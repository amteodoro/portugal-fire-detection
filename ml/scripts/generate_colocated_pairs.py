"""
Co-Located Tile Generator.

Creates paired tiles from Sentinel-2 and Sentinel-3 that show the EXACT same
geographic location. This ensures optical and thermal data can be directly compared.

Pipeline:
1. Load FIRMS fire detections
2. Find fires that have coverage from BOTH S2 and S3
3. For each fire, extract tiles from both satellites at the EXACT same coordinate
4. Save as linked pairs with metadata

Output structure:
- data/samples/colocated/
  - fire_location_001/
    - s2_tile.npy
    - s3_tile.npy
    - metadata.json
  - no_fire_location_001/
    - s2_tile.npy
    - s3_tile.npy
    - metadata.json
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime
import yaml
import random
import re

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import rowcol
from rasterio.warp import transform as reproject_coords
from tqdm import tqdm

# Import Sentinel-3 processor
import sys
sys.path.insert(0, str(Path(__file__).parent))
from sentinel3_processor import Sentinel3Processor

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ColocatedTileGenerator:
    """Generate co-located tile pairs from S2 and S3."""
    
    # Portugal bounding box (exclude Spain)
    PORTUGAL_BOUNDS = {
        'lon_min': -9.5, 'lon_max': -6.2,  # Tightened to exclude most of Spain
        'lat_min': 36.95, 'lat_max': 42.15
    }
    
    def __init__(self, config_path: str = "ml/configs/config.yaml", tile_size: int = None):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.s2_dir = Path(self.config['data']['raw_dir'])
        self.s3_dir = Path(self.config['data']['sentinel3_dir'])
        self.output_dir = Path("data/samples/colocated")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Tile size from parameter, config, or default
        self.tile_size = tile_size or self.config.get('tiles', {}).get('size', 128)
        self.s3_processor = Sentinel3Processor()
        
        # S2 bands - expanded to include RGB for True Color visualization
        # Order: B02(Blue), B03(Green), B04(Red), B08(NIR), B11(SWIR1), B12(SWIR2)
        self.s2_bands = ['B02', 'B03', 'B04', 'B08', 'B11', 'B12']
    
    def is_in_portugal(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Portugal bounds."""
        return (self.PORTUGAL_BOUNDS['lat_min'] <= lat <= self.PORTUGAL_BOUNDS['lat_max'] and
                self.PORTUGAL_BOUNDS['lon_min'] <= lon <= self.PORTUGAL_BOUNDS['lon_max'])
    
    def find_colocated_fires(self) -> List[Dict]:
        """
        Find FIRMS fires that have BOTH S2 and S3 scene coverage.
        
        Returns list of dicts with fire info and scene paths.
        """
        matched_path = Path("ml/data/processed/matched_dataset.csv")
        if not matched_path.exists():
            logger.error("Matched dataset not found. Run matching first.")
            return []
        
        df = pd.read_csv(matched_path)
        
        # Group by fire_id
        fires_with_both = []
        
        for fire_id, group in df.groupby('fire_id'):
            satellites = group['satellite'].unique()
            
            if 'sentinel2' in satellites and 'sentinel3' in satellites:
                s2_rows = group[group['satellite'] == 'sentinel2']
                s3_rows = group[group['satellite'] == 'sentinel3']
                
                if s2_rows.empty or s3_rows.empty:
                    continue
                
                # Get fire coordinates
                lat = group.iloc[0]['fire_lat']
                lon = group.iloc[0]['fire_lon']
                
                # Filter to Portugal only
                if not self.is_in_portugal(lat, lon):
                    continue
                
                # Get best S2 scene (lowest cloud cover)
                s2_best = s2_rows.loc[s2_rows['cloud_cover'].idxmin()]
                # Get nearest S3 scene (by date)
                s3_best = s3_rows.iloc[0]
                
                fires_with_both.append({
                    'fire_id': fire_id,
                    'lat': lat,
                    'lon': lon,
                    'fire_date': group.iloc[0]['fire_date'],
                    'fire_confidence': group.iloc[0]['fire_confidence'],
                    's2_scene_id': s2_best['scene_id'],
                    's2_scene_path': s2_best['scene_path'],
                    's2_date': s2_best['scene_date'],
                    's3_scene_id': s3_best['scene_id'],
                    's3_scene_path': s3_best['scene_path'],
                    's3_date': s3_best['scene_date'],
                })
        
        logger.info(f"Found {len(fires_with_both)} fires with both S2 and S3 coverage in Portugal")
        return fires_with_both
    
    def load_s2_bands(self, scene_path: str) -> Optional[Dict]:
        """Load Sentinel-2 bands from scene directory, handling different resolutions."""
        scene_dir = Path(scene_path)
        
        if not scene_dir.exists():
            return None
        
        bands_data = {'bands': {}}
        
        for band_name in self.s2_bands:
            # Find band file
            band_files = list(scene_dir.glob(f"*{band_name}*.tif"))
            if not band_files:
                band_files = list(scene_dir.glob(f"*{band_name.lower()}*.tif"))
            
            if not band_files:
                return None
            
            try:
                with rasterio.open(band_files[0]) as src:
                    bands_data['bands'][band_name] = {
                        'data': src.read(1),
                        'transform': src.transform,
                        'crs': src.crs,
                        'shape': src.shape
                    }
            except Exception as e:
                logger.warning(f"Failed to load {band_name}: {e}")
                return None
        
        return bands_data
    
    def extract_s2_tile(self, bands_data: Dict, lon: float, lat: float) -> Optional[Tuple[np.ndarray, Dict]]:
        """Extract S2 tile at exact coordinate, handling different band resolutions."""
        from scipy.ndimage import zoom
        
        tiles = []
        ref_shape = None
        
        for band_name in self.s2_bands:
            band_info = bands_data['bands'][band_name]
            data = band_info['data']
            transform = band_info['transform']
            crs = band_info['crs']
            h, w = band_info['shape']
            
            # Reproject lon/lat to raster CRS
            xs, ys = reproject_coords('EPSG:4326', crs, [lon], [lat])
            x, y = xs[0], ys[0]
            
            # Convert to pixel coordinates
            row, col = rowcol(transform, x, y)
            
            half_size = self.tile_size // 2
            row_start = row - half_size
            col_start = col - half_size
            
            # Check bounds
            if row_start < 0 or col_start < 0:
                return None
            if row_start + self.tile_size > h or col_start + self.tile_size > w:
                return None
            
            # Extract tile
            tile = data[row_start:row_start+self.tile_size, col_start:col_start+self.tile_size]
            
            # Track reference shape (should be tile_size x tile_size)
            if ref_shape is None:
                ref_shape = tile.shape
            
            # Resample if needed (shouldn't be needed if we extracted correctly)
            if tile.shape != ref_shape:
                zoom_factors = (ref_shape[0] / tile.shape[0], ref_shape[1] / tile.shape[1])
                tile = zoom(tile, zoom_factors, order=1)
            
            tiles.append(tile)
        
        tile_array = np.stack(tiles, axis=0).astype(np.float32)
        
        # Normalize
        for i in range(tile_array.shape[0]):
            band = tile_array[i]
            valid = band[band > 0]
            if len(valid) > 0:
                p99 = np.percentile(valid, 99)
                if p99 > 0:
                    tile_array[i] = band / p99
        
        metadata = {
            'fire_pixel_row': self.tile_size // 2,  # Fire is at center
            'fire_pixel_col': self.tile_size // 2,
        }
        
        return tile_array, metadata
    
    def find_best_s3_scene(self, lon: float, lat: float) -> Optional[Tuple[Path, Dict]]:
        """Find the S3 scene with best coverage for the given coordinates."""
        all_scenes = []
        for date_dir in self.s3_dir.iterdir():
            if date_dir.is_dir():
                for scene_dir in date_dir.iterdir():
                    if scene_dir.is_dir():
                        all_scenes.append(scene_dir)
        
        best_scene = None
        best_row_distance = float('inf')
        best_bands = None
        
        for scene in all_scenes:
            bands = self.s3_processor.load_scene_bands(scene)
            if not bands:
                continue
            
            lat_grid = bands['lat']
            lon_grid = bands['lon']
            
            center = self.s3_processor.find_nearest_pixel(lat_grid, lon_grid, lat, lon)
            if not center:
                continue
            
            row, col = center
            # Calculate how far from edges
            row_margin = min(row, lat_grid.shape[0] - row)
            col_margin = min(col, lat_grid.shape[1] - col)
            
            # We need at least tile_size/2 margin on all sides
            min_margin = self.tile_size // 2
            if row_margin >= min_margin and col_margin >= min_margin:
                # This scene has good coverage - use it
                return scene, bands
            
            # Track best partial coverage
            total_margin = row_margin + col_margin
            if total_margin > best_row_distance:
                continue  # Worse than current best
        
        return None
    
    def extract_s3_tile(self, scene_path: str, lon: float, lat: float) -> Optional[Tuple[np.ndarray, Dict]]:
        """Extract S3 tile at exact coordinate, searching all scenes for best coverage."""
        # First try the provided scene
        bands_data = self.s3_processor.load_scene_bands(Path(scene_path))
        
        if bands_data is not None:
            result = self.s3_processor.extract_tile_at_location(
                bands_data, lon, lat, self.tile_size
            )
            if result is not None:
                tile_array, metadata = result
                metadata['fire_pixel_row'] = self.tile_size // 2
                metadata['fire_pixel_col'] = self.tile_size // 2
                metadata['s3_scene_used'] = scene_path
                return tile_array.astype(np.float32), metadata
        
        # If that fails, search ALL S3 scenes for best coverage
        best_result = self.find_best_s3_scene(lon, lat)
        if best_result is None:
            return None
        
        scene, bands_data = best_result
        result = self.s3_processor.extract_tile_at_location(
            bands_data, lon, lat, self.tile_size
        )
        
        if result is None:
            return None
        
        tile_array, metadata = result
        metadata['fire_pixel_row'] = self.tile_size // 2
        metadata['fire_pixel_col'] = self.tile_size // 2
        metadata['s3_scene_used'] = str(scene)
        return tile_array.astype(np.float32), metadata
    
    def generate_colocated_pairs(self, max_pairs: int = 50) -> int:
        """Generate co-located tile pairs."""
        logger.info("Finding fires with both S2 and S3 coverage...")
        fires = self.find_colocated_fires()
        
        if not fires:
            logger.error("No co-located fires found!")
            return 0
        
        # Limit to max_pairs
        if len(fires) > max_pairs:
            fires = random.sample(fires, max_pairs)
        
        successful = 0
        
        for i, fire in enumerate(tqdm(fires, desc="Generating co-located pairs")):
            try:
                # Create output directory for this pair
                pair_dir = self.output_dir / f"fire_{i+1:04d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                
                # Load and extract S2 tile
                s2_bands = self.load_s2_bands(fire['s2_scene_path'])
                if s2_bands is None:
                    logger.debug(f"Failed to load S2 bands from {fire['s2_scene_path']}")
                    continue
                
                s2_result = self.extract_s2_tile(s2_bands, fire['lon'], fire['lat'])
                if s2_result is None:
                    logger.debug(f"Failed to extract S2 tile at {fire['lat']}, {fire['lon']}")
                    continue
                
                s2_tile, s2_meta = s2_result
                
                # Extract S3 tile at SAME coordinate
                s3_result = self.extract_s3_tile(fire['s3_scene_path'], fire['lon'], fire['lat'])
                if s3_result is None:
                    logger.debug(f"Failed to extract S3 tile from {fire['s3_scene_path']}")
                    continue
                
                s3_tile, s3_meta = s3_result
                
                logger.info(f"SUCCESS: Generated pair for {fire['fire_id']} at {fire['lat']:.3f}, {fire['lon']:.3f}")
                
                # Save tiles
                np.save(pair_dir / "s2_tile.npy", s2_tile)
                np.save(pair_dir / "s3_tile.npy", s3_tile)
                
                # Save metadata
                metadata = {
                    'fire_id': fire['fire_id'],
                    'lat': fire['lat'],
                    'lon': fire['lon'],
                    'fire_date': fire['fire_date'],
                    'fire_confidence': float(fire['fire_confidence']) if pd.notna(fire['fire_confidence']) else None,
                    'has_fire': True,
                    's2': {
                        'scene_id': fire['s2_scene_id'],
                        'scene_date': fire['s2_date'],
                        'fire_pixel_row': s2_meta['fire_pixel_row'],
                        'fire_pixel_col': s2_meta['fire_pixel_col'],
                    },
                    's3': {
                        'scene_id': fire['s3_scene_id'],
                        'scene_date': fire['s3_date'],
                        'fire_pixel_row': s3_meta['fire_pixel_row'],
                        'fire_pixel_col': s3_meta['fire_pixel_col'],
                    }
                }
                
                with open(pair_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                successful += 1
                
            except Exception as e:
                logger.warning(f"Failed to process fire {fire['fire_id']}: {e}")
                continue
        
        logger.info(f"\nGenerated {successful} co-located tile pairs")
        logger.info(f"Saved to: {self.output_dir}")
        
        return successful
    
    def generate_no_fire_pairs(self, max_pairs: int = 20) -> int:
        """Generate co-located pairs for locations WITHOUT fires."""
        logger.info("Generating no-fire co-located pairs...")
        
        # Find S2 scenes
        s2_scenes = list(self.s2_dir.glob("S2*"))
        if not s2_scenes:
            logger.warning("No S2 scenes found")
            return 0
        
        # Find S3 scenes
        s3_scenes = list(self.s3_dir.glob("*.zip")) + list(self.s3_dir.glob("*.SEN3"))
        if not s3_scenes:
            logger.warning("No S3 scenes found")
            return 0
        
        successful = 0
        attempts = 0
        max_attempts = max_pairs * 5
        
        # Load fire locations to avoid them
        matched_path = Path("ml/data/processed/matched_dataset.csv")
        fire_locs = set()
        if matched_path.exists():
            df = pd.read_csv(matched_path)
            for _, row in df.iterrows():
                # Round to 0.05 degrees (~5km) for exclusion
                lat_key = round(row['fire_lat'], 2)
                lon_key = round(row['fire_lon'], 2)
                fire_locs.add((lat_key, lon_key))
        
        while successful < max_pairs and attempts < max_attempts:
            attempts += 1
            
            # Random location in Portugal
            lat = random.uniform(37.0, 42.0)
            lon = random.uniform(-9.4, -6.3)
            
            # Skip if near a fire
            if (round(lat, 2), round(lon, 2)) in fire_locs:
                continue
            
            # Skip if not in Portugal
            if not self.is_in_portugal(lat, lon):
                continue
            
            try:
                # Try random S2 scene
                s2_scene = random.choice(s2_scenes)
                s2_bands = self.load_s2_bands(str(s2_scene))
                if s2_bands is None:
                    continue
                
                s2_result = self.extract_s2_tile(s2_bands, lon, lat)
                if s2_result is None:
                    continue
                
                # Try random S3 scene
                s3_scene = random.choice(s3_scenes)
                s3_result = self.extract_s3_tile(str(s3_scene), lon, lat)
                if s3_result is None:
                    continue
                
                s2_tile, s2_meta = s2_result
                s3_tile, s3_meta = s3_result
                
                # Save
                pair_dir = self.output_dir / f"no_fire_{successful+1:04d}"
                pair_dir.mkdir(parents=True, exist_ok=True)
                
                np.save(pair_dir / "s2_tile.npy", s2_tile)
                np.save(pair_dir / "s3_tile.npy", s3_tile)
                
                metadata = {
                    'fire_id': None,
                    'lat': lat,
                    'lon': lon,
                    'has_fire': False,
                    's2': {
                        'scene_id': s2_scene.name,
                        'fire_pixel_row': None,
                        'fire_pixel_col': None,
                    },
                    's3': {
                        'scene_id': s3_scene.name,
                        'fire_pixel_row': None,
                        'fire_pixel_col': None,
                    }
                }
                
                with open(pair_dir / "metadata.json", 'w') as f:
                    json.dump(metadata, f, indent=2, default=str)
                
                successful += 1
                
            except Exception as e:
                continue
        
        logger.info(f"Generated {successful} no-fire co-located pairs")
        return successful


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate co-located S2+S3 tile pairs")
    parser.add_argument('--config', default='ml/configs/config.yaml')
    parser.add_argument('--tile-size', type=int, default=None, help='Tile size in pixels (default: from config)')
    parser.add_argument('--max-fire-pairs', type=int, default=30)
    parser.add_argument('--max-nofire-pairs', type=int, default=10)
    args = parser.parse_args()
    
    generator = ColocatedTileGenerator(args.config, tile_size=args.tile_size)
    
    logger.info(f"Using tile size: {generator.tile_size}x{generator.tile_size} pixels")
    
    # Clear existing
    import shutil
    if generator.output_dir.exists():
        shutil.rmtree(generator.output_dir)
    generator.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate fire pairs
    fire_count = generator.generate_colocated_pairs(args.max_fire_pairs)
    
    # Generate no-fire pairs
    nofire_count = generator.generate_no_fire_pairs(args.max_nofire_pairs)
    
    logger.info("\n" + "="*50)
    logger.info("CO-LOCATED PAIR GENERATION COMPLETE")
    logger.info("="*50)
    logger.info(f"Tile size: {generator.tile_size}x{generator.tile_size}")
    logger.info(f"Fire pairs: {fire_count}")
    logger.info(f"No-fire pairs: {nofire_count}")
    logger.info(f"Total: {fire_count + nofire_count}")
    logger.info(f"Output: {generator.output_dir}")
    
    if fire_count + nofire_count == 0:
        logger.warning("")
        logger.warning("No co-located pairs could be generated!")
        logger.warning("Suggestions:")
        logger.warning(f"  1. Try smaller tile size (current: {generator.tile_size})")
        logger.warning("  2. Fetch more Sentinel-3 data (expand date range)")
        logger.warning("  3. Check that S2 and S3 data overlap geographically")


if __name__ == "__main__":
    main()
