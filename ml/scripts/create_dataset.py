"""
Create training dataset by extracting tiles from Sentinel-2 scenes.

This script reads the matched FIRMS-scene dataset and extracts image tiles
centered around fire locations (positive samples) and random non-fire locations
(negative samples). The tiles are saved as numpy arrays with labels.
"""

import json
import logging
import random
from pathlib import Path
from typing import Tuple, List, Dict, Optional
import yaml

import numpy as np
import pandas as pd
import rasterio
from rasterio.windows import Window
from rasterio.transform import rowcol
from rasterio.warp import transform as reproject_coords
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DatasetGenerator:
    """Generate training dataset from matched fire-scene pairs."""
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize generator with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tile_size = self.config['tiles']['size']
        self.negative_ratio = self.config['tiles']['negative_sample_ratio']
        self.bands = self.config['sentinel2']['bands']
        self.band_names = self.config['sentinel2']['band_names']
        self.band_save_names = self.config['sentinel2'].get('band_save_names', self.bands)
        
        self.tiles_dir = Path(self.config['data']['tiles_dir'])
        self.labels_dir = Path(self.config['data']['labels_dir'])
        self.tiles_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        
        # Use fixed seed for reproducible train/val/test splits
        # but keep spatial offsets random to prevent center bias
        random.seed(self.config['splits']['random_seed'])
        np.random.seed(self.config['splits']['random_seed'])
        
        # Create separate random state for spatial offsets (not seeded)
        self.spatial_random = random.Random()  # Uses system entropy
    
    def load_matched_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load the matched FIRMS-scene dataset."""
        df = pd.DataFrame(dataset_path)
        logger.info(f"Loaded {len(df)} fire-scene matches from {dataset_path}")
        return df
    
    def load_scene_bands(self, scene_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all required bands from a Sentinel-2 scene.
        
        Args:
            scene_path: Path to scene directory
            
        Returns:
            Dictionary mapping band names to numpy arrays
        """
        bands_data = {}
        
        for i, band in enumerate(self.bands):
            # Use band_save_names for file lookup (e.g., B04, B08, etc.)
            file_band_name = self.band_save_names[i] if i < len(self.band_save_names) else band
            band_file = scene_path / f"{file_band_name}.tif"
            
            if not band_file.exists():
                logger.warning(f"Band file not found: {band_file}")
                return None
            
            try:
                with rasterio.open(band_file) as src:
                    bands_data[band] = src.read(1)  # Read first band
                    
                    # Store transform and profile for later use
                    if 'transform' not in bands_data:
                        bands_data['transform'] = src.transform
                        bands_data['crs'] = src.crs
                        bands_data['profile'] = src.profile
            
            except Exception as e:
                logger.error(f"Error reading band {band} from {band_file}: {e}")
                return None
        
        return bands_data
    
    def extract_tile_at_location(
        self,
        bands_data: Dict[str, np.ndarray],
        lon: float,
        lat: float,
        tile_size: int = 256
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Extract a tile centered at a geographic location.
        
        Args:
            bands_data: Dictionary of band arrays and metadata
            lon: Longitude of center point (EPSG:4326)
            lat: Latitude of center point (EPSG:4326)
            tile_size: Size of tile in pixels
            
        Returns:
            Tuple of (tile array, metadata dict with fire location), or None if invalid
        """
        transform = bands_data['transform']
        crs = bands_data['crs']
        
        # Reproject lon/lat (EPSG:4326) to raster CRS
        xs, ys = reproject_coords('EPSG:4326', crs, [lon], [lat])
        x, y = xs[0], ys[0]
        
        # Convert reprojected coordinates to pixel coordinates
        row, col = rowcol(transform, x, y)
        
        # Calculate tile boundaries with random offset to prevent center bias
        half_size = tile_size // 2
        
        # Add random offset (±25% of half_size) so fire isn't always centered
        # Use separate random state to ensure true randomness (not seeded)
        offset_range = int(half_size * 0.25)
        row_offset = self.spatial_random.randint(-offset_range, offset_range)
        col_offset = self.spatial_random.randint(-offset_range, offset_range)
        
        row_start = max(0, row - half_size + row_offset)
        col_start = max(0, col - half_size + col_offset)
        
        # Extract tile from each band
        tiles = []
        
        for band in self.bands:
            band_array = bands_data[band]
            
            # Check if tile is within bounds
            if (row_start + tile_size > band_array.shape[0] or
                col_start + tile_size > band_array.shape[1]):
                return None
            
            tile = band_array[row_start:row_start+tile_size,
                             col_start:col_start+tile_size]
            
            # Check for no-data values (typically 0 or very negative)
            if np.sum(tile == 0) > (tile_size * tile_size * 0.5):
                return None  # Too much no-data
            
            tiles.append(tile)
        
        # Stack bands: shape (bands, height, width)
        multi_band_tile = np.stack(tiles, axis=0)
        
        # Calculate where the fire center is within this tile
        # (accounting for the offset we applied)
        fire_row_in_tile = half_size - row_offset
        fire_col_in_tile = half_size - col_offset
        
        # Return tile and metadata with fire location and scene position
        metadata = {
            'fire_pixel_row': int(fire_row_in_tile),
            'fire_pixel_col': int(fire_col_in_tile),
            'offset_applied': [int(row_offset), int(col_offset)],  # List for JSON serialization
            # Store tile position in original scene for RGB extraction
            'scene_row_start': int(row_start),
            'scene_col_start': int(col_start),
            'tile_size': int(tile_size)
        }
        
        return multi_band_tile, metadata
    
    def generate_negative_samples(
        self,
        bands_data: Dict[str, np.ndarray],
        fire_locations: List[Tuple[float, float]],
        n_samples: int,
        min_distance_km: float = 5.0
    ) -> List[np.ndarray]:
        """
        Generate negative (non-fire) samples from the same scene.
        
        Args:
            bands_data: Dictionary of band arrays
            fire_locations: List of (lon, lat) tuples for fires in this scene
            n_samples: Number of negative samples to generate
            min_distance_km: Minimum distance from any fire (km)
            
        Returns:
            List of tile arrays
        """
        negative_tiles = []
        
        # Get scene dimensions
        sample_band = bands_data[self.bands[0]]
        height, width = sample_band.shape
        transform = bands_data['transform']
        
        # Convert min distance to degrees (approximate)
        min_distance_deg = min_distance_km / 111.0  # ~111 km per degree
        
        max_attempts = n_samples * 10
        attempts = 0
        
        while len(negative_tiles) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Random pixel location
            row = random.randint(self.tile_size, height - self.tile_size)
            col = random.randint(self.tile_size, width - self.tile_size)
            
            # Convert pixel to raster CRS coordinates
            x, y = rasterio.transform.xy(transform, row, col)
            
            # Convert to lon/lat (EPSG:4326) for distance check
            crs = bands_data['crs']
            lons, lats = reproject_coords(crs, 'EPSG:4326', [x], [y])
            lon, lat = lons[0], lats[0]
            
            # Check distance from all fires
            too_close = False
            for fire_lon, fire_lat in fire_locations:
                dist = np.sqrt((lon - fire_lon)**2 + (lat - fire_lat)**2)
                if dist < min_distance_deg:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Extract tile (no fire metadata needed for negatives)
            result = self.extract_tile_at_location(bands_data, lon, lat, self.tile_size)
            
            if result is not None:
                tile, _ = result  # Ignore metadata for negative samples
                negative_tiles.append(tile)
        
        if len(negative_tiles) < n_samples:
            logger.warning(f"Only generated {len(negative_tiles)}/{n_samples} negative samples")
        
        return negative_tiles
    
    def normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Normalize tile values to 0-1 range.
        
        Args:
            tile: Tile array of shape (bands, height, width)
            
        Returns:
            Normalized tile
        """
        # Clip extreme values (typical Sentinel-2 range is 0-10000)
        tile = np.clip(tile, 0, 10000)
        
        # Normalize to 0-1
        tile = tile / 10000.0
        
        return tile.astype(np.float32)
    
    def save_tile(
        self,
        tile: np.ndarray,
        label: int,
        scene_id: str,
        fire_id: str,
        sample_type: str,
        fire_location: Optional[Dict] = None
    ) -> Path:
        """
        Save a tile and its label.
        
        Args:
            tile: Tile array
            label: 0 for no-fire, 1 for fire
            scene_id: Sentinel-2 scene ID
            fire_id: FIRMS fire ID
            sample_type: 'positive' or 'negative'
            
        Returns:
            Path to saved tile
        """
        # Generate unique filename
        tile_id = f"{scene_id}_{fire_id}_{sample_type}_{random.randint(1000, 9999)}"
        
        # Save tile as numpy array
        tile_path = self.tiles_dir / f"{tile_id}.npy"
        np.save(tile_path, tile)
        
        # Save label metadata
        label_data = {
            'tile_id': tile_id,
            'label': label,
            'scene_id': scene_id,
            'fire_id': fire_id,
            'sample_type': sample_type,
            'bands': self.band_names,
            'shape': list(tile.shape)
        }
        
        # Add fire location if provided (for positive samples)
        if fire_location is not None:
            label_data['fire_pixel_row'] = int(fire_location['fire_pixel_row'])
            label_data['fire_pixel_col'] = int(fire_location['fire_pixel_col'])
            # Store scene coordinates for RGB extraction later  
            if fire_location.get('scene_row_start') is not None:
                label_data['scene_row_start'] = int(fire_location['scene_row_start'])
            if fire_location.get('scene_col_start') is not None:
                label_data['scene_col_start'] = int(fire_location['scene_col_start'])
            if fire_location.get('tile_size') is not None:
                label_data['tile_size'] = int(fire_location['tile_size'])
        
        label_path = self.labels_dir / f"{tile_id}.json"
        with open(label_path, 'w') as f:
            json.dump(label_data, f, indent=2)
        
        return tile_path
    
    def process_scene(self, scene_matches: pd.DataFrame) -> Dict[str, int]:
        """
        Process all fires in a single scene and generate tiles.
        
        Args:
            scene_matches: DataFrame of fires in this scene
            
        Returns:
            Dictionary with counts of positive and negative samples generated
        """
        scene_id = scene_matches.iloc[0]['scene_id']
        scene_path = Path(scene_matches.iloc[0]['scene_path'])
        
        logger.info(f"Processing scene {scene_id} with {len(scene_matches)} fires")
        
        # Load all bands
        bands_data = self.load_scene_bands(scene_path)
        
        if bands_data is None:
            logger.error(f"Failed to load bands for scene {scene_id}")
            return {'positive': 0, 'negative': 0}
        
        positive_count = 0
        negative_count = 0
        
        # Extract positive samples (fires)
        fire_locations = []
        
        for _, row in scene_matches.iterrows():
            lon, lat = row['fire_lon'], row['fire_lat']
            fire_locations.append((lon, lat))
            
            # Extract tile centered on fire
            result = self.extract_tile_at_location(bands_data, lon, lat, self.tile_size)
            
            if result is not None:
                tile, fire_metadata = result
                
                # Normalize
                tile = self.normalize_tile(tile)
                
                # Save with fire location metadata
                self.save_tile(tile, label=1, scene_id=scene_id,
                             fire_id=str(row['fire_id']), sample_type='positive',
                             fire_location=fire_metadata)
                positive_count += 1
        
        # Generate negative samples
        n_negative = len(fire_locations) * self.negative_ratio
        negative_tiles = self.generate_negative_samples(
            bands_data, fire_locations, int(n_negative)
        )
        
        for i, tile in enumerate(negative_tiles):
            # Normalize
            tile = self.normalize_tile(tile)
            
            # Save
            self.save_tile(tile, label=0, scene_id=scene_id,
                         fire_id='none', sample_type='negative')
            negative_count += 1
        
        logger.info(f"Scene {scene_id}: {positive_count} positive, {negative_count} negative samples")
        
        return {'positive': positive_count, 'negative': negative_count}
    
    def run(self, matched_dataset_path: str = "ml/data/processed/matched_dataset.csv"):
        """
        Main execution: generate complete dataset.
        
        Args:
            matched_dataset_path: Path to matched FIRMS-scene CSV
        """
        # Load matched dataset
        logger.info("Loading matched dataset...")
        matched_df = pd.read_csv(matched_dataset_path)
        
        if matched_df.empty:
            logger.error("Matched dataset is empty")
            return
        
        # Group by scene
        scene_groups = matched_df.groupby('scene_id')
        logger.info(f"Processing {len(scene_groups)} scenes")
        
        total_positive = 0
        total_negative = 0
        
        # Process each scene
        for scene_id, scene_matches in tqdm(scene_groups, desc="Processing scenes"):
            try:
                counts = self.process_scene(scene_matches)
                total_positive += counts['positive']
                total_negative += counts['negative']
            except Exception as e:
                logger.error(f"Error processing scene {scene_id}: {e}")
                continue
        
        # Generate summary
        logger.info("\n" + "="*50)
        logger.info("DATASET GENERATION SUMMARY")
        logger.info("="*50)
        logger.info(f"Total positive samples (fire): {total_positive}")
        logger.info(f"Total negative samples (no-fire): {total_negative}")
        total_samples = total_positive + total_negative
        logger.info(f"Total samples: {total_samples}")
        if total_samples > 0:
            logger.info(f"Class balance: {total_positive / total_samples:.2%} fire")
        else:
            logger.warning("No samples generated. Check that Sentinel-2 band files exist.")
        logger.info(f"Tiles saved to: {self.tiles_dir}")
        logger.info(f"Labels saved to: {self.labels_dir}")
        
        # Save summary
        total_samples = total_positive + total_negative
        summary = {
            'total_positive': total_positive,
            'total_negative': total_negative,
            'total_samples': total_samples,
            'class_balance': total_positive / total_samples if total_samples > 0 else 0,
            'tile_size': self.tile_size,
            'bands': self.band_names
        }
        
        summary_path = Path(self.config['data']['processed_dir']) / 'dataset_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Summary saved to: {summary_path}")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate training dataset from matched fires and scenes")
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Path to config file')
    parser.add_argument('--matched-dataset', default='ml/data/processed/matched_dataset.csv',
                       help='Path to matched dataset CSV')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config)
    generator.run(args.matched_dataset)


if __name__ == "__main__":
    main()
