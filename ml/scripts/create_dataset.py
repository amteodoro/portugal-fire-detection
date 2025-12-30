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

# Import Sentinel-3 processor
from sentinel3_processor import Sentinel3Processor

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
        
        # Initialize Sentinel-3 processor
        self.s3_processor = Sentinel3Processor()
    
    def load_matched_dataset(self, dataset_path: str) -> pd.DataFrame:
        """Load the matched FIRMS-scene dataset."""
        df = pd.DataFrame(dataset_path)
        logger.info(f"Loaded {len(df)} fire-scene matches from {dataset_path}")
        return df
    
    def load_scene_bands(self, scene_path: Path) -> Optional[Dict[str, np.ndarray]]:
        """
        Load all required bands from a Sentinel-2 scene.
        
        All bands are resampled to match the highest resolution (10m) for consistent
        pixel coordinates across bands. Sentinel-2 has mixed resolutions:
        - B02, B03, B04, B08: 10m (10980x10980 pixels per granule)
        - B05, B06, B07, B8A, B11, B12: 20m (5490x5490 pixels per granule)
        
        Args:
            scene_path: Path to scene directory
            
        Returns:
            Dictionary mapping band names to numpy arrays (all same resolution)
        """
        from scipy.ndimage import zoom
        
        bands_data = {}
        target_shape = None  # Will be set from highest resolution band
        
        for i, band in enumerate(self.bands):
            # Use band_save_names for file lookup (e.g., B04, B08, etc.)
            file_band_name = self.band_save_names[i] if i < len(self.band_save_names) else band
            band_file = scene_path / f"{file_band_name}.tif"
            
            if not band_file.exists():
                logger.warning(f"Band file not found: {band_file}")
                return None
            
            try:
                with rasterio.open(band_file) as src:
                    data = src.read(1)  # Read first band
                    
                    # Store transform/CRS from this band
                    band_transform = src.transform
                    band_crs = src.crs
                    band_resolution = abs(band_transform[0])  # Pixel size in meters
                    
                    # Set target shape from first (highest resolution) band
                    if target_shape is None:
                        target_shape = data.shape
                        bands_data['transform'] = band_transform
                        bands_data['crs'] = band_crs
                        bands_data['profile'] = src.profile
                        bands_data['resolution'] = band_resolution
                        logger.debug(f"Target resolution: {band_resolution}m, shape: {target_shape}")
                    
                    # Resample if needed (20m -> 10m)
                    if data.shape != target_shape:
                        scale_factor = data.shape[0] / target_shape[0]
                        if scale_factor < 1:  # Need to upsample
                            zoom_factor = target_shape[0] / data.shape[0]
                            logger.debug(f"Resampling {file_band_name} from {data.shape} to {target_shape} (zoom={zoom_factor:.1f}x)")
                            data = zoom(data, zoom_factor, order=1)  # Bilinear interpolation
                    
                    bands_data[band] = data
            
            except Exception as e:
                logger.error(f"Error reading band {band} from {band_file}: {e}")
                return None
        
        # Log summary
        logger.info(f"Loaded {len(self.bands)} bands from {scene_path.name}, all resampled to {target_shape}")
        
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
        
        # --- SNAP TO HOTSPOT LOGIC ---
        # FIRMS data has low resolution (~1km). The actual fire in Sentinel-2 (20m)
        # might be offset. We search for the brightest pixel in SWIR2 (Band 12)
        # within a ~1km radius (50 pixels) and snap the center to it.
        
        # SWIR2 is usually the last band (index 3) or explicitly named 'swir2' or 'B12'
        swir2_band_name = self.bands[-1]  # Assuming SWIR2 is last as per config
        if swir2_band_name in bands_data:
            swir2_data = bands_data[swir2_band_name]
            h, w = swir2_data.shape
            
            search_radius = 50  # ~1000m radius at 20m resolution
            r_min = max(0, row - search_radius)
            r_max = min(h, row + search_radius)
            c_min = max(0, col - search_radius)
            c_max = min(w, col + search_radius)
            
            # Extract window
            window = swir2_data[r_min:r_max, c_min:c_max]
            
            if window.size > 0:
                # Find local maximum in this window
                # We normalize index relative to window, then add window offset
                try:
                    local_max_idx = np.unravel_index(np.argmax(window), window.shape)
                    max_r = r_min + local_max_idx[0]
                    max_c = c_min + local_max_idx[1]
                    
                    # Update row, col to snapped location
                    # Only snap if the hotspot is significantly bright (e.g. > 1000, fires are usually > 4000-5000)
                    if window[local_max_idx] > 1000:
                        row, col = int(max_r), int(max_c)
                except Exception as e:
                    logger.warning(f"Failed to snap to hotspot: {e}")

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
        # IMPORTANT: Must use actual row_start/col_start (after clipping), not the unclipped values!
        # The fire is at pixel (row, col) in the scene.
        # After extraction starting at (row_start, col_start), the fire is at:
        fire_row_in_tile = row - row_start
        fire_col_in_tile = col - col_start
        
        # Validate fire is actually within the tile (should always be true if no bugs)
        if not (0 <= fire_row_in_tile < tile_size and 0 <= fire_col_in_tile < tile_size):
            logger.warning(f"Fire location ({fire_row_in_tile}, {fire_col_in_tile}) outside tile bounds!")
            # Clamp to valid range as fallback
            fire_row_in_tile = max(0, min(tile_size - 1, fire_row_in_tile))
            fire_col_in_tile = max(0, min(tile_size - 1, fire_col_in_tile))
        
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
            
            # Extract tile (include metadata for location)
            result = self.extract_tile_at_location(bands_data, lon, lat, self.tile_size)
            
            if result is not None:
                tile, metadata = result
                # Add tile center coordinates to metadata
                metadata['tile_center_lat'] = lat
                metadata['tile_center_lon'] = lon
                negative_tiles.append((tile, metadata))
        
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
        
        # Add location metadata (for both positive and negative samples)
        if fire_location is not None:
            # Tile center coordinates (for map display)
            if 'tile_center_lat' in fire_location:
                label_data['tile_center_lat'] = float(fire_location['tile_center_lat'])
            if 'tile_center_lon' in fire_location:
                label_data['tile_center_lon'] = float(fire_location['tile_center_lon'])
            
            # Fire pixel location within tile (positive samples)
            if 'fire_pixel_row' in fire_location:
                label_data['fire_pixel_row'] = int(fire_location['fire_pixel_row'])
            if 'fire_pixel_col' in fire_location:
                label_data['fire_pixel_col'] = int(fire_location['fire_pixel_col'])
            
            # Scene coordinates for RGB extraction later  
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
        satellite = scene_matches.iloc[0].get('satellite', 'sentinel2')
        
        logger.info(f"Processing {satellite} scene {scene_id} with {len(scene_matches)} fires")
        
        # Detect satellite type and load bands accordingly
        is_sentinel3 = satellite == 'sentinel3' or 'sentinel3' in str(scene_path).lower()
        
        if is_sentinel3:
            # Use Sentinel-3 processor
            bands_data = self.s3_processor.load_scene_bands(scene_path)
        else:
            # Use Sentinel-2 processor (existing method)
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
            if is_sentinel3:
                result = self.s3_processor.extract_tile_at_location(bands_data, lon, lat, self.tile_size)
            else:
                result = self.extract_tile_at_location(bands_data, lon, lat, self.tile_size)
            
            if result is not None:
                tile, fire_metadata = result
                
                # Add tile center coordinates (same as fire location for positive samples)
                fire_metadata['tile_center_lat'] = lat
                fire_metadata['tile_center_lon'] = lon
                
                # Normalize (Sentinel-3 tiles are already normalized in extract_tile_at_location)
                if not is_sentinel3:
                    tile = self.normalize_tile(tile)
                else:
                    tile = tile.astype(np.float32)  # Ensure float32 dtype
                
                # Save with fire location metadata
                self.save_tile(tile, label=1, scene_id=scene_id,
                             fire_id=str(row['fire_id']), sample_type='positive',
                             fire_location=fire_metadata)
                positive_count += 1
        
        # Generate negative samples
        n_negative = len(fire_locations) * self.negative_ratio
        
        if is_sentinel3:
            negative_tiles = self.s3_processor.generate_negative_samples(
                bands_data, fire_locations, int(n_negative), self.tile_size
            )
        else:
            negative_tiles = self.generate_negative_samples(
                bands_data, fire_locations, int(n_negative)
            )
        
        for i, item in enumerate(negative_tiles):
            # Handle both old format (tile only) and new format (tile, metadata)
            if isinstance(item, tuple):
                tile, tile_metadata = item
            else:
                tile = item
                tile_metadata = None
            
            # Normalize (Sentinel-3 tiles are already normalized in extract_tile_at_location)
            if not is_sentinel3:
                tile = self.normalize_tile(tile)
            else:
                tile = tile.astype(np.float32)  # Ensure float32 dtype
            
            # Save with location metadata
            self.save_tile(tile, label=0, scene_id=scene_id,
                         fire_id='none', sample_type='negative',
                         fire_location=tile_metadata)
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
        
        # Count Sentinel-2 vs Sentinel-3 scenes
        if 'satellite' in matched_df.columns:
            s2_count = len(matched_df[matched_df['satellite'] == 'sentinel2'])
            s3_count = len(matched_df[matched_df['satellite'] == 'sentinel3'])
            logger.info(f"Dataset contains {s2_count} Sentinel-2 matches and {s3_count} Sentinel-3 matches")
            
            # Sort to prioritize Sentinel-2 (for faster debugging/verification)
            # Sentinel-2=0, Sentinel-3=1
            matched_df['sat_order'] = matched_df['satellite'].map({'sentinel2': 0, 'sentinel3': 1})
            matched_df = matched_df.sort_values('sat_order')
            
        else:
            logger.info(f"Dataset contains {len(matched_df)} matches (satellite type not specified)")
        
        # Group by scene
        scene_groups = matched_df.groupby('scene_id', sort=False)
        logger.info(f"Processing {len(scene_groups)} scenes (Sentinel-2 + Sentinel-3 with thermal bands)")
        
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
    parser.add_argument('--tile-size', type=int, default=None, 
                       help='Tile size in pixels (default: from config)')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator(args.config)
    
    # Override tile size if provided via CLI
    if args.tile_size:
        generator.tile_size = args.tile_size
        logger.info(f"Using CLI tile size: {generator.tile_size}x{generator.tile_size}")
    else:
        logger.info(f"Using config tile size: {generator.tile_size}x{generator.tile_size}")
    
    generator.run(args.matched_dataset)


if __name__ == "__main__":
    main()
