"""
Sentinel-3 SLSTR thermal data processor for fire detection.

Processes Sentinel-3 SLSTR NetCDF files to extract thermal bands
(F1: 3.74µm, F2: 10.85µm) for active fire detection.
"""

import logging
import zipfile
from pathlib import Path
from typing import Dict, Optional, Tuple
import tempfile
import shutil

import numpy as np
import xarray as xr
from scipy.interpolate import griddata

logger = logging.getLogger(__name__)


class Sentinel3Processor:
    """Process Sentinel-3 SLSTR thermal data for fire detection."""
    
    def __init__(self):
        """Initialize processor."""
        # Thermal bands for fire detection
        # F1 uses fire grid (fn/fo), F2 uses infrared grid (in/io)
        self.thermal_bands = {
            'F1': ('F1_BT', 'f'),  # 3.74µm - Mid-infrared, fire grid
            'F2': ('F2_BT', 'i')   # 10.85µm - Thermal infrared, infrared grid
        }
        
        # SWIR bands (500m resolution, useful context)
        self.swir_bands = {
            'S5': ('S5_radiance', 'a'),  # 1.6µm
            'S6': ('S6_radiance', 'a')   # 2.25µm
        }
    
    def extract_zip(self, zip_path: Path) -> Optional[Path]:
        """
        Extract Sentinel-3 ZIP file to temporary directory.
        
        Args:
            zip_path: Path to .zip file
            
        Returns:
            Path to extracted .SEN3 directory, or None if failed
        """
        try:
            temp_dir = Path(tempfile.mkdtemp(prefix='s3_'))
            
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Find the .SEN3 directory
            sen3_dirs = list(temp_dir.glob('*.SEN3'))
            if not sen3_dirs:
                logger.error(f"No .SEN3 directory found in {zip_path}")
                shutil.rmtree(temp_dir)
                return None
            
            return sen3_dirs[0]
        
        except Exception as e:
            logger.error(f"Error extracting {zip_path}: {e}")
            return None
    
    def load_band(
        self,
        sen3_dir: Path,
        band_prefix: str,
        grid_type: str,
        view: str = 'n'
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Load a single SLSTR band from NetCDF.
        
        Args:
            sen3_dir: Path to .SEN3 directory
            band_prefix: Band prefix (e.g., 'F1_BT', 'F2_BT')
            grid_type: Grid type - 'f' for fire grid, 'i' for infrared, 'a' for radiance
            view: 'n' for nadir, 'o' for oblique (default: nadir)
            
        Returns:
            Tuple of (data array, metadata dict) or None
        """
        # Construct filename pattern
        # F1_BT uses fire grid: F1_BT_fn.nc (fire grid nadir)
        # F2_BT uses infrared grid: F2_BT_in.nc (infrared nadir)
        # S5/S6 use radiance grid: S5_radiance_an.nc (radiance nadir)
        suffix = f'{grid_type}{view}.nc'
        band_file = sen3_dir / f"{band_prefix}_{suffix}"
        
        if not band_file.exists():
            # Try alternate view if nadir not found
            if view == 'n':
                logger.debug(f"Nadir view not found for {band_prefix}, trying oblique")
                return self.load_band(sen3_dir, band_prefix, grid_type, view='o')
            logger.debug(f"Band file not found: {band_file}")
            return None
        
        try:
            with xr.open_dataset(band_file) as ds:
                # Get the main variable (usually first data variable)
                var_names = [v for v in ds.data_vars if not v.endswith('_quality')]
                if not var_names:
                    logger.error(f"No data variables found in {band_file}")
                    return None
                
                var_name = var_names[0]
                data = ds[var_name].values
                
                # Get coordinate arrays
                metadata = {
                    'shape': data.shape,
                    'view': view,
                    'variable': var_name
                }
                
                # Extract lat/lon grids if available
                if 'latitude_in' in ds or 'latitude_an' in ds:
                    lat_var = 'latitude_in' if 'latitude_in' in ds else 'latitude_an'
                    lon_var = 'longitude_in' if 'longitude_in' in ds else 'longitude_an'
                    metadata['lat'] = ds[lat_var].values
                    metadata['lon'] = ds[lon_var].values
                
                return data, metadata
        
        except Exception as e:
            logger.error(f"Error reading {band_file}: {e}")
            return None
    
    def load_scene_bands(
        self,
        scene_path: Path
    ) -> Optional[Dict[str, np.ndarray]]:
        """
        Load thermal and SWIR bands from Sentinel-3 scene.
        
        Args:
            scene_path: Path to scene directory containing .zip file
            
        Returns:
            Dictionary with band data and metadata, or None
        """
        # Find ZIP file - may be directly in scene_path or in a same-named subdirectory
        zip_files = list(scene_path.glob('*.zip'))
        
        if not zip_files:
            # Check for nested structure: scene_id/scene_id.zip
            zip_files = list(scene_path.glob('**/*.zip'))
        
        if not zip_files:
            logger.error(f"No ZIP file found in {scene_path} or subdirectories")
            return None
        
        zip_path = zip_files[0]
        logger.debug(f"Found ZIP: {zip_path}")
        
        # Extract ZIP
        sen3_dir = self.extract_zip(zip_path)
        if not sen3_dir:
            return None
        
        try:
            bands_data = {}
            
            # Load thermal bands (F1, F2) - these are 1km resolution
            for band_name, (band_prefix, grid_type) in self.thermal_bands.items():
                result = self.load_band(sen3_dir, band_prefix, grid_type)
                if result is None:
                    logger.debug(f"Could not load {band_name}")
                    continue
                
                data, metadata = result
                bands_data[band_name] = data
                bands_data[f'{band_name}_meta'] = metadata
            
            # Load SWIR bands (S5, S6) - these are 500m resolution (optional)
            for band_name, (band_prefix, grid_type) in self.swir_bands.items():
                result = self.load_band(sen3_dir, band_prefix, grid_type)
                if result is None:
                    logger.debug(f"Could not load {band_name}")
                    continue
                
                data, metadata = result
                bands_data[band_name] = data
                bands_data[f'{band_name}_meta'] = metadata
            
            # Check if we have minimum required bands (at least one thermal band)
            if 'F1' not in bands_data and 'F2' not in bands_data:
                logger.debug("Missing both thermal bands F1 and F2")
                return None
            
            # Load geodetic coordinates from separate file
            # Use fire grid (fn) if F1 available, else infrared grid (in) for F2
            if 'F1' in bands_data:
                grid_suffix = 'fn'  # Fire grid nadir
            else:
                grid_suffix = 'in'  # Infrared grid nadir
            
            geodetic_file = sen3_dir / f"geodetic_{grid_suffix}.nc"
            if geodetic_file.exists():
                try:
                    with xr.open_dataset(geodetic_file) as geo_ds:
                        bands_data['lat'] = geo_ds[f'latitude_{grid_suffix}'].values
                        bands_data['lon'] = geo_ds[f'longitude_{grid_suffix}'].values
                        logger.debug(f"Loaded geodetic coordinates from {geodetic_file.name}")
                except Exception as e:
                    logger.warning(f"Error loading geodetic file {geodetic_file}: {e}")
                    return None
            else:
                logger.warning(f"Geodetic file not found: {geodetic_file}")
                return None
            
            return bands_data
        
        finally:
            # Cleanup temporary directory
            if sen3_dir and sen3_dir.exists():
                shutil.rmtree(sen3_dir.parent)
    
    def find_nearest_pixel(
        self,
        lat_grid: np.ndarray,
        lon_grid: np.ndarray,
        target_lat: float,
        target_lon: float
    ) -> Optional[Tuple[int, int]]:
        """
        Find nearest pixel to target lat/lon in a grid.
        
        Args:
            lat_grid: 2D array of latitudes
            lon_grid: 2D array of longitudes
            target_lat: Target latitude
            target_lon: Target longitude
            
        Returns:
            Tuple of (row, col) indices, or None
        """
        try:
            # Calculate distance from target
            dist = np.sqrt((lat_grid - target_lat)**2 + (lon_grid - target_lon)**2)
            
            # Find minimum distance
            min_idx = np.unravel_index(np.argmin(dist), dist.shape)
            
            return min_idx
        
        except Exception as e:
            logger.error(f"Error finding nearest pixel: {e}")
            return None
    
    def extract_tile_at_location(
        self,
        bands_data: Dict[str, np.ndarray],
        lon: float,
        lat: float,
        tile_size: int = 256
    ) -> Optional[Tuple[np.ndarray, Dict]]:
        """
        Extract thermal tile centered at geographic location.
        
        FIXED: Always outputs 4 bands compatible with Sentinel-2 format:
        - Band 0: Red proxy (from S5 SWIR 1.6µm, or scaled F1)
        - Band 1: NIR proxy (from S6 SWIR 2.25µm, or scaled F2)
        - Band 2: F1 thermal (3.74µm) - mapped to SWIR1 slot
        - Band 3: F2 thermal (10.85µm) - mapped to SWIR2 slot
        
        Args:
            bands_data: Dictionary of band arrays and metadata
            lon: Longitude of center point
            lat: Latitude of center point
            tile_size: Size of tile in pixels
            
        Returns:
            Tuple of (tile array with 4 bands, metadata dict), or None
        """
        if 'lat' not in bands_data or 'lon' not in bands_data:
            logger.error("No lat/lon grids available")
            return None
        
        lat_grid = bands_data['lat']
        lon_grid = bands_data['lon']
        
        # Find nearest pixel
        center = self.find_nearest_pixel(lat_grid, lon_grid, lat, lon)
        if center is None:
            return None
        
        row, col = center
        half_size = tile_size // 2
        
        # Calculate tile boundaries
        row_start = max(0, row - half_size)
        row_end = min(lat_grid.shape[0], row + half_size)
        col_start = max(0, col - half_size)
        col_end = min(lat_grid.shape[1], col + half_size)
        
        # Check if tile is large enough
        if (row_end - row_start) < tile_size * 0.8 or (col_end - col_start) < tile_size * 0.8:
            logger.debug("Tile too close to edge")
            return None
        
        # Helper function to extract and validate a single band tile
        def extract_band_tile(band_name: str) -> Optional[np.ndarray]:
            if band_name not in bands_data:
                return None
            
            band_array = bands_data[band_name]
            
            # Handle different grid resolutions (SWIR is 500m, thermal is 1km)
            # Scale row/col indices for bands with different resolution
            scale_factor = band_array.shape[0] / lat_grid.shape[0]
            
            if scale_factor != 1.0:
                # Adjust indices for this band's resolution
                band_row_start = int(row_start * scale_factor)
                band_row_end = int(row_end * scale_factor)
                band_col_start = int(col_start * scale_factor)
                band_col_end = int(col_end * scale_factor)
            else:
                band_row_start, band_row_end = row_start, row_end
                band_col_start, band_col_end = col_start, col_end
            
            # Clamp to array bounds
            band_row_start = max(0, band_row_start)
            band_row_end = min(band_array.shape[0], band_row_end)
            band_col_start = max(0, band_col_start)
            band_col_end = min(band_array.shape[1], band_col_end)
            
            # Extract tile
            tile = band_array[band_row_start:band_row_end, band_col_start:band_col_end]
            
            # Calculate required padding to reach tile_size
            actual_rows, actual_cols = tile.shape
            target_size = tile_size if scale_factor == 1.0 else int(tile_size * scale_factor)
            
            pad_rows = max(0, target_size - actual_rows)
            pad_cols = max(0, target_size - actual_cols)
            
            # Pad if needed (pad at the end to preserve fire location)
            if pad_rows > 0 or pad_cols > 0:
                tile = np.pad(tile, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
            
            # Ensure exact size (may need to crop if slightly larger)
            tile = tile[:target_size, :target_size]
            
            # Resample to standard tile_size if different resolution
            if scale_factor != 1.0 and target_size != tile_size:
                from scipy.ndimage import zoom
                zoom_factor = tile_size / target_size
                tile = zoom(tile, zoom_factor, order=1)
                tile = tile[:tile_size, :tile_size]  # Ensure exact size after zoom
            
            # Check for fill values or invalid data
            valid_mask = (tile > -999) & (tile < 1e10)
            if np.sum(valid_mask) < (tile_size * tile_size * 0.3):  # Relaxed threshold
                logger.debug(f"Too many invalid pixels in {band_name}: {np.sum(~valid_mask)}/{tile.size}")
                return None
            
            # Replace fill values with 0
            tile = np.where(valid_mask, tile, 0)
            return tile
        
        # Extract available bands
        f1_tile = extract_band_tile('F1')
        f2_tile = extract_band_tile('F2')
        s5_tile = extract_band_tile('S5')
        s6_tile = extract_band_tile('S6')
        
        # Need at least one thermal band
        if f1_tile is None and f2_tile is None:
            logger.debug("No valid thermal bands found")
            return None
        
        # Initialize 4-band output array
        output_tile = np.zeros((4, tile_size, tile_size), dtype=np.float32)
        
        # Band 0 (Red proxy): Prefer S5 (1.6µm SWIR), else scale F1
        if s5_tile is not None:
            # S5 is radiance, normalize to 0-1 range (typical S5 range: 0-100)
            output_tile[0] = np.clip(s5_tile / 100.0, 0, 1)
        elif f1_tile is not None:
            # F1 is brightness temperature (Kelvin), scale to reflectance-like range
            # Typical range: 250-350K, map to 0-1
            output_tile[0] = np.clip((f1_tile - 250) / 100.0, 0, 1)
        elif f2_tile is not None:
            # Fallback: use F2 scaled
            output_tile[0] = np.clip((f2_tile - 250) / 100.0, 0, 1)
        
        # Band 1 (NIR proxy): Prefer S6 (2.25µm SWIR), else scale F2
        if s6_tile is not None:
            # S6 is radiance, normalize to 0-1 range
            output_tile[1] = np.clip(s6_tile / 100.0, 0, 1)
        elif f2_tile is not None:
            # F2 is brightness temperature, scale to reflectance-like range
            output_tile[1] = np.clip((f2_tile - 250) / 100.0, 0, 1)
        elif f1_tile is not None:
            # Fallback: use F1 scaled
            output_tile[1] = np.clip((f1_tile - 250) / 100.0, 0, 1)
        
        # Band 2 (SWIR1 slot): F1 thermal (3.74µm)
        if f1_tile is not None:
            # Normalize F1: typical fire range 300-400K, map to 0-1
            # For visualization, fires (>350K) should be bright
            output_tile[2] = np.clip((f1_tile - 270) / 100.0, 0, 1)
        elif f2_tile is not None:
            # If F1 missing, duplicate F2
            output_tile[2] = np.clip((f2_tile - 270) / 100.0, 0, 1)
        
        # Band 3 (SWIR2 slot): F2 thermal (10.85µm)
        if f2_tile is not None:
            # Normalize F2: typical range 280-320K, map to 0-1
            output_tile[3] = np.clip((f2_tile - 270) / 80.0, 0, 1)
        elif f1_tile is not None:
            # If F2 missing, duplicate F1
            output_tile[3] = np.clip((f1_tile - 270) / 80.0, 0, 1)
        
        # Calculate fire location within tile (centered by design for S3)
        fire_row_in_tile = row - row_start
        fire_col_in_tile = col - col_start
        
        # Track which original bands were available
        available_bands = []
        if f1_tile is not None:
            available_bands.append('F1')
        if f2_tile is not None:
            available_bands.append('F2')
        if s5_tile is not None:
            available_bands.append('S5')
        if s6_tile is not None:
            available_bands.append('S6')
        
        metadata = {
            'fire_pixel_row': int(fire_row_in_tile),
            'fire_pixel_col': int(fire_col_in_tile),
            'scene_row_start': int(row_start),
            'scene_col_start': int(col_start),
            'tile_size': int(tile_size),
            'bands': ['red_proxy', 'nir_proxy', 'swir1_thermal_f1', 'swir2_thermal_f2'],  # Output band names
            'source_bands': available_bands,  # Original S3 bands used
            'satellite': 'sentinel3',
            'confirmed_fire_location': True  # S3 tiles center fires by design
        }
        
        logger.debug(f"Created 4-band S3 tile from source bands: {available_bands}")
        
        return output_tile, metadata

    def normalize_tile(self, tile: np.ndarray, is_thermal: bool = True) -> np.ndarray:
        """
        Normalize Sentinel-3 tile values.
        
        Args:
            tile: Tile array of shape (bands, height, width)
            is_thermal: True if thermal bands (brightness temperature)
            
        Returns:
            Normalized tile
        """
        tile = tile.copy()
        
        # Replace fill values with 0
        tile[tile < -999] = 0
        tile[tile > 1e10] = 0
        
        if is_thermal:
            # Thermal bands are in Kelvin (typically 250-350K)
            # Normalize to 0-1 range
            # Typical range: 250K (cold) to 400K (very hot fire)
            tile = np.clip(tile, 250, 400)
            tile = (tile - 250) / 150.0
        else:
            # SWIR radiance - normalize by typical max values
            tile = np.clip(tile, 0, 500)
            tile = tile / 500.0
        
        return tile.astype(np.float32)
    
    def generate_negative_samples(
        self,
        bands_data: Dict[str, np.ndarray],
        fire_locations: list,
        n_samples: int,
        tile_size: int = 256,
        min_distance_deg: float = 0.05
    ) -> list:
        """
        Generate negative samples from Sentinel-3 scene.
        
        Args:
            bands_data: Dictionary of band arrays
            fire_locations: List of (lon, lat) fire locations
            n_samples: Number of negative samples to generate
            tile_size: Tile size in pixels
            min_distance_deg: Minimum distance from fires in degrees
            
        Returns:
            List of tile arrays
        """
        import random
        
        negative_tiles = []
        
        if 'lat' not in bands_data or 'lon' not in bands_data:
            return []
        
        lat_grid = bands_data['lat']
        lon_grid = bands_data['lon']
        height, width = lat_grid.shape
        
        max_attempts = n_samples * 10
        attempts = 0
        
        while len(negative_tiles) < n_samples and attempts < max_attempts:
            attempts += 1
            
            # Random pixel location
            row = random.randint(tile_size, height - tile_size)
            col = random.randint(tile_size, width - tile_size)
            
            # Get lat/lon at this location
            lat = lat_grid[row, col]
            lon = lon_grid[row, col]
            
            # Check distance from all fires
            too_close = False
            for fire_lon, fire_lat in fire_locations:
                dist = np.sqrt((lon - fire_lon)**2 + (lat - fire_lat)**2)
                if dist < min_distance_deg:
                    too_close = True
                    break
            
            if too_close:
                continue
            
            # Extract tile
            result = self.extract_tile_at_location(bands_data, lon, lat, tile_size)
            
            if result is not None:
                tile, _ = result
                negative_tiles.append(tile)
        
        return negative_tiles

