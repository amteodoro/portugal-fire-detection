"""
Download Sentinel-2 imagery using OpenEO API for fire detection training.

Based on working Copernicus OpenEO implementation.
Uses OAuth2 authentication for reliable data access.
"""

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional
import yaml
import warnings

warnings.filterwarnings('ignore')

try:
    import openeo
except ImportError:
    print("Error: openeo library not installed")
    print("Install it with: uv pip install openeo")
    exit(1)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class Sentinel2OpenEODownloader:
    """Download and manage Sentinel-2 imagery using OpenEO."""
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize downloader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.connection = None
    
    def authenticate(self):
        """Connect and authenticate with OpenEO Copernicus backend."""
        logger.info("Connecting to OpenEO Copernicus Data Space...")
        
        try:
            # Connect to Copernicus OpenEO backend (working endpoint)
            self.connection = openeo.connect("https://openeo.dataspace.copernicus.eu")
            
            # Authenticate using OAuth2 (opens browser for login)
            self.connection.authenticate_oidc()
            
            logger.info("✓ Connected and authenticated with OpenEO")
            return True
            
        except Exception as e:
            logger.error(f"✗ Authentication failed: {e}")
            logger.error("Please ensure you have Copernicus account credentials")
            logger.error("Sign up at: https://dataspace.copernicus.eu/")
            return False
    
    def download_scene_batch(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        bands: Optional[List[str]] = None,
        max_cloud_cover: Optional[float] = None
    ) -> Optional[Path]:
        """
        Download Sentinel-2 scene using OpenEO batch jobs (for large areas).
        
        Args:
            bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            bands: List of Sentinel-2 band names (e.g., ['B04', 'B08', 'B11', 'B12'])
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            
        Returns:
            Path to the downloaded scene directory
        """
        if not self.connection:
            logger.error("Not authenticated. Call authenticate() first.")
            return None
        
        if bands is None:
            # Use config bands, but convert to S2 format (B04 instead of 'red')
            config_bands = self.config['sentinel2']['bands']
            # Map STAC names to S2 band names
            band_map = {'red': 'B04', 'nir': 'B08', 'swir16': 'B11', 'swir22': 'B12',
                       'blue': 'B02', 'green': 'B03'}
            bands = [band_map.get(b, b) for b in config_bands]
        
        if max_cloud_cover is None:
            max_cloud_cover = self.config['sentinel2']['max_cloud_cover']
        
        logger.info("Requesting Sentinel-2 data via OpenEO (Batch Job)")
        logger.info(f"  Bbox: {bbox}")
        logger.info(f"  Dates: {start_date} to {end_date}")
        logger.info(f"  Bands: {bands}")
        logger.info(f"  Max cloud: {max_cloud_cover}%")
        
        try:
            # Create spatial extent (west, south, east, north)
            spatial_extent = {
                "west": bbox[0],
                "south": bbox[1], 
                "east": bbox[2],
                "north": bbox[3]
            }
            
            # Load Sentinel-2 L2A collection
            cube = self.connection.load_collection(
                "SENTINEL2_L2A",
                spatial_extent=spatial_extent,
                temporal_extent=[f"{start_date}T00:00:00Z", f"{end_date}T23:59:59Z"],
                bands=bands
            )
            
            logger.info("✓ Data cube created successfully")
            
            # Generate unique scene ID from parameters
            scene_id = f"openeo_batch_{start_date}_{end_date}".replace('-', '')
            scene_id += f"_{abs(int(bbox[0]*10))}_{abs(int(bbox[1]*10))}"
            
            scene_dir = self.raw_dir / start_date / scene_id
            scene_dir.mkdir(parents=True, exist_ok=True)
            
            # Create and execute batch job
            logger.info("Creating batch job...")
            job = cube.save_result(format="GTiff").create_job(
                title=f"Sentinel-2 {start_date} to {end_date}",
                description=f"Bbox: {bbox}, Bands: {bands}"
            )
            
            logger.info(f"✓ Batch job created: {job.job_id}")
            logger.info("Starting batch job and waiting for completion...")
            logger.info("This may take several minutes. Progress updates will appear below.")
            
            # Start and wait with custom logging
            job.start_and_wait(
                print=lambda msg: logger.info(msg),
                max_poll_interval=30  # Check status every ~30 seconds
            )
            
            logger.info("✓ Batch job completed successfully")
            logger.info("Downloading results...")
            
            # Download results
            job.download_results(scene_dir)
            
            logger.info(f"✓ Downloaded results to {scene_dir}")
            
            # Rename files to match expected band names
            band_save_names = self.config['sentinel2'].get('band_save_names', bands)
            downloaded_files = list(scene_dir.glob("*.tif"))
            
            if len(downloaded_files) == len(bands):
                for i, (band, save_name) in enumerate(zip(bands, band_save_names)):
                    if i < len(downloaded_files):
                        old_path = downloaded_files[i]
                        new_path = scene_dir / f"{save_name}.tif"
                        if old_path != new_path and not new_path.exists():
                            old_path.rename(new_path)
                            logger.info(f"  Renamed {old_path.name} → {save_name}.tif")
            
            # Save metadata
            metadata = {
                "id": scene_id,
                "source": "openeo_batch",
                "job_id": job.job_id,
                "collection": "SENTINEL2_L2A",
                "bbox": bbox,
                "temporal_extent": [start_date, end_date],
                "bands": bands,
                "band_save_names": band_save_names,
                "max_cloud_cover": max_cloud_cover,
                "download_date": datetime.now().isoformat()
            }
            
            metadata_path = scene_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"✓ Successfully downloaded scene to {scene_dir}")
            return scene_dir
            
        except Exception as e:
            logger.error(f"Error downloading via OpenEO batch job: {e}")
            logger.exception(e)
            return None
    
    def run(
        self,
        bbox: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None
    ):
        """
        Main execution: authenticate and download scene using batch jobs.
        
        Args:
            bbox: Bounding box override
            start_date: Start date override
            end_date: End date override
        """
        # Authenticate first
        if not self.authenticate():
            logger.error("Authentication failed. Cannot proceed.")
            return
        
        # Use config defaults if not specified
        if bbox is None:
            bbox = self.config['region']['bbox']
        if start_date is None:
            start_date = self.config['time_range']['start_date']
        if end_date is None:
            end_date = self.config['time_range']['end_date']
        
        # Download scene using batch job (works for large areas)
        result = self.download_scene_batch(bbox, start_date, end_date)
        
        if result:
            logger.info(f"✓ Download complete: {result}")
        else:
            logger.error("✗ Download failed")


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Download Sentinel-2 imagery using OpenEO (more reliable than STAC)"
    )
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Path to config file')
    parser.add_argument('--bbox', nargs=4, type=float, help='Bounding box: lon_min lat_min lon_max lat_max')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-cloud', type=float, help='Maximum cloud cover percentage')
    
    args = parser.parse_args()
    
    downloader = Sentinel2OpenEODownloader(args.config)
    
    # Override cloud cover if specified
    if args.max_cloud:
        downloader.config['sentinel2']['max_cloud_cover'] = args.max_cloud
    
    downloader.run(
        bbox=args.bbox,
        start_date=args.start_date,
        end_date=args.end_date
    )


if __name__ == "__main__":
    main()
