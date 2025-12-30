"""
Download Sentinel-2 imagery from STAC catalog for fire detection training.

This script queries the STAC API for Sentinel-2 L2A scenes over a specified
region and time period, then downloads the relevant spectral bands.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import yaml

import httpx
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# STAC API endpoints
STAC_API_URL = "https://earth-search.aws.element84.com/v1"


class Sentinel2Downloader:
    """Download and manage Sentinel-2 imagery."""
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize downloader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.raw_dir = Path(self.config['data']['raw_dir'])
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = httpx.AsyncClient(timeout=60.0)
        
    async def search_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        max_cloud_cover: Optional[float] = None
    ) -> List[Dict]:
        """
        Search for Sentinel-2 scenes in STAC catalog.
        
        Args:
            bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            max_cloud_cover: Maximum cloud cover percentage (0-100)
            
        Returns:
            List of STAC items (scenes)
        """
        if max_cloud_cover is None:
            max_cloud_cover = self.config['sentinel2']['max_cloud_cover']
        
        search_params = {
            "collections": [self.config['sentinel2']['collection']],
            "bbox": bbox,
            "datetime": f"{start_date}T00:00:00Z/{end_date}T23:59:59Z",
            "query": {
                "eo:cloud_cover": {"lt": max_cloud_cover}
            },
            "limit": 500  # Adjust as needed
        }
        
        logger.info(f"Searching STAC catalog: {start_date} to {end_date}")
        logger.info(f"Bbox: {bbox}, Max cloud cover: {max_cloud_cover}%")
        
        try:
            response = await self.client.post(
                f"{STAC_API_URL}/search",
                json=search_params
            )
            response.raise_for_status()
            data = response.json()
            
            features = data.get('features', [])
            logger.info(f"Found {len(features)} Sentinel-2 scenes")
            
            return features
            
        except httpx.HTTPError as e:
            logger.error(f"Error searching STAC: {e}")
            return []
    
    async def download_band(
        self,
        asset_url: str,
        output_path: Path,
        band_name: str
    ) -> bool:
        """
        Download a single band from S3/HTTP.
        
        Args:
            asset_url: URL to the band asset
            output_path: Local path to save the band
            band_name: Name of the band for logging
            
        Returns:
            True if successful, False otherwise
        """
        if output_path.exists():
            logger.debug(f"Band {band_name} already exists at {output_path}")
            return True
        
        try:
            logger.info(f"Downloading {band_name} from {asset_url[:80]}...")
            
            async with self.client.stream('GET', asset_url) as response:
                response.raise_for_status()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                with open(output_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=8192):
                        f.write(chunk)
            
            logger.info(f"Saved {band_name} to {output_path}")
            return True
            
        except httpx.HTTPError as e:
            logger.error(f"Error downloading {band_name}: {e}")
            return False
    
    async def download_scene(
        self,
        scene: Dict,
        bands: Optional[List[str]] = None
    ) -> Optional[Path]:
        """
        Download specified bands from a Sentinel-2 scene.
        
        Args:
            scene: STAC item containing scene metadata
            bands: List of band names to download (e.g., ['B04', 'B08'])
            
        Returns:
            Path to the scene directory, or None if failed
        """
        if bands is None:
            # Download both training bands and RGB bands
            bands = self.config['sentinel2']['bands']
            rgb_bands = self.config['sentinel2'].get('rgb_bands', [])
            
            # Combine and deduplicate (B04/red appears in both)
            all_bands = list(bands)
            for rgb_band in rgb_bands:
                if rgb_band not in all_bands:
                    all_bands.append(rgb_band)
            bands = all_bands
        
        # Get band save names mapping
        band_save_names = self.config['sentinel2'].get('band_save_names', [])
        rgb_save_names = self.config['sentinel2'].get('rgb_save_names', [])
        
        # Create combined save names mapping
        save_names_map = {}
        for i, band in enumerate(self.config['sentinel2']['bands']):
            if i < len(band_save_names):
                save_names_map[band] = band_save_names[i]
        
        rgb_bands_list = self.config['sentinel2'].get('rgb_bands', [])
        for i, band in enumerate(rgb_bands_list):
            if i < len(rgb_save_names):
                save_names_map[band] = rgb_save_names[i]
        
        scene_id = scene['id']
        scene_date = scene['properties'].get('datetime', '').split('T')[0]
        
        # Create scene directory
        scene_dir = self.raw_dir / scene_date / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scene metadata
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(scene, f, indent=2)
        
        # Download each band
        assets = scene.get('assets', {})
        download_tasks = []
        
        for band in bands:
            if band.lower() in assets:
                asset = assets[band.lower()]
                asset_url = asset.get('href')
                
                if asset_url:
                    # Use mapped save name if available, otherwise use band name
                    save_name = save_names_map.get(band, band)
                    output_path = scene_dir / f"{save_name}.tif"
                    download_tasks.append(
                        self.download_band(asset_url, output_path, save_name)
                    )
                else:
                    logger.warning(f"No href found for band {band} in scene {scene_id}")
            else:
                logger.warning(f"Band {band} not found in scene {scene_id}. Available: {list(assets.keys())[:10]}")
        
        # Download all bands concurrently
        if download_tasks:
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            
            if all(r is True for r in results):
                logger.info(f"Successfully downloaded scene {scene_id}")
                return scene_dir
            else:
                logger.warning(f"Some bands failed to download for scene {scene_id}")
                return scene_dir
        
        return None
    
    async def download_scenes_batch(
        self,
        scenes: List[Dict],
        max_concurrent: int = 3
    ) -> List[Path]:
        """
        Download multiple scenes with concurrency control.
        
        Args:
            scenes: List of STAC items to download
            max_concurrent: Maximum number of concurrent downloads
            
        Returns:
            List of paths to downloaded scene directories
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(scene):
            async with semaphore:
                return await self.download_scene(scene)
        
        tasks = [download_with_semaphore(scene) for scene in scenes]
        
        results = []
        for i, coro in enumerate(asyncio.as_completed(tasks)):
            try:
                result = await coro
                if result:
                    results.append(result)
                    logger.info(f"Progress: {len(results)}/{len(tasks)} scenes downloaded")
            except Exception as e:
                logger.error(f"Error in batch download: {e}")
        
        return results
    
    async def run(
        self,
        bbox: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        max_scenes: Optional[int] = None
    ):
        """
        Main execution: search and download scenes.
        
        Args:
            bbox: Bounding box override
            start_date: Start date override
            end_date: End date override
            max_scenes: Maximum number of scenes to download
        """
        # Use config defaults if not specified
        if bbox is None:
            bbox = self.config['region']['bbox']
        if start_date is None:
            start_date = self.config['time_range']['start_date']
        if end_date is None:
            end_date = self.config['time_range']['end_date']
        
        # Search for scenes
        scenes = await self.search_scenes(bbox, start_date, end_date)
        
        if not scenes:
            logger.warning("No scenes found matching criteria")
            return
        
        # Limit number of scenes if specified
        if max_scenes:
            scenes = scenes[:max_scenes]
            logger.info(f"Limited to {max_scenes} scenes")
        
        # Download scenes
        logger.info(f"Starting download of {len(scenes)} scenes...")
        downloaded = await self.download_scenes_batch(scenes)
        
        logger.info(f"Successfully downloaded {len(downloaded)}/{len(scenes)} scenes")
        logger.info(f"Data saved to: {self.raw_dir}")
        
        await self.client.aclose()


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Sentinel-2 imagery for fire detection")
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Path to config file')
    parser.add_argument('--bbox', nargs=4, type=float, help='Bounding box: lon_min lat_min lon_max lat_max')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--max-scenes', type=int, help='Maximum number of scenes to download')
    parser.add_argument('--max-cloud', type=float, help='Maximum cloud cover percentage')
    
    args = parser.parse_args()
    
    downloader = Sentinel2Downloader(args.config)
    
    # Override cloud cover if specified
    if args.max_cloud:
        downloader.config['sentinel2']['max_cloud_cover'] = args.max_cloud
    
    await downloader.run(
        bbox=args.bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        max_scenes=args.max_scenes
    )


if __name__ == "__main__":
    asyncio.run(main())
