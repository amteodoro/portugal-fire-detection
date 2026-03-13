"""
Download Sentinel-3 SLSTR and OLCI data from STAC catalog for fire detection.

Sentinel-3 SLSTR provides thermal IR bands (3.74µm, 10.85µm) for active fire detection
and FRP estimation, while OLCI provides optical bands for burn scar mapping.
"""

import asyncio
import json
import logging
import os
import webbrowser
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
import yaml

import httpx
from tqdm import tqdm
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# OpenSearch API endpoint for Copernicus Data Space Ecosystem
# Sentinel-3 is primarily accessed via OpenSearch, not STAC
OPENSEARCH_API_URL = "https://catalogue.dataspace.copernicus.eu/resto/api/collections/Sentinel3/search.json"
TOKEN_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/token"
DEVICE_AUTH_URL = "https://identity.dataspace.copernicus.eu/auth/realms/CDSE/protocol/openid-connect/auth/device"
TOKEN_CACHE_FILE = Path.home() / ".copernicus_token_cache.json"


class Sentinel3Downloader:
    """Download and manage Sentinel-3 SLSTR and OLCI imagery."""

    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize downloader with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.sentinel3_dir = Path(self.config['data']['sentinel3_dir'])
        self.sentinel3_dir.mkdir(parents=True, exist_ok=True)

        # Load OAuth client credentials
        self.client_id = os.getenv('COPERNICUS_CLIENT_ID')
        self.client_secret = os.getenv('COPERNICUS_CLIENT_SECRET')

        if not self.client_id or not self.client_secret:
            logger.warning("COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET not set in .env")
            logger.warning("Downloads will fail. Register OAuth client at: https://dataspace.copernicus.eu/")

        self.client = httpx.AsyncClient(
            timeout=120.0,
            follow_redirects=True,
        )
        self.access_token = None
        
    def _load_cached_token(self) -> Optional[str]:
        """Load cached access token if it exists and is still valid."""
        try:
            if TOKEN_CACHE_FILE.exists():
                with open(TOKEN_CACHE_FILE, 'r') as f:
                    cache = json.load(f)
                    # Simple expiry check (tokens typically last 10 minutes)
                    expires_at = cache.get('expires_at', 0)
                    if datetime.now().timestamp() < expires_at:
                        return cache.get('access_token')
        except Exception as e:
            logger.debug(f"Could not load cached token: {e}")
        return None
    
    def _save_token_cache(self, access_token: str, expires_in: int = 600):
        """Save access token to cache file."""
        try:
            cache = {
                'access_token': access_token,
                'expires_at': datetime.now().timestamp() + expires_in - 60  # 1 min buffer
            }
            with open(TOKEN_CACHE_FILE, 'w') as f:
                json.dump(cache, f)
            TOKEN_CACHE_FILE.chmod(0o600)  # Secure permissions
        except Exception as e:
            logger.warning(f"Could not save token cache: {e}")
    
    async def get_access_token(self) -> Optional[str]:
        """
        Get OAuth2 access token for CDSE file downloads.

        Priority order:
        1. COPERNICUS_ACCESS_TOKEN env var (pre-issued token, skips OAuth entirely)
        2. Cached token from a previous OAuth flow
        3. Password grant with cdse-public client (required for file downloads)
        4. client_credentials grant (fallback; works for APIs but NOT for downloads)

        Returns:
            Access token string, or None if authentication fails
        """
        # 1. Pre-issued token bypass
        direct_token = os.getenv("COPERNICUS_ACCESS_TOKEN", "").strip()
        if direct_token:
            logger.info("Using COPERNICUS_ACCESS_TOKEN from environment")
            return direct_token

        # 2. Try cached token first
        cached_token = self._load_cached_token()
        if cached_token:
            logger.info("Using cached authentication token")
            return cached_token

        # 3. Password grant using cdse-public (the CDSE-recommended flow for downloads)
        #    See: https://documentation.dataspace.copernicus.eu/APIs/Token.html
        username = os.getenv("COPERNICUS_USERNAME", "").strip()
        password = os.getenv("COPERNICUS_PASSWORD", "").strip()
        if username and password:
            try:
                logger.info("Authenticating with Copernicus Data Space (password grant)...")
                response = await self.client.post(
                    TOKEN_URL,
                    data={
                        "client_id": "cdse-public",
                        "username": username,
                        "password": password,
                        "grant_type": "password",
                    },
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                response.raise_for_status()
                token_data = response.json()
                access_token = token_data.get("access_token")
                expires_in = token_data.get("expires_in", 600)
                if access_token:
                    logger.info("✓ Successfully authenticated via password grant")
                    self._save_token_cache(access_token, expires_in)
                    return access_token
                logger.error("No access token in password grant response")
            except httpx.HTTPError as e:
                logger.warning(f"Password grant failed: {e}")
                try:
                    logger.warning(f"Response: {response.json()}")
                except Exception:
                    pass

        # 4. client_credentials fallback (note: CDSE download endpoint rejects these tokens)
        if not self.client_id or not self.client_secret:
            logger.error(
                "No credentials configured for downloads. "
                "Set COPERNICUS_USERNAME and COPERNICUS_PASSWORD in .env "
                "(these are your https://dataspace.copernicus.eu/ login credentials). "
                "Alternatively paste a pre-issued Bearer token into COPERNICUS_ACCESS_TOKEN."
            )
            return None

        try:
            logger.info("Authenticating with Copernicus Data Space (client credentials)...")
            response = await self.client.post(
                TOKEN_URL,
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            response.raise_for_status()
            token_data = response.json()
            access_token = token_data.get("access_token")
            expires_in = token_data.get("expires_in", 600)
            if access_token:
                logger.warning(
                    "Authenticated via client_credentials. "
                    "Note: the CDSE download endpoint may reject this token with 401. "
                    "Set COPERNICUS_USERNAME + COPERNICUS_PASSWORD in .env for reliable downloads."
                )
                self._save_token_cache(access_token, expires_in)
                return access_token
            logger.error("No access token in client_credentials response")
            return None
        except httpx.HTTPError as e:
            logger.error(f"Authentication failed: {e}")
            try:
                error_detail = response.json()
                logger.error(f"Error details: {error_detail}")
                if error_detail.get("error") == "invalid_client":
                    logger.error(
                        "The OAuth client credentials were rejected. "
                        "Go to https://dataspace.copernicus.eu/ → User Settings → OAuth Clients "
                        "to verify or recreate the client."
                    )
            except Exception:
                pass
            return None
    
    # ------------------------------------------------------------------
    # S3 download helpers (preferred over OAuth-based HTTP download)
    # Credentials: https://eodata-s3keysmanager.dataspace.copernicus.eu/
    # Set COPERNICUS_S3_ACCESS_KEY and COPERNICUS_S3_SECRET_KEY in .env
    # ------------------------------------------------------------------

    def _get_s3_resource(self):
        """Return a boto3 S3 resource for the CDSE eodata bucket, or None."""
        access_key = os.getenv('COPERNICUS_S3_ACCESS_KEY', '').strip()
        secret_key = os.getenv('COPERNICUS_S3_SECRET_KEY', '').strip()
        if not access_key or not secret_key:
            return None
        try:
            import boto3
            return boto3.resource(
                's3',
                endpoint_url='https://eodata.dataspace.copernicus.eu',
                aws_access_key_id=access_key,
                aws_secret_access_key=secret_key,
                region_name='default',
            )
        except ImportError:
            logger.warning("boto3 not installed; S3 download unavailable")
            return None

    _S3_INSTRUMENT_DIRS = {
        'SL_1_RBT___': 'SLSTR/SL_1_RBT___',
        'SL_2_FRP___': 'SLSTR/SL_2_FRP___',
        'SL_2_LST___': 'SLSTR/SL_2_LST___',
        'OL_1_EFR___': 'OLCI/OL_1_EFR___',
        'OL_2_LFR___': 'OLCI/OL_2_LFR___',
    }

    def _s3_prefix(self, scene: Dict) -> Optional[str]:
        """Return the S3 object prefix for a scene, or None if not derivable."""
        props = scene.get('properties', {})
        title = props.get('title', '').strip()
        if not title:
            return None
        if not title.endswith('.SEN3'):
            title = title + '.SEN3'

        start_date = props.get('startDate', props.get('datetime', ''))
        if not start_date:
            return None
        year, month, day = start_date[:10].split('-')

        product_type = props.get('productType', '')
        inst_dir = self._S3_INSTRUMENT_DIRS.get(product_type)
        if not inst_dir:
            logger.warning(f"No S3 path known for product type {product_type!r}")
            return None

        return f"Sentinel-3/{inst_dir}/{year}/{month}/{day}/{title}/"

    def _sync_s3_download(self, prefix: str, output_dir: Path) -> bool:
        """Synchronously download all objects under prefix to output_dir."""
        s3 = self._get_s3_resource()
        if s3 is None:
            return False
        try:
            bucket = s3.Bucket('eodata')
            objects = list(bucket.objects.filter(Prefix=prefix))
            if not objects:
                logger.warning(f"S3: no objects at s3://eodata/{prefix}")
                return False

            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"S3: downloading {len(objects)} files from s3://eodata/{prefix}")
            for obj in objects:
                filename = Path(obj.key).name
                if not filename:
                    continue
                local_path = output_dir / filename
                if local_path.exists():
                    logger.debug(f"S3: skip {filename} (exists)")
                    continue
                logger.info(f"S3: {filename} ({obj.size / (1024*1024):.1f} MB)")
                bucket.download_file(obj.key, str(local_path))
            logger.info(f"S3 download complete → {output_dir}")
            return True
        except Exception as e:
            logger.error(f"S3 download failed: {e}")
            return False

    async def _download_via_s3(self, scene: Dict, output_dir: Path) -> bool:
        """Try to download a scene via S3. Non-blocking (runs in thread executor)."""
        prefix = self._s3_prefix(scene)
        if not prefix:
            return False
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._sync_s3_download, prefix, output_dir)

    async def search_scenes(
        self,
        bbox: List[float],
        start_date: str,
        end_date: str,
        instrument: str = "slstr"  # "slstr" or "olci"
    ) -> List[Dict]:
        """
        Search for Sentinel-3 scenes using OpenSearch API.
        
        Args:
            bbox: Bounding box [lon_min, lat_min, lon_max, lat_max]
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            instrument: "slstr" for thermal/fire data, "olci" for optical
            
        Returns:
            List of scenes (converted to STAC-like format)
        """
        # Map instrument to Sentinel-3 product types
        product_type_map = {
            "slstr": "SL_1_RBT___",  # SLSTR Level-1 Radiance and Brightness Temperature
            "olci": "OL_1_EFR___"    # OLCI Level-1 Full Resolution
        }
        
        product_type = product_type_map.get(instrument.lower(), "SL_1_RBT___")
        
        # OpenSearch parameters for Copernicus
        search_params = {
            "box": f"{bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}",  # lon_min,lat_min,lon_max,lat_max
            "startDate": f"{start_date}T00:00:00Z",
            "completionDate": f"{end_date}T23:59:59Z",
            "productType": product_type,
            "maxRecords": 100,  # Limit results
            "sortParam": "startDate",
            "sortOrder": "descending"
        }
        
        logger.info(f"Searching OpenSearch for Sentinel-3 {instrument.upper()}: {start_date} to {end_date}")
        logger.info(f"Bbox: {bbox}")
        logger.info(f"Product Type: {product_type}")
        
        try:
            response = await self.client.get(
                OPENSEARCH_API_URL,
                params=search_params
            )
            response.raise_for_status()
            data = response.json()
            
            # OpenSearch returns features in a different format
            features = data.get('features', [])
            logger.info(f"Found {len(features)} Sentinel-3 {instrument.upper()} scenes")
            
            return features
            
        except httpx.HTTPError as e:
            logger.error(f"Error searching OpenSearch: {e}")
            try:
                error_detail = response.json()
                logger.error(f"API response: {error_detail}")
            except:
                pass
            logger.warning("Note: Copernicus Data Space may require authentication for full access")
            return []
    
    async def download_band(
        self,
        asset_url: str,
        output_path: Path,
        band_name: str,
        _retry: bool = True
    ) -> bool:
        """
        Download a single band from Copernicus.
        
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
        
        # Ensure we have authentication token
        if not self.access_token:
            self.access_token = await self.get_access_token()
            if not self.access_token:
                logger.error("Cannot download without authentication token")
                return False
        
        try:
            # httpx strips the Authorization header from cross-host redirect legs
            # (RFC-compliant behaviour; no hook or Auth class can reliably override it).
            # The download URL redirects from catalogue.dataspace.copernicus.eu to
            # download.dataspace.copernicus.eu, so we resolve the redirect manually:
            # step 1 — HEAD request to get the Location URL without following it;
            # step 2 — stream directly from that URL with the token header attached.
            auth_header = {"Authorization": f"Bearer {self.access_token}"}
            pre = await self.client.head(
                asset_url, follow_redirects=False, headers=auth_header
            )
            download_url = pre.headers.get("location", asset_url) if pre.is_redirect else asset_url
            logger.info(f"Downloading {band_name} from {download_url[:80]}...")

            async with self.client.stream(
                "GET", download_url, follow_redirects=False, headers=auth_header
            ) as response:
                response.raise_for_status()
                
                output_path.parent.mkdir(parents=True, exist_ok=True)
                
                # Get file size if available
                total_size = response.headers.get('content-length')
                if total_size:
                    total_size = int(total_size)
                    logger.info(f"File size: {total_size / (1024*1024):.1f} MB")
                
                downloaded = 0
                last_log_mb = 0
                
                with open(output_path, 'wb') as f:
                    async for chunk in response.aiter_bytes(chunk_size=1024*1024):  # 1MB chunks
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Log progress every 50MB
                        current_mb = downloaded / (1024*1024)
                        if current_mb - last_log_mb >= 50:
                            if total_size:
                                progress = (downloaded / total_size) * 100
                                logger.info(f"Progress: {current_mb:.0f}MB / {total_size/(1024*1024):.0f}MB ({progress:.1f}%)")
                            else:
                                logger.info(f"Downloaded: {current_mb:.0f}MB")
                            last_log_mb = current_mb
            
            logger.info(f"Saved {band_name} to {output_path} ({downloaded/(1024*1024):.1f}MB)")
            return True
            
        except httpx.HTTPStatusError as e:
            # Handle HTTP errors with status codes
            status = e.response.status_code if e.response is not None else None
            logger.error(f"Error downloading {band_name}: {e}")
            if status == 401:
                logger.error("Authentication failed. Token may have expired. Refreshing token and retrying once...")
                # Clear cached token so next call fetches a fresh one
                self.access_token = None
                try:
                    if TOKEN_CACHE_FILE.exists():
                        TOKEN_CACHE_FILE.unlink()
                except Exception:
                    pass
                if _retry:
                    self.access_token = await self.get_access_token()
                    return await self.download_band(asset_url, output_path, band_name, _retry=False)
            elif status == 403:
                logger.error("Access forbidden. Check your OAuth client permissions.")
            return False
        except httpx.HTTPError as e:
            logger.error(f"Network error downloading {band_name}: {e}")
            return False
    
    async def download_slstr_scene(
        self,
        scene: Dict,
        bands: Optional[List[str]] = None
    ) -> Optional[Path]:
        """
        Download SLSTR bands from a Sentinel-3 scene.
        
        Args:
            scene: STAC item containing scene metadata
            bands: List of SLSTR band names (e.g., ['F1', 'F2', 'S5', 'S6'])
            
        Returns:
            Path to the scene directory, or None if failed
        """
        if bands is None:
            # Get all configured SLSTR bands
            slstr_config = self.config['sentinel3']['slstr_bands']
            bands = (slstr_config.get('thermal', []) + 
                    slstr_config.get('swir', []) + 
                    slstr_config.get('visible', []))
        
        # Handle both STAC and OpenSearch formats
        scene_id = scene.get('id') or scene.get('properties', {}).get('title', 'unknown')
        scene_props = scene.get('properties', {})
        scene_date = scene_props.get('datetime', scene_props.get('startDate', '')).split('T')[0]
        
        # Create scene directory
        scene_dir = self.sentinel3_dir / "slstr" / scene_date / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scene metadata
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(scene, f, indent=2)

        # Try S3 first — bypasses OAuth entirely
        if await self._download_via_s3(scene, scene_dir):
            logger.info(f"Successfully downloaded SLSTR scene {scene_id} via S3")
            return scene_dir

        # Fall back to HTTP download (requires valid user token)
        logger.info(f"S3 unavailable, falling back to HTTP for SLSTR scene {scene_id}")
        download_url = None
        services = scene_props.get('services', {})
        download_service = services.get('download', {})
        download_url = download_service.get('url')

        if download_url:
            logger.info(f"Downloading full SLSTR product from OpenSearch: {scene_id}")
            output_path = scene_dir / f"{scene_id}.zip"
            download_tasks = [self.download_band(download_url, output_path, "product")]
        else:
            assets = scene.get('assets', {})
            download_tasks = []
            for band in bands:
                possible_keys = [
                    band.lower(),
                    f"slstr_{band.lower()}",
                    f"{band.lower()}_radiance",
                    f"{band.lower()}_bt",
                ]
                asset = None
                for key in possible_keys:
                    if key in assets:
                        asset = assets[key]
                        break
                if asset:
                    asset_url = asset.get('href')
                    if asset_url:
                        output_path = scene_dir / f"{band}.nc"
                        download_tasks.append(self.download_band(asset_url, output_path, band))
                else:
                    logger.warning(f"Band {band} not found in scene {scene_id}")

        if download_tasks:
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            if all(r is True for r in results):
                logger.info(f"Successfully downloaded SLSTR scene {scene_id}")
                return scene_dir
            else:
                logger.warning(f"Some bands failed for SLSTR scene {scene_id}")
                return None

        return None
    
    async def download_olci_scene(
        self,
        scene: Dict,
        bands: Optional[List[str]] = None
    ) -> Optional[Path]:
        """
        Download OLCI bands from a Sentinel-3 scene.
        
        Args:
            scene: STAC item containing scene metadata
            bands: List of OLCI band names (e.g., ['Oa08', 'Oa17', 'Oa21'])
            
        Returns:
            Path to the scene directory, or None if failed
        """
        if bands is None:
            bands = self.config['sentinel3']['olci_bands']
        
        # Handle both STAC and OpenSearch formats
        scene_id = scene.get('id') or scene.get('properties', {}).get('title', 'unknown')
        scene_props = scene.get('properties', {})
        scene_date = scene_props.get('datetime', scene_props.get('startDate', '')).split('T')[0]
        
        # Create scene directory
        scene_dir = self.sentinel3_dir / "olci" / scene_date / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scene metadata
        metadata_path = scene_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(scene, f, indent=2)

        # Try S3 first — bypasses OAuth entirely
        if await self._download_via_s3(scene, scene_dir):
            logger.info(f"Successfully downloaded OLCI scene {scene_id} via S3")
            return scene_dir

        # Fall back to HTTP download
        logger.info(f"S3 unavailable, falling back to HTTP for OLCI scene {scene_id}")
        download_url = None
        services = scene_props.get('services', {})
        download_service = services.get('download', {})
        download_url = download_service.get('url')

        if download_url:
            logger.info(f"Downloading full OLCI product from OpenSearch: {scene_id}")
            output_path = scene_dir / f"{scene_id}.zip"
            download_tasks = [self.download_band(download_url, output_path, "product")]
        else:
            assets = scene.get('assets', {})
            download_tasks = []
            for band in bands:
                possible_keys = [
                    band.lower(),
                    f"olci_{band.lower()}",
                    f"{band.lower()}_radiance",
                ]
                asset = None
                for key in possible_keys:
                    if key in assets:
                        asset = assets[key]
                        break
                if asset:
                    asset_url = asset.get('href')
                    if asset_url:
                        output_path = scene_dir / f"{band}.nc"
                        download_tasks.append(self.download_band(asset_url, output_path, band))
                else:
                    logger.warning(f"Band {band} not found in scene {scene_id}")

        if download_tasks:
            results = await asyncio.gather(*download_tasks, return_exceptions=True)
            if all(r is True for r in results):
                logger.info(f"Successfully downloaded OLCI scene {scene_id}")
                return scene_dir
            else:
                logger.warning(f"Some bands failed for OLCI scene {scene_id}")
                return None

        return None
    
    async def download_scenes_batch(
        self,
        scenes: List[Dict],
        instrument: str = "slstr",
        max_concurrent: int = 2
    ) -> List[Path]:
        """
        Download multiple scenes with concurrency control.
        
        Args:
            scenes: List of STAC items to download
            instrument: "slstr" or "olci"
            max_concurrent: Maximum number of concurrent downloads
            
        Returns:
            List of paths to downloaded scene directories
        """
        semaphore = asyncio.Semaphore(max_concurrent)
        
        async def download_with_semaphore(scene):
            async with semaphore:
                if instrument.lower() == "slstr":
                    return await self.download_slstr_scene(scene)
                else:
                    return await self.download_olci_scene(scene)
        
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
        instruments: Optional[List[str]] = None,
        max_scenes: Optional[int] = None
    ):
        """
        Main execution: search and download scenes.
        
        Args:
            bbox: Bounding box override
            start_date: Start date override
            end_date: End date override
            instruments: List of instruments to download ["slstr", "olci"]
            max_scenes: Maximum number of scenes per instrument
        """
        # Use config defaults if not specified
        if bbox is None:
            bbox = self.config['region']['bbox']
        if start_date is None:
            start_date = self.config['time_range']['start_date']
        if end_date is None:
            end_date = self.config['time_range']['end_date']
        if instruments is None:
            instruments = ["slstr"]  # Default to thermal/fire detection
        
        # Get authentication token before starting downloads
        self.access_token = await self.get_access_token()
        if not self.access_token:
            logger.error("Failed to authenticate. Cannot proceed with downloads.")
            logger.error(
                "To fix this, go to https://dataspace.copernicus.eu/ → User Settings → "
                "OAuth Clients and create a new OAuth client (or regenerate its secret). "
                "Set COPERNICUS_CLIENT_ID and COPERNICUS_CLIENT_SECRET in .env, "
                "or paste a pre-issued Bearer token into COPERNICUS_ACCESS_TOKEN."
            )
            await self.client.aclose()
            return
        
        total_downloaded = 0
        
        for instrument in instruments:
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Sentinel-3 {instrument.upper()}")
            logger.info(f"{'='*60}")
            
            # Search for scenes
            scenes = await self.search_scenes(bbox, start_date, end_date, instrument)
            
            if not scenes:
                logger.warning(f"No {instrument.upper()} scenes found matching criteria")
                continue
            
            # Limit number of scenes if specified
            if max_scenes:
                scenes = scenes[:max_scenes]
                logger.info(f"Limited to {max_scenes} scenes")
            
            # Download scenes
            logger.info(f"Starting download of {len(scenes)} {instrument.upper()} scenes...")
            downloaded = await self.download_scenes_batch(scenes, instrument)
            
            logger.info(f"Successfully downloaded {len(downloaded)}/{len(scenes)} {instrument.upper()} scenes")
            total_downloaded += len(downloaded)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Total downloaded: {total_downloaded} scenes")
        logger.info(f"Data saved to: {self.sentinel3_dir}")
        logger.info(f"{'='*60}")
        
        await self.client.aclose()


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download Sentinel-3 imagery for fire detection")
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Path to config file')
    parser.add_argument('--bbox', nargs=4, type=float, help='Bounding box: lon_min lat_min lon_max lat_max')
    parser.add_argument('--start-date', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', help='End date (YYYY-MM-DD)')
    parser.add_argument('--instruments', nargs='+', choices=['slstr', 'olci'], 
                       default=['slstr'], help='Instruments to download')
    parser.add_argument('--max-scenes', type=int, help='Maximum number of scenes per instrument')
    
    args = parser.parse_args()
    
    downloader = Sentinel3Downloader(args.config)
    
    await downloader.run(
        bbox=args.bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        instruments=args.instruments,
        max_scenes=args.max_scenes
    )


if __name__ == "__main__":
    asyncio.run(main())
