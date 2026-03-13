"""
Fetch a live 4-channel Sentinel-2 tile centered on a lat/lon point.

Uses Cloud Optimised GeoTIFF (COG) windowed HTTP reads so only the
required ~1.28 km² patch is downloaded from each band file, typically
completing in 2–8 seconds across all 4 bands.

Output is a float32 (4, 128, 128) array with p99-normalised values in
[0, 1], identical format to the training tiles in ml/data/tiles/.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import httpx
import numpy as np

logger = logging.getLogger(__name__)

STAC_URL = "https://earth-search.aws.element84.com/v1"
LIVE_TILE_CACHE_DIR = Path("/app/data/live_tiles")
TILE_SIZE = 128  # pixels

# (stac_asset_key, output_channel_index, native_resolution_metres)
_BANDS = [
    ("red",    0, 10),  # B04
    ("nir",    1, 10),  # B08
    ("swir16", 2, 20),  # B11
    ("swir22", 3, 20),  # B12
]


async def search_scene_for_point(
    lat: float,
    lon: float,
    days_back: int = 30,
    max_cloud: int = 30,
) -> Optional[dict]:
    """Return the most recent low-cloud S2 scene covering (lat, lon)."""
    margin = 0.05  # ~5.5 km at Portugal latitude
    bbox = [lon - margin, lat - margin, lon + margin, lat + margin]
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days_back)

    body = {
        "collections": ["sentinel-2-l2a"],
        "bbox": bbox,
        "datetime": f"{start.isoformat()}/{end.isoformat()}",
        "query": {"eo:cloud_cover": {"lt": max_cloud}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
        "limit": 5,
    }
    async with httpx.AsyncClient(timeout=15) as client:
        r = await client.post(f"{STAC_URL}/search", json=body)
        r.raise_for_status()
    features = r.json().get("features", [])
    return features[0] if features else None


def _read_band_window_sync(
    url: str, lat: float, lon: float, native_res: int
) -> np.ndarray:
    """
    Synchronous COG windowed read.  Called via asyncio.to_thread so it
    doesn't block the event loop.

    For 10 m bands, reads a 128×128 native-resolution window.
    For 20 m bands, reads a 64×64 native-resolution window and asks
    rasterio to resample it to 128×128 on the fly.
    """
    import rasterio
    from rasterio.windows import Window
    from pyproj import Transformer

    native_half = TILE_SIZE // 2 if native_res == 10 else TILE_SIZE // 4

    with rasterio.open(url) as src:
        transformer = Transformer.from_crs(
            "EPSG:4326", src.crs, always_xy=True
        )
        x, y = transformer.transform(lon, lat)
        row, col = src.index(x, y)

        window = Window(
            col_off=col - native_half,
            row_off=row - native_half,
            width=native_half * 2,
            height=native_half * 2,
        )
        data = src.read(
            1,
            window=window,
            boundless=True,
            fill_value=0,
            out_shape=(TILE_SIZE, TILE_SIZE),
        ).astype(np.float32)

    return data


def _p99_normalize(band: np.ndarray) -> np.ndarray:
    valid = band[band > 0]
    if valid.size == 0:
        return band
    p99 = float(np.percentile(valid, 99))
    return np.clip(band / p99, 0.0, 1.0) if p99 > 0 else band


async def fetch_tile_for_point(
    lat: float,
    lon: float,
    days_back: int = 30,
    max_cloud: int = 30,
) -> Optional[dict]:
    """
    Return a dict with keys:
        tile        – np.ndarray (4, 128, 128) float32
        scene_id    – str
        scene_date  – str  (YYYY-MM-DD)
        cloud_cover – float | None
        source      – "live"
    Returns None if no suitable scene is found or a read error occurs.
    """
    scene = await search_scene_for_point(lat, lon, days_back, max_cloud)
    if not scene:
        logger.info("No S2 scene found for (%.4f, %.4f)", lat, lon)
        return None

    assets = scene.get("assets", {})
    scene_id = scene.get("id", "unknown")
    scene_date = (scene.get("properties", {}).get("datetime") or "")[:10]
    cloud_cover = scene.get("properties", {}).get("eo:cloud_cover")

    # Check disk cache
    LIVE_TILE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_npy = LIVE_TILE_CACHE_DIR / f"{scene_id}_{lat:.4f}_{lon:.4f}.npy"
    cache_json = LIVE_TILE_CACHE_DIR / f"{scene_id}_{lat:.4f}_{lon:.4f}.json"
    if cache_npy.exists() and cache_json.exists():
        logger.info("Live tile cache hit: %s", cache_npy.name)
        return {
            "tile": np.load(cache_npy),
            **json.loads(cache_json.read_text()),
        }

    # Resolve band URLs
    band_urls = []
    for band_key, _, native_res in _BANDS:
        href = assets.get(band_key, {}).get("href")
        if not href:
            logger.warning("Scene %s missing asset '%s'", scene_id, band_key)
            return None
        band_urls.append((href, native_res))

    # Read all 4 bands concurrently (each is a blocking COG read in a thread)
    try:
        results = await asyncio.gather(
            *[
                asyncio.to_thread(_read_band_window_sync, url, lat, lon, res)
                for url, res in band_urls
            ]
        )
    except Exception as exc:
        logger.error("Error reading S2 bands for scene %s: %s", scene_id, exc)
        return None

    tile = np.stack([_p99_normalize(b) for b in results], axis=0)

    meta = {
        "scene_id": scene_id,
        "scene_date": scene_date,
        "cloud_cover": round(float(cloud_cover), 1) if cloud_cover is not None else None,
        "source": "live",
    }
    np.save(cache_npy, tile)
    cache_json.write_text(json.dumps(meta))
    logger.info(
        "Live tile fetched and cached: scene=%s date=%s cloud=%.1f%%",
        scene_id,
        scene_date,
        cloud_cover or 0,
    )
    return {"tile": tile, **meta}
