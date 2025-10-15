from datetime import datetime, timedelta, timezone
from src.utils.config import settings
from src.utils.geo import parse_bbox
from src.utils.http import fetch
from src.utils.files import save_image_bytes
from src.utils.caching import file_cache_load, file_cache_store
import httpx
import os

DEFAULT_COLLECTION = "sentinel-2-l2a"  # Earth Search naming

async def search_items(days: int = 7, cloud_pct: int = 20, limit: int = 12) -> dict:
    bbox = parse_bbox(settings.PORTUGAL_BBOX)
    cache_key = f"s2_search_{days}_{cloud_pct}_{limit}"
    cached = file_cache_load("/app/data/sentinel2", cache_key, settings.S2_TTL)
    if cached:
        return cached
    end = datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    body = {
        "collections": [DEFAULT_COLLECTION],
        "bbox": list(bbox),
        "datetime": f"{start.isoformat()}/{end.isoformat()}",
        "limit": limit,
        "query": {"eo:cloud_cover": {"lt": cloud_pct}},
        "sortby": [{"field": "properties.datetime", "direction": "desc"}],
    }
    url = f"{settings.STAC_CATALOG_URL}/search"
    # Earth Search requires POST for /search
    async with httpx.AsyncClient(timeout=20) as client:
        r = await client.post(url, json=body)
        r.raise_for_status()
        data = r.json()
    file_cache_store("/app/data/sentinel2", cache_key, data)
    return data

async def fetch_thumbnail(item: dict) -> dict:
    assets = item.get("assets", {})
    thumb = assets.get("thumbnail") or assets.get("preview") or {}
    href = thumb.get("href")
    if not href:
        return {}
    filename = os.path.basename(href).split("?")[0] or "s2_thumb.jpg"
    path = save_image_bytes(await fetch(href), "sentinel2", filename=filename, optimize=False)
    return {
        "id": item.get("id"),
        "source": "SENTINEL-2",
        "datetime": item.get("properties", {}).get("datetime"),
        "bbox": item.get("bbox", []),
        "path": path,
        "cloud_cover": item.get("properties", {}).get("eo:cloud_cover"),
    }
