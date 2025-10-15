from datetime import datetime, timezone
from src.utils.config import settings
from src.utils.geo import parse_bbox
from src.utils.http import fetch
from src.utils.files import save_image_bytes
from src.utils.caching import file_cache_load, file_cache_store
import os

def _wms_params(bbox: tuple[float, float, float, float]):
    lon_min, lat_min, lon_max, lat_max = bbox
    return {
        "service": "WMS",
        "request": "GetMap",
        "version": "1.1.1",
        "layers": settings.EUMETSAT_FRP_LAYER,
        "styles": "",
        "format": "image/png",
        "transparent": "true",
        "srs": "EPSG:4326",
        "bbox": f"{lon_min},{lat_min},{lon_max},{lat_max}",
        "width": 1024,
        "height": 1024,
    }

async def fetch_frp_png() -> dict:
    bbox = parse_bbox(settings.PORTUGAL_BBOX)
    cache_key = "eumetsat_frp_png_latest"
    existing = file_cache_load("/app/data/eumetsat", cache_key, settings.EUMETSAT_TTL)
    if existing and os.path.exists(existing.get("path", "")):
        # Validate the cached file is not too small (error response)
        file_size = os.path.getsize(existing.get("path", ""))
        if file_size > 1000:  # Valid image should be larger than 1KB
            return existing
    
    if not settings.EUMETSAT_WMS_URL or not settings.EUMETSAT_FRP_LAYER:
        return {"error": "EUMETSAT WMS URL or layer not configured"}
    
    try:
        headers = {}
        if settings.EUMETSAT_TOKEN:
            headers["Authorization"] = f"Bearer {settings.EUMETSAT_TOKEN}"
        
        img = await fetch(settings.EUMETSAT_WMS_URL, params=_wms_params(bbox), headers=headers)
        
        # Validate response size
        if len(img) < 1000:
            return {"error": "WMS returned invalid/small response - check credentials or layer name"}
        
        path = save_image_bytes(img, "eumetsat", filename=None, optimize=True)
        meta = {
            "id": os.path.basename(path),
            "source": "EUMETSAT_FRP",
            "datetime": datetime.now(timezone.utc).isoformat(),
            "bbox": list(bbox),
            "path": path,
        }
        file_cache_store("/app/data/eumetsat", cache_key, meta)
        return meta
    except Exception as e:
        return {"error": f"EUMETSAT WMS error: {str(e)}"}
