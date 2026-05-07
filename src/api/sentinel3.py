import json
import logging
import re
from pathlib import Path

from fastapi import APIRouter, Request

from src.utils.limiter import limiter
from src.utils.demo import is_demo
from src.services.snapshot import read_sentinel3_scenes, read_sentinel3_fire_pixels

logger = logging.getLogger(__name__)

router = APIRouter()

S3_ZIP_DIR = Path("data/ml/sentinel3")
LABELS_DIR = Path("ml/data/labels")

_FILENAME_RE = re.compile(r"^(S3[AB])_([A-Z0-9_]+?)_{4}(\d{8}T\d{6})_")


def _parse_scene(path: Path) -> dict | None:
    m = _FILENAME_RE.match(path.name)
    if not m:
        return None
    satellite, instrument, raw_dt = m.group(1), m.group(2), m.group(3)
    date_str = f"{raw_dt[:4]}-{raw_dt[4:6]}-{raw_dt[6:8]}"
    size_mb = round(path.stat().st_size / (1024 * 1024), 1)
    return {
        "scene_id": path.stem,
        "satellite": satellite,
        "instrument": instrument,
        "date": date_str,
        "size_mb": size_mb,
    }


@router.get("/scenes")
@limiter.limit("30/minute")
async def list_scenes(request: Request):
    """List available Sentinel-3 scene ZIPs sorted by date descending."""
    if is_demo():
        return read_sentinel3_scenes()
    if not S3_ZIP_DIR.exists():
        return []

    scenes = []
    for p in S3_ZIP_DIR.glob("*.zip"):
        parsed = _parse_scene(p)
        if parsed:
            scenes.append(parsed)

    scenes.sort(key=lambda s: s["date"], reverse=True)
    return scenes


@router.get("/fire-pixels")
@limiter.limit("30/minute")
async def fire_pixels(request: Request):
    """Return GeoJSON FeatureCollection of Sentinel-3 fire tile labels."""
    if is_demo():
        return read_sentinel3_fire_pixels()
    empty_fc = {"type": "FeatureCollection", "features": []}

    if not LABELS_DIR.exists():
        return empty_fc

    features = []
    for p in LABELS_DIR.glob("*.json"):
        try:
            data = json.loads(p.read_text())
        except Exception:
            logger.warning("Skipping malformed label JSON: %s", p)
            continue

        if data.get("satellite") != "sentinel3" or data.get("label") != 1:
            continue

        lat = data.get("tile_center_lat")
        lon = data.get("tile_center_lon")
        if lat is None or lon is None:
            continue

        scene_id = data.get("scene_id") or ""
        satellite, instrument, date_str = None, None, None
        m = _FILENAME_RE.match(scene_id)
        if m:
            satellite = m.group(1)           # S3A or S3B
            instrument = m.group(2)          # e.g. SL_2_FRP
            raw_dt = m.group(3)              # YYYYMMDDTHHmmss
            date_str = f"{raw_dt[:4]}-{raw_dt[4:6]}-{raw_dt[6:8]}"

        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {
                "tile_id": data.get("tile_id"),
                "scene_id": scene_id,
                "satellite": satellite,
                "instrument": instrument,
                "date": date_str,
                "tile_center_lat": lat,
                "tile_center_lon": lon,
                "label": data.get("label"),
            },
        })

    return {"type": "FeatureCollection", "features": features}
