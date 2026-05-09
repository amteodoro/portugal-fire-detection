"""Read functions for the public-demo snapshot.

Every function in this module is a pure file read of `data/snapshot/`.
None of them call external APIs or databases. They are the *only* way
demo-mode handlers should fetch data.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

from src.utils.config import settings

logger = logging.getLogger(__name__)


def _root() -> Path:
    return Path(settings.SNAPSHOT_DIR)


def _read_json(rel: str) -> Optional[dict | list]:
    p = _root() / rel
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("snapshot read failed for %s: %s", rel, exc)
        return None


def read_manifest() -> dict:
    return _read_json("manifest.json") or {}


def read_firms(product: str, region: str = "mainland") -> dict:
    """Return a GeoJSON FeatureCollection for (product, region).

    Falls back to an empty collection if the file is missing.
    """
    fname = f"firms/{product.lower()}_{region.lower()}.geojson"
    fc = _read_json(fname)
    if fc and fc.get("type") == "FeatureCollection":
        return fc
    return {"type": "FeatureCollection", "features": []}


def read_sentinel2_manifest() -> dict:
    return _read_json("sentinel2/manifest.json") or {"items": []}


def read_sentinel3_scenes() -> dict:
    return _read_json("sentinel3/scenes.geojson") or {
        "type": "FeatureCollection", "features": []
    }


def read_sentinel3_fire_pixels() -> dict:
    return _read_json("sentinel3/fire_pixels.geojson") or {
        "type": "FeatureCollection", "features": []
    }


def read_predictions() -> dict:
    return _read_json("tiles/predictions.json") or {"tiles": []}


def find_nearest_prediction(lat: float, lon: float) -> Optional[dict]:
    tiles = read_predictions().get("tiles", [])
    if not tiles:
        return None
    return min(
        tiles,
        key=lambda t: (t["lat"] - lat) ** 2 + (t["lon"] - lon) ** 2,
    )


def find_prediction_by_id(tile_id: str) -> Optional[dict]:
    for t in read_predictions().get("tiles", []):
        if t.get("tile_id") == tile_id:
            return t
    return None


def read_aois() -> list[dict]:
    aois = _read_json("aois.json")
    return aois if isinstance(aois, list) else []


def read_metrics() -> Optional[dict]:
    return _read_json("metrics.json")


def read_test_predictions() -> Optional[dict]:
    return _read_json("test_predictions.json")


def eumetsat_frp_path() -> Optional[Path]:
    p = _root() / "eumetsat_frp_sample.png"
    return p if p.exists() else None


def tile_npy_path(tile_id: str) -> Optional[Path]:
    p = _root() / "tiles" / f"{tile_id}.npy"
    return p if p.exists() else None


def tile_png_path(tile_id: str) -> Optional[Path]:
    p = _root() / "tiles" / f"{tile_id}.png"
    return p if p.exists() else None


async def read_history(
    start: str,
    end: str,
    bbox: Optional[tuple[float, float, float, float]] = None,
    source: Optional[str] = None,
    limit: int = 5000,
) -> dict:
    """Query the bundled fires.sqlite snapshot.

    Args mirror /api/firms/history's query params. Returns the same
    GeoJSON FeatureCollection shape so the API handler can return our
    result unchanged.
    """
    import aiosqlite

    db_path = _root() / "fires.sqlite"
    if not db_path.exists():
        return {"type": "FeatureCollection", "features": []}

    sql = (
        "SELECT id, source, detected_at, frp, confidence, region, lat, lon"
        " FROM fire_events"
        " WHERE date(detected_at) BETWEEN :start AND :end"
    )
    params: dict = {"start": start, "end": end}

    if source:
        sql += " AND source = :source"
        params["source"] = source.upper()
    if bbox:
        sql += (
            " AND lon BETWEEN :lon_min AND :lon_max"
            " AND lat BETWEEN :lat_min AND :lat_max"
        )
        lon_min, lat_min, lon_max, lat_max = bbox
        params.update(
            lon_min=lon_min, lat_min=lat_min,
            lon_max=lon_max, lat_max=lat_max,
        )

    sql += " ORDER BY detected_at DESC LIMIT :limit"
    params["limit"] = limit

    async with aiosqlite.connect(db_path) as conn:
        conn.row_factory = aiosqlite.Row
        async with conn.execute(sql, params) as cur:
            rows = await cur.fetchall()

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [r["lon"], r["lat"]]},
            "properties": {
                "id": r["id"],
                "source": r["source"],
                "detected_at": r["detected_at"],
                "frp": r["frp"],
                "confidence": r["confidence"],
                "region": r["region"],
            },
        }
        for r in rows
    ]
    return {"type": "FeatureCollection", "features": features}
