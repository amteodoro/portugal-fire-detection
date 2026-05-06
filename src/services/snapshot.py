"""Read functions for the public-demo snapshot.

Every function in this module is a pure file read of `data/snapshot/`.
None of them call external APIs or databases. They are the *only* way
demo-mode handlers should fetch data.
"""
from __future__ import annotations

import json
import logging
from functools import lru_cache
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


@lru_cache(maxsize=1)
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
