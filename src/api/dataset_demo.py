"""Demo-mode read-only dataset router.

Mirrors the live dataset endpoints' response shape, but reads from the
bundled `data/snapshot/dataset/` directory. All mutating endpoints are
intentionally absent — the UI hides those buttons in demo mode anyway,
but if anything still calls them they get a 404 from FastAPI.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

SNAPSHOT_ROOT = Path("data/snapshot")
DATASET_ROOT = SNAPSHOT_ROOT / "dataset"


def _read_json(path: Path) -> Any:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


@router.get("/summary")
async def get_dataset_summary() -> dict:
    """Return the snapshot's frozen dataset summary, shaped like the live
    `/api/dataset/summary` response so the UI renders without changes."""
    summary = _read_json(DATASET_ROOT / "summary.json")
    if summary is not None:
        return summary

    # Fallback: synthesize a minimal, shape-compatible summary from the
    # top-level snapshot manifest. Numbers may be coarse but the page
    # will render with no errors.
    manifest = _read_json(SNAPSHOT_ROOT / "manifest.json") or {}
    counts = manifest.get("counts", {})
    rng = manifest.get("snapshot_range", {})
    start = rng.get("start", "—")
    end = rng.get("end", "—")
    total_fires = sum(
        v for k, v in counts.items()
        if k.startswith(("VIIRS_", "MODIS_")) and isinstance(v, int)
    )
    return {
        "sentinel2": {
            "total_scenes": counts.get("sentinel2_items", 0),
            "unique_dates": 0,
            "date_range": {"start": start, "end": end},
        },
        "sentinel3": {
            "total_scenes": counts.get("sentinel3_scenes", 0),
            "slstr_scenes": counts.get("sentinel3_scenes", 0),
            "olci_scenes": 0,
        },
        "firms": {
            "total_fire_detections": total_fires,
            "regions": [
                {"name": "mainland", "total_fires": total_fires},
            ],
        },
        "processed": {
            "total_tiles": 0,
            "fire_tiles": 0,
            "no_fire_tiles": 0,
            "fire_ratio": 0,
            "splits": {
                "train": {"fire": 0, "no_fire": 0, "total": 0, "fire_ratio": 0},
                "val":   {"fire": 0, "no_fire": 0, "total": 0, "fire_ratio": 0},
                "test":  {"fire": 0, "no_fire": 0, "total": 0, "fire_ratio": 0},
            },
        },
        "status": {"ready_for_training": False},
        "samples": [],
        "demo": True,
    }


def _sample_response(json_path: Path, base_path: str) -> dict:
    """Common shape for tile-samples, dual-satellite-samples, colocated-samples."""
    samples = _read_json(json_path)
    if not samples:
        return {
            "samples": [],
            "message": "No samples bundled in this demo snapshot.",
            "can_generate": False,
        }
    return {
        "samples": samples,
        "base_path": base_path,
        "can_generate": True,
    }


@router.get("/tile-samples")
async def tile_samples() -> dict:
    return _sample_response(
        DATASET_ROOT / "tile_samples" / "samples_info.json",
        "/static/snapshot/dataset/tile_samples",
    )


@router.get("/dual-satellite-samples")
async def dual_satellite_samples() -> dict:
    return _sample_response(
        DATASET_ROOT / "dual_view" / "samples_info.json",
        "/static/snapshot/dataset/dual_view",
    )


@router.get("/colocated-samples")
async def colocated_samples() -> dict:
    samples = _read_json(DATASET_ROOT / "colocated" / "samples_info.json")
    if not samples:
        return {
            "samples": [],
            "message": "No co-located pairs bundled in this demo snapshot.",
            "can_generate": False,
            "pairs_exist": False,
        }
    return {
        "samples": samples,
        "base_path": "/static/snapshot/dataset/colocated",
        "can_generate": True,
        "pairs_exist": True,
    }


@router.get("/temporal-sequence-samples")
async def temporal_sequence_samples() -> dict:
    return {
        "samples": [],
        "message": "Temporal-sequence visualisations are not bundled in the demo snapshot.",
        "can_generate": False,
    }


# ── Demo "regenerate" endpoints ──────────────────────────────────────
# In demo mode these don't actually compute anything — they re-pick a
# random subset of the bundled samples so the user can shuffle the
# visible carousel. Stream a single SSE event so the existing frontend
# progress UI doesn't break.

class _NSamples(BaseModel):
    n_samples: int | None = 8
    seed: int | None = None


def _shuffle_subset(json_path: Path, n: int, seed: int | None) -> int:
    """Persist a re-ordered subset back to the JSON file. Returns count."""
    samples = _read_json(json_path)
    if not isinstance(samples, list) or not samples:
        return 0
    rng = random.Random(seed)
    rng.shuffle(samples)
    picked = samples[: max(1, min(n, len(samples)))]
    json_path.write_text(json.dumps(picked, indent=2))
    return len(picked)


@router.post("/generate-tile-samples")
async def generate_tile_samples(req: _NSamples):
    path = DATASET_ROOT / "tile_samples" / "samples_info.json"
    if not path.exists():
        raise HTTPException(404, "No tile samples bundled in this demo snapshot.")
    n = _shuffle_subset(path, req.n_samples or 8, req.seed)
    return {"status": "ok", "demo": True, "samples": n}


@router.post("/generate-dual-satellite-samples")
async def generate_dual_samples(req: _NSamples):
    path = DATASET_ROOT / "dual_view" / "samples_info.json"
    if not path.exists():
        raise HTTPException(404, "No dual-satellite samples bundled in this demo snapshot.")
    n = _shuffle_subset(path, req.n_samples or 6, req.seed)
    return {"status": "ok", "demo": True, "samples": n}
