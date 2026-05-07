import io
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import Response
from pydantic import BaseModel
from src.utils.limiter import limiter
from src.utils.demo import is_demo
from src.services import snapshot as snap

logger = logging.getLogger(__name__)

router = APIRouter()

# Tiles with lat/lon encoded in filename live in the splits directories.
# Pattern: {id}_{source}_{date}_{lat}_{lon}_{label}_{idx}.npy
SPLITS_DIR = Path("ml/data/processed/splits")
_TILE_COORD_RE = re.compile(
    r"^.+_(?P<lat>-?\d+\.\d+)_(?P<lon>-?\d+\.\d+)_(?:positive|negative)_\d+$"
)

# Lazy tile index: stem -> (lat, lon, path)
_tile_index: Optional[dict[str, tuple[float, float, Path]]] = None


def _build_tile_index() -> dict[str, tuple[float, float, Path]]:
    index: dict[str, tuple[float, float, Path]] = {}
    if not SPLITS_DIR.exists():
        return index
    for npy in SPLITS_DIR.rglob("*.npy"):
        m = _TILE_COORD_RE.match(npy.stem)
        if m:
            index[npy.stem] = (float(m.group("lat")), float(m.group("lon")), npy)
    logger.info("Built tile index: %d geo-tagged tiles", len(index))
    return index


def _get_tile_index() -> dict[str, tuple[float, float, Path]]:
    global _tile_index
    if _tile_index is None:
        _tile_index = _build_tile_index()
    return _tile_index


class PredictRequest(BaseModel):
    tile_id: Optional[str] = None
    lat: Optional[float] = None
    lon: Optional[float] = None
    live: bool = False          # If True, fetch a fresh S2 tile via COG
    days_back: int = 30         # How many days back to search for a scene
    max_cloud: int = 30         # Max cloud cover % for live fetch


METRICS_PATH = Path("ml/training/checkpoints/test_metrics.json")
PREDICTIONS_PATH = Path("ml/training/checkpoints/test_predictions.json")


@router.get("/metrics")
@limiter.limit("60/minute")
async def inference_metrics(request: Request):
    """Return test-set evaluation metrics for the loaded model."""
    if is_demo():
        m = snap.read_metrics()
        if m is None:
            raise HTTPException(status_code=404, detail="No metrics in snapshot")
        return m
    if not METRICS_PATH.exists():
        raise HTTPException(status_code=404, detail="Test metrics not yet computed. Run ml/scripts/evaluate_model.py first.")
    import json as _json
    return _json.loads(METRICS_PATH.read_text())


@router.get("/test-results")
@limiter.limit("30/minute")
async def test_results(
    request: Request,
    outcome: Optional[str] = None,
    limit: int = 20,
    offset: int = 0,
):
    """
    Return per-tile test-set inference results.

    Query params:
    - outcome: filter by 'tp', 'fp', 'fn', 'tn', or leave empty for all
    - limit: max items to return (default 20, max 100)
    - offset: pagination offset
    """
    if is_demo():
        data = snap.read_test_predictions()
        if data is None:
            raise HTTPException(status_code=404, detail="Not available in demo snapshot")
        from collections import Counter
        tiles = data.get("tiles", [])
        counts = Counter(t["outcome"] for t in tiles)
        if outcome:
            valid_outcomes = {"tp", "fp", "fn", "tn"}
            if outcome not in valid_outcomes:
                raise HTTPException(status_code=400, detail=f"outcome must be one of {valid_outcomes}")
            tiles = [t for t in tiles if t["outcome"] == outcome]
        total = len(tiles)
        limit_eff = min(limit, 100)
        page = tiles[offset: offset + limit_eff]
        return {
            "model_version": data.get("model_version", "snapshot"),
            "threshold": data.get("threshold", 0.5),
            "counts": dict(counts),
            "total_filtered": total,
            "offset": offset,
            "limit": limit_eff,
            "tiles": page,
        }
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Per-tile predictions not yet computed. "
                "Run: docker exec <container> uv run python ml/scripts/evaluate_model.py"
            ),
        )
    import json as _json
    data = _json.loads(PREDICTIONS_PATH.read_text())
    tiles = data["tiles"]

    valid_outcomes = {"tp", "fp", "fn", "tn"}
    if outcome and outcome not in valid_outcomes:
        raise HTTPException(status_code=400, detail=f"outcome must be one of {valid_outcomes}")

    if outcome:
        tiles = [t for t in tiles if t["outcome"] == outcome]

    total = len(tiles)
    limit = min(limit, 100)
    page = tiles[offset: offset + limit]

    # Count by outcome
    from collections import Counter
    counts = Counter(t["outcome"] for t in data["tiles"])

    return {
        "model_version": data["model_version"],
        "threshold": data["threshold"],
        "counts": dict(counts),
        "total_filtered": total,
        "offset": offset,
        "limit": limit,
        "tiles": page,
    }


@router.get("/model-info")
@limiter.limit("60/minute")
async def model_info(request: Request):
    """Return full model metadata: type, training date, data stats, val/test metrics."""
    if is_demo():
        info = {"model_version": "snapshot", "demo": True}
        m = snap.read_metrics()
        if m:
            info["test_metrics"] = m
        return info
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference model not loaded")

    info: dict = {
        "model_version": predictor.model_version,
        "model_type": getattr(predictor, "model_type", "mobilevit"),
        "threshold": predictor.threshold,
    }

    # Include training_metadata if saved in checkpoint
    checkpoint_path = Path("ml/training/checkpoints") / predictor.model_version
    if checkpoint_path.exists():
        try:
            import torch as _torch
            ckpt = _torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
            if "training_metadata" in ckpt:
                info["training_metadata"] = ckpt["training_metadata"]
            if "config" in ckpt:
                cfg = ckpt["config"]
                info["training_config"] = {
                    "model_name": cfg.get("model", {}).get("model_name"),
                    "input_channels": cfg.get("model", {}).get("input_channels"),
                    "epochs_trained": ckpt.get("epoch"),
                }
        except Exception as exc:
            logger.warning("Could not read checkpoint metadata: %s", exc)

    # Include test metrics if available
    if METRICS_PATH.exists():
        import json as _json
        try:
            info["test_metrics"] = _json.loads(METRICS_PATH.read_text())
        except Exception:
            pass

    return info


@router.get("/status")
@limiter.limit("60/minute")
async def inference_status(request: Request):
    if is_demo():
        snapshot_tiles = len(snap.read_predictions().get("tiles", []))
        return {
            "loaded": False,
            "demo": True,
            "model_version": None,
            "threshold": None,
            "snapshot_tiles": snapshot_tiles,
        }
    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        return {"loaded": False, "demo": False, "model_version": None, "threshold": None}
    index = _get_tile_index()
    return {
        "loaded": True,
        "demo": False,
        "model_version": predictor.model_version,
        "threshold": predictor.threshold,
        "tiles_indexed": len(index),
    }


@router.post("/predict")
@limiter.limit("20/minute")
async def predict(request: Request, body: PredictRequest):
    if is_demo():
        # No model loaded; resolve to a snapshot prediction.
        pred = None
        if body.tile_id:
            pred = snap.find_prediction_by_id(body.tile_id)
        if pred is None and body.lat is not None and body.lon is not None:
            pred = snap.find_nearest_prediction(body.lat, body.lon)
        if pred is None:
            raise HTTPException(
                status_code=503,
                detail="No demo predictions available",
            )
        return {
            "tile_id": pred["tile_id"],
            "tile_lat": pred["lat"],
            "tile_lon": pred["lon"],
            "fire_probability": pred["fire_probability"],
            "is_fire": pred.get("is_fire", pred["fire_probability"] >= 0.5),
            "confidence": pred["fire_probability"],
            "model_version": "snapshot",
            "source": "demo_snapshot",
            "scene_date": pred.get("scene_date"),
            "cloud_cover": pred.get("cloud_cover"),
            "note": "Demo mode — pre-computed prediction from snapshot",
        }

    predictor = getattr(request.app.state, "predictor", None)
    if predictor is None:
        raise HTTPException(status_code=503, detail="Inference model not loaded")

    # Live mode: fetch a fresh Sentinel-2 tile via COG window reads
    if body.live and body.lat is not None and body.lon is not None:
        from src.services.live_tile import fetch_tile_for_point
        live_data = await fetch_tile_for_point(
            body.lat, body.lon,
            days_back=body.days_back,
            max_cloud=body.max_cloud,
        )
        if live_data is not None:
            result = predictor.predict_from_array(live_data["tile"])
            return {
                "tile_id": live_data["scene_id"],
                "tile_lat": body.lat,
                "tile_lon": body.lon,
                "fire_probability": round(result.fire_probability, 4),
                "is_fire": result.is_fire,
                "confidence": round(result.confidence, 4),
                "model_version": result.model_version,
                "source": "live",
                "scene_date": live_data.get("scene_date"),
                "cloud_cover": live_data.get("cloud_cover"),
            }
        # Fall back to nearest training tile with a note
        logger.info(
            "Live tile unavailable for (%.4f, %.4f), falling back to training tile",
            body.lat, body.lon,
        )

    tile_id, tile, tile_lat, tile_lon = _resolve_tile(body)
    result = predictor.predict_from_array(tile)
    return {
        "tile_id": tile_id,
        "tile_lat": tile_lat,
        "tile_lon": tile_lon,
        "fire_probability": round(result.fire_probability, 4),
        "is_fire": result.is_fire,
        "confidence": round(result.confidence, 4),
        "model_version": result.model_version,
        "source": "training_tile",
        "scene_date": None,
        "cloud_cover": None,
    }


_SAFE_TILE_ID_RE = re.compile(r"^[\w\-\.]+$")
TILES_DIR_RAW = Path("ml/data/tiles")


@router.get("/tile-image/{tile_id}")
@limiter.limit("120/minute")
async def tile_image(request: Request, tile_id: str, size: int = 256):
    """
    Return a false-color PNG for a given tile_id.

    Renders SWIR2→R, NIR→G, Red→B (fires appear bright red/white).
    Optional ?size= param (128–512, default 256).
    """
    if is_demo():
        from fastapi.responses import FileResponse
        png = snap.tile_png_path(tile_id)
        if png is None:
            raise HTTPException(status_code=404, detail=f"Tile not found: {tile_id}")
        return FileResponse(png, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})
    if not _SAFE_TILE_ID_RE.match(tile_id):
        raise HTTPException(status_code=400, detail="Invalid tile_id")
    size = max(128, min(size, 512))

    tile_path = TILES_DIR_RAW / f"{tile_id}.npy"
    if not tile_path.exists():
        # Also check splits
        for candidate in Path("ml/data/processed/splits").rglob(f"{tile_id}.npy"):
            if candidate.exists():
                tile_path = candidate
                break
        else:
            raise HTTPException(status_code=404, detail=f"Tile not found: {tile_id}")

    try:
        from PIL import Image as PilImage

        tile = np.load(tile_path)  # shape (4, H, W): Red, NIR, SWIR1, SWIR2

        def _norm(band: np.ndarray) -> np.ndarray:
            valid = band[np.isfinite(band)]
            if len(valid) == 0:
                return np.zeros(band.shape, dtype=np.uint8)
            p2, p98 = np.percentile(valid, (2, 98))
            if p98 - p2 < 1e-6:
                p2, p98 = valid.min(), valid.max()
            if p98 - p2 < 1e-6:
                return np.full(band.shape, 128, dtype=np.uint8)
            return np.clip((band - p2) / (p98 - p2) * 255, 0, 255).astype(np.uint8)

        # False color: SWIR2(3)→R, NIR(1)→G, Red(0)→B
        r = _norm(tile[3]) if tile.shape[0] > 3 else _norm(tile[-1])
        g = _norm(tile[1]) if tile.shape[0] > 1 else _norm(tile[0])
        b = _norm(tile[0])
        rgb = np.stack([r, g, b], axis=-1)  # (H, W, 3)

        img = PilImage.fromarray(rgb, "RGB")
        if img.size != (size, size):
            img = img.resize((size, size), PilImage.NEAREST)

        buf = io.BytesIO()
        img.save(buf, format="PNG", optimize=False)
        buf.seek(0)
        return Response(
            content=buf.read(),
            media_type="image/png",
            headers={"Cache-Control": "public, max-age=3600"},
        )
    except Exception as exc:
        logger.error("tile-image render failed for %s: %s", tile_id, exc)
        raise HTTPException(status_code=500, detail="Failed to render tile image")


@router.get("/roc-data")
@limiter.limit("30/minute")
async def roc_data(request: Request):
    """Return (fpr, tpr) arrays for drawing the ROC curve, plus threshold array."""
    if is_demo():
        data = snap.read_test_predictions()
        if data is None:
            raise HTTPException(status_code=404, detail="Not available in demo snapshot")
        return data.get("roc", {"fpr": [], "tpr": [], "thresholds": [], "auc": None})
    if not PREDICTIONS_PATH.exists():
        raise HTTPException(status_code=404, detail="Predictions not yet computed")
    import json as _json

    data = _json.loads(PREDICTIONS_PATH.read_text())
    tiles = data["tiles"]

    y_true = np.array([t["y_true"] for t in tiles])
    y_prob = np.array([t["fire_probability"] for t in tiles])

    thresholds = np.linspace(0, 1, 201)
    fprs, tprs = [], []
    n_pos = float((y_true == 1).sum())
    n_neg = float((y_true == 0).sum())
    for thr in thresholds:
        pred = (y_prob >= thr).astype(int)
        tp = float(((pred == 1) & (y_true == 1)).sum())
        fp = float(((pred == 1) & (y_true == 0)).sum())
        tprs.append(tp / n_pos if n_pos > 0 else 0.0)
        fprs.append(fp / n_neg if n_neg > 0 else 0.0)

    return {
        "fpr": [round(v, 4) for v in fprs],
        "tpr": [round(v, 4) for v in tprs],
        "thresholds": [round(float(t), 4) for t in thresholds],
        "auc": data.get("auc") or round(float(np.trapezoid(tprs[::-1], fprs[::-1])), 4),
        "operating_threshold": data["threshold"],
    }


def _resolve_tile(body: PredictRequest) -> tuple[str, np.ndarray, Optional[float], Optional[float]]:
    if body.tile_id is not None:
        tile_id, tile, lat, lon = _load_tile_by_id(body.tile_id)
        return tile_id, tile, lat, lon
    if body.lat is not None and body.lon is not None:
        return _find_nearest_tile(body.lat, body.lon)
    raise HTTPException(
        status_code=400,
        detail="Provide either tile_id or lat/lon coordinates",
    )


def _load_tile_by_id(tile_id: str) -> tuple[str, np.ndarray, Optional[float], Optional[float]]:
    index = _get_tile_index()
    if tile_id in index:
        lat, lon, path = index[tile_id]
        return tile_id, np.load(path), lat, lon
    # Fallback: check ml/data/tiles/
    fallback = Path("ml/data/tiles") / f"{tile_id}.npy"
    if fallback.exists():
        return tile_id, np.load(fallback), None, None
    raise HTTPException(status_code=404, detail=f"Tile not found: {tile_id}")


def _find_nearest_tile(lat: float, lon: float) -> tuple[str, np.ndarray, float, float]:
    index = _get_tile_index()
    if not index:
        raise HTTPException(status_code=404, detail="No geo-tagged tiles available")

    best_id = min(index, key=lambda k: (index[k][0] - lat) ** 2 + (index[k][1] - lon) ** 2)
    t_lat, t_lon, path = index[best_id]
    return best_id, np.load(path), t_lat, t_lon
