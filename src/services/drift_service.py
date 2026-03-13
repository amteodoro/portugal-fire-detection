"""Model drift detection service.

Runs weekly to compare live model predictions against the baseline test F1.
Logs a warning (and optionally emails) if drift exceeds 5%.

Only runs when a predictor is loaded and the tile index is non-empty.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

METRICS_PATH = Path("ml/training/checkpoints/test_metrics.json")
DRIFT_LOG_PATH = Path("ml/training/checkpoints/drift_log.json")


async def check_drift(app_state) -> dict:
    """Run drift check.

    Compares model predictions on nearest training tiles for the most recent
    FIRMS detections against the baseline test F1. Logs result and returns
    a report dict.

    Args:
        app_state: FastAPI app.state (needs .predictor attribute)
    """
    predictor = getattr(app_state, "predictor", None)
    if predictor is None:
        logger.info("Drift check: no predictor loaded, skipping")
        return {"skipped": True, "reason": "no_predictor"}

    if not METRICS_PATH.exists():
        logger.info("Drift check: no baseline metrics, skipping")
        return {"skipped": True, "reason": "no_baseline_metrics"}

    baseline = json.loads(METRICS_PATH.read_text())
    baseline_f1 = baseline.get("f1")
    if baseline_f1 is None:
        return {"skipped": True, "reason": "baseline_f1_missing"}

    # Load tile index
    from src.api.inference import _get_tile_index
    index = _get_tile_index()
    if not index:
        return {"skipped": True, "reason": "no_tiles"}

    # Sample up to 100 tiles from the index (stratified: half positive, half other)
    import numpy as np

    tile_items = list(index.items())
    rng = np.random.default_rng(seed=42)
    sample_size = min(100, len(tile_items))
    sampled = rng.choice(len(tile_items), size=sample_size, replace=False)

    correct = 0
    total = 0
    fire_true = []
    fire_pred = []

    for idx in sampled:
        stem, (lat, lon, path) = tile_items[idx]
        try:
            tile = np.load(path)
            result = predictor.predict_from_array(tile)
            # Infer ground truth from filename: "positive" → 1, "negative" → 0
            gt = 1 if "positive" in stem else 0
            pred = 1 if result.is_fire else 0
            fire_true.append(gt)
            fire_pred.append(pred)
            if gt == pred:
                correct += 1
            total += 1
        except Exception as exc:
            logger.debug("Drift tile error: %s: %s", path, exc)

    if total == 0:
        return {"skipped": True, "reason": "no_samples_processed"}

    from sklearn.metrics import f1_score

    live_f1 = float(f1_score(fire_true, fire_pred, zero_division=0))
    drift = baseline_f1 - live_f1
    drift_pct = abs(drift) * 100

    report = {
        "baseline_f1": round(baseline_f1, 4),
        "live_f1": round(live_f1, 4),
        "drift": round(drift, 4),
        "drift_pct": round(drift_pct, 2),
        "samples": total,
        "drift_warning": drift_pct > 5.0,
    }

    if report["drift_warning"]:
        logger.warning(
            "MODEL DRIFT DETECTED: baseline F1=%.4f, live F1=%.4f, drift=%.1f%%",
            baseline_f1,
            live_f1,
            drift_pct,
        )
        # Send email alert if alert service is configured
        try:
            from src.services.alert_service import is_configured, _send_email
            from src.utils.config import settings

            if is_configured():
                _send_email(
                    to=settings.FROM_EMAIL,
                    subject=f"[Drift Alert] Model F1 dropped {drift_pct:.1f}%",
                    body=(
                        f"Model drift detected!\n\n"
                        f"Baseline F1: {baseline_f1:.4f}\n"
                        f"Live F1:     {live_f1:.4f}\n"
                        f"Drift:       {drift_pct:.1f}%\n\n"
                        f"Consider retraining the model.\n"
                    ),
                )
        except Exception as exc:
            logger.debug("Drift alert email failed: %s", exc)
    else:
        logger.info(
            "Drift check: baseline F1=%.4f, live F1=%.4f, drift=%.1f%% (OK)",
            baseline_f1,
            live_f1,
            drift_pct,
        )

    # Persist drift log
    try:
        from datetime import datetime, timezone

        log: list = []
        if DRIFT_LOG_PATH.exists():
            try:
                log = json.loads(DRIFT_LOG_PATH.read_text())
            except Exception:
                log = []
        report_with_ts = {"timestamp": datetime.now(timezone.utc).isoformat(), **report}
        log.append(report_with_ts)
        DRIFT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        DRIFT_LOG_PATH.write_text(json.dumps(log[-50:], indent=2))  # keep last 50
    except Exception as exc:
        logger.warning("Could not write drift log: %s", exc)

    return report
