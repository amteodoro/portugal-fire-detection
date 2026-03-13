"""Evaluate best_model.pth on the held-out test set and write metrics JSON."""

import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from ml.inference.predictor import FirePredictor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

CHECKPOINT = Path("ml/training/checkpoints/best_model.pth")
TILES_DIR = Path("ml/data/tiles")
LABELS_DIR = Path("ml/data/labels")
TEST_SPLIT = Path("ml/data/processed/test_split.json")
OUTPUT = Path("ml/training/checkpoints/test_metrics.json")
PREDICTIONS_OUTPUT = Path("ml/training/checkpoints/test_predictions.json")


def main():
    if not CHECKPOINT.exists():
        logger.error("Checkpoint not found: %s", CHECKPOINT)
        sys.exit(1)

    logger.info("Loading predictor from %s", CHECKPOINT)
    predictor = FirePredictor(str(CHECKPOINT))

    logger.info("Loading test split from %s", TEST_SPLIT)
    test_ids = json.loads(TEST_SPLIT.read_text())["tile_ids"]
    logger.info("Test set size: %d tiles", len(test_ids))

    y_true, y_prob = [], []
    per_tile: list = []
    skipped = 0

    for i, tile_id in enumerate(test_ids):
        tile_path = TILES_DIR / f"{tile_id}.npy"
        label_path = LABELS_DIR / f"{tile_id}.json"

        if not tile_path.exists() or not label_path.exists():
            skipped += 1
            continue

        tile = np.load(tile_path)
        label_data = json.loads(label_path.read_text())
        label = label_data["label"]

        result = predictor.predict_from_array(tile)
        y_true.append(label)
        y_prob.append(result.fire_probability)

        # Parse lat/lon from tile_id (fire tiles encode coords in filename)
        lat, lon = None, None
        parts = tile_id.split("_")
        for j in range(len(parts) - 1):
            try:
                maybe_lat = float(parts[j])
                maybe_lon = float(parts[j + 1])
                if 30 < maybe_lat < 50 and -15 < maybe_lon < 5:
                    lat, lon = maybe_lat, maybe_lon
                    break
            except ValueError:
                continue

        per_tile.append({
            "tile_id": tile_id,
            "y_true": label,
            "fire_probability": round(float(result.fire_probability), 4),
            "lat": lat,
            "lon": lon,
            "scene_id": label_data.get("scene_id"),
            "sample_type": label_data.get("sample_type"),
        })

        if (i + 1) % 100 == 0:
            logger.info("Progress: %d/%d", i + 1, len(test_ids))

    logger.info("Evaluated %d tiles (%d skipped)", len(y_true), skipped)

    y_true_arr = np.array(y_true)
    y_prob_arr = np.array(y_prob)
    y_pred = (y_prob_arr >= predictor.threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true_arr == 1)).sum())
    fp = int(((y_pred == 1) & (y_true_arr == 0)).sum())
    fn = int(((y_pred == 0) & (y_true_arr == 1)).sum())
    tn = int(((y_pred == 0) & (y_true_arr == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / len(y_true)

    # AUC via trapezoidal rule
    thresholds = np.linspace(0, 1, 201)
    tprs, fprs = [], []
    for t in thresholds:
        pred = (y_prob_arr >= t).astype(int)
        tprs.append(((pred == 1) & (y_true_arr == 1)).sum() / max((y_true_arr == 1).sum(), 1))
        fprs.append(((pred == 1) & (y_true_arr == 0)).sum() / max((y_true_arr == 0).sum(), 1))
    auc = float(np.trapezoid(tprs[::-1], fprs[::-1]))

    metrics = {
        "model_version": CHECKPOINT.name,
        "threshold": predictor.threshold,
        "n_samples": len(y_true),
        "n_skipped": skipped,
        "accuracy": round(accuracy, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "auc": round(auc, 4),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "class_counts": {
            "fire": int((y_true_arr == 1).sum()),
            "no_fire": int((y_true_arr == 0).sum()),
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(metrics, indent=2))
    logger.info("Metrics saved to %s", OUTPUT)

    # Annotate per-tile records with predicted label and outcome
    threshold = predictor.threshold
    for rec in per_tile:
        rec["y_pred"] = int(rec["fire_probability"] >= threshold)
        gt, pred = rec["y_true"], rec["y_pred"]
        if gt == 1 and pred == 1:
            rec["outcome"] = "tp"
        elif gt == 0 and pred == 1:
            rec["outcome"] = "fp"
        elif gt == 1 and pred == 0:
            rec["outcome"] = "fn"
        else:
            rec["outcome"] = "tn"

    predictions = {
        "model_version": CHECKPOINT.name,
        "threshold": threshold,
        "n_samples": len(per_tile),
        "tiles": per_tile,
    }
    PREDICTIONS_OUTPUT.write_text(json.dumps(predictions, indent=2))
    logger.info("Per-tile predictions saved to %s", PREDICTIONS_OUTPUT)

    logger.info(
        "Test results — Accuracy: %.4f  Precision: %.4f  Recall: %.4f  F1: %.4f  AUC: %.4f",
        accuracy, precision, recall, f1, auc,
    )
    logger.info("Confusion matrix: TP=%d  FP=%d  FN=%d  TN=%d", tp, fp, fn, tn)


if __name__ == "__main__":
    main()
