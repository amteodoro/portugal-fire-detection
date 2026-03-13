import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    fire_probability: float  # 0.0–1.0
    is_fire: bool            # True if above threshold
    confidence: float        # Same as fire_probability
    model_version: str       # Checkpoint filename
    model_type: str = "mobilevit"  # 'mobilevit' | 'multimodal'


def _load_model_from_config(config: dict):
    """Create model matching config['model_type'] (default: mobilevit)."""
    model_type = config.get("model_type", "mobilevit")
    if model_type == "multimodal":
        from ml.models.multimodal_model import create_multimodal_model
        return create_multimodal_model(config), model_type
    from ml.models.model import create_model
    return create_model(config), model_type


class FirePredictor:
    def __init__(self, checkpoint_path: str, device: str = "cpu"):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        self.config = checkpoint["config"]

        # MODEL_TYPE env var overrides checkpoint config (useful for testing)
        env_model_type = os.environ.get("MODEL_TYPE")
        if env_model_type:
            self.config["model_type"] = env_model_type

        self.model, self.model_type = _load_model_from_config(self.config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.eval()
        self.device = device
        self.threshold = self.config.get("inference", {}).get("confidence_threshold", 0.7)
        self.model_version = Path(checkpoint_path).name
        logger.info(
            "Loaded FirePredictor: model=%s type=%s threshold=%.2f",
            self.model_version,
            self.model_type,
            self.threshold,
        )

    def predict_from_array(
        self,
        tile: np.ndarray,
        s3_tile: Optional[np.ndarray] = None,
    ) -> PredictionResult:
        """Predict from a tile array.

        Args:
            tile:     S2 tile, shape (4, H, W), values 0–10000.
            s3_tile:  S3 tile, shape (2, H, W), values 0–10000 (optional).
                      Only used when model_type == 'multimodal'.
        """
        tile_norm = tile.astype(np.float32) / 10000.0
        tensor = torch.from_numpy(tile_norm).unsqueeze(0)  # (1, 4, H, W)
        mean = tensor.mean(dim=(2, 3), keepdim=True)
        std = tensor.std(dim=(2, 3), keepdim=True) + 1e-8
        tensor = (tensor - mean) / std
        tensor = tensor.to(self.device)

        s3_tensor = None
        if self.model_type == "multimodal" and s3_tile is not None:
            s3_norm = s3_tile.astype(np.float32) / 10000.0
            s3_tensor = torch.from_numpy(s3_norm).unsqueeze(0)
            s3_mean = s3_tensor.mean(dim=(2, 3), keepdim=True)
            s3_std = s3_tensor.std(dim=(2, 3), keepdim=True) + 1e-8
            s3_tensor = (s3_tensor - s3_mean) / s3_std
            s3_tensor = s3_tensor.to(self.device)

        with torch.no_grad():
            if self.model_type == "multimodal":
                logits = self.model(tensor, s3_tensor)
            else:
                logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            fire_prob = probs[0, 1].item()

        return PredictionResult(
            fire_probability=fire_prob,
            is_fire=fire_prob >= self.threshold,
            confidence=fire_prob,
            model_version=self.model_version,
            model_type=self.model_type,
        )
