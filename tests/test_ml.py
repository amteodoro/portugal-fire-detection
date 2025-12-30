"""
Unit tests for ML components.

Run with: docker-compose exec web uv run pytest tests/test_ml.py
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import numpy as np
import torch


class TestFireDetectionDataset:
    """Tests for ml/models/dataset.py"""

    def test_dataset_initialization(self, tmp_path, sample_tile_data):
        """Test dataset initializes correctly with tiles and labels."""
        from ml.models.dataset import FireDetectionDataset

        # Create test tile and label files
        tiles_dir = tmp_path / "tiles"
        labels_dir = tmp_path / "labels"
        tiles_dir.mkdir()
        labels_dir.mkdir()

        tile_id = sample_tile_data["label"]["tile_id"]
        np.save(tiles_dir / f"{tile_id}.npy", sample_tile_data["tile"])
        with open(labels_dir / f"{tile_id}.json", "w") as f:
            json.dump(sample_tile_data["label"], f)

        dataset = FireDetectionDataset(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_dir),
            normalize=False
        )

        assert len(dataset) == 1

    def test_dataset_getitem(self, tmp_path, sample_tile_data):
        """Test dataset __getitem__ returns correct format."""
        from ml.models.dataset import FireDetectionDataset

        tiles_dir = tmp_path / "tiles"
        labels_dir = tmp_path / "labels"
        tiles_dir.mkdir()
        labels_dir.mkdir()

        tile_id = sample_tile_data["label"]["tile_id"]
        np.save(tiles_dir / f"{tile_id}.npy", sample_tile_data["tile"])
        with open(labels_dir / f"{tile_id}.json", "w") as f:
            json.dump(sample_tile_data["label"], f)

        dataset = FireDetectionDataset(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_dir),
            normalize=False
        )

        tile_tensor, label, metadata = dataset[0]

        assert isinstance(tile_tensor, torch.Tensor)
        assert tile_tensor.shape == (4, 128, 128)
        assert label == 1  # Fire label
        assert metadata["tile_id"] == tile_id

    def test_dataset_normalization(self, tmp_path, sample_tile_data):
        """Test dataset normalization works correctly."""
        from ml.models.dataset import FireDetectionDataset

        tiles_dir = tmp_path / "tiles"
        labels_dir = tmp_path / "labels"
        tiles_dir.mkdir()
        labels_dir.mkdir()

        tile_id = sample_tile_data["label"]["tile_id"]
        np.save(tiles_dir / f"{tile_id}.npy", sample_tile_data["tile"])
        with open(labels_dir / f"{tile_id}.json", "w") as f:
            json.dump(sample_tile_data["label"], f)

        dataset = FireDetectionDataset(
            tiles_dir=str(tiles_dir),
            labels_dir=str(labels_dir),
            normalize=True
        )

        tile_tensor, _, _ = dataset[0]

        # After normalization, mean should be close to 0
        for c in range(4):
            channel_mean = tile_tensor[c].mean().item()
            assert abs(channel_mean) < 0.1, f"Channel {c} mean not normalized"

    def test_dataset_empty_raises_error(self, tmp_path):
        """Test dataset raises error when no tiles found."""
        from ml.models.dataset import FireDetectionDataset

        tiles_dir = tmp_path / "empty_tiles"
        labels_dir = tmp_path / "empty_labels"
        tiles_dir.mkdir()
        labels_dir.mkdir()

        with pytest.raises(ValueError, match="No tiles found"):
            FireDetectionDataset(
                tiles_dir=str(tiles_dir),
                labels_dir=str(labels_dir)
            )


class TestCollateFn:
    """Tests for collate function."""

    def test_collate_filters_none(self):
        """Test collate function filters out None values."""
        from ml.models.dataset import collate_fn_filter_none

        batch = [
            (torch.randn(4, 128, 128), 0, {"tile_id": "1"}),
            None,
            (torch.randn(4, 128, 128), 1, {"tile_id": "2"}),
            None,
        ]

        inputs, labels, metadata = collate_fn_filter_none(batch)

        assert inputs.shape == (2, 4, 128, 128)
        assert len(labels) == 2
        assert len(metadata) == 2

    def test_collate_empty_batch(self):
        """Test collate function handles all-None batch."""
        from ml.models.dataset import collate_fn_filter_none

        batch = [None, None, None]

        inputs, labels, metadata = collate_fn_filter_none(batch)

        assert inputs.shape == (0,)
        assert labels.shape == (0,)
        assert metadata == []


class TestDataTransforms:
    """Tests for data augmentation transforms."""

    def test_get_transforms_disabled(self):
        """Test get_data_transforms with augmentation disabled."""
        from ml.models.dataset import get_data_transforms

        config = {"augmentation": {"enabled": False}}
        transforms = get_data_transforms(config)

        assert transforms["train"] is None
        assert transforms["val"] is None
        assert transforms["test"] is None

    def test_get_transforms_enabled(self):
        """Test get_data_transforms with augmentation enabled."""
        from ml.models.dataset import get_data_transforms

        config = {
            "augmentation": {
                "enabled": True,
                "flip_horizontal": True,
                "flip_vertical": True,
                "rotation": True,
            }
        }
        transforms = get_data_transforms(config)

        assert transforms["train"] is not None
        assert transforms["val"] is None  # No augmentation for val/test


class TestModelCreation:
    """Tests for model creation."""

    def test_create_model_mobilevit(self):
        """Test creating MobileViT model."""
        from ml.models.model import create_model

        config = {
            "model": {
                "type": "huggingface_transformer",
                "model_name": "apple/mobilevit-small",
                "num_classes": 2,
                "input_channels": 4,
                "pretrained": False,  # Don't download weights in tests
                "freeze_backbone": False,
            }
        }

        model = create_model(config)

        assert model is not None
        assert model.num_classes == 2
        assert model.input_channels == 4

    def test_model_forward_pass(self):
        """Test model forward pass with dummy input."""
        from ml.models.model import create_model

        config = {
            "model": {
                "type": "huggingface_transformer",
                "model_name": "apple/mobilevit-small",
                "num_classes": 2,
                "input_channels": 4,
                "pretrained": False,
                "freeze_backbone": False,
            }
        }

        model = create_model(config)
        model.eval()

        # Dummy input (batch_size=2, channels=4, height=256, width=256)
        dummy_input = torch.randn(2, 4, 256, 256)

        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (2, 2)  # batch_size, num_classes

    def test_model_count_parameters(self):
        """Test counting model parameters."""
        from ml.models.model import create_model

        config = {
            "model": {
                "type": "huggingface_transformer",
                "model_name": "apple/mobilevit-small",
                "num_classes": 2,
                "input_channels": 4,
                "pretrained": False,
                "freeze_backbone": False,
            }
        }

        model = create_model(config)
        param_count = model.count_parameters()

        assert param_count > 0
        assert isinstance(param_count, int)


class TestHuggingFaceFireDetector:
    """Tests for HuggingFace model wrapper."""

    def test_adapt_first_layer_mobilevit(self):
        """Test first layer adaptation for MobileViT."""
        from ml.models.model import HuggingFaceFireDetector

        model = HuggingFaceFireDetector(
            model_name="apple/mobilevit-small",
            num_classes=2,
            input_channels=4,
            pretrained=False
        )

        # Check that model accepts 4-channel input
        dummy_input = torch.randn(1, 4, 256, 256)
        with torch.no_grad():
            output = model(dummy_input)

        assert output.shape == (1, 2)

    def test_freeze_backbone(self):
        """Test backbone freezing."""
        from ml.models.model import HuggingFaceFireDetector

        model = HuggingFaceFireDetector(
            model_name="apple/mobilevit-small",
            num_classes=2,
            input_channels=4,
            pretrained=False,
            freeze_backbone=True
        )

        # Count trainable vs frozen parameters
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())

        # With frozen backbone, trainable should be much less than total
        assert trainable < total
        assert trainable > 0  # Classifier head should be trainable


class TestTrainingUtils:
    """Tests for training utilities."""

    def test_early_stopping_init(self):
        """Test EarlyStopping initialization."""
        from ml.models.train import EarlyStopping

        es = EarlyStopping(patience=10, min_delta=0.001)

        assert es.patience == 10
        assert es.min_delta == 0.001
        assert es.counter == 0
        assert es.best_loss is None
        assert es.early_stop is False

    def test_early_stopping_improvement(self):
        """Test EarlyStopping with improving loss."""
        from ml.models.train import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.01)

        es(1.0)  # First loss
        assert es.best_loss == 1.0
        assert es.counter == 0

        es(0.5)  # Improvement
        assert es.best_loss == 0.5
        assert es.counter == 0

    def test_early_stopping_triggers(self):
        """Test EarlyStopping triggers after patience exceeded."""
        from ml.models.train import EarlyStopping

        es = EarlyStopping(patience=3, min_delta=0.01)

        es(1.0)  # First loss
        es(1.0)  # No improvement
        es(1.0)  # No improvement
        es(1.0)  # No improvement - should trigger

        assert es.early_stop is True
