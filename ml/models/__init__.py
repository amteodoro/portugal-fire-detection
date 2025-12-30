"""
Fire detection model training package.
"""

from .model import HuggingFaceFireDetector, create_model
from .dataset import FireDetectionDataset, create_dataloaders

__all__ = [
    'HuggingFaceFireDetector',
    'create_model',
    'FireDetectionDataset',
    'create_dataloaders'
]
