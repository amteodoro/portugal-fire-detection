"""
Fire detection model training package.
"""

from .model import FireDetectionClassifier, create_model
from .dataset import FireDetectionDataset, create_dataloaders

__all__ = [
    'FireDetectionClassifier',
    'create_model',
    'FireDetectionDataset',
    'create_dataloaders'
]
