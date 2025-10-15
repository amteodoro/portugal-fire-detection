"""
PyTorch Dataset for fire detection from Sentinel-2 tiles.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from typing import Tuple, Optional, Dict
import yaml


class FireDetectionDataset(Dataset):
    """Dataset for fire detection from Sentinel-2 tiles."""
    
    def __init__(
        self,
        split_dir: Path,
        transform=None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            split_dir: Path to split directory (train/val/test)
            transform: Optional transforms to apply
            normalize: Whether to normalize bands
        """
        self.split_dir = Path(split_dir)
        self.transform = transform
        self.normalize = normalize
        
        # Collect all tiles
        self.samples = []
        self.labels = []
        
        # Fire tiles (class 1)
        fire_dir = self.split_dir / "fire"
        if fire_dir.exists():
            for tile_path in fire_dir.glob("*.npy"):
                self.samples.append(tile_path)
                self.labels.append(1)
        
        # No-fire tiles (class 0)
        no_fire_dir = self.split_dir / "no_fire"
        if no_fire_dir.exists():
            for tile_path in no_fire_dir.glob("*.npy"):
                self.samples.append(tile_path)
                self.labels.append(0)
        
        print(f"Loaded {len(self.samples)} samples from {split_dir.name}")
        print(f"  Fire: {sum(self.labels)}, No-fire: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get item by index.
        
        Returns:
            Tuple of (tile, label)
        """
        # Load tile
        tile_path = self.samples[idx]
        tile = np.load(tile_path)  # Shape: (H, W, C) or (C, H, W)
        
        # Ensure channel-first format (C, H, W)
        if tile.shape[-1] == 4:  # (H, W, C)
            tile = np.transpose(tile, (2, 0, 1))
        
        # Convert to float32
        tile = tile.astype(np.float32)
        
        # Normalize if requested
        if self.normalize:
            tile = self._normalize_tile(tile)
        
        # Apply transforms
        if self.transform:
            tile = self.transform(tile)
        
        # Convert to tensor
        tile = torch.from_numpy(tile)
        label = self.labels[idx]
        
        return tile, label
    
    def _normalize_tile(self, tile: np.ndarray) -> np.ndarray:
        """
        Normalize tile bands.
        
        Uses per-band normalization based on typical Sentinel-2 ranges.
        """
        # Typical Sentinel-2 L2A reflectance ranges (0-10000)
        # Normalize to [0, 1] range
        tile = np.clip(tile / 10000.0, 0, 1)
        
        # Standardize (optional, based on ImageNet stats or custom)
        # For now, just scale to [0, 1]
        return tile
    
    def get_class_weights(self) -> torch.Tensor:
        """Calculate class weights for imbalanced dataset."""
        labels = np.array(self.labels)
        unique, counts = np.unique(labels, return_counts=True)
        weights = 1.0 / counts
        weights = weights / weights.sum() * len(unique)
        return torch.FloatTensor(weights)


def get_transforms(augment: bool = False):
    """
    Get data transforms/augmentation.
    
    Args:
        augment: Whether to apply augmentation
        
    Returns:
        Transform function
    """
    import torchvision.transforms as T
    
    transforms = []
    
    if augment:
        # Random horizontal flip
        transforms.append(T.RandomHorizontalFlip(p=0.5))
        # Random vertical flip
        transforms.append(T.RandomVerticalFlip(p=0.5))
        # Random rotation (90, 180, 270 degrees)
        transforms.append(T.RandomRotation(degrees=[0, 90, 180, 270]))
        # Random brightness/contrast
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    
    return T.Compose(transforms) if transforms else None


def create_dataloaders(
    config: Dict,
    batch_size: Optional[int] = None,
    num_workers: int = 4
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train/val/test dataloaders.
    
    Args:
        config: Configuration dictionary
        batch_size: Batch size (overrides config if provided)
        num_workers: Number of dataloader workers
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    splits_dir = Path(config['data']['processed_dir']) / "splits"
    
    if batch_size is None:
        batch_size = config['training']['batch_size']
    
    # Training set with augmentation
    train_transform = get_transforms(augment=config['augmentation']['enabled'])
    train_dataset = FireDetectionDataset(
        splits_dir / "train",
        transform=train_transform,
        normalize=True
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Validation set (no augmentation)
    val_dataset = FireDetectionDataset(
        splits_dir / "val",
        transform=None,
        normalize=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Test set (no augmentation)
    test_dataset = FireDetectionDataset(
        splits_dir / "test",
        transform=None,
        normalize=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Test dataset loading
    import yaml
    
    with open("ml/configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    try:
        train_loader, val_loader, test_loader = create_dataloaders(config, batch_size=4)
        
        print("\nDataset loaded successfully!")
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        print(f"Test batches: {len(test_loader)}")
        
        # Test loading a batch
        batch, labels = next(iter(train_loader))
        print(f"\nBatch shape: {batch.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Labels: {labels}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure splits exist. Run: python ml/models/prepare_splits.py")
