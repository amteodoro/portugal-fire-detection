"""
PyTorch Dataset for fire detection training tiles.

Loads preprocessed .npy tiles and corresponding labels from JSON files.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms as transforms

logger = logging.getLogger(__name__)


class FireDetectionDataset(Dataset):
    """Dataset for fire/no-fire classification from Sentinel-2 tiles."""
    
    def __init__(
        self,
        tiles_dir: str,
        labels_dir: str,
        split_file: Optional[str] = None,
        transform: Optional[transforms.Compose] = None,
        normalize: bool = True
    ):
        """
        Initialize dataset.
        
        Args:
            tiles_dir: Directory containing .npy tile files
            labels_dir: Directory containing .json label files
            split_file: Optional path to split file (train/val/test indices)
            transform: Optional torchvision transforms
            normalize: Whether to normalize tile values
        """
        self.tiles_dir = Path(tiles_dir)
        self.labels_dir = Path(labels_dir)
        self.transform = transform
        self.normalize = normalize
        
        # Load all tile paths
        all_tiles = sorted(list(self.tiles_dir.glob("*.npy")))
        
        # Filter by split if provided
        if split_file:
            with open(split_file, 'r') as f:
                split_data = json.load(f)
                split_tile_ids = set(split_data['tile_ids'])
            
            self.tiles = [t for t in all_tiles if t.stem in split_tile_ids]
            logger.info(f"Loaded {len(self.tiles)} tiles from split: {split_file}")
        else:
            self.tiles = all_tiles
            logger.info(f"Loaded {len(self.tiles)} tiles (no split filter)")
        
        if len(self.tiles) == 0:
            raise ValueError(f"No tiles found in {tiles_dir}")
    
    def __len__(self) -> int:
        return len(self.tiles)
    
    def __getitem__(self, idx: int) -> Optional[Tuple[torch.Tensor, int, Dict]]:
        """
        Get a single training sample.
        
        Returns:
            Tuple of (tile_tensor, label, metadata) or None if sample fails to load
        """
        try:
            tile_path = self.tiles[idx]
            label_path = self.labels_dir / f"{tile_path.stem}.json"
            
            # Load tile (shape: [4, 256, 256])
            tile = np.load(tile_path)
            
            # Load label
            with open(label_path, 'r') as f:
                label_data = json.load(f)
            
            label = label_data['label']  # 0 or 1
            
            # Convert to torch tensor
            tile_tensor = torch.from_numpy(tile).float()
            
            # Normalize if requested (tiles are already in 0-1 range from preprocessing)
            if self.normalize:
                # Per-channel normalization (compute mean/std across spatial dimensions)
                mean = tile_tensor.mean(dim=(1, 2), keepdim=True)
                std = tile_tensor.std(dim=(1, 2), keepdim=True) + 1e-8
                tile_tensor = (tile_tensor - mean) / std
            
            # Apply transforms if provided
            if self.transform:
                tile_tensor = self.transform(tile_tensor)
            
            # Return tile, label, and metadata
            metadata = {
                'tile_id': label_data['tile_id'],
                'scene_id': label_data.get('scene_id', ''),
                'fire_pixel_row': label_data.get('fire_pixel_row'),
                'fire_pixel_col': label_data.get('fire_pixel_col')
            }
            
            return tile_tensor, label, metadata
        
        except Exception as e:
            logger.warning(f"Failed to load sample {idx} ({tile_path.stem if 'tile_path' in locals() else 'unknown'}): {e}")
            return None


def collate_fn_filter_none(batch):
    """
    Custom collate function that filters out None values from batch.
    
    Args:
        batch: List of samples from dataset (may contain None)
        
    Returns:
        Tuple of (inputs, labels, metadata) with None values filtered out
    """
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If all samples failed, return empty batch
    if len(batch) == 0:
        return torch.empty(0), torch.empty(0), []
    
    # Separate into inputs, labels, and metadata
    inputs = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch], dtype=torch.long)
    metadata = [item[2] for item in batch]
    
    return inputs, labels, metadata


def get_data_transforms(config: Dict) -> Dict[str, transforms.Compose]:
    """
    Get train and validation transforms from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with 'train' and 'val' transforms
    """
    aug_config = config.get('augmentation', {})
    
    train_transforms = []
    
    # Data augmentation for training
    if aug_config.get('enabled', False):
        if aug_config.get('flip_horizontal', False):
            train_transforms.append(transforms.RandomHorizontalFlip(p=0.5))
        
        if aug_config.get('flip_vertical', False):
            train_transforms.append(transforms.RandomVerticalFlip(p=0.5))
        
        if aug_config.get('rotation', False):
            # Random rotation (any angle)
            train_transforms.append(transforms.RandomRotation(degrees=[-180, 180]))
    
    train_transform = transforms.Compose(train_transforms) if train_transforms else None
    
    # No augmentation for validation/test
    val_transform = None
    
    return {
        'train': train_transform,
        'val': val_transform,
        'test': val_transform
    }


def create_dataloaders(config: Dict) -> Dict[str, torch.utils.data.DataLoader]:
    """
    Create train, validation, and test dataloaders.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary with 'train', 'val', 'test' dataloaders
    """
    tiles_dir = config['data']['tiles_dir']
    labels_dir = config['data']['labels_dir']
    processed_dir = Path(config['data']['processed_dir'])
    
    batch_size = config['training']['batch_size']
    
    # Get transforms
    transforms_dict = get_data_transforms(config)
    
    # Create datasets
    datasets = {}
    for split in ['train', 'val', 'test']:
        split_file = processed_dir / f"{split}_split.json"
        
        split_file_to_pass = None
        if split_file.exists():
            split_file_to_pass = split_file
        else:
            if split in ['train', 'val']:
                logger.warning(f"Required split file not found: {split_file}. Proceeding without split filter.")
            else:
                logger.warning(f"Split file not found: {split_file}. Skipping test dataset creation.")
                continue

        datasets[split] = FireDetectionDataset(
            tiles_dir=tiles_dir,
            labels_dir=labels_dir,
            split_file=split_file_to_pass,
            transform=transforms_dict[split],
            normalize=True
        )
    
    # Create dataloaders
    dataloaders = {}
    for split, dataset in datasets.items():
        shuffle = (split == 'train')
        
        dataloaders[split] = torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Avoid Docker shared memory issues
            pin_memory=False,  # No GPU in container
            drop_last=(split == 'train'),  # Drop last incomplete batch for training
            collate_fn=collate_fn_filter_none  # Filter out None values from failed samples
        )
        
        logger.info(f"{split.capitalize()} dataloader: {len(dataset)} samples, {len(dataloaders[split])} batches")
    
    return dataloaders
