"""
Prepare train/val/test splits from processed fire dataset.

Reads tiles from processed directory and creates stratified splits.
"""

import json
import shutil
from pathlib import Path
from typing import Dict, List
import yaml
import numpy as np
from sklearn.model_selection import train_test_split


def load_config(config_path: str = "ml/configs/config.yaml") -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def collect_tiles(processed_dir: Path) -> Dict[str, List[Path]]:
    """
    Collect all tiles from processed directory.
    
    Returns:
        Dictionary with 'fire' and 'no_fire' tile paths
    """
    tiles = {
        'fire': [],
        'no_fire': []
    }
    
    # Look for tiles in processed directory structure
    if not processed_dir.exists():
        print(f"Processed directory not found: {processed_dir}")
        return tiles
    
    # Find all .npy tile files
    for tile_file in processed_dir.rglob("*.npy"):
        # Determine if fire or no-fire based on path or metadata
        # Assume tiles with 'fire' in path or associated with FIRMS are positive
        if 'fire' in str(tile_file).lower() or 'positive' in str(tile_file).lower():
            tiles['fire'].append(tile_file)
        else:
            tiles['no_fire'].append(tile_file)
    
    return tiles


def create_splits(
    tiles: Dict[str, List[Path]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, Dict[str, List[Path]]]:
    """
    Create stratified train/val/test splits.
    
    Args:
        tiles: Dictionary of fire and no_fire tile paths
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with splits: {split: {class: [paths]}}
    """
    np.random.seed(random_seed)
    
    splits = {
        'train': {'fire': [], 'no_fire': []},
        'val': {'fire': [], 'no_fire': []},
        'test': {'fire': [], 'no_fire': []}
    }
    
    # Split each class separately (stratified)
    for class_name, tile_paths in tiles.items():
        if len(tile_paths) == 0:
            print(f"Warning: No tiles found for class {class_name}")
            continue
        
        # First split: train vs (val + test)
        train_tiles, temp_tiles = train_test_split(
            tile_paths,
            train_size=train_ratio,
            random_state=random_seed
        )
        
        # Second split: val vs test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_tiles, test_tiles = train_test_split(
            temp_tiles,
            train_size=val_size,
            random_state=random_seed
        )
        
        splits['train'][class_name] = train_tiles
        splits['val'][class_name] = val_tiles
        splits['test'][class_name] = test_tiles
    
    return splits


def save_splits(
    splits: Dict[str, Dict[str, List[Path]]],
    output_dir: Path,
    symlink: bool = True
):
    """
    Save splits to organized directory structure.
    
    Args:
        splits: Dictionary with splits
        output_dir: Output directory for splits
        symlink: If True, create symlinks; otherwise copy files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for split_name, split_data in splits.items():
        split_dir = output_dir / split_name
        split_dir.mkdir(exist_ok=True)
        
        for class_name, tile_paths in split_data.items():
            class_dir = split_dir / class_name
            class_dir.mkdir(exist_ok=True)
            
            print(f"  {split_name}/{class_name}: {len(tile_paths)} tiles")
            
            for tile_path in tile_paths:
                dest = class_dir / tile_path.name
                
                if symlink:
                    # Create symlink to original file
                    if dest.exists():
                        dest.unlink()
                    dest.symlink_to(tile_path.resolve())
                else:
                    # Copy file
                    shutil.copy2(tile_path, dest)
    
    # Save split metadata
    metadata = {
        'splits': {
            split: {
                class_name: len(paths)
                for class_name, paths in split_data.items()
            }
            for split, split_data in splits.items()
        }
    }
    
    metadata_path = output_dir / "splits_info.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nSaved split metadata to {metadata_path}")


def main():
    """Main execution."""
    print("=" * 80)
    print("Preparing Train/Val/Test Splits")
    print("=" * 80)
    
    # Load config
    config = load_config()
    
    # Use tiles_dir instead of processed_dir
    tiles_dir = Path(config['data']['tiles_dir'])
    output_dir = Path(config['data']['processed_dir']) / "splits"
    
    # Collect tiles
    print(f"\nCollecting tiles from {tiles_dir}...")
    tiles = collect_tiles(tiles_dir)
    
    print(f"\nFound:")
    print(f"  Fire tiles: {len(tiles['fire'])}")
    print(f"  No-fire tiles: {len(tiles['no_fire'])}")
    
    if len(tiles['fire']) == 0 and len(tiles['no_fire']) == 0:
        print("\n✗ No tiles found. Run dataset generation first:")
        print("  docker compose exec web python ml/data/create_dataset.py")
        return
    
    # Create splits
    print("\nCreating splits...")
    splits = create_splits(
        tiles,
        train_ratio=config['splits']['train'],
        val_ratio=config['splits']['val'],
        test_ratio=config['splits']['test'],
        random_seed=config['splits']['random_seed']
    )
    
    print("\nSplit distribution:")
    # Save splits
    save_splits(splits, output_dir, symlink=True)
    
    print("\n" + "=" * 80)
    print("✓ Splits prepared successfully!")
    print(f"Output directory: {output_dir}")
    print("=" * 80)


if __name__ == "__main__":
    main()
