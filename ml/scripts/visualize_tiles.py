"""
Visualize training tiles with false color composites.

Creates preview images of fire and no-fire tiles with fire locations marked.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict
import yaml

import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TileVisualizer:
    """Visualize training tiles with false color composites."""
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize visualizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tiles_dir = Path(self.config['data']['tiles_dir'])
        self.labels_dir = Path(self.config['data']['labels_dir'])
        self.output_dir = Path("data/samples/tiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Band indices (for 4-band: R, NIR, SWIR1, SWIR2)
        self.band_names = self.config['sentinel2']['band_names']
        self.raw_dir = Path(self.config['data']['raw_dir'])
        
        # RGB band names for true color
        self.rgb_save_names = self.config['sentinel2'].get('rgb_save_names', ['B02', 'B03', 'B04'])
    
    def load_rgb_for_tile(self, tile_path: Path) -> Optional[np.ndarray]:
        """
        Load true RGB bands from the source scene for a tile.
        
        Args:
            tile_path: Path to .npy tile file
            
        Returns:
            RGB array (H, W, 3) if available, None otherwise
        """
        # Parse tile filename to get scene info
        # Format: sceneId_fireId_type_random.npy
        filename = tile_path.stem
        parts = filename.split('_')
        
        if len(parts) < 4:
            return None
        
        # Reconstruct scene ID (first parts until we hit fire_id)
        # Typical scene ID: S2A_29SPB_20251009_0_L2A
        scene_id = '_'.join(parts[:5])  # Assume scene ID has 5 parts
        
        # Find scene date from tile path or scene_id
        # Scene directories are organized as: raw_dir/YYYY-MM-DD/scene_id/
        scene_dirs = list(self.raw_dir.glob(f"*/{scene_id}"))
        
        if not scene_dirs:
            logger.debug(f"Scene directory not found for {scene_id}")
            return None
        
        scene_dir = scene_dirs[0]
        
        # Load label to get tile extraction parameters
        label_path = self.labels_dir / f"{tile_path.stem}.json"
        if not label_path.exists():
            return None
        
        with open(label_path, 'r') as f:
            label_data = json.load(f)
        
        # Get tile position in scene
        row_start = label_data.get('scene_row_start')
        col_start = label_data.get('scene_col_start')
        tile_size = label_data.get('tile_size', 256)
        
        if row_start is None or col_start is None:
            logger.debug(f"Missing scene coordinates for {tile_path.stem}")
            return None
        
        # Load RGB bands from scene
        rgb_arrays = []
        
        for band_name in self.rgb_save_names:  # B02, B03, B04
            band_file = scene_dir / f"{band_name}.tif"
            
            if not band_file.exists():
                logger.debug(f"RGB band {band_name} not found in {scene_dir}")
                return None
            
            try:
                with rasterio.open(band_file) as src:
                    # Read the same tile region
                    data = src.read(1)
                    tile_data = data[row_start:row_start+tile_size, 
                                   col_start:col_start+tile_size]
                    rgb_arrays.append(tile_data)
            except Exception as e:
                logger.debug(f"Error reading RGB band {band_name}: {e}")
                return None
        
        # Stack as (H, W, 3)
        rgb = np.stack(rgb_arrays, axis=-1)
        return rgb
    
    def normalize_for_display(self, band_data: np.ndarray, percentile: int = 98) -> np.ndarray:
        """
        Normalize band data for display.
        
        Args:
            band_data: Raw band array
            percentile: Percentile for clipping
            
        Returns:
            Normalized array in [0, 1]
        """
        p_low, p_high = np.percentile(band_data, (2, percentile))
        normalized = np.clip((band_data - p_low) / (p_high - p_low), 0, 1)
        return normalized
    
    def create_false_color(self, tile: np.ndarray) -> np.ndarray:
        """
        Create false color composite (SWIR2-NIR-Red).
        
        Fire shows up as bright red/white in this composite.
        
        Args:
            tile: Numpy array of shape (4, H, W) with bands [R, NIR, SWIR1, SWIR2]
            
        Returns:
            RGB image array
        """
        # Extract bands
        red = tile[0]      # B04
        nir = tile[1]      # B08
        swir2 = tile[3]    # B12
        
        # Create false color composite: SWIR2 (Red), NIR (Green), Red (Blue)
        r = self.normalize_for_display(swir2)
        g = self.normalize_for_display(nir)
        b = self.normalize_for_display(red)
        
        # Stack into RGB
        rgb = np.stack([r, g, b], axis=-1)
        
        return rgb
    
    def visualize_tile(
        self,
        tile_path: Path,
        label: int,
        output_path: Path,
        fire_location: Optional[Dict] = None
    ):
        """
        Create visualization of a single tile.
        
        Args:
            tile_path: Path to .npy tile file
            label: 0 for no-fire, 1 for fire
            output_path: Path to save visualization
            fire_location: (row, col) of fire center if known
        """
        # Load tile
        tile = np.load(tile_path)
        
        # Create false color composite
        false_color = self.create_false_color(tile)
        
        # Try to load true RGB
        true_rgb = self.load_rgb_for_tile(tile_path)
        
        if true_rgb is not None:
            # Two panels: false color and true RGB
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Left: False color (fire detection)
            ax1.imshow(false_color)
            ax1.set_title('Fire Detection\n(SWIR2-NIR-Red)', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            # Right: True RGB
            rgb_normalized = self.normalize_for_display(true_rgb, percentile=98)
            ax2.imshow(rgb_normalized)
            ax2.set_title('True Color\n(Blue-Green-Red)', fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            axes = [ax1, ax2]
        else:
            # Single panel: false color only (no RGB available)
            fig, ax1 = plt.subplots(1, 1, figsize=(8, 8))
            
            ax1.imshow(false_color)
            ax1.set_title('Fire Detection\n(SWIR2-NIR-Red)', fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            axes = [ax1]
        
        # Mark fire location for positive samples
        if label == 1 and fire_location is not None:
            # Use actual fire coordinates from label metadata
            fire_row = fire_location.get('fire_pixel_row', tile.shape[1] // 2)
            fire_col = fire_location.get('fire_pixel_col', tile.shape[2] // 2)
            
            # Draw crosshair on all subplots
            for ax_subplot in axes:
                circle = Circle((fire_col, fire_row), radius=10, 
                              color='red', fill=False, linewidth=2)
                ax_subplot.add_patch(circle)
                
                circle2 = Circle((fire_col, fire_row), radius=2, 
                               color='red', fill=True)
                ax_subplot.add_patch(circle2)
                
                ax_subplot.text(fire_col, fire_row - 20, 'Fire Location', 
                       color='red', fontsize=10, fontweight='bold',
                       ha='center', va='bottom',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Add overall title
        label_text = "FIRE DETECTION" if label == 1 else "NO FIRE"
        color = 'red' if label == 1 else 'green'
        fig.suptitle(label_text, fontsize=16, fontweight='bold', color=color, y=0.98)
        
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Saved visualization: {output_path.name}")
    
    def create_sample_grid(self, n_samples: int = 6):
        """
        Create a grid of sample visualizations.
        
        Args:
            n_samples: Number of samples to visualize (half fire, half no-fire)
        """
        logger.info(f"Creating {n_samples} sample visualizations...")
        
        # Find tile files
        fire_tiles = list(self.tiles_dir.rglob("*positive*.npy"))
        no_fire_tiles = list(self.tiles_dir.rglob("*negative*.npy"))
        
        logger.info(f"Found {len(fire_tiles)} fire tiles, {len(no_fire_tiles)} no-fire tiles")
        
        if not fire_tiles and not no_fire_tiles:
            logger.error("No tiles found. Generate dataset first.")
            return
        
        # Sample tiles
        n_fire = min(n_samples // 2, len(fire_tiles))
        n_no_fire = min(n_samples // 2, len(no_fire_tiles))
        
        import random
        # Seeding now handled by caller; if no seed set, Python uses system entropy
        
        sampled_fire = random.sample(fire_tiles, n_fire) if fire_tiles else []
        sampled_no_fire = random.sample(no_fire_tiles, n_no_fire) if no_fire_tiles else []
        
        # Visualize each sample
        sample_info = []
        
        for i, tile_path in enumerate(sampled_fire):
            output_path = self.output_dir / f"fire_sample_{i+1}.png"
            
            # Load label to get fire location
            label_path = self.labels_dir / f"{tile_path.stem}.json"
            fire_location = None
            if label_path.exists():
                with open(label_path, 'r') as f:
                    label_data = json.load(f)
                    if 'fire_pixel_row' in label_data and 'fire_pixel_col' in label_data:
                        fire_location = {
                            'fire_pixel_row': label_data['fire_pixel_row'],
                            'fire_pixel_col': label_data['fire_pixel_col']
                        }
            
            self.visualize_tile(tile_path, label=1, output_path=output_path, 
                              fire_location=fire_location)
            sample_info.append({
                "type": "fire",
                "filename": output_path.name,
                "tile_path": str(tile_path)
            })
        
        for i, tile_path in enumerate(sampled_no_fire):
            output_path = self.output_dir / f"no_fire_sample_{i+1}.png"
            self.visualize_tile(tile_path, label=0, output_path=output_path)
            sample_info.append({
                "type": "no_fire",
                "filename": output_path.name,
                "tile_path": str(tile_path)
            })
        
        # Save sample info
        info_path = self.output_dir / "samples_info.json"
        with open(info_path, 'w') as f:
            json.dump(sample_info, f, indent=2)
        
        logger.info(f"\n{'='*50}")
        logger.info(f"Created {len(sample_info)} visualizations")
        logger.info(f"Fire samples: {n_fire}")
        logger.info(f"No-fire samples: {n_no_fire}")
        logger.info(f"Saved to: {self.output_dir}")
        logger.info(f"{'='*50}")


def main():
    """CLI entry point."""
    import argparse
    import random
    
    parser = argparse.ArgumentParser(description="Visualize training tiles")
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Config file path')
    parser.add_argument('--n-samples', type=int, default=6, help='Number of samples to visualize')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for sampling (optional)')
    
    args = parser.parse_args()
    
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
    
    visualizer = TileVisualizer(args.config)
    visualizer.create_sample_grid(args.n_samples)


if __name__ == "__main__":
    main()
