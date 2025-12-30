"""
Visualize training tiles with comprehensive views.

Creates preview images showing:
- All 4 spectral/thermal bands individually
- False color composite for fire detection
- Location map showing tile position in Portugal/Iberia

Enhanced version with geographic context and multi-band display.
"""

import json
import logging
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import yaml
import re

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
from matplotlib.gridspec import GridSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class TileVisualizer:
    """Visualize training tiles with all bands and location context."""
    
    # Portugal/Iberia bounding box for the overview map
    IBERIA_BOUNDS = {
        'lon_min': -10.5,
        'lon_max': -5.5,
        'lat_min': 36.5,
        'lat_max': 42.5
    }
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize visualizer with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tiles_dir = Path(self.config['data']['tiles_dir'])
        self.labels_dir = Path(self.config['data']['labels_dir'])
        self.output_dir = Path("data/samples/tiles")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Band names for display
        self.s2_band_names = ['Red (B04)', 'NIR (B08)', 'SWIR1 (B11)', 'SWIR2 (B12)']
        self.s3_band_names = ['Red Proxy', 'NIR Proxy', 'F1 (3.74um)', 'F2 (10.85um)']
    
    def normalize_for_display(self, band_data: np.ndarray, percentile: int = 98) -> np.ndarray:
        """
        Normalize band data for display with improved handling.
        
        Args:
            band_data: Raw band array
            percentile: Percentile for clipping
            
        Returns:
            Normalized array in [0, 1]
        """
        # Remove NaN and inf values
        valid_data = band_data[np.isfinite(band_data)]
        
        if len(valid_data) == 0:
            return np.zeros_like(band_data, dtype=float)
        
        # Calculate percentiles on valid data only
        p_low, p_high = np.percentile(valid_data, (2, percentile))
        
        # Check if data has range
        if p_high - p_low < 1e-6:
            p_low = valid_data.min()
            p_high = valid_data.max()
            
            if p_high - p_low < 1e-6:
                return np.full_like(band_data, 0.5, dtype=float)
        
        # Normalize
        normalized = np.clip((band_data - p_low) / (p_high - p_low), 0, 1)
        return normalized
    
    def create_false_color(self, tile: np.ndarray) -> np.ndarray:
        """
        Create false color composite (SWIR2-NIR-Red) for fire detection.
        
        Args:
            tile: Numpy array of shape (4, H, W) with bands [R, NIR, SWIR1, SWIR2]
            
        Returns:
            RGB array (H, W, 3) for display
        """
        if tile.shape[0] < 4:
            raise ValueError(f"Expected 4 bands, got {tile.shape[0]}")
        
        # False color: SWIR2 -> Red, NIR -> Green, Red -> Blue
        # This highlights fires as bright red/white spots
        false_color = np.stack([
            self.normalize_for_display(tile[3]),  # SWIR2 -> Red channel
            self.normalize_for_display(tile[1]),  # NIR -> Green channel
            self.normalize_for_display(tile[0])   # Red -> Blue channel
        ], axis=-1)
        
        return false_color
    
    def parse_fire_coords(self, fire_id: str) -> Optional[Tuple[float, float]]:
        """
        Parse latitude and longitude from fire_id string.
        
        Fire IDs are formatted like: VIIRS_2025-07-28_41.81562_-8.25049
        
        Returns:
            Tuple of (lat, lon) or None if parsing fails
        """
        if not fire_id or fire_id == 'none':
            return None
        
        # Try to extract last two numeric values (lat, lon)
        parts = fire_id.split('_')
        if len(parts) >= 4:
            try:
                lat = float(parts[-2])
                lon = float(parts[-1])
                # Validate reasonable bounds for Portugal
                if 35 < lat < 45 and -12 < lon < 0:
                    return (lat, lon)
            except (ValueError, IndexError):
                pass
        
        return None
    
    def draw_portugal_outline(self, ax):
        """
        Draw a simplified Portugal outline on the map axes.
        Uses approximate polygon coordinates.
        """
        # Simplified Portugal mainland border (approximate)
        portugal_outline = [
            (-8.9, 41.9), (-8.1, 42.0), (-7.4, 41.9), (-6.9, 41.95),
            (-6.6, 41.7), (-6.5, 41.3), (-6.8, 40.9), (-6.9, 40.4),
            (-7.0, 39.9), (-7.3, 39.5), (-7.0, 39.0), (-7.4, 38.2),
            (-7.5, 37.5), (-7.9, 37.0), (-8.8, 36.95), (-8.9, 37.5),
            (-9.0, 38.0), (-9.5, 38.7), (-9.2, 39.3), (-9.0, 39.6),
            (-9.1, 40.0), (-8.8, 40.7), (-8.9, 41.2), (-8.9, 41.9)
        ]
        
        lons = [p[0] for p in portugal_outline]
        lats = [p[1] for p in portugal_outline]
        ax.plot(lons, lats, 'b-', linewidth=2, alpha=0.7)
        ax.fill(lons, lats, color='lightblue', alpha=0.3)
        
        # Add some city markers for reference
        cities = {
            'Lisboa': (-9.14, 38.72),
            'Porto': (-8.61, 41.15),
            'Coimbra': (-8.43, 40.21),
            'Faro': (-7.93, 37.02)
        }
        for city, (lon, lat) in cities.items():
            ax.plot(lon, lat, 'ko', markersize=4)
            ax.annotate(city, (lon, lat), fontsize=7, ha='left', 
                       xytext=(3, 3), textcoords='offset points')
    
    def visualize_tile(self, 
                      tile_path: Path, 
                      label: int, 
                      output_path: Path,
                      fire_location: Optional[Dict] = None):
        """
        Create a comprehensive visualization of a tile.
        
        Layout (3 rows):
        - Row 1: 4 individual bands
        - Row 2: False color composite, Fire detection overlay, Thermal view
        - Row 3: Location map (spanning width)
        
        Args:
            tile_path: Path to .npy tile file
            label: 0 = no fire, 1 = fire
            output_path: Where to save the visualization
            fire_location: Optional dict with fire coordinates
        """
        # Load tile
        tile = np.load(tile_path)
        
        # Load label data
        label_path = self.labels_dir / f"{tile_path.stem}.json"
        label_data = None
        if label_path.exists():
            with open(label_path, 'r') as f:
                label_data = json.load(f)
        
        # Detect satellite type from scene_id
        scene_id = label_data.get('scene_id', '') if label_data else ''
        is_s2 = scene_id.startswith('S2')
        is_s3 = not is_s2 and len(scene_id) > 10 and '-' in scene_id
        
        # Get tile center coordinates (for map display)
        # Priority: 1) tile_center_lat/lon in label, 2) parse from fire_id, 3) None
        tile_coords = None
        if label_data:
            if 'tile_center_lat' in label_data and 'tile_center_lon' in label_data:
                tile_coords = (label_data['tile_center_lat'], label_data['tile_center_lon'])
            elif label_data.get('fire_id'):
                tile_coords = self.parse_fire_coords(label_data['fire_id'])
        
        # Get fire pixel location within tile
        fire_pixel = None
        if label_data:
            fire_row = label_data.get('fire_pixel_row')
            fire_col = label_data.get('fire_pixel_col')
            if fire_row is not None and fire_col is not None:
                fire_pixel = (fire_row, fire_col)
        
        # Create figure with custom grid
        # 3 rows: bands (4 cols), composites (2 cols), map (1 wide)
        fig = plt.figure(figsize=(16, 20))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 0.8],
                     hspace=0.25, wspace=0.15)
        sat_type = "Sentinel-2" if is_s2 else "Sentinel-3"
        
        # ========================================
        # ROW 1: Individual Bands / Thermal Views
        # ========================================
        
        if is_s3:
            # SENTINEL-3: Show thermal bands only (skip sparse S5/S6)
            # Panel 1: F1 thermal (3.74um)
            ax1 = fig.add_subplot(gs[0, 0])
            if tile.shape[0] >= 3:
                f1 = self.normalize_for_display(tile[2], percentile=99)
                ax1.imshow(f1, cmap='inferno')
                ax1.set_title('F1 Thermal (3.74um)\nActive fire detection', fontsize=11, fontweight='bold')
                if fire_pixel and label == 1:
                    ax1.add_patch(Circle((fire_pixel[1], fire_pixel[0]), radius=8, color='cyan', fill=False, linewidth=2))
            ax1.axis('off')
            
            # Panel 2: F2 thermal (10.85um)
            ax2 = fig.add_subplot(gs[0, 1])
            if tile.shape[0] >= 4:
                f2 = self.normalize_for_display(tile[3], percentile=99)
                ax2.imshow(f2, cmap='inferno')
                ax2.set_title('F2 Thermal (10.85um)\nBackground temperature', fontsize=11, fontweight='bold')
                if fire_pixel and label == 1:
                    ax2.add_patch(Circle((fire_pixel[1], fire_pixel[0]), radius=8, color='cyan', fill=False, linewidth=2))
            ax2.axis('off')
            
            # Panel 3: F1-F2 difference (fire anomaly)
            ax3 = fig.add_subplot(gs[0, 2])
            if tile.shape[0] >= 4:
                f1_raw = tile[2]
                f2_raw = tile[3]
                diff = f1_raw - f2_raw
                diff_norm = self.normalize_for_display(diff, percentile=98)
                ax3.imshow(diff_norm, cmap='RdYlBu_r')
                ax3.set_title('F1-F2 Difference\nFire anomaly = red', fontsize=11, fontweight='bold')
                if fire_pixel and label == 1:
                    ax3.add_patch(Circle((fire_pixel[1], fire_pixel[0]), radius=8, color='black', fill=False, linewidth=2))
            ax3.axis('off')
            
            # Panel 4: Thermal composite (F1=R, F2=G, F1=B)
            ax4 = fig.add_subplot(gs[0, 3])
            if tile.shape[0] >= 4:
                f1_norm = self.normalize_for_display(tile[2], percentile=99)
                f2_norm = self.normalize_for_display(tile[3], percentile=99)
                composite = np.stack([f1_norm, f2_norm, f1_norm * 0.5], axis=-1)
                ax4.imshow(composite)
                ax4.set_title('Thermal Composite\nFires = pink/white', fontsize=11, fontweight='bold')
                if fire_pixel and label == 1:
                    ax4.add_patch(Circle((fire_pixel[1], fire_pixel[0]), radius=8, color='cyan', fill=False, linewidth=2))
            ax4.axis('off')
            
        else:
            # SENTINEL-2: Show all 4 optical bands
            band_names = self.s2_band_names
            for i in range(4):
                ax = fig.add_subplot(gs[0, i])
                if i < tile.shape[0]:
                    band = tile[i]
                    band_norm = self.normalize_for_display(band, percentile=98)
                    cmap = 'gray'
                    ax.imshow(band_norm, cmap=cmap)
                    ax.set_title(f'Band {i+1}: {band_names[i]}', fontsize=11, fontweight='bold')
                    
                    if fire_pixel and label == 1:
                        ax.add_patch(Circle((fire_pixel[1], fire_pixel[0]), radius=8, color='red', fill=False, linewidth=2))
                else:
                    ax.text(0.5, 0.5, 'N/A', ha='center', va='center', fontsize=14, transform=ax.transAxes)
                ax.axis('off')
        
        # ========================================
        # ROW 2: Composite Views
        # ========================================
        
        # Panel 2.1: False Color Composite
        ax_fc = fig.add_subplot(gs[1, 0:2])  # Span 2 columns
        try:
            false_color = self.create_false_color(tile)
            ax_fc.imshow(false_color)
            ax_fc.set_title('False Color Composite\n(SWIR2-NIR-Red: Fires=red/white)', 
                           fontsize=11, fontweight='bold')
            
            if fire_pixel and label == 1:
                circle = Circle((fire_pixel[1], fire_pixel[0]), radius=12,
                               color='lime', fill=False, linewidth=3)
                ax_fc.add_patch(circle)
                ax_fc.annotate('FIRE', (fire_pixel[1], fire_pixel[0] - 25),
                             color='lime', fontsize=12, fontweight='bold',
                             ha='center', va='bottom',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        except Exception as e:
            ax_fc.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center',
                      transform=ax_fc.transAxes)
        ax_fc.axis('off')
        
        # Panel 2.2: Thermal/SWIR Fire View
        ax_th = fig.add_subplot(gs[1, 2:4])  # Span 2 columns
        try:
            if is_s3:
                # For S3: Use F1-F2 difference to highlight fires
                f1 = tile[2] if tile.shape[0] > 2 else tile[0]
                f2 = tile[3] if tile.shape[0] > 3 else tile[1]
                thermal = f1  # F1 responds stronger to active fires
                title = 'F1 Thermal (3.74um)\nActive fires = bright hot spots'
            else:
                # For S2: Use SWIR2
                thermal = tile[3] if tile.shape[0] > 3 else tile[-1]
                title = 'SWIR2 Band (B12)\nFires = bright yellow/white'
            
            thermal_norm = self.normalize_for_display(thermal, percentile=99)
            ax_th.imshow(thermal_norm, cmap='inferno')
            ax_th.set_title(title, fontsize=11, fontweight='bold')
            
            if fire_pixel and label == 1:
                circle = Circle((fire_pixel[1], fire_pixel[0]), radius=12,
                               color='cyan', fill=False, linewidth=3)
                ax_th.add_patch(circle)
                ax_th.annotate('FIRE', (fire_pixel[1], fire_pixel[0] - 25),
                             color='cyan', fontsize=12, fontweight='bold',
                             ha='center', va='bottom',
                             bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))
        except Exception as e:
            ax_th.text(0.5, 0.5, f'Error: {str(e)[:30]}', ha='center', va='center',
                      transform=ax_th.transAxes)
        ax_th.axis('off')
        
        # ========================================
        # ROW 3: Location Map
        # ========================================
        ax_map = fig.add_subplot(gs[2, :])
        
        # Set up map bounds (Portugal + Spain border region)
        ax_map.set_xlim(self.IBERIA_BOUNDS['lon_min'], self.IBERIA_BOUNDS['lon_max'])
        ax_map.set_ylim(self.IBERIA_BOUNDS['lat_min'], self.IBERIA_BOUNDS['lat_max'])
        ax_map.set_aspect('equal')
        
        # Draw Portugal outline
        self.draw_portugal_outline(ax_map)
        
        # Add grid
        ax_map.grid(True, linestyle='--', alpha=0.5)
        ax_map.set_xlabel('Longitude', fontsize=10)
        ax_map.set_ylabel('Latitude', fontsize=10)
        
        # Mark tile location
        if tile_coords:
            lat, lon = tile_coords
            ax_map.plot(lon, lat, 'r*', markersize=20, markeredgecolor='black', 
                       markeredgewidth=1, label='Tile Location')
            ax_map.annotate(f'  Loc: {lat:.3f}N, {abs(lon):.3f}W',
                          (lon, lat), fontsize=10, fontweight='bold',
                          color='darkred',
                          xytext=(10, 10), textcoords='offset points')
            map_title = f'Location: {lat:.4f}N, {abs(lon):.4f}W'
        else:
            map_title = 'Location: Coordinates not available'
        
        # Add title with scene info
        scene_info = scene_id[:30] if scene_id else 'Unknown'
        ax_map.set_title(f'{map_title}\nScene: {scene_info}', 
                        fontsize=12, fontweight='bold')
        
        # ========================================
        # Overall Title
        # ========================================
        label_text = "FIRE DETECTION" if label == 1 else "NO FIRE"
        color = 'red' if label == 1 else 'green'
        
        fig.suptitle(f"{label_text} ({sat_type})", 
                    fontsize=18, fontweight='bold', color=color, y=0.98)
        
        # Save
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        
        logger.info(f"Saved visualization: {output_path.name}")
    
    def create_sample_grid(self, n_samples: int = 6):
        """
        Create sample visualizations.
        
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
        
        sampled_fire = random.sample(fire_tiles, n_fire) if fire_tiles else []
        sampled_no_fire = random.sample(no_fire_tiles, n_no_fire) if no_fire_tiles else []
        
        # Visualize each sample
        sample_info = []
        
        for i, tile_path in enumerate(sampled_fire):
            output_path = self.output_dir / f"fire_sample_{i+1}.png"
            self.visualize_tile(tile_path, label=1, output_path=output_path)
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
