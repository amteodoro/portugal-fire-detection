"""
Dual-Satellite Fire Visualization.

Creates side-by-side visualizations showing the SAME fire detection
from both Sentinel-2 (optical) and Sentinel-3 (thermal) perspectives.

This helps validate that:
1. FIRMS fire detections are correctly matched to both satellites
2. The fire location marker aligns with actual hotspots in both views
3. The thermal and optical signatures correlate
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
import yaml
import random

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class DualSatelliteVisualizer:
    """Visualize fires from both Sentinel-2 and Sentinel-3."""
    
    # Portugal bounds for map
    IBERIA_BOUNDS = {
        'lon_min': -10.5, 'lon_max': -5.5,
        'lat_min': 36.5, 'lat_max': 42.5
    }
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tiles_dir = Path(self.config['data']['tiles_dir'])
        self.labels_dir = Path(self.config['data']['labels_dir'])
        self.output_dir = Path("data/samples/dual_satellite")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Band names
        self.s2_bands = ['Red (B04)', 'NIR (B08)', 'SWIR1 (B11)', 'SWIR2 (B12)']
        self.s3_bands = ['F1 (3.74um)', 'F2 (10.85um)', 'F1-F2 Diff', 'Thermal Composite']
    
    def normalize(self, data: np.ndarray, pct: int = 98) -> np.ndarray:
        """Normalize array for display."""
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return np.zeros_like(data)
        p_low, p_high = np.percentile(valid, (2, pct))
        if p_high - p_low < 1e-6:
            return np.full_like(data, 0.5, dtype=float)
        return np.clip((data - p_low) / (p_high - p_low), 0, 1)
    
    def find_dual_satellite_fires(self) -> List[Dict]:
        """
        Find fires that have tiles from BOTH Sentinel-2 and Sentinel-3.
        
        Returns list of dicts with s2_tile_path, s3_tile_path, fire_id, coords
        """
        # Get all label files
        label_files = list(self.labels_dir.glob("*.json"))
        
        # Group by fire_id
        fire_tiles = {}  # fire_id -> {'s2': [...], 's3': [...]}
        
        for lf in label_files:
            with open(lf) as f:
                data = json.load(f)
            
            if data.get('label') != 1:  # Only positive samples
                continue
                
            fire_id = data.get('fire_id', '')
            scene_id = data.get('scene_id', '')
            
            if not fire_id or fire_id == 'none':
                continue
            
            if fire_id not in fire_tiles:
                fire_tiles[fire_id] = {'s2': [], 's3': []}
            
            # Determine satellite type
            is_s2 = scene_id.startswith('S2')
            
            tile_path = self.tiles_dir / f"{lf.stem}.npy"
            if tile_path.exists():
                entry = {
                    'tile_path': tile_path,
                    'label_path': lf,
                    'scene_id': scene_id,
                    'data': data
                }
                if is_s2:
                    fire_tiles[fire_id]['s2'].append(entry)
                else:
                    fire_tiles[fire_id]['s3'].append(entry)
        
        # Find fires with both
        dual_fires = []
        for fire_id, tiles in fire_tiles.items():
            if tiles['s2'] and tiles['s3']:
                dual_fires.append({
                    'fire_id': fire_id,
                    's2_tile': tiles['s2'][0],  # Take first S2 tile
                    's3_tile': tiles['s3'][0],  # Take first S3 tile
                })
        
        logger.info(f"Found {len(dual_fires)} fires with both S2 and S3 tiles")
        return dual_fires
    
    def draw_portugal(self, ax):
        """Draw simplified Portugal outline."""
        outline = [
            (-8.9, 41.9), (-8.1, 42.0), (-7.4, 41.9), (-6.9, 41.95),
            (-6.6, 41.7), (-6.5, 41.3), (-6.8, 40.9), (-6.9, 40.4),
            (-7.0, 39.9), (-7.3, 39.5), (-7.0, 39.0), (-7.4, 38.2),
            (-7.5, 37.5), (-7.9, 37.0), (-8.8, 36.95), (-8.9, 37.5),
            (-9.0, 38.0), (-9.5, 38.7), (-9.2, 39.3), (-9.0, 39.6),
            (-9.1, 40.0), (-8.8, 40.7), (-8.9, 41.2), (-8.9, 41.9)
        ]
        lons, lats = zip(*outline)
        ax.plot(lons, lats, 'b-', linewidth=2, alpha=0.7)
        ax.fill(lons, lats, color='lightblue', alpha=0.3)
    
    def visualize_dual(self, fire_data: Dict, output_path: Path):
        """
        Create dual-satellite visualization for one fire.
        
        Layout:
        - Row 1: S2 bands (4 panels)
        - Row 2: S3 thermal views (4 panels)
        - Row 3: S2 composite, S3 composite, Location map
        """
        s2_info = fire_data['s2_tile']
        s3_info = fire_data['s3_tile']
        fire_id = fire_data['fire_id']
        
        # Load tiles
        s2_tile = np.load(s2_info['tile_path'])
        s3_tile = np.load(s3_info['tile_path'])
        
        # Get fire pixel locations
        s2_data = s2_info['data']
        s3_data = s3_info['data']
        
        s2_fire_pixel = None
        if 'fire_pixel_row' in s2_data and 'fire_pixel_col' in s2_data:
            s2_fire_pixel = (s2_data['fire_pixel_row'], s2_data['fire_pixel_col'])
        
        s3_fire_pixel = None
        if 'fire_pixel_row' in s3_data and 'fire_pixel_col' in s3_data:
            s3_fire_pixel = (s3_data['fire_pixel_row'], s3_data['fire_pixel_col'])
        
        # Get coordinates
        coords = None
        if 'tile_center_lat' in s2_data:
            coords = (s2_data['tile_center_lat'], s2_data['tile_center_lon'])
        elif 'tile_center_lat' in s3_data:
            coords = (s3_data['tile_center_lat'], s3_data['tile_center_lon'])
        
        # Create figure
        fig = plt.figure(figsize=(18, 16))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.2],
                     hspace=0.3, wspace=0.15)
        
        # ===== ROW 1: Sentinel-2 Bands =====
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            if i < s2_tile.shape[0]:
                band = self.normalize(s2_tile[i])
                ax.imshow(band, cmap='gray')
                ax.set_title(f'S2: {self.s2_bands[i]}', fontsize=10, fontweight='bold')
                if s2_fire_pixel:
                    ax.add_patch(Circle((s2_fire_pixel[1], s2_fire_pixel[0]), 
                                       radius=8, color='red', fill=False, linewidth=2))
            ax.axis('off')
        
        # ===== ROW 2: Sentinel-3 Thermal =====
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            
            if i == 0 and s3_tile.shape[0] >= 3:
                # F1 thermal
                f1 = self.normalize(s3_tile[2], pct=99)
                ax.imshow(f1, cmap='inferno')
                ax.set_title('S3: F1 (3.74um)', fontsize=10, fontweight='bold')
            elif i == 1 and s3_tile.shape[0] >= 4:
                # F2 thermal
                f2 = self.normalize(s3_tile[3], pct=99)
                ax.imshow(f2, cmap='inferno')
                ax.set_title('S3: F2 (10.85um)', fontsize=10, fontweight='bold')
            elif i == 2 and s3_tile.shape[0] >= 4:
                # F1-F2 difference
                diff = s3_tile[2] - s3_tile[3]
                diff_norm = self.normalize(diff)
                ax.imshow(diff_norm, cmap='RdYlBu_r')
                ax.set_title('S3: F1-F2 (Fire Anomaly)', fontsize=10, fontweight='bold')
            elif i == 3 and s3_tile.shape[0] >= 4:
                # Composite
                f1_n = self.normalize(s3_tile[2], pct=99)
                f2_n = self.normalize(s3_tile[3], pct=99)
                composite = np.stack([f1_n, f2_n, f1_n * 0.5], axis=-1)
                ax.imshow(composite)
                ax.set_title('S3: Thermal Composite', fontsize=10, fontweight='bold')
            
            if s3_fire_pixel:
                ax.add_patch(Circle((s3_fire_pixel[1], s3_fire_pixel[0]),
                                   radius=8, color='cyan', fill=False, linewidth=2))
            ax.axis('off')
        
        # ===== ROW 3: Composites + Map =====
        
        # S2 False Color
        ax_s2 = fig.add_subplot(gs[2, 0:2])
        if s2_tile.shape[0] >= 4:
            false_color = np.stack([
                self.normalize(s2_tile[3]),  # SWIR2 -> R
                self.normalize(s2_tile[1]),  # NIR -> G
                self.normalize(s2_tile[0])   # Red -> B
            ], axis=-1)
            ax_s2.imshow(false_color)
            ax_s2.set_title('Sentinel-2 False Color\n(SWIR2-NIR-Red: Fires=red/white)', 
                           fontsize=11, fontweight='bold')
            if s2_fire_pixel:
                ax_s2.add_patch(Circle((s2_fire_pixel[1], s2_fire_pixel[0]),
                                      radius=12, color='lime', fill=False, linewidth=3))
                ax_s2.annotate('FIRE', (s2_fire_pixel[1], s2_fire_pixel[0] - 20),
                              color='lime', fontsize=10, fontweight='bold',
                              ha='center', bbox=dict(facecolor='black', alpha=0.7))
        ax_s2.axis('off')
        
        # S3 Thermal view
        ax_s3 = fig.add_subplot(gs[2, 2])
        if s3_tile.shape[0] >= 3:
            thermal = self.normalize(s3_tile[2], pct=99)
            ax_s3.imshow(thermal, cmap='inferno')
            ax_s3.set_title('Sentinel-3 F1 Thermal\n(Active fires=bright)', 
                           fontsize=11, fontweight='bold')
            if s3_fire_pixel:
                ax_s3.add_patch(Circle((s3_fire_pixel[1], s3_fire_pixel[0]),
                                      radius=12, color='cyan', fill=False, linewidth=3))
                ax_s3.annotate('FIRE', (s3_fire_pixel[1], s3_fire_pixel[0] - 20),
                              color='cyan', fontsize=10, fontweight='bold',
                              ha='center', bbox=dict(facecolor='black', alpha=0.7))
        ax_s3.axis('off')
        
        # Location Map
        ax_map = fig.add_subplot(gs[2, 3])
        ax_map.set_xlim(self.IBERIA_BOUNDS['lon_min'], self.IBERIA_BOUNDS['lon_max'])
        ax_map.set_ylim(self.IBERIA_BOUNDS['lat_min'], self.IBERIA_BOUNDS['lat_max'])
        ax_map.set_aspect('equal')
        self.draw_portugal(ax_map)
        ax_map.grid(True, linestyle='--', alpha=0.5)
        
        if coords:
            lat, lon = coords
            ax_map.plot(lon, lat, 'r*', markersize=15, markeredgecolor='black')
            ax_map.set_title(f'Location\n{lat:.3f}N, {abs(lon):.3f}W', 
                           fontsize=10, fontweight='bold')
        else:
            ax_map.set_title('Location\n(coords unavailable)', fontsize=10)
        ax_map.set_xlabel('Lon', fontsize=8)
        ax_map.set_ylabel('Lat', fontsize=8)
        
        # Overall title
        fig.suptitle(f'DUAL SATELLITE VIEW: {fire_id}', 
                    fontsize=14, fontweight='bold', color='darkred', y=0.98)
        
        # Save
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: {output_path.name}")
    
    def create_samples(self, n_samples: int = 6):
        """Generate sample dual-satellite visualizations."""
        logger.info(f"Creating {n_samples} dual-satellite visualizations...")
        
        dual_fires = self.find_dual_satellite_fires()
        
        if not dual_fires:
            logger.error("No fires found with both S2 and S3 tiles!")
            return
        
        # Sample
        n = min(n_samples, len(dual_fires))
        samples = random.sample(dual_fires, n)
        
        for i, fire_data in enumerate(samples):
            output_path = self.output_dir / f"dual_fire_{i+1}.png"
            try:
                self.visualize_dual(fire_data, output_path)
            except Exception as e:
                logger.error(f"Error visualizing {fire_data['fire_id']}: {e}")
        
        logger.info(f"\nGenerated {n} dual-satellite visualizations in {self.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Dual-satellite fire visualization")
    parser.add_argument('--config', default='ml/configs/config.yaml')
    parser.add_argument('--n-samples', type=int, default=6)
    args = parser.parse_args()
    
    viz = DualSatelliteVisualizer(args.config)
    viz.create_samples(args.n_samples)


if __name__ == "__main__":
    main()
