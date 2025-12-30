"""
Visualize Co-Located Tile Pairs.

Creates side-by-side visualizations of S2 and S3 data for the SAME location.
Each panel clearly shows:
- S2 optical bands and composite
- S3 thermal bands and composite
- Acquisition timestamps
- Fire marker (if applicable)
- Location on Portugal map
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, List
import yaml
import random

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.gridspec import GridSpec

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ColocatedVisualizer:
    """Visualize co-located S2+S3 tile pairs."""
    
    PORTUGAL_BOUNDS = {
        'lon_min': -9.8, 'lon_max': -6.0,
        'lat_min': 36.8, 'lat_max': 42.2
    }
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.pairs_dir = Path("data/samples/colocated")
        self.output_dir = Path("data/samples/colocated_viz")
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def normalize(self, data: np.ndarray, pct: int = 98) -> np.ndarray:
        """Normalize for display."""
        valid = data[np.isfinite(data) & (data > 0)]
        if len(valid) == 0:
            return np.zeros_like(data)
        p_low = np.percentile(valid, 2)
        p_high = np.percentile(valid, pct)
        if p_high - p_low < 1e-6:
            return np.full_like(data, 0.5, dtype=float)
        return np.clip((data - p_low) / (p_high - p_low), 0, 1)
    
    def draw_portugal(self, ax):
        """Draw Portugal outline."""
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
        
        # Add cities
        cities = [
            ('Lisbon', -9.14, 38.72),
            ('Porto', -8.61, 41.15),
            ('Coimbra', -8.43, 40.21),
            ('Faro', -7.93, 37.02),
        ]
        for name, lon, lat in cities:
            ax.plot(lon, lat, 'ko', markersize=4)
            ax.annotate(name, (lon, lat), fontsize=7, xytext=(3, 3), 
                       textcoords='offset points')
    
    def load_pair(self, pair_dir: Path) -> Optional[Dict]:
        """Load a co-located pair."""
        s2_path = pair_dir / "s2_tile.npy"
        s3_path = pair_dir / "s3_tile.npy"
        meta_path = pair_dir / "metadata.json"
        
        if not all(p.exists() for p in [s2_path, s3_path, meta_path]):
            return None
        
        try:
            s2_tile = np.load(s2_path)
            s3_tile = np.load(s3_path)
            with open(meta_path) as f:
                metadata = json.load(f)
            
            return {
                's2_tile': s2_tile,
                's3_tile': s3_tile,
                'metadata': metadata,
                'pair_name': pair_dir.name
            }
        except Exception as e:
            logger.warning(f"Failed to load {pair_dir}: {e}")
            return None
    
    def visualize_pair(self, pair: Dict, output_path: Path):
        """
        Create visualization for one co-located pair.
        
        Layout:
        - Title with fire status and location
        - Row 1: S2 bands (4 panels)
        - Row 2: S3 bands (4 panels)
        - Row 3: S2 composite, S3 thermal, Map
        """
        s2_tile = pair['s2_tile']
        s3_tile = pair['s3_tile']
        meta = pair['metadata']
        
        has_fire = meta.get('has_fire', False)
        lat = meta['lat']
        lon = meta['lon']
        
        # Get fire pixel locations
        s2_fire = None
        s3_fire = None
        if has_fire:
            s2_info = meta.get('s2', {})
            s3_info = meta.get('s3', {})
            if s2_info.get('fire_pixel_row') is not None:
                s2_fire = (s2_info['fire_pixel_row'], s2_info['fire_pixel_col'])
            if s3_info.get('fire_pixel_row') is not None:
                s3_fire = (s3_info['fire_pixel_row'], s3_info['fire_pixel_col'])
        
        # Get dates
        s2_date = meta.get('s2', {}).get('scene_date', 'Unknown')
        s3_date = meta.get('s3', {}).get('scene_date', 'Unknown')
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1, 1, 1.2],
                     hspace=0.25, wspace=0.15)
        
        # ===== ROW 1: S2 Bands =====
        s2_bands = ['Red (B04)', 'NIR (B08)', 'SWIR1 (B11)', 'SWIR2 (B12)']
        for i in range(4):
            ax = fig.add_subplot(gs[0, i])
            if i < s2_tile.shape[0]:
                ax.imshow(self.normalize(s2_tile[i]), cmap='gray')
                ax.set_title(f'S2: {s2_bands[i]}', fontsize=10, fontweight='bold')
                if s2_fire:
                    ax.add_patch(Circle((s2_fire[1], s2_fire[0]), 
                                       radius=10, color='red', fill=False, linewidth=2))
            ax.axis('off')
        
        # Row 1 label
        fig.text(0.02, 0.82, f'SENTINEL-2 (Optical) - {s2_date}', 
                fontsize=11, fontweight='bold', color='darkblue')
        
        # ===== ROW 2: S3 Bands =====
        s3_labels = ['F1 (3.74um)', 'F2 (10.85um)', 'F1-F2 Diff', 'Thermal Comp']
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            
            if i == 0 and s3_tile.shape[0] >= 3:
                ax.imshow(self.normalize(s3_tile[2], 99), cmap='inferno')
            elif i == 1 and s3_tile.shape[0] >= 4:
                ax.imshow(self.normalize(s3_tile[3], 99), cmap='inferno')
            elif i == 2 and s3_tile.shape[0] >= 4:
                diff = s3_tile[2] - s3_tile[3]
                ax.imshow(self.normalize(diff), cmap='RdYlBu_r')
            elif i == 3 and s3_tile.shape[0] >= 4:
                f1 = self.normalize(s3_tile[2], 99)
                f2 = self.normalize(s3_tile[3], 99)
                comp = np.stack([f1, f2, f1 * 0.5], axis=-1)
                ax.imshow(comp)
            
            ax.set_title(f'S3: {s3_labels[i]}', fontsize=10, fontweight='bold')
            if s3_fire:
                ax.add_patch(Circle((s3_fire[1], s3_fire[0]),
                                   radius=10, color='cyan', fill=False, linewidth=2))
            ax.axis('off')
        
        # Row 2 label
        fig.text(0.02, 0.52, f'SENTINEL-3 (Thermal) - {s3_date}', 
                fontsize=11, fontweight='bold', color='darkred')
        
        # ===== ROW 3: Composites + Map =====
        
        # S2 False Color Composite
        ax_s2 = fig.add_subplot(gs[2, 0:2])
        if s2_tile.shape[0] >= 4:
            false_color = np.stack([
                self.normalize(s2_tile[3]),  # SWIR2 -> R
                self.normalize(s2_tile[1]),  # NIR -> G
                self.normalize(s2_tile[0])   # Red -> B
            ], axis=-1)
            ax_s2.imshow(false_color)
        ax_s2.set_title(f'S2 False Color (SWIR2-NIR-Red)\n{s2_date}', 
                       fontsize=11, fontweight='bold')
        if s2_fire:
            ax_s2.add_patch(Circle((s2_fire[1], s2_fire[0]),
                                  radius=15, color='lime', fill=False, linewidth=3))
            ax_s2.annotate('FIRE', (s2_fire[1], s2_fire[0] - 25),
                          color='lime', fontsize=12, fontweight='bold',
                          ha='center', bbox=dict(facecolor='black', alpha=0.7))
        ax_s2.axis('off')
        
        # S3 Thermal Composite
        ax_s3 = fig.add_subplot(gs[2, 2])
        if s3_tile.shape[0] >= 3:
            f1 = self.normalize(s3_tile[2], 99)
            ax_s3.imshow(f1, cmap='inferno')
        ax_s3.set_title(f'S3 F1 Thermal\n{s3_date}', fontsize=11, fontweight='bold')
        if s3_fire:
            ax_s3.add_patch(Circle((s3_fire[1], s3_fire[0]),
                                  radius=15, color='cyan', fill=False, linewidth=3))
            ax_s3.annotate('FIRE', (s3_fire[1], s3_fire[0] - 25),
                          color='cyan', fontsize=12, fontweight='bold',
                          ha='center', bbox=dict(facecolor='black', alpha=0.7))
        ax_s3.axis('off')
        
        # Location Map
        ax_map = fig.add_subplot(gs[2, 3])
        ax_map.set_xlim(self.PORTUGAL_BOUNDS['lon_min'], self.PORTUGAL_BOUNDS['lon_max'])
        ax_map.set_ylim(self.PORTUGAL_BOUNDS['lat_min'], self.PORTUGAL_BOUNDS['lat_max'])
        ax_map.set_aspect('equal')
        self.draw_portugal(ax_map)
        ax_map.grid(True, linestyle='--', alpha=0.4)
        
        # Mark location
        marker_color = 'red' if has_fire else 'green'
        ax_map.plot(lon, lat, '*', markersize=20, color=marker_color, 
                   markeredgecolor='white', markeredgewidth=1)
        ax_map.set_title(f'Portugal\n{lat:.3f}N, {abs(lon):.3f}W', 
                        fontsize=10, fontweight='bold')
        ax_map.set_xlabel('Longitude', fontsize=8)
        ax_map.set_ylabel('Latitude', fontsize=8)
        ax_map.tick_params(labelsize=7)
        
        # Overall title
        status = "FIRE DETECTED" if has_fire else "NO FIRE"
        status_color = 'red' if has_fire else 'green'
        fire_id = meta.get('fire_id', 'N/A')
        if fire_id and has_fire:
            title = f'{status} - {fire_id}'
        else:
            title = f'{status} - Location: {lat:.3f}N, {abs(lon):.3f}W'
        
        fig.suptitle(title, fontsize=14, fontweight='bold', color=status_color, y=0.98)
        
        # Save
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: {output_path.name}")
    
    def create_samples(self, n_samples: int = 10):
        """Generate visualizations for co-located pairs."""
        logger.info(f"Creating {n_samples} co-located visualizations...")
        
        # Find all pairs
        pair_dirs = sorted([d for d in self.pairs_dir.iterdir() if d.is_dir()])
        
        if not pair_dirs:
            logger.error(f"No pair directories found in {self.pairs_dir}")
            return
        
        # Separate fire and no-fire
        fire_dirs = [d for d in pair_dirs if d.name.startswith('fire_')]
        nofire_dirs = [d for d in pair_dirs if d.name.startswith('no_fire_')]
        
        logger.info(f"Found {len(fire_dirs)} fire pairs, {len(nofire_dirs)} no-fire pairs")
        
        # Balance selection
        n_fire = min(n_samples * 2 // 3, len(fire_dirs))
        n_nofire = min(n_samples - n_fire, len(nofire_dirs))
        
        selected = []
        if fire_dirs:
            selected.extend(random.sample(fire_dirs, min(n_fire, len(fire_dirs))))
        if nofire_dirs:
            selected.extend(random.sample(nofire_dirs, min(n_nofire, len(nofire_dirs))))
        
        random.shuffle(selected)
        
        # Generate visualizations
        samples_info = []
        for i, pair_dir in enumerate(selected):
            pair = self.load_pair(pair_dir)
            if pair is None:
                continue
            
            output_path = self.output_dir / f"colocated_{i+1}.png"
            try:
                self.visualize_pair(pair, output_path)
                samples_info.append({
                    'filename': output_path.name,
                    'has_fire': pair['metadata'].get('has_fire', False),
                    'pair_name': pair['pair_name']
                })
            except Exception as e:
                logger.error(f"Error visualizing {pair_dir.name}: {e}")
        
        # Save info
        with open(self.output_dir / "samples_info.json", 'w') as f:
            json.dump(samples_info, f, indent=2)
        
        logger.info(f"\nGenerated {len(samples_info)} visualizations in {self.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Visualize co-located S2+S3 pairs")
    parser.add_argument('--config', default='ml/configs/config.yaml')
    parser.add_argument('--n-samples', type=int, default=10)
    args = parser.parse_args()
    
    viz = ColocatedVisualizer(args.config)
    viz.create_samples(args.n_samples)


if __name__ == "__main__":
    main()
