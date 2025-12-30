"""
Temporal Sequence Visualization.

Shows the same geographic location captured by Sentinel-2 and Sentinel-3
at different times, in chronological order.

For each location:
- Shows the earlier satellite acquisition (with timestamp)
- Shows the later satellite acquisition (with timestamp)
- Marks fire location ONLY if FIRMS detected fire at that acquisition time
- Includes both fire and no-fire examples
"""

import json
import logging
from pathlib import Path
from typing import Optional, Dict, Tuple, List
from datetime import datetime, timedelta
import yaml
import random

import numpy as np
import pandas as pd
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


class TemporalSequenceVisualizer:
    """Visualize temporal sequences of S2 and S3 data for the same location."""
    
    IBERIA_BOUNDS = {
        'lon_min': -10.5, 'lon_max': -5.5,
        'lat_min': 36.5, 'lat_max': 42.5
    }
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tiles_dir = Path(self.config['data']['tiles_dir'])
        self.labels_dir = Path(self.config['data']['labels_dir'])
        self.output_dir = Path("data/samples/temporal_sequence")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load matched dataset
        self.matched_df = pd.read_csv("ml/data/processed/matched_dataset.csv")
        
    def normalize(self, data: np.ndarray, pct: int = 98) -> np.ndarray:
        """Normalize for display."""
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return np.zeros_like(data)
        p_low, p_high = np.percentile(valid, (2, pct))
        if p_high - p_low < 1e-6:
            return np.full_like(data, 0.5, dtype=float)
        return np.clip((data - p_low) / (p_high - p_low), 0, 1)
    
    def find_temporal_pairs(self) -> List[Dict]:
        """
        Find locations with both S2 and S3 data, ordered chronologically.
        
        Returns list of dicts with:
        - location (lat, lon)
        - s2_tile, s3_tile info
        - s2_datetime, s3_datetime
        - has_fire (bool)
        - fire dates (if any)
        """
        # Group matched data by approximate location (round to 0.1 degree = ~10km)
        self.matched_df['loc_key'] = (
            self.matched_df['fire_lat'].round(1).astype(str) + '_' + 
            self.matched_df['fire_lon'].round(1).astype(str)
        )
        
        temporal_pairs = []
        
        # Get all label files and organize by fire_id
        label_files = list(self.labels_dir.glob("*.json"))
        fire_tiles = {}  # fire_id -> {s2: [...], s3: [...]}
        
        for lf in label_files:
            with open(lf) as f:
                data = json.load(f)
            
            fire_id = data.get('fire_id', 'none')
            scene_id = data.get('scene_id', '')
            
            # Get coordinates
            lat = data.get('tile_center_lat')
            lon = data.get('tile_center_lon')
            
            if lat is None or lon is None:
                # Try to parse from fire_id
                if fire_id and fire_id != 'none':
                    parts = fire_id.split('_')
                    if len(parts) >= 4:
                        try:
                            lat = float(parts[-2])
                            lon = float(parts[-1])
                        except:
                            continue
            
            if lat is None:
                continue
            
            loc_key = f"{round(lat, 1)}_{round(lon, 1)}"
            
            if loc_key not in fire_tiles:
                fire_tiles[loc_key] = {
                    's2': [], 's3': [],
                    'lat': lat, 'lon': lon,
                    'fire_ids': set()
                }
            
            # Get scene date from matched_df
            scene_row = self.matched_df[self.matched_df['scene_id'] == scene_id]
            scene_date = None
            if not scene_row.empty:
                scene_date = scene_row.iloc[0].get('scene_date')
            
            tile_path = self.tiles_dir / f"{lf.stem}.npy"
            if not tile_path.exists():
                continue
            
            is_s2 = scene_id.startswith('S2')
            entry = {
                'tile_path': tile_path,
                'label_path': lf,
                'scene_id': scene_id,
                'scene_date': scene_date,
                'label': data.get('label', 0),
                'fire_id': fire_id,
                'data': data
            }
            
            if is_s2:
                fire_tiles[loc_key]['s2'].append(entry)
            else:
                fire_tiles[loc_key]['s3'].append(entry)
            
            if fire_id != 'none':
                fire_tiles[loc_key]['fire_ids'].add(fire_id)
        
        # Build temporal pairs
        for loc_key, tiles in fire_tiles.items():
            if tiles['s2'] and tiles['s3']:
                # Pick one S2 and one S3 tile
                s2 = tiles['s2'][0]
                s3 = tiles['s3'][0]
                
                # Determine chronological order
                s2_date = s2.get('scene_date')
                s3_date = s3.get('scene_date')
                
                if s2_date and s3_date:
                    try:
                        s2_dt = pd.to_datetime(s2_date)
                        s3_dt = pd.to_datetime(s3_date)
                        
                        if s2_dt <= s3_dt:
                            first, second = s2, s3
                            first_type, second_type = 'S2', 'S3'
                        else:
                            first, second = s3, s2
                            first_type, second_type = 'S3', 'S2'
                    except:
                        first, second = s2, s3
                        first_type, second_type = 'S2', 'S3'
                else:
                    first, second = s2, s3
                    first_type, second_type = 'S2', 'S3'
                
                has_fire = bool(tiles['fire_ids'])
                
                temporal_pairs.append({
                    'loc_key': loc_key,
                    'lat': tiles['lat'],
                    'lon': tiles['lon'],
                    'first': first,
                    'first_type': first_type,
                    'second': second,
                    'second_type': second_type,
                    'has_fire': has_fire,
                    'fire_ids': list(tiles['fire_ids'])
                })
        
        # Separate fire and no-fire
        fire_pairs = [p for p in temporal_pairs if p['has_fire']]
        no_fire_pairs = [p for p in temporal_pairs if not p['has_fire']]
        
        logger.info(f"Found {len(fire_pairs)} fire locations, {len(no_fire_pairs)} no-fire locations with both satellites")
        
        return temporal_pairs
    
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
    
    def create_s2_composite(self, tile: np.ndarray) -> np.ndarray:
        """Create false color composite from S2 tile."""
        if tile.shape[0] >= 4:
            return np.stack([
                self.normalize(tile[3]),  # SWIR2 -> R
                self.normalize(tile[1]),  # NIR -> G
                self.normalize(tile[0])   # Red -> B
            ], axis=-1)
        return np.zeros((tile.shape[1], tile.shape[2], 3))
    
    def create_s3_thermal(self, tile: np.ndarray) -> np.ndarray:
        """Create thermal composite from S3 tile."""
        if tile.shape[0] >= 4:
            f1 = self.normalize(tile[2], pct=99)
            f2 = self.normalize(tile[3], pct=99)
            return np.stack([f1, f2, f1 * 0.5], axis=-1)
        elif tile.shape[0] >= 3:
            f1 = self.normalize(tile[2], pct=99)
            return np.stack([f1, f1, f1], axis=-1)
        return np.zeros((tile.shape[1], tile.shape[2], 3))
    
    def visualize_pair(self, pair: Dict, output_path: Path):
        """
        Create temporal sequence visualization.
        
        Layout:
        - Top: Title with location and fire status
        - Row 1: Earlier acquisition (4 panels: bands or composites)
        - Row 2: Later acquisition (4 panels: bands or composites)
        - Bottom: Location map
        """
        first = pair['first']
        second = pair['second']
        first_type = pair['first_type']
        second_type = pair['second_type']
        
        # Load tiles
        tile1 = np.load(first['tile_path'])
        tile2 = np.load(second['tile_path'])
        
        # Get fire pixel locations
        fire_pixel1 = None
        fire_pixel2 = None
        
        if first['data'].get('fire_pixel_row') is not None:
            fire_pixel1 = (first['data']['fire_pixel_row'], first['data']['fire_pixel_col'])
        if second['data'].get('fire_pixel_row') is not None:
            fire_pixel2 = (second['data']['fire_pixel_row'], second['data']['fire_pixel_col'])
        
        # Create figure
        fig = plt.figure(figsize=(18, 14))
        gs = GridSpec(3, 5, figure=fig, height_ratios=[1, 1, 0.6],
                     width_ratios=[1, 1, 1, 1, 0.8], hspace=0.3, wspace=0.15)
        
        # Format dates
        date1 = first.get('scene_date', 'Unknown date')
        date2 = second.get('scene_date', 'Unknown date')
        
        # Determine labels based on whether this sample has fire
        label1 = first['label']
        label2 = second['label']
        
        # ===== ROW 1: First (Earlier) Acquisition =====
        row1_title = f"{first_type} - {date1}"
        if label1 == 1:
            row1_title += " [FIRE DETECTED]"
        
        if first_type == 'S2':
            bands = ['Red (B04)', 'NIR (B08)', 'SWIR1 (B11)', 'SWIR2 (B12)']
            for i in range(4):
                ax = fig.add_subplot(gs[0, i])
                if i < tile1.shape[0]:
                    ax.imshow(self.normalize(tile1[i]), cmap='gray')
                    ax.set_title(f'S2: {bands[i]}', fontsize=9)
                    if fire_pixel1 and label1 == 1:
                        ax.add_patch(Circle((fire_pixel1[1], fire_pixel1[0]), 
                                           radius=8, color='red', fill=False, linewidth=2))
                ax.axis('off')
        else:  # S3
            panels = ['F1 (3.74um)', 'F2 (10.85um)', 'F1-F2 Diff', 'Thermal Comp']
            for i in range(4):
                ax = fig.add_subplot(gs[0, i])
                if i == 0 and tile1.shape[0] >= 3:
                    ax.imshow(self.normalize(tile1[2], 99), cmap='inferno')
                elif i == 1 and tile1.shape[0] >= 4:
                    ax.imshow(self.normalize(tile1[3], 99), cmap='inferno')
                elif i == 2 and tile1.shape[0] >= 4:
                    diff = tile1[2] - tile1[3]
                    ax.imshow(self.normalize(diff), cmap='RdYlBu_r')
                elif i == 3 and tile1.shape[0] >= 4:
                    ax.imshow(self.create_s3_thermal(tile1))
                ax.set_title(f'S3: {panels[i]}', fontsize=9)
                if fire_pixel1 and label1 == 1:
                    ax.add_patch(Circle((fire_pixel1[1], fire_pixel1[0]),
                                       radius=8, color='cyan', fill=False, linewidth=2))
                ax.axis('off')
        
        # Add row 1 label
        fig.text(0.02, 0.78, row1_title, fontsize=12, fontweight='bold',
                color='darkgreen' if label1 == 0 else 'darkred')
        
        # ===== ROW 2: Second (Later) Acquisition =====
        row2_title = f"{second_type} - {date2}"
        if label2 == 1:
            row2_title += " [FIRE DETECTED]"
        
        if second_type == 'S2':
            bands = ['Red (B04)', 'NIR (B08)', 'SWIR1 (B11)', 'SWIR2 (B12)']
            for i in range(4):
                ax = fig.add_subplot(gs[1, i])
                if i < tile2.shape[0]:
                    ax.imshow(self.normalize(tile2[i]), cmap='gray')
                    ax.set_title(f'S2: {bands[i]}', fontsize=9)
                    if fire_pixel2 and label2 == 1:
                        ax.add_patch(Circle((fire_pixel2[1], fire_pixel2[0]),
                                           radius=8, color='red', fill=False, linewidth=2))
                ax.axis('off')
        else:  # S3
            panels = ['F1 (3.74um)', 'F2 (10.85um)', 'F1-F2 Diff', 'Thermal Comp']
            for i in range(4):
                ax = fig.add_subplot(gs[1, i])
                if i == 0 and tile2.shape[0] >= 3:
                    ax.imshow(self.normalize(tile2[2], 99), cmap='inferno')
                elif i == 1 and tile2.shape[0] >= 4:
                    ax.imshow(self.normalize(tile2[3], 99), cmap='inferno')
                elif i == 2 and tile2.shape[0] >= 4:
                    diff = tile2[2] - tile2[3]
                    ax.imshow(self.normalize(diff), cmap='RdYlBu_r')
                elif i == 3 and tile2.shape[0] >= 4:
                    ax.imshow(self.create_s3_thermal(tile2))
                ax.set_title(f'S3: {panels[i]}', fontsize=9)
                if fire_pixel2 and label2 == 1:
                    ax.add_patch(Circle((fire_pixel2[1], fire_pixel2[0]),
                                       radius=8, color='cyan', fill=False, linewidth=2))
                ax.axis('off')
        
        # Add row 2 label
        fig.text(0.02, 0.44, row2_title, fontsize=12, fontweight='bold',
                color='darkgreen' if label2 == 0 else 'darkred')
        
        # ===== ROW 3: Composites + Map =====
        
        # Composite 1 (earlier)
        ax_comp1 = fig.add_subplot(gs[2, 0:2])
        if first_type == 'S2':
            ax_comp1.imshow(self.create_s2_composite(tile1))
            ax_comp1.set_title(f'{first_type} False Color\n{date1}', fontsize=10, fontweight='bold')
        else:
            ax_comp1.imshow(self.create_s3_thermal(tile1))
            ax_comp1.set_title(f'{first_type} Thermal\n{date1}', fontsize=10, fontweight='bold')
        if fire_pixel1 and label1 == 1:
            ax_comp1.add_patch(Circle((fire_pixel1[1], fire_pixel1[0]),
                                     radius=12, color='lime', fill=False, linewidth=3))
        ax_comp1.axis('off')
        
        # Composite 2 (later)
        ax_comp2 = fig.add_subplot(gs[2, 2:4])
        if second_type == 'S2':
            ax_comp2.imshow(self.create_s2_composite(tile2))
            ax_comp2.set_title(f'{second_type} False Color\n{date2}', fontsize=10, fontweight='bold')
        else:
            ax_comp2.imshow(self.create_s3_thermal(tile2))
            ax_comp2.set_title(f'{second_type} Thermal\n{date2}', fontsize=10, fontweight='bold')
        if fire_pixel2 and label2 == 1:
            ax_comp2.add_patch(Circle((fire_pixel2[1], fire_pixel2[0]),
                                     radius=12, color='lime', fill=False, linewidth=3))
        ax_comp2.axis('off')
        
        # Location map
        ax_map = fig.add_subplot(gs[2, 4])
        ax_map.set_xlim(self.IBERIA_BOUNDS['lon_min'], self.IBERIA_BOUNDS['lon_max'])
        ax_map.set_ylim(self.IBERIA_BOUNDS['lat_min'], self.IBERIA_BOUNDS['lat_max'])
        ax_map.set_aspect('equal')
        self.draw_portugal(ax_map)
        ax_map.grid(True, linestyle='--', alpha=0.5)
        
        lat, lon = pair['lat'], pair['lon']
        marker_color = 'red' if pair['has_fire'] else 'green'
        ax_map.plot(lon, lat, '*', markersize=15, color=marker_color, markeredgecolor='black')
        ax_map.set_title(f'{lat:.2f}N, {abs(lon):.2f}W', fontsize=9, fontweight='bold')
        ax_map.tick_params(labelsize=7)
        
        # Overall title
        status = "FIRE AREA" if pair['has_fire'] else "NO FIRE"
        status_color = 'red' if pair['has_fire'] else 'green'
        fig.suptitle(f'TEMPORAL SEQUENCE: {status}', fontsize=16, fontweight='bold', 
                    color=status_color, y=0.98)
        
        plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='white')
        plt.close()
        logger.info(f"Saved: {output_path.name}")
    
    def create_samples(self, n_samples: int = 8):
        """Generate temporal sequence visualizations (balanced fire/no-fire)."""
        logger.info(f"Creating {n_samples} temporal sequence visualizations...")
        
        pairs = self.find_temporal_pairs()
        
        if not pairs:
            logger.error("No temporal pairs found!")
            return
        
        # Split into fire and no-fire
        fire_pairs = [p for p in pairs if p['has_fire']]
        no_fire_pairs = [p for p in pairs if not p['has_fire']]
        
        # Balance samples
        n_fire = min(n_samples // 2, len(fire_pairs))
        n_no_fire = min(n_samples - n_fire, len(no_fire_pairs))
        
        # If not enough no-fire, take more fire
        if n_no_fire < n_samples // 2 and len(fire_pairs) > n_fire:
            n_fire = min(n_samples - n_no_fire, len(fire_pairs))
        
        selected = []
        if fire_pairs:
            selected.extend(random.sample(fire_pairs, n_fire))
        if no_fire_pairs:
            selected.extend(random.sample(no_fire_pairs, n_no_fire))
        
        random.shuffle(selected)
        
        for i, pair in enumerate(selected):
            output_path = self.output_dir / f"temporal_{i+1}.png"
            try:
                self.visualize_pair(pair, output_path)
            except Exception as e:
                logger.error(f"Error visualizing {pair['loc_key']}: {e}")
        
        # Save info
        info = [{'filename': f"temporal_{i+1}.png", 'has_fire': p['has_fire']} 
                for i, p in enumerate(selected)]
        with open(self.output_dir / "samples_info.json", 'w') as f:
            json.dump(info, f, indent=2)
        
        logger.info(f"\nGenerated {len(selected)} temporal visualizations:")
        logger.info(f"  Fire samples: {sum(1 for p in selected if p['has_fire'])}")
        logger.info(f"  No-fire samples: {sum(1 for p in selected if not p['has_fire'])}")
        logger.info(f"  Saved to: {self.output_dir}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Temporal sequence visualization")
    parser.add_argument('--config', default='ml/configs/config.yaml')
    parser.add_argument('--n-samples', type=int, default=8)
    args = parser.parse_args()
    
    viz = TemporalSequenceVisualizer(args.config)
    viz.create_samples(args.n_samples)


if __name__ == "__main__":
    main()
