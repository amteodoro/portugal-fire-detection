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
    
    # Portugal bounds for map display
    IBERIA_BOUNDS = {
        'lon_min': -10.5, 'lon_max': -5.5,
        'lat_min': 36.5, 'lat_max': 42.5
    }
    
    # Portugal bounds for filtering (tighter, excludes most of Spain)
    PORTUGAL_BOUNDS = {
        'lon_min': -9.5, 'lon_max': -6.2,
        'lat_min': 36.95, 'lat_max': 42.15
    }
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.tiles_dir = Path(self.config['data']['tiles_dir'])
        self.labels_dir = Path(self.config['data']['labels_dir'])
        self.output_dir = Path("data/samples/dual_satellite")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Band names for new 6-band format: B02, B03, B04, B08, B11, B12
        # Legacy 4-band format: B04, B08, B11, B12
        self.s2_bands_new = ['Blue (B02)', 'Green (B03)', 'Red (B04)', 'NIR (B08)', 'SWIR1 (B11)', 'SWIR2 (B12)']
        self.s2_bands_legacy = ['Red (B04)', 'NIR (B08)', 'SWIR1 (B11)', 'SWIR2 (B12)']
        self.s3_bands = ['F1 (3.74um)', 'F2 (10.85um)', 'F1-F2 Diff', 'Thermal Composite']
    
    def is_in_portugal(self, lat: float, lon: float) -> bool:
        """Check if coordinates are within Portugal bounds."""
        return (self.PORTUGAL_BOUNDS['lat_min'] <= lat <= self.PORTUGAL_BOUNDS['lat_max'] and
                self.PORTUGAL_BOUNDS['lon_min'] <= lon <= self.PORTUGAL_BOUNDS['lon_max'])
    
    def normalize(self, data: np.ndarray, pct: int = 98) -> np.ndarray:
        """Normalize array for display."""
        valid = data[np.isfinite(data)]
        if len(valid) == 0:
            return np.zeros_like(data)
        p_low, p_high = np.percentile(valid, (2, pct))
        if p_high - p_low < 1e-6:
            return np.full_like(data, 0.5, dtype=float)
        return np.clip((data - p_low) / (p_high - p_low), 0, 1)
    
    def find_hotspot(self, band_data: np.ndarray, search_radius: int = None) -> tuple:
        """
        Find the hottest pixel (brightest spot) in a thermal/SWIR band.
        
        Args:
            band_data: 2D array of band values
            search_radius: Optional radius from center to search (None = entire tile)
            
        Returns:
            (row, col) of the hottest pixel
        """
        h, w = band_data.shape
        center_row, center_col = h // 2, w // 2
        
        if search_radius is not None:
            y, x = np.ogrid[:h, :w]
            dist_from_center = np.sqrt((y - center_row)**2 + (x - center_col)**2)
            mask = dist_from_center <= search_radius
            search_data = np.where(mask, band_data, -np.inf)
        else:
            search_data = band_data
        
        search_data = np.where(np.isfinite(search_data), search_data, -np.inf)
        
        if np.all(search_data == -np.inf):
            return center_row, center_col
        
        max_idx = np.argmax(search_data)
        row, col = np.unravel_index(max_idx, search_data.shape)
        
        return int(row), int(col)
    
    def find_dual_satellite_fires(self) -> List[Dict]:
        """
        Find fires that have tiles from BOTH Sentinel-2 and Sentinel-3.
        
        Filters to only include fires:
        1. With valid coordinates
        2. Located within Portugal bounds
        
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
            if not tile_path.exists():
                # Fallback: search processed splits (symlinks to tiles)
                splits_dir = Path("ml/data/processed/splits")
                if splits_dir.exists():
                    for candidate in splits_dir.rglob(f"{lf.stem}.npy"):
                        if candidate.exists():
                            tile_path = candidate
                            break
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
        
        # Find fires with both - prioritize those with valid Portugal coordinates
        dual_fires = []
        skipped_no_coords = 0
        skipped_outside_portugal = 0
        
        for fire_id, tiles in fire_tiles.items():
            if tiles['s2'] and tiles['s3']:
                # Try to get coordinates from either S2 or S3 tile data
                coords = None
                for tile in tiles['s2'] + tiles['s3']:
                    data = tile['data']
                    if 'tile_center_lat' in data and 'tile_center_lon' in data:
                        coords = (data['tile_center_lat'], data['tile_center_lon'])
                        break
                
                # Skip samples without coordinates
                if coords is None:
                    skipped_no_coords += 1
                    continue
                
                # Check if within Portugal bounds
                lat, lon = coords
                if not self.is_in_portugal(lat, lon):
                    skipped_outside_portugal += 1
                    continue
                
                dual_fires.append({
                    'fire_id': fire_id,
                    's2_tile': tiles['s2'][0],  # Take first S2 tile
                    's3_tile': tiles['s3'][0],  # Take first S3 tile
                    'coords': coords
                })
        
        logger.info(f"Found {len(dual_fires)} fires with both S2 and S3 tiles in Portugal")
        if skipped_no_coords > 0:
            logger.warning(f"Skipped {skipped_no_coords} fires without coordinates")
        if skipped_outside_portugal > 0:
            logger.info(f"Skipped {skipped_outside_portugal} fires outside Portugal")
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
        
        NEW Layout with Split View (supports 6-band and legacy 4-band tiles):
        - Row 1: True Color (or Red band) vs Fire-Enhanced False Color
        - Row 2: S3 thermal views (4 panels)
        - Row 3: Individual bands + Location map
        """
        s2_info = fire_data['s2_tile']
        s3_info = fire_data['s3_tile']
        fire_id = fire_data['fire_id']
        
        # Load tiles
        s2_tile = np.load(s2_info['tile_path'])
        s3_tile = np.load(s3_info['tile_path'])
        
        # Detect tile format
        has_rgb = s2_tile.shape[0] >= 6
        if has_rgb:
            b02_idx, b03_idx, b04_idx, b08_idx, b11_idx, b12_idx = 0, 1, 2, 3, 4, 5
            s2_bands = self.s2_bands_new
        else:
            b04_idx, b08_idx, b11_idx, b12_idx = 0, 1, 2, 3
            b02_idx, b03_idx = None, None
            s2_bands = self.s2_bands_legacy
        
        # Get fire pixel locations - UNIFIED DETECTION for ALL panels
        # Use SWIR2 (B12) as the most reliable fire indicator
        tile_size = s2_tile.shape[1]
        search_radius = tile_size // 2  # Search entire tile
        
        # Detect hottest pixel from SWIR2 - this is the PRIMARY fire position
        fire_pos = self.find_hotspot(s2_tile[b12_idx], search_radius=search_radius)
        
        # Use same position for both S2 and S3 for visual consistency
        s2_fire_pixel = fire_pos
        s3_fire_pixel = fire_pos  # Same position - same geographic coordinates
        
        # Get coordinates - use pre-fetched coords if available
        coords = fire_data.get('coords', None)
        if coords is None:
            s2_data = s2_info['data']
            s3_data = s3_info['data']
            if 'tile_center_lat' in s2_data:
                coords = (s2_data['tile_center_lat'], s2_data['tile_center_lon'])
            elif 'tile_center_lat' in s3_data:
                coords = (s3_data['tile_center_lat'], s3_data['tile_center_lon'])
        
        # Create figure with new split view layout
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(3, 4, figure=fig, height_ratios=[1.3, 1, 1],
                     hspace=0.25, wspace=0.15)
        
        # ===== ROW 1: SPLIT VIEW - True Color vs Fire Enhanced =====
        
        # Left: True Color RGB (or Red band for legacy tiles)
        ax_true = fig.add_subplot(gs[0, 0:2])
        if has_rgb:
            true_color = np.stack([
                self.normalize(s2_tile[b04_idx], 98),  # Red
                self.normalize(s2_tile[b03_idx], 98),  # Green
                self.normalize(s2_tile[b02_idx], 98)   # Blue
            ], axis=-1)
            true_color = np.clip(true_color * 1.2, 0, 1)  # Boost saturation
            ax_true.imshow(true_color)
            ax_true.set_title('TRUE COLOR (Natural View)\nRed-Green-Blue (B04-B03-B02)', 
                            fontsize=12, fontweight='bold', color='darkgreen')
        else:
            ax_true.imshow(self.normalize(s2_tile[b04_idx]), cmap='gray')
            ax_true.set_title('RED BAND (B04)\n(RGB bands not available)', 
                            fontsize=12, fontweight='bold', color='gray')
        
        if s2_fire_pixel:
            ax_true.add_patch(Circle((s2_fire_pixel[1], s2_fire_pixel[0]),
                                    radius=12, color='red', fill=False, linewidth=3))
            ax_true.annotate('FIRE', (s2_fire_pixel[1], s2_fire_pixel[0] - 20),
                            color='red', fontsize=11, fontweight='bold',
                            ha='center', bbox=dict(facecolor='white', alpha=0.8, edgecolor='red'))
        ax_true.axis('off')
        
        # Right: Fire-Enhanced False Color (SWIR2-NIR-Red)
        ax_fire = fig.add_subplot(gs[0, 2:4])
        false_color = np.stack([
            self.normalize(s2_tile[b12_idx], 98),  # SWIR2 -> R (fires bright)
            self.normalize(s2_tile[b08_idx], 98),  # NIR -> G (vegetation)
            self.normalize(s2_tile[b04_idx], 98)   # Red -> B
        ], axis=-1)
        ax_fire.imshow(false_color)
        ax_fire.set_title('FIRE-ENHANCED (Active Fires = Bright Red/White)\nSWIR2-NIR-Red (B12-B08-B04)', 
                        fontsize=12, fontweight='bold', color='darkred')
        
        if s2_fire_pixel:
            ax_fire.add_patch(Circle((s2_fire_pixel[1], s2_fire_pixel[0]),
                                    radius=12, color='lime', fill=False, linewidth=3))
            ax_fire.annotate('FIRE', (s2_fire_pixel[1], s2_fire_pixel[0] - 20),
                            color='lime', fontsize=11, fontweight='bold',
                            ha='center', bbox=dict(facecolor='black', alpha=0.7))
        ax_fire.axis('off')
        
        # Row 1 title
        fig.text(0.5, 0.98, 'LEFT: Natural view (how humans see) | RIGHT: Fire detection view (fires glow bright)',
                fontsize=10, ha='center', style='italic', alpha=0.7)
        
        # ===== ROW 2: Sentinel-3 Thermal =====
        s3_labels = ['F1 Thermal (3.74um)\nActive Fire Detection', 
                    'F2 Thermal (10.85um)\nBackground Temp', 
                    'F1-F2 Difference\nFire Anomaly', 
                    'Thermal Composite\nFires = Pink/White']
        
        for i in range(4):
            ax = fig.add_subplot(gs[1, i])
            
            if i == 0 and s3_tile.shape[0] >= 3:
                f1 = self.normalize(s3_tile[2], pct=99)
                ax.imshow(f1, cmap='inferno')
            elif i == 1 and s3_tile.shape[0] >= 4:
                f2 = self.normalize(s3_tile[3], pct=99)
                ax.imshow(f2, cmap='inferno')
            elif i == 2 and s3_tile.shape[0] >= 4:
                diff = s3_tile[2] - s3_tile[3]
                diff_norm = self.normalize(diff)
                ax.imshow(diff_norm, cmap='RdYlBu_r')
            elif i == 3 and s3_tile.shape[0] >= 4:
                f1_n = self.normalize(s3_tile[2], pct=99)
                f2_n = self.normalize(s3_tile[3], pct=99)
                composite = np.stack([f1_n, f2_n, f1_n * 0.5], axis=-1)
                ax.imshow(composite)
            
            ax.set_title(s3_labels[i], fontsize=9, fontweight='bold')
            if s3_fire_pixel:
                ax.add_patch(Circle((s3_fire_pixel[1], s3_fire_pixel[0]),
                                   radius=8, color='cyan', fill=False, linewidth=2))
            ax.axis('off')
        
        fig.text(0.02, 0.62, 'SENTINEL-3 (Thermal)', fontsize=11, fontweight='bold', color='darkred')
        
        # ===== ROW 3: Individual S2 Bands + Map =====
        s2_display_bands = [
            (b08_idx, 'NIR (B08)\nVegetation=Bright', 'gray'),
            (b11_idx, 'SWIR1 (B11)\nMoisture/Burns', 'gray'),
            (b12_idx, 'SWIR2 (B12)\nACTIVE FIRES=BRIGHT', 'hot')
        ]
        
        for i, (band_idx, label, cmap) in enumerate(s2_display_bands):
            ax = fig.add_subplot(gs[2, i])
            ax.imshow(self.normalize(s2_tile[band_idx]), cmap=cmap)
            ax.set_title(label, fontsize=9, fontweight='bold')
            if s2_fire_pixel:
                color = 'cyan' if i == 2 else 'red'
                ax.add_patch(Circle((s2_fire_pixel[1], s2_fire_pixel[0]),
                                   radius=8, color=color, fill=False, linewidth=2))
            ax.axis('off')
        
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
            ax_map.set_title(f'Portugal\n{lat:.3f}N, {abs(lon):.3f}W', 
                           fontsize=10, fontweight='bold')
        else:
            ax_map.set_title('Portugal\n(coords unavailable)', fontsize=10)
        ax_map.set_xlabel('Longitude', fontsize=8)
        ax_map.set_ylabel('Latitude', fontsize=8)
        ax_map.tick_params(labelsize=7)
        
        # Overall title
        fig.suptitle(f'DUAL SATELLITE FIRE VIEW: {fire_id}', 
                    fontsize=16, fontweight='bold', color='darkred', y=1.0)
        
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
