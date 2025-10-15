"""
Summarize local dataset and generate false color preview images.

Analyzes Sentinel-2 scenes, FIRMS data, and processed tiles.
Generates false color composites for visualization.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
import numpy as np
from PIL import Image
import yaml


def load_config(config_path: str = "ml/configs/config.yaml") -> Dict:
    """Load configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def analyze_sentinel2_scenes(raw_dir: Path) -> Dict:
    """
    Analyze Sentinel-2 scenes in raw directory.
    
    Returns:
        Dictionary with scene statistics
    """
    scenes = []
    
    if not raw_dir.exists():
        return {'total': 0, 'scenes': [], 'dates': []}
    
    # Find all scene directories
    for date_dir in raw_dir.iterdir():
        if not date_dir.is_dir():
            continue
        
        for scene_dir in date_dir.iterdir():
            if not scene_dir.is_dir():
                continue
            
            # Check for metadata
            metadata_path = scene_dir / "metadata.json"
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = {}
            
            # Count bands
            bands = list(scene_dir.glob("*.tif"))
            
            scene_info = {
                'id': scene_dir.name,
                'date': date_dir.name,
                'path': str(scene_dir),
                'bands': len(bands),
                'band_files': [b.name for b in bands],
                'metadata': metadata
            }
            scenes.append(scene_info)
    
    # Get unique dates
    dates = sorted(set(s['date'] for s in scenes))
    
    return {
        'total': len(scenes),
        'scenes': scenes,
        'dates': dates,
        'date_range': (dates[0], dates[-1]) if dates else (None, None)
    }


def analyze_firms_data(firms_dir: Path) -> Dict:
    """
    Analyze FIRMS fire detection data.
    
    Returns:
        Dictionary with fire statistics
    """
    if not firms_dir.exists():
        return {'total_files': 0, 'regions': []}
    
    fires_by_region = {}
    
    # Find all FIRMS GeoJSON files
    for geojson_file in firms_dir.rglob("*.geojson"):
        region = geojson_file.parent.name
        
        try:
            with open(geojson_file, 'r') as f:
                data = json.load(f)
            
            num_fires = len(data.get('features', []))
            
            if region not in fires_by_region:
                fires_by_region[region] = {
                    'files': [],
                    'total_fires': 0
                }
            
            fires_by_region[region]['files'].append({
                'file': geojson_file.name,
                'fires': num_fires
            })
            fires_by_region[region]['total_fires'] += num_fires
        except:
            pass
    
    return {
        'total_files': sum(len(r['files']) for r in fires_by_region.values()),
        'regions': [
            {
                'name': region,
                'total_fires': info['total_fires'],
                'files': len(info['files'])
            }
            for region, info in fires_by_region.items()
        ],
        'total_fires': sum(r['total_fires'] for r in fires_by_region.values())
    }


def analyze_processed_tiles(processed_dir: Path) -> Dict:
    """
    Analyze processed tiles for training.
    
    Returns:
        Dictionary with tile statistics
    """
    stats = {
        'fire_tiles': 0,
        'no_fire_tiles': 0,
        'splits': {
            'train': {'fire': 0, 'no_fire': 0},
            'val': {'fire': 0, 'no_fire': 0},
            'test': {'fire': 0, 'no_fire': 0}
        }
    }
    
    if not processed_dir.exists():
        return stats
    
    # Check for splits directory
    splits_dir = processed_dir / "splits"
    if splits_dir.exists():
        for split in ['train', 'val', 'test']:
            split_dir = splits_dir / split
            if split_dir.exists():
                fire_dir = split_dir / "fire"
                no_fire_dir = split_dir / "no_fire"
                
                if fire_dir.exists():
                    stats['splits'][split]['fire'] = len(list(fire_dir.glob("*.npy")))
                if no_fire_dir.exists():
                    stats['splits'][split]['no_fire'] = len(list(no_fire_dir.glob("*.npy")))
    
    # Total counts
    stats['fire_tiles'] = sum(s['fire'] for s in stats['splits'].values())
    stats['no_fire_tiles'] = sum(s['no_fire'] for s in stats['splits'].values())
    stats['total_tiles'] = stats['fire_tiles'] + stats['no_fire_tiles']
    
    return stats


def create_false_color_composite(
    scene_dir: Path,
    output_path: Path,
    size: tuple = (512, 512)
) -> Optional[Path]:
    """
    Create false color composite (SWIR2-NIR-Red) for fire detection.
    
    Args:
        scene_dir: Path to scene directory
        output_path: Output path for composite image
        size: Output size (width, height)
    
    Returns:
        Path to created image or None
    """
    try:
        import rasterio
        from rasterio.enums import Resampling
        
        # Find band files (B12, B08, B04)
        b12_path = scene_dir / "B12.tif"  # SWIR2
        b08_path = scene_dir / "B08.tif"  # NIR
        b04_path = scene_dir / "B04.tif"  # Red
        
        if not all([b12_path.exists(), b08_path.exists(), b04_path.exists()]):
            return None
        
        # Read bands
        with rasterio.open(b12_path) as src:
            b12 = src.read(1, out_shape=size, resampling=Resampling.bilinear)
        
        with rasterio.open(b08_path) as src:
            b08 = src.read(1, out_shape=size, resampling=Resampling.bilinear)
        
        with rasterio.open(b04_path) as src:
            b04 = src.read(1, out_shape=size, resampling=Resampling.bilinear)
        
        # Normalize to 0-255
        def normalize(band):
            # Clip to typical Sentinel-2 range
            band = np.clip(band, 0, 3000)
            # Stretch to 0-255
            band = ((band / 3000.0) * 255).astype(np.uint8)
            return band
        
        r = normalize(b12)  # SWIR2 -> Red channel (fires appear bright)
        g = normalize(b08)  # NIR -> Green channel
        b = normalize(b04)  # Red -> Blue channel
        
        # Stack and create image
        rgb = np.dstack([r, g, b])
        img = Image.fromarray(rgb, mode='RGB')
        
        # Save
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path, 'PNG')
        
        return output_path
    
    except Exception as e:
        print(f"Error creating false color composite: {e}")
        return None


def generate_sample_images(
    scenes: List[Dict],
    output_dir: Path,
    num_samples: int = 5
) -> List[Dict]:
    """
    Generate false color sample images from scenes.
    
    Args:
        scenes: List of scene info dicts
        output_dir: Output directory for images
        num_samples: Number of samples to generate
        
    Returns:
        List of generated image info
    """
    samples = []
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Select representative scenes
    selected_scenes = scenes[:num_samples] if len(scenes) >= num_samples else scenes
    
    for i, scene in enumerate(selected_scenes):
        scene_path = Path(scene['path'])
        output_path = output_dir / f"sample_{i}_{scene['date']}_{scene['id'][:20]}.png"
        
        result = create_false_color_composite(scene_path, output_path)
        
        if result:
            samples.append({
                'scene_id': scene['id'],
                'date': scene['date'],
                'image_path': str(output_path.relative_to(Path.cwd())),
                'image_url': f"/static/samples/{output_path.name}"
            })
    
    return samples


def generate_summary(config: Optional[Dict] = None) -> Dict:
    """
    Generate complete dataset summary.
    
    Returns:
        Comprehensive summary dictionary
    """
    if config is None:
        config = load_config()
    
    print("Generating dataset summary...")
    
    # Paths
    raw_dir = Path(config['data']['raw_dir'])
    firms_dir = Path(config['data']['firms_cache'])
    processed_dir = Path(config['data']['processed_dir'])
    samples_dir = Path("data/samples")
    
    # Analyze components
    print("  Analyzing Sentinel-2 scenes...")
    sentinel2_stats = analyze_sentinel2_scenes(raw_dir)
    
    print("  Analyzing FIRMS data...")
    firms_stats = analyze_firms_data(firms_dir)
    
    print("  Analyzing processed tiles...")
    tile_stats = analyze_processed_tiles(processed_dir)
    
    # Generate sample images
    print("  Generating false color samples...")
    sample_images = generate_sample_images(
        sentinel2_stats['scenes'],
        samples_dir,
        num_samples=5
    )
    
    # Compile summary
    summary = {
        'generated_at': datetime.now().isoformat(),
        'sentinel2': {
            'total_scenes': sentinel2_stats['total'],
            'date_range': {
                'start': sentinel2_stats['date_range'][0],
                'end': sentinel2_stats['date_range'][1]
            },
            'unique_dates': len(sentinel2_stats['dates']),
            'scenes_by_date': {
                date: len([s for s in sentinel2_stats['scenes'] if s['date'] == date])
                for date in sentinel2_stats['dates']
            }
        },
        'firms': {
            'total_fire_detections': firms_stats.get('total_fires', 0),
            'total_files': firms_stats['total_files'],
            'regions': firms_stats['regions']
        },
        'processed': {
            'total_tiles': tile_stats['total_tiles'],
            'fire_tiles': tile_stats['fire_tiles'],
            'no_fire_tiles': tile_stats['no_fire_tiles'],
            'fire_ratio': tile_stats['fire_tiles'] / tile_stats['total_tiles'] if tile_stats['total_tiles'] > 0 else 0,
            'splits': tile_stats['splits']
        },
        'samples': sample_images,
        'status': {
            'has_sentinel2_data': sentinel2_stats['total'] > 0,
            'has_firms_data': firms_stats.get('total_fires', 0) > 0,
            'has_processed_tiles': tile_stats['total_tiles'] > 0,
            'ready_for_training': tile_stats['total_tiles'] > 0 and all(
                tile_stats['splits'][s]['fire'] > 0 and tile_stats['splits'][s]['no_fire'] > 0
                for s in ['train', 'val', 'test']
            )
        }
    }
    
    # Save summary
    summary_path = Path("data/dataset_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\n✓ Summary saved to {summary_path}")
    
    return summary


def print_summary(summary: Dict):
    """Print summary to console."""
    print("\n" + "=" * 80)
    print("Dataset Summary")
    print("=" * 80)
    
    # Sentinel-2
    s2 = summary['sentinel2']
    print(f"\n📡 Sentinel-2 Scenes")
    print(f"  Total scenes: {s2['total_scenes']}")
    print(f"  Date range: {s2['date_range']['start']} to {s2['date_range']['end']}")
    print(f"  Unique dates: {s2['unique_dates']}")
    
    # FIRMS
    firms = summary['firms']
    print(f"\n🔥 FIRMS Fire Detections")
    print(f"  Total fire detections: {firms['total_fire_detections']}")
    print(f"  Files: {firms['total_files']}")
    for region in firms['regions']:
        print(f"    - {region['name']}: {region['total_fires']} fires ({region['files']} files)")
    
    # Processed
    proc = summary['processed']
    print(f"\n🎯 Processed Training Data")
    print(f"  Total tiles: {proc['total_tiles']}")
    print(f"  Fire tiles: {proc['fire_tiles']} ({proc['fire_ratio']:.1%})")
    print(f"  No-fire tiles: {proc['no_fire_tiles']}")
    
    print(f"\n  Splits:")
    for split, counts in proc['splits'].items():
        total = counts['fire'] + counts['no_fire']
        print(f"    {split.capitalize()}: {total} tiles ({counts['fire']} fire, {counts['no_fire']} no-fire)")
    
    # Samples
    print(f"\n🖼️  Generated {len(summary['samples'])} false color sample images")
    
    # Status
    status = summary['status']
    print(f"\n✅ Status")
    print(f"  Sentinel-2 data: {'✓' if status['has_sentinel2_data'] else '✗'}")
    print(f"  FIRMS data: {'✓' if status['has_firms_data'] else '✗'}")
    print(f"  Processed tiles: {'✓' if status['has_processed_tiles'] else '✗'}")
    print(f"  Ready for training: {'✓' if status['ready_for_training'] else '✗'}")
    
    print("\n" + "=" * 80)


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Summarize dataset and generate previews")
    parser.add_argument('--config', default='ml/configs/config.yaml',
                       help='Path to config file')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of sample images to generate')
    
    args = parser.parse_args()
    
    # Generate summary
    summary = generate_summary()
    
    # Print to console
    print_summary(summary)


if __name__ == "__main__":
    main()
