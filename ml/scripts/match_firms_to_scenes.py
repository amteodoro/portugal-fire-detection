"""
Match FIRMS fire detections with Sentinel-2 scenes.

This script reads FIRMS hotspot data and pairs it with available Sentinel-2 scenes
based on spatial and temporal proximity. The output is a dataset of fire events
with their corresponding satellite imagery.
"""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Tuple
import yaml

import pandas as pd
from shapely.geometry import Point, box
from shapely.ops import unary_union

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FIRMSSceneMatcher:
    """Match FIRMS fire detections with Sentinel-2 scenes."""
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize matcher with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.temporal_window = self.config['firms']['temporal_window_days']
        self.min_confidence = self.config['firms']['min_confidence']
        self.raw_scenes_dir = Path(self.config['data']['raw_dir'])
        self.output_dir = Path(self.config['data']['processed_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def load_firms_data(self, firms_dir: str = "data/firms") -> pd.DataFrame:
        """
        Load FIRMS fire detection data from cached JSON files.
        
        Args:
            firms_dir: Directory containing FIRMS cache files
            
        Returns:
            DataFrame with columns: lat, lon, acq_date, acq_time, confidence, frp, source
        """
        firms_path = Path(firms_dir)
        
        if not firms_path.exists():
            logger.error(f"FIRMS directory not found: {firms_path}")
            return pd.DataFrame()
        
        all_fires = []
        
        # Read all FIRMS cache files
        for cache_file in firms_path.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                if 'features' in data:
                    for feature in data['features']:
                        props = feature['properties']
                        coords = feature['geometry']['coordinates']
                        
                        fire = {
                            'lon': coords[0],
                            'lat': coords[1],
                            'acq_time': props.get('acq_time'),
                            'confidence': props.get('confidence', 0),
                            'frp': props.get('frp'),
                            'source': props.get('source', 'VIIRS'),
                            'id': props.get('id')
                        }
                        
                        # Parse acquisition date from ISO timestamp
                        if fire['acq_time']:
                            try:
                                dt = datetime.fromisoformat(fire['acq_time'].replace('Z', '+00:00'))
                                fire['acq_date'] = dt.date()
                            except:
                                continue
                        
                        all_fires.append(fire)
            
            except Exception as e:
                logger.warning(f"Error reading {cache_file}: {e}")
                continue
        
        if not all_fires:
            logger.warning("No FIRMS data found")
            return pd.DataFrame()
        
        df = pd.DataFrame(all_fires)
        
        # Filter by confidence
        df = df[df['confidence'] >= self.min_confidence]
        
        logger.info(f"Loaded {len(df)} FIRMS fire detections (confidence >= {self.min_confidence})")
        
        return df
    
    def load_available_scenes(self) -> pd.DataFrame:
        """
        Load metadata for all downloaded Sentinel-2 scenes.
        
        Returns:
            DataFrame with scene metadata including bbox and date
        """
        scenes = []
        
        # Scan raw data directory for scene metadata
        for metadata_file in self.raw_scenes_dir.rglob("metadata.json"):
            try:
                with open(metadata_file, 'r') as f:
                    scene_data = json.load(f)
                
                scene_id = scene_data['id']
                properties = scene_data['properties']
                
                # Extract bounding box
                bbox_geom = scene_data.get('bbox')
                if bbox_geom and len(bbox_geom) == 4:
                    bbox = box(bbox_geom[0], bbox_geom[1], bbox_geom[2], bbox_geom[3])
                elif 'geometry' in scene_data:
                    # Fallback: use geometry coordinates
                    coords = scene_data['geometry']['coordinates'][0]
                    lons = [c[0] for c in coords]
                    lats = [c[1] for c in coords]
                    bbox = box(min(lons), min(lats), max(lons), max(lats))
                else:
                    logger.warning(f"No bbox found for scene {scene_id}")
                    continue
                
                # Extract acquisition date
                datetime_str = properties.get('datetime')
                if datetime_str:
                    acq_date = datetime.fromisoformat(datetime_str.replace('Z', '+00:00')).date()
                else:
                    acq_date = None
                
                scene = {
                    'scene_id': scene_id,
                    'acq_date': acq_date,
                    'bbox': bbox,
                    'cloud_cover': properties.get('eo:cloud_cover'),
                    'path': metadata_file.parent
                }
                
                scenes.append(scene)
            
            except Exception as e:
                logger.warning(f"Error reading scene metadata {metadata_file}: {e}")
                continue
        
        if not scenes:
            logger.warning("No Sentinel-2 scenes found")
            return pd.DataFrame()
        
        df = pd.DataFrame(scenes)
        logger.info(f"Loaded {len(df)} Sentinel-2 scenes")
        
        return df
    
    def match_fire_to_scenes(
        self,
        fire_row: pd.Series,
        scenes_df: pd.DataFrame
    ) -> List[Dict]:
        """
        Find Sentinel-2 scenes that match a fire detection.
        
        Matching criteria:
        - Scene bbox contains fire location
        - Scene date within temporal window of fire date
        
        Args:
            fire_row: Single fire detection record
            scenes_df: DataFrame of available scenes
            
        Returns:
            List of matching scene dictionaries
        """
        fire_point = Point(fire_row['lon'], fire_row['lat'])
        fire_date = fire_row['acq_date']
        
        matches = []
        
        for _, scene in scenes_df.iterrows():
            # Check spatial match
            if not scene['bbox'].contains(fire_point):
                continue
            
            # Check temporal match
            date_diff = abs((scene['acq_date'] - fire_date).days)
            if date_diff > self.temporal_window:
                continue
            
            match = {
                'fire_id': fire_row.get('id'),
                'fire_lat': fire_row['lat'],
                'fire_lon': fire_row['lon'],
                'fire_date': fire_date.isoformat(),
                'fire_confidence': fire_row['confidence'],
                'fire_frp': fire_row['frp'],
                'fire_source': fire_row['source'],
                'scene_id': scene['scene_id'],
                'scene_date': scene['acq_date'].isoformat(),
                'scene_path': str(scene['path']),
                'cloud_cover': scene['cloud_cover'],
                'date_diff_days': date_diff
            }
            
            matches.append(match)
        
        return matches
    
    def create_matched_dataset(
        self,
        firms_df: pd.DataFrame,
        scenes_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create a dataset of fire-scene matches.
        
        Args:
            firms_df: DataFrame of FIRMS fire detections
            scenes_df: DataFrame of available Sentinel-2 scenes
            
        Returns:
            DataFrame with matched fire-scene pairs
        """
        all_matches = []
        
        logger.info("Matching FIRMS detections to Sentinel-2 scenes...")
        
        for idx, fire in firms_df.iterrows():
            matches = self.match_fire_to_scenes(fire, scenes_df)
            all_matches.extend(matches)
            
            if (idx + 1) % 100 == 0:
                logger.info(f"Processed {idx + 1}/{len(firms_df)} fires, found {len(all_matches)} matches")
        
        if not all_matches:
            logger.warning("No matches found between FIRMS and Sentinel-2 scenes")
            return pd.DataFrame()
        
        matched_df = pd.DataFrame(all_matches)
        
        logger.info(f"Total matches: {len(matched_df)}")
        logger.info(f"Unique fires with matches: {matched_df['fire_id'].nunique()}")
        logger.info(f"Unique scenes with fires: {matched_df['scene_id'].nunique()}")
        
        return matched_df
    
    def generate_statistics(self, matched_df: pd.DataFrame) -> Dict:
        """Generate statistics about the matched dataset."""
        stats = {
            'total_matches': len(matched_df),
            'unique_fires': matched_df['fire_id'].nunique(),
            'unique_scenes': matched_df['scene_id'].nunique(),
            'avg_cloud_cover': matched_df['cloud_cover'].mean(),
            'avg_date_diff_days': matched_df['date_diff_days'].mean(),
            'fire_sources': matched_df['fire_source'].value_counts().to_dict(),
            'date_range': {
                'start': matched_df['fire_date'].min(),
                'end': matched_df['fire_date'].max()
            }
        }
        
        return stats
    
    def run(self, output_name: str = "matched_dataset.csv") -> Path:
        """
        Main execution: match FIRMS to scenes and save dataset.
        
        Args:
            output_name: Output CSV filename
            
        Returns:
            Path to saved dataset CSV
        """
        # Load FIRMS data
        logger.info("Loading FIRMS fire detections...")
        firms_df = self.load_firms_data()
        
        if firms_df.empty:
            logger.error("No FIRMS data available. Please fetch fire data first.")
            return None
        
        # Load available Sentinel-2 scenes
        logger.info("Loading Sentinel-2 scene metadata...")
        scenes_df = self.load_available_scenes()
        
        if scenes_df.empty:
            logger.error("No Sentinel-2 scenes available. Please download scenes first.")
            return None
        
        # Match fires to scenes
        matched_df = self.create_matched_dataset(firms_df, scenes_df)
        
        if matched_df.empty:
            logger.error("No matches found")
            return None
        
        # Save matched dataset
        output_path = self.output_dir / output_name
        matched_df.to_csv(output_path, index=False)
        logger.info(f"Saved matched dataset to: {output_path}")
        
        # Generate and save statistics
        stats = self.generate_statistics(matched_df)
        stats_path = self.output_dir / "matching_statistics.json"
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        logger.info(f"Saved statistics to: {stats_path}")
        
        # Print summary
        logger.info("\n" + "="*50)
        logger.info("MATCHING SUMMARY")
        logger.info("="*50)
        for key, value in stats.items():
            if isinstance(value, dict):
                logger.info(f"{key}:")
                for k, v in value.items():
                    logger.info(f"  {k}: {v}")
            else:
                logger.info(f"{key}: {value}")
        
        return output_path


def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Match FIRMS fires to Sentinel-2 scenes")
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Path to config file')
    parser.add_argument('--output', default='matched_dataset.csv', help='Output CSV filename')
    
    args = parser.parse_args()
    
    matcher = FIRMSSceneMatcher(args.config)
    matcher.run(args.output)


if __name__ == "__main__":
    main()
