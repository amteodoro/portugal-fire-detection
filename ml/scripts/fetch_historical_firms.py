"""
Fetch historical FIRMS fire detection data from the archive.

The FIRMS NRT API only provides 10 days of recent data. For older fire data,
we need to use date-based requests which access the FIRMS archive.

This script fetches fire data for custom date ranges and saves it to the
FIRMS cache directory for use in ML training.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional
import yaml

import httpx

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# FIRMS API base URL
FIRMS_AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"


class HistoricalFIRMSFetcher:
    """Fetch historical FIRMS data using date-based API."""
    
    def __init__(self, config_path: str = "ml/configs/config.yaml"):
        """Initialize fetcher with configuration."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Get API key from environment
        import os
        from dotenv import load_dotenv
        
        # Load .env file
        load_dotenv()
        self.api_key = os.getenv("FIRMS_API_KEY")
        
        if not self.api_key:
            raise ValueError("FIRMS_API_KEY not found. Set it in .env file")
        
        self.cache_dir = Path("data/firms")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.client = httpx.AsyncClient(timeout=120.0)
    
    def _parse_csv_to_geojson(self, csv_text: str, product: str) -> dict:
        """
        Parse FIRMS CSV response to GeoJSON format.
        
        Args:
            csv_text: CSV response from FIRMS API
            product: 'VIIRS' or 'MODIS'
            
        Returns:
            GeoJSON FeatureCollection
        """
        import csv
        import io
        
        features = []
        
        # Check for errors
        if csv_text.startswith("error") or "error" in csv_text.lower()[:100]:
            logger.error(f"FIRMS API Error: {csv_text[:200]}")
            return {"type": "FeatureCollection", "features": []}
        
        # Parse CSV
        reader = csv.DictReader(io.StringIO(csv_text))
        
        for row in reader:
            try:
                lat = float(row.get("latitude"))
                lon = float(row.get("longitude"))
                date = row.get("acq_date")
                time = row.get("acq_time")
                
                # Format datetime
                dt = f"{date}T{time[:2]}:{time[2:]}:00Z" if date and time else None
                
                # Normalize confidence
                if product == "VIIRS":
                    conf_str = row.get("confidence", "nominal").lower()
                    conf_map = {"low": 20, "nominal": 60, "high": 90}
                    confidence = conf_map.get(conf_str, 50)
                else:
                    confidence = float(row.get("confidence", 50))
                
                # FRP (Fire Radiative Power)
                frp = float(row.get("frp")) if row.get("frp") else None
                
                feature = {
                    "type": "Feature",
                    "geometry": {
                        "type": "Point",
                        "coordinates": [lon, lat]
                    },
                    "properties": {
                        "id": f"{product}_{date}_{lat}_{lon}",
                        "source": product,
                        "acq_time": dt,
                        "confidence": confidence,
                        "frp": frp,
                        "bright_ti4": float(row.get("bright_ti4")) if row.get("bright_ti4") else None,
                        "scan": row.get("scan"),
                        "track": row.get("track")
                    }
                }
                
                features.append(feature)
                
            except (ValueError, TypeError, KeyError) as e:
                logger.debug(f"Skipping malformed row: {e}")
                continue
        
        return {"type": "FeatureCollection", "features": features}
    
    async def fetch_date_range(
        self,
        product: str,
        bbox: List[float],
        start_date: str,
        end_date: str
    ) -> dict:
        """
        Fetch FIRMS data for a date range.
        
        Args:
            product: 'VIIRS' or 'MODIS'
            bbox: [lon_min, lat_min, lon_max, lat_max]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            GeoJSON FeatureCollection
        """
        # Map product name to FIRMS product code
        product_codes = {
            "VIIRS": "VIIRS_SNPP_NRT",
            "MODIS": "MODIS_NRT"
        }
        
        product_code = product_codes.get(product.upper())
        if not product_code:
            raise ValueError(f"Invalid product: {product}")
        
        # Format bbox
        lon_min, lat_min, lon_max, lat_max = bbox
        area_spec = f"{lon_min},{lat_min},{lon_max},{lat_max}"
        
        # Format date range
        date_param = f"{start_date},{end_date}"
        
        # Construct URL
        url = f"{FIRMS_AREA_BASE}/{self.api_key}/{product_code}/{area_spec}/{date_param}"
        
        logger.info(f"Fetching {product} data: {start_date} to {end_date}")
        logger.info(f"URL: {url[:100]}...")
        
        try:
            response = await self.client.get(url)
            response.raise_for_status()
            
            csv_text = response.text
            logger.info(f"Received {len(csv_text)} bytes")
            
            # Parse to GeoJSON
            geojson = self._parse_csv_to_geojson(csv_text, product.upper())
            
            logger.info(f"Parsed {len(geojson['features'])} fire detections")
            
            return geojson
            
        except httpx.HTTPError as e:
            logger.error(f"HTTP error fetching FIRMS data: {e}")
            return {"type": "FeatureCollection", "features": []}
    
    async def fetch_multi_month(
        self,
        product: str,
        bbox: List[float],
        start_date: str,
        end_date: str,
        chunk_days: int = 10
    ) -> dict:
        """
        Fetch data for long date ranges by splitting into chunks.
        
        FIRMS API works best with ~10 day chunks. This method splits
        longer ranges into multiple requests and combines results.
        
        Args:
            product: 'VIIRS' or 'MODIS'
            bbox: [lon_min, lat_min, lon_max, lat_max]
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            chunk_days: Days per chunk (default 10)
            
        Returns:
            Combined GeoJSON FeatureCollection
        """
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")
        
        total_days = (end - start).days
        
        if total_days <= chunk_days:
            # Single request
            return await self.fetch_date_range(product, bbox, start_date, end_date)
        
        # Split into chunks
        logger.info(f"Splitting {total_days} days into ~{chunk_days}-day chunks")
        
        all_features = []
        current = start
        
        while current <= end:
            chunk_end = min(current + timedelta(days=chunk_days - 1), end)
            
            chunk_start_str = current.strftime("%Y-%m-%d")
            chunk_end_str = chunk_end.strftime("%Y-%m-%d")
            
            geojson = await self.fetch_date_range(product, bbox, chunk_start_str, chunk_end_str)
            all_features.extend(geojson.get("features", []))
            
            current = chunk_end + timedelta(days=1)
            
            # Rate limiting
            await asyncio.sleep(0.5)
        
        logger.info(f"Total features collected: {len(all_features)}")
        
        return {"type": "FeatureCollection", "features": all_features}
    
    def save_to_cache(self, geojson: dict, product: str, bbox: List[float], date_range: str):
        """
        Save fetched data to FIRMS cache directory.
        
        Args:
            geojson: GeoJSON FeatureCollection
            product: 'VIIRS' or 'MODIS'
            bbox: Bounding box
            date_range: Date range string for filename
        """
        bbox_str = ",".join(map(str, bbox))
        filename = f"firms_{product}_{bbox_str}_{date_range}.json"
        filepath = self.cache_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(geojson, f, indent=2)
        
        logger.info(f"Saved {len(geojson['features'])} features to {filepath}")
    
    async def run(
        self,
        products: List[str] = ["VIIRS", "MODIS"],
        bbox: Optional[List[float]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        region_name: Optional[str] = None
    ):
        """
        Main execution: fetch historical FIRMS data.
        
        Args:
            products: List of products to fetch
            bbox: Bounding box override
            start_date: Start date override
            end_date: End date override
            region_name: Region name for cache filename
        """
        # Use config defaults if not specified
        if bbox is None:
            bbox = self.config['region']['bbox']
        
        if start_date is None or end_date is None:
            # Default to fire season (June-September) of previous year
            current_year = datetime.now().year
            start_date = f"{current_year - 1}-06-01"
            end_date = f"{current_year - 1}-09-30"
        
        logger.info(f"Fetching historical FIRMS data")
        logger.info(f"Date range: {start_date} to {end_date}")
        logger.info(f"Bbox: {bbox}")
        logger.info(f"Products: {products}")
        
        for product in products:
            logger.info(f"\n{'='*50}")
            logger.info(f"Fetching {product}")
            logger.info(f"{'='*50}")
            
            try:
                geojson = await self.fetch_multi_month(
                    product, bbox, start_date, end_date
                )
                
                # Save to cache
                date_label = f"{start_date}_to_{end_date}" if region_name is None else region_name
                self.save_to_cache(geojson, product, bbox, date_label)
                
            except Exception as e:
                logger.error(f"Error fetching {product}: {e}")
                continue
        
        await self.client.aclose()
        
        logger.info("\n" + "="*50)
        logger.info("Historical FIRMS fetch complete")
        logger.info("="*50)


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Fetch historical FIRMS fire detection data"
    )
    parser.add_argument('--config', default='ml/configs/config.yaml', help='Config file path')
    parser.add_argument('--bbox', nargs=4, type=float, help='Bbox: lon_min lat_min lon_max lat_max')
    parser.add_argument('--start-date', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end-date', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--products', nargs='+', default=['VIIRS', 'MODIS'],
                       help='Products to fetch (VIIRS, MODIS)')
    parser.add_argument('--region', help='Region name for cache filename')
    
    args = parser.parse_args()
    
    fetcher = HistoricalFIRMSFetcher(args.config)
    
    await fetcher.run(
        products=args.products,
        bbox=args.bbox,
        start_date=args.start_date,
        end_date=args.end_date,
        region_name=args.region
    )


if __name__ == "__main__":
    asyncio.run(main())
