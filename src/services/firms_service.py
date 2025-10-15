import csv
import io
from datetime import datetime, timezone
from src.utils.config import settings
from src.utils.http import fetch
from src.utils.caching import cache_get, cache_set, file_cache_load, file_cache_store
from src.utils.geo import parse_bbox

FIRMS_COUNTRY_BASE = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"
FIRMS_AREA_BASE = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"

PRODUCTS = {
    "VIIRS": "VIIRS_SNPP_NRT",
    "MODIS": "MODIS_NRT",
}

def _unify_confidence(product: str, value: str) -> float | None:
    # VIIRS: "low|nominal|high"; MODIS: "0-100"
    try:
        if product == "VIIRS":
            mapping = {"low": 20, "nominal": 60, "high": 90}
            return float(mapping.get(value.lower(), 50))
        return float(value)
    except Exception:
        return None

def _parse_csv(product: str, content: bytes) -> list[dict]:
    rows = []
    text = content.decode("utf-8")
    
    # Check if response is an error message
    if text.startswith("error") or "error" in text.lower()[:100]:
        print(f"FIRMS API Error: {text[:200]}")
        return []
    
    # Check if response is empty or just headers
    lines = text.strip().split("\n")
    if len(lines) <= 1:
        print(f"FIRMS API returned no data (only headers or empty)")
        return []
    
    reader = csv.DictReader(io.StringIO(text))
    for i, r in enumerate(reader):
        try:
            # Skip rows with missing critical data
            if not r.get("latitude") or not r.get("longitude"):
                continue
                
            lat = float(r.get("latitude"))
            lon = float(r.get("longitude"))
            date = r.get("acq_date")
            time_ = r.get("acq_time")
            dt = f"{date}T{time_[:2]}:{time_[2:]}:00Z" if date and time_ else datetime.now(timezone.utc).isoformat()
            conf = _unify_confidence(product, r.get("confidence", "") or r.get("confidence_text", ""))
            frp = float(r["frp"]) if r.get("frp") else None
            rows.append({
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    "id": r.get("track") or r.get("scan") or f"{product}_{dt}_{lat:.4f}_{lon:.4f}",
                    "source": product,
                    "acq_time": dt,
                    "confidence": conf,
                    "frp": frp,
                },
            })
        except (ValueError, TypeError, KeyError) as e:
            # Skip malformed rows
            print(f"Skipping malformed row {i}: {e}")
            continue
    
    return rows

async def fetch_country(product: str, days: int = 1) -> dict:
    """Deprecated Portugal-only fetch kept for compatibility."""
    product_key = PRODUCTS[product]
    api_key = settings.FIRMS_API_KEY
    if not api_key:
        return {"type": "FeatureCollection", "features": []}
    url = f"{FIRMS_COUNTRY_BASE}/{api_key}/{product_key}/PRT/{days}"
    content = await fetch(url)
    features = _parse_csv(product, content)
    return {"type": "FeatureCollection", "features": features}


async def fetch_bbox(product: str, days: int = 1) -> dict:
    """Fetch FIRMS data for the configured bbox (from .env)."""
    product_key = PRODUCTS[product]
    api_key = settings.FIRMS_API_KEY
    if not api_key:
        return {"type": "FeatureCollection", "features": []}

    # FIRMS area API expects: [west,south,east,north] = [lon_min,lat_min,lon_max,lat_max]
    bbox_str = settings.PORTUGAL_BBOX
    lon_min, lat_min, lon_max, lat_max = parse_bbox(bbox_str)
    area_spec = f"{lon_min},{lat_min},{lon_max},{lat_max}"

    # Cache keys include bbox to avoid stale results
    ck = f"firms_{product}_{bbox_str}_{days}"
    fk = ck
    cached = cache_get(ck)
    if cached:
        return cached
    file_cached = file_cache_load("/app/data/firms", fk, settings.FIRMS_TTL)
    if file_cached:
        cache_set(ck, file_cached, ttl=settings.FIRMS_TTL)
        return file_cached

    url = f"{FIRMS_AREA_BASE}/{api_key}/{product_key}/{area_spec}/{days}"
    print(f"Fetching FIRMS data from: {url[:100]}...")
    content = await fetch(url)
    print(f"FIRMS response size: {len(content)} bytes")
    print(f"FIRMS response first 500 chars: {content[:500]}")
    features = _parse_csv(product, content)
    print(f"Parsed {len(features)} fire features")
    fc = {"type": "FeatureCollection", "features": features}
    file_cache_store("/app/data/firms", fk, fc)
    cache_set(ck, fc, ttl=settings.FIRMS_TTL)
    return fc
