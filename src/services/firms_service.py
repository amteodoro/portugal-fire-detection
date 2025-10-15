import csv
import io
from datetime import datetime, timezone
from src.utils.config import settings
from src.utils.http import fetch
from src.utils.caching import cache_get, cache_set, file_cache_load, file_cache_store

FIRMS_BASE = "https://firms.modaps.eosdis.nasa.gov/api/country/csv"

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
    reader = csv.DictReader(io.StringIO(content.decode("utf-8")))
    for r in reader:
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
    return rows

async def fetch_country(product: str, days: int = 1) -> dict:
    product_key = PRODUCTS[product]
    api_key = settings.FIRMS_API_KEY
    if not api_key:
        return {"type": "FeatureCollection", "features": []}
    # Cache keys
    ck = f"firms_{product}_{days}"
    fk = ck
    cached = cache_get(ck)
    if cached:
        return cached
    file_cached = file_cache_load("/app/data/firms", fk, settings.FIRMS_TTL)
    if file_cached:
        cache_set(ck, file_cached, ttl=settings.FIRMS_TTL)
        return file_cached
    url = f"{FIRMS_BASE}/{api_key}/{product_key}/PRT/{days}"
    content = await fetch(url)
    features = _parse_csv(product, content)
    fc = {"type": "FeatureCollection", "features": features}
    file_cache_store("/app/data/firms", fk, fc)
    cache_set(ck, fc, ttl=settings.FIRMS_TTL)
    return fc
