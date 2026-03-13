import csv
import io
import logging
from datetime import datetime, timezone
from src.utils.config import settings
from src.utils.http import fetch
from src.utils.caching import cache_get, cache_set, file_cache_load, file_cache_store
from src.utils.geo import parse_bbox

logger = logging.getLogger(__name__)

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
        logger.warning("FIRMS API returned an error response")
        return []

    # Check if response is empty or just headers
    lines = text.strip().split("\n")
    if len(lines) <= 1:
        logger.info("FIRMS API returned no data (only headers or empty)")
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
            dt = (
                f"{date}T{time_[:2]}:{time_[2:]}:00Z"
                if date and time_
                else datetime.now(timezone.utc).isoformat()
            )
            conf = _unify_confidence(
                product, r.get("confidence", "") or r.get("confidence_text", "")
            )
            frp = float(r["frp"]) if r.get("frp") else None
            rows.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "Point", "coordinates": [lon, lat]},
                    "properties": {
                        "id": r.get("track")
                        or r.get("scan")
                        or f"{product}_{dt}_{lat:.4f}_{lon:.4f}",
                        "source": product,
                        "acq_time": dt,
                        "confidence": conf,
                        "frp": frp,
                    },
                }
            )
        except (ValueError, TypeError, KeyError):
            # Skip malformed rows
            logger.debug("Skipping malformed row %d", i)
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


async def fetch_bbox(product: str, days: int = 1, bbox_override: str = None) -> dict:
    """Fetch FIRMS data for the configured bbox (from .env) or override bbox.

    Args:
        product: 'VIIRS' or 'MODIS'
        days: Number of days to fetch (1-10, limited by FIRMS NRT API)
        bbox_override: Optional bbox string 'lon_min,lat_min,lon_max,lat_max' to override .env config
    """
    product_key = PRODUCTS[product]
    api_key = settings.FIRMS_API_KEY
    if not api_key:
        return {"type": "FeatureCollection", "features": []}

    # Use override bbox if provided, otherwise use configured bbox from .env
    bbox_str = bbox_override if bbox_override else settings.PORTUGAL_BBOX

    # FIRMS area API expects: [west,south,east,north] = [lon_min,lat_min,lon_max,lat_max]
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

    # FIRMS NRT API supports 1-10 days only via the /area/{days} endpoint
    # For historical data beyond 10 days, the Archive API would be needed (different endpoints/auth)
    url = f"{FIRMS_AREA_BASE}/{api_key}/{product_key}/{area_spec}/{days}"
    logger.info("Fetching FIRMS data: product=%s, days=%d", product_key, days)
    content = await fetch(url)
    logger.debug("FIRMS response size: %d bytes", len(content))
    features = _parse_csv(product, content)
    logger.info("Parsed %d fire features", len(features))

    fc = {"type": "FeatureCollection", "features": features}
    file_cache_store("/app/data/firms", fk, fc)
    cache_set(ck, fc, ttl=settings.FIRMS_TTL)

    # Persist to database when configured (fire-and-forget, non-critical)
    from src.db.session import is_configured
    if is_configured():
        try:
            await _upsert_fire_events(features, bbox_str)
        except Exception as exc:
            logger.warning("DB upsert failed (non-critical): %s", exc)

    return fc


async def _upsert_fire_events(features: list[dict], region: str) -> None:
    """Upsert FIRMS fire events into the database, deduplicated by source+location+date."""
    from src.db.session import get_session
    from src.db.models import FireEvent
    from sqlalchemy import text

    async with get_session() as session:
        for f in features:
            props = f["properties"]
            lon, lat = f["geometry"]["coordinates"]
            detected_at_str = props.get("acq_time", "")
            try:
                detected_at = datetime.fromisoformat(
                    detected_at_str.replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                detected_at = datetime.now(timezone.utc)

            # Deduplicate: skip if same source + location (rounded) + date already exists
            date_only = detected_at.date().isoformat()
            lat_r = round(lat, 4)
            lon_r = round(lon, 4)
            exists = await session.execute(
                text(
                    "SELECT 1 FROM fire_events WHERE source = :src"
                    " AND detected_at::date = :dt"
                    " AND ST_DWithin(location::geography,"
                    "   ST_MakePoint(:lon, :lat)::geography, 500)"
                    " LIMIT 1"
                ),
                {"src": props["source"], "dt": date_only, "lat": lat_r, "lon": lon_r},
            )
            if exists.fetchone():
                continue

            conf = props.get("confidence")
            conf_str = None
            if conf is not None:
                conf_str = str(int(conf)) if conf > 1 else str(conf)

            session.add(
                FireEvent(
                    source=props["source"],
                    detected_at=detected_at,
                    location=f"SRID=4326;POINT({lon} {lat})",
                    frp=props.get("frp"),
                    confidence=conf_str,
                    region=region,
                )
            )
        await session.commit()
