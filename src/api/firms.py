import logging
from datetime import date
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from src.services.firms_service import fetch_bbox
from src.utils.geo import parse_bbox
from src.utils.limiter import limiter

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/viirs")
@limiter.limit("30/minute")
async def viirs(request: Request, days: int = Query(1, ge=1, le=10), bbox: Optional[str] = None):
    """Fetch VIIRS fire detections. NRT API supports 1-10 days only."""
    if bbox:
        try:
            parse_bbox(bbox)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return await fetch_bbox("VIIRS", days, bbox_override=bbox)


@router.get("/modis")
@limiter.limit("30/minute")
async def modis(request: Request, days: int = Query(1, ge=1, le=10), bbox: Optional[str] = None):
    """Fetch MODIS fire detections. NRT API supports 1-10 days only."""
    if bbox:
        try:
            parse_bbox(bbox)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    return await fetch_bbox("MODIS", days, bbox_override=bbox)


@router.get("/history")
@limiter.limit("30/minute")
async def history(
    request: Request,
    start: date = Query(..., description="Start date YYYY-MM-DD"),
    end: date = Query(..., description="End date YYYY-MM-DD"),
    bbox: Optional[str] = Query(None, description="lon_min,lat_min,lon_max,lat_max"),
    source: Optional[str] = Query(None, description="VIIRS or MODIS"),
    limit: int = Query(5000, ge=1, le=50000),
):
    """Query historical fire events from the database.

    Returns GeoJSON FeatureCollection. Requires DATABASE_URL to be configured.
    """
    from src.db.session import is_configured

    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail="Historical data unavailable — database not configured (set DATABASE_URL)",
        )

    if end < start:
        raise HTTPException(status_code=400, detail="end must be >= start")

    bbox_coords = None
    if bbox:
        try:
            bbox_coords = parse_bbox(bbox)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    from sqlalchemy import text
    from src.db.session import get_session

    # Build query with optional filters
    conditions = ["detected_at::date BETWEEN :start AND :end"]
    params: dict = {"start": start.isoformat(), "end": end.isoformat(), "limit": limit}

    if source:
        conditions.append("source = :source")
        params["source"] = source.upper()

    if bbox_coords:
        lon_min, lat_min, lon_max, lat_max = bbox_coords
        conditions.append(
            "ST_Within(location::geometry, "
            "ST_MakeEnvelope(:lon_min, :lat_min, :lon_max, :lat_max, 4326))"
        )
        params.update(
            {"lon_min": lon_min, "lat_min": lat_min, "lon_max": lon_max, "lat_max": lat_max}
        )

    where = " AND ".join(conditions)
    sql = text(
        f"SELECT id, source, detected_at, frp, confidence, region,"
        f" ST_Y(location::geometry) AS lat, ST_X(location::geometry) AS lon"
        f" FROM fire_events WHERE {where}"
        f" ORDER BY detected_at DESC LIMIT :limit"
    )

    try:
        async with get_session() as session:
            result = await session.execute(sql, params)
            rows = result.fetchall()
    except Exception as exc:
        logger.error("History query failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed")

    features = [
        {
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
            "properties": {
                "id": row.id,
                "source": row.source,
                "detected_at": row.detected_at.isoformat(),
                "frp": row.frp,
                "confidence": row.confidence,
                "region": row.region,
            },
        }
        for row in rows
    ]
    return {"type": "FeatureCollection", "features": features}


def register_jobs(scheduler: AsyncIOScheduler):
    async def fetch_viirs():
        await fetch_bbox("VIIRS", 1)

    async def fetch_modis():
        await fetch_bbox("MODIS", 1)

    scheduler.add_job(fetch_viirs, "interval", minutes=30, id="firms_viirs")
    scheduler.add_job(fetch_modis, "interval", minutes=30, id="firms_modis")
