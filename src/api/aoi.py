"""Area of Interest (AOI) saved searches.

Allows users to define named geographic polygons and query fires within them.
Requires DATABASE_URL to be configured.
"""
from __future__ import annotations

import logging
import uuid
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from pydantic import BaseModel, field_validator

from src.utils.limiter import limiter

logger = logging.getLogger(__name__)
router = APIRouter()


# ─── Pydantic schemas ────────────────────────────────────────────────────────


class AOICreate(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    boundary: dict  # GeoJSON Polygon geometry

    @field_validator("boundary")
    @classmethod
    def validate_boundary(cls, v: dict) -> dict:
        if v.get("type") != "Polygon":
            raise ValueError("boundary must be a GeoJSON Polygon geometry")
        coords = v.get("coordinates")
        if not coords or not isinstance(coords, list) or len(coords[0]) < 4:
            raise ValueError("Polygon must have at least 4 coordinate pairs")
        return v


class AOIResponse(BaseModel):
    id: str
    name: Optional[str]
    email: Optional[str]
    created_at: str
    active: bool


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _require_db():
    from src.db.session import is_configured

    if not is_configured():
        raise HTTPException(
            status_code=503,
            detail="AOI features unavailable — database not configured (set DATABASE_URL)",
        )


def _geojson_polygon_to_wkt(geojson: dict) -> str:
    """Convert a GeoJSON Polygon geometry to WKT (POLYGON(...))."""
    rings = geojson["coordinates"]
    ring_strs = []
    for ring in rings:
        pairs = ", ".join(f"{lon} {lat}" for lon, lat in ring)
        ring_strs.append(f"({pairs})")
    return f"POLYGON({', '.join(ring_strs)})"


# ─── Endpoints ───────────────────────────────────────────────────────────────


@router.post("", status_code=201)
@limiter.limit("20/minute")
async def create_aoi(request: Request, body: AOICreate):
    """Create a new Area of Interest."""
    _require_db()

    wkt = _geojson_polygon_to_wkt(body.boundary)

    from sqlalchemy import text
    from src.db.session import get_session

    aoi_id = uuid.uuid4()
    sql = text(
        "INSERT INTO areas_of_interest (id, name, email, boundary)"
        " VALUES (:id, :name, :email, ST_GeomFromText(:wkt, 4326))"
        " RETURNING id, name, email, created_at, active"
    )
    try:
        async with get_session() as session:
            result = await session.execute(
                sql,
                {
                    "id": str(aoi_id),
                    "name": body.name,
                    "email": body.email,
                    "wkt": wkt,
                },
            )
            row = result.fetchone()
            await session.commit()
    except Exception as exc:
        logger.error("AOI create failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to create AOI")

    return {
        "id": str(row.id),
        "name": row.name,
        "email": row.email,
        "created_at": row.created_at.isoformat(),
        "active": row.active,
    }


@router.get("")
@limiter.limit("60/minute")
async def list_aois(request: Request):
    """List all active Areas of Interest."""
    _require_db()

    from sqlalchemy import text
    from src.db.session import get_session

    sql = text(
        "SELECT id, name, email, created_at, active,"
        " ST_AsGeoJSON(boundary::geometry) AS boundary_geojson"
        " FROM areas_of_interest WHERE active = true ORDER BY created_at DESC"
    )
    try:
        async with get_session() as session:
            result = await session.execute(sql)
            rows = result.fetchall()
    except Exception as exc:
        logger.error("AOI list failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database query failed")

    import json

    return [
        {
            "id": str(row.id),
            "name": row.name,
            "email": row.email,
            "created_at": row.created_at.isoformat(),
            "active": row.active,
            "boundary": json.loads(row.boundary_geojson) if row.boundary_geojson else None,
        }
        for row in rows
    ]


@router.delete("/{aoi_id}", status_code=204)
@limiter.limit("20/minute")
async def delete_aoi(request: Request, aoi_id: str):
    """Soft-delete an Area of Interest (sets active=false)."""
    _require_db()

    try:
        uuid.UUID(aoi_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid AOI id")

    from sqlalchemy import text
    from src.db.session import get_session

    sql = text(
        "UPDATE areas_of_interest SET active = false WHERE id = :id AND active = true"
        " RETURNING id"
    )
    try:
        async with get_session() as session:
            result = await session.execute(sql, {"id": aoi_id})
            row = result.fetchone()
            await session.commit()
    except Exception as exc:
        logger.error("AOI delete failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not row:
        raise HTTPException(status_code=404, detail="AOI not found")


@router.get("/{aoi_id}/fires")
@limiter.limit("30/minute")
async def aoi_fires(
    request: Request,
    aoi_id: str,
    days: int = Query(7, ge=1, le=90, description="How many days back to search"),
):
    """Get recent fires within an AOI boundary using PostGIS ST_Within."""
    _require_db()

    try:
        uuid.UUID(aoi_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid AOI id")

    from sqlalchemy import text
    from src.db.session import get_session

    sql = text(
        "SELECT fe.id, fe.source, fe.detected_at, fe.frp, fe.confidence,"
        " ST_Y(fe.location::geometry) AS lat, ST_X(fe.location::geometry) AS lon"
        " FROM fire_events fe"
        " JOIN areas_of_interest aoi ON aoi.id = :aoi_id"
        " WHERE ST_Within(fe.location::geometry, aoi.boundary::geometry)"
        "   AND fe.detected_at >= NOW() - INTERVAL ':days days'"
        "   AND aoi.active = true"
        " ORDER BY fe.detected_at DESC"
        " LIMIT 1000"
    )
    # Use string formatting for the interval (safe — days is validated int)
    sql = text(
        "SELECT fe.id, fe.source, fe.detected_at, fe.frp, fe.confidence,"
        " ST_Y(fe.location::geometry) AS lat, ST_X(fe.location::geometry) AS lon"
        " FROM fire_events fe"
        " JOIN areas_of_interest aoi ON aoi.id = :aoi_id"
        f" WHERE ST_Within(fe.location::geometry, aoi.boundary::geometry)"
        f"   AND fe.detected_at >= NOW() - INTERVAL '{days} days'"
        f"   AND aoi.active = true"
        f" ORDER BY fe.detected_at DESC LIMIT 1000"
    )
    try:
        async with get_session() as session:
            result = await session.execute(sql, {"aoi_id": aoi_id})
            rows = result.fetchall()
    except Exception as exc:
        logger.error("AOI fires query failed: %s", exc)
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
            },
        }
        for row in rows
    ]
    return {"type": "FeatureCollection", "features": features, "count": len(features)}


@router.post("/{aoi_id}/subscribe", status_code=200)
@limiter.limit("20/minute")
async def subscribe_aoi(request: Request, aoi_id: str, body: dict):
    """Add or update an email subscription for AOI fire alerts."""
    _require_db()

    try:
        uuid.UUID(aoi_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid AOI id")

    email = body.get("email", "").strip()
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Valid email required")

    from sqlalchemy import text
    from src.db.session import get_session

    sql = text(
        "UPDATE areas_of_interest SET email = :email"
        " WHERE id = :id AND active = true RETURNING id"
    )
    try:
        async with get_session() as session:
            result = await session.execute(sql, {"id": aoi_id, "email": email})
            row = result.fetchone()
            await session.commit()
    except Exception as exc:
        logger.error("AOI subscribe failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not row:
        raise HTTPException(status_code=404, detail="AOI not found")

    return {"subscribed": True, "email": email, "aoi_id": aoi_id}


@router.delete("/{aoi_id}/subscribe", status_code=200)
@limiter.limit("20/minute")
async def unsubscribe_aoi(request: Request, aoi_id: str):
    """Remove email subscription from an AOI."""
    _require_db()

    try:
        uuid.UUID(aoi_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid AOI id")

    from sqlalchemy import text
    from src.db.session import get_session

    sql = text(
        "UPDATE areas_of_interest SET email = NULL"
        " WHERE id = :id AND active = true RETURNING id"
    )
    try:
        async with get_session() as session:
            result = await session.execute(sql, {"id": aoi_id})
            row = result.fetchone()
            await session.commit()
    except Exception as exc:
        logger.error("AOI unsubscribe failed: %s", exc)
        raise HTTPException(status_code=500, detail="Database error")

    if not row:
        raise HTTPException(status_code=404, detail="AOI not found")

    return {"subscribed": False, "aoi_id": aoi_id}
