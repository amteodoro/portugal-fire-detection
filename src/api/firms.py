from fastapi import APIRouter, Query
from src.services.firms_service import fetch_bbox
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from typing import Optional

router = APIRouter()


@router.get("/viirs")
async def viirs(days: int = Query(1, ge=1, le=10), bbox: Optional[str] = None):
    """Fetch VIIRS fire detections. NRT API supports 1-10 days only."""
    return await fetch_bbox("VIIRS", days, bbox_override=bbox)


@router.get("/modis")
async def modis(days: int = Query(1, ge=1, le=10), bbox: Optional[str] = None):
    """Fetch MODIS fire detections. NRT API supports 1-10 days only."""
    return await fetch_bbox("MODIS", days, bbox_override=bbox)


def register_jobs(scheduler: AsyncIOScheduler):
    async def fetch_viirs():
        await fetch_bbox("VIIRS", 1)

    async def fetch_modis():
        await fetch_bbox("MODIS", 1)

    scheduler.add_job(fetch_viirs, "interval", minutes=30, id="firms_viirs")
    scheduler.add_job(fetch_modis, "interval", minutes=30, id="firms_modis")
