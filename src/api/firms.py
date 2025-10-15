from fastapi import APIRouter, Query
from src.services.firms_service import fetch_country
from apscheduler.schedulers.asyncio import AsyncIOScheduler

router = APIRouter()

@router.get("/viirs")
async def viirs(days: int = Query(1, ge=1, le=365)):
    return await fetch_country("VIIRS", days)

@router.get("/modis")
async def modis(days: int = Query(1, ge=1, le=365)):
    return await fetch_country("MODIS", days)

def register_jobs(scheduler: AsyncIOScheduler):
    async def fetch_viirs():
        await fetch_country("VIIRS", 1)
    
    async def fetch_modis():
        await fetch_country("MODIS", 1)
    
    scheduler.add_job(fetch_viirs, "interval", minutes=30, id="firms_viirs")
    scheduler.add_job(fetch_modis, "interval", minutes=30, id="firms_modis")
