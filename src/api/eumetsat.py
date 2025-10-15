from fastapi import APIRouter
from src.services.eumetsat_service import fetch_frp_png
from apscheduler.schedulers.asyncio import AsyncIOScheduler

router = APIRouter()

@router.get("/frp/latest")
async def frp_latest():
    return await fetch_frp_png()

def register_jobs(scheduler: AsyncIOScheduler):
    scheduler.add_job(frp_latest, "interval", hours=1, id="eumetsat_frp")
