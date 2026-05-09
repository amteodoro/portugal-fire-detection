from fastapi import APIRouter
from src.services.eumetsat_service import fetch_frp_png
from src.services.snapshot import eumetsat_frp_path
from src.utils.demo import is_demo
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from datetime import datetime, timezone

router = APIRouter()


@router.get("/frp/latest")
async def frp_latest():
    if is_demo():
        path = eumetsat_frp_path()
        if path is None:
            return {"error": "EUMETSAT FRP unavailable in demo snapshot"}
        return {
            "id": path.name,
            "source": "EUMETSAT_FRP",
            "datetime": datetime.now(timezone.utc).isoformat(),
            "path": str(path),
        }
    return await fetch_frp_png()


def register_jobs(scheduler: AsyncIOScheduler):
    if is_demo():
        return
    scheduler.add_job(frp_latest, "interval", hours=1, id="eumetsat_frp")
