from fastapi import APIRouter, Query
from src.services.sentinel_service import search_items, fetch_thumbnail
from src.services.snapshot import read_sentinel2_manifest
from src.utils.demo import is_demo
from apscheduler.schedulers.asyncio import AsyncIOScheduler

router = APIRouter()


@router.get("/search")
async def search(
    days: int = Query(7, ge=1, le=30),
    cloud_pct: int = Query(20, ge=0, le=100),
    limit: int = Query(12, ge=1, le=50),
):
    if is_demo():
        return read_sentinel2_manifest()
    return await search_items(days, cloud_pct, limit)


@router.get("/thumbnail")
async def thumbnail(days: int = Query(7), cloud_pct: int = Query(20)):
    if is_demo():
        items = read_sentinel2_manifest().get("items", [])
        return items[0] if items else {}
    data = await search_items(days, cloud_pct, 1)
    items = data.get("features", [])
    if not items:
        return {}
    return await fetch_thumbnail(items[0])


def register_jobs(scheduler: AsyncIOScheduler):
    if is_demo():
        return
    async def search_daily():
        await search_items(7, 20, 10)

    scheduler.add_job(search_daily, "cron", hour=2, id="s2_search")
