import os
import time
import glob
from apscheduler.schedulers.asyncio import AsyncIOScheduler

def cleanup_data_dir(retention_days: int = 14):
    cutoff = time.time() - retention_days * 86400
    for sub in ("firms", "eumetsat", "sentinel2"):
        for f in glob.glob(f"/app/data/{sub}/*"):
            try:
                if os.path.getmtime(f) < cutoff:
                    os.remove(f)
            except Exception:
                pass

def schedule_cleanup(scheduler: AsyncIOScheduler):
    scheduler.add_job(cleanup_data_dir, "cron", hour=3, id="cleanup")
