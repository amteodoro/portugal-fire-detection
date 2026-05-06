import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from src.utils.config import settings
from src.utils.limiter import limiter
from src.api import firms, eumetsat, sentinel, sentinel3, images, dataset
from src.api import dataset_summary, dataset_scenes, dataset_training
from src.utils.cleanup import schedule_cleanup
from apscheduler.schedulers.asyncio import AsyncIOScheduler

logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        checkpoint = settings.MODEL_CHECKPOINT_PATH
        if Path(checkpoint).exists():
            try:
                from ml.inference.predictor import FirePredictor
                app.state.predictor = FirePredictor(checkpoint)
                logger.info("Fire predictor loaded from %s", checkpoint)
            except Exception as e:
                logger.warning("Failed to load fire predictor: %s", e)
                app.state.predictor = None
        else:
            app.state.predictor = None
            logger.warning("No checkpoint found at %s — inference disabled", checkpoint)
        from src.db.session import init_db
        init_db()
        try:
            yield
        finally:
            from src.utils.http import close_client
            from src.db.session import close_db
            await close_client()
            await close_db()

    app = FastAPI(title=settings.APP_NAME, lifespan=lifespan)

    # Rate limiting
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    if settings.APP_ENV == "production" and not origins:
        raise RuntimeError(
            "CORS_ORIGINS must be set in production (APP_ENV=production). "
            "Set it to a comma-separated list of allowed origins, e.g. https://yourdomain.com"
        )
    if not origins:
        origins = ["http://localhost", "http://localhost:8000"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type"],
    )
    if Path("data/samples").exists():
        app.mount("/static/samples", StaticFiles(directory="data/samples"), name="samples")
    app.mount("/static", StaticFiles(directory="src/static"), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/static/index.html")

    @app.get("/api/health")
    async def health():
        from src.utils.demo import is_demo
        body = {"status": "ok", "demo": is_demo()}
        if is_demo():
            try:
                from src.services.snapshot import read_manifest
                manifest = read_manifest()
                if manifest:
                    body["snapshot_range"] = manifest.get("snapshot_range")
            except ImportError:
                pass
        return body

    @app.get("/api/config")
    async def config():
        from src.utils.geo import parse_bbox

        bbox = parse_bbox(settings.PORTUGAL_BBOX)
        lon_min, lat_min, lon_max, lat_max = bbox
        center_lat = (lat_min + lat_max) / 2
        center_lon = (lon_min + lon_max) / 2
        return {
            "bbox": list(bbox),
            "center": [center_lat, center_lon],
            "app_name": settings.APP_NAME,
        }

    from src.api import inference, aoi
    app.include_router(firms.router, prefix="/api/firms", tags=["FIRMS"])
    app.include_router(eumetsat.router, prefix="/api/eumetsat", tags=["EUMETSAT"])
    app.include_router(sentinel.router, prefix="/api/sentinel2", tags=["Sentinel-2"])
    app.include_router(images.router, prefix="/api/images", tags=["Images"])
    app.include_router(dataset.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_summary.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_scenes.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(dataset_training.router, prefix="/api/dataset", tags=["Dataset"])
    app.include_router(inference.router, prefix="/api/inference", tags=["Inference"])
    app.include_router(sentinel3.router, prefix="/api/sentinel3", tags=["Sentinel-3"])
    app.include_router(aoi.router, prefix="/api/aoi", tags=["AOI"])

    if settings.SCHEDULER_ENABLED:
        scheduler = AsyncIOScheduler(timezone=settings.TIMEZONE)
        firms.register_jobs(scheduler)
        eumetsat.register_jobs(scheduler)
        sentinel.register_jobs(scheduler)
        schedule_cleanup(scheduler)

        # Fire alert job — runs every 30 min, skips gracefully if not configured
        async def _run_alerts():
            from src.services.alert_service import check_and_alert
            sent = await check_and_alert()
            if sent:
                logger.info("Alert job: sent %d alert(s)", sent)

        scheduler.add_job(_run_alerts, "interval", minutes=30, id="fire_alerts")

        # Drift detection — weekly, Sunday 03:00 UTC
        async def _run_drift():
            from src.services.drift_service import check_drift
            report = await check_drift(app.state)
            logger.info("Drift check report: %s", report)

        scheduler.add_job(
            _run_drift, "cron", day_of_week="sun", hour=3, id="drift_check"
        )
        scheduler.start()

    return app


app = create_app()
