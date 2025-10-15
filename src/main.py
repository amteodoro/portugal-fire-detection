from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import RedirectResponse
from src.utils.config import settings
from src.api import firms, eumetsat, sentinel, images
from src.utils.cleanup import schedule_cleanup
from apscheduler.schedulers.asyncio import AsyncIOScheduler

def create_app() -> FastAPI:
    app = FastAPI(title=settings.APP_NAME)
    origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins or ["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.mount("/static", StaticFiles(directory="src/static"), name="static")

    @app.get("/", include_in_schema=False)
    async def root():
        return RedirectResponse(url="/static/index.html")

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    app.include_router(firms.router, prefix="/api/firms", tags=["FIRMS"])
    app.include_router(eumetsat.router, prefix="/api/eumetsat", tags=["EUMETSAT"])
    app.include_router(sentinel.router, prefix="/api/sentinel2", tags=["Sentinel-2"])
    app.include_router(images.router, prefix="/api/images", tags=["Images"])

    if settings.SCHEDULER_ENABLED:
        scheduler = AsyncIOScheduler(timezone=settings.TIMEZONE)
        firms.register_jobs(scheduler)
        eumetsat.register_jobs(scheduler)
        sentinel.register_jobs(scheduler)
        schedule_cleanup(scheduler)
        scheduler.start()

    return app

app = create_app()
