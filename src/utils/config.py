from pydantic import BaseModel
import os


class Settings(BaseModel):
    APP_NAME: str = os.getenv("APP_NAME", "Portugal Fire Detection")
    CORS_ORIGINS: str = os.getenv("CORS_ORIGINS", "*")
    PORTUGAL_BBOX: str = os.getenv("PORTUGAL_BBOX", "-9.5,36.8,-6.0,42.3")
    SCHEDULER_ENABLED: bool = os.getenv("SCHEDULER_ENABLED", "false").lower() == "true"
    TIMEZONE: str = os.getenv("TIMEZONE", "Europe/Lisbon")
    FIRMS_API_KEY: str = os.getenv("FIRMS_API_KEY", "")
    EUMETSAT_WMS_URL: str = os.getenv("EUMETSAT_WMS_URL", "")
    EUMETSAT_FRP_LAYER: str = os.getenv("EUMETSAT_FRP_LAYER", "")
    EUMETSAT_TOKEN: str = os.getenv("EUMETSAT_TOKEN", "")
    STAC_CATALOG_URL: str = os.getenv(
        "STAC_CATALOG_URL", "https://earth-search.aws.element84.com/v1"
    )
    COPERNICUS_CLIENT_ID: str = os.getenv("COPERNICUS_CLIENT_ID", "")
    COPERNICUS_CLIENT_SECRET: str = os.getenv("COPERNICUS_CLIENT_SECRET", "")
    COPERNICUS_USERNAME: str = os.getenv("COPERNICUS_USERNAME", "")
    COPERNICUS_PASSWORD: str = os.getenv("COPERNICUS_PASSWORD", "")
    FIRMS_TTL: int = int(os.getenv("FIRMS_TTL", "900"))
    EUMETSAT_TTL: int = int(os.getenv("EUMETSAT_TTL", "1800"))
    S2_TTL: int = int(os.getenv("S2_TTL", "86400"))


settings = Settings()
