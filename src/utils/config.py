from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    APP_NAME: str = "Portugal Fire Detection"
    CORS_ORIGINS: str = ""
    PORTUGAL_BBOX: str = "-9.5,36.8,-6.0,42.3"
    SCHEDULER_ENABLED: bool = False
    TIMEZONE: str = "Europe/Lisbon"
    FIRMS_API_KEY: str = ""
    EUMETSAT_WMS_URL: str = ""
    EUMETSAT_FRP_LAYER: str = ""
    EUMETSAT_TOKEN: str = ""
    STAC_CATALOG_URL: str = "https://earth-search.aws.element84.com/v1"
    COPERNICUS_CLIENT_ID: str = ""
    COPERNICUS_CLIENT_SECRET: str = ""
    COPERNICUS_USERNAME: str = ""
    COPERNICUS_PASSWORD: str = ""
    COPERNICUS_ACCESS_TOKEN: str = ""
    COPERNICUS_S3_ACCESS_KEY: str = ""
    COPERNICUS_S3_SECRET_KEY: str = ""
    FIRMS_TTL: int = 900
    EUMETSAT_TTL: int = 1800
    S2_TTL: int = 86400
    MODEL_CHECKPOINT_PATH: str = "ml/training/checkpoints/best_model.pth"
    APP_ENV: str = "development"
    DATABASE_URL: str = ""
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_USER: str = ""
    SMTP_PASS: str = ""
    FROM_EMAIL: str = "alerts@fire-detection.local"
    DEMO_MODE: bool = False
    SNAPSHOT_DIR: str = "data/snapshot"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
