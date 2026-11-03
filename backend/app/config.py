"""
Application configuration — reads from environment variables.
"""

from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    """Central config pulled from env vars / .env file."""

    # ── App ──
    app_name: str = "SIH-2K26 Cyclone Prediction API"
    environment: str = "development"
    debug: bool = True

    # ── Database ──
    database_url: str = "postgresql+psycopg_async://cyclone@localhost:5432/cyclone_db"
    database_url_sync: str = "postgresql+psycopg://cyclone@localhost:5432/cyclone_db"

    # ── MinIO ──
    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = ""
    minio_secure: bool = False

    # ── MLflow ──
    mlflow_tracking_uri: str = "http://localhost:5001"

    # ── Prefect ──
    prefect_api_url: str = "http://localhost:4200/api"

    # ── CORS ──
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:8080",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ]

    # ── Coastline data path ──
    coastline_shapefile: str = "/app/data/coastline/ne_110m_coastline.shp"
    land_shapefile: str = "/app/data/coastline/ne_110m_land.shp"

    # ── Gemini AI ──
    gemini_api_key: str = ""
    gemini_live_model: str = "models/gemini-3.1-flash-live-preview"
    gemini_chat_model: str = "models/gemini-3.6-flash"

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Cached settings singleton."""
    return Settings()
