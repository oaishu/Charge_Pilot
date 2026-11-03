"""Settings for the standalone ML serving service (ml/service).

Fully standalone: this service owns a **dedicated database** (default
``chargepilot_ml``) — it never points at the main backend's demo database and
never touches it. Bring the ``ml/`` folder to a fresh machine, point
``DATABASE_URL`` at any reachable Postgres (or a SQLite file) and bootstrap:

    python scripts_setup_db.py     # creates DB + tables + demo data

Env files, in priority order (later wins):
1. ``ml/service/.env``  — service-specific overrides
2. ``ml/.env``          — shared ML-side config (DB URL, JWT, ACN token)
3. process environment variables always beat files"""
from typing import List

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- App ---
    APP_NAME: str = "ChargePilot ML Service"
    APP_VERSION: str = "0.1.0"
    API_V1_PREFIX: str = "/api/v1"
    DEBUG: bool = False

    # --- Database: the ML service's OWN database (never the backend's) ---
    DATABASE_URL: str = "postgresql+asyncpg://chargepilot:chargepilot_dev@localhost:5432/chargepilot_ml"

    # --- CORS ---
    CORS_ORIGINS: List[str] = ["http://localhost:3000"]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def split_origins(cls, v):
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # --- JWT (this service's own tokens; no shared secret with the backend) ---
    JWT_SECRET_KEY: str = "change-me-in-prod"
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30

    # --- ML artifacts (the ml/ folder's own bundle output) ---
    ML_ARTIFACTS_DIR: str = "ml/artifacts"
    ML_FORECAST_MODEL_VERSION: str = "baseline-v0.1"
    ML_FAULT_MODEL_VERSION: str = "guardrail-v0.1"


def get_settings() -> Settings:
    return Settings()


settings = get_settings()
