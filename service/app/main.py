"""Standalone ML serving service entry point.

Runs fully independently of ``backend/`` — own config/env, own engine wiring,
own artifact loader resolving ``ml/artifacts``, own database (created and
seeded by ``scripts_setup_db.py``) and own JWT secret for its own users.

Bootstrap on a fresh machine:  python scripts_setup_db.py
Run:                           python -m uvicorn app.main:app --port 8001
"""
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text

from app.core.config import settings
from app.core.database import engine
from app.core.errors import install_error_handlers
from app.core.ml_artifacts import staged_versions
from app.routers import auth, faults, forecasts, insights, telemetry

logger = logging.getLogger("ml.service")

app = FastAPI(
    title="ChargePilot ML Service",
    description=(
        "Standalone serving surface for the models trained in the ml/ notebooks: "
        "demand forecasts (baseline + RandomForest bounded-ratio correction), "
        "fault detection (trained GBDT + guardrail safety layer, fed by the "
        "telemetry ingestion endpoint) and operator insights. Owns a dedicated "
        "database with its own users and demo data (bootstrap via "
        "scripts_setup_db.py); artifacts are loaded from ml/artifacts with the "
        "beats_baseline promotion guard."
    ),
    version=settings.APP_VERSION,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
install_error_handlers(app)

app.include_router(auth.router, prefix=settings.API_V1_PREFIX)
app.include_router(forecasts.router, prefix=settings.API_V1_PREFIX)
app.include_router(faults.router, prefix=settings.API_V1_PREFIX)
app.include_router(insights.router, prefix=settings.API_V1_PREFIX)
app.include_router(telemetry.router, prefix=settings.API_V1_PREFIX)


@app.get("/health")
async def health():
    """Liveness + staged model versions (and DB reachability)."""
    db_status = "ok"
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
    except Exception:  # noqa: BLE001 - health must never raise
        logger.exception("health DB ping failed")
        db_status = "unreachable"
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "database": db_status,
        "models": staged_versions(),
    }
