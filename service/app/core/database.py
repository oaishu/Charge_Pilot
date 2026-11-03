"""Async SQLAlchemy setup + bootstrap helpers for the standalone ML service.

This service owns a **dedicated database** (default ``chargepilot_ml``) that
never shares tables with the main backend's DB. Beyond engine/session wiring,
this module provides the bootstrap used by ``scripts_setup_db.py``:

* ``create_database_if_missing`` — connect to the server's admin DB
  (``postgres`` on Postgres) and ``CREATE DATABASE`` when absent; no-op on
  SQLite (the file is created on first connect).
* ``create_tables`` — ``Base.metadata.create_all`` so a fresh DB gets the full
  schema without any migration tooling.
"""
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy import text

from app.core.config import settings
from app.models import Base

engine = create_async_engine(settings.DATABASE_URL, echo=False, pool_pre_ping=True)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False, class_=AsyncSession)


async def get_db():
    """FastAPI dependency yielding an async session."""
    async with AsyncSessionLocal() as session:
        yield session


def _make_admin_url(database_url: str) -> str:
    """The URL to connect with when the target database may not exist yet."""
    head, _, _ = database_url.rpartition("/")
    return f"{head}/postgres"  # the server's default admin database


async def create_database_if_missing(database_url: str) -> bool:
    """Create the target database if it doesn't exist yet.

    Returns True when a database was created. On SQLite this is a no-op
    (the driver creates the file automatically).
    """
    if database_url.startswith("sqlite"):
        return False

    admin_engine = create_async_engine(_make_admin_url(database_url), isolation_level="AUTOCOMMIT")
    try:
        db_name = database_url.rsplit("/", 1)[1].split("?")[0]
        async with admin_engine.connect() as conn:
            exists = await conn.execute(
                text("SELECT 1 FROM pg_database WHERE datname = :name"), {"name": db_name}
            )
            if exists.scalar() is not None:
                return False
            await conn.execute(text(f'CREATE DATABASE "{db_name}"'))
            return True
    finally:
        await admin_engine.dispose()


async def create_tables() -> None:
    """Create all tables from the ORM metadata (idempotent)."""
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
