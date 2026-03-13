"""Async SQLAlchemy session factory.

Only initialised when DATABASE_URL is set in config. All callers must check
``is_configured()`` before using ``get_session()``.
"""
from __future__ import annotations

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from src.utils.config import settings

logger = logging.getLogger(__name__)

_engine = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def is_configured() -> bool:
    return bool(settings.DATABASE_URL)


def init_db() -> None:
    """Initialise engine + session factory. Call once at app startup."""
    global _engine, _session_factory
    if not settings.DATABASE_URL:
        logger.info("DATABASE_URL not set — database features disabled")
        return
    _engine = create_async_engine(
        settings.DATABASE_URL,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=10,
        echo=False,
    )
    _session_factory = async_sessionmaker(_engine, expire_on_commit=False)
    logger.info("Database engine initialised: %s", settings.DATABASE_URL.split("@")[-1])


async def close_db() -> None:
    global _engine
    if _engine is not None:
        await _engine.dispose()
        _engine = None


@asynccontextmanager
async def get_session() -> AsyncGenerator[AsyncSession, None]:
    if _session_factory is None:
        raise RuntimeError("Database not initialised — set DATABASE_URL")
    async with _session_factory() as session:
        yield session
