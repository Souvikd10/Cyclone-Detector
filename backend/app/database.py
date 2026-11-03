"""
Async SQLAlchemy engine + session factory with PostGIS support and resilient fallback.
"""

import sys
import asyncio
import logging
from typing import AsyncGenerator
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase
from app.config import get_settings

# On Windows with Python 3.8+, psycopg requires WindowsSelectorEventLoopPolicy for async I/O
if sys.platform == "win32":
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

logger = logging.getLogger(__name__)
settings = get_settings()


def normalize_async_db_url(url: str) -> str:
    """Ensure database URL uses an installed async driver on Windows."""
    if url.startswith("postgres://"):
        return url.replace("postgres://", "postgresql+psycopg_async://", 1)
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg_async://", 1)
    if url.startswith("postgresql+asyncpg://"):
        try:
            import asyncpg  # noqa: F401
            return url
        except ImportError:
            return url.replace("postgresql+asyncpg://", "postgresql+psycopg_async://", 1)
    return url


db_url = normalize_async_db_url(settings.database_url)

pool_kwargs = {"pool_pre_ping": True}
if not db_url.startswith("sqlite"):
    pool_kwargs.update({"pool_size": 10, "max_overflow": 5})

try:
    engine = create_async_engine(
        db_url,
        echo=False,
        **pool_kwargs,
    )
except Exception as e:
    logger.warning(f"Could not create async engine for {db_url}: {e}")
    # Fallback to local sqlite memory if engine creation fails
    engine = create_async_engine("sqlite+aiosqlite:///:memory:", echo=False)

async_session = async_sessionmaker(
    engine,
    class_=AsyncSession,
    expire_on_commit=False,
)

_db_connected = False


def is_db_connected() -> bool:
    """Return True if the database was successfully connected to and initialized."""
    return _db_connected


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


async def get_db() -> AsyncGenerator[AsyncSession | None, None]:
    """
    FastAPI dependency — yields an async DB session when connected,
    or None if the database is offline (mock fallback mode).
    """
    global _db_connected
    if not _db_connected:
        yield None
        return

    try:
        async with async_session() as session:
            try:
                yield session
                await session.commit()
            except Exception:
                await session.rollback()
                raise
            finally:
                await session.close()
    except Exception as e:
        logger.warning(f"Database session error: {e}")
        yield None


async def init_db():
    """Create all tables if DB is available; otherwise run in standalone mock mode."""
    global _db_connected
    try:
        async with engine.begin() as conn:
            import app.models.storm  # noqa: F401
            import app.models.prediction  # noqa: F401
            import app.models.feedback  # noqa: F401
            await conn.run_sync(Base.metadata.create_all)
        _db_connected = True
        logger.info("Database connected & tables initialized successfully ✓")
    except Exception as e:
        _db_connected = False
        logger.warning(
            f"Database connection unavailable or uninitialized ({type(e).__name__}: {e}). "
            "Running in standalone mock mode (all endpoints active with built-in NIO storm data)."
        )


async def close_db():
    """Dispose of the connection pool."""
    global _db_connected
    if _db_connected:
        try:
            await engine.dispose()
        except Exception:
            pass
