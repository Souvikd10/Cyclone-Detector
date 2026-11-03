"""
SIH-2K26 Cyclone Prediction API — Main Application Entry Point.

FastAPI app with CORS, lifespan management, and all routers mounted.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import get_settings
from app.database import init_db, close_db
from app.routers import storms, impact, alerts, feedback, websocket, seed, voice

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    logger.info("=" * 60)
    logger.info(f"  {settings.app_name}")
    logger.info(f"  Environment: {settings.environment}")
    logger.info("=" * 60)

    # Startup: create tables
    logger.info("Initializing database...")
    await init_db()
    logger.info("Database initialized ✓")

    yield

    # Shutdown: close connections
    logger.info("Shutting down...")
    await close_db()
    logger.info("Database connections closed ✓")


# ── Create FastAPI app ──
app = FastAPI(
    title=settings.app_name,
    description=(
        "Backend API for the SIH-2K26 Cyclone Prediction & Decision-Support System.\n\n"
        "**Stages:**\n"
        "- Stage 14: Landfall estimation & coastal impact zones\n"
        "- Stage 16: CAP 1.2 alert generation (PROTOTYPE)\n"
        "- Stage 17: Human-in-the-loop feedback system\n\n"
        "**Note:** All alerts carry `status=Test`. This is a prototype system."
    ),
    version="0.1.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# ── CORS middleware ──
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Mount routers ──
app.include_router(storms.router, prefix="")
app.include_router(impact.router, prefix="")
app.include_router(alerts.router, prefix="")
app.include_router(feedback.router, prefix="")
app.include_router(websocket.router, prefix="")
app.include_router(seed.router, prefix="")
app.include_router(voice.router, prefix="")


# ── Health check ──
@app.get("/", tags=["Health"])
async def root():
    """Health check / API info endpoint."""
    return {
        "name": settings.app_name,
        "version": "0.1.0",
        "status": "running",
        "environment": settings.environment,
        "docs": "/docs",
        "endpoints": {
            "storms": "GET /storms",
            "latest_prediction": "GET /storms/{id}/latest",
            "prediction_history": "GET /storms/{id}/history?from=&to=",
            "impact": "GET /storms/{id}/impact",
            "alerts": "GET /alerts/{storm_id}",
            "feedback": "POST /storms/{id}/feedback",
            "websocket": "WS /ws/storms/{id}",
            "voice_token": "POST /voice/token",
            "seed": "POST /seed",
        },
    }


@app.get("/health", tags=["Health"])
async def health_check():
    """Detailed health check with dependency status."""
    from sqlalchemy import text
    from app.database import async_session

    checks = {"api": "healthy"}

    # Check database
    try:
        async with async_session() as session:
            await session.execute(text("SELECT 1"))
        checks["database"] = "healthy"
    except Exception as e:
        checks["database"] = f"unhealthy: {str(e)}"

    overall = "healthy" if all(v == "healthy" for v in checks.values()) else "degraded"

    return {
        "status": overall,
        "checks": checks,
    }
