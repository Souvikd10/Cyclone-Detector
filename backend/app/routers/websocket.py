"""
WebSocket router:
  WS /ws/storms/{id} — push live prediction updates to connected dashboards
"""

import logging
import asyncio
from uuid import UUID

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.models.prediction import Prediction
from app.core.websocket_manager import manager

logger = logging.getLogger(__name__)

router = APIRouter(tags=["WebSocket"])


@router.websocket("/ws/storms/{storm_id}")
async def storm_websocket(
    websocket: WebSocket,
    storm_id: UUID,
):
    """
    WebSocket endpoint for live prediction updates.

    Once connected, the client receives:
    1. The latest prediction immediately (as a welcome message)
    2. Any new predictions as they are created (via broadcast)

    Sends heartbeat pings every 30 seconds to keep the connection alive.
    """
    storm_id_str = str(storm_id)

    await manager.connect(websocket, storm_id_str)

    try:
        # Send initial state — latest prediction
        async with get_db_session() as db:
            result = await db.execute(
                select(Prediction)
                .where(Prediction.storm_id == storm_id)
                .order_by(Prediction.timestamp.desc())
                .limit(1)
            )
            prediction = result.scalar_one_or_none()

            if prediction:
                await manager.send_personal(websocket, {
                    "type": "initial",
                    "storm_id": storm_id_str,
                    "prediction": _serialize_prediction(prediction),
                })
            else:
                await manager.send_personal(websocket, {
                    "type": "initial",
                    "storm_id": storm_id_str,
                    "prediction": None,
                    "message": "No predictions available yet",
                })

        # Keep alive loop — heartbeat + listen for client messages
        while True:
            try:
                # Wait for client messages (or timeout for heartbeat)
                data = await asyncio.wait_for(
                    websocket.receive_text(),
                    timeout=30.0,
                )
                # Handle client messages (e.g., "ping" → "pong")
                if data.strip().lower() == "ping":
                    await manager.send_personal(websocket, {"type": "pong"})

            except asyncio.TimeoutError:
                # Send heartbeat
                await manager.send_personal(websocket, {
                    "type": "heartbeat",
                    "connections": manager.get_connection_count(storm_id_str),
                })

    except WebSocketDisconnect:
        manager.disconnect(websocket, storm_id_str)
        logger.info(f"Client disconnected from storm {storm_id_str}")
    except Exception as e:
        manager.disconnect(websocket, storm_id_str)
        logger.error(f"WebSocket error for storm {storm_id_str}: {e}")


def _serialize_prediction(prediction: Prediction) -> dict:
    """Serialize a prediction ORM object for WebSocket transmission."""
    return {
        "id": str(prediction.id),
        "storm_id": str(prediction.storm_id),
        "timestamp": prediction.timestamp.isoformat() if prediction.timestamp else None,
        "detection": prediction.detection,
        "structure": prediction.structure,
        "current_state": prediction.current_state,
        "forecast": prediction.forecast,
        "ri": prediction.ri,
        "uncertainty": prediction.uncertainty,
        "xai": prediction.xai,
        "impact": prediction.impact,
        "model_run_id": prediction.model_run_id,
        "created_at": prediction.created_at.isoformat() if prediction.created_at else None,
    }


async def get_db_session():
    """Create a standalone async session for WebSocket use (not a FastAPI dependency)."""
    from app.database import async_session
    return async_session()
