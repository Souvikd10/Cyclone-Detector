"""
WebSocket connection manager for live prediction broadcasting.
"""

import json
import logging
from typing import Any
from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    """
    Manages WebSocket connections grouped by storm_id.
    Broadcasts prediction updates to all clients watching a given storm.
    """

    def __init__(self):
        # storm_id -> list of connected WebSocket clients
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, websocket: WebSocket, storm_id: str):
        """Accept a connection and register it under the storm_id."""
        await websocket.accept()
        if storm_id not in self._connections:
            self._connections[storm_id] = []
        self._connections[storm_id].append(websocket)
        logger.info(
            f"WS client connected for storm {storm_id}. "
            f"Total connections for this storm: {len(self._connections[storm_id])}"
        )

    def disconnect(self, websocket: WebSocket, storm_id: str):
        """Remove a connection from the registry."""
        if storm_id in self._connections:
            self._connections[storm_id] = [
                ws for ws in self._connections[storm_id] if ws != websocket
            ]
            if not self._connections[storm_id]:
                del self._connections[storm_id]
        logger.info(f"WS client disconnected from storm {storm_id}")

    async def broadcast(self, storm_id: str, data: dict[str, Any]):
        """Send a JSON message to all clients watching a storm."""
        if storm_id not in self._connections:
            return

        message = json.dumps(data, default=str)
        disconnected = []

        for websocket in self._connections[storm_id]:
            try:
                await websocket.send_text(message)
            except Exception:
                disconnected.append(websocket)

        # Clean up dead connections
        for ws in disconnected:
            self.disconnect(ws, storm_id)

    async def send_personal(self, websocket: WebSocket, data: dict[str, Any]):
        """Send a message to a single client."""
        message = json.dumps(data, default=str)
        await websocket.send_text(message)

    def get_connection_count(self, storm_id: str) -> int:
        """Number of active connections for a storm."""
        return len(self._connections.get(storm_id, []))

    def get_total_connections(self) -> int:
        """Total number of active connections across all storms."""
        return sum(len(conns) for conns in self._connections.values())


# Singleton instance
manager = ConnectionManager()
