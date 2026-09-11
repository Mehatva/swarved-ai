import asyncio
import json
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect


router = APIRouter(
    tags=["Threat Analytics"],
)


class ConnectionManager:
    """
    Manage active WebSocket connections for the live threat feed.
    """

    def __init__(self) -> None:
        self.active_connections: set[WebSocket] = set()
        self._lock = asyncio.Lock()

    async def connect(self, websocket: WebSocket) -> None:
        """
        Accept and register a new WebSocket connection.
        """
        await websocket.accept()

        async with self._lock:
            self.active_connections.add(websocket)

    async def disconnect(self, websocket: WebSocket) -> None:
        """
        Remove a WebSocket connection from the active connection pool.
        """
        async with self._lock:
            self.active_connections.discard(websocket)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """
        Broadcast a JSON message to all currently connected clients.

        Failed connections are removed automatically.
        """
        async with self._lock:
            connections = list(self.active_connections)

        if not connections:
            return

        payload = json.dumps(message, default=str)

        results = await asyncio.gather(
            *(
                connection.send_text(payload)
                for connection in connections
            ),
            return_exceptions=True,
        )

        failed_connections = [
            connection
            for connection, result in zip(connections, results)
            if isinstance(result, Exception)
        ]

        if failed_connections:
            async with self._lock:
                for connection in failed_connections:
                    self.active_connections.discard(connection)


connection_manager = ConnectionManager()


@router.websocket("/ws/threat-feed")
async def threat_feed(websocket: WebSocket) -> None:
    """
    WebSocket endpoint for the live threat analytics dashboard.
    """
    await connection_manager.connect(websocket)

    try:
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        await connection_manager.disconnect(websocket)

    except Exception:
        await connection_manager.disconnect(websocket)