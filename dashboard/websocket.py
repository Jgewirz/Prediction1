"""
WebSocket connection manager for real-time dashboard updates.

Handles:
- Client connection lifecycle
- Broadcasting decisions, KPIs, alerts
- Topic-based subscriptions
- Auto-reconnection support
"""
import asyncio
import json
from datetime import datetime
from typing import Set, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum

from fastapi import WebSocket, WebSocketDisconnect
from loguru import logger


class MessageType(str, Enum):
    """WebSocket message types."""
    DECISION = "decision"
    KPI_UPDATE = "kpi_update"
    ALERT = "alert"
    STATUS = "status"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    CONNECTED = "connected"
    SUBSCRIBED = "subscribed"


@dataclass
class WebSocketMessage:
    """Structured WebSocket message."""
    type: MessageType
    data: Dict[str, Any]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_json(self) -> str:
        return json.dumps({
            "type": self.type.value,
            "data": self.data,
            "timestamp": self.timestamp
        })


class ConnectionManager:
    """
    Manages WebSocket connections for real-time dashboard updates.

    Features:
    - Multiple concurrent client connections
    - Topic-based subscriptions (decisions, kpis, alerts)
    - Heartbeat for connection health
    - Thread-safe broadcasting
    """

    def __init__(self):
        # All active WebSocket connections
        self.active_connections: Set[WebSocket] = set()

        # Topic subscriptions: topic -> set of websockets
        self.subscriptions: Dict[str, Set[WebSocket]] = {
            "decisions": set(),
            "kpis": set(),
            "alerts": set(),
            "status": set(),
            "all": set()  # Subscribers to everything
        }

        # Connection metadata
        self.connection_info: Dict[WebSocket, Dict[str, Any]] = {}

        # Lock for thread-safe operations
        self._lock = asyncio.Lock()

        # Heartbeat interval (seconds)
        self.heartbeat_interval = 30

    async def connect(self, websocket: WebSocket, client_id: Optional[str] = None) -> None:
        """Accept a new WebSocket connection."""
        await websocket.accept()

        async with self._lock:
            self.active_connections.add(websocket)
            self.connection_info[websocket] = {
                "client_id": client_id or f"client_{id(websocket)}",
                "connected_at": datetime.utcnow().isoformat(),
                "subscriptions": set()
            }

        # Send welcome message
        welcome = WebSocketMessage(
            type=MessageType.CONNECTED,
            data={
                "message": "Connected to Kalshi Trading Bot Dashboard",
                "client_id": self.connection_info[websocket]["client_id"],
                "available_topics": list(self.subscriptions.keys())
            }
        )
        await websocket.send_text(welcome.to_json())

        logger.info(f"WebSocket client connected: {self.connection_info[websocket]['client_id']}")

    async def disconnect(self, websocket: WebSocket) -> None:
        """Handle client disconnection."""
        async with self._lock:
            self.active_connections.discard(websocket)

            # Remove from all subscriptions
            for topic_subscribers in self.subscriptions.values():
                topic_subscribers.discard(websocket)

            # Get client ID for logging
            client_id = self.connection_info.get(websocket, {}).get("client_id", "unknown")
            self.connection_info.pop(websocket, None)

        logger.info(f"WebSocket client disconnected: {client_id}")

    async def subscribe(self, websocket: WebSocket, topics: list[str]) -> None:
        """Subscribe a client to specific topics."""
        async with self._lock:
            for topic in topics:
                if topic in self.subscriptions:
                    self.subscriptions[topic].add(websocket)
                    if websocket in self.connection_info:
                        self.connection_info[websocket]["subscriptions"].add(topic)

        # Confirm subscription
        confirm = WebSocketMessage(
            type=MessageType.SUBSCRIBED,
            data={"topics": topics}
        )
        try:
            await websocket.send_text(confirm.to_json())
        except Exception:
            pass

    async def unsubscribe(self, websocket: WebSocket, topics: list[str]) -> None:
        """Unsubscribe a client from specific topics."""
        async with self._lock:
            for topic in topics:
                if topic in self.subscriptions:
                    self.subscriptions[topic].discard(websocket)
                    if websocket in self.connection_info:
                        self.connection_info[websocket]["subscriptions"].discard(topic)

    async def broadcast(self, message: WebSocketMessage, topic: str = "all") -> int:
        """
        Broadcast a message to all clients subscribed to a topic.

        Returns the number of clients that received the message.
        """
        sent_count = 0

        async with self._lock:
            # Get subscribers for this topic + "all" subscribers
            subscribers = self.subscriptions.get(topic, set()) | self.subscriptions.get("all", set())

            if not subscribers:
                return 0

            # Send to all subscribers (copy set to avoid modification during iteration)
            for websocket in list(subscribers):
                try:
                    await websocket.send_text(message.to_json())
                    sent_count += 1
                except Exception as e:
                    logger.debug(f"Failed to send to client: {e}")
                    # Client may have disconnected, remove from all subscriptions
                    self.active_connections.discard(websocket)
                    for topic_subscribers in self.subscriptions.values():
                        topic_subscribers.discard(websocket)

        return sent_count

    async def broadcast_decision(self, decision: Dict[str, Any]) -> int:
        """Broadcast a new betting decision to subscribers."""
        message = WebSocketMessage(
            type=MessageType.DECISION,
            data=decision
        )
        return await self.broadcast(message, topic="decisions")

    async def broadcast_kpi_update(self, kpis: Dict[str, Any]) -> int:
        """Broadcast updated KPIs to subscribers."""
        message = WebSocketMessage(
            type=MessageType.KPI_UPDATE,
            data=kpis
        )
        return await self.broadcast(message, topic="kpis")

    async def broadcast_alert(self, alert: Dict[str, Any]) -> int:
        """Broadcast an alert to subscribers."""
        message = WebSocketMessage(
            type=MessageType.ALERT,
            data=alert
        )
        return await self.broadcast(message, topic="alerts")

    async def broadcast_status(self, status: Dict[str, Any]) -> int:
        """Broadcast bot status update."""
        message = WebSocketMessage(
            type=MessageType.STATUS,
            data=status
        )
        return await self.broadcast(message, topic="status")

    async def send_heartbeat(self) -> None:
        """Send heartbeat to all connected clients."""
        message = WebSocketMessage(
            type=MessageType.HEARTBEAT,
            data={"connections": len(self.active_connections)}
        )
        await self.broadcast(message, topic="all")

    async def heartbeat_loop(self) -> None:
        """Background task to send periodic heartbeats."""
        while True:
            await asyncio.sleep(self.heartbeat_interval)
            if self.active_connections:
                await self.send_heartbeat()

    def get_connection_count(self) -> int:
        """Get the number of active connections."""
        return len(self.active_connections)

    def get_stats(self) -> Dict[str, Any]:
        """Get WebSocket connection statistics."""
        return {
            "total_connections": len(self.active_connections),
            "subscriptions": {
                topic: len(subscribers)
                for topic, subscribers in self.subscriptions.items()
            },
            "clients": [
                {
                    "client_id": info.get("client_id"),
                    "connected_at": info.get("connected_at"),
                    "subscriptions": list(info.get("subscriptions", []))
                }
                for info in self.connection_info.values()
            ]
        }


# Global connection manager instance
manager = ConnectionManager()


async def handle_websocket(websocket: WebSocket) -> None:
    """
    Handle a WebSocket connection lifecycle.

    Message protocol:
    - Client sends: {"action": "subscribe", "topics": ["decisions", "kpis"]}
    - Client sends: {"action": "unsubscribe", "topics": ["alerts"]}
    - Client sends: {"action": "ping"}
    - Server sends: WebSocketMessage objects
    """
    await manager.connect(websocket)

    try:
        while True:
            # Wait for client messages
            data = await websocket.receive_text()

            try:
                message = json.loads(data)
                action = message.get("action")

                if action == "subscribe":
                    topics = message.get("topics", ["all"])
                    await manager.subscribe(websocket, topics)
                    logger.debug(f"Client subscribed to: {topics}")

                elif action == "unsubscribe":
                    topics = message.get("topics", [])
                    await manager.unsubscribe(websocket, topics)
                    logger.debug(f"Client unsubscribed from: {topics}")

                elif action == "ping":
                    pong = WebSocketMessage(
                        type=MessageType.HEARTBEAT,
                        data={"pong": True}
                    )
                    await websocket.send_text(pong.to_json())

            except json.JSONDecodeError:
                error = WebSocketMessage(
                    type=MessageType.ERROR,
                    data={"message": "Invalid JSON"}
                )
                await websocket.send_text(error.to_json())

    except WebSocketDisconnect:
        await manager.disconnect(websocket)
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
        await manager.disconnect(websocket)
