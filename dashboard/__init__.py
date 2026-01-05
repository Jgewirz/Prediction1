"""Dashboard package for Kalshi Trading Bot.

Real-time monitoring dashboard with WebSocket support.

Usage:
    # Start the dashboard server
    python -m dashboard.api

    # Broadcast from trading bot
    from dashboard.broadcaster import broadcast_decision, broadcast_kpi_update
"""
from dashboard.broadcaster import (
    broadcast_decision,
    broadcast_kpi_update,
    broadcast_alert,
    broadcast_status,
    DashboardBroadcaster
)

__all__ = [
    "broadcast_decision",
    "broadcast_kpi_update",
    "broadcast_alert",
    "broadcast_status",
    "DashboardBroadcaster"
]
