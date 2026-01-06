"""
Dashboard Broadcaster - Helper module for trading bot to send real-time updates.

Usage in trading_bot.py:
    from dashboard.broadcaster import broadcast_decision, broadcast_kpi_update, broadcast_alert

    # After saving a decision
    await broadcast_decision(decision_data)

    # After a batch of decisions
    await broadcast_kpi_update()

    # On important events
    await broadcast_alert("Daily loss limit reached!", severity="critical")
"""
import os
import httpx
from typing import Dict, Any, Optional
from loguru import logger

# Dashboard API URL - configurable via environment variable for Render deployment
# Set DASHBOARD_URL to the Render web service URL (e.g., https://kalshi-trading-bot-le8x.onrender.com)
DASHBOARD_API_URL = os.getenv("DASHBOARD_URL", "http://localhost:8000")


async def broadcast_decision(decision: Dict[str, Any], base_url: str = DASHBOARD_API_URL) -> bool:
    """
    Broadcast a new betting decision to all connected dashboard clients.

    Args:
        decision: Dictionary with decision data (decision_id, action, market_title, etc.)
        base_url: Dashboard API URL (default: http://localhost:8000)

    Returns:
        True if broadcast succeeded, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{base_url}/api/broadcast/decision",
                json=decision
            )
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Decision broadcast to {result.get('sent_to', 0)} clients")
                return True
            else:
                logger.warning(f"Decision broadcast failed: {response.status_code}")
                return False
    except httpx.ConnectError:
        # Dashboard not running - this is expected in many cases
        logger.debug("Dashboard not available for broadcast")
        return False
    except Exception as e:
        logger.debug(f"Decision broadcast error: {e}")
        return False


async def broadcast_kpi_update(base_url: str = DASHBOARD_API_URL) -> bool:
    """
    Trigger a KPI recalculation and broadcast to all connected clients.

    Call this after saving decisions to update dashboard KPIs in real-time.

    Returns:
        True if broadcast succeeded, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(f"{base_url}/api/broadcast/kpis")
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"KPI broadcast to {result.get('sent_to', 0)} clients")
                return True
            else:
                logger.warning(f"KPI broadcast failed: {response.status_code}")
                return False
    except httpx.ConnectError:
        logger.debug("Dashboard not available for KPI broadcast")
        return False
    except Exception as e:
        logger.debug(f"KPI broadcast error: {e}")
        return False


async def broadcast_alert(
    message: str,
    severity: str = "info",
    details: Optional[Dict[str, Any]] = None,
    base_url: str = DASHBOARD_API_URL
) -> bool:
    """
    Broadcast an alert to all connected dashboard clients.

    Args:
        message: Alert message text
        severity: One of: "info", "warning", "error", "critical"
        details: Optional dictionary with additional alert data
        base_url: Dashboard API URL

    Returns:
        True if broadcast succeeded, False otherwise
    """
    alert_data = {
        "message": message,
        "severity": severity,
        "details": details or {}
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{base_url}/api/broadcast/alert",
                json=alert_data
            )
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Alert broadcast to {result.get('sent_to', 0)} clients")
                return True
            else:
                logger.warning(f"Alert broadcast failed: {response.status_code}")
                return False
    except httpx.ConnectError:
        logger.debug("Dashboard not available for alert broadcast")
        return False
    except Exception as e:
        logger.debug(f"Alert broadcast error: {e}")
        return False


async def broadcast_status(
    bot_running: bool,
    mode: str = "dry_run",
    additional_info: Optional[Dict[str, Any]] = None,
    base_url: str = DASHBOARD_API_URL
) -> bool:
    """
    Broadcast bot status update to all connected clients.

    Args:
        bot_running: Whether the bot is currently running
        mode: "dry_run" or "live"
        additional_info: Optional additional status information
        base_url: Dashboard API URL

    Returns:
        True if broadcast succeeded, False otherwise
    """
    status_data = {
        "bot_running": bot_running,
        "mode": mode,
        **(additional_info or {})
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{base_url}/api/broadcast/status",
                json=status_data
            )
            if response.status_code == 200:
                result = response.json()
                logger.debug(f"Status broadcast to {result.get('sent_to', 0)} clients")
                return True
            else:
                logger.warning(f"Status broadcast failed: {response.status_code}")
                return False
    except httpx.ConnectError:
        logger.debug("Dashboard not available for status broadcast")
        return False
    except Exception as e:
        logger.debug(f"Status broadcast error: {e}")
        return False


async def broadcast_workflow_step(
    step_number: int,
    step_name: str,
    status: str = "running",
    description: str = "",
    details: Optional[Dict[str, Any]] = None,
    base_url: str = DASHBOARD_API_URL
) -> bool:
    """
    Broadcast a workflow step update to the dashboard.

    Streams CLI progress to connected dashboard clients in real-time.

    Args:
        step_number: Step number (1-10)
        step_name: Short step name (e.g., "Fetching events")
        status: One of "running", "completed", "failed", "skipped"
        description: Optional longer description
        details: Optional dict with step-specific data (e.g., {"events_found": 25})
        base_url: Dashboard API URL

    Returns:
        True if broadcast succeeded, False otherwise
    """
    from datetime import datetime

    step_data = {
        "step_number": step_number,
        "step_name": step_name,
        "status": status,
        "description": description,
        "details": details or {},
        "timestamp": datetime.now().isoformat()
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{base_url}/api/broadcast/workflow",
                json=step_data
            )
            if response.status_code == 200:
                logger.debug(f"Workflow step {step_number} ({step_name}) broadcast")
                return True
            else:
                logger.warning(f"Workflow broadcast failed: {response.status_code}")
                return False
    except httpx.ConnectError:
        logger.debug("Dashboard not available for workflow broadcast")
        return False
    except Exception as e:
        logger.debug(f"Workflow broadcast error: {e}")
        return False


async def broadcast_cli_log(
    level: str,
    message: str,
    source: str = "bot",
    details: Optional[Dict[str, Any]] = None,
    base_url: str = DASHBOARD_API_URL
) -> bool:
    """
    Broadcast a CLI log entry to the dashboard.

    Args:
        level: Log level (info, warning, error, debug, success)
        message: Log message text
        source: Source of the log (bot, position_monitor, scheduler, etc.)
        details: Optional dictionary with additional log data
        base_url: Dashboard API URL

    Returns:
        True if broadcast succeeded, False otherwise
    """
    from datetime import datetime

    log_data = {
        "level": level,
        "message": message,
        "source": source,
        "details": details or {},
        "timestamp": datetime.now().isoformat()
    }

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{base_url}/api/broadcast/cli_log",
                json=log_data
            )
            if response.status_code == 200:
                return True
            else:
                return False
    except httpx.ConnectError:
        return False
    except Exception:
        return False


async def broadcast_account_update(
    account_data: Dict[str, Any],
    base_url: str = DASHBOARD_API_URL
) -> bool:
    """
    Broadcast real Kalshi account data to the dashboard.

    Args:
        account_data: Dictionary with balance, positions, P&L, etc.
        base_url: Dashboard API URL

    Returns:
        True if broadcast succeeded, False otherwise
    """
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                f"{base_url}/api/broadcast/account",
                json=account_data
            )
            if response.status_code == 200:
                return True
            else:
                return False
    except httpx.ConnectError:
        return False
    except Exception:
        return False


class DashboardBroadcaster:
    """
    Context manager for dashboard broadcasting.

    Usage:
        async with DashboardBroadcaster() as broadcaster:
            await broadcaster.decision({...})
            await broadcaster.alert("Warning!", severity="warning")
    """

    def __init__(self, base_url: str = DASHBOARD_API_URL, enabled: bool = True):
        self.base_url = base_url
        self.enabled = enabled
        self.decisions_sent = 0
        self.alerts_sent = 0

    async def __aenter__(self):
        if self.enabled:
            await broadcast_status(bot_running=True, base_url=self.base_url)
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.enabled:
            if exc_type:
                await broadcast_alert(
                    f"Bot stopped with error: {exc_val}",
                    severity="error",
                    base_url=self.base_url
                )
            await broadcast_status(bot_running=False, base_url=self.base_url)
            await broadcast_kpi_update(base_url=self.base_url)

    async def decision(self, decision_data: Dict[str, Any]) -> bool:
        if not self.enabled:
            return False
        result = await broadcast_decision(decision_data, base_url=self.base_url)
        if result:
            self.decisions_sent += 1
        return result

    async def kpis(self) -> bool:
        if not self.enabled:
            return False
        return await broadcast_kpi_update(base_url=self.base_url)

    async def alert(self, message: str, severity: str = "info", details: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        result = await broadcast_alert(message, severity, details, base_url=self.base_url)
        if result:
            self.alerts_sent += 1
        return result

    async def status(self, bot_running: bool, mode: str = "dry_run", **kwargs) -> bool:
        if not self.enabled:
            return False
        return await broadcast_status(bot_running, mode, kwargs, base_url=self.base_url)

    async def workflow(self, step_number: int, step_name: str, status: str = "running",
                       description: str = "", details: Optional[Dict[str, Any]] = None) -> bool:
        if not self.enabled:
            return False
        return await broadcast_workflow_step(step_number, step_name, status, description, details, base_url=self.base_url)
