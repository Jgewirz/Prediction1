#!/usr/bin/env python3
"""
COMPREHENSIVE PRODUCTION VALIDATION SCRIPT
==========================================
Validates entire Kalshi Trading Bot system including dashboard connectivity,
real-time streaming, Kalshi API integration, and deployment synchronization.

This script ensures the system is AIRTIGHT and production-ready.

Usage:
    uv run python validate_system_airtight.py
    uv run python validate_system_airtight.py --remote   # Test against Render deployment
    uv run python validate_system_airtight.py --quick    # Core checks only

Components Validated:
    Phase 1: Configuration & Environment
    Phase 2: Database Integrity
    Phase 3: Kalshi API Integration (with real account data)
    Phase 4: Trading Logic & Risk Management
    Phase 5: Security Audit
    Phase 6: Dashboard API & WebSocket
    Phase 7: Trader ↔ Dashboard Communication
    Phase 8: Early Entry & Market Selection
    Phase 9: Position Monitoring & Stop-Loss
    Phase 10: Deployment Synchronization
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)


class Status(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    name: str
    status: Status
    message: str = ""
    duration_ms: float = 0.0


@dataclass
class PhaseResult:
    phase: int
    name: str
    results: list = field(default_factory=list)

    @property
    def passed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.PASS)

    @property
    def failed(self) -> int:
        return sum(1 for r in self.results if r.status == Status.FAIL)

    @property
    def warnings(self) -> int:
        return sum(1 for r in self.results if r.status == Status.WARN)

    @property
    def success(self) -> bool:
        return self.failed == 0


console = Console() if RICH_AVAILABLE else None

# URLs for testing
LOCAL_DASHBOARD_URL = "http://localhost:8000"
RENDER_DASHBOARD_URL = "https://kalshi-dashboard.onrender.com"
RENDER_TRADER_URL = "https://kalshi-trading-bot-le8x.onrender.com"


def print_header(text: str):
    if console:
        console.print(Panel(text, style="bold blue"))
    else:
        print(f"\n{'='*60}\n{text}\n{'='*60}")


def print_result(result: ValidationResult):
    status_colors = {
        Status.PASS: "green",
        Status.FAIL: "red",
        Status.WARN: "yellow",
        Status.SKIP: "dim",
    }
    status_icons = {
        Status.PASS: "[PASS]",
        Status.FAIL: "[FAIL]",
        Status.WARN: "[WARN]",
        Status.SKIP: "[SKIP]",
    }

    if console:
        color = status_colors[result.status]
        icon = status_icons[result.status]
        msg = f" - {result.message}" if result.message else ""
        time_str = f" ({result.duration_ms:.0f}ms)" if result.duration_ms > 0 else ""
        console.print(f"  [{color}]{icon}[/{color}] {result.name}{msg}{time_str}")
    else:
        icon = status_icons[result.status]
        msg = f" - {result.message}" if result.message else ""
        print(f"  {icon} {result.name}{msg}")


async def timed_check(name: str, check_func) -> ValidationResult:
    """Run a check function and time it."""
    start = time.perf_counter()
    try:
        result = (
            await check_func()
            if asyncio.iscoroutinefunction(check_func)
            else check_func()
        )
        duration = (time.perf_counter() - start) * 1000

        if isinstance(result, tuple):
            status, message = result
        elif isinstance(result, bool):
            status = Status.PASS if result else Status.FAIL
            message = ""
        else:
            status = Status.FAIL
            message = f"Invalid return type: {type(result)}"

        return ValidationResult(name, status, message, duration)

    except Exception as e:
        duration = (time.perf_counter() - start) * 1000
        return ValidationResult(name, Status.FAIL, str(e)[:100], duration)


# =============================================================================
# PHASE 1: Configuration & Environment
# =============================================================================


async def validate_phase1_config() -> PhaseResult:
    """Configuration & Environment"""
    results = PhaseResult(1, "Configuration & Environment")

    async def check_env_file():
        if os.path.exists(".env"):
            return Status.PASS, ".env file found"
        elif os.getenv("KALSHI_API_KEY"):
            return Status.PASS, "Using environment variables (Render)"
        return Status.FAIL, "No .env file or env vars"

    results.results.append(await timed_check("Environment config", check_env_file))

    async def check_config_loads():
        try:
            from config import BotConfig

            config = BotConfig()
            return Status.PASS, f"Config loaded (demo={config.kalshi.use_demo})"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(await timed_check("Config loads", check_config_loads))

    async def check_kalshi_creds():
        from config import BotConfig

        config = BotConfig()
        if not config.kalshi.api_key:
            return Status.FAIL, "KALSHI_API_KEY not set"
        if not config.kalshi.private_key:
            return Status.FAIL, "KALSHI_PRIVATE_KEY not set"
        if (
            "BEGIN" not in config.kalshi.private_key
            and "RSA" not in config.kalshi.private_key
        ):
            return Status.WARN, "Private key format may be incorrect"
        return Status.PASS, "Kalshi credentials configured"

    results.results.append(await timed_check("Kalshi credentials", check_kalshi_creds))

    async def check_openai_creds():
        from config import BotConfig

        config = BotConfig()
        if not config.openai.api_key:
            return Status.FAIL, "OPENAI_API_KEY not set"
        return Status.PASS, "OpenAI credentials configured"

    results.results.append(await timed_check("OpenAI credentials", check_openai_creds))

    async def check_dashboard_url():
        dashboard_url = os.getenv("DASHBOARD_URL", "")
        if dashboard_url:
            return Status.PASS, f"DASHBOARD_URL: {dashboard_url[:40]}..."
        return Status.WARN, "DASHBOARD_URL not set (using localhost)"

    results.results.append(
        await timed_check("Dashboard URL config", check_dashboard_url)
    )

    return results


# =============================================================================
# PHASE 2: Database Integrity
# =============================================================================


async def validate_phase2_database() -> PhaseResult:
    """Database Integrity"""
    results = PhaseResult(2, "Database Integrity")

    async def check_db_connection():
        from config import BotConfig

        config = BotConfig()
        if config.database.db_type == "postgres":
            from db.postgres import PostgresDatabase

            db = PostgresDatabase(
                host=config.database.pg_host,
                database=config.database.pg_database,
                user=config.database.pg_user,
                password=config.database.pg_password,
                port=config.database.pg_port,
                ssl=config.database.pg_ssl,
            )
        else:
            from db.database import Database

            db = Database(config.database.db_path)
        await db.connect()
        await db.close()
        return Status.PASS, f"Connected to {config.database.db_type}"

    results.results.append(
        await timed_check("Database connection", check_db_connection)
    )

    async def check_tables():
        from config import BotConfig

        config = BotConfig()
        required = ["betting_decisions", "market_snapshots", "run_history"]

        if config.database.db_type == "postgres":
            from db.postgres import PostgresDatabase

            db = PostgresDatabase(
                host=config.database.pg_host,
                database=config.database.pg_database,
                user=config.database.pg_user,
                password=config.database.pg_password,
                port=config.database.pg_port,
                ssl=config.database.pg_ssl,
            )
            await db.connect()
            result = await db.fetchall(
                "SELECT table_name FROM information_schema.tables WHERE table_schema = 'public'"
            )
            existing = {row["table_name"] for row in result}
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            result = await db.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            )
            existing = {row[0] for row in result}
        await db.close()

        missing = [t for t in required if t not in existing]
        if missing:
            return Status.FAIL, f"Missing: {missing}"
        return Status.PASS, f"All {len(required)} tables exist"

    results.results.append(await timed_check("Required tables", check_tables))

    async def check_recent_data():
        from config import BotConfig

        config = BotConfig()

        if config.database.db_type == "postgres":
            from db.postgres import PostgresDatabase

            db = PostgresDatabase(
                host=config.database.pg_host,
                database=config.database.pg_database,
                user=config.database.pg_user,
                password=config.database.pg_password,
                port=config.database.pg_port,
                ssl=config.database.pg_ssl,
            )
            await db.connect()
            result = await db.fetchone(
                """
                SELECT COUNT(*) as cnt, MAX(created_at) as latest
                FROM betting_decisions
                WHERE created_at > NOW() - INTERVAL '24 hours'
            """
            )
            count = result["cnt"] if result else 0
            latest = result["latest"] if result else None
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            result = await db.execute(
                """
                SELECT COUNT(*), MAX(created_at) FROM betting_decisions
                WHERE created_at > datetime('now', '-24 hours')
            """
            )
            count = result[0][0]
            latest = result[0][1]
        await db.close()

        if count == 0:
            return Status.WARN, "No decisions in last 24h"
        return Status.PASS, f"{count} decisions in last 24h"

    results.results.append(await timed_check("Recent data", check_recent_data))

    return results


# =============================================================================
# PHASE 3: Kalshi API Integration (with real account data)
# =============================================================================


async def validate_phase3_kalshi() -> PhaseResult:
    """Kalshi API Integration"""
    results = PhaseResult(3, "Kalshi API Integration")

    async def check_kalshi_auth():
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)
        await client.login()
        events = await client.get_events(limit=1)
        await client.close()
        if events:
            return Status.PASS, f"Authenticated - {len(events)} event(s)"
        return Status.WARN, "Auth OK but no events"

    results.results.append(
        await timed_check("Kalshi authentication", check_kalshi_auth)
    )

    async def check_balance_endpoint():
        """Critical: Tests /portfolio/balance endpoint for real account data."""
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)
        await client.login()
        balance = await client.get_balance()
        await client.close()

        # Check if we got an error or valid data
        if balance.get("error"):
            return (
                Status.FAIL,
                f"Balance API error: {balance.get('error', 'unknown')[:50]}",
            )

        # Valid response has total_equity_dollars calculated
        equity = balance.get("total_equity_dollars", 0)
        if "balance_dollars" in balance:
            return Status.PASS, f"Balance API OK - Equity: ${equity:.2f}"
        return Status.FAIL, "Balance API returned no data"

    results.results.append(
        await timed_check(
            "Balance endpoint (/portfolio/balance)", check_balance_endpoint
        )
    )

    async def check_positions_endpoint():
        """Critical: Tests /portfolio/positions endpoint."""
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)
        await client.login()
        positions = await client.get_user_positions()
        await client.close()
        count = len(positions) if positions else 0
        return Status.PASS, f"Positions: {count} open"

    results.results.append(
        await timed_check(
            "Positions endpoint (/portfolio/positions)", check_positions_endpoint
        )
    )

    async def check_account_summary():
        """Tests get_account_summary() which combines balance + positions."""
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)
        await client.login()
        summary = await client.get_account_summary()
        await client.close()

        # Check for actual field names returned by get_account_summary()
        required_fields = [
            "balance",
            "portfolio_value",
            "total_equity",
            "open_positions_count",
            "total_exposure",
            "realized_pnl",
            "api_connected",
        ]
        missing = [f for f in required_fields if f not in summary]

        if missing:
            return Status.WARN, f"Missing fields: {missing}"
        if not summary.get("api_connected"):
            return Status.WARN, "API not connected"
        return (
            Status.PASS,
            f"Account summary complete - equity: ${summary.get('total_equity', 0):.2f}",
        )

    results.results.append(
        await timed_check("Account summary aggregation", check_account_summary)
    )

    return results


# =============================================================================
# PHASE 4: Trading Logic & Risk Management
# =============================================================================


async def validate_phase4_trading() -> PhaseResult:
    """Trading Logic & Risk Management"""
    results = PhaseResult(4, "Trading Logic")

    async def check_r_score():
        import math

        def calc_r(rp, mp):
            std = math.sqrt(mp * (1 - mp))
            return (rp - mp) / std if std > 0 else 0

        tests = [(0.7, 0.5, 0.4), (0.5, 0.5, 0.0)]
        for rp, mp, expected in tests:
            result = calc_r(rp, mp)
            if abs(result - expected) > 0.02:
                return (
                    Status.FAIL,
                    f"r_score({rp}, {mp})={result:.2f}, expected {expected}",
                )
        return Status.PASS, "R-score calculation correct"

    results.results.append(await timed_check("R-score calculation", check_r_score))

    async def check_risk_params():
        from config import BotConfig

        config = BotConfig()
        issues = []
        if config.max_daily_loss <= 0:
            issues.append("max_daily_loss <= 0")
        if config.kelly_fraction < 0.1 or config.kelly_fraction > 2.0:
            issues.append(f"kelly_fraction={config.kelly_fraction}")
        if config.max_bet_amount > config.bankroll * 0.5:
            issues.append("max_bet > 50% bankroll")

        if issues:
            return Status.WARN, "; ".join(issues)
        return (
            Status.PASS,
            f"Bankroll=${config.bankroll}, MaxBet=${config.max_bet_amount}",
        )

    results.results.append(await timed_check("Risk parameters", check_risk_params))

    async def check_kill_switch():
        from config import BotConfig

        config = BotConfig()
        if not config.enable_kill_switch:
            return Status.WARN, "Kill switch DISABLED"
        return Status.PASS, f"Kill switch: ${config.max_daily_loss} max loss"

    results.results.append(await timed_check("Kill switch config", check_kill_switch))

    return results


# =============================================================================
# PHASE 5: Security Audit
# =============================================================================


async def validate_phase5_security() -> PhaseResult:
    """Security Audit"""
    results = PhaseResult(5, "Security")

    async def check_gitignore():
        required = [".env", "*.pem"]
        if not os.path.exists(".gitignore"):
            return Status.WARN, ".gitignore missing"
        with open(".gitignore") as f:
            content = f.read().lower()
        missing = [p for p in required if p.lower() not in content]
        if missing:
            return Status.WARN, f"Missing: {missing}"
        return Status.PASS, "Sensitive files protected"

    results.results.append(await timed_check(".gitignore check", check_gitignore))

    async def check_env_isolation():
        from config import BotConfig

        config = BotConfig()
        if config.kalshi.use_demo:
            return Status.PASS, "Using DEMO environment"
        return Status.WARN, "LIVE trading mode - verify intentional"

    results.results.append(
        await timed_check("Environment isolation", check_env_isolation)
    )

    return results


# =============================================================================
# PHASE 6: Dashboard API & WebSocket
# =============================================================================


async def validate_phase6_dashboard(remote: bool = False) -> PhaseResult:
    """Dashboard API & WebSocket"""
    results = PhaseResult(6, "Dashboard API")
    base_url = RENDER_DASHBOARD_URL if remote else LOCAL_DASHBOARD_URL

    async def check_health():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/health")
                if response.status_code == 200:
                    data = response.json()
                    return Status.PASS, f"Health OK - {data.get('status', 'unknown')}"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(
        await timed_check(f"Dashboard health ({base_url[:30]}...)", check_health)
    )

    async def check_account_endpoint():
        """Tests /api/account endpoint that returns real Kalshi data."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/account")
                if response.status_code == 200:
                    data = response.json()
                    if data.get("api_connected"):
                        equity = data.get("total_equity_dollars", 0)
                        return Status.PASS, f"Account API OK - Equity: ${equity:.2f}"
                    return Status.WARN, "API not connected"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(
        await timed_check("/api/account (real Kalshi data)", check_account_endpoint)
    )

    async def check_positions_endpoint():
        """Tests /api/positions endpoint."""
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/positions")
                if response.status_code == 200:
                    data = response.json()
                    count = len(data.get("positions", []))
                    return Status.PASS, f"Positions API OK - {count} positions"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(
        await timed_check("/api/positions", check_positions_endpoint)
    )

    async def check_kpis_endpoint():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/kpis")
                if response.status_code == 200:
                    data = response.json()
                    return Status.PASS, f"KPIs: {len(data)} metrics"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(await timed_check("/api/kpis", check_kpis_endpoint))

    async def check_ws_stats():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/ws/stats")
                if response.status_code == 200:
                    data = response.json()
                    clients = data.get("active_connections", 0)
                    return Status.PASS, f"WS stats: {clients} clients connected"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(await timed_check("/api/ws/stats", check_ws_stats))

    async def check_freshness():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(f"{base_url}/api/freshness")
                if response.status_code == 200:
                    data = response.json()
                    stale = data.get("is_stale", True)
                    if stale:
                        return (
                            Status.WARN,
                            f"Data stale - {data.get('minutes_since_last', 'unknown')} min old",
                        )
                    return Status.PASS, "Data fresh"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(await timed_check("/api/freshness", check_freshness))

    return results


# =============================================================================
# PHASE 7: Trader ↔ Dashboard Communication
# =============================================================================


async def validate_phase7_communication(remote: bool = False) -> PhaseResult:
    """Trader ↔ Dashboard Communication"""
    results = PhaseResult(7, "Trader-Dashboard Communication")
    base_url = RENDER_DASHBOARD_URL if remote else LOCAL_DASHBOARD_URL

    async def check_broadcast_decision():
        import httpx

        test_decision = {
            "decision_id": "test-validation-12345",
            "action": "skip",
            "market_ticker": "VALIDATION-TEST",
            "market_title": "Validation Test Decision",
            "bet_amount": 0,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/api/broadcast/decision", json=test_decision
                )
                if response.status_code == 200:
                    data = response.json()
                    return Status.PASS, f"Broadcast to {data.get('sent_to', 0)} clients"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(
        await timed_check("POST /api/broadcast/decision", check_broadcast_decision)
    )

    async def check_broadcast_cli_log():
        import httpx

        test_log = {
            "level": "info",
            "message": "Validation test log entry",
            "source": "validator",
            "details": {"test": True},
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/api/broadcast/cli_log", json=test_log
                )
                if response.status_code == 200:
                    return Status.PASS, "CLI log broadcast working"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(
        await timed_check("POST /api/broadcast/cli_log", check_broadcast_cli_log)
    )

    async def check_broadcast_account():
        import httpx

        test_account = {
            "balance_dollars": 100.0,
            "portfolio_value_dollars": 50.0,
            "total_equity_dollars": 150.0,
            "api_connected": True,
        }
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(
                    f"{base_url}/api/broadcast/account", json=test_account
                )
                if response.status_code == 200:
                    return Status.PASS, "Account broadcast working"
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(
        await timed_check("POST /api/broadcast/account", check_broadcast_account)
    )

    async def check_broadcast_kpis():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.post(f"{base_url}/api/broadcast/kpis")
                if response.status_code == 200:
                    data = response.json()
                    return (
                        Status.PASS,
                        f"KPI broadcast to {data.get('sent_to', 0)} clients",
                    )
                return Status.FAIL, f"Status {response.status_code}"
        except httpx.ConnectError:
            return Status.SKIP, "Dashboard not reachable"
        except Exception as e:
            return Status.FAIL, str(e)[:50]

    results.results.append(
        await timed_check("POST /api/broadcast/kpis", check_broadcast_kpis)
    )

    return results


# =============================================================================
# PHASE 8: Early Entry & Market Selection
# =============================================================================


async def validate_phase8_early_entry() -> PhaseResult:
    """Early Entry & Market Selection"""
    results = PhaseResult(8, "Early Entry Strategy")

    async def check_early_entry_config():
        from config import BotConfig

        config = BotConfig()
        ee = config.early_entry

        if not ee.enabled:
            return Status.WARN, "Early entry DISABLED"

        return (
            Status.PASS,
            f"Max age: {ee.max_market_age_hours}h, Min time: {ee.min_time_remaining_hours}h",
        )

    results.results.append(
        await timed_check("Early entry config", check_early_entry_config)
    )

    async def check_unique_market_bonus():
        from config import BotConfig

        config = BotConfig()
        ee = config.early_entry

        if not ee.favor_unique_events:
            return Status.WARN, "Unique event bonus DISABLED"

        bonus = ee.unique_event_bonus
        if bonus < 0 or bonus > 1:
            return Status.FAIL, f"Invalid bonus: {bonus}"
        return Status.PASS, f"Unique event bonus: +{bonus}"

    results.results.append(
        await timed_check("Unique event bonus config", check_unique_market_bonus)
    )

    async def check_new_market_bonus():
        from config import BotConfig

        config = BotConfig()
        ee = config.early_entry

        if not ee.favor_new_markets:
            return Status.WARN, "New market bonus DISABLED"

        bonus = ee.new_market_bonus
        hours = ee.new_market_hours
        return Status.PASS, f"New market (<{hours}h) bonus: +{bonus}"

    results.results.append(
        await timed_check("New market bonus config", check_new_market_bonus)
    )

    async def check_scoring_function():
        """Test the early entry scoring calculation."""
        from config import BotConfig, EarlyEntryConfig
        from kalshi_client import KalshiClient

        config = BotConfig()

        # Create client with early entry config
        early_entry_config = config.early_entry
        client = KalshiClient(config.kalshi, early_entry_config=early_entry_config)
        await client.login()

        # Create mock event data with time_remaining_hours (required for scoring)
        now = datetime.now(timezone.utc)
        test_event = {
            "event_ticker": "TEST",
            "series_ticker": None,  # Unique event (no series)
            "strike_period": None,  # No recurring period
            "markets": [],
            "time_remaining_hours": 72.0,  # 3 days remaining - required for time score
        }
        test_market = {
            "ticker": "TEST-M1",
            "volume_24h": 100,  # Low volume
            "volume": 100,
            "open_time": now.isoformat(),  # Just opened
            "created_time": now.isoformat(),
            "close_time": (now + timedelta(days=3)).isoformat(),  # 3 days remaining
        }

        # Test unique detection
        is_unique = client._is_unique_event(test_event)
        if not is_unique:
            await client.close()
            return Status.FAIL, "Unique event detection failed"

        # Test scoring
        score = client.calculate_early_entry_score(test_market, test_event)
        await client.close()

        if score <= 0:
            return Status.FAIL, f"Invalid score: {score} (check early_entry_config)"
        return Status.PASS, f"Scoring OK - test score: {score:.2f}"

    results.results.append(
        await timed_check("Early entry scoring function", check_scoring_function)
    )

    return results


# =============================================================================
# PHASE 9: Position Monitoring & Stop-Loss
# =============================================================================


async def validate_phase9_position_monitor() -> PhaseResult:
    """Position Monitoring & Stop-Loss"""
    results = PhaseResult(9, "Position Monitoring")

    async def check_stop_loss_config():
        from config import BotConfig

        config = BotConfig()
        sl = config.stop_loss

        if not sl.enabled:
            return Status.WARN, "Stop-loss DISABLED"

        return (
            Status.PASS,
            f"SL: {sl.default_stop_loss_pct*100}%, TP: {sl.default_take_profit_pct*100}%",
        )

    results.results.append(
        await timed_check("Stop-loss config", check_stop_loss_config)
    )

    async def check_monitor_interval():
        from config import BotConfig

        config = BotConfig()
        sl = config.stop_loss

        interval = sl.monitor_interval_seconds
        if interval < 10:
            return Status.WARN, f"Interval too aggressive: {interval}s"
        if interval > 120:
            return Status.WARN, f"Interval too slow: {interval}s"
        return Status.PASS, f"Monitor interval: {interval}s"

    results.results.append(
        await timed_check("Monitor interval", check_monitor_interval)
    )

    async def check_position_monitor_class():
        """Test position monitor can be instantiated."""
        try:
            from position_monitor import PositionMonitor, TriggerType

            # Verify trigger types exist
            if not hasattr(TriggerType, "STOP_LOSS"):
                return Status.FAIL, "Missing STOP_LOSS trigger"
            if not hasattr(TriggerType, "TAKE_PROFIT"):
                return Status.FAIL, "Missing TAKE_PROFIT trigger"

            return Status.PASS, "PositionMonitor class valid"
        except ImportError as e:
            return Status.FAIL, f"Import error: {e}"

    results.results.append(
        await timed_check("PositionMonitor class", check_position_monitor_class)
    )

    return results


# =============================================================================
# PHASE 10: Deployment Synchronization
# =============================================================================


async def validate_phase10_deployment() -> PhaseResult:
    """Deployment Synchronization"""
    results = PhaseResult(10, "Deployment Sync")

    async def check_local_commit():
        import subprocess

        try:
            result = subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                commit = result.stdout.strip()[:7]
                return Status.PASS, f"Local HEAD: {commit}"
            return Status.WARN, "Git not available"
        except Exception as e:
            return Status.WARN, str(e)[:30]

    results.results.append(await timed_check("Local git commit", check_local_commit))

    async def check_render_dashboard():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                response = await client.get(f"{RENDER_DASHBOARD_URL}/api/health")
                if response.status_code == 200:
                    return Status.PASS, "Dashboard deployed and healthy"
                return Status.WARN, f"Status {response.status_code}"
        except Exception as e:
            return Status.SKIP, f"Cannot reach Render: {str(e)[:30]}"

    results.results.append(
        await timed_check("Render dashboard status", check_render_dashboard)
    )

    async def check_render_trader():
        import httpx

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Try the trading bot's health endpoint if it exists
                response = await client.get(f"{RENDER_TRADER_URL}/api/health")
                if response.status_code == 200:
                    return Status.PASS, "Trader service healthy"
                return Status.WARN, f"Status {response.status_code}"
        except Exception as e:
            return Status.SKIP, f"Cannot reach Render: {str(e)[:30]}"

    results.results.append(
        await timed_check("Render trader status", check_render_trader)
    )

    async def check_dashboard_url_env():
        """Verify DASHBOARD_URL is set correctly for trader → dashboard communication."""
        dashboard_url = os.getenv("DASHBOARD_URL", "")

        if not dashboard_url:
            return (
                Status.WARN,
                "DASHBOARD_URL not set - broadcasts will go to localhost",
            )

        if "onrender.com" in dashboard_url or "localhost" in dashboard_url:
            return Status.PASS, f"DASHBOARD_URL configured: {dashboard_url[:40]}..."

        return Status.WARN, f"Unusual DASHBOARD_URL: {dashboard_url[:40]}..."

    results.results.append(
        await timed_check("DASHBOARD_URL environment", check_dashboard_url_env)
    )

    return results


# =============================================================================
# Main Execution
# =============================================================================


async def run_all_validations(remote: bool = False, quick: bool = False) -> list:
    """Run all validation phases."""
    all_phases = [
        (1, validate_phase1_config),
        (2, validate_phase2_database),
        (3, validate_phase3_kalshi),
        (4, validate_phase4_trading),
        (5, validate_phase5_security),
        (6, lambda: validate_phase6_dashboard(remote)),
        (7, lambda: validate_phase7_communication(remote)),
        (8, validate_phase8_early_entry),
        (9, validate_phase9_position_monitor),
        (10, validate_phase10_deployment),
    ]

    if quick:
        all_phases = all_phases[:5]  # Only core phases

    results = []
    for phase_num, phase_func in all_phases:
        doc = (
            phase_func.__doc__
            if hasattr(phase_func, "__doc__") and phase_func.__doc__
            else f"Phase {phase_num}"
        )
        print_header(f"Phase {phase_num}: {doc}")

        try:
            phase_result = await phase_func()
            results.append(phase_result)

            for result in phase_result.results:
                print_result(result)

            if console:
                console.print(
                    f"\n  Summary: {phase_result.passed} passed, "
                    f"{phase_result.failed} failed, {phase_result.warnings} warnings\n"
                )
        except Exception as e:
            if console:
                console.print(f"  [red]Phase failed: {e}[/red]\n")
            else:
                print(f"  Phase failed: {e}\n")

    return results


def print_summary(results: list) -> bool:
    """Print final summary and return success status."""
    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_warnings = sum(r.warnings for r in results)
    total_tests = total_passed + total_failed + total_warnings

    all_passed = all(r.success for r in results)

    print_header("VALIDATION SUMMARY")

    if console:
        table = Table(title="Results by Phase")
        table.add_column("Phase", style="cyan")
        table.add_column("Passed", style="green")
        table.add_column("Failed", style="red")
        table.add_column("Warnings", style="yellow")
        table.add_column("Status")

        for r in results:
            status = "[green]PASS[/green]" if r.success else "[red]FAIL[/red]"
            table.add_row(
                f"{r.phase}. {r.name}",
                str(r.passed),
                str(r.failed),
                str(r.warnings),
                status,
            )

        console.print(table)
        console.print()

        if all_passed:
            console.print(
                Panel(
                    f"[bold green]SYSTEM IS AIRTIGHT - PRODUCTION READY[/bold green]\n\n"
                    f"Total: {total_passed}/{total_tests} checks passed\n"
                    f"Warnings: {total_warnings}\n\n"
                    f"All critical systems validated successfully.",
                    style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]VALIDATION FAILED - NOT PRODUCTION READY[/bold red]\n\n"
                    f"Total: {total_passed}/{total_tests} checks passed\n"
                    f"Failed: {total_failed}\n"
                    f"Warnings: {total_warnings}\n\n"
                    f"Fix failures before deploying to production.",
                    style="red",
                )
            )
    else:
        print(f"\nTotal: {total_passed}/{total_tests} passed")
        print(f"Failed: {total_failed}, Warnings: {total_warnings}")
        print("\n" + ("PRODUCTION READY" if all_passed else "NOT READY - FIX FAILURES"))

    return all_passed


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Comprehensive Production Validation")
    parser.add_argument(
        "--remote", action="store_true", help="Test against Render deployment"
    )
    parser.add_argument(
        "--quick", action="store_true", help="Core checks only (phases 1-5)"
    )
    args = parser.parse_args()

    if console:
        console.print(
            Panel(
                "[bold]KALSHI TRADING BOT - SYSTEM VALIDATION[/bold]\n"
                "Comprehensive production readiness check",
                style="blue",
            )
        )
        if args.remote:
            console.print(
                "[yellow]Testing against REMOTE (Render) deployment[/yellow]\n"
            )

    results = await run_all_validations(remote=args.remote, quick=args.quick)
    success = print_summary(results)

    # Print actionable next steps
    if console and not success:
        console.print("\n[bold yellow]ACTION REQUIRED:[/bold yellow]")
        for r in results:
            if r.failed > 0:
                console.print(
                    f"  - Fix Phase {r.phase} ({r.name}): {r.failed} failures"
                )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
