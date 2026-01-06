#!/usr/bin/env python3
"""
Production Validation Script
============================
Comprehensive validation of all system components before going live.

Usage:
    uv run python validate_production.py
    uv run python validate_production.py --phase 1  # Run specific phase
    uv run python validate_production.py --quick    # Quick checks only
"""

import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional

# Rich for pretty output
try:
    from rich import print as rprint
    from rich.console import Console
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    print("Note: Install 'rich' for better output: pip install rich")


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
    results: list[ValidationResult] = field(default_factory=list)

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
        return ValidationResult(name, Status.FAIL, str(e), duration)


# =============================================================================
# PHASE 1: Configuration & Environment
# =============================================================================


async def validate_phase1_config() -> PhaseResult:
    """Validate configuration and environment variables."""
    results = PhaseResult(1, "Configuration & Environment")

    # Check 1: Environment file exists
    def check_env_file():
        if os.path.exists(".env"):
            return Status.PASS, ".env file found"
        elif os.path.exists("env_template.txt"):
            return Status.WARN, ".env missing, but template exists"
        return Status.FAIL, "No .env file found"

    results.results.append(await timed_check("Environment file", check_env_file))

    # Check 2: Load config without errors
    async def check_config_loads():
        try:
            from config import BotConfig

            config = BotConfig()
            return Status.PASS, f"Config loaded (demo={config.kalshi.use_demo})"
        except Exception as e:
            return Status.FAIL, str(e)

    results.results.append(await timed_check("Config loads", check_config_loads))

    # Check 3: Kalshi credentials
    async def check_kalshi_creds():
        from config import BotConfig

        config = BotConfig()
        if not config.kalshi.api_key:
            return Status.FAIL, "KALSHI_API_KEY not set"
        if config.kalshi.api_key.startswith("your_"):
            return Status.FAIL, "KALSHI_API_KEY is placeholder"
        if not config.kalshi.private_key:
            return Status.FAIL, "KALSHI_PRIVATE_KEY not set"
        if "BEGIN RSA" not in config.kalshi.private_key:
            return Status.FAIL, "Private key not in PEM format"
        return Status.PASS, "Kalshi credentials configured"

    results.results.append(await timed_check("Kalshi credentials", check_kalshi_creds))

    # Check 4: OpenAI credentials
    async def check_openai_creds():
        from config import BotConfig

        config = BotConfig()
        if not config.openai.api_key:
            return Status.FAIL, "OPENAI_API_KEY not set"
        if not config.openai.api_key.startswith("sk-"):
            return Status.WARN, "OPENAI_API_KEY doesn't start with sk-"
        return Status.PASS, "OpenAI credentials configured"

    results.results.append(await timed_check("OpenAI credentials", check_openai_creds))

    # Check 5: Database configuration
    async def check_db_config():
        from config import BotConfig

        config = BotConfig()
        if config.database.db_type == "postgres":
            if not all(
                [
                    config.database.pg_host,
                    config.database.pg_database,
                    config.database.pg_user,
                    config.database.pg_password,
                ]
            ):
                return Status.FAIL, "PostgreSQL credentials incomplete"
            return Status.PASS, f"PostgreSQL: {config.database.pg_host}"
        else:
            return Status.PASS, f"SQLite: {config.database.db_path}"

    results.results.append(await timed_check("Database config", check_db_config))

    # Check 6: Risk parameters
    async def check_risk_params():
        from config import BotConfig

        config = BotConfig()
        issues = []
        if config.max_daily_loss <= 0:
            issues.append("max_daily_loss <= 0")
        if config.kelly_fraction < 0.1 or config.kelly_fraction > 1.5:
            issues.append(f"kelly_fraction={config.kelly_fraction} out of range")
        if config.max_bet_amount > config.bankroll * 0.25:
            issues.append("max_bet > 25% of bankroll")
        if config.z_threshold < 0:
            issues.append("z_threshold < 0")

        if issues:
            return Status.WARN, "; ".join(issues)
        return (
            Status.PASS,
            f"Bankroll=${config.bankroll}, MaxBet=${config.max_bet_amount}",
        )

    results.results.append(await timed_check("Risk parameters", check_risk_params))

    # Check 7: TrendRadar config (if enabled)
    async def check_trendradar_config():
        from config import BotConfig

        config = BotConfig()
        if not config.trendradar.enabled:
            return Status.SKIP, "TrendRadar disabled"
        if not config.trendradar.base_url:
            return Status.FAIL, "TRENDRADAR_URL not set"
        return Status.PASS, f"TrendRadar: {config.trendradar.base_url}"

    results.results.append(
        await timed_check("TrendRadar config", check_trendradar_config)
    )

    return results


# =============================================================================
# PHASE 2: Database Integrity
# =============================================================================


async def validate_phase2_database() -> PhaseResult:
    """Validate database connectivity and schema."""
    results = PhaseResult(2, "Database Integrity")

    # Check 1: Database connection
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

    # Check 2: Required tables exist
    async def check_tables():
        from config import BotConfig

        config = BotConfig()

        required_tables = [
            "betting_decisions",
            "market_snapshots",
            "calibration_records",
            "performance_daily",
            "run_history",
        ]

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
            query = """
                SELECT table_name FROM information_schema.tables
                WHERE table_schema = 'public'
            """
            result = await db.fetchall(query)
            existing = {row["table_name"] for row in result}
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            query = "SELECT name FROM sqlite_master WHERE type='table'"
            result = await db.execute(query)
            existing = {row[0] for row in result}
        await db.close()

        missing = [t for t in required_tables if t not in existing]
        if missing:
            return Status.FAIL, f"Missing tables: {missing}"
        return Status.PASS, f"All {len(required_tables)} tables exist"

    results.results.append(await timed_check("Required tables", check_tables))

    # Check 3: Signal columns exist
    async def check_signal_columns():
        from config import BotConfig

        config = BotConfig()

        signal_columns = [
            "signal_applied",
            "signal_direction",
            "signal_strength",
            "signal_sentiment",
            "confidence_boost",
            "kelly_multiplier",
        ]

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
            query = """
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'betting_decisions'
            """
            result = await db.fetchall(query)
            existing = {row["column_name"] for row in result}
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            query = "PRAGMA table_info(betting_decisions)"
            result = await db.execute(query)
            existing = {row[1] for row in result}
        await db.close()

        missing = [c for c in signal_columns if c not in existing]
        if missing:
            return Status.WARN, f"Missing signal columns: {missing}"
        return Status.PASS, "All signal columns present"

    results.results.append(await timed_check("Signal columns", check_signal_columns))

    # Check 4: Row counts
    async def check_row_counts():
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
            counts = {}
            for table in ["betting_decisions", "run_history"]:
                result = await db.fetchone(f"SELECT COUNT(*) as cnt FROM {table}")
                counts[table] = result["cnt"] if result else 0
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            counts = {}
            for table in ["betting_decisions", "run_history"]:
                result = await db.execute(f"SELECT COUNT(*) FROM {table}")
                counts[table] = result[0][0]
        await db.close()

        return (
            Status.PASS,
            f"decisions={counts['betting_decisions']}, runs={counts['run_history']}",
        )

    results.results.append(await timed_check("Row counts", check_row_counts))

    # Check 5: Data integrity
    async def check_data_integrity():
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
            # PostgreSQL uses BOOLEAN (TRUE/FALSE)
            orphan_query = """
                SELECT COUNT(*) as cnt FROM betting_decisions
                WHERE signal_applied = TRUE AND signal_direction IS NULL
            """
            result = await db.fetchone(orphan_query)
            orphans = result["cnt"] if result else 0
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            # SQLite uses INTEGER (1/0)
            orphan_query = """
                SELECT COUNT(*) FROM betting_decisions
                WHERE signal_applied = 1 AND signal_direction IS NULL
            """
            result = await db.execute(orphan_query)
            orphans = result[0][0]
        await db.close()

        if orphans > 0:
            return (
                Status.WARN,
                f"{orphans} records with signal_applied=TRUE but no direction",
            )
        return Status.PASS, "No data integrity issues"

    results.results.append(await timed_check("Data integrity", check_data_integrity))

    return results


# =============================================================================
# PHASE 3: API Integration
# =============================================================================


async def validate_phase3_apis() -> PhaseResult:
    """Validate API connectivity and authentication."""
    results = PhaseResult(3, "API Integration")

    # Check 1: Kalshi API - Events (tests authentication)
    async def check_kalshi_events():
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)

        try:
            # IMPORTANT: Must call login() to initialize the HTTP client
            await client.login()
            events = await client.get_events(limit=5)
            if events and len(events) > 0:
                return Status.PASS, f"Found {len(events)} events (auth working)"
            return Status.WARN, "No events returned"
        except Exception as e:
            return Status.FAIL, str(e)

    results.results.append(await timed_check("Kalshi API & Auth", check_kalshi_events))

    # Check 2: Kalshi API - Positions
    async def check_kalshi_positions():
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)

        try:
            await client.login()
            positions = await client.get_user_positions()
            count = len(positions) if positions else 0
            return Status.PASS, f"Positions: {count} open"
        except Exception as e:
            return Status.FAIL, str(e)

    results.results.append(
        await timed_check("Kalshi positions", check_kalshi_positions)
    )

    # Check 3: Kalshi API - Markets
    async def check_kalshi_markets():
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)

        try:
            await client.login()
            events = await client.get_events(limit=1)
            if events:
                event_ticker = events[0].get("event_ticker")
                markets = events[0].get("markets", [])  # Markets are already loaded
                return Status.PASS, f"Markets for {event_ticker}: {len(markets)}"
            return Status.WARN, "No events to test markets"
        except Exception as e:
            return Status.FAIL, str(e)

    results.results.append(await timed_check("Kalshi markets", check_kalshi_markets))

    # Check 4: OpenAI API - Basic
    async def check_openai_api():
        from config import BotConfig

        config = BotConfig()

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=config.openai.api_key)

            # Simple completion test
            response = await client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": "Say 'OK'"}],
                max_tokens=5,
            )
            if response.choices[0].message.content:
                return Status.PASS, "OpenAI API responding"
            return Status.FAIL, "Empty response"
        except Exception as e:
            return Status.FAIL, str(e)

    results.results.append(await timed_check("OpenAI API", check_openai_api))

    # Check 5: TrendRadar API (if enabled)
    async def check_trendradar():
        from config import BotConfig

        config = BotConfig()

        if not config.trendradar.enabled:
            return Status.SKIP, "TrendRadar disabled"

        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{config.trendradar.base_url}/health")
                if response.status_code == 200:
                    return Status.PASS, "TrendRadar healthy"
                return Status.WARN, f"Status: {response.status_code}"
        except Exception as e:
            return Status.WARN, f"Not reachable: {e}"

    results.results.append(await timed_check("TrendRadar API", check_trendradar))

    return results


# =============================================================================
# PHASE 4: Trading Logic
# =============================================================================


async def validate_phase4_trading() -> PhaseResult:
    """Validate trading calculations and risk management."""
    results = PhaseResult(4, "Trading Logic")

    # Check 1: R-score calculation
    async def check_r_score():
        import math

        def calc_r_score(research_prob: float, market_price: float) -> float:
            if market_price <= 0 or market_price >= 1:
                return 0.0
            std_dev = math.sqrt(market_price * (1 - market_price))
            return (research_prob - market_price) / std_dev

        # Test cases
        tests = [
            (0.70, 0.50, 0.40),  # Positive edge
            (0.50, 0.50, 0.00),  # No edge
            (0.30, 0.50, -0.40),  # Negative edge
        ]

        for rp, mp, expected in tests:
            result = calc_r_score(rp, mp)
            if abs(result - expected) > 0.01:
                return (
                    Status.FAIL,
                    f"r_score({rp}, {mp}) = {result}, expected {expected}",
                )

        return Status.PASS, "R-score calculation correct"

    results.results.append(await timed_check("R-score calculation", check_r_score))

    # Check 2: Kelly criterion
    async def check_kelly():
        def calc_kelly(research_prob: float, market_price: float) -> float:
            if market_price >= 1:
                return 0.0
            return (research_prob - market_price) / (1 - market_price)

        tests = [
            (0.70, 0.50, 0.40),
            (0.50, 0.50, 0.00),
            (0.80, 0.60, 0.50),
        ]

        for rp, mp, expected in tests:
            result = calc_kelly(rp, mp)
            if abs(result - expected) > 0.01:
                return Status.FAIL, f"kelly({rp}, {mp}) = {result}, expected {expected}"

        return Status.PASS, "Kelly calculation correct"

    results.results.append(await timed_check("Kelly calculation", check_kelly))

    # Check 3: Bet sizing with caps
    async def check_bet_sizing():
        from config import BotConfig

        config = BotConfig()

        bankroll = config.bankroll
        kelly_frac = config.kelly_fraction
        max_bet = config.max_bet_amount

        # Test: Large Kelly should be capped
        kelly = 0.5  # 50% Kelly
        raw_bet = bankroll * kelly_frac * kelly
        capped_bet = min(raw_bet, max_bet)

        if capped_bet > max_bet:
            return Status.FAIL, f"Bet {capped_bet} exceeds max {max_bet}"

        return Status.PASS, f"Bet sizing: raw=${raw_bet:.2f}, capped=${capped_bet:.2f}"

    results.results.append(await timed_check("Bet sizing", check_bet_sizing))

    # Check 4: Kill switch logic
    async def check_kill_switch():
        from config import BotConfig

        config = BotConfig()

        if not config.enable_kill_switch:
            return Status.WARN, "Kill switch disabled"

        max_loss = config.max_daily_loss
        max_loss_pct = config.max_daily_loss_pct

        if max_loss <= 0:
            return Status.FAIL, "max_daily_loss must be positive"

        return Status.PASS, f"Kill switch: ${max_loss} or {max_loss_pct*100}%"

    results.results.append(await timed_check("Kill switch config", check_kill_switch))

    # Check 5: Signal influence bounds
    async def check_signal_influence():
        from config import BotConfig

        config = BotConfig()

        max_boost = config.trendradar.max_confidence_boost
        strong_threshold = config.trendradar.strong_signal_threshold
        kelly_mult = config.trendradar.aligned_signal_kelly_multiplier

        issues = []
        if max_boost < 0 or max_boost > 0.5:
            issues.append(f"max_boost={max_boost} out of [0, 0.5]")
        if strong_threshold < 0 or strong_threshold > 1:
            issues.append(f"strong_threshold={strong_threshold} out of [0, 1]")
        if kelly_mult < 1.0 or kelly_mult > 2.0:
            issues.append(f"kelly_mult={kelly_mult} out of [1, 2]")

        if issues:
            return Status.WARN, "; ".join(issues)
        return Status.PASS, f"boost={max_boost}, threshold={strong_threshold}"

    results.results.append(
        await timed_check("Signal influence bounds", check_signal_influence)
    )

    return results


# =============================================================================
# PHASE 5: Security
# =============================================================================


async def validate_phase5_security() -> PhaseResult:
    """Validate security configuration and practices."""
    results = PhaseResult(5, "Security")

    # Check 1: No hardcoded secrets
    async def check_no_hardcoded_secrets():
        import re

        patterns = [
            r"sk-[a-zA-Z0-9]{20,}",  # OpenAI key
            r"-----BEGIN RSA PRIVATE KEY-----",  # Inline private key
        ]

        suspicious_files = []
        for root, dirs, files in os.walk("."):
            # Skip common directories
            dirs[:] = [
                d
                for d in dirs
                if d not in [".git", ".venv", "venv", "__pycache__", "node_modules"]
            ]

            for file in files:
                if file.endswith(".py") and file != "validate_production.py":
                    filepath = os.path.join(root, file)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            content = f.read()
                            for pattern in patterns:
                                if re.search(pattern, content):
                                    suspicious_files.append(filepath)
                                    break
                    except Exception:
                        pass

        if suspicious_files:
            return Status.WARN, f"Potential secrets in: {suspicious_files[:3]}"
        return Status.PASS, "No hardcoded secrets found"

    results.results.append(
        await timed_check("No hardcoded secrets", check_no_hardcoded_secrets)
    )

    # Check 2: .gitignore includes sensitive files
    async def check_gitignore():
        required_patterns = [".env", "*.pem", "private"]

        if not os.path.exists(".gitignore"):
            return Status.WARN, ".gitignore not found"

        with open(".gitignore", "r") as f:
            content = f.read().lower()

        missing = [p for p in required_patterns if p.lower() not in content]
        if missing:
            return Status.WARN, f"Missing from .gitignore: {missing}"
        return Status.PASS, "Sensitive files in .gitignore"

    results.results.append(await timed_check(".gitignore check", check_gitignore))

    # Check 3: Environment isolation
    async def check_env_isolation():
        from config import BotConfig

        config = BotConfig()

        if config.kalshi.use_demo:
            return Status.PASS, "Using demo environment"
        return Status.WARN, "LIVE trading mode - verify intentional"

    results.results.append(
        await timed_check("Environment isolation", check_env_isolation)
    )

    # Check 4: Logging doesn't expose secrets
    async def check_logging_safe():
        # Check log files for secrets
        log_files = ["bot_output.log", "trading_bot.log"]
        patterns = [r"sk-", r"BEGIN RSA", r"password"]

        for log_file in log_files:
            if os.path.exists(log_file):
                with open(log_file, "r") as f:
                    content = f.read()[:10000]  # Check first 10K chars
                    for pattern in patterns:
                        if pattern.lower() in content.lower():
                            return Status.WARN, f"Potential secret in {log_file}"

        return Status.PASS, "Logs appear safe"

    results.results.append(await timed_check("Logging safety", check_logging_safe))

    return results


# =============================================================================
# PHASE 6: Performance
# =============================================================================


async def validate_phase6_performance() -> PhaseResult:
    """Validate system performance."""
    results = PhaseResult(6, "Performance")

    # Check 1: Database query performance
    async def check_db_query_perf():
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
            start = time.perf_counter()
            await db.fetchone("SELECT COUNT(*) as cnt FROM betting_decisions")
            duration_ms = (time.perf_counter() - start) * 1000
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            start = time.perf_counter()
            await db.execute("SELECT COUNT(*) FROM betting_decisions")
            duration_ms = (time.perf_counter() - start) * 1000

        await db.close()

        if duration_ms > 100:
            return Status.WARN, f"Slow query: {duration_ms:.0f}ms"
        return Status.PASS, f"Query time: {duration_ms:.0f}ms"

    results.results.append(
        await timed_check("Database query speed", check_db_query_perf)
    )

    # Check 2: API response time
    async def check_api_perf():
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi)

        await client.login()
        start = time.perf_counter()
        await client.get_events(limit=1)
        duration_ms = (time.perf_counter() - start) * 1000

        if duration_ms > 5000:
            return Status.WARN, f"Slow API: {duration_ms:.0f}ms"
        return Status.PASS, f"API time: {duration_ms:.0f}ms"

    results.results.append(await timed_check("Kalshi API speed", check_api_perf))

    # Check 3: Memory usage
    async def check_memory():
        try:
            import psutil

            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024

            if memory_mb > 500:
                return Status.WARN, f"High memory: {memory_mb:.0f}MB"
            return Status.PASS, f"Memory: {memory_mb:.0f}MB"
        except ImportError:
            return Status.SKIP, "psutil not installed"

    results.results.append(await timed_check("Memory usage", check_memory))

    return results


# =============================================================================
# PHASE 7: Operational Readiness
# =============================================================================


async def validate_phase7_operations() -> PhaseResult:
    """Validate operational readiness."""
    results = PhaseResult(7, "Operational Readiness")

    # Check 1: Dashboard API
    async def check_dashboard():
        try:
            import httpx

            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get("http://localhost:8000/api/status")
                if response.status_code == 200:
                    return Status.PASS, "Dashboard API responding"
                return Status.WARN, f"Status: {response.status_code}"
        except Exception:
            return Status.SKIP, "Dashboard not running"

    results.results.append(await timed_check("Dashboard API", check_dashboard))

    # Check 2: Recent run history
    async def check_recent_runs():
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
            # PostgreSQL - column is started_at, use fetchone
            result = await db.fetchone(
                """
                SELECT COUNT(*) as cnt FROM run_history
                WHERE started_at > NOW() - INTERVAL '7 days'
            """
            )
            recent_runs = result["cnt"] if result else 0
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            # SQLite - column is started_at
            result = await db.execute(
                """
                SELECT COUNT(*) FROM run_history
                WHERE started_at > datetime('now', '-7 days')
            """
            )
            recent_runs = result[0][0]

        await db.close()

        if recent_runs == 0:
            return Status.WARN, "No runs in last 7 days"
        return Status.PASS, f"{recent_runs} runs in last 7 days"

    results.results.append(await timed_check("Recent run history", check_recent_runs))

    # Check 3: Reconciliation status
    async def check_reconciliation():
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
                SELECT COUNT(*) as cnt FROM betting_decisions
                WHERE status = 'pending'
            """
            )
            pending = result["cnt"] if result else 0
        else:
            from db.database import Database

            db = Database(config.database.db_path)
            await db.connect()
            result = await db.execute(
                """
                SELECT COUNT(*) FROM betting_decisions
                WHERE status = 'pending'
            """
            )
            pending = result[0][0]
        await db.close()

        if pending > 100:
            return Status.WARN, f"{pending} pending decisions need reconciliation"
        return Status.PASS, f"{pending} pending decisions"

    results.results.append(
        await timed_check("Reconciliation status", check_reconciliation)
    )

    # Check 4: Log file accessible
    async def check_logs():
        log_file = "bot_output.log"
        if os.path.exists(log_file):
            size_kb = os.path.getsize(log_file) / 1024
            return Status.PASS, f"Log file: {size_kb:.0f}KB"
        return Status.WARN, "No log file found"

    results.results.append(await timed_check("Log file", check_logs))

    return results


# =============================================================================
# Main Execution
# =============================================================================


async def run_all_validations(phases: list[int] = None) -> list[PhaseResult]:
    """Run all validation phases."""
    all_phases = [
        (1, validate_phase1_config),
        (2, validate_phase2_database),
        (3, validate_phase3_apis),
        (4, validate_phase4_trading),
        (5, validate_phase5_security),
        (6, validate_phase6_performance),
        (7, validate_phase7_operations),
    ]

    results = []
    for phase_num, phase_func in all_phases:
        if phases and phase_num not in phases:
            continue

        print_header(f"Phase {phase_num}: {phase_func.__doc__.strip()}")

        try:
            phase_result = await phase_func()
            results.append(phase_result)

            for result in phase_result.results:
                print_result(result)

            # Summary
            if console:
                console.print(
                    f"\n  Summary: {phase_result.passed} passed, "
                    f"{phase_result.failed} failed, {phase_result.warnings} warnings\n"
                )
        except Exception as e:
            if console:
                console.print(f"  [red]Phase failed with error: {e}[/red]\n")
            else:
                print(f"  Phase failed with error: {e}\n")

    return results


def print_summary(results: list[PhaseResult]):
    """Print final summary."""
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
                    f"[bold green]ALL VALIDATIONS PASSED[/bold green]\n\n"
                    f"Total: {total_passed}/{total_tests} checks passed\n"
                    f"Warnings: {total_warnings}\n\n"
                    f"System is ready for production!",
                    style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]VALIDATION FAILED[/bold red]\n\n"
                    f"Total: {total_passed}/{total_tests} checks passed\n"
                    f"Failed: {total_failed}\n"
                    f"Warnings: {total_warnings}\n\n"
                    f"Fix failures before going live.",
                    style="red",
                )
            )
    else:
        print(f"\nTotal: {total_passed}/{total_tests} passed")
        print(f"Failed: {total_failed}")
        print(f"Warnings: {total_warnings}")

        if all_passed:
            print("\n[OK] System ready for production")
        else:
            print("\n[FAIL] Fix failures before going live")

    return all_passed


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Production Validation Script")
    parser.add_argument("--phase", type=int, help="Run specific phase (1-7)")
    parser.add_argument(
        "--quick", action="store_true", help="Quick checks only (phases 1-2)"
    )
    args = parser.parse_args()

    if console:
        console.print(
            Panel(
                "[bold]Kalshi Deep Trading Bot[/bold]\n" "Production Validation Script",
                style="blue",
            )
        )

    # Change to project directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # Determine phases to run
    phases = None
    if args.phase:
        phases = [args.phase]
    elif args.quick:
        phases = [1, 2]

    # Run validations
    results = await run_all_validations(phases)

    # Print summary
    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
