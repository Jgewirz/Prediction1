#!/usr/bin/env python3
"""
Unified validation script for Kalshi Deep Trading Bot.

Combines all validation phases into a single script with --phase flags:
  - system:    Config, credentials, database connectivity
  - pipeline:  Market fetch, research, decision generation
  - dashboard: API endpoints, WebSocket connectivity

Usage:
    python validate.py                    # Run all phases
    python validate.py --phase system     # Config, credentials, database
    python validate.py --phase pipeline   # Market fetch, research, decisions
    python validate.py --phase dashboard  # API endpoints, WebSocket
    python validate.py --quick            # Core checks only (phases 1-3)
    python validate.py --remote           # Test against Render deployment
"""

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime, timedelta
from typing import Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


class ValidationResult:
    """Container for validation results."""

    def __init__(self, name: str):
        self.name = name
        self.passed = 0
        self.failed = 0
        self.skipped = 0
        self.errors: list[str] = []
        self.warnings: list[str] = []
        self.start_time = time.time()
        self.duration = 0.0

    def pass_check(self, message: str = ""):
        self.passed += 1
        if message:
            print(f"  [PASS] {message}")

    def fail_check(self, message: str):
        self.failed += 1
        self.errors.append(message)
        print(f"  [FAIL] {message}")

    def skip_check(self, message: str):
        self.skipped += 1
        print(f"  [SKIP] {message}")

    def warn(self, message: str):
        self.warnings.append(message)
        print(f"  [WARN] {message}")

    def finalize(self):
        self.duration = time.time() - self.start_time

    @property
    def success(self) -> bool:
        return self.failed == 0


def print_header(title: str):
    """Print a section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_summary(results: list[ValidationResult]):
    """Print validation summary."""
    print_header("VALIDATION SUMMARY")

    total_passed = sum(r.passed for r in results)
    total_failed = sum(r.failed for r in results)
    total_skipped = sum(r.skipped for r in results)
    total_duration = sum(r.duration for r in results)

    for r in results:
        status = "PASS" if r.success else "FAIL"
        print(f"  {r.name}: {status} ({r.passed} passed, {r.failed} failed, {r.skipped} skipped) [{r.duration:.2f}s]")
        for err in r.errors:
            print(f"    - {err}")

    print(f"\n  Total: {total_passed} passed, {total_failed} failed, {total_skipped} skipped")
    print(f"  Duration: {total_duration:.2f}s")

    if total_failed == 0:
        print("\n  All validations PASSED!")
        return True
    else:
        print(f"\n  {total_failed} validation(s) FAILED")
        return False


# =============================================================================
# PHASE 1: SYSTEM VALIDATION
# =============================================================================

def validate_system() -> ValidationResult:
    """Validate system configuration, credentials, and database."""
    result = ValidationResult("System")
    print_header("Phase 1: System Validation")

    # 1. Check environment variables
    print("\n  Checking environment variables...")
    required_vars = [
        "KALSHI_API_KEY",
        "KALSHI_PRIVATE_KEY",
        "OPENAI_API_KEY",
    ]

    for var in required_vars:
        if os.getenv(var):
            result.pass_check(f"{var} is set")
        else:
            result.fail_check(f"{var} is not set")

    # 2. Load configuration
    print("\n  Loading configuration...")
    try:
        from config import load_config
        config = load_config()
        result.pass_check("Configuration loaded successfully")

        # Check Kalshi environment
        if config.kalshi.use_demo:
            result.warn("Running in DEMO mode")
        else:
            result.pass_check("Running in PRODUCTION mode")

    except Exception as e:
        result.fail_check(f"Configuration failed: {e}")
        result.finalize()
        return result

    # 3. Database connectivity
    print("\n  Testing database connectivity...")
    try:
        if config.database.enable_db:
            if config.database.db_type == "postgres":
                # Test PostgreSQL connection
                import asyncpg

                async def test_pg():
                    conn = await asyncpg.connect(
                        host=config.database.pg_host,
                        database=config.database.pg_database,
                        user=config.database.pg_user,
                        password=config.database.pg_password,
                        ssl="require" if config.database.pg_ssl == "require" else None,
                    )
                    version = await conn.fetchval("SELECT version()")
                    await conn.close()
                    return version

                version = asyncio.run(test_pg())
                result.pass_check(f"PostgreSQL connected")
            else:
                # SQLite
                import sqlite3
                conn = sqlite3.connect(config.database.db_path)
                conn.close()
                result.pass_check(f"SQLite connected: {config.database.db_path}")
        else:
            result.skip_check("Database disabled in config")

    except Exception as e:
        result.fail_check(f"Database connection failed: {e}")

    # 4. Kalshi API connectivity
    print("\n  Testing Kalshi API connectivity...")
    try:
        from kalshi_client import KalshiClient
        client = KalshiClient(config.kalshi)

        # Test exchange status
        status = client.get_exchange_status()
        if status:
            result.pass_check(f"Kalshi API connected")
        else:
            result.fail_check("Kalshi API returned empty status")

    except Exception as e:
        result.fail_check(f"Kalshi API failed: {e}")

    # 5. OpenAI API connectivity
    print("\n  Testing OpenAI API connectivity...")
    try:
        from openai import OpenAI
        client = OpenAI(api_key=config.openai.api_key)
        models = client.models.list()
        result.pass_check("OpenAI API connected")
    except Exception as e:
        result.fail_check(f"OpenAI API failed: {e}")

    result.finalize()
    return result


# =============================================================================
# PHASE 2: PIPELINE VALIDATION
# =============================================================================

def validate_pipeline() -> ValidationResult:
    """Validate trading pipeline: market fetch, research, decisions."""
    result = ValidationResult("Pipeline")
    print_header("Phase 2: Pipeline Validation")

    # Load config
    try:
        from config import load_config
        config = load_config()
    except Exception as e:
        result.fail_check(f"Configuration failed: {e}")
        result.finalize()
        return result

    # 1. Fetch events
    print("\n  Fetching events from Kalshi...")
    try:
        from kalshi_client import KalshiClient
        client = KalshiClient(config.kalshi)
        events = client.get_events(limit=5, status="open")

        if events and len(events) > 0:
            result.pass_check(f"Fetched {len(events)} events")
            sample_event = events[0]

            # Validate event structure
            required_fields = ["event_ticker", "title"]
            for field in required_fields:
                if field in sample_event:
                    result.pass_check(f"Event has '{field}' field")
                else:
                    result.fail_check(f"Event missing '{field}' field")
        else:
            result.fail_check("No events returned")

    except Exception as e:
        result.fail_check(f"Event fetch failed: {e}")
        result.finalize()
        return result

    # 2. Fetch markets
    print("\n  Fetching markets...")
    try:
        if events:
            event_ticker = events[0].get("event_ticker")
            markets = client.get_markets(event_ticker=event_ticker, limit=3)

            if markets and len(markets) > 0:
                result.pass_check(f"Fetched {len(markets)} markets for {event_ticker}")
                sample_market = markets[0]

                # Validate market structure
                required_fields = ["ticker", "yes_bid", "yes_ask"]
                for field in required_fields:
                    if field in sample_market:
                        result.pass_check(f"Market has '{field}' field")
                    else:
                        result.fail_check(f"Market missing '{field}' field")
            else:
                result.warn(f"No markets for event {event_ticker}")
    except Exception as e:
        result.fail_check(f"Market fetch failed: {e}")

    # 3. Test research client
    print("\n  Testing research client...")
    try:
        from research_client import OctagonClient
        research_client = OctagonClient(config)

        # Quick connectivity test
        result.pass_check("Research client initialized")
    except Exception as e:
        result.fail_check(f"Research client failed: {e}")

    # 4. Test betting models
    print("\n  Testing betting models...")
    try:
        from betting_models import BettingDecision, MarketAnalysis

        # Create test decision
        decision = BettingDecision(
            action="buy_yes",
            amount=10.0,
            confidence=0.75,
            reasoning="Test decision",
            market_ticker="TEST-MARKET",
            market_title="Test Market",
            event_ticker="TEST-EVENT",
            research_probability=0.65,
            market_price=0.50,
        )
        result.pass_check("BettingDecision model works")

    except Exception as e:
        result.fail_check(f"Betting models failed: {e}")

    # 5. Test risk metrics calculation
    print("\n  Testing risk metrics...")
    try:
        from trading_bot import calculate_risk_adjusted_metrics

        metrics = calculate_risk_adjusted_metrics(
            research_prob=0.65,
            market_price=0.50,
        )

        if "r_score" in metrics and "kelly_fraction" in metrics:
            result.pass_check(f"Risk metrics: R={metrics['r_score']:.2f}, Kelly={metrics['kelly_fraction']:.2f}")
        else:
            result.fail_check("Risk metrics missing required fields")

    except Exception as e:
        result.fail_check(f"Risk metrics failed: {e}")

    result.finalize()
    return result


# =============================================================================
# PHASE 3: DASHBOARD VALIDATION
# =============================================================================

def validate_dashboard(remote_url: str = None) -> ValidationResult:
    """Validate dashboard API endpoints and WebSocket."""
    result = ValidationResult("Dashboard")
    print_header("Phase 3: Dashboard Validation")

    base_url = remote_url or "http://localhost:8000"
    print(f"  Testing dashboard at: {base_url}")

    try:
        import httpx
    except ImportError:
        result.fail_check("httpx not installed - run: pip install httpx")
        result.finalize()
        return result

    async def test_endpoints():
        async with httpx.AsyncClient(timeout=10) as client:
            # 1. Health check
            print("\n  Testing health endpoint...")
            try:
                resp = await client.get(f"{base_url}/health")
                if resp.status_code == 200:
                    result.pass_check("Health endpoint OK")
                else:
                    result.fail_check(f"Health returned {resp.status_code}")
            except httpx.ConnectError:
                result.skip_check("Dashboard not running")
                return
            except Exception as e:
                result.fail_check(f"Health check failed: {e}")
                return

            # 2. API endpoints
            endpoints = [
                ("/api/kpis", "KPIs"),
                ("/api/decisions?limit=5", "Decisions"),
                ("/api/status", "Status"),
            ]

            print("\n  Testing API endpoints...")
            for path, name in endpoints:
                try:
                    resp = await client.get(f"{base_url}{path}")
                    if resp.status_code == 200:
                        result.pass_check(f"{name} endpoint OK")
                    else:
                        result.fail_check(f"{name} returned {resp.status_code}")
                except Exception as e:
                    result.fail_check(f"{name} failed: {e}")

            # 3. WebSocket test (if available)
            print("\n  Testing WebSocket...")
            try:
                import websockets

                async with websockets.connect(f"ws://{base_url.replace('http://', '')}/ws/live") as ws:
                    result.pass_check("WebSocket connected")
            except ImportError:
                result.skip_check("websockets not installed")
            except Exception as e:
                result.warn(f"WebSocket connection failed: {e}")

    try:
        asyncio.run(test_endpoints())
    except Exception as e:
        result.fail_check(f"Dashboard validation failed: {e}")

    result.finalize()
    return result


# =============================================================================
# MAIN
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Unified validation script for Kalshi Deep Trading Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python validate.py                    # Run all phases
    python validate.py --phase system     # Config, credentials, database
    python validate.py --phase pipeline   # Market fetch, research, decisions
    python validate.py --phase dashboard  # API endpoints, WebSocket
    python validate.py --quick            # Core checks only (phases 1-3)
    python validate.py --remote https://my-bot.onrender.com
        """,
    )

    parser.add_argument(
        "--phase",
        choices=["system", "pipeline", "dashboard", "all"],
        default="all",
        help="Validation phase to run (default: all)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick validation (system + pipeline only)",
    )
    parser.add_argument(
        "--remote",
        metavar="URL",
        help="Test against remote deployment URL",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("  Kalshi Deep Trading Bot - Unified Validation")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    results = []

    # Determine which phases to run
    phases_to_run = []
    if args.phase == "all":
        phases_to_run = ["system", "pipeline", "dashboard"]
    elif args.phase:
        phases_to_run = [args.phase]

    if args.quick:
        phases_to_run = ["system", "pipeline"]

    # Run selected phases
    if "system" in phases_to_run:
        results.append(validate_system())

    if "pipeline" in phases_to_run:
        results.append(validate_pipeline())

    if "dashboard" in phases_to_run:
        results.append(validate_dashboard(args.remote))

    # Print summary
    success = print_summary(results)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
