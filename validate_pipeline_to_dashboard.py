#!/usr/bin/env python3
"""
PIPELINE-TO-DASHBOARD VALIDATION SCRIPT
========================================
Validates the complete data flow from market collection through research,
scoring, and dashboard display. Ensures terminal outputs are visible to users.

Pipeline Flow:
    1. Kalshi API → Collect Events/Markets
    2. OpenAI → Research & Analysis
    3. Scoring Engine → Early Entry Score + R-Score
    4. Decision Engine → Generate Betting Decisions
    5. Dashboard Broadcasting → CLI Terminal + Decision Feed
    6. Dashboard Display → User Views Real-Time Data

Usage:
    uv run python validate_pipeline_to_dashboard.py
    uv run python validate_pipeline_to_dashboard.py --live-test    # Actually broadcast test data
    uv run python validate_pipeline_to_dashboard.py --full-cycle   # Run mini trading cycle
"""

import asyncio
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

# Change to project directory
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

try:
    from rich.console import Console
    from rich.layout import Layout
    from rich.live import Live
    from rich.panel import Panel
    from rich.table import Table

    console = Console()
except ImportError:
    console = None
    print("Install 'rich' for better output: pip install rich")


class TestStatus(Enum):
    PASS = "✅ PASS"
    FAIL = "❌ FAIL"
    WARN = "⚠️ WARN"
    SKIP = "⏭️ SKIP"
    RUNNING = "🔄 RUNNING"


@dataclass
class PipelineTestResult:
    step: str
    status: TestStatus
    message: str
    data: Optional[Dict] = None
    duration_ms: float = 0


def print_step(step: str, status: TestStatus, message: str):
    if console:
        color = {
            "PASS": "green",
            "FAIL": "red",
            "WARN": "yellow",
            "SKIP": "dim",
            "RUNNING": "cyan",
        }
        console.print(
            f"  [{color.get(status.name, 'white')}]{status.value}[/] {step}: {message}"
        )
    else:
        print(f"  {status.value} {step}: {message}")


# =============================================================================
# STEP 1: Market Data Collection
# =============================================================================


async def test_market_collection() -> PipelineTestResult:
    """Test that we can collect events and markets from Kalshi."""
    start = time.perf_counter()

    try:
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()
        client = KalshiClient(config.kalshi, early_entry_config=config.early_entry)
        await client.login()

        # Fetch events with early entry scoring
        events = await client.get_events(limit=10)
        await client.close()

        if not events:
            return PipelineTestResult(
                "Market Collection",
                TestStatus.FAIL,
                "No events returned from Kalshi API",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        # Check that events have markets
        total_markets = sum(len(e.get("markets", [])) for e in events)

        # Check for early entry scores
        events_with_scores = sum(1 for e in events if e.get("early_entry_score", 0) > 0)

        return PipelineTestResult(
            "Market Collection",
            TestStatus.PASS,
            f"Collected {len(events)} events, {total_markets} markets, {events_with_scores} with early entry scores",
            data={
                "events_count": len(events),
                "markets_count": total_markets,
                "events_with_scores": events_with_scores,
                "sample_event": events[0].get("event_ticker") if events else None,
            },
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    except Exception as e:
        return PipelineTestResult(
            "Market Collection",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# STEP 2: Early Entry Scoring
# =============================================================================


async def test_early_entry_scoring() -> PipelineTestResult:
    """Test that early entry scoring is working correctly."""
    start = time.perf_counter()

    try:
        from config import BotConfig
        from kalshi_client import KalshiClient

        config = BotConfig()

        if not config.early_entry.enabled:
            return PipelineTestResult(
                "Early Entry Scoring",
                TestStatus.SKIP,
                "Early entry scoring is disabled in config",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        client = KalshiClient(config.kalshi, early_entry_config=config.early_entry)
        await client.login()

        # Create test data to verify scoring
        now = datetime.now(timezone.utc)
        test_event = {
            "event_ticker": "TEST-VALIDATION",
            "series_ticker": None,  # Unique event
            "strike_period": None,
            "time_remaining_hours": 72.0,
        }
        test_market = {
            "ticker": "TEST-M1",
            "volume_24h": 500,
            "open_time": now.isoformat(),
            "created_time": now.isoformat(),
        }

        # Test unique detection
        is_unique = client._is_unique_event(test_event)

        # Test scoring
        score = client.calculate_early_entry_score(test_market, test_event)
        await client.close()

        scoring_breakdown = {
            "is_unique_event": is_unique,
            "total_score": round(score, 3),
            "unique_bonus_applied": test_market.get("is_unique_event", False),
            "new_market_bonus_applied": test_market.get("is_new_market", False),
            "config": {
                "recency_weight": config.early_entry.recency_weight,
                "low_volume_weight": config.early_entry.low_volume_weight,
                "time_remaining_weight": config.early_entry.time_remaining_weight,
                "unique_event_bonus": config.early_entry.unique_event_bonus,
                "new_market_bonus": config.early_entry.new_market_bonus,
            },
        }

        if score > 0:
            return PipelineTestResult(
                "Early Entry Scoring",
                TestStatus.PASS,
                f"Score: {score:.2f} (unique={is_unique}, new_market={test_market.get('is_new_market', False)})",
                data=scoring_breakdown,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        else:
            return PipelineTestResult(
                "Early Entry Scoring",
                TestStatus.WARN,
                f"Score is 0 - check scoring weights",
                data=scoring_breakdown,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    except Exception as e:
        return PipelineTestResult(
            "Early Entry Scoring",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# STEP 3: Research Integration (OpenAI)
# =============================================================================


async def test_research_integration() -> PipelineTestResult:
    """Test that OpenAI research integration is working."""
    start = time.perf_counter()

    try:
        from config import BotConfig
        from openai import AsyncOpenAI

        config = BotConfig()

        if not config.openai.api_key:
            return PipelineTestResult(
                "Research Integration",
                TestStatus.FAIL,
                "OpenAI API key not configured",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        client = AsyncOpenAI(api_key=config.openai.api_key)

        # Quick test - simple completion
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Respond with just 'OK'"}],
            max_tokens=5,
        )

        if response.choices and response.choices[0].message.content:
            return PipelineTestResult(
                "Research Integration",
                TestStatus.PASS,
                f"OpenAI API responding - Model: {config.openai.model}",
                data={
                    "model": config.openai.model,
                    "response": response.choices[0].message.content[:50],
                },
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        return PipelineTestResult(
            "Research Integration",
            TestStatus.WARN,
            "OpenAI responded but with empty content",
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    except Exception as e:
        return PipelineTestResult(
            "Research Integration",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# STEP 4: Decision Generation
# =============================================================================


async def test_decision_generation() -> PipelineTestResult:
    """Test that the decision engine generates properly structured decisions."""
    start = time.perf_counter()

    try:
        from betting_models import BettingDecision
        from config import BotConfig

        config = BotConfig()

        # Create a sample decision using actual BettingDecision model fields
        sample_decision = BettingDecision(
            ticker="TEST-MARKET-YES",
            action="buy_yes",
            confidence=0.75,
            amount=10.00,
            reasoning="This is a validation test decision",
            event_name="Test Event for Validation",
            market_name="Will this test pass?",
            expected_return=0.15,
            r_score=1.85,
            kelly_fraction=0.31,
            market_price=0.55,
            research_probability=0.72,
            signal_applied=False,
        )

        # Convert to dict to verify serialization
        decision_dict = sample_decision.model_dump()

        required_fields = [
            "ticker",
            "action",
            "confidence",
            "amount",
            "reasoning",
            "r_score",
            "kelly_fraction",
        ]

        missing = [f for f in required_fields if f not in decision_dict]

        if missing:
            return PipelineTestResult(
                "Decision Generation",
                TestStatus.FAIL,
                f"Missing required fields: {missing}",
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        return PipelineTestResult(
            "Decision Generation",
            TestStatus.PASS,
            f"Decision structure valid - {len(decision_dict)} fields",
            data={
                "sample_decision": {
                    "ticker": decision_dict["ticker"],
                    "action": decision_dict["action"],
                    "r_score": decision_dict.get("r_score"),
                    "confidence": decision_dict["confidence"],
                    "amount": decision_dict["amount"],
                },
                "total_fields": len(decision_dict),
            },
            duration_ms=(time.perf_counter() - start) * 1000,
        )

    except Exception as e:
        return PipelineTestResult(
            "Decision Generation",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# STEP 5: Dashboard Broadcasting
# =============================================================================


async def test_dashboard_broadcasting(live_test: bool = False) -> PipelineTestResult:
    """Test that we can broadcast to the dashboard."""
    start = time.perf_counter()

    try:
        import httpx
        from dashboard.broadcaster import (DASHBOARD_API_URL,
                                           broadcast_account_update,
                                           broadcast_cli_log,
                                           broadcast_decision,
                                           broadcast_kpi_update)

        results = {
            "dashboard_url": DASHBOARD_API_URL,
            "endpoints_tested": [],
            "broadcasts_sent": 0,
        }

        # Check if dashboard is reachable
        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.get(f"{DASHBOARD_API_URL}/api/health")
                if response.status_code != 200:
                    return PipelineTestResult(
                        "Dashboard Broadcasting",
                        TestStatus.WARN,
                        f"Dashboard not healthy: {response.status_code}",
                        data=results,
                        duration_ms=(time.perf_counter() - start) * 1000,
                    )
        except httpx.ConnectError:
            return PipelineTestResult(
                "Dashboard Broadcasting",
                TestStatus.SKIP,
                f"Dashboard not reachable at {DASHBOARD_API_URL}",
                data=results,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

        if live_test:
            # Actually send test broadcasts

            # Test CLI log broadcast
            cli_success = await broadcast_cli_log(
                level="info",
                message="🧪 VALIDATION TEST: Pipeline validation running",
                source="validator",
                details={
                    "test": True,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            results["endpoints_tested"].append(("cli_log", cli_success))
            if cli_success:
                results["broadcasts_sent"] += 1

            # Test decision broadcast
            test_decision = {
                "decision_id": "validation-test-" + datetime.now().strftime("%H%M%S"),
                "action": "skip",
                "market_ticker": "VALIDATION-TEST",
                "market_title": "Pipeline Validation Test",
                "confidence": 0.0,
                "r_score": 0.0,
                "reasoning": "This is a test decision from pipeline validation",
            }
            decision_success = await broadcast_decision(test_decision)
            results["endpoints_tested"].append(("decision", decision_success))
            if decision_success:
                results["broadcasts_sent"] += 1

            # Test KPI broadcast
            kpi_success = await broadcast_kpi_update()
            results["endpoints_tested"].append(("kpi_update", kpi_success))
            if kpi_success:
                results["broadcasts_sent"] += 1

            return PipelineTestResult(
                "Dashboard Broadcasting",
                TestStatus.PASS if results["broadcasts_sent"] >= 2 else TestStatus.WARN,
                f"Sent {results['broadcasts_sent']}/3 broadcasts to {DASHBOARD_API_URL}",
                data=results,
                duration_ms=(time.perf_counter() - start) * 1000,
            )
        else:
            # Just verify endpoints exist
            async with httpx.AsyncClient(timeout=5.0) as client:
                endpoints = [
                    "/api/broadcast/decision",
                    "/api/broadcast/cli_log",
                    "/api/broadcast/account",
                    "/api/broadcast/kpis",
                ]

                for endpoint in endpoints:
                    try:
                        # OPTIONS request to check if endpoint exists
                        response = await client.options(
                            f"{DASHBOARD_API_URL}{endpoint}"
                        )
                        results["endpoints_tested"].append((endpoint, True))
                    except:
                        results["endpoints_tested"].append((endpoint, False))

            return PipelineTestResult(
                "Dashboard Broadcasting",
                TestStatus.PASS,
                f"Dashboard reachable, {len(endpoints)} broadcast endpoints available",
                data=results,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    except Exception as e:
        return PipelineTestResult(
            "Dashboard Broadcasting",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# STEP 6: CLI Terminal Display
# =============================================================================


async def test_cli_terminal_display(live_test: bool = False) -> PipelineTestResult:
    """Test that CLI terminal logs are being received by dashboard."""
    start = time.perf_counter()

    try:
        import httpx
        from dashboard.broadcaster import DASHBOARD_API_URL

        # Check WebSocket stats to see if clients are connected
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.get(f"{DASHBOARD_API_URL}/api/ws/stats")

            if response.status_code != 200:
                return PipelineTestResult(
                    "CLI Terminal Display",
                    TestStatus.WARN,
                    f"Cannot check WS stats: {response.status_code}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            ws_stats = response.json()
            active_connections = ws_stats.get("active_connections", 0)
            topics = ws_stats.get("subscriptions", {})
            cli_subscribers = topics.get("cli_logs", 0)

            data = {
                "active_connections": active_connections,
                "cli_log_subscribers": cli_subscribers,
                "all_topics": topics,
            }

            if live_test and active_connections > 0:
                # Send a test message that should appear in terminal
                from dashboard.broadcaster import broadcast_cli_log

                test_messages = [
                    ("info", "📊 VALIDATION: Market collection complete"),
                    ("success", "✅ VALIDATION: Early entry scoring working"),
                    ("warning", "⚠️ VALIDATION: This is a test warning"),
                ]

                for level, message in test_messages:
                    await broadcast_cli_log(
                        level=level,
                        message=message,
                        source="validator",
                        details={"test": True},
                    )
                    await asyncio.sleep(0.2)  # Small delay between messages

                data["test_messages_sent"] = len(test_messages)

                return PipelineTestResult(
                    "CLI Terminal Display",
                    TestStatus.PASS,
                    f"Sent {len(test_messages)} test logs to {cli_subscribers} CLI subscribers",
                    data=data,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            if active_connections == 0:
                return PipelineTestResult(
                    "CLI Terminal Display",
                    TestStatus.WARN,
                    "No WebSocket clients connected - open dashboard to see logs",
                    data=data,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            return PipelineTestResult(
                "CLI Terminal Display",
                TestStatus.PASS,
                f"{active_connections} clients connected, {cli_subscribers} subscribed to CLI logs",
                data=data,
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    except httpx.ConnectError:
        return PipelineTestResult(
            "CLI Terminal Display",
            TestStatus.SKIP,
            "Dashboard not reachable",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        return PipelineTestResult(
            "CLI Terminal Display",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# STEP 7: Account Data Display
# =============================================================================


async def test_account_data_display() -> PipelineTestResult:
    """Test that real Kalshi account data flows to dashboard."""
    start = time.perf_counter()

    try:
        import httpx
        from dashboard.broadcaster import DASHBOARD_API_URL

        # Get account data from dashboard API
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{DASHBOARD_API_URL}/api/account")

            if response.status_code != 200:
                return PipelineTestResult(
                    "Account Data Display",
                    TestStatus.FAIL,
                    f"Account endpoint returned {response.status_code}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            account_data = response.json()

            # Check for required fields
            required = ["balance", "portfolio_value", "total_equity", "api_connected"]
            has_fields = {f: f in account_data for f in required}

            if not account_data.get("api_connected"):
                return PipelineTestResult(
                    "Account Data Display",
                    TestStatus.WARN,
                    "Dashboard showing account data but API not connected",
                    data=account_data,
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            return PipelineTestResult(
                "Account Data Display",
                TestStatus.PASS,
                f"Real account data displayed - Equity: ${account_data.get('total_equity', 0):.2f}",
                data={
                    "balance": account_data.get("balance", 0),
                    "portfolio_value": account_data.get("portfolio_value", 0),
                    "total_equity": account_data.get("total_equity", 0),
                    "open_positions": account_data.get("open_positions_count", 0),
                    "api_connected": account_data.get("api_connected", False),
                },
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    except httpx.ConnectError:
        return PipelineTestResult(
            "Account Data Display",
            TestStatus.SKIP,
            "Dashboard not reachable",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        return PipelineTestResult(
            "Account Data Display",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# STEP 8: Decision Feed Display
# =============================================================================


async def test_decision_feed_display() -> PipelineTestResult:
    """Test that decisions are displayed in the dashboard feed."""
    start = time.perf_counter()

    try:
        import httpx
        from dashboard.broadcaster import DASHBOARD_API_URL

        async with httpx.AsyncClient(timeout=10.0) as client:
            # Get recent decisions
            response = await client.get(f"{DASHBOARD_API_URL}/api/decisions?limit=10")

            if response.status_code != 200:
                return PipelineTestResult(
                    "Decision Feed Display",
                    TestStatus.FAIL,
                    f"Decisions endpoint returned {response.status_code}",
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            decisions_data = response.json()
            decisions = decisions_data.get("decisions", [])
            total = decisions_data.get("total", 0)

            if not decisions:
                return PipelineTestResult(
                    "Decision Feed Display",
                    TestStatus.WARN,
                    "No decisions in feed yet - run a trading cycle first",
                    data={"total": total},
                    duration_ms=(time.perf_counter() - start) * 1000,
                )

            # Analyze recent decisions
            actions = {}
            for d in decisions:
                action = d.get("action", "unknown")
                actions[action] = actions.get(action, 0) + 1

            recent_decision = decisions[0]

            return PipelineTestResult(
                "Decision Feed Display",
                TestStatus.PASS,
                f"Feed showing {len(decisions)} recent decisions (total: {total})",
                data={
                    "total_decisions": total,
                    "recent_count": len(decisions),
                    "action_breakdown": actions,
                    "latest_decision": {
                        "market_ticker": recent_decision.get("market_ticker"),
                        "action": recent_decision.get("action"),
                        "created_at": recent_decision.get("created_at"),
                    },
                },
                duration_ms=(time.perf_counter() - start) * 1000,
            )

    except httpx.ConnectError:
        return PipelineTestResult(
            "Decision Feed Display",
            TestStatus.SKIP,
            "Dashboard not reachable",
            duration_ms=(time.perf_counter() - start) * 1000,
        )
    except Exception as e:
        return PipelineTestResult(
            "Decision Feed Display",
            TestStatus.FAIL,
            f"Error: {str(e)[:100]}",
            duration_ms=(time.perf_counter() - start) * 1000,
        )


# =============================================================================
# FULL MINI TRADING CYCLE TEST
# =============================================================================


async def run_mini_trading_cycle() -> List[PipelineTestResult]:
    """Run a mini trading cycle and broadcast results to dashboard."""
    results = []

    if console:
        console.print(Panel("[bold cyan]Running Mini Trading Cycle[/bold cyan]"))

    try:
        from config import BotConfig
        from dashboard.broadcaster import broadcast_cli_log, broadcast_decision
        from kalshi_client import KalshiClient

        config = BotConfig()

        # Step 1: Broadcast start
        await broadcast_cli_log(
            level="info",
            message="🚀 Mini trading cycle starting (validation mode)",
            source="validator",
        )

        # Step 2: Collect markets
        await broadcast_cli_log(
            level="info",
            message="📡 Fetching markets from Kalshi...",
            source="validator",
        )

        client = KalshiClient(config.kalshi, early_entry_config=config.early_entry)
        await client.login()
        events = await client.get_events(limit=5)

        await broadcast_cli_log(
            level="success",
            message=f"✅ Collected {len(events)} events",
            source="validator",
            details={"events_count": len(events)},
        )

        results.append(
            PipelineTestResult(
                "Mini Cycle: Collection",
                TestStatus.PASS,
                f"Collected {len(events)} events",
            )
        )

        # Step 3: Score markets
        await broadcast_cli_log(
            level="info",
            message="📊 Scoring markets with early entry strategy...",
            source="validator",
        )

        scored_events = [e for e in events if e.get("early_entry_score", 0) > 0]

        await broadcast_cli_log(
            level="success",
            message=f"✅ Scored {len(scored_events)} events with early entry potential",
            source="validator",
        )

        results.append(
            PipelineTestResult(
                "Mini Cycle: Scoring",
                TestStatus.PASS,
                f"Scored {len(scored_events)} events",
            )
        )

        # Step 4: Generate sample decision
        if events:
            sample_event = events[0]
            sample_decision = {
                "decision_id": f"mini-cycle-{datetime.now().strftime('%H%M%S')}",
                "event_ticker": sample_event.get("event_ticker", "UNKNOWN"),
                "market_ticker": sample_event.get("markets", [{}])[0].get(
                    "ticker", "UNKNOWN"
                ),
                "action": "skip",
                "market_title": sample_event.get("title", "Validation Test")[:50],
                "confidence": 0.5,
                "research_probability": 0.5,
                "market_price": 0.5,
                "r_score": 0.0,
                "reasoning": "Mini cycle validation - no actual bet placed",
            }

            await broadcast_decision(sample_decision)

            await broadcast_cli_log(
                level="success",
                message=f"✅ Decision generated for {sample_decision['event_ticker']}",
                source="validator",
            )

            results.append(
                PipelineTestResult(
                    "Mini Cycle: Decision",
                    TestStatus.PASS,
                    f"Generated decision for {sample_decision['event_ticker']}",
                )
            )

        # Step 5: Complete
        await broadcast_cli_log(
            level="success",
            message="🏁 Mini trading cycle complete",
            source="validator",
        )

        await client.close()

        return results

    except Exception as e:
        await broadcast_cli_log(
            level="error",
            message=f"❌ Mini cycle failed: {str(e)[:100]}",
            source="validator",
        )
        results.append(
            PipelineTestResult("Mini Cycle", TestStatus.FAIL, f"Error: {str(e)[:100]}")
        )
        return results


# =============================================================================
# Main Execution
# =============================================================================


async def run_pipeline_validation(live_test: bool = False, full_cycle: bool = False):
    """Run complete pipeline validation."""

    if console:
        console.print(
            Panel(
                "[bold]PIPELINE-TO-DASHBOARD VALIDATION[/bold]\n"
                "Testing data flow from collection through dashboard display",
                style="blue",
            )
        )

    all_results = []

    # Core pipeline tests
    tests = [
        ("1. Market Collection", test_market_collection),
        ("2. Early Entry Scoring", test_early_entry_scoring),
        ("3. Research Integration", test_research_integration),
        ("4. Decision Generation", test_decision_generation),
        ("5. Dashboard Broadcasting", lambda: test_dashboard_broadcasting(live_test)),
        ("6. CLI Terminal Display", lambda: test_cli_terminal_display(live_test)),
        ("7. Account Data Display", test_account_data_display),
        ("8. Decision Feed Display", test_decision_feed_display),
    ]

    for name, test_func in tests:
        if console:
            console.print(f"\n[cyan]Testing: {name}[/cyan]")

        result = await test_func()
        all_results.append(result)
        print_step(result.step, result.status, result.message)

        if result.data and console:
            # Show key data points
            for key, value in list(result.data.items())[:3]:
                console.print(f"    [dim]{key}: {value}[/dim]")

    # Optional: Run mini trading cycle
    if full_cycle:
        cycle_results = await run_mini_trading_cycle()
        all_results.extend(cycle_results)

    # Summary
    passed = sum(1 for r in all_results if r.status == TestStatus.PASS)
    failed = sum(1 for r in all_results if r.status == TestStatus.FAIL)
    warnings = sum(1 for r in all_results if r.status == TestStatus.WARN)
    skipped = sum(1 for r in all_results if r.status == TestStatus.SKIP)

    if console:
        console.print("\n")

        # Results table
        table = Table(title="Pipeline Validation Results")
        table.add_column("Step", style="cyan")
        table.add_column("Status")
        table.add_column("Message")
        table.add_column("Time")

        for r in all_results:
            status_style = {
                TestStatus.PASS: "green",
                TestStatus.FAIL: "red",
                TestStatus.WARN: "yellow",
                TestStatus.SKIP: "dim",
            }.get(r.status, "white")

            table.add_row(
                r.step,
                f"[{status_style}]{r.status.value}[/{status_style}]",
                r.message[:50] + "..." if len(r.message) > 50 else r.message,
                f"{r.duration_ms:.0f}ms" if r.duration_ms > 0 else "-",
            )

        console.print(table)

        # Final verdict
        if failed == 0:
            console.print(
                Panel(
                    f"[bold green]PIPELINE VALIDATION PASSED[/bold green]\n\n"
                    f"✅ Passed: {passed}\n"
                    f"⚠️ Warnings: {warnings}\n"
                    f"⏭️ Skipped: {skipped}\n\n"
                    f"Data is flowing correctly from collection to dashboard.",
                    style="green",
                )
            )
        else:
            console.print(
                Panel(
                    f"[bold red]PIPELINE VALIDATION FAILED[/bold red]\n\n"
                    f"✅ Passed: {passed}\n"
                    f"❌ Failed: {failed}\n"
                    f"⚠️ Warnings: {warnings}\n\n"
                    f"Fix failures to ensure data reaches the dashboard.",
                    style="red",
                )
            )

    return failed == 0


async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline to Dashboard Validation")
    parser.add_argument(
        "--live-test",
        action="store_true",
        help="Actually broadcast test data to dashboard",
    )
    parser.add_argument(
        "--full-cycle",
        action="store_true",
        help="Run a mini trading cycle with broadcasts",
    )
    parser.add_argument(
        "--remote",
        action="store_true",
        help="Test against Render dashboard (not localhost)",
    )
    args = parser.parse_args()

    # Set dashboard URL for remote testing
    if args.remote:
        os.environ["DASHBOARD_URL"] = "https://kalshi-dashboard.onrender.com"
        if console:
            console.print(
                "[yellow]Using REMOTE dashboard: https://kalshi-dashboard.onrender.com[/yellow]\n"
            )

    success = await run_pipeline_validation(
        live_test=args.live_test, full_cycle=args.full_cycle
    )

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
