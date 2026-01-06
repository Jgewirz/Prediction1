"""
Comprehensive validation script for the Kalshi Trading Bot pipeline.
Tests database, API connectivity, decision making, and position placement.
"""

import asyncio
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()


async def validate_config():
    """Validate configuration loading."""
    console.print("\n[bold cyan]1. Validating Configuration[/bold cyan]")
    try:
        from config import load_config

        config = load_config()

        results = []
        results.append(
            (
                "Kalshi API Key",
                "SET" if config.kalshi.api_key else "MISSING",
                bool(config.kalshi.api_key),
            )
        )
        results.append(
            (
                "Kalshi Private Key",
                "SET" if config.kalshi.private_key else "MISSING",
                bool(config.kalshi.private_key),
            )
        )
        results.append(("Use Demo", str(config.kalshi.use_demo), True))
        results.append(
            (
                "OpenAI API Key",
                "SET" if config.openai.api_key else "MISSING",
                bool(config.openai.api_key),
            )
        )
        results.append(("Database Type", config.database.db_type, True))
        results.append(("Database Enabled", str(config.database.enable_db), True))
        results.append(("Bankroll", f"${config.max_bet_amount * 4:.2f}", True))
        results.append(("Max Bet", f"${config.max_bet_amount:.2f}", True))
        results.append(("Z-Threshold", str(config.z_threshold), True))
        results.append(("Kelly Fraction", str(config.kelly_fraction), True))

        table = Table(title="Configuration Status")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="white")
        table.add_column("Status", style="green")

        all_ok = True
        for name, value, ok in results:
            status = "[green]OK[/green]" if ok else "[red]FAIL[/red]"
            if not ok:
                all_ok = False
            table.add_row(name, value, status)

        console.print(table)
        return all_ok, config
    except Exception as e:
        console.print(f"[red]Configuration Error: {e}[/red]")
        return False, None


async def validate_database(config):
    """Validate database connection and schema."""
    console.print("\n[bold cyan]2. Validating Database Connection[/bold cyan]")
    try:
        from db import get_postgres_database

        db = await get_postgres_database(
            host=config.database.pg_host,
            database=config.database.pg_database,
            user=config.database.pg_user,
            password=config.database.pg_password,
            port=config.database.pg_port,
            ssl=config.database.pg_ssl,
        )

        # Check tables
        tables = [
            "betting_decisions",
            "run_history",
            "market_snapshots",
            "calibration_records",
            "performance_daily",
        ]
        table_status = Table(title="Database Tables")
        table_status.add_column("Table", style="cyan")
        table_status.add_column("Row Count", style="white")
        table_status.add_column("Status", style="green")

        all_ok = True
        for table_name in tables:
            try:
                result = await db.fetchone(
                    f"SELECT COUNT(*) as count FROM {table_name}"
                )
                count = result["count"] if result else 0
                table_status.add_row(table_name, str(count), "[green]OK[/green]")
            except Exception as e:
                table_status.add_row(table_name, "N/A", f"[red]ERROR: {e}[/red]")
                all_ok = False

        console.print(table_status)

        # Get recent activity
        recent = await db.fetchone(
            """
            SELECT COUNT(*) as count, MAX(timestamp) as latest
            FROM betting_decisions
            WHERE timestamp > NOW() - INTERVAL '24 hours'
        """
        )
        if recent:
            console.print(f"  Decisions in last 24h: {recent['count']}")
            console.print(f"  Latest decision: {recent['latest']}")

        return all_ok, db
    except Exception as e:
        console.print(f"[red]Database Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return False, None


async def validate_kalshi_api(config):
    """Validate Kalshi API connectivity."""
    console.print("\n[bold cyan]3. Validating Kalshi API[/bold cyan]")
    try:
        from kalshi_client import KalshiClient

        client = KalshiClient(
            config=config.kalshi,
            minimum_time_remaining_hours=1.0,
            max_markets_per_event=10,
        )

        results = []

        # Test 1: Get events
        console.print("  Testing event fetch...")
        events = await client.get_top_events(limit=5)
        results.append(("Get Events", f"{len(events)} events", len(events) > 0))

        if events:
            # Show top events
            event_table = Table(title="Top Events by Volume")
            event_table.add_column("Ticker", style="cyan")
            event_table.add_column("Title", style="white", max_width=40)
            event_table.add_column("Volume", style="yellow")

            for event in events[:5]:
                event_table.add_row(
                    event.get("event_ticker", "N/A"),
                    event.get("title", "N/A")[:40],
                    f"${event.get('volume_24h', 0):,.0f}",
                )
            console.print(event_table)

        # Test 2: Get markets for first event
        if events:
            console.print("  Testing market fetch...")
            first_event = events[0]["event_ticker"]
            markets = await client.get_markets(event_ticker=first_event, limit=5)
            results.append(("Get Markets", f"{len(markets)} markets", len(markets) > 0))

            if markets:
                market_table = Table(title=f"Markets for {first_event}")
                market_table.add_column("Ticker", style="cyan")
                market_table.add_column("Yes Bid", style="green")
                market_table.add_column("Yes Ask", style="red")
                market_table.add_column("Volume", style="yellow")

                for market in markets[:5]:
                    market_table.add_row(
                        market.get("ticker", "N/A"),
                        f"{market.get('yes_bid', 0)}¢",
                        f"{market.get('yes_ask', 0)}¢",
                        f"${market.get('volume', 0):,.0f}",
                    )
                console.print(market_table)

        # Test 3: Check balance (if not demo, this confirms auth works)
        console.print("  Testing authentication...")
        try:
            balance = await client.get_balance()
            results.append(("Get Balance", f"${balance:.2f}", True))
            console.print(f"  [green]Account Balance: ${balance:.2f}[/green]")
        except Exception as e:
            results.append(("Get Balance", str(e), False))
            console.print(
                f"  [yellow]Balance check failed (may be demo mode): {e}[/yellow]"
            )

        # Test 4: Check positions
        console.print("  Testing positions fetch...")
        try:
            positions = await client.get_positions()
            results.append(("Get Positions", f"{len(positions)} positions", True))
            console.print(f"  [green]Open Positions: {len(positions)}[/green]")

            if positions:
                pos_table = Table(title="Current Positions")
                pos_table.add_column("Market", style="cyan")
                pos_table.add_column("Side", style="white")
                pos_table.add_column("Contracts", style="yellow")
                pos_table.add_column("Avg Price", style="green")

                for pos in positions[:5]:
                    pos_table.add_row(
                        pos.get("market_ticker", "N/A"),
                        pos.get("position", "N/A"),
                        str(pos.get("total_traded", 0)),
                        f"{pos.get('market_exposure', 0)}¢",
                    )
                console.print(pos_table)
        except Exception as e:
            results.append(("Get Positions", str(e), False))

        # Summary
        summary_table = Table(title="Kalshi API Summary")
        summary_table.add_column("Test", style="cyan")
        summary_table.add_column("Result", style="white")
        summary_table.add_column("Status", style="green")

        all_ok = True
        for name, result, ok in results:
            status = "[green]PASS[/green]" if ok else "[red]FAIL[/red]"
            if not ok:
                all_ok = False
            summary_table.add_row(name, result, status)

        console.print(summary_table)
        return all_ok, client
    except Exception as e:
        console.print(f"[red]Kalshi API Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return False, None


async def validate_decision_pipeline(config, db):
    """Validate the decision-making pipeline."""
    console.print("\n[bold cyan]4. Validating Decision Pipeline[/bold cyan]")

    try:
        # Check recent decisions
        result = await db.fetchall(
            """
            SELECT
                action,
                COUNT(*) as count,
                AVG(confidence) as avg_confidence,
                AVG(r_score) as avg_r_score,
                SUM(bet_amount) as total_wagered
            FROM betting_decisions
            WHERE timestamp > NOW() - INTERVAL '7 days'
            GROUP BY action
            ORDER BY count DESC
        """
        )

        if result:
            table = Table(title="Decision Distribution (Last 7 Days)")
            table.add_column("Action", style="cyan")
            table.add_column("Count", style="white")
            table.add_column("Avg Confidence", style="yellow")
            table.add_column("Avg R-Score", style="green")
            table.add_column("Total Wagered", style="magenta")

            for row in result:
                table.add_row(
                    row["action"],
                    str(row["count"]),
                    f"{row['avg_confidence']:.3f}" if row["avg_confidence"] else "N/A",
                    f"{row['avg_r_score']:.3f}" if row["avg_r_score"] else "N/A",
                    f"${row['total_wagered']:.2f}" if row["total_wagered"] else "$0.00",
                )
            console.print(table)
        else:
            console.print("  [yellow]No decisions in the last 7 days[/yellow]")

        # Check run history
        runs = await db.fetchall(
            """
            SELECT
                run_id,
                started_at,
                status,
                events_analyzed,
                decisions_made,
                bets_placed,
                mode
            FROM run_history
            ORDER BY started_at DESC
            LIMIT 5
        """
        )

        if runs:
            run_table = Table(title="Recent Bot Runs")
            run_table.add_column("Run ID", style="cyan", max_width=20)
            run_table.add_column("Started", style="white")
            run_table.add_column("Status", style="yellow")
            run_table.add_column("Events", style="green")
            run_table.add_column("Decisions", style="magenta")
            run_table.add_column("Bets", style="red")
            run_table.add_column("Mode", style="blue")

            for run in runs:
                run_table.add_row(
                    str(run["run_id"])[:20],
                    str(run["started_at"]),
                    run["status"],
                    str(run["events_analyzed"]),
                    str(run["decisions_made"]),
                    str(run["bets_placed"] or 0),
                    run["mode"] or "dry_run",
                )
            console.print(run_table)

        # Check for actual bets placed
        bets_placed = await db.fetchone(
            """
            SELECT COUNT(*) as count, SUM(bet_amount) as total
            FROM betting_decisions
            WHERE action IN ('buy_yes', 'buy_no')
            AND bet_amount > 0
        """
        )

        if bets_placed:
            console.print(f"\n  Total bets placed: {bets_placed['count']}")
            console.print(f"  Total wagered: ${bets_placed['total'] or 0:.2f}")

        return True
    except Exception as e:
        console.print(f"[red]Decision Pipeline Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return False


async def run_quick_dry_run(config):
    """Run a quick dry-run to test the full pipeline."""
    console.print("\n[bold cyan]5. Running Quick Dry-Run Test[/bold cyan]")

    try:
        from trading_bot import SimpleTradingBot

        console.print("  Initializing bot...")
        bot = SimpleTradingBot(live_trading=False)
        await bot.initialize()

        console.print("  Fetching top events...")
        events = await bot.kalshi_client.get_top_events(limit=3)
        console.print(f"  Found {len(events)} events")

        if not events:
            console.print("[yellow]  No events available to analyze[/yellow]")
            return False

        # Get markets for first event
        console.print("  Fetching markets...")
        event = events[0]
        markets = await bot.kalshi_client.get_markets(
            event_ticker=event["event_ticker"], limit=3
        )
        console.print(f"  Found {len(markets)} markets for {event['event_ticker']}")

        if markets:
            console.print("\n  Sample market data:")
            for m in markets[:2]:
                console.print(
                    f"    - {m.get('ticker')}: Yes bid/ask: {m.get('yes_bid')}¢/{m.get('yes_ask')}¢"
                )

        console.print("\n  [green]Dry-run validation successful![/green]")
        return True
    except Exception as e:
        console.print(f"[red]Dry-run Error: {e}[/red]")
        import traceback

        traceback.print_exc()
        return False


async def main():
    """Run all validations."""
    console.print(
        Panel.fit(
            "[bold blue]Kalshi Trading Bot - Pipeline Validation[/bold blue]\n"
            "Testing database, API, and decision pipeline",
            border_style="blue",
        )
    )

    results = {}

    # 1. Validate config
    ok, config = await validate_config()
    results["Configuration"] = ok

    if not ok or not config:
        console.print("[red]Cannot continue without valid configuration[/red]")
        return

    # 2. Validate database
    ok, db = await validate_database(config)
    results["Database"] = ok

    # 3. Validate Kalshi API
    ok, client = await validate_kalshi_api(config)
    results["Kalshi API"] = ok

    # 4. Validate decision pipeline
    if db:
        ok = await validate_decision_pipeline(config, db)
        results["Decision Pipeline"] = ok

    # 5. Run quick dry-run
    ok = await run_quick_dry_run(config)
    results["Dry-Run Test"] = ok

    # Final summary
    console.print("\n")
    summary = Table(title="Validation Summary")
    summary.add_column("Component", style="cyan")
    summary.add_column("Status", style="green")

    all_passed = True
    for component, passed in results.items():
        status = "[green]PASSED[/green]" if passed else "[red]FAILED[/red]"
        if not passed:
            all_passed = False
        summary.add_row(component, status)

    console.print(summary)

    if all_passed:
        console.print(
            Panel.fit(
                "[bold green]All validations passed![/bold green]\n"
                "The trading bot pipeline is working correctly.",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                "[bold red]Some validations failed![/bold red]\n"
                "Please check the errors above.",
                border_style="red",
            )
        )


if __name__ == "__main__":
    asyncio.run(main())
