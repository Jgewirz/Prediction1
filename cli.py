"""
CLI Commands for Kalshi Trading Bot
====================================
Position monitoring, liquidation, and management commands.

Usage:
    uv run python cli.py positions         # Show all positions
    uv run python cli.py liquidate TICKER  # Liquidate specific position
    uv run python cli.py liquidate-all     # Emergency liquidate all
    uv run python cli.py monitor           # Start position monitor only
    uv run python cli.py triggers          # Show positions near triggers
    uv run python cli.py exits             # Show exit history
    uv run python cli.py set-sl DECISION_ID 0.10  # Set stop-loss to 10%
    uv run python cli.py set-tp DECISION_ID 0.25  # Set take-profit to 25%
"""

import asyncio
import sys
from datetime import datetime, timezone
from typing import Optional

import click
# Local imports
from config import load_config
from db import get_database, get_postgres_database
from kalshi_client import KalshiClient
from loguru import logger
from position_monitor import PositionMonitor, StopLossConfig
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress
from rich.table import Table

console = Console()


async def get_db_and_client():
    """Initialize database and Kalshi client."""
    config = load_config()

    # Get database
    if config.database.db_type.lower() == "postgres":
        db = await get_postgres_database(
            host=config.database.pg_host,
            database=config.database.pg_database,
            user=config.database.pg_user,
            password=config.database.pg_password,
            port=config.database.pg_port,
            ssl=config.database.pg_ssl,
        )
    else:
        db = await get_database(config.database.db_path)

    # Get Kalshi client
    kalshi = KalshiClient(config.kalshi)
    await kalshi.login()

    return config, db, kalshi


@click.group()
def cli():
    """Kalshi Trading Bot CLI - Position Management Commands"""
    pass


@cli.command()
@click.option("--live", is_flag=True, help="Show live positions from API (not just DB)")
def positions(live: bool):
    """Show all current positions with P&L."""

    async def _show_positions():
        config, db, kalshi = await get_db_and_client()

        if live:
            # Get positions directly from Kalshi API
            api_positions = await kalshi.get_user_positions()

            table = Table(title="Live Positions (from Kalshi API)")
            table.add_column("Ticker", style="cyan")
            table.add_column("Side", style="green")
            table.add_column("Contracts", justify="right")
            table.add_column("Avg Price", justify="right")

            for pos in api_positions:
                position_size = pos.get("position", 0)
                if position_size == 0:
                    continue

                side = "YES" if position_size > 0 else "NO"
                contracts = abs(position_size)

                table.add_row(
                    pos.get("ticker", ""),
                    side,
                    str(contracts),
                    "-",  # API doesn't provide avg price directly
                )

            console.print(table)
            console.print(
                f"\nTotal positions: {len([p for p in api_positions if p.get('position', 0) != 0])}"
            )

        else:
            # Get positions from database
            query = """
                SELECT
                    decision_id, market_ticker, action,
                    filled_price_cents, filled_contracts,
                    current_price_cents, unrealized_pnl_dollars, unrealized_pnl_pct,
                    stop_loss_pct, take_profit_pct,
                    high_water_mark_cents, last_price_update
                FROM betting_decisions
                WHERE status = 'pending'
                  AND action IN ('buy_yes', 'buy_no')
                  AND filled_contracts > 0
                  AND exit_order_id IS NULL
                ORDER BY unrealized_pnl_dollars DESC
            """

            try:
                rows = await db.fetchall(query)
            except Exception as e:
                console.print(f"[red]Error querying database: {e}[/red]")
                return

            table = Table(title="Positions from Database")
            table.add_column("Ticker", style="cyan", max_width=30)
            table.add_column("Side", style="green")
            table.add_column("Contracts", justify="right")
            table.add_column("Entry", justify="right")
            table.add_column("Current", justify="right")
            table.add_column("P&L $", justify="right")
            table.add_column("P&L %", justify="right")
            table.add_column("SL %", justify="right", style="red")
            table.add_column("TP %", justify="right", style="green")

            total_pnl = 0
            for row in rows:
                pnl = float(row.get("unrealized_pnl_dollars") or 0)
                pnl_pct = float(row.get("unrealized_pnl_pct") or 0) * 100
                total_pnl += pnl

                pnl_style = "green" if pnl >= 0 else "red"

                table.add_row(
                    row["market_ticker"][:30],
                    "YES" if row["action"] == "buy_yes" else "NO",
                    str(row.get("filled_contracts") or 0),
                    f"{row.get('filled_price_cents') or 0}c",
                    f"{row.get('current_price_cents') or 0}c",
                    f"[{pnl_style}]${pnl:.2f}[/{pnl_style}]",
                    f"[{pnl_style}]{pnl_pct:.1f}%[/{pnl_style}]",
                    f"{(row.get('stop_loss_pct') or 0.15) * 100:.0f}%",
                    f"{(row.get('take_profit_pct') or 0.30) * 100:.0f}%",
                )

            console.print(table)

            total_style = "green" if total_pnl >= 0 else "red"
            console.print(f"\n[bold]Total positions:[/bold] {len(rows)}")
            console.print(
                f"[bold]Total Unrealized P&L:[/bold] [{total_style}]${total_pnl:.2f}[/{total_style}]"
            )

        await kalshi.close()

    asyncio.run(_show_positions())


@cli.command()
@click.argument("ticker")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def liquidate(ticker: str, confirm: bool):
    """Liquidate a specific position by ticker."""

    if not confirm:
        if not click.confirm(f"Are you sure you want to liquidate {ticker}?"):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    async def _liquidate():
        config, db, kalshi = await get_db_and_client()

        console.print(f"[yellow]Liquidating position: {ticker}...[/yellow]")

        result = await kalshi.liquidate_position(ticker)

        if result.get("success"):
            console.print(
                Panel(
                    f"[green]Position liquidated successfully![/green]\n\n"
                    f"Ticker: {ticker}\n"
                    f"Side: {result.get('side', 'unknown').upper()}\n"
                    f"Contracts: {result.get('contracts', 0)}\n"
                    f"Order ID: {result.get('order_id', 'N/A')}",
                    title="Liquidation Complete",
                )
            )
        else:
            console.print(
                f"[red]Liquidation failed: {result.get('error', 'Unknown error')}[/red]"
            )

        await kalshi.close()

    asyncio.run(_liquidate())


@cli.command("liquidate-all")
@click.option("--confirm", is_flag=True, help="Skip confirmation prompt")
def liquidate_all(confirm: bool):
    """EMERGENCY: Liquidate ALL open positions."""

    console.print("[bold red]WARNING: This will close ALL positions![/bold red]")

    if not confirm:
        if not click.confirm(
            "Are you ABSOLUTELY sure you want to liquidate ALL positions?"
        ):
            console.print("[yellow]Cancelled.[/yellow]")
            return

    async def _liquidate_all():
        config, db, kalshi = await get_db_and_client()

        console.print("[red bold]EMERGENCY LIQUIDATION IN PROGRESS...[/red bold]")

        result = await kalshi.liquidate_all_positions()

        if result.get("success"):
            liquidated = result.get("liquidated", [])
            failed = result.get("failed", [])

            console.print(
                Panel(
                    f"[green]Liquidation complete![/green]\n\n"
                    f"Positions closed: {len(liquidated)}\n"
                    f"Failed: {len(failed)}",
                    title="Emergency Liquidation Results",
                )
            )

            if liquidated:
                table = Table(title="Liquidated Positions")
                table.add_column("Ticker")
                table.add_column("Side")
                table.add_column("Contracts")
                table.add_column("Order ID")

                for pos in liquidated:
                    table.add_row(
                        pos.get("ticker", ""),
                        pos.get("side", "").upper(),
                        str(pos.get("contracts", 0)),
                        pos.get("order_id", "")[:20] + "...",
                    )

                console.print(table)

            if failed:
                console.print("\n[red]Failed liquidations:[/red]")
                for pos in failed:
                    console.print(f"  - {pos.get('ticker')}: {pos.get('error')}")

        else:
            console.print(
                f"[red]Liquidation failed: {result.get('error', 'Unknown error')}[/red]"
            )

        await kalshi.close()

    asyncio.run(_liquidate_all())


@cli.command()
@click.option("--interval", default=30, help="Monitor interval in seconds")
def monitor(interval: int):
    """Start position monitor (stop-loss/take-profit) only."""

    async def _run_monitor():
        config, db, kalshi = await get_db_and_client()

        sl_config = StopLossConfig(
            enabled=True,
            default_stop_loss_pct=config.stop_loss.default_stop_loss_pct,
            default_take_profit_pct=config.stop_loss.default_take_profit_pct,
            monitor_interval_seconds=interval,
            trailing_stop_enabled=config.stop_loss.trailing_stop_enabled,
            trailing_stop_pct=config.stop_loss.trailing_stop_pct,
        )

        def on_trigger(event):
            console.print(
                Panel(
                    f"[bold {'red' if event.trigger_type.value == 'stop_loss' else 'green'}]"
                    f"{event.trigger_type.value.upper()} TRIGGERED[/bold]\n\n"
                    f"Ticker: {event.position.market_ticker}\n"
                    f"P&L: ${event.position.unrealized_pnl_dollars:.2f} "
                    f"({event.position.unrealized_pnl_pct * 100:.1f}%)\n"
                    f"Price: {event.trigger_price_cents}c",
                    title="Trigger Alert",
                )
            )

        monitor = PositionMonitor(
            config=config,
            kalshi=kalshi,
            db=db,
            sl_config=sl_config,
            on_trigger=on_trigger,
        )

        console.print(
            f"[green]Starting position monitor (interval: {interval}s)...[/green]"
        )
        console.print("Press Ctrl+C to stop.\n")

        await monitor.start()

        try:
            while True:
                await asyncio.sleep(1)
        except KeyboardInterrupt:
            console.print("\n[yellow]Stopping monitor...[/yellow]")
            await monitor.stop()

        await kalshi.close()

    asyncio.run(_run_monitor())


@cli.command()
@click.option("--threshold", default=0.05, help="Distance threshold (e.g., 0.05 = 5%)")
def triggers(threshold: float):
    """Show positions near stop-loss or take-profit triggers."""

    async def _show_triggers():
        config, db, kalshi = await get_db_and_client()

        query = """
            SELECT
                decision_id, market_ticker, action,
                filled_price_cents, filled_contracts,
                current_price_cents, unrealized_pnl_pct,
                stop_loss_pct, take_profit_pct
            FROM betting_decisions
            WHERE status = 'pending'
              AND action IN ('buy_yes', 'buy_no')
              AND filled_contracts > 0
              AND exit_order_id IS NULL
        """

        try:
            rows = await db.fetchall(query)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        near_sl = []
        near_tp = []

        for row in rows:
            pnl_pct = float(row.get("unrealized_pnl_pct") or 0)
            sl_pct = float(row.get("stop_loss_pct") or 0.15)
            tp_pct = float(row.get("take_profit_pct") or 0.30)

            sl_distance = pnl_pct + sl_pct
            tp_distance = tp_pct - pnl_pct

            if 0 < sl_distance < threshold:
                near_sl.append({**dict(row), "distance": sl_distance})
            if 0 < tp_distance < threshold:
                near_tp.append({**dict(row), "distance": tp_distance})

        if near_sl:
            table = Table(
                title=f"[red]Near Stop-Loss (within {threshold*100:.0f}%)[/red]"
            )
            table.add_column("Ticker")
            table.add_column("Distance", justify="right")
            table.add_column("P&L %", justify="right")

            for pos in sorted(near_sl, key=lambda x: x["distance"]):
                table.add_row(
                    pos["market_ticker"][:30],
                    f"{pos['distance']*100:.1f}%",
                    f"{float(pos.get('unrealized_pnl_pct') or 0)*100:.1f}%",
                )

            console.print(table)

        if near_tp:
            table = Table(
                title=f"[green]Near Take-Profit (within {threshold*100:.0f}%)[/green]"
            )
            table.add_column("Ticker")
            table.add_column("Distance", justify="right")
            table.add_column("P&L %", justify="right")

            for pos in sorted(near_tp, key=lambda x: x["distance"]):
                table.add_row(
                    pos["market_ticker"][:30],
                    f"{pos['distance']*100:.1f}%",
                    f"{float(pos.get('unrealized_pnl_pct') or 0)*100:.1f}%",
                )

            console.print(table)

        if not near_sl and not near_tp:
            console.print(
                f"[dim]No positions within {threshold*100:.0f}% of triggers.[/dim]"
            )

        await kalshi.close()

    asyncio.run(_show_triggers())


@cli.command()
@click.option("--days", default=7, help="Number of days to look back")
@click.option("--reason", default=None, help="Filter by exit reason")
def exits(days: int, reason: Optional[str]):
    """Show exit history (stop-loss, take-profit, manual)."""

    async def _show_exits():
        config, db, kalshi = await get_db_and_client()

        conditions = ["exit_order_id IS NOT NULL"]
        if reason:
            conditions.append(f"exit_reason = '{reason}'")

        query = f"""
            SELECT
                market_ticker, action, exit_reason,
                filled_price_cents, exit_price_cents,
                exit_pnl_dollars, exit_timestamp
            FROM betting_decisions
            WHERE {' AND '.join(conditions)}
            ORDER BY exit_timestamp DESC
            LIMIT 50
        """

        try:
            rows = await db.fetchall(query)
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            return

        if not rows:
            console.print("[dim]No exits found.[/dim]")
            return

        table = Table(title=f"Exit History (last {days} days)")
        table.add_column("Ticker", max_width=25)
        table.add_column("Side")
        table.add_column("Reason")
        table.add_column("Entry", justify="right")
        table.add_column("Exit", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("Time")

        total_pnl = 0
        for row in rows:
            pnl = float(row.get("exit_pnl_dollars") or 0)
            total_pnl += pnl
            pnl_style = "green" if pnl >= 0 else "red"

            reason_style = {
                "stop_loss": "red",
                "take_profit": "green",
                "trailing_stop": "yellow",
                "manual": "blue",
            }.get(row.get("exit_reason"), "white")

            table.add_row(
                row["market_ticker"][:25],
                "YES" if row["action"] == "buy_yes" else "NO",
                f"[{reason_style}]{row.get('exit_reason', 'unknown')}[/{reason_style}]",
                f"{row.get('filled_price_cents') or 0}c",
                f"{row.get('exit_price_cents') or 0}c",
                f"[{pnl_style}]${pnl:.2f}[/{pnl_style}]",
                str(row.get("exit_timestamp", ""))[:16],
            )

        console.print(table)

        total_style = "green" if total_pnl >= 0 else "red"
        console.print(f"\n[bold]Total exits:[/bold] {len(rows)}")
        console.print(
            f"[bold]Total P&L:[/bold] [{total_style}]${total_pnl:.2f}[/{total_style}]"
        )

        await kalshi.close()

    asyncio.run(_show_exits())


@cli.command("set-sl")
@click.argument("decision_id")
@click.argument("stop_loss_pct", type=float)
def set_stop_loss(decision_id: str, stop_loss_pct: float):
    """Set stop-loss percentage for a position."""

    async def _set_sl():
        config, db, kalshi = await get_db_and_client()

        query = """
            UPDATE betting_decisions
            SET stop_loss_pct = $1
            WHERE decision_id = $2
            RETURNING decision_id, market_ticker, stop_loss_pct
        """

        try:
            row = await db.fetchone(query, stop_loss_pct, decision_id)
            if row:
                console.print(
                    f"[green]Updated stop-loss for {row['market_ticker']} to {stop_loss_pct*100:.0f}%[/green]"
                )
            else:
                console.print(f"[red]Position not found: {decision_id}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        await kalshi.close()

    asyncio.run(_set_sl())


@cli.command("set-tp")
@click.argument("decision_id")
@click.argument("take_profit_pct", type=float)
def set_take_profit(decision_id: str, take_profit_pct: float):
    """Set take-profit percentage for a position."""

    async def _set_tp():
        config, db, kalshi = await get_db_and_client()

        query = """
            UPDATE betting_decisions
            SET take_profit_pct = $1
            WHERE decision_id = $2
            RETURNING decision_id, market_ticker, take_profit_pct
        """

        try:
            row = await db.fetchone(query, take_profit_pct, decision_id)
            if row:
                console.print(
                    f"[green]Updated take-profit for {row['market_ticker']} to {take_profit_pct*100:.0f}%[/green]"
                )
            else:
                console.print(f"[red]Position not found: {decision_id}[/red]")
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        await kalshi.close()

    asyncio.run(_set_tp())


@cli.command()
def balance():
    """Show current Kalshi account balance."""

    async def _show_balance():
        config, db, kalshi = await get_db_and_client()

        try:
            # Get balance from API (if available)
            positions = await kalshi.get_user_positions()
            position_count = len([p for p in positions if p.get("position", 0) != 0])

            console.print(
                Panel(
                    f"[bold]Environment:[/bold] {'DEMO' if config.kalshi.use_demo else 'PRODUCTION'}\n"
                    f"[bold]Open positions:[/bold] {position_count}\n"
                    f"[bold]Configured bankroll:[/bold] ${config.bankroll:.2f}",
                    title="Account Status",
                )
            )
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

        await kalshi.close()

    asyncio.run(_show_balance())


if __name__ == "__main__":
    cli()
