"""
Reconciliation Engine for Kalshi Deep Trading Bot.

Automatically tracks outcomes and calculates P&L for betting decisions.
"""

import asyncio
from datetime import date, datetime
# Type alias for database (can be SQLite or PostgreSQL)
from typing import Any, Dict, List, Optional, Union

from config import load_config
from db import Database, get_database
from db.postgres import PostgresDatabase, get_postgres_database
from db.queries import Queries
from kalshi_client import KalshiClient
from loguru import logger
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

DatabaseType = Union[Database, PostgresDatabase]


class ReconciliationEngine:
    """Automated outcome tracking and P&L calculation."""

    def __init__(self, db: Database, kalshi_client: KalshiClient):
        self.db = db
        self.kalshi_client = kalshi_client
        self.console = Console()

    async def reconcile_pending_decisions(self) -> Dict[str, Any]:
        """
        Check and update outcomes for all pending decisions.

        Returns:
            Dict with reconciliation summary statistics
        """
        self.console.print("[bold blue]Starting reconciliation...[/bold blue]")

        # Get all pending decisions
        pending = await self.db.get_pending_decisions()

        if not pending:
            self.console.print("[yellow]No pending decisions to reconcile[/yellow]")
            return {"checked": 0, "settled": 0, "still_pending": 0}

        self.console.print(f"Found {len(pending)} pending decisions to check")

        # Get unique market tickers
        tickers = list(set(d["market_ticker"] for d in pending))
        self.console.print(f"Checking {len(tickers)} unique markets")

        # Fetch market status in batches
        settled_markets = await self.kalshi_client.get_settled_markets(tickers)

        # Create lookup dict
        settled_lookup = {m["ticker"]: m for m in settled_markets}

        # Process each pending decision
        settled_count = 0
        still_pending = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Processing decisions...", total=len(pending))

            for decision in pending:
                ticker = decision["market_ticker"]
                progress.update(task, description=f"Processing {ticker[:20]}...")

                if ticker in settled_lookup:
                    market = settled_lookup[ticker]
                    outcome = market.get("result", "")

                    if outcome:
                        # Calculate P&L
                        payout, profit_loss = self.calculate_pnl(
                            action=decision["action"],
                            amount=decision["bet_amount"],
                            outcome=outcome,
                        )

                        # Update decision in database
                        await self.db.update_decision_outcome(
                            decision_id=decision["decision_id"],
                            outcome=outcome,
                            payout=payout,
                            profit_loss=profit_loss,
                        )

                        # Also update calibration if exists
                        await self._update_calibration_for_decision(
                            decision, outcome, payout
                        )

                        settled_count += 1
                        logger.info(
                            f"Settled {ticker}: {outcome}, P&L: ${profit_loss:.2f}"
                        )
                else:
                    still_pending += 1

                progress.advance(task)

        # Update daily performance aggregations
        await self.aggregate_daily_performance()

        summary = {
            "checked": len(pending),
            "settled": settled_count,
            "still_pending": still_pending,
            "markets_checked": len(tickers),
            "markets_settled": len(settled_markets),
        }

        self._print_reconciliation_summary(summary)
        return summary

    def calculate_pnl(
        self, action: str, amount: float, outcome: str
    ) -> tuple[float, float]:
        """
        Calculate profit/loss for a single decision.

        Args:
            action: buy_yes or buy_no
            amount: Bet amount in dollars
            outcome: yes or no

        Returns:
            Tuple of (payout, profit_loss)
        """
        # Normalize outcome
        outcome = outcome.lower().strip()

        if action == "buy_yes":
            if outcome == "yes":
                # Win: payout is full contract value ($1 per contract)
                # Assuming amount was spent at market price
                payout = amount * 2  # Simplified: double your money on win
                profit_loss = payout - amount
            else:
                # Lose: lose entire bet
                payout = 0.0
                profit_loss = -amount
        elif action == "buy_no":
            if outcome == "no":
                # Win
                payout = amount * 2
                profit_loss = payout - amount
            else:
                # Lose
                payout = 0.0
                profit_loss = -amount
        else:
            # Skip or unknown action
            payout = 0.0
            profit_loss = 0.0

        return payout, profit_loss

    async def _update_calibration_for_decision(
        self, decision: Dict[str, Any], outcome: str, payout: float
    ) -> None:
        """Update calibration record if one exists for this decision."""
        try:
            # Try to find matching calibration record
            ticker = decision["market_ticker"]
            outcome_value = 1.0 if outcome.lower() == "yes" else 0.0

            # Check if there's a calibration record for this ticker
            query = """
                UPDATE calibration_records
                SET outcome = ?, actual_payout = ?, resolved_timestamp = ?
                WHERE ticker = ? AND outcome IS NULL
            """
            await self.db.execute(
                query, (outcome_value, payout, datetime.utcnow().isoformat(), ticker)
            )
            await self.db.commit()

        except Exception as e:
            logger.debug(
                f"Could not update calibration for {decision.get('market_ticker')}: {e}"
            )

    async def aggregate_daily_performance(
        self, target_date: Optional[date] = None
    ) -> None:
        """
        Recalculate daily performance metrics.

        Args:
            target_date: Specific date to aggregate, or None for all dates with settled bets
        """
        try:
            if target_date:
                dates_to_process = [target_date]
            else:
                # Get all unique dates with settled decisions
                query = """
                    SELECT DISTINCT DATE(timestamp) as date
                    FROM betting_decisions
                    WHERE status = 'settled'
                """
                rows = await self.db.fetchall(query)
                dates_to_process = [row["date"] for row in rows]

            for d in dates_to_process:
                # Calculate metrics for this date
                query = """
                    SELECT
                        COUNT(*) as total_bets,
                        SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_bets,
                        SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_bets,
                        SUM(bet_amount) as total_wagered,
                        SUM(payout_amount) as total_payout,
                        SUM(profit_loss) as net_profit_loss,
                        AVG(bet_amount) as avg_bet_size,
                        AVG(r_score) as avg_r_score
                    FROM betting_decisions
                    WHERE DATE(timestamp) = ? AND status = 'settled' AND action != 'skip'
                """
                row = await self.db.fetchone(query, (d,))

                if row and row["total_bets"] > 0:
                    roi = (
                        (row["net_profit_loss"] / row["total_wagered"] * 100)
                        if row["total_wagered"]
                        else 0
                    )
                    win_rate = (
                        row["winning_bets"] / row["total_bets"]
                        if row["total_bets"]
                        else 0
                    )

                    # Count pending for this date
                    pending_query = """
                        SELECT COUNT(*) as pending
                        FROM betting_decisions
                        WHERE DATE(timestamp) = ? AND status = 'pending' AND action != 'skip'
                    """
                    pending_row = await self.db.fetchone(pending_query, (d,))
                    pending_count = pending_row["pending"] if pending_row else 0

                    metrics = {
                        "date": d,
                        "total_bets": row["total_bets"],
                        "winning_bets": row["winning_bets"],
                        "losing_bets": row["losing_bets"],
                        "pending_bets": pending_count,
                        "total_wagered": row["total_wagered"] or 0,
                        "total_payout": row["total_payout"] or 0,
                        "net_profit_loss": row["net_profit_loss"] or 0,
                        "roi_percent": roi,
                        "win_rate": win_rate,
                        "avg_bet_size": row["avg_bet_size"] or 0,
                        "avg_r_score": row["avg_r_score"],
                    }

                    await self.db.upsert_daily_performance(metrics)
                    logger.debug(f"Updated daily performance for {d}")

        except Exception as e:
            logger.error(f"Error aggregating daily performance: {e}")

    def _print_reconciliation_summary(self, summary: Dict[str, Any]) -> None:
        """Print a summary of reconciliation results."""
        table = Table(title="Reconciliation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Decisions Checked", str(summary["checked"]))
        table.add_row("Markets Checked", str(summary["markets_checked"]))
        table.add_row("Newly Settled", str(summary["settled"]))
        table.add_row("Still Pending", str(summary["still_pending"]))

        self.console.print(table)

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get overall performance summary from database."""
        stats = await self.db.get_statistics()

        # Get daily P&L
        daily_pnl = await self.db.fetchall(Queries.GET_DAILY_PNL)

        # Get R-score effectiveness
        r_score_stats = await self.db.fetchall(Queries.GET_R_SCORE_EFFECTIVENESS)

        return {
            "overall": stats,
            "daily_pnl": daily_pnl,
            "r_score_effectiveness": r_score_stats,
        }

    async def print_performance_report(self) -> None:
        """Print a detailed performance report."""
        summary = await self.get_performance_summary()
        overall = summary["overall"]

        # Overall stats table
        self.console.print("\n[bold]Overall Performance[/bold]")
        overall_table = Table()
        overall_table.add_column("Metric", style="cyan")
        overall_table.add_column("Value", style="green")

        overall_table.add_row("Total Decisions", str(overall.get("total_decisions", 0)))
        overall_table.add_row("Settled", str(overall.get("settled_decisions", 0)))
        overall_table.add_row("Pending", str(overall.get("pending_decisions", 0)))
        overall_table.add_row("Total P&L", f"${overall.get('total_pnl', 0):.2f}")
        overall_table.add_row("Win Rate", f"{overall.get('win_rate', 0)*100:.1f}%")

        self.console.print(overall_table)

        # Daily P&L table
        if summary["daily_pnl"]:
            self.console.print("\n[bold]Daily P&L (Last 10 Days)[/bold]")
            daily_table = Table()
            daily_table.add_column("Date")
            daily_table.add_column("Bets")
            daily_table.add_column("Wins")
            daily_table.add_column("P&L")
            daily_table.add_column("Wagered")

            for row in summary["daily_pnl"][:10]:
                pnl_color = "green" if row["daily_pnl"] >= 0 else "red"
                daily_table.add_row(
                    str(row["date"]),
                    str(row["total_bets"]),
                    str(row["wins"]),
                    f"[{pnl_color}]${row['daily_pnl']:.2f}[/{pnl_color}]",
                    f"${row['total_wagered']:.2f}",
                )

            self.console.print(daily_table)

        # R-score effectiveness
        if summary["r_score_effectiveness"]:
            self.console.print("\n[bold]R-Score Effectiveness[/bold]")
            rscore_table = Table()
            rscore_table.add_column("R-Score Bucket")
            rscore_table.add_column("Count")
            rscore_table.add_column("Win Rate")
            rscore_table.add_column("Total P&L")

            for row in summary["r_score_effectiveness"]:
                pnl_color = "green" if (row["total_pnl"] or 0) >= 0 else "red"
                rscore_table.add_row(
                    row["r_score_bucket"],
                    str(row["count"]),
                    f"{(row['win_rate'] or 0)*100:.1f}%",
                    f"[{pnl_color}]${row['total_pnl'] or 0:.2f}[/{pnl_color}]",
                )

            self.console.print(rscore_table)


async def run_reconciliation():
    """Run the reconciliation process."""
    config = load_config()
    console = Console()

    try:
        # Initialize database based on DB_TYPE configuration
        db_type = config.database.db_type.lower()
        if db_type == "postgres":
            db = await get_postgres_database(
                host=config.database.pg_host,
                database=config.database.pg_database,
                user=config.database.pg_user,
                password=config.database.pg_password,
                port=config.database.pg_port,
                ssl=config.database.pg_ssl,
            )
            console.print("[green][OK] PostgreSQL database connected (Neon)[/green]")
        else:
            db = await get_database(config.database.db_path)
            console.print("[green][OK] SQLite database connected[/green]")

        # Initialize Kalshi client
        kalshi = KalshiClient(
            config.kalshi,
            config.minimum_time_remaining_hours,
            config.max_markets_per_event,
        )
        await kalshi.login()

        # Run reconciliation
        engine = ReconciliationEngine(db, kalshi)
        await engine.reconcile_pending_decisions()

        # Print performance report
        await engine.print_performance_report()

    except Exception as e:
        console.print(f"[red]Reconciliation error: {e}[/red]")
        logger.exception("Reconciliation failed")

    finally:
        if "kalshi" in locals():
            await kalshi.close()


if __name__ == "__main__":
    asyncio.run(run_reconciliation())
