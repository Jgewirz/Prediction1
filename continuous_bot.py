"""
Continuous trading bot with APScheduler.
Runs trading cycles every 15 minutes with automatic error recovery.
"""
import asyncio
import argparse
import signal
import sys
from datetime import datetime
from pathlib import Path
from loguru import logger
from rich.console import Console

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from trading_bot import SimpleTradingBot
from reconciliation import run_reconciliation
from config import load_config
from position_monitor import PositionMonitor, create_position_monitor
from dashboard.broadcaster import broadcast_status, broadcast_alert, broadcast_kpi_update


class ContinuousTradingBot:
    """Continuous trading bot that runs on a schedule."""

    def __init__(self, live_trading: bool = False):
        self.config = load_config()
        self.live_trading = live_trading
        self.scheduler = AsyncIOScheduler()
        self.console = Console()
        self.running = True
        self.consecutive_failures = 0
        self.total_runs = 0
        self.successful_runs = 0
        self.last_run_time = None
        self.position_monitor: PositionMonitor = None

        # Create logs directory if it doesn't exist
        Path("logs").mkdir(exist_ok=True)

    async def trading_job(self):
        """Execute a single trading cycle."""
        self.total_runs += 1
        run_start = datetime.now()

        self.console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        self.console.print(f"[bold cyan]Starting trading run #{self.total_runs} at {run_start.strftime('%Y-%m-%d %H:%M:%S')}[/bold cyan]")
        self.console.print(f"[bold cyan]{'='*60}[/bold cyan]\n")

        logger.info(f"=== Starting trading run #{self.total_runs} ===")

        # Broadcast status: trading run starting
        await broadcast_status(
            bot_running=True,
            mode='live' if self.live_trading else 'dry_run',
            additional_info={
                'run_number': self.total_runs,
                'run_start': run_start.isoformat(),
                'status': 'running'
            }
        )

        try:
            bot = SimpleTradingBot(live_trading=self.live_trading)
            await bot.run()

            # Initialize position monitor with bot's clients if not already done
            if self.position_monitor is None and self.config.stop_loss.enabled and self.live_trading:
                try:
                    self.position_monitor = PositionMonitor(
                        config=self.config,
                        kalshi=bot.kalshi_client,
                        db=bot.db
                    )
                    self.console.print("[green]Position monitor initialized[/green]")
                    logger.info("Position monitor initialized successfully")
                except Exception as e:
                    logger.error(f"Failed to initialize position monitor: {e}")

            self.consecutive_failures = 0
            self.successful_runs += 1
            self.last_run_time = datetime.now()

            duration = (datetime.now() - run_start).total_seconds()
            logger.info(f"Trading run #{self.total_runs} completed in {duration:.1f}s")
            self.console.print(f"\n[green]Trading run #{self.total_runs} completed in {duration:.1f}s[/green]")

            # Broadcast status: run completed successfully
            await broadcast_status(
                bot_running=True,
                mode='live' if self.live_trading else 'dry_run',
                additional_info={
                    'run_number': self.total_runs,
                    'last_run_completed': self.last_run_time.isoformat(),
                    'duration_seconds': duration,
                    'status': 'idle',
                    'successful_runs': self.successful_runs
                }
            )
            # Update KPIs after successful run
            await broadcast_kpi_update()

        except Exception as e:
            self.consecutive_failures += 1
            logger.error(f"Trading run failed (failure #{self.consecutive_failures}): {e}")
            self.console.print(f"\n[red]Trading run #{self.total_runs} failed: {e}[/red]")

            # Broadcast alert on failure
            await broadcast_alert(
                message=f"Trading run #{self.total_runs} failed: {str(e)[:100]}",
                severity="error",
                details={
                    'run_number': self.total_runs,
                    'consecutive_failures': self.consecutive_failures,
                    'error': str(e)
                }
            )

            if self.consecutive_failures >= self.config.scheduler.max_consecutive_failures:
                logger.critical("Max consecutive failures reached, pausing trading")
                self.console.print(f"[bold red]CRITICAL: {self.consecutive_failures} consecutive failures - trading paused[/bold red]")
                # Broadcast critical alert
                await broadcast_alert(
                    message=f"CRITICAL: {self.consecutive_failures} consecutive failures - trading paused",
                    severity="critical",
                    details={'consecutive_failures': self.consecutive_failures}
                )

        # Show next scheduled run
        next_run = self.scheduler.get_job('trading')
        if next_run and next_run.next_run_time:
            self.console.print(f"[dim]Next run scheduled at: {next_run.next_run_time.strftime('%H:%M:%S')}[/dim]")

    async def reconciliation_job(self):
        """Reconcile outcomes for settled markets."""
        logger.info("Starting reconciliation job")
        self.console.print("\n[yellow]Running reconciliation...[/yellow]")
        try:
            await run_reconciliation()
            logger.info("Reconciliation completed")
            self.console.print("[green]Reconciliation completed[/green]")
        except Exception as e:
            logger.error(f"Reconciliation failed: {e}")
            self.console.print(f"[red]Reconciliation failed: {e}[/red]")

    async def health_check_job(self):
        """Check health of external services."""
        logger.debug("Running health check")
        # Future: Check Kalshi API, TrendRadar, Database connectivity
        # For now, just log that we're alive
        pass

    async def position_monitor_job(self):
        """Check positions for stop-loss/take-profit triggers."""
        if not self.config.stop_loss.enabled:
            return

        if self.position_monitor is None:
            logger.debug("Position monitor not initialized, skipping check")
            return

        try:
            triggers = await self.position_monitor.check_all_positions()
            if triggers:
                for trigger in triggers:
                    self.console.print(
                        f"[yellow]EXIT TRIGGERED: {trigger.trigger_type.value.upper()} "
                        f"on {trigger.position.market_ticker} "
                        f"P&L: ${trigger.position.unrealized_pnl_dollars:.2f}[/yellow]"
                    )
                    logger.info(f"Exit triggered: {trigger.to_dict()}")
        except Exception as e:
            logger.error(f"Position monitor check failed: {e}")

    def setup_signal_handlers(self):
        """Setup graceful shutdown handlers."""
        def shutdown(signum, frame):
            logger.info(f"Received signal {signum}, shutting down...")
            self.console.print(f"\n[yellow]Received shutdown signal, stopping...[/yellow]")
            self.running = False
            if self.scheduler.running:
                self.scheduler.shutdown(wait=False)

        # Handle both SIGINT (Ctrl+C) and SIGTERM
        signal.signal(signal.SIGINT, shutdown)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, shutdown)

    async def run(self):
        """Start the continuous trading system."""
        self.setup_signal_handlers()

        # Configure logging for continuous operation
        logger.add(
            "logs/trading_bot_{time:YYYY-MM-DD}.log",
            rotation="1 day",
            retention="7 days",
            level="INFO",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {message}"
        )

        # Display startup banner
        mode = "LIVE TRADING" if self.live_trading else "DRY RUN"
        mode_color = "red" if self.live_trading else "green"

        self.console.print("\n[bold blue]" + "=" * 60 + "[/bold blue]")
        self.console.print("[bold blue]      KALSHI CONTINUOUS TRADING BOT[/bold blue]")
        self.console.print("[bold blue]" + "=" * 60 + "[/bold blue]")
        self.console.print(f"[{mode_color}]Mode: {mode}[/{mode_color}]")
        self.console.print(f"Trading interval: {self.config.scheduler.trading_interval_minutes} minutes")
        self.console.print(f"Reconciliation interval: {self.config.scheduler.reconciliation_interval_minutes} minutes")
        self.console.print(f"Max consecutive failures: {self.config.scheduler.max_consecutive_failures}")
        self.console.print("[dim]Press Ctrl+C to stop[/dim]")
        self.console.print("[bold blue]" + "=" * 60 + "[/bold blue]\n")

        logger.info(f"Starting continuous trading bot ({mode} mode)")
        logger.info(f"Trading interval: {self.config.scheduler.trading_interval_minutes} minutes")
        logger.info(f"Reconciliation interval: {self.config.scheduler.reconciliation_interval_minutes} minutes")

        # Schedule jobs
        self.scheduler.add_job(
            self.trading_job,
            IntervalTrigger(minutes=self.config.scheduler.trading_interval_minutes),
            id='trading',
            name='Trading Cycle',
            max_instances=1,  # Prevent overlapping runs
            coalesce=True     # Combine missed runs
        )

        self.scheduler.add_job(
            self.reconciliation_job,
            IntervalTrigger(minutes=self.config.scheduler.reconciliation_interval_minutes),
            id='reconciliation',
            name='Reconciliation',
            max_instances=1
        )

        self.scheduler.add_job(
            self.health_check_job,
            IntervalTrigger(minutes=self.config.scheduler.health_check_interval_minutes),
            id='health_check',
            name='Health Check'
        )

        # Schedule position monitoring if enabled
        if self.config.stop_loss.enabled:
            self.scheduler.add_job(
                self.position_monitor_job,
                IntervalTrigger(seconds=self.config.stop_loss.monitor_interval_seconds),
                id='position_monitor',
                name='Position Monitor',
                max_instances=1
            )
            self.console.print(f"Position monitoring interval: {self.config.stop_loss.monitor_interval_seconds} seconds")

        # Start scheduler
        self.scheduler.start()

        # Run first trading job immediately
        await self.trading_job()

        # Keep running until shutdown signal
        try:
            while self.running:
                await asyncio.sleep(1)
        except (KeyboardInterrupt, SystemExit):
            pass
        finally:
            if self.scheduler.running:
                self.scheduler.shutdown()
            self.console.print(f"\n[bold]Bot stopped. Total runs: {self.total_runs}, Successful: {self.successful_runs}[/bold]")
            logger.info(f"Bot stopped. Total runs: {self.total_runs}, Successful: {self.successful_runs}")


def main():
    """Entry point for continuous bot."""
    parser = argparse.ArgumentParser(
        description="Kalshi Continuous Trading Bot - Runs trading cycles on a schedule"
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Enable live trading (default: dry run mode)"
    )
    parser.add_argument(
        "--version",
        action="version",
        version="Kalshi Trading Bot v1.0.0"
    )
    args = parser.parse_args()

    bot = ContinuousTradingBot(live_trading=args.live)
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
