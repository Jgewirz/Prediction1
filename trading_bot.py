"""
Simple Kalshi trading bot with Octagon research and OpenAI decision making.
"""
import asyncio
import argparse
import json
import csv
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from loguru import logger
import re

from kalshi_client import KalshiClient
from research_client import OctagonClient
from betting_models import BettingDecision, MarketAnalysis, ProbabilityExtraction
from config import load_config
from trendradar_client import TrendRadarClient, TrendingSignal, SignalConfig, calculate_signal_influence, format_signals_for_research
from db import get_database, Database, get_postgres_database, PostgresDatabase
from db.database import close_database
from db.postgres import close_postgres_database
from db.queries import Queries
from calibration_tracker import CalibrationCurve, get_calibration_curve
from dashboard.broadcaster import broadcast_decision, broadcast_kpi_update, broadcast_alert, broadcast_status, broadcast_workflow_step
import openai
import uuid

# Type alias for database (can be SQLite or PostgreSQL)
from typing import Union
DatabaseType = Union[Database, PostgresDatabase]


class SimpleTradingBot:
    """Simple trading bot that follows a clear workflow."""
    
    def __init__(self, live_trading: bool = False, max_close_ts: Optional[int] = None):
        self.config = load_config()
        # Override dry_run based on CLI parameter
        self.config.dry_run = not live_trading
        self.console = Console()
        self.kalshi_client = None
        self.research_client = None
        self.openai_client = None
        self.trendradar_client: Optional[TrendRadarClient] = None
        self.event_signals: Dict[str, List[TrendingSignal]] = {}  # Cache signals per event
        self.db: Optional[Database] = None
        self.run_id: str = str(uuid.uuid4())
        self.max_close_ts = max_close_ts

        # Initialize calibration curve for probability calibration (Phase 1 calibration)
        self.calibration_curve: Optional[CalibrationCurve] = None
        self._calibration_enabled = os.environ.get('CALIBRATION_ENABLED', 'true').lower() == 'true'
        self._calibration_min_samples = int(os.environ.get('CALIBRATION_MIN_SAMPLES', '20'))
        
    async def initialize(self):
        """Initialize all API clients."""
        self.console.print("[bold blue]Initializing trading bot...[/bold blue]")
        
        # Initialize clients
        self.kalshi_client = KalshiClient(
            self.config.kalshi,
            self.config.minimum_time_remaining_hours,
            self.config.max_markets_per_event,
            max_close_ts=self.max_close_ts,
            early_entry_config=self.config.early_entry,
        )
        self.research_client = OctagonClient(self.config.octagon, self.config.openai)
        self.openai_client = openai.AsyncOpenAI(api_key=self.config.openai.api_key)

        # Initialize database if enabled
        if self.config.database.enable_db:
            try:
                db_type = self.config.database.db_type.lower()

                if db_type == "postgres":
                    # Use PostgreSQL (Neon)
                    self.db = await get_postgres_database(
                        host=self.config.database.pg_host,
                        database=self.config.database.pg_database,
                        user=self.config.database.pg_user,
                        password=self.config.database.pg_password,
                        port=self.config.database.pg_port,
                        ssl=self.config.database.pg_ssl
                    )
                    self.console.print("[green][OK] PostgreSQL database connected (Neon)[/green]")
                else:
                    # Use SQLite
                    self.db = await get_database(self.config.database.db_path)
                    self.console.print("[green][OK] SQLite database connected[/green]")

                # Record run start
                await self.db.start_run(self.run_id, {
                    'mode': 'dry_run' if self.config.dry_run else 'live',
                    'environment': 'demo' if self.config.kalshi.use_demo else 'production',
                    'max_events': self.config.max_events_to_analyze,
                    'z_threshold': self.config.z_threshold,
                    'kelly_fraction': self.config.kelly_fraction
                })
            except Exception as e:
                logger.warning(f"Failed to initialize database: {e}")
                self.db = None

        # Test connections
        await self.kalshi_client.login()
        self.console.print("[green][OK] Kalshi API connected[/green]")
        self.console.print("[green][OK] Research API ready (GPT-4o)[/green]")
        self.console.print("[green][OK] OpenAI API ready[/green]")

        # Initialize TrendRadar client if enabled
        if self.config.trendradar.enabled:
            self.trendradar_client = TrendRadarClient(
                base_url=self.config.trendradar.base_url,
                timeout=self.config.trendradar.timeout,
                enabled=True,
                cache_ttl=getattr(self.config.trendradar, 'cache_ttl_seconds', 300.0),
                max_retries=self.config.trendradar.max_retries,
                retry_backoff_base=self.config.trendradar.retry_backoff_base,
                circuit_failure_threshold=self.config.trendradar.circuit_failure_threshold,
                circuit_reset_seconds=self.config.trendradar.circuit_reset_seconds
            )
            # Test connection
            if await self.trendradar_client.health_check():
                logger.info(
                    f"TrendRadar connected | "
                    f"url={self.config.trendradar.base_url} | "
                    f"timeout={self.config.trendradar.timeout}s | "
                    f"max_retries={self.config.trendradar.max_retries} | "
                    f"circuit_threshold={self.config.trendradar.circuit_failure_threshold}"
                )
                self.console.print("[green][OK] TrendRadar connected (Western Financial News)[/green]")
            else:
                logger.warning(
                    f"TrendRadar health check FAILED | "
                    f"url={self.config.trendradar.base_url} | "
                    f"action=disabling_integration | "
                    f"impact=trading_without_news_signals"
                )
                self.console.print("[yellow][!] TrendRadar not reachable - continuing without news signals[/yellow]")
                self.trendradar_client.enabled = False
        else:
            logger.info("TrendRadar integration disabled via config (TRENDRADAR_ENABLED=false)")
            self.console.print("[blue]TrendRadar integration disabled[/blue]")

        # Initialize calibration curve (loads from disk if previously fitted)
        if self._calibration_enabled:
            self.calibration_curve = get_calibration_curve(
                min_samples=self._calibration_min_samples
            )
            if self.calibration_curve.is_fitted:
                self.console.print(
                    f"[green][OK] Calibration curve loaded "
                    f"(fitted on {self.calibration_curve.sample_count} samples)[/green]"
                )
            else:
                self.console.print(
                    f"[blue]Calibration curve not yet fitted "
                    f"(need {self._calibration_min_samples} settled outcomes)[/blue]"
                )
        else:
            self.console.print("[blue]Calibration disabled (CALIBRATION_ENABLED=false)[/blue]")

        # Show environment info
        env_color = "green" if self.config.kalshi.use_demo else "yellow"
        env_name = "DEMO" if self.config.kalshi.use_demo else "PRODUCTION"
        mode = "DRY RUN" if self.config.dry_run else "LIVE TRADING"
        
        self.console.print(f"\n[{env_color}]Environment: {env_name}[/{env_color}]")
        self.console.print(f"[blue]Mode: {mode}[/blue]")
        self.console.print(f"[blue]Max events to analyze: {self.config.max_events_to_analyze}[/blue]")
        self.console.print(f"[blue]Research batch size: {self.config.research_batch_size}[/blue]")
        self.console.print(f"[blue]Skip existing positions: {self.config.skip_existing_positions}[/blue]")
        self.console.print(f"[blue]Minimum time to event strike: {self.config.minimum_time_remaining_hours} hours (for events with strike_date)[/blue]")
        self.console.print(f"[blue]Max markets per event: {self.config.max_markets_per_event}[/blue]")
        self.console.print(f"[blue]Max bet amount: ${self.config.max_bet_amount}[/blue]")
        hedging_status = "Enabled" if self.config.enable_hedging else "Disabled"
        self.console.print(f"[blue]Risk hedging: {hedging_status} (ratio: {self.config.hedge_ratio}, min confidence: {self.config.min_confidence_for_hedging})[/blue]")
        
        # Show risk-adjusted trading settings
        self.console.print(f"[blue]R-score filtering: Enabled (z-threshold: {self.config.z_threshold})[/blue]")
        if self.config.enable_kelly_sizing:
            self.console.print(f"[blue]Kelly sizing: Enabled (fraction: {self.config.kelly_fraction}, bankroll: ${self.config.bankroll})[/blue]")
        self.console.print(f"[blue]Portfolio selection: {self.config.portfolio_selection_method} (max positions: {self.config.max_portfolio_positions})[/blue]\n")
        if self.max_close_ts is not None:
            hours_from_now = (self.max_close_ts - int(time.time())) / 3600
            # Show one decimal hour precision
            self.console.print(f"[blue]Market expiration filter: close before ~{hours_from_now:.1f} hours from now[/blue]")

    async def check_kill_switch(self) -> bool:
        """
        Check if daily loss limit has been hit.

        Returns:
            True if kill switch is triggered (trading should stop), False otherwise.
        """
        if not self.config.enable_kill_switch:
            return False

        if not self.db:
            logger.warning("Kill switch check skipped: database not available")
            return False

        try:
            # Query today's P&L from database
            row = await self.db.fetchone(Queries.GET_TODAY_PNL)
            if row:
                raw_pnl = row.get('daily_pnl', 0.0) if isinstance(row, dict) else (row[0] if row[0] is not None else 0.0)
                daily_pnl = float(raw_pnl) if raw_pnl is not None else 0.0

                # Check if we've exceeded the daily loss limit
                if daily_pnl < -self.config.max_daily_loss:
                    logger.warning(
                        f"Kill switch triggered: daily loss ${abs(daily_pnl):.2f} "
                        f"exceeds limit ${self.config.max_daily_loss:.2f}"
                    )
                    return True

                # Also check percentage-based limit
                pct_loss = abs(daily_pnl) / self.config.bankroll if self.config.bankroll > 0 else 0
                if daily_pnl < 0 and pct_loss > self.config.max_daily_loss_pct:
                    logger.warning(
                        f"Kill switch triggered: daily loss {pct_loss:.1%} "
                        f"exceeds limit {self.config.max_daily_loss_pct:.1%}"
                    )
                    return True

                # Log current status
                if daily_pnl != 0:
                    logger.info(f"Daily P&L: ${daily_pnl:.2f} (limit: ${self.config.max_daily_loss:.2f})")

        except Exception as e:
            logger.error(f"Failed to check kill switch: {e}")
            # Don't block trading on kill switch check failure
            return False

        return False

    def calculate_risk_adjusted_metrics(self, research_prob: float, market_price: float, action: str) -> dict:
        """
        Calculate hedge-fund style risk-adjusted metrics.
        
        Args:
            research_prob: Research probability (0-1)
            market_price: Market price (0-1) 
            action: "buy_yes" or "buy_no"
            
        Returns:
            dict with expected_return, r_score, kelly_fraction
        """
        try:
            # Adjust probabilities based on action
            if action == "buy_yes":
                p = research_prob  # Our probability of YES
                y = market_price   # Market price of YES
            elif action == "buy_no":
                p = 1 - research_prob  # Our probability of NO (1 - research_prob_of_yes)
                y = market_price       # Market price of NO
            else:
                return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}
            
            # Prevent division by zero and invalid probabilities
            if y <= 0 or y >= 1 or p <= 0 or p >= 1:
                return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}
            
            # Expected return on capital: E[R] = (p-y)/y
            expected_return = (p - y) / y
            
            # Risk-Adjusted Edge (R-score): (p-y)/sqrt(p*(1-p)) 
            # This is the z-score - how many standard deviations away from fair value
            variance = p * (1 - p)
            if variance <= 0:
                return {"expected_return": expected_return, "r_score": 0.0, "kelly_fraction": 0.0}
            
            r_score = (p - y) / math.sqrt(variance)
            
            # Kelly fraction: f_kelly = (p-y)/(1-y)
            # This gives the optimal fraction of bankroll to bet
            if y >= 1:
                kelly_fraction = 0.0
            else:
                kelly_fraction = (p - y) / (1 - y)
                # Ensure Kelly fraction is reasonable (between 0 and 1)
                kelly_fraction = max(0.0, min(1.0, kelly_fraction))
            
            return {
                "expected_return": expected_return,
                "r_score": r_score, 
                "kelly_fraction": kelly_fraction
            }
            
        except Exception as e:
            logger.warning(f"Error calculating risk metrics: {e}")
            return {"expected_return": 0.0, "r_score": 0.0, "kelly_fraction": 0.0}
    
    def calculate_kelly_position_size(self, kelly_fraction: float) -> float:
        """
        Calculate position size using fractional Kelly criterion.
        
        Args:
            kelly_fraction: Optimal Kelly fraction (0-1)
            
        Returns:
            Position size in dollars
        """
        if not self.config.enable_kelly_sizing or kelly_fraction <= 0:
            return self.config.max_bet_amount
        
        # Apply fractional Kelly (e.g., half-Kelly)
        adjusted_kelly = kelly_fraction * self.config.kelly_fraction
        
        # Calculate position size as fraction of bankroll
        kelly_bet_size = self.config.bankroll * adjusted_kelly
        
        # Apply maximum bet fraction constraint
        max_allowed = self.config.bankroll * self.config.max_kelly_bet_fraction
        kelly_bet_size = min(kelly_bet_size, max_allowed)
        
        # Apply absolute maximum bet limit
        kelly_bet_size = min(kelly_bet_size, self.config.max_bet_amount)
        
        # Ensure minimum bet size
        kelly_bet_size = max(kelly_bet_size, 1.0)
        
        return kelly_bet_size

    def calculate_dynamic_kelly(self, base_kelly: float, r_score: float, confidence: float,
                                 signal_strength: float = 0.0) -> float:
        """
        Calculate Kelly fraction dynamically based on bet quality (R-score, confidence, calibration).

        Higher R-scores and confidence warrant closer-to-full Kelly sizing.
        Lower quality bets use more conservative sizing.
        Calibration accuracy influences confidence penalty severity.

        Args:
            base_kelly: Base Kelly fraction from probability edge
            r_score: Risk-adjusted edge score (z-score)
            confidence: Model's confidence in the probability estimate (0-1)
            signal_strength: TrendRadar signal strength (0-1), 0 if no signal

        Returns:
            Quality-adjusted Kelly fraction
        """
        # R-score multiplier: scale from 0.5 (at threshold) to 1.0 (at R-score 3.0+)
        # At R-score 0.8 (threshold), use 0.5x multiplier
        # At R-score 3.0+, use full 1.0x multiplier
        r_score_normalized = min(max((r_score - self.config.z_threshold) / 2.2, 0), 1)
        r_score_multiplier = 0.5 + (r_score_normalized * 0.5)  # 0.5 to 1.0

        # Confidence multiplier: severity depends on calibration accuracy
        # If we have a well-calibrated system, be less harsh on confidence penalty
        if self.calibration_curve and self.calibration_curve.is_fitted:
            # Well-calibrated system: use confidence^1.5 (less severe)
            confidence_multiplier = confidence ** 1.5
        else:
            # Not calibrated yet: use confidence^2 (more conservative)
            confidence_multiplier = confidence ** 2
        confidence_multiplier = max(confidence_multiplier, 0.25)  # Floor at 0.25

        # Signal alignment bonus (if TrendRadar signals are aligned)
        signal_multiplier = 1.0
        if signal_strength > 0.5:
            signal_multiplier = 1.0 + (signal_strength * 0.2)  # Up to 20% boost

        # Combined quality multiplier
        quality_multiplier = r_score_multiplier * confidence_multiplier * signal_multiplier

        # Apply to base Kelly
        adjusted_kelly = base_kelly * quality_multiplier * self.config.kelly_fraction

        return adjusted_kelly

    def apply_calibration(self, raw_prob: float) -> float:
        """
        Apply probability calibration if available.

        Args:
            raw_prob: Raw model probability (0-100 scale)

        Returns:
            Calibrated probability (0-100 scale), or raw if no calibration
        """
        if self.calibration_curve and self.calibration_curve.is_fitted:
            calibrated = self.calibration_curve.calibrate(raw_prob)
            # Log significant adjustments
            if abs(calibrated - raw_prob) > 5:
                logger.debug(f"Calibration: {raw_prob:.1f}% -> {calibrated:.1f}%")
            return calibrated
        return raw_prob

    def calculate_quality_adjusted_position_size(self, kelly_fraction: float, r_score: float,
                                                  confidence: float) -> float:
        """
        Calculate position size using dynamic Kelly based on bet quality.

        Args:
            kelly_fraction: Raw Kelly fraction from edge calculation
            r_score: Risk-adjusted edge score
            confidence: Model confidence in probability estimate

        Returns:
            Position size in dollars
        """
        if not self.config.enable_kelly_sizing or kelly_fraction <= 0:
            return self.config.max_bet_amount

        # Apply dynamic Kelly adjustment based on quality
        adjusted_kelly = self.calculate_dynamic_kelly(kelly_fraction, r_score, confidence)

        # Calculate position size as fraction of bankroll
        kelly_bet_size = self.config.bankroll * adjusted_kelly

        # Apply maximum bet fraction constraint
        max_allowed = self.config.bankroll * self.config.max_kelly_bet_fraction
        kelly_bet_size = min(kelly_bet_size, max_allowed)

        # Apply absolute maximum bet limit
        kelly_bet_size = min(kelly_bet_size, self.config.max_bet_amount)

        # Ensure minimum bet size
        kelly_bet_size = max(kelly_bet_size, 1.0)

        return kelly_bet_size

    def apply_portfolio_selection(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """
        Apply portfolio selection to hold only the N highest R-scores.
        Step 4: Portfolio view - hold the N highest R-scores subject to limits.
        """
        if self.config.portfolio_selection_method == "legacy":
            # Skip portfolio optimization, use existing logic
            return analysis
        
        # Filter out skip decisions for ranking
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        skip_decisions = [d for d in analysis.decisions if d.action == "skip"]
        
        if not actionable_decisions:
            return analysis
        
        if self.config.portfolio_selection_method == "top_r_scores":
            # Sort by R-score (highest first)
            actionable_decisions.sort(key=lambda d: d.r_score or -999, reverse=True)
            
            # Select top N positions
            max_positions = self.config.max_portfolio_positions
            selected_decisions = actionable_decisions[:max_positions]
            
            # Convert remaining to skip decisions
            rejected_decisions = []
            for decision in actionable_decisions[max_positions:]:
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Portfolio limit: R-score {decision.r_score:.2f} ranked #{len(selected_decisions)+1}",
                    event_name=decision.event_name,
                    market_name=decision.market_name,
                    expected_return=decision.expected_return,
                    r_score=decision.r_score,
                    kelly_fraction=decision.kelly_fraction,
                    market_price=decision.market_price,
                    research_probability=decision.research_probability
                )
                rejected_decisions.append(skip_decision)
            
            if rejected_decisions:
                logger.info(f"Portfolio selection: kept top {len(selected_decisions)} positions, "
                           f"rejected {len(rejected_decisions)} lower R-score positions")
            
            # Combine selected decisions with all skip decisions
            analysis.decisions = selected_decisions + skip_decisions + rejected_decisions
            
        elif self.config.portfolio_selection_method == "diversified":
            # Future enhancement: could implement diversification by event category, etc.
            # For now, fall back to top R-scores
            return self.apply_portfolio_selection(analysis, event_ticker)

        return analysis

    def _parse_datetime(self, value: Optional[str]) -> Optional[datetime]:
        """Parse a datetime string to timezone-naive datetime for PostgreSQL."""
        if not value:
            return None
        try:
            # Handle ISO format with Z suffix
            if value.endswith('Z'):
                value = value[:-1] + '+00:00'
            dt = datetime.fromisoformat(value)
            # Convert to UTC and remove timezone for PostgreSQL compatibility
            if dt.tzinfo is not None:
                from datetime import timezone
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except (ValueError, AttributeError) as e:
            logger.warning(f"Failed to parse datetime '{value}': {e}")
            return None

    async def get_top_events(self) -> List[Dict[str, Any]]:
        """Get top events sorted by 24-hour volume."""
        self.console.print("[bold]Step 1: Fetching top events...[/bold]")
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
            transient=True,
        ) as progress:
            task = progress.add_task("Fetching events...", total=None)
            
            try:
                # Get a larger pool of events to ensure we have enough after filtering positions
                # Use 3x the target amount to account for events with existing positions
                fetch_limit = self.config.max_events_to_analyze * 3
                events = await self.kalshi_client.get_events(limit=fetch_limit)
                self.console.print(f"[blue]• Fetched {len(events)} events (will filter to top {self.config.max_events_to_analyze} after position filtering)[/blue]")
                
                self.console.print(f"[green][OK] Found {len(events)} events[/green]")
                
                # Show top 10 events
                table = Table(title="Top 10 Events by 24h Volume")
                table.add_column("Event Ticker", style="cyan")
                table.add_column("Title", style="yellow")
                table.add_column("24h Volume", style="magenta", justify="right")
                table.add_column("Time Remaining", style="blue", justify="right")
                table.add_column("Category", style="green")
                table.add_column("Mutually Exclusive", style="red", justify="center")
                
                for event in events[:10]:
                    time_remaining = event.get('time_remaining_hours')
                    if time_remaining is None:
                        time_str = "No date set"
                    elif time_remaining > 24:
                        time_str = f"{time_remaining/24:.1f} days"
                    else:
                        time_str = f"{time_remaining:.1f} hours"
                    
                    table.add_row(
                        event.get('event_ticker', 'N/A'),
                        event.get('title', 'N/A')[:35] + ("..." if len(event.get('title', '')) > 35 else ""),
                        f"{event.get('volume_24h', 0):,}",
                        time_str,
                        event.get('category', 'N/A'),
                        "YES" if event.get('mutually_exclusive', False) else "NO"
                    )
                
                self.console.print(table)
                return events
                
            except Exception as e:
                self.console.print(f"[red]Error fetching events: {e}[/red]")
                return []
    
    async def get_markets_for_events(self, events: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Get markets for each event (uses pre-loaded markets from events)."""
        self.console.print(f"\n[bold]Step 2: Processing markets for {len(events)} events...[/bold]")
        
        event_markets = {}
        
        for event in events:
            event_ticker = event.get('event_ticker', '')
            if not event_ticker:
                continue
            
            # Use pre-loaded markets from the event data
            markets = event.get('markets', [])
            total_markets = event.get('total_markets', len(markets))
            
            if markets:
                # Convert to the format expected by the rest of the system
                simple_markets = []
                for market in markets:
                    simple_markets.append({
                        "ticker": market.get("ticker", ""),
                        "title": market.get("title", ""),
                        "subtitle": market.get("subtitle", ""),
                        "volume": market.get("volume", 0),
                        "open_time": market.get("open_time", ""),
                        "close_time": market.get("close_time", ""),
                    })
                
                event_markets[event_ticker] = {
                    'event': event,
                    'markets': simple_markets
                }
                
                if total_markets > len(markets):
                    self.console.print(f"[green][OK] Using top {len(markets)} markets for {event_ticker} (from {total_markets} total)[/green]")
                else:
                    self.console.print(f"[green][OK] Using {len(markets)} markets for {event_ticker}[/green]")
            else:
                self.console.print(f"[yellow][!] No markets found for {event_ticker}[/yellow]")
        
        total_markets = sum(len(data['markets']) for data in event_markets.values())
        self.console.print(f"[green][OK] Processing {total_markets} total markets across {len(event_markets)} events[/green]")
        return event_markets
    
    async def filter_markets_by_positions(self, event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Filter out individual markets where we already have positions (market-level, not event-level)."""
        if self.config.dry_run or not self.config.skip_existing_positions:
            # Skip position filtering in dry run mode or if disabled
            return event_markets

        self.console.print(f"\n[bold]Step 2.5: Filtering individual markets by existing positions...[/bold]")

        filtered_event_markets = {}
        total_markets_before = 0
        total_markets_after = 0
        skipped_markets = 0

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            # Count total markets for progress
            total_markets = sum(len(data['markets']) for data in event_markets.values())
            task = progress.add_task("Checking existing positions...", total=total_markets)

            for event_ticker, data in event_markets.items():
                event = data['event']
                markets = data['markets']
                total_markets_before += len(markets)

                # Filter individual markets (not entire event)
                filtered_markets = []

                for market in markets:
                    ticker = market.get('ticker', '')
                    if not ticker:
                        progress.update(task, advance=1)
                        continue

                    try:
                        # Check if we already have a position in THIS market
                        has_position = await self.kalshi_client.has_position_in_market(ticker)
                        if has_position:
                            self.console.print(f"[yellow][!] Skipping {ticker}: Has existing position[/yellow]")
                            skipped_markets += 1
                        else:
                            # Keep this market
                            filtered_markets.append(market)

                    except Exception as e:
                        logger.warning(f"Could not check position for {ticker}: {e}")
                        # If we can't check, assume no position and keep the market
                        filtered_markets.append(market)

                    progress.update(task, advance=1)

                # Keep event if any markets remain after filtering
                if filtered_markets:
                    filtered_event_markets[event_ticker] = {
                        'event': event,
                        'markets': filtered_markets
                    }
                    total_markets_after += len(filtered_markets)
                    if len(filtered_markets) < len(markets):
                        self.console.print(f"[blue]• Event {event_ticker}: Kept {len(filtered_markets)}/{len(markets)} markets[/blue]")
                    else:
                        self.console.print(f"[green][OK] Event {event_ticker}: All {len(markets)} markets available[/green]")
                else:
                    self.console.print(f"[yellow][!] Event {event_ticker}: All markets have positions, skipping[/yellow]")

        # Show filtering summary
        events_with_some_markets = len(filtered_event_markets)
        self.console.print(f"\n[blue]Position filtering summary (market-level):[/blue]")
        self.console.print(f"[blue]• Total markets checked: {total_markets_before}[/blue]")
        self.console.print(f"[blue]• Markets with positions (skipped): {skipped_markets}[/blue]")
        self.console.print(f"[blue]• Markets remaining for research: {total_markets_after}[/blue]")
        self.console.print(f"[blue]• Events with available markets: {events_with_some_markets}[/blue]")

        if events_with_some_markets == 0:
            self.console.print("[yellow][!] No markets remaining after position filtering[/yellow]")

        return filtered_event_markets

    async def fetch_trending_signals(self, event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, List[TrendingSignal]]:
        """Fetch trending signals for all events from TrendRadar (Step 3.25)."""
        if not self.trendradar_client or not self.trendradar_client.enabled:
            return {}

        self.console.print(f"\n[bold]Step 3.25: Fetching trending signals for {len(event_markets)} events...[/bold]")

        signals_by_event = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Fetching news signals...", total=len(event_markets))

            for event_ticker, data in event_markets.items():
                event = data['event']
                event_title = event.get('title', '')
                event_category = event.get('category', '')

                try:
                    signals = await self.trendradar_client.get_signals_for_event(
                        event_title=event_title,
                        event_category=event_category
                    )

                    if signals:
                        signals_by_event[event_ticker] = signals
                        strong_count = sum(1 for s in signals if s.is_strong)
                        self.console.print(
                            f"[green][OK] {event_ticker}: {len(signals)} signals "
                            f"({strong_count} strong)[/green]"
                        )

                except Exception as e:
                    logger.warning(f"Error fetching signals for {event_ticker}: {e}")

                progress.update(task, advance=1)

        # Cache signals for use in decision making
        self.event_signals = signals_by_event

        total_signals = sum(len(s) for s in signals_by_event.values())
        self.console.print(f"[green][OK] Fetched {total_signals} signals for {len(signals_by_event)} events[/green]")

        return signals_by_event

    def _parse_probabilities_from_research(self, research_text: str, markets: List[Dict[str, Any]]) -> Dict[str, float]:
        """Parse probability predictions from Octagon research text."""
        probabilities = {}
        
        # Look for patterns with both ticker names and market titles
        for market in markets:
            ticker = market.get('ticker', '')
            title = market.get('title', '')
            if not ticker:
                continue
                
            # Try different patterns to find probability for this market
            # Look for both ticker and title patterns
            search_terms = [ticker]
            if title:
                # Add key words from the title for better matching
                search_terms.append(title)
                # Extract key identifying words (avoid common words)
                title_words = [w for w in title.split() if len(w) > 3 and w.lower() not in ['will', 'the', 'win', 'be', 'a', 'be', 'of', 'and', 'or', 'for', 'to', 'in', 'on', 'at', 'with', 'from', 'by', 'about', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'up', 'down', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once']]
                search_terms.extend(title_words)
            
            found_probability = None
            for term in search_terms:
                if not term:
                    continue
                    
                # Try different patterns to find probability for this term
                patterns = [
                    rf"{re.escape(term)}[:\s]*(\d+\.?\d*)%",
                    rf"{re.escape(term)}[:\s]*(\d+)%",
                    rf"(\d+\.?\d*)%[:\s]*{re.escape(term)}",
                    rf"(\d+)%[:\s]*{re.escape(term)}",
                    rf"probability.*{re.escape(term)}[:\s]*(\d+\.?\d*)%",
                    rf"{re.escape(term)}.*probability.*?(\d+\.?\d*)%",
                    rf"{re.escape(term)}.*(\d+\.?\d*)%.*probability",
                    rf"probability.*(\d+\.?\d*)%.*{re.escape(term)}",
                    # More flexible patterns for natural language
                    rf"{re.escape(term)}.*?(\d+\.?\d*)%",
                    rf"(\d+\.?\d*)%.*?{re.escape(term)}",
                ]
                
                for pattern in patterns:
                    matches = re.findall(pattern, research_text, re.IGNORECASE | re.DOTALL)
                    if matches:
                        try:
                            prob = float(matches[0])
                            if 0 <= prob <= 100:
                                found_probability = prob
                                break
                        except ValueError:
                            continue
                
                if found_probability is not None:
                    break
            
            if found_probability is not None:
                probabilities[ticker] = found_probability
                logger.info(f"Found probability for {ticker}: {found_probability}%")
            else:
                logger.warning(f"No probability found for {ticker} (title: {title})")
                # Show first 200 chars of research text for debugging
                sample_text = research_text[:200].replace('\n', ' ')
                logger.debug(f"Research sample: {sample_text}...")
             
        return probabilities

    async def research_events(self, event_markets: Dict[str, Dict[str, Any]], signals_by_event: Dict[str, List[TrendingSignal]] = None) -> Dict[str, str]:
        """Research each event and its markets using Octagon Deep Research."""
        self.console.print(f"\n[bold]Step 3: Researching {len(event_markets)} events...[/bold]")

        if signals_by_event is None:
            signals_by_event = {}

        research_results = {}

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Researching events...", total=len(event_markets))

            # Research events in batches to avoid rate limits
            batch_size = self.config.research_batch_size
            event_items = list(event_markets.items())

            for i in range(0, len(event_items), batch_size):
                batch = event_items[i:i + batch_size]
                self.console.print(f"[blue]Processing research batch {i//batch_size + 1} with {len(batch)} events[/blue]")

                # Research batch in parallel with per-event timeout
                tasks = []
                for event_ticker, data in batch:
                    event = data['event']
                    markets = data['markets']
                    if event and markets:
                        # Build trending context from signals if available
                        trending_context = ""
                        if event_ticker in signals_by_event:
                            trending_context = format_signals_for_research(signals_by_event[event_ticker])

                        coro = self.research_client.research_event(event, markets, trending_context)
                        # Apply per-event timeout to avoid hanging the whole batch
                        tasks.append(asyncio.wait_for(coro, timeout=self.config.research_timeout_seconds))
                    else:
                        tasks.append(asyncio.sleep(0, result=None))
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for (event_ticker, data), result in zip(batch, results):
                        if not isinstance(result, Exception) and result:
                            research_results[event_ticker] = result
                            progress.update(task, advance=1)
                            self.console.print(f"[green][OK] Researched {event_ticker}[/green]")
                        else:
                            err = result
                            if isinstance(result, asyncio.TimeoutError):
                                err = f"Timeout after {self.config.research_timeout_seconds}s"
                            self.console.print(f"[red][X] Failed to research {event_ticker}: {err}[/red]")
                            progress.update(task, advance=1)
                
                except Exception as e:
                    self.console.print(f"[red]Batch research error: {e}[/red]")
                    # Ensure progress advances for this entire batch even on error
                    progress.update(task, advance=len(batch))
                
                # Brief pause between batches
                await asyncio.sleep(1)
        
        self.console.print(f"[green][OK] Completed research on {len(research_results)} events[/green]")
    
        
        return research_results
    
    async def _extract_probabilities_for_event(self, event_ticker: str, research_text: str, 
                                              event_markets: Dict[str, Dict[str, Any]]) -> tuple[str, Optional[ProbabilityExtraction]]:
        """Extract probabilities for a single event."""
        try:
            # Get market information for this event
            event_data = event_markets.get(event_ticker, {})
            markets = event_data.get('markets', [])
            event_info = event_data.get('event', {})
            
            # Prepare market information for the prompt
            market_info = []
            for market in markets:
                market_info.append({
                    'ticker': market.get('ticker', ''),
                    'title': market.get('title', ''),
                    'yes_mid_price': market.get('yes_mid_price', 0),
                    'no_mid_price': market.get('no_mid_price', 0)
                })
            
            # Create prompt for probability extraction with calibration-aware constraints
            prompt = f"""
            Based on the following deep research, extract CALIBRATED probability estimates for each market.

            Event: {event_info.get('title', event_ticker)}

            Markets:
            {json.dumps(market_info, indent=2)}

            Research Results:
            {research_text}

            EXTRACTION RULES (FOLLOW EXACTLY):
            1. If research provides EXPLICIT probability estimates (percentages, odds), USE THOSE EXACT VALUES
            2. If research provides ranges (e.g., "60-70%"), use the MIDPOINT
            3. If research has NO quantitative probability data for a market, set confidence=0.2 (very low)
            4. DO NOT invent or guess probabilities - only extract what the research explicitly supports
            5. Include direct quotes or citations from the research to justify each probability

            CALIBRATION-AWARE CONFIDENCE SCORING:
            - confidence 0.8-1.0: Research has explicit numerical probabilities with clear justification
            - confidence 0.6-0.7: Research has strong qualitative indicators pointing to specific outcome
            - confidence 0.4-0.5: Research has moderate evidence but some uncertainty acknowledged
            - confidence 0.2-0.3: Research is vague, conflicting signals, or lacks specific data

            OVERCONFIDENCE CHECKS:
            - Does the research acknowledge uncertainty? Higher confidence if yes.
            - Are there counterarguments mentioned? Reduce confidence if ignored.
            - Is the probability near 50%? Require MORE evidence for high confidence.
            - Is the probability extreme (>80% or <20%)? Require STRONGEST evidence.

            For each market, provide:
            1. research_probability: The probability (0-100%) - must be justified by research
            2. reasoning: Must include direct evidence/quotes AND uncertainty factors
            3. confidence: 0-1 based on calibration rules above

            IMPORTANT: Extreme probabilities (>85% or <15%) require explicit, verifiable evidence.
            Default to more moderate estimates when evidence is circumstantial.
            """

            # Calibration-aware system prompt
            system_prompt = (
                "You are a calibrated prediction market analyst. Your probability extractions are "
                "tracked for accuracy over time. Extract probabilities conservatively - only claim "
                "high confidence when research provides explicit numerical estimates. When uncertain, "
                "use confidence < 0.5 and keep probabilities closer to 50%."
            )

            # Use Responses API structured outputs
            from openai_utils import responses_parse_pydantic
            extraction = await responses_parse_pydantic(
                self.openai_client,
                model=self.config.openai.model if self.config.openai.model else "gpt-5",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                response_format=ProbabilityExtraction,
                reasoning_effort="high",
                text_verbosity="high",
            )

            return event_ticker, extraction
            
        except Exception as e:
            logger.error(f"Error extracting probabilities for {event_ticker}: {e}")
            return event_ticker, None

    async def extract_probabilities(self, research_results: Dict[str, str], 
                                  event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, ProbabilityExtraction]:
        """Extract structured probabilities from research results using GPT-5 in parallel."""
        self.console.print(f"\n[bold]Step 3.5: Extracting probabilities from research...[/bold]")
        
        probability_extractions = {}
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Extracting probabilities...", total=len(research_results))
            
            # Create tasks for all events to run in parallel
            tasks = []
            for event_ticker, research_text in research_results.items():
                task_coroutine = self._extract_probabilities_for_event(event_ticker, research_text, event_markets)
                tasks.append(task_coroutine)
            
            # Run all probability extractions in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in probability extraction: {result}")
                    progress.update(task, advance=1)
                    continue
                    
                event_ticker, extraction = result
                if extraction is not None:
                    probability_extractions[event_ticker] = extraction
                    self.console.print(f"[green][OK] Extracted probabilities for {event_ticker}[/green]")
                    
                    # Display extracted probabilities
                    self.console.print(f"[blue]Extracted probabilities for {event_ticker}:[/blue]")
                    for market_prob in extraction.markets:
                        self.console.print(f"  {market_prob.ticker}: {market_prob.research_probability:.1f}%")
                else:
                    self.console.print(f"[red][X] Failed to extract probabilities for {event_ticker}[/red]")
                
                progress.update(task, advance=1)
        
        self.console.print(f"[green][OK] Extracted probabilities for {len(probability_extractions)} events[/green]")
        return probability_extractions
    
    
    
    async def get_market_odds(self, event_markets: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
        """Fetch current market odds for all markets."""
        self.console.print(f"\n[bold]Step 4: Fetching current market odds...[/bold]")
        
        market_odds = {}
        all_tickers = []
        
        # Collect all market tickers
        for event_ticker, data in event_markets.items():
            markets = data['markets']
            for market in markets:
                ticker = market.get('ticker', '')
                if ticker:
                    all_tickers.append(ticker)
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Fetching market odds...", total=len(all_tickers))
            
            # Fetch odds in batches to avoid overwhelming the API
            batch_size = 20
            for i in range(0, len(all_tickers), batch_size):
                batch = all_tickers[i:i + batch_size]
                
                # Fetch batch in parallel
                tasks = []
                for ticker in batch:
                    tasks.append(self.kalshi_client.get_market_with_odds(ticker))
                
                try:
                    results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    for ticker, result in zip(batch, results):
                        if not isinstance(result, Exception) and result:
                            market_odds[ticker] = result
                            progress.update(task, advance=1)
                        else:
                            self.console.print(f"[red][X] Failed to get odds for {ticker}[/red]")
                            progress.update(task, advance=1)
                
                except Exception as e:
                    self.console.print(f"[red]Batch odds fetch error: {e}[/red]")
                    progress.update(task, advance=len(batch))
                
                # Brief pause between batches
                await asyncio.sleep(0.2)
        
        self.console.print(f"[green][OK] Fetched odds for {len(market_odds)} markets[/green]")
        return market_odds
    
    async def _get_betting_decisions_for_event(self, event_ticker: str, data: Dict[str, Any], 
                                             probability_extraction: ProbabilityExtraction, 
                                             market_odds: Dict[str, Dict[str, Any]]) -> tuple[str, Optional[MarketAnalysis]]:
        """Get betting decisions for a single event with error handling."""
        try:
            # Get event-specific decisions
            event_analysis = await self._get_event_betting_decisions(
                event_ticker, data, probability_extraction, market_odds
            )
            return event_ticker, event_analysis
        except Exception as e:
            logger.error(f"Error generating decisions for {event_ticker}: {e}")
            return event_ticker, None

    async def get_betting_decisions(self, event_markets: Dict[str, Dict[str, Any]], 
                                   probability_extractions: Dict[str, ProbabilityExtraction], 
                                   market_odds: Dict[str, Dict[str, Any]]) -> MarketAnalysis:
        """Use OpenAI to make structured betting decisions per event in parallel."""
        self.console.print(f"\n[bold]Step 5: Generating betting decisions...[/bold]")
        
        # Process events in parallel for better performance
        all_decisions = []
        total_recommended_bet = 0.0
        high_confidence_bets = 0
        event_summaries = []
        
        # Filter to events that have both research results and markets
        processable_events = [
            (event_ticker, data) for event_ticker, data in event_markets.items()
            if event_ticker in probability_extractions and data['markets']
        ]
        
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=self.console,
        ) as progress:
            task = progress.add_task("Generating betting decisions...", total=len(processable_events))
            
            # Create tasks for all events to run in parallel
            tasks = []
            for event_ticker, data in processable_events:
                task_coroutine = self._get_betting_decisions_for_event(
                    event_ticker, data, probability_extractions[event_ticker], market_odds
                )
                tasks.append(task_coroutine)
            
            # Run all betting decision generations in parallel
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for result in results:
                if isinstance(result, Exception):
                    logger.error(f"Exception in betting decisions generation: {result}")
                    progress.update(task, advance=1)
                    continue
                
                event_ticker, event_analysis = result
                if event_analysis is not None:
                    # Display decisions for this event
                    self._display_event_decisions(event_ticker, event_analysis)
                    
                    # Aggregate results
                    all_decisions.extend(event_analysis.decisions)
                    total_recommended_bet += event_analysis.total_recommended_bet
                    high_confidence_bets += event_analysis.high_confidence_bets
                    event_summaries.append(f"{event_ticker}: {event_analysis.summary}")
                    
                    self.console.print(f"[green][OK] Generated {len(event_analysis.decisions)} decisions for {event_ticker}[/green]")
                else:
                    self.console.print(f"[red][X] Failed to generate decisions for {event_ticker}[/red]")
                
                progress.update(task, advance=1)
        
        # Generate hedge decisions for risk management
        hedge_decisions = self._generate_hedge_decisions(all_decisions)
        if hedge_decisions:
            all_decisions.extend(hedge_decisions)
            # Update totals to include hedge amounts
            hedge_bet_total = sum(d.amount for d in hedge_decisions)
            total_recommended_bet += hedge_bet_total
            self.console.print(f"[blue]💡 Generated {len(hedge_decisions)} hedge bets (${hedge_bet_total:.2f}) for risk management[/blue]")
        
        # Create combined analysis
        analysis = MarketAnalysis(
            decisions=all_decisions,
            total_recommended_bet=total_recommended_bet,
            high_confidence_bets=high_confidence_bets,
            summary=f"Analyzed {len(processable_events)} events. " + " | ".join(event_summaries[:3]) + 
                   (f" and {len(event_summaries) - 3} more..." if len(event_summaries) > 3 else "")
        )
        
        # Show overall decision summary
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        self.console.print(f"\n[green][OK] Generated {len(analysis.decisions)} total decisions ({len(actionable_decisions)} actionable)[/green]")
        
        # Display consolidated summary table
        if actionable_decisions:
            table = Table(title="All Betting Decisions Summary", show_lines=True)
            table.add_column("Type", style="bright_blue", justify="center", width=8)
            table.add_column("Event", style="bright_blue", width=22)
            table.add_column("Market", style="cyan", width=32)
            table.add_column("Action", style="yellow", justify="center", width=10)
            table.add_column("Confidence", style="magenta", justify="right", width=10)
            table.add_column("Amount", style="green", justify="right", width=10)
            table.add_column("Reasoning", style="blue", width=65)
            
            for decision in actionable_decisions:
                # Use human-readable names if available
                event_name = decision.event_name if decision.event_name else "Unknown Event"
                market_name = decision.market_name if decision.market_name else decision.ticker
                
                # Determine bet type
                bet_type = "🛡️ Hedge" if decision.is_hedge else "💰 Main"
                
                table.add_row(
                    bet_type,
                    event_name,
                    market_name,
                    decision.action.upper().replace('_', ' '),
                    f"{decision.confidence:.2f}",
                    f"${decision.amount:.2f}",
                    decision.reasoning
                )
            
            self.console.print(table)
        else:
            self.console.print("[yellow]No actionable betting decisions generated[/yellow]")
        
        # Show summary
        self.console.print(f"\n[blue]Total recommended bet: ${analysis.total_recommended_bet:.2f}[/blue]")
        self.console.print(f"[blue]High confidence bets: {analysis.high_confidence_bets}[/blue]")
        self.console.print(f"[blue]Strategy: {analysis.summary}[/blue]")
        
        return analysis
    
    def _generate_hedge_decisions(self, main_decisions: List[BettingDecision]) -> List[BettingDecision]:
        """Generate hedge decisions to minimize risk for main betting decisions."""
        if not self.config.enable_hedging:
            return []
        
        hedge_decisions = []
        
        for main_decision in main_decisions:
            # Skip if this is already a hedge or a skip
            if main_decision.is_hedge or main_decision.action == "skip":
                continue
                
            # Only hedge if confidence is below threshold (higher risk bets)
            if main_decision.confidence >= self.config.min_confidence_for_hedging:
                continue
                
            # Calculate hedge amount
            hedge_amount = min(
                main_decision.amount * self.config.hedge_ratio,
                self.config.max_hedge_amount
            )
            
            # Only create hedge if amount is meaningful (at least $1)
            if hedge_amount < 1.0:
                continue
            
            # Create opposite hedge position
            hedge_action = "buy_no" if main_decision.action == "buy_yes" else "buy_yes"
            
            hedge_decision = BettingDecision(
                ticker=main_decision.ticker,
                action=hedge_action,
                confidence=0.8,  # Hedge has high confidence (it's risk management)
                amount=hedge_amount,
                reasoning=f"Risk hedge: {self.config.hedge_ratio*100:.0f}% hedge for {main_decision.action} (confidence {main_decision.confidence:.2f} < {self.config.min_confidence_for_hedging:.2f})",
                event_name=main_decision.event_name,
                market_name=main_decision.market_name,
                is_hedge=True,
                hedge_for=main_decision.ticker,
                hedge_ratio=self.config.hedge_ratio
            )
            
            hedge_decisions.append(hedge_decision)
        
        return hedge_decisions
    
    def _display_event_decisions(self, event_ticker: str, event_analysis: MarketAnalysis):
        """Display the betting decisions for a single event."""
        # Filter to actionable decisions (not skip)
        actionable_decisions = [
            decision for decision in event_analysis.decisions 
            if decision.action != "skip"
        ]
        
        # Check if any decisions were adjusted due to mutually exclusive constraint
        mutually_exclusive_adjustments = [
            decision for decision in event_analysis.decisions 
            if decision.action != "skip" and "Mutually exclusive hedge" in decision.reasoning
        ]
        
        # Check if strategic filtering was applied
        strategic_filtering_skips = [
            decision for decision in event_analysis.decisions 
            if decision.action == "skip" and "Strategic filter" in decision.reasoning
        ]
        
        if not actionable_decisions:
            self.console.print(f"[yellow]No actionable decisions for {event_ticker}[/yellow]")
            return
        
        # Create event-specific table
        event_name = actionable_decisions[0].event_name if actionable_decisions else "Unknown Event"
        table = Table(title=f"Betting Decisions for {event_name}", show_lines=True)
        table.add_column("Type", style="bright_blue", justify="center", width=8)
        table.add_column("Market", style="cyan", width=40)
        table.add_column("Action", style="yellow", justify="center", width=10)
        table.add_column("Confidence", style="magenta", justify="right", width=10)
        table.add_column("Amount", style="green", justify="right", width=10)
        table.add_column("Reasoning", style="blue", width=70)
        
        for decision in actionable_decisions:
            # Use human-readable market name if available, otherwise generate from ticker
            market_display = decision.market_name if decision.market_name else self._generate_readable_market_name(decision.ticker)
            
            # Determine bet type
            bet_type = "🛡️ Hedge" if decision.is_hedge else "💰 Main"
            
            table.add_row(
                bet_type,
                market_display,
                decision.action.upper().replace('_', ' '),
                f"{decision.confidence:.2f}",
                f"${decision.amount:.2f}",
                decision.reasoning
            )
        
        self.console.print(table)
        
        # Show mutually exclusive strategy info if applicable
        if mutually_exclusive_adjustments:
            self.console.print(f"[blue]ℹ Strategic hedge betting: {len(mutually_exclusive_adjustments)} positions sized for mutually exclusive event[/blue]")
        
        # Show strategic filtering info if applicable
        if strategic_filtering_skips:
            self.console.print(f"[yellow]⚡ Strategic filtering: {len(strategic_filtering_skips)} lower-value opportunities skipped, focused on best positions[/yellow]")
        
        # Show event summary
        if event_analysis.total_recommended_bet > 0:
            self.console.print(f"[blue]Event total: ${event_analysis.total_recommended_bet:.2f} | High confidence: {event_analysis.high_confidence_bets}[/blue]")
    
    async def _get_event_betting_decisions(self, event_ticker: str, event_data: Dict[str, Any], 
                                         probability_extraction: ProbabilityExtraction, market_odds: Dict[str, Dict[str, Any]]) -> MarketAnalysis:
        """Get betting decisions for a single event."""
        event_info = event_data['event']
        markets = event_data['markets']
        
        # Include market odds in the data
        markets_with_odds = []
        for market in markets:
            ticker = market.get('ticker', '')
            market_data = {
                'ticker': ticker,
                'title': market.get('title', ''),
                'volume': market.get('volume', 0)
            }
            
            # Add current market odds if available
            if ticker in market_odds:
                odds = market_odds[ticker]
                
                yes_bid = odds.get('yes_bid', 0)
                no_bid = odds.get('no_bid', 0)
                yes_ask = odds.get('yes_ask', 0)
                no_ask = odds.get('no_ask', 0)
                
                # Calculate mid-prices with validation
                # Only calculate mid-price if both bid and ask are > 0
                yes_mid_price = None
                no_mid_price = None
                
                if yes_bid > 0 and yes_ask > 0:
                    yes_mid_price = (yes_bid + yes_ask) / 2
                elif yes_ask > 0:
                    # If only ask is available, use ask price as approximation
                    yes_mid_price = yes_ask
                elif yes_bid > 0:
                    # If only bid is available, use bid price as approximation
                    yes_mid_price = yes_bid
                
                if no_bid > 0 and no_ask > 0:
                    no_mid_price = (no_bid + no_ask) / 2
                elif no_ask > 0:
                    # If only ask is available, use ask price as approximation
                    no_mid_price = no_ask
                elif no_bid > 0:
                    # If only bid is available, use bid price as approximation
                    no_mid_price = no_bid
                
                # Log warning if we couldn't calculate proper mid-prices
                if yes_mid_price is None or no_mid_price is None:
                    logger.warning(f"Market {ticker}: Missing bid/ask data - yes_bid={yes_bid}, yes_ask={yes_ask}, no_bid={no_bid}, no_ask={no_ask}")
                    # Fallback: use 0 for missing prices
                    yes_mid_price = yes_mid_price or 0
                    no_mid_price = no_mid_price or 0
                
                market_data.update({
                    'yes_bid': yes_bid,
                    'no_bid': no_bid,
                    'yes_ask': yes_ask,
                    'no_ask': no_ask,
                    'status': odds.get('status', ''),
                    # Calculate implied probabilities from mid-prices
                    'yes_mid_price': yes_mid_price,
                    'no_mid_price': no_mid_price,
                })
            
            markets_with_odds.append(market_data)
        
        # Skip one-sided markets where either YES or NO mid price is missing/zero
        filtered_markets_with_odds = []
        for md in markets_with_odds:
            yes_mid = md.get('yes_mid_price')
            no_mid = md.get('no_mid_price')
            if yes_mid in (None, 0) or no_mid in (None, 0):
                logger.info(f"Skipping market {md.get('ticker', '')}: missing yes/no price (yes_mid={yes_mid}, no_mid={no_mid})")
                continue
            filtered_markets_with_odds.append(md)
        markets_with_odds = filtered_markets_with_odds

        # Create single event data with structured probabilities
        is_mutually_exclusive = event_info.get('mutually_exclusive', False)
        single_event_data = {
            'event_ticker': event_ticker,
            'event_title': event_info.get('title', ''),
            'event_category': event_info.get('category', ''),
            'event_volume': event_info.get('volume', 0),
            'time_remaining_hours': event_info.get('time_remaining_hours'),
            'strike_date': event_info.get('strike_date', ''),
            'strike_period': event_info.get('strike_period', ''),
            'mutually_exclusive': is_mutually_exclusive,
            'markets': markets_with_odds,
            'research_summary': probability_extraction.overall_summary,
            'market_probabilities': [
                {
                    'ticker': mp.ticker,
                    'title': mp.title,
                    'research_probability': mp.research_probability,
                    'reasoning': mp.reasoning,
                    'confidence': mp.confidence
                }
                for mp in probability_extraction.markets
            ]
        }
        
        # Create prompt for OpenAI
        mutually_exclusive_guidance = ""
        if is_mutually_exclusive:
            mutually_exclusive_guidance = """
            
            🚨 MUTUALLY EXCLUSIVE EVENT - STRATEGIC HEDGE BETTING:
            This event is MUTUALLY EXCLUSIVE - only ONE outcome can be true.
            
            STRATEGIC BETTING APPROACH:
            - You CAN place multiple bets (YES and NO) with different position sizes
            - Focus on creating a profitable hedge portfolio across outcomes
            - Primary strategy: Find the best value opportunity for your largest YES bet
            - Secondary strategy: Place smaller YES bets on other good value opportunities
            - Tertiary strategy: Place NO bets on clearly overpriced outcomes
            - Key principle: Position sizing should reflect probability and value, not equal amounts
            
            POSITION SIZING GUIDELINES:
            - Largest bet: Best value opportunity (highest edge)
            - Medium bets: Good value opportunities (moderate edge)
            - Small bets: Decent value or hedge positions
            - Consider potential profit/loss scenarios across different outcomes
            
            Example: If researching shows A=50%, B=30%, C=20% but market prices A=30%, B=40%, C=30%:
            - BUY YES A (large) - undervalued by 20 points
            - BUY NO B (medium) - overvalued by 10 points  
            - BUY YES C (small) - undervalued by 10 points
            """
        else:
            mutually_exclusive_guidance = """
            
            NON-MUTUALLY EXCLUSIVE EVENT:
            Multiple outcomes in this event can be true simultaneously.
            You can place multiple YES bets if there are multiple good opportunities.
            Position sizing should reflect individual value opportunities.
            """
        
        prompt = f"""
        You are a professional prediction market trader. Based on the research provided for this event AND the current market odds, 
        make betting decisions for the individual markets within this event.
        
        Max bet per market: ${self.config.max_bet_amount}
        {mutually_exclusive_guidance}
        
        IMPORTANT TRADING CONSTRAINTS:
        - You can ONLY buy YES or buy NO positions (no shorting or sophisticated selling)
        - When you buy YES, you profit if the outcome is YES
        - When you buy NO, you profit if the outcome is NO
        - Prices are in cents (0-100, where 50 = 50% probability)
        - Look for value opportunities where research probability differs significantly from market odds
        
        STRUCTURED PROBABILITY DATA AVAILABLE:
        - Each market has a research_probability (0-100%) extracted from deep research
        - Each market has detailed reasoning for the probability estimate
        - Use these precise probabilities to calculate edges and alpha ratios
        - The research_summary provides overall context
        
        Event Data, Research Probabilities, and Current Market Odds:
        {json.dumps(single_event_data, indent=2)}
        
        For each market, decide:
        1. Action: "buy_yes", "buy_no", or "skip"
        2. Confidence: 0-1 (only bet if confidence > 0.75 for primary positions, > 0.85 for hedge positions)
        3. Amount: How much to bet (max ${self.config.max_bet_amount})
        4. Reasoning: Brief explanation comparing research prediction to current market odds
        
        STRATEGIC BETTING APPROACH - BE HIGHLY SELECTIVE:
        - SKIP most markets - only bet on exceptional opportunities
        - Primary position: Find the ONE best value opportunity (highest edge × confidence)
        - Secondary positions: Maximum 1-2 hedge bets only if they're truly exceptional
        - Minimum edge requirement: Research probability must differ by at least 5 percentage points from market odds
        - Focus on quality over quantity - better to make 1 great bet than 5 mediocre ones
        
        RISK-ADJUSTED FILTERING (HEDGE-FUND STYLE):
        - Only place bets with strong statistical edge: R-score (z-score) >= {self.config.z_threshold}
        - R-score measures how many standard deviations away the market is from fair value
        - Higher R-scores indicate higher conviction opportunities with better risk-adjusted returns
        - Example: R-score of 2.0 means the market is 2 standard deviations mis-priced (97.5th percentile opportunity)
        - SKIP all bets with R-score below the threshold - focus on exceptional statistical opportunities
        
        POSITION SIZING STRATEGY:
        - Use Kelly criterion for optimal position sizing based on edge and risk
        - Higher R-score opportunities get larger position sizes (within risk limits)
        - Maximum position size capped at ${self.config.max_bet_amount} and {self.config.max_kelly_bet_fraction*100}% of bankroll
        - Most markets: SKIP - only bet on the highest R-score opportunities
        
        RISK-ADJUSTED EDGE CALCULATION:
        - Bot automatically calculates R-score (z-score) for statistical edge measurement
        - R-score accounts for both probability difference AND volatility/risk
        - Higher R-scores indicate better risk-adjusted opportunities
        - Kelly sizing automatically optimizes position size based on edge and risk
        
        Return your analysis in the specified JSON format.
        """
        
        try:
            # Use Responses API structured outputs
            from openai_utils import responses_parse_pydantic
            analysis = await responses_parse_pydantic(
                self.openai_client,
                model=self.config.openai.model,
                messages=[
                    {"role": "system", "content": "You are a professional prediction market trader."},
                    {"role": "user", "content": prompt}
                ],
                response_format=MarketAnalysis,
                reasoning_effort="high",
                text_verbosity="high",
            )
            
            # Enrich decisions with human-readable names
            analysis = self._add_human_readable_names(analysis, event_info, markets)
            
            # Apply alpha threshold validation to ensure minimum edge requirements
            analysis = self._apply_alpha_threshold_validation(analysis, event_ticker, markets, probability_extraction, market_odds)
            
            # Apply strategic filtering to ensure selective betting
            analysis = self._apply_strategic_filtering(analysis, event_ticker)
            
            # Apply portfolio selection to hold only the N highest R-scores
            analysis = self.apply_portfolio_selection(analysis, event_ticker)
            
            # Post-process for mutually exclusive events: ensure only one YES bet
            if is_mutually_exclusive:
                analysis = self._enforce_mutually_exclusive_constraint(analysis, event_ticker)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating decisions for {event_ticker}: {e}")
            # Return empty analysis for this event
            return MarketAnalysis(
                decisions=[],
                total_recommended_bet=0.0,
                high_confidence_bets=0,
                summary=f"Error generating decisions for {event_ticker}"
            )
    
    def _add_human_readable_names(self, analysis: MarketAnalysis, event_info: Dict[str, Any], markets: List[Dict[str, Any]]) -> MarketAnalysis:
        """Add human-readable names to betting decisions."""
        # Create lookup for market names
        market_names = {market.get('ticker', ''): market.get('title', '') for market in markets}
        
        # Update each decision with readable names
        for decision in analysis.decisions:
            decision.event_name = event_info.get('title', '')
            decision.market_name = market_names.get(decision.ticker, decision.ticker)
        
        return analysis
    
    def _generate_readable_market_name(self, ticker: str) -> str:
        """Generate a readable market name from ticker."""
        return ticker.replace('-', ' ').replace('_', ' ').title()
    
    def _enforce_mutually_exclusive_constraint(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """Ensure only one YES bet for mutually exclusive events."""
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        yes_bets = [d for d in actionable_decisions if d.action == "buy_yes"]
        
        if len(yes_bets) <= 1:
            # Already compliant or no YES bets
            return analysis
        
        # Multiple YES bets found - keep only the highest confidence one
        logger.warning(f"Event {event_ticker}: Multiple YES bets found in mutually exclusive event. Keeping only highest confidence bet.")
        
        # Sort YES bets by confidence (highest first)
        yes_bets.sort(key=lambda x: x.confidence, reverse=True)
        best_yes_bet = yes_bets[0]
        
        # Convert other YES bets to SKIP
        filtered_decisions = []
        for decision in analysis.decisions:
            if decision.action == "buy_yes" and decision.ticker != best_yes_bet.ticker:
                # Convert to skip
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to mutually exclusive constraint (kept {best_yes_bet.ticker} with higher confidence)",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                filtered_decisions.append(skip_decision)
            else:
                filtered_decisions.append(decision)
        
        # Update the analysis
        analysis.decisions = filtered_decisions
        
        # Recalculate totals
        analysis.total_recommended_bet = sum(d.amount for d in filtered_decisions)
        analysis.high_confidence_bets = len([d for d in filtered_decisions if d.action != "skip" and d.confidence > 0.7])
        
        return analysis
    
    def _apply_alpha_threshold_validation(self, analysis: MarketAnalysis, event_ticker: str, markets: List[Dict[str, Any]], probability_extraction: ProbabilityExtraction, market_odds: Dict[str, Dict[str, Any]]) -> MarketAnalysis:
        """Apply risk-adjusted threshold validation and enrich decisions with metrics."""
        validated_decisions = []
        
        for decision in analysis.decisions:
            if decision.action == "skip":
                validated_decisions.append(decision)
                continue
            
            # Find the market probability for this decision
            market_prob = None
            for market_data in probability_extraction.markets:
                if market_data.ticker == decision.ticker:
                    market_prob = market_data.research_probability
                    break
            
            if market_prob is None:
                # If we can't find probability data, skip the bet
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to missing probability data",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                validated_decisions.append(skip_decision)
                continue
            
            # Find current market price from market_odds dictionary
            market_odds_data = None
            if decision.ticker in market_odds:
                ticker_odds = market_odds[decision.ticker]
                if decision.action == "buy_yes":
                    market_odds_data = ticker_odds.get('yes_ask', 0) / 100.0  # Convert to probability
                elif decision.action == "buy_no":
                    market_odds_data = ticker_odds.get('no_ask', 0) / 100.0  # Convert to probability
            
            if market_odds_data is None or market_odds_data == 0:
                # If we can't find market odds, skip the bet
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to missing market odds",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                validated_decisions.append(skip_decision)
                continue
            
            # Convert probabilities to 0-1 range
            research_prob = market_prob / 100.0
            
            # Calculate risk-adjusted metrics
            risk_metrics = self.calculate_risk_adjusted_metrics(
                research_prob, market_odds_data, decision.action
            )

            # Apply TrendRadar signal influence
            signal_influence = {"confidence_boost": 0.0, "kelly_multiplier": 1.0, "should_override_skip": False, "reasoning": "", "signal_direction": None}
            best_signal = None  # Track the best matching signal for persistence
            if event_ticker in self.event_signals:
                for signal in self.event_signals[event_ticker]:
                    # Check if this signal matches the decision's market
                    if signal.matches_event(decision.market_name or decision.ticker):
                        signal_config = SignalConfig(
                            max_confidence_boost=self.config.trendradar.max_confidence_boost,
                            strong_signal_threshold=self.config.trendradar.strong_signal_threshold,
                            min_source_count=self.config.trendradar.min_source_count,
                            aligned_signal_kelly_multiplier=self.config.trendradar.aligned_signal_kelly_multiplier,
                            enable_skip_override=self.config.trendradar.enable_skip_override,
                            skip_override_min_strength=self.config.trendradar.skip_override_min_strength,
                            skip_override_min_sources=self.config.trendradar.skip_override_min_sources
                        )
                        influence = calculate_signal_influence(
                            signal, decision.action, decision.confidence, signal_config
                        )
                        # Take the strongest influence (highest absolute boost)
                        if abs(influence["confidence_boost"]) > abs(signal_influence["confidence_boost"]):
                            signal_influence = influence
                            best_signal = signal  # Track the best signal for persistence

            # Apply confidence boost from signals
            original_confidence = decision.confidence
            boosted_confidence = decision.confidence + signal_influence["confidence_boost"]
            boosted_confidence = max(0.0, min(1.0, boosted_confidence))

            # Structured logging for signal influence (for auditing)
            if signal_influence["confidence_boost"] != 0:
                logger.info(
                    f"Signal influence applied | "
                    f"ticker={decision.ticker} | "
                    f"direction={signal_influence['signal_direction']} | "
                    f"confidence_before={original_confidence:.3f} | "
                    f"confidence_after={boosted_confidence:.3f} | "
                    f"boost={signal_influence['confidence_boost']:+.3f} | "
                    f"kelly_mult={signal_influence['kelly_multiplier']:.2f} | "
                    f"override_skip={signal_influence['should_override_skip']}"
                )

            decision.confidence = boosted_confidence

            # Attach TrendRadar signal data to decision for persistence
            if best_signal is not None:
                decision.signal_applied = True
                decision.signal_direction = signal_influence.get("signal_direction")
                decision.signal_topic = best_signal.topic
                decision.signal_sentiment = best_signal.sentiment
                decision.signal_strength = best_signal.strength
                decision.signal_source_count = best_signal.source_count
                decision.confidence_boost = signal_influence.get("confidence_boost", 0.0)
                decision.kelly_multiplier = signal_influence.get("kelly_multiplier", 1.0)
                decision.override_skip_triggered = signal_influence.get("should_override_skip", False)
                decision.signal_reasoning = signal_influence.get("reasoning", "")

            # Apply threshold filtering - use R-score filtering by default
            should_accept = False
            rejection_reason = ""

            # Use R-score (z-score) filtering - the new standard
            if risk_metrics["r_score"] >= self.config.z_threshold:
                should_accept = True
            elif signal_influence["should_override_skip"]:
                # Strong signal can override skip decision
                should_accept = True
                logger.info(f"Signal override for {decision.ticker}: {signal_influence['reasoning']}")
            else:
                rejection_reason = f"R-score {risk_metrics['r_score']:.2f} below z-threshold {self.config.z_threshold:.2f}"

            if should_accept:
                # Calculate quality-adjusted Kelly position size (uses R-score and confidence)
                if self.config.enable_kelly_sizing:
                    kelly_size = self.calculate_quality_adjusted_position_size(
                        risk_metrics["kelly_fraction"],
                        risk_metrics["r_score"],
                        decision.confidence
                    )
                    # Apply signal Kelly multiplier
                    kelly_before = kelly_size
                    kelly_size *= signal_influence["kelly_multiplier"]
                    kelly_size = min(kelly_size, self.config.max_bet_amount)

                    # Log Kelly multiplier application (for auditing)
                    if signal_influence["kelly_multiplier"] != 1.0:
                        logger.info(
                            f"Kelly multiplier applied | "
                            f"ticker={decision.ticker} | "
                            f"multiplier={signal_influence['kelly_multiplier']:.2f} | "
                            f"kelly_before=${kelly_before:.2f} | "
                            f"kelly_after=${kelly_size:.2f}"
                        )

                    decision.amount = kelly_size
                
                # Enrich decision with risk metrics
                decision.expected_return = risk_metrics["expected_return"]
                decision.r_score = risk_metrics["r_score"]
                decision.kelly_fraction = risk_metrics["kelly_fraction"]
                decision.market_price = market_odds_data
                decision.research_probability = research_prob
                
                validated_decisions.append(decision)
            else:
                # Convert to skip if threshold not met
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped: {rejection_reason}",
                    event_name=decision.event_name,
                    market_name=decision.market_name,
                    expected_return=risk_metrics["expected_return"],
                    r_score=risk_metrics["r_score"],
                    kelly_fraction=risk_metrics["kelly_fraction"],
                    market_price=market_odds_data,
                    research_probability=research_prob,
                    # Copy TrendRadar signal data from original decision
                    signal_applied=getattr(decision, 'signal_applied', False),
                    signal_direction=getattr(decision, 'signal_direction', None),
                    signal_topic=getattr(decision, 'signal_topic', None),
                    signal_sentiment=getattr(decision, 'signal_sentiment', None),
                    signal_strength=getattr(decision, 'signal_strength', None),
                    signal_source_count=getattr(decision, 'signal_source_count', None),
                    confidence_boost=getattr(decision, 'confidence_boost', 0.0),
                    kelly_multiplier=getattr(decision, 'kelly_multiplier', 1.0),
                    override_skip_triggered=getattr(decision, 'override_skip_triggered', False),
                    signal_reasoning=getattr(decision, 'signal_reasoning', None)
                )
                validated_decisions.append(skip_decision)
                logger.info(f"Filtered out {decision.ticker} - {rejection_reason}")
        
        # Update analysis with validated decisions
        analysis.decisions = validated_decisions
        
        # Recalculate totals
        analysis.total_recommended_bet = sum(d.amount for d in validated_decisions if d.action != "skip")
        analysis.high_confidence_bets = len([d for d in validated_decisions if d.action != "skip" and d.confidence > 0.7])
        
        return analysis
    
    def _apply_strategic_filtering(self, analysis: MarketAnalysis, event_ticker: str) -> MarketAnalysis:
        """Apply strategic filtering to ensure selective betting."""
        filtered_decisions = []
        
        for decision in analysis.decisions:
            if decision.action == "skip":
                filtered_decisions.append(decision)
                continue
            
            # Apply confidence threshold filter
            if decision.confidence < 0.6:
                skip_decision = BettingDecision(
                    ticker=decision.ticker,
                    action="skip",
                    confidence=decision.confidence,
                    amount=0.0,
                    reasoning=f"Skipped due to low confidence: {decision.confidence:.2f} < 0.6",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                filtered_decisions.append(skip_decision)
                continue
            
            # Apply bet amount limits
            max_bet = min(decision.amount, self.config.max_bet_amount)
            if max_bet != decision.amount:
                # Adjust bet amount to maximum allowed
                adjusted_decision = BettingDecision(
                    ticker=decision.ticker,
                    action=decision.action,
                    confidence=decision.confidence,
                    amount=max_bet,
                    reasoning=f"Amount adjusted to max limit: {decision.reasoning}",
                    event_name=decision.event_name,
                    market_name=decision.market_name
                )
                filtered_decisions.append(adjusted_decision)
            else:
                filtered_decisions.append(decision)
        
        # Update analysis with filtered decisions
        analysis.decisions = filtered_decisions
        
        # Recalculate totals
        analysis.total_recommended_bet = sum(d.amount for d in filtered_decisions if d.action != "skip")
        analysis.high_confidence_bets = len([d for d in filtered_decisions if d.action != "skip" and d.confidence > 0.7])
        
        return analysis
    
    async def place_bets(self, analysis: MarketAnalysis, market_odds: Dict[str, Dict[str, Any]], probability_extractions: Dict[str, ProbabilityExtraction]):
        """Place bets based on the analysis."""
        self.console.print(f"\n[bold]Step 6: Placing bets...[/bold]")
        
        if not analysis.decisions:
            self.console.print("[yellow]No betting decisions to execute[/yellow]")
            return
        
        actionable_decisions = [d for d in analysis.decisions if d.action != "skip"]
        
        if not actionable_decisions:
            self.console.print("[yellow]No actionable betting decisions to execute[/yellow]")
            return
        
        self.console.print(f"Found {len(actionable_decisions)} actionable decisions")
        
        for decision in actionable_decisions:
            if self.config.dry_run:
                self.console.print(f"[blue]DRY RUN: Would place {decision.action} bet of ${decision.amount} on {decision.ticker}[/blue]")
            else:
                # Convert action to Kalshi side format
                side = "yes" if decision.action == "buy_yes" else "no"
                result = await self.kalshi_client.place_order(decision.ticker, side, decision.amount)
                
                if result.get("success"):
                    self.console.print(f"[green][OK] Placed {decision.action} bet of ${decision.amount} on {decision.ticker}[/green]")
                else:
                    self.console.print(f"[red][X] Failed to place bet on {decision.ticker}: {result.get('error', 'Unknown error')}[/red]")
        
        if self.config.dry_run:
            self.console.print("\n[yellow]DRY RUN MODE: No actual bets were placed[/yellow]")
        else:
            self.console.print(f"\n[green][OK] Completed bet placement[/green]")
 
    def save_betting_decisions_to_csv(self, 
                                     analysis: MarketAnalysis, 
                                     research_results: Dict[str, str],
                                     probability_extractions: Dict[str, ProbabilityExtraction],
                                     market_odds: Dict[str, Dict[str, Any]],
                                     event_markets: Dict[str, Dict[str, Any]]) -> str:
        """
        Save betting decisions to a timestamped CSV file including raw research data.
        
        Args:
            analysis: The final betting decisions
            research_results: Raw research results by event ticker
            probability_extractions: Structured probability data by event ticker
            market_odds: Current market odds
            event_markets: Event and market information
            
        Returns:
            str: Path to the created CSV file
        """
        # Create output directory
        output_dir = Path("betting_decisions")
        output_dir.mkdir(exist_ok=True)
        
        # Generate timestamp for filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"betting_decisions_{timestamp}.csv"
        filepath = output_dir / filename
        
        # Prepare CSV data
        csv_data = []
        
        for decision in analysis.decisions:
            # Find corresponding research and market data
            event_ticker = None
            raw_research = ""
            research_summary = ""
            research_probability = None
            research_reasoning = ""
            market_yes_price = None
            market_no_price = None
            event_title = ""
            market_title = ""
            
            # Find the event ticker for this market
            for evt_ticker, data in event_markets.items():
                for market in data['markets']:
                    if market.get('ticker') == decision.ticker:
                        event_ticker = evt_ticker
                        event_title = data['event'].get('title', '')
                        market_title = market.get('title', '')
                        break
                if event_ticker:
                    break
            
            # Get raw research
            if event_ticker and event_ticker in research_results:
                raw_research = research_results[event_ticker]
            
            # Get probability extraction data
            if event_ticker and event_ticker in probability_extractions:
                extraction = probability_extractions[event_ticker]
                research_summary = extraction.overall_summary
                
                # Find market-specific probability
                for market_prob in extraction.markets:
                    if market_prob.ticker == decision.ticker:
                        research_probability = market_prob.research_probability
                        research_reasoning = market_prob.reasoning
                        break
            
            # Get market odds
            if decision.ticker in market_odds:
                odds = market_odds[decision.ticker]
                
                # Pull raw prices directly from API
                yes_bid = odds.get('yes_bid', 0)
                no_bid = odds.get('no_bid', 0)
                yes_ask = odds.get('yes_ask', 0)
                no_ask = odds.get('no_ask', 0)
                
                # If either side has no liquidity (both bid and ask are 0), skip this market in CSV
                has_yes_side = (yes_bid > 0) or (yes_ask > 0)
                has_no_side = (no_bid > 0) or (no_ask > 0)
                if not (has_yes_side and has_no_side):
                    # One-sided or illiquid market — exclude from CSV
                    continue

                # Use API ask prices directly for CSV yes/no price columns
                market_yes_price = yes_ask if yes_ask > 0 else None
                market_no_price = no_ask if no_ask > 0 else None
                # Also compute mids for reference (will be appended as extra columns if present)
                market_yes_mid = (yes_bid + yes_ask) / 2 if yes_bid > 0 and yes_ask > 0 else None
                market_no_mid = (no_bid + no_ask) / 2 if no_bid > 0 and no_ask > 0 else None
            
            # Skip one-sided markets in CSV output
            if (market_yes_price is None or market_yes_price == 0) or (market_no_price is None or market_no_price == 0):
                # No valid market on one side; exclude from CSV
                continue
            
            # Legacy edge calculation removed - now using R-score instead
            
            # Build base row (existing columns preserved)
            csv_row = {
                # Basic info
                'timestamp': datetime.now().isoformat(),
                'event_ticker': event_ticker or '',
                'event_title': event_title,
                'market_ticker': decision.ticker,
                'market_title': market_title,
                
                # Decision details
                'action': decision.action,
                'bet_amount': decision.amount,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                
                # Research data (original scale 0-100%)
                'research_probability': research_probability,
                'research_reasoning': research_reasoning,
                
                # Market data (original scale 0-100 cents)
                'market_yes_price': market_yes_price,
                'market_no_price': market_no_price,
                'market_yes_mid': market_yes_mid,
                'market_no_mid': market_no_mid,
                
                # Risk-adjusted metrics (hedge-fund style)
                'expected_return': decision.expected_return,
                'r_score': decision.r_score,
                'kelly_fraction': decision.kelly_fraction,
                
                # Normalized probabilities used in calculations (0-1 scale)
                'calc_market_prob': decision.market_price,  # Market probability used in R-score calc
                'calc_research_prob': decision.research_probability,  # Research probability used in R-score calc
                
                # Hedging info
                'is_hedge': decision.is_hedge,
                'hedge_for': decision.hedge_for or '',
                
                # Research context
                'research_summary': research_summary,
                'raw_research': raw_research.replace('\n', ' ').replace('\r', ' ') if raw_research else ''
            }

            # Enrich with full market attributes (do not remove any existing columns)
            # Put title fields first among the newly added market columns
            market_enriched = {}
            if decision.ticker in market_odds:
                m = market_odds[decision.ticker]
                # Title-related fields first
                market_enriched['market_title_full'] = m.get('title')
                market_enriched['market_subtitle'] = m.get('subtitle')
                market_enriched['market_yes_sub_title'] = m.get('yes_sub_title')
                market_enriched['market_no_sub_title'] = m.get('no_sub_title')
                
                # Remaining attributes (aligned to Kalshi market schema when present)
                market_enriched.update({
                    'market_event_ticker': m.get('event_ticker'),
                    'market_market_type': m.get('market_type'),
                    'market_open_time': m.get('open_time'),
                    'market_close_time': m.get('close_time'),
                    'market_expiration_time': m.get('expiration_time'),
                    'market_latest_expiration_time': m.get('latest_expiration_time'),
                    'market_settlement_timer_seconds': m.get('settlement_timer_seconds'),
                    'market_status': m.get('status'),
                    'market_response_price_units': m.get('response_price_units'),
                    'market_notional_value': m.get('notional_value'),
                    'market_tick_size': m.get('tick_size'),
                    'market_yes_bid': m.get('yes_bid'),
                    'market_yes_ask': m.get('yes_ask'),
                    'market_no_bid': m.get('no_bid'),
                    'market_no_ask': m.get('no_ask'),
                    'market_last_price': m.get('last_price'),
                    'market_previous_yes_bid': m.get('previous_yes_bid'),
                    'market_previous_yes_ask': m.get('previous_yes_ask'),
                    'market_previous_price': m.get('previous_price'),
                    'market_volume': m.get('volume'),
                    'market_volume_24h': m.get('volume_24h'),
                    'market_liquidity': m.get('liquidity'),
                    'market_open_interest': m.get('open_interest'),
                    'market_result': m.get('result'),
                    'market_can_close_early': m.get('can_close_early'),
                    'market_expiration_value': m.get('expiration_value'),
                    'market_category': m.get('category'),
                    'market_risk_limit_cents': m.get('risk_limit_cents'),
                    'market_rules_primary': m.get('rules_primary'),
                    'market_rules_secondary': m.get('rules_secondary'),
                    'market_settlement_value': m.get('settlement_value'),
                    'market_settlement_value_dollars': m.get('settlement_value_dollars'),
                })
            
            csv_row.update(market_enriched)
            csv_data.append(csv_row)
        
        # Write to CSV
        if csv_data:
            fieldnames = [
                # Basic info
                'timestamp', 'event_ticker', 'event_title', 'market_ticker', 'market_title',
                # Decision details  
                'action', 'bet_amount', 'confidence', 'reasoning',
                # Research data (original scale 0-100%)
                'research_probability', 'research_reasoning',
                # Market data (original scale 0-100 cents)
                'market_yes_price', 'market_no_price',
                # Risk-adjusted metrics (hedge-fund style)
                'expected_return', 'r_score', 'kelly_fraction',
                # Normalized probabilities used in calculations (0-1 scale)
                'calc_market_prob', 'calc_research_prob',
                # Hedging info
                'is_hedge', 'hedge_for',
                # Research context
                'research_summary', 'raw_research'
            ]

            # Append additional market attributes, with titles first among the new section
            additional_market_fields = [
                'market_title_full', 'market_subtitle', 'market_yes_sub_title', 'market_no_sub_title',
                'market_event_ticker', 'market_market_type', 'market_open_time', 'market_close_time',
                'market_expiration_time', 'market_latest_expiration_time', 'market_settlement_timer_seconds',
                'market_status', 'market_response_price_units', 'market_notional_value', 'market_tick_size',
                'market_yes_bid', 'market_yes_ask', 'market_no_bid', 'market_no_ask', 'market_last_price',
                'market_previous_yes_bid', 'market_previous_yes_ask', 'market_previous_price',
                'market_volume', 'market_volume_24h', 'market_liquidity', 'market_open_interest',
                'market_result', 'market_can_close_early', 'market_expiration_value', 'market_category',
                'market_risk_limit_cents', 'market_rules_primary', 'market_rules_secondary',
                'market_settlement_value', 'market_settlement_value_dollars'
            ]

            # Include any keys present in rows (regardless of None), ordered with the preferred list first
            base_set = set(fieldnames)
            present_keys = set()
            for row in csv_data:
                for k in row.keys():
                    if k not in base_set:
                        present_keys.add(k)

            ordered_extras = [f for f in additional_market_fields if f in present_keys]
            # Include any other unexpected keys deterministically (sorted for stability)
            remaining_extras = sorted(k for k in present_keys if k not in set(ordered_extras))
            fieldnames.extend(ordered_extras + remaining_extras)
            
            with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
            
            logger.info(f"Saved {len(csv_data)} betting decisions to {filepath}")
            self.console.print(f"[bold green][OK][/bold green] Betting decisions saved to: [blue]{filepath}[/blue]")
        else:
            logger.warning("No betting decisions to save")
            self.console.print("[yellow]No betting decisions to save[/yellow]")
        
        return str(filepath)

    async def save_decisions_to_db(
        self,
        analysis: MarketAnalysis,
        research_results: Dict[str, str],
        probability_extractions: Dict[str, ProbabilityExtraction],
        market_odds: Dict[str, Dict[str, Any]],
        event_markets: Dict[str, Dict[str, Any]]
    ) -> int:
        """
        Save betting decisions to SQLite database.

        Args:
            analysis: The final betting decisions
            research_results: Raw research results by event ticker
            probability_extractions: Structured probability data by event ticker
            market_odds: Current market odds
            event_markets: Event and market information

        Returns:
            int: Number of decisions saved
        """
        if not self.db:
            logger.warning("Database not initialized, skipping DB save")
            return 0

        logger.info(f"Starting save_decisions_to_db with {len(analysis.decisions)} decisions")

        decisions_to_save = []
        timestamp_now = datetime.now()  # asyncpg requires datetime object, not string

        for decision in analysis.decisions:
            # Find corresponding event and research data
            event_ticker = None
            raw_research = ""
            research_summary = ""
            research_probability = None
            research_reasoning = ""

            # Find the event ticker for this market
            for evt_ticker, data in event_markets.items():
                for market in data['markets']:
                    if market.get('ticker') == decision.ticker:
                        event_ticker = evt_ticker
                        break
                if event_ticker:
                    break

            # Get raw research
            if event_ticker and event_ticker in research_results:
                raw_research = research_results[event_ticker]

            # Get probability extraction data
            if event_ticker and event_ticker in probability_extractions:
                extraction = probability_extractions[event_ticker]
                research_summary = extraction.overall_summary

                for market_prob in extraction.markets:
                    if market_prob.ticker == decision.ticker:
                        research_probability = market_prob.research_probability
                        research_reasoning = market_prob.reasoning
                        break

            # Get market odds
            market_data = market_odds.get(decision.ticker, {})

            # Generate unique decision ID (use ISO format for timestamp)
            decision_id = f"{decision.ticker}_{timestamp_now.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

            decision_record = {
                'decision_id': decision_id,
                'timestamp': timestamp_now,
                'event_ticker': event_ticker or '',
                'event_title': decision.event_name or '',
                'market_ticker': decision.ticker,
                'market_title': decision.market_name or '',
                'action': decision.action,
                'bet_amount': decision.amount,
                'confidence': decision.confidence,
                'reasoning': decision.reasoning,
                'research_probability': research_probability,
                'research_reasoning': research_reasoning,
                'research_summary': research_summary,
                'raw_research': raw_research[:10000] if raw_research else '',  # Truncate for DB
                'market_yes_price': market_data.get('yes_ask'),
                'market_no_price': market_data.get('no_ask'),
                'market_yes_mid': (market_data.get('yes_bid', 0) + market_data.get('yes_ask', 0)) / 2 if market_data.get('yes_bid') and market_data.get('yes_ask') else None,
                'market_no_mid': (market_data.get('no_bid', 0) + market_data.get('no_ask', 0)) / 2 if market_data.get('no_bid') and market_data.get('no_ask') else None,
                'expected_return': decision.expected_return,
                'r_score': decision.r_score,
                'kelly_fraction': decision.kelly_fraction,
                'calc_market_prob': decision.market_price,
                'calc_research_prob': decision.research_probability,
                'is_hedge': decision.is_hedge if decision.is_hedge else False,
                'hedge_for': decision.hedge_for,
                'market_yes_bid': market_data.get('yes_bid'),
                'market_yes_ask': market_data.get('yes_ask'),
                'market_no_bid': market_data.get('no_bid'),
                'market_no_ask': market_data.get('no_ask'),
                'market_volume': market_data.get('volume'),
                'market_status': market_data.get('status'),
                'market_close_time': self._parse_datetime(market_data.get('close_time')),
                'status': 'pending' if decision.action != 'skip' else 'skipped',
                # Run tracking
                'run_mode': 'live' if not self.config.dry_run else 'dry_run',
                'run_id': self.run_id,
                # TrendRadar signal influence
                'signal_applied': getattr(decision, 'signal_applied', False),
                'signal_direction': getattr(decision, 'signal_direction', None),
                'signal_topic': getattr(decision, 'signal_topic', None),
                'signal_sentiment': getattr(decision, 'signal_sentiment', None),
                'signal_strength': getattr(decision, 'signal_strength', None),
                'signal_source_count': getattr(decision, 'signal_source_count', None),
                'confidence_boost': getattr(decision, 'confidence_boost', 0.0),
                'kelly_multiplier': getattr(decision, 'kelly_multiplier', 1.0),
                'override_skip_triggered': getattr(decision, 'override_skip_triggered', False),
                'signal_reasoning': getattr(decision, 'signal_reasoning', None)
            }

            decisions_to_save.append(decision_record)
            logger.debug(f"Prepared decision record for {decision.ticker}")

        logger.info(f"Prepared {len(decisions_to_save)} decision records for database")

        # Batch insert all decisions
        if decisions_to_save:
            try:
                count = await self.db.insert_decisions_batch(decisions_to_save)
                logger.info(f"Saved {count} decisions to database")
                self.console.print(f"[green][OK] Saved {count} decisions to SQLite database[/green]")
                return count
            except Exception as e:
                logger.error(f"Failed to save decisions to database: {e}")
                self.console.print(f"[red]Failed to save to database: {e}[/red]")
                return 0

        return 0

    async def run(self):
        """Main bot execution."""
        try:
            await self.initialize()

            # Check kill switch before trading
            if await self.check_kill_switch():
                self.console.print("[bold red]Kill switch active - daily loss limit exceeded. Trading halted.[/bold red]")
                if self.db:
                    await self.db.complete_run(
                        run_id=self.run_id,
                        events=0,
                        markets=0,
                        decisions=0,
                        bets=0,
                        wagered=0.0,
                        status='halted',
                        error='Kill switch triggered'
                    )
                return

            # Execute the main workflow
            # Step 1: Fetch events
            await broadcast_workflow_step(1, "Fetching events", "running",
                "Querying Kalshi API for early entry opportunities")
            events = await self.get_top_events()
            early_entry_enabled = self.config.early_entry.enabled if hasattr(self.config, 'early_entry') else False
            await broadcast_workflow_step(1, "Fetching events", "completed",
                details={"events_found": len(events), "early_entry_mode": early_entry_enabled})
            if not events:
                await broadcast_workflow_step(1, "Fetching events", "failed", "No events found")
                self.console.print("[red]No events found. Exiting.[/red]")
                return

            # Step 2: Process markets
            await broadcast_workflow_step(2, "Processing markets", "running",
                f"Processing markets for {len(events)} events")
            event_markets = await self.get_markets_for_events(events)
            total_markets = sum(len(data['markets']) for data in event_markets.values())
            await broadcast_workflow_step(2, "Processing markets", "completed",
                details={"events_count": len(event_markets), "markets_count": total_markets})
            if not event_markets:
                await broadcast_workflow_step(2, "Processing markets", "failed", "No markets found")
                self.console.print("[red]No markets found. Exiting.[/red]")
                return

            # Step 2.5: Filter positions
            await broadcast_workflow_step(3, "Filtering positions", "running",
                "Checking for existing positions to avoid duplicates")
            markets_before = sum(len(data['markets']) for data in event_markets.values())
            event_markets = await self.filter_markets_by_positions(event_markets)
            markets_after = sum(len(data['markets']) for data in event_markets.values())
            await broadcast_workflow_step(3, "Filtering positions", "completed",
                details={"filtered_count": markets_before - markets_after, "remaining_count": markets_after})
            if not event_markets:
                self.console.print("[red]No markets remaining after position filtering. Exiting.[/red]")
                return
            
            # Limit to max_events_to_analyze after position filtering
            if len(event_markets) > self.config.max_events_to_analyze:
                # Sort filtered events by volume_24h and take top N
                filtered_events_list = []
                for event_ticker, data in event_markets.items():
                    event = data['event']
                    volume_24h = event.get('volume_24h', 0)
                    filtered_events_list.append((event_ticker, data, volume_24h))

                # Sort by volume_24h (descending) and take top max_events_to_analyze
                filtered_events_list.sort(key=lambda x: x[2], reverse=True)
                top_events = filtered_events_list[:self.config.max_events_to_analyze]

                # Rebuild event_markets dict with only top events
                event_markets = {event_ticker: data for event_ticker, data, _ in top_events}

                self.console.print(f"[blue]* Limited to top {len(event_markets)} events by volume after position filtering[/blue]")

            # Step 4: Fetch trending signals from TrendRadar
            await broadcast_workflow_step(4, "Fetching signals", "running",
                "Querying TrendRadar for news sentiment signals")
            signals_by_event = await self.fetch_trending_signals(event_markets)
            total_signals = sum(len(s) for s in signals_by_event.values())
            await broadcast_workflow_step(4, "Fetching signals", "completed",
                details={"signals_count": total_signals, "events_with_signals": len(signals_by_event)})

            # Step 5: Research events
            await broadcast_workflow_step(5, "Researching events", "running",
                f"Deep research on {len(event_markets)} events via GPT-4o")
            research_results = await self.research_events(event_markets, signals_by_event)
            await broadcast_workflow_step(5, "Researching events", "completed",
                details={"events_researched": len(research_results)})
            if not research_results:
                await broadcast_workflow_step(5, "Researching events", "failed", "No research results")
                self.console.print("[red]No research results. Exiting.[/red]")
                return

            # Step 6: Extract probabilities
            await broadcast_workflow_step(6, "Extracting probabilities", "running",
                "Using GPT-5 to extract structured probabilities from research")
            probability_extractions = await self.extract_probabilities(research_results, event_markets)
            await broadcast_workflow_step(6, "Extracting probabilities", "completed",
                details={"events_processed": len(probability_extractions)})
            if not probability_extractions:
                await broadcast_workflow_step(6, "Extracting probabilities", "failed", "No probability extractions")
                self.console.print("[red]No probability extractions. Exiting.[/red]")
                return

            # Step 7: Fetch market odds
            await broadcast_workflow_step(7, "Fetching odds", "running",
                "Getting current bid/ask prices from Kalshi")
            market_odds = await self.get_market_odds(event_markets)
            await broadcast_workflow_step(7, "Fetching odds", "completed",
                details={"markets_count": len(market_odds)})
            if not market_odds:
                await broadcast_workflow_step(7, "Fetching odds", "failed", "No market odds found")
                self.console.print("[red]No market odds found. Exiting.[/red]")
                return

            # Step 8: Generate betting decisions
            await broadcast_workflow_step(8, "Generating decisions", "running",
                "Calculating R-scores and Kelly fractions for betting decisions")
            analysis = await self.get_betting_decisions(event_markets, probability_extractions, market_odds)
            actionable_count = len([d for d in analysis.decisions if d.action != "skip"])
            skip_count = len([d for d in analysis.decisions if d.action == "skip"])
            await broadcast_workflow_step(8, "Generating decisions", "completed",
                details={"decisions_count": len(analysis.decisions), "actionable_count": actionable_count, "skip_count": skip_count})
            
            # Step 9: Save & Broadcast
            await broadcast_workflow_step(9, "Saving decisions", "running",
                "Saving decisions to database and broadcasting to dashboard")

            # Save betting decisions to CSV with research data (if enabled)
            if self.config.database.save_to_csv:
                self.save_betting_decisions_to_csv(
                    analysis=analysis,
                    research_results=research_results,
                    probability_extractions=probability_extractions,
                    market_odds=market_odds,
                    event_markets=event_markets
                )

            # Save betting decisions to SQLite database
            decisions_saved = await self.save_decisions_to_db(
                analysis=analysis,
                research_results=research_results,
                probability_extractions=probability_extractions,
                market_odds=market_odds,
                event_markets=event_markets
            )

            # Broadcast decisions to dashboard for real-time updates
            if decisions_saved > 0:
                logger.info(f"Broadcasting {decisions_saved} decisions to dashboard")
                for decision in analysis.decisions:
                    market_data = market_odds.get(decision.ticker, {})
                    await broadcast_decision({
                        "decision_id": f"{decision.ticker}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        "timestamp": datetime.now().isoformat(),
                        "market_ticker": decision.ticker,
                        "market_title": decision.market_name or "",
                        "event_title": decision.event_name or "",
                        "action": decision.action,
                        "bet_amount": decision.amount,
                        "confidence": decision.confidence,
                        "r_score": decision.r_score,
                        "kelly_fraction": decision.kelly_fraction,
                        "expected_return": decision.expected_return,
                        "research_probability": decision.research_probability,
                        "market_price": decision.market_price,
                        "reasoning": decision.reasoning[:200] if decision.reasoning else "",
                        "run_mode": "live" if not self.config.dry_run else "dry_run",
                        "signal_applied": getattr(decision, 'signal_applied', False),
                        "signal_direction": getattr(decision, 'signal_direction', None),
                    })
                # Trigger KPI recalculation on dashboard
                await broadcast_kpi_update()

            await broadcast_workflow_step(9, "Saving decisions", "completed",
                details={"saved_count": decisions_saved, "broadcast_status": "completed"})

            # Step 10: Place bets
            await broadcast_workflow_step(10, "Placing bets", "running",
                f"Executing {actionable_count} bet orders on Kalshi")
            await self.place_bets(analysis, market_odds, probability_extractions)
            total_wagered = sum(d.amount for d in analysis.decisions if d.action != "skip")
            await broadcast_workflow_step(10, "Placing bets", "completed",
                details={"bets_placed": actionable_count, "total_wagered": total_wagered, "mode": "live" if not self.config.dry_run else "dry_run"})

            # Record successful run completion
            if self.db:
                actionable = [d for d in analysis.decisions if d.action != 'skip']
                await self.db.complete_run(
                    run_id=self.run_id,
                    events=len(event_markets),
                    markets=sum(len(d['markets']) for d in event_markets.values()),
                    decisions=len(analysis.decisions),
                    bets=len(actionable),
                    wagered=sum(d.amount for d in actionable),
                    status='completed'
                )

            self.console.print("\n[bold green]Bot execution completed![/bold green]")

        except Exception as e:
            self.console.print(f"[red]Bot execution error: {e}[/red]")
            logger.exception("Bot execution failed")

            # Record failed run
            if self.db:
                await self.db.complete_run(
                    run_id=self.run_id,
                    events=0, markets=0, decisions=0, bets=0, wagered=0.0,
                    status='failed',
                    error=str(e)
                )

        finally:
            # Clean up
            if self.research_client:
                await self.research_client.close()
            if self.kalshi_client:
                await self.kalshi_client.close()
            # Note: Database connection is managed globally, don't close here


async def run_migration():
    """Run data migration from CSV and JSON to SQLite."""
    from migrations import migrate_csv_files, migrate_calibration_json

    config = load_config()
    console = Console()

    if not config.database.enable_db:
        console.print("[red]Database is disabled. Enable it in config to run migration.[/red]")
        return

    try:
        db = await get_database(config.database.db_path)

        console.print("[bold blue]Starting data migration...[/bold blue]\n")

        # Migrate CSV files
        csv_summary = await migrate_csv_files(db, "betting_decisions", archive=False)

        # Migrate calibration JSON
        json_summary = await migrate_calibration_json(db, "calibration_data.json", backup=True)

        console.print("\n[bold green]Migration complete![/bold green]")

    except Exception as e:
        console.print(f"[red]Migration failed: {e}[/red]")
        logger.exception("Migration failed")


async def show_statistics():
    """Show performance statistics from the database."""
    from reconciliation import ReconciliationEngine

    config = load_config()
    console = Console()

    if not config.database.enable_db:
        console.print("[red]Database is disabled. Enable it in config to use statistics.[/red]")
        return

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
                ssl=config.database.pg_ssl
            )
            console.print("[green][OK] PostgreSQL database connected (Neon)[/green]")
        else:
            db = await get_database(config.database.db_path)
            console.print("[green][OK] SQLite database connected[/green]")

        # Create a dummy kalshi client (not needed for stats)
        kalshi = KalshiClient(
            config.kalshi,
            config.minimum_time_remaining_hours,
            config.max_markets_per_event
        )

        engine = ReconciliationEngine(db, kalshi)
        await engine.print_performance_report()

    except Exception as e:
        console.print(f"[red]Error loading statistics: {e}[/red]")
        logger.exception("Statistics failed")


async def main(live_trading: bool = False, max_close_ts: Optional[int] = None):
    """Main entry point."""
    bot = SimpleTradingBot(live_trading=live_trading, max_close_ts=max_close_ts)
    await bot.run()


def cli():
    """Command line interface entry point."""
    parser = argparse.ArgumentParser(
        description="Simple Kalshi trading bot with Octagon research and OpenAI decision making",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  uv run trading-bot                    # Run bot in dry run mode (default)
  uv run trading-bot --live             # Run bot with live trading enabled
  uv run trading-bot --help            # Show this help message
  
Configuration:
  Create a .env file with your API keys:
    KALSHI_API_KEY=your_kalshi_api_key
    KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\\n...\\n-----END RSA PRIVATE KEY-----"
    OCTAGON_API_KEY=your_octagon_api_key
    OPENAI_API_KEY=your_openai_api_key
    
  Optional settings:
    KALSHI_USE_DEMO=true               # Use demo environment (default: true)
    MAX_EVENTS_TO_ANALYZE=50           # Max events to analyze (default: 50)
    MAX_BET_AMOUNT=25.0                # Max bet per market (default: 25.0)
    RESEARCH_BATCH_SIZE=10             # Parallel research requests (default: 10)
    SKIP_EXISTING_POSITIONS=true       # Skip markets with existing positions (default: true)
    Z_THRESHOLD=1.5                    # Minimum R-score (z-score) for betting (default: 1.5)
    KELLY_FRACTION=0.5                 # Fraction of Kelly to use for position sizing (default: 0.5)
    BANKROLL=1000.0                    # Total bankroll for Kelly calculations (default: 1000.0)
    
  Trading modes:
    Default: Dry run mode - shows what trades would be made without placing real bets
    --live: Live trading mode - actually places bets (use with caution!)
        """
    )
    
    parser.add_argument(
        '--live',
        action='store_true',
        help='Enable live trading (default: dry run mode)'
    )
    parser.add_argument(
        '--max-expiration-hours',
        type=int,
        default=None,
        dest='max_expiration_hours',
        help='Only include markets that close within this many hours from now (minimum 1 hour).'
    )

    parser.add_argument(
        '--reconcile',
        action='store_true',
        help='Run outcome reconciliation to update P&L for settled markets'
    )

    parser.add_argument(
        '--stats',
        action='store_true',
        help='Show performance statistics from the database'
    )

    parser.add_argument(
        '--migrate',
        action='store_true',
        help='Migrate existing CSV and JSON data to SQLite database'
    )

    parser.add_argument(
        '--version',
        action='version',
        version='Kalshi Trading Bot 1.0.0'
    )

    # Parse arguments
    args = parser.parse_args()

    # Try to load config and run bot
    try:
        if args.reconcile:
            # Run reconciliation mode
            from reconciliation import run_reconciliation
            asyncio.run(run_reconciliation())
        elif args.stats:
            # Show statistics mode
            asyncio.run(show_statistics())
        elif args.migrate:
            # Run migration mode
            asyncio.run(run_migration())
        else:
            # Normal trading mode
            max_close_ts = None
            if args.max_expiration_hours is not None:
                hours = max(1, args.max_expiration_hours)
                max_close_ts = int(time.time()) + (hours * 3600)
            asyncio.run(main(live_trading=args.live, max_close_ts=max_close_ts))
    except Exception as e:
        console = Console()
        console.print(f"[red]Error: {e}[/red]")
        console.print("\n[yellow]Please check your .env file configuration.[/yellow]")
        console.print("[yellow]Run with --help for more information.[/yellow]")
        sys.exit(1)


if __name__ == "__main__":
    cli() 