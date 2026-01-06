"""
Position Monitor Service
========================
Real-time monitoring of open positions with stop-loss/take-profit triggers.

Usage:
    monitor = PositionMonitor(config, kalshi_client, database)
    await monitor.start()

    # Or run single check
    triggers = await monitor.check_all_positions()
"""

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

try:
    from config import BotConfig
    from kalshi_client import KalshiClient
except ImportError:
    BotConfig = Any
    KalshiClient = Any


class TriggerType(Enum):
    """Types of exit triggers."""

    STOP_LOSS = "stop_loss"
    TAKE_PROFIT = "take_profit"
    TRAILING_STOP = "trailing_stop"
    MANUAL = "manual"
    EXPIRY = "expiry"


@dataclass
class StopLossConfig:
    """Stop-loss and take-profit configuration."""

    enabled: bool = True

    # Default thresholds (can be overridden per-position)
    default_stop_loss_pct: float = 0.15  # Exit if down 15%
    default_take_profit_pct: float = 0.30  # Exit if up 30%

    # Trailing stop (optional)
    trailing_stop_enabled: bool = False
    trailing_stop_pct: float = 0.10  # Trail 10% below high water mark

    # Monitoring settings
    monitor_interval_seconds: int = 30  # Check positions every 30s
    price_staleness_seconds: int = 60  # Re-fetch if price older than 60s

    # Execution settings
    use_market_orders: bool = False  # True = market order, False = limit at bid
    slippage_tolerance_pct: float = 0.02  # 2% slippage tolerance for limits
    retry_failed_exits: bool = True
    max_exit_retries: int = 3

    # Risk controls
    max_simultaneous_exits: int = 5  # Don't exit more than 5 at once
    min_position_age_seconds: int = 300  # Don't exit positions < 5 min old

    # Notifications
    alert_on_trigger: bool = True
    alert_on_near_trigger_pct: float = 0.05  # Alert when within 5% of trigger


@dataclass
class PositionState:
    """Current state of a monitored position."""

    decision_id: str
    market_ticker: str
    side: str  # "yes" or "no"

    # Entry info
    entry_price_cents: int
    entry_contracts: int
    entry_timestamp: datetime
    entry_cost_dollars: float

    # Current state
    current_price_cents: int
    current_contracts: int
    last_update: datetime

    # P&L
    unrealized_pnl_dollars: float = 0.0
    unrealized_pnl_pct: float = 0.0

    # Triggers
    stop_loss_pct: float = 0.15
    take_profit_pct: float = 0.30
    trailing_stop_pct: Optional[float] = None
    high_water_mark_cents: int = 0

    # Status
    is_active: bool = True
    exit_pending: bool = False

    def calculate_pnl(self) -> None:
        """Calculate current P&L based on prices."""
        if self.entry_price_cents == 0:
            return

        # Value calculation (contracts * price in dollars)
        entry_value = self.entry_contracts * self.entry_price_cents / 100
        current_value = self.current_contracts * self.current_price_cents / 100

        self.unrealized_pnl_dollars = current_value - entry_value
        self.unrealized_pnl_pct = (
            (self.current_price_cents - self.entry_price_cents) / self.entry_price_cents
            if self.entry_price_cents > 0
            else 0.0
        )

        # Update high water mark for trailing stops
        if self.current_price_cents > self.high_water_mark_cents:
            self.high_water_mark_cents = self.current_price_cents

    def check_stop_loss(self) -> bool:
        """Check if stop-loss should trigger."""
        return self.unrealized_pnl_pct <= -self.stop_loss_pct

    def check_take_profit(self) -> bool:
        """Check if take-profit should trigger."""
        return self.unrealized_pnl_pct >= self.take_profit_pct

    def check_trailing_stop(self) -> bool:
        """Check if trailing stop should trigger."""
        if self.trailing_stop_pct is None or self.high_water_mark_cents == 0:
            return False

        trail_trigger = self.high_water_mark_cents * (1 - self.trailing_stop_pct)
        return self.current_price_cents <= trail_trigger

    def get_stop_loss_price(self) -> int:
        """Get the price at which stop-loss would trigger."""
        return int(self.entry_price_cents * (1 - self.stop_loss_pct))

    def get_take_profit_price(self) -> int:
        """Get the price at which take-profit would trigger."""
        return int(self.entry_price_cents * (1 + self.take_profit_pct))

    def get_trailing_stop_price(self) -> Optional[int]:
        """Get the current trailing stop price."""
        if self.trailing_stop_pct is None or self.high_water_mark_cents == 0:
            return None
        return int(self.high_water_mark_cents * (1 - self.trailing_stop_pct))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_id": self.decision_id,
            "ticker": self.market_ticker,
            "side": self.side,
            "contracts": self.current_contracts,
            "entry_price": self.entry_price_cents,
            "current_price": self.current_price_cents,
            "pnl_dollars": round(self.unrealized_pnl_dollars, 2),
            "pnl_pct": round(self.unrealized_pnl_pct * 100, 2),
            "stop_loss_pct": self.stop_loss_pct,
            "take_profit_pct": self.take_profit_pct,
            "stop_loss_price": self.get_stop_loss_price(),
            "take_profit_price": self.get_take_profit_price(),
            "trailing_stop_price": self.get_trailing_stop_price(),
            "high_water_mark": self.high_water_mark_cents,
            "is_active": self.is_active,
            "exit_pending": self.exit_pending,
            "last_update": self.last_update.isoformat() if self.last_update else None,
        }


@dataclass
class TriggerEvent:
    """Event fired when a trigger is hit."""

    position: PositionState
    trigger_type: TriggerType
    trigger_price_cents: int
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "decision_id": self.position.decision_id,
            "ticker": self.position.market_ticker,
            "side": self.position.side,
            "trigger_type": self.trigger_type.value,
            "trigger_price": self.trigger_price_cents,
            "entry_price": self.position.entry_price_cents,
            "contracts": self.position.current_contracts,
            "pnl_dollars": round(self.position.unrealized_pnl_dollars, 2),
            "pnl_pct": round(self.position.unrealized_pnl_pct * 100, 2),
            "timestamp": self.timestamp.isoformat(),
        }


class PositionMonitor:
    """
    Monitors open positions and triggers stop-loss/take-profit exits.

    Usage:
        monitor = PositionMonitor(config, kalshi_client, database)
        await monitor.start()

        # Or run single check
        await monitor.check_all_positions()
    """

    def __init__(
        self,
        config: BotConfig,
        kalshi: KalshiClient,
        db: Any,  # PostgresDatabase or Database
        sl_config: Optional[StopLossConfig] = None,
        on_trigger: Optional[Callable[[TriggerEvent], None]] = None,
        on_position_update: Optional[Callable[[PositionState], None]] = None,
    ):
        self.config = config
        self.sl_config = sl_config or StopLossConfig()
        self.kalshi = kalshi
        self.db = db
        self.on_trigger = on_trigger
        self.on_position_update = on_position_update

        self._positions: Dict[str, PositionState] = {}
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._exit_queue: asyncio.Queue = asyncio.Queue()
        self._stats = {
            "checks": 0,
            "triggers": 0,
            "successful_exits": 0,
            "failed_exits": 0,
        }

    @property
    def is_running(self) -> bool:
        """Check if monitor is running."""
        return self._running

    @property
    def position_count(self) -> int:
        """Get count of monitored positions."""
        return len(self._positions)

    @property
    def stats(self) -> Dict[str, int]:
        """Get monitoring statistics."""
        return self._stats.copy()

    async def start(self) -> None:
        """Start the monitoring loop."""
        if self._running:
            logger.warning("Position monitor already running")
            return

        self._running = True
        logger.info("Starting position monitor...")

        # Load existing positions from database
        await self._load_open_positions()

        # Start monitoring loop
        self._task = asyncio.create_task(self._monitor_loop())
        logger.info(f"Position monitor started with {len(self._positions)} positions")

    async def stop(self) -> None:
        """Stop the monitoring loop gracefully."""
        logger.info("Stopping position monitor...")
        self._running = False

        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

        logger.info("Position monitor stopped")

    async def _monitor_loop(self) -> None:
        """Main monitoring loop - runs until stopped."""
        while self._running:
            try:
                triggers = await self.check_all_positions()

                if triggers:
                    logger.info(f"Triggered {len(triggers)} exits this cycle")

                await asyncio.sleep(self.sl_config.monitor_interval_seconds)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitor loop: {e}")
                await asyncio.sleep(5)  # Brief pause on error

    async def _load_open_positions(self) -> None:
        """Load open positions from database."""
        query = """
            SELECT
                decision_id, market_ticker, action,
                filled_price_cents, filled_contracts, filled_timestamp,
                fill_cost_dollars, stop_loss_pct, take_profit_pct,
                trailing_stop_pct, high_water_mark_cents,
                current_price_cents, unrealized_pnl_dollars, unrealized_pnl_pct
            FROM betting_decisions
            WHERE status = 'pending'
              AND action IN ('buy_yes', 'buy_no')
              AND filled_contracts > 0
              AND sl_tp_enabled = TRUE
              AND exit_order_id IS NULL
        """

        try:
            rows = await self.db.fetchall(query)
        except Exception as e:
            logger.warning(f"Could not load positions from database: {e}")
            rows = []

        for row in rows:
            side = "yes" if row["action"] == "buy_yes" else "no"
            entry_price = row.get("filled_price_cents") or 0
            entry_contracts = row.get("filled_contracts") or 0

            position = PositionState(
                decision_id=row["decision_id"],
                market_ticker=row["market_ticker"],
                side=side,
                entry_price_cents=entry_price,
                entry_contracts=entry_contracts,
                entry_timestamp=row.get("filled_timestamp")
                or datetime.now(timezone.utc),
                entry_cost_dollars=row.get("fill_cost_dollars") or 0,
                current_price_cents=row.get("current_price_cents") or entry_price,
                current_contracts=entry_contracts,
                last_update=datetime.now(timezone.utc),
                stop_loss_pct=row.get("stop_loss_pct")
                or self.sl_config.default_stop_loss_pct,
                take_profit_pct=row.get("take_profit_pct")
                or self.sl_config.default_take_profit_pct,
                trailing_stop_pct=row.get("trailing_stop_pct"),
                high_water_mark_cents=row.get("high_water_mark_cents") or entry_price,
            )

            self._positions[row["decision_id"]] = position

        logger.info(f"Loaded {len(self._positions)} open positions for monitoring")

    async def check_all_positions(self) -> List[TriggerEvent]:
        """Check all positions for trigger conditions."""
        triggers: List[TriggerEvent] = []
        self._stats["checks"] += 1

        if not self._positions:
            return triggers

        # Batch fetch current prices for all tickers
        tickers = list(set(p.market_ticker for p in self._positions.values()))
        prices = await self._fetch_current_prices(tickers)

        exits_this_cycle = 0

        for decision_id, position in list(self._positions.items()):
            if not position.is_active or position.exit_pending:
                continue

            # Check max simultaneous exits
            if exits_this_cycle >= self.sl_config.max_simultaneous_exits:
                logger.warning("Max simultaneous exits reached, deferring remaining")
                break

            # Update current price
            price_data = prices.get(position.market_ticker, {})
            if position.side == "yes":
                new_price = price_data.get("yes_bid", position.current_price_cents)
            else:
                new_price = price_data.get("no_bid", position.current_price_cents)

            if new_price > 0:
                position.current_price_cents = new_price

            position.last_update = datetime.now(timezone.utc)
            position.calculate_pnl()

            # Notify position update
            if self.on_position_update:
                try:
                    self.on_position_update(position)
                except Exception as e:
                    logger.error(f"Error in position update callback: {e}")

            # Check triggers
            trigger = await self._check_triggers(position)
            if trigger:
                self._stats["triggers"] += 1
                triggers.append(trigger)

                # Execute exit
                success = await self._execute_exit(trigger)
                if success:
                    exits_this_cycle += 1

        # Update database with current prices/P&L
        await self._update_position_prices()

        return triggers

    async def _fetch_current_prices(self, tickers: List[str]) -> Dict[str, Dict]:
        """Fetch current prices for multiple tickers."""
        prices = {}

        for ticker in tickers:
            try:
                market = await self.kalshi.get_market_with_odds(ticker)
                prices[ticker] = {
                    "yes_bid": market.get("yes_bid", 0),
                    "yes_ask": market.get("yes_ask", 0),
                    "no_bid": market.get("no_bid", 0),
                    "no_ask": market.get("no_ask", 0),
                }
            except Exception as e:
                logger.error(f"Error fetching price for {ticker}: {e}")

        return prices

    async def _check_triggers(self, position: PositionState) -> Optional[TriggerEvent]:
        """Check if any trigger condition is met."""

        # Check minimum position age
        if position.entry_timestamp:
            age = (
                datetime.now(timezone.utc) - position.entry_timestamp
            ).total_seconds()
            if age < self.sl_config.min_position_age_seconds:
                return None

        # Check triggers in priority order: stop-loss > trailing > take-profit
        if position.check_stop_loss():
            logger.warning(
                f"STOP-LOSS triggered: {position.market_ticker} "
                f"P&L: {position.unrealized_pnl_pct:.1%} (limit: -{position.stop_loss_pct:.1%})"
            )
            return TriggerEvent(
                position=position,
                trigger_type=TriggerType.STOP_LOSS,
                trigger_price_cents=position.current_price_cents,
            )

        if position.check_trailing_stop():
            logger.warning(
                f"TRAILING-STOP triggered: {position.market_ticker} "
                f"Price: {position.current_price_cents}c (HWM: {position.high_water_mark_cents}c)"
            )
            return TriggerEvent(
                position=position,
                trigger_type=TriggerType.TRAILING_STOP,
                trigger_price_cents=position.current_price_cents,
            )

        if position.check_take_profit():
            logger.info(
                f"TAKE-PROFIT triggered: {position.market_ticker} "
                f"P&L: {position.unrealized_pnl_pct:.1%} (target: +{position.take_profit_pct:.1%})"
            )
            return TriggerEvent(
                position=position,
                trigger_type=TriggerType.TAKE_PROFIT,
                trigger_price_cents=position.current_price_cents,
            )

        # Check for near-trigger alerts
        if self.sl_config.alert_on_trigger:
            self._check_near_trigger_alerts(position)

        return None

    def _check_near_trigger_alerts(self, position: PositionState) -> None:
        """Check if position is near a trigger and log warning."""
        near_pct = self.sl_config.alert_on_near_trigger_pct

        # Near stop-loss
        stop_distance = position.unrealized_pnl_pct + position.stop_loss_pct
        if 0 < stop_distance < near_pct:
            logger.warning(
                f"NEAR STOP-LOSS: {position.market_ticker} is {stop_distance:.1%} from trigger"
            )

        # Near take-profit
        tp_distance = position.take_profit_pct - position.unrealized_pnl_pct
        if 0 < tp_distance < near_pct:
            logger.info(
                f"NEAR TAKE-PROFIT: {position.market_ticker} is {tp_distance:.1%} from trigger"
            )

    async def _execute_exit(self, trigger: TriggerEvent) -> bool:
        """Execute the exit order for a triggered position."""
        position = trigger.position
        position.exit_pending = True

        retries = 0
        max_retries = (
            self.sl_config.max_exit_retries if self.sl_config.retry_failed_exits else 1
        )

        while retries < max_retries:
            try:
                # Determine order type and price
                if self.sl_config.use_market_orders:
                    price_cents = None
                    order_type = "market"
                else:
                    # Use current bid with slippage tolerance
                    slippage = int(
                        position.current_price_cents
                        * self.sl_config.slippage_tolerance_pct
                    )
                    price_cents = max(1, position.current_price_cents - slippage)
                    order_type = "limit"

                # Place sell order
                result = await self.kalshi.sell_position(
                    ticker=position.market_ticker,
                    side=position.side,
                    contracts=position.current_contracts,
                    price_cents=price_cents,
                    order_type=order_type,
                )

                if result.get("success"):
                    # Record exit in database
                    await self._record_exit(position, trigger, result)

                    # Remove from active monitoring
                    position.is_active = False
                    if position.decision_id in self._positions:
                        del self._positions[position.decision_id]

                    self._stats["successful_exits"] += 1

                    # Fire callback
                    if self.on_trigger:
                        try:
                            self.on_trigger(trigger)
                        except Exception as e:
                            logger.error(f"Error in trigger callback: {e}")

                    logger.info(
                        f"Exit executed: {position.market_ticker} via {trigger.trigger_type.value} "
                        f"Order ID: {result.get('order_id')} P&L: ${position.unrealized_pnl_dollars:.2f}"
                    )
                    return True
                else:
                    logger.error(
                        f"Exit failed (attempt {retries + 1}): {result.get('error')}"
                    )
                    retries += 1

            except Exception as e:
                logger.error(f"Error executing exit (attempt {retries + 1}): {e}")
                retries += 1

            if retries < max_retries:
                await asyncio.sleep(2**retries)  # Exponential backoff

        # All retries failed
        self._stats["failed_exits"] += 1
        position.exit_pending = False
        logger.error(
            f"Failed to exit {position.market_ticker} after {max_retries} attempts"
        )
        return False

    async def _record_exit(
        self, position: PositionState, trigger: TriggerEvent, result: Dict
    ) -> None:
        """Record the exit in the database."""
        try:
            # Update betting_decisions
            await self.db.execute(
                """
                UPDATE betting_decisions
                SET exit_order_id = $1,
                    exit_price_cents = $2,
                    exit_contracts = $3,
                    exit_timestamp = $4,
                    exit_reason = $5,
                    exit_pnl_dollars = $6,
                    status = 'closed'
                WHERE decision_id = $7
                """,
                result.get("order_id"),
                trigger.trigger_price_cents,
                position.current_contracts,
                datetime.now(timezone.utc),
                trigger.trigger_type.value,
                position.unrealized_pnl_dollars,
                position.decision_id,
            )

            # Record exit event
            await self.db.execute(
                """
                INSERT INTO exit_events (
                    decision_id, market_ticker, trigger_type, trigger_price_cents,
                    entry_price_cents, contracts, side,
                    unrealized_pnl_dollars, unrealized_pnl_pct,
                    order_id, execution_status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11)
                """,
                position.decision_id,
                position.market_ticker,
                trigger.trigger_type.value,
                trigger.trigger_price_cents,
                position.entry_price_cents,
                position.current_contracts,
                position.side,
                position.unrealized_pnl_dollars,
                position.unrealized_pnl_pct,
                result.get("order_id"),
                "pending",
            )
        except Exception as e:
            logger.error(f"Error recording exit: {e}")

    async def _update_position_prices(self) -> None:
        """Batch update position prices in database."""
        for position in self._positions.values():
            try:
                await self.db.execute(
                    """
                    UPDATE betting_decisions
                    SET current_price_cents = $1,
                        unrealized_pnl_dollars = $2,
                        unrealized_pnl_pct = $3,
                        high_water_mark_cents = $4,
                        last_price_update = $5
                    WHERE decision_id = $6
                    """,
                    position.current_price_cents,
                    position.unrealized_pnl_dollars,
                    position.unrealized_pnl_pct,
                    position.high_water_mark_cents,
                    datetime.now(timezone.utc),
                    position.decision_id,
                )
            except Exception as e:
                logger.error(f"Error updating position {position.decision_id}: {e}")

    # === Manual Controls ===

    async def add_position(self, decision_id: str) -> bool:
        """Manually add a position to monitoring by decision_id."""
        if decision_id in self._positions:
            logger.warning(f"Position {decision_id} already monitored")
            return False

        query = """
            SELECT
                decision_id, market_ticker, action,
                filled_price_cents, filled_contracts, filled_timestamp,
                fill_cost_dollars, stop_loss_pct, take_profit_pct,
                trailing_stop_pct, high_water_mark_cents
            FROM betting_decisions
            WHERE decision_id = $1
        """

        try:
            row = await self.db.fetchone(query, decision_id)
            if not row:
                logger.error(f"Decision {decision_id} not found")
                return False

            side = "yes" if row["action"] == "buy_yes" else "no"
            entry_price = row.get("filled_price_cents") or 0

            position = PositionState(
                decision_id=row["decision_id"],
                market_ticker=row["market_ticker"],
                side=side,
                entry_price_cents=entry_price,
                entry_contracts=row.get("filled_contracts") or 0,
                entry_timestamp=row.get("filled_timestamp")
                or datetime.now(timezone.utc),
                entry_cost_dollars=row.get("fill_cost_dollars") or 0,
                current_price_cents=entry_price,
                current_contracts=row.get("filled_contracts") or 0,
                last_update=datetime.now(timezone.utc),
                stop_loss_pct=row.get("stop_loss_pct")
                or self.sl_config.default_stop_loss_pct,
                take_profit_pct=row.get("take_profit_pct")
                or self.sl_config.default_take_profit_pct,
                trailing_stop_pct=row.get("trailing_stop_pct"),
                high_water_mark_cents=row.get("high_water_mark_cents") or entry_price,
            )

            self._positions[decision_id] = position
            logger.info(f"Added position {decision_id} to monitoring")
            return True

        except Exception as e:
            logger.error(f"Error adding position: {e}")
            return False

    async def remove_position(self, decision_id: str) -> bool:
        """Remove a position from monitoring (without exiting)."""
        if decision_id in self._positions:
            del self._positions[decision_id]
            logger.info(f"Removed position {decision_id} from monitoring")
            return True
        return False

    async def update_triggers(
        self,
        decision_id: str,
        stop_loss_pct: Optional[float] = None,
        take_profit_pct: Optional[float] = None,
        trailing_stop_pct: Optional[float] = None,
    ) -> bool:
        """Update trigger levels for a position."""
        if decision_id not in self._positions:
            logger.error(f"Position {decision_id} not found")
            return False

        position = self._positions[decision_id]

        if stop_loss_pct is not None:
            position.stop_loss_pct = stop_loss_pct
        if take_profit_pct is not None:
            position.take_profit_pct = take_profit_pct
        if trailing_stop_pct is not None:
            position.trailing_stop_pct = trailing_stop_pct

        # Update in database
        try:
            await self.db.execute(
                """
                UPDATE betting_decisions
                SET stop_loss_pct = $1, take_profit_pct = $2, trailing_stop_pct = $3
                WHERE decision_id = $4
                """,
                position.stop_loss_pct,
                position.take_profit_pct,
                position.trailing_stop_pct,
                decision_id,
            )
            logger.info(
                f"Updated triggers for {decision_id}: SL={stop_loss_pct}, TP={take_profit_pct}"
            )
            return True
        except Exception as e:
            logger.error(f"Error updating triggers: {e}")
            return False

    async def manual_exit(self, decision_id: str, reason: str = "manual") -> bool:
        """Manually trigger an exit for a position."""
        if decision_id not in self._positions:
            logger.error(f"Position {decision_id} not found")
            return False

        position = self._positions[decision_id]
        trigger = TriggerEvent(
            position=position,
            trigger_type=TriggerType.MANUAL,
            trigger_price_cents=position.current_price_cents,
        )

        return await self._execute_exit(trigger)

    def get_position(self, decision_id: str) -> Optional[PositionState]:
        """Get a specific position by decision_id."""
        return self._positions.get(decision_id)

    def get_positions_summary(self) -> Dict[str, Any]:
        """Get summary of all monitored positions."""
        total_pnl = sum(p.unrealized_pnl_dollars for p in self._positions.values())
        total_value = sum(
            p.current_contracts * p.current_price_cents / 100
            for p in self._positions.values()
        )

        return {
            "count": len(self._positions),
            "total_unrealized_pnl": round(total_pnl, 2),
            "total_value": round(total_value, 2),
            "stats": self._stats.copy(),
            "positions": [p.to_dict() for p in self._positions.values()],
        }

    def get_positions_by_ticker(self, ticker: str) -> List[PositionState]:
        """Get all positions for a specific ticker."""
        return [p for p in self._positions.values() if p.market_ticker == ticker]

    def get_positions_near_trigger(
        self, threshold_pct: float = 0.05
    ) -> List[PositionState]:
        """Get positions that are near a trigger threshold."""
        near_positions = []

        for position in self._positions.values():
            # Near stop-loss
            stop_distance = position.unrealized_pnl_pct + position.stop_loss_pct
            if 0 < stop_distance < threshold_pct:
                near_positions.append(position)
                continue

            # Near take-profit
            tp_distance = position.take_profit_pct - position.unrealized_pnl_pct
            if 0 < tp_distance < threshold_pct:
                near_positions.append(position)

        return near_positions


# === Convenience functions for integration ===


async def create_position_monitor(
    config: BotConfig, kalshi: KalshiClient, db: Any, start_immediately: bool = True
) -> PositionMonitor:
    """Factory function to create and optionally start a position monitor."""
    sl_config = StopLossConfig(
        enabled=getattr(config, "sl_tp_enabled", True),
        default_stop_loss_pct=getattr(config, "default_stop_loss_pct", 0.15),
        default_take_profit_pct=getattr(config, "default_take_profit_pct", 0.30),
        monitor_interval_seconds=getattr(config, "monitor_interval_seconds", 30),
    )

    monitor = PositionMonitor(config=config, kalshi=kalshi, db=db, sl_config=sl_config)

    if start_immediately:
        await monitor.start()

    return monitor
