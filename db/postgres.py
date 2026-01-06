"""
Async PostgreSQL database connection manager for Kalshi Deep Trading Bot.
Uses asyncpg for high-performance async PostgreSQL access.
"""

import asyncio
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import asyncpg
from loguru import logger

# Singleton instance
_pg_database_instance: Optional["PostgresDatabase"] = None
_lock = asyncio.Lock()


class PostgresDatabase:
    """Async PostgreSQL database connection manager."""

    def __init__(
        self,
        host: str,
        database: str,
        user: str,
        password: str,
        port: int = 5432,
        ssl: str = "require",
    ):
        self.host = host
        self.database = database
        self.user = user
        self.password = password
        self.port = port
        self.ssl = ssl
        self._pool: Optional[asyncpg.Pool] = None
        self._initialized = False

    async def connect(self) -> None:
        """Connect to the database and create connection pool."""
        if self._pool is not None:
            return

        logger.info(f"Connecting to PostgreSQL database: {self.host}/{self.database}")

        try:
            self._pool = await asyncpg.create_pool(
                host=self.host,
                port=self.port,
                database=self.database,
                user=self.user,
                password=self.password,
                ssl=self.ssl,
                min_size=2,
                max_size=10,
                command_timeout=60,
            )

            self._initialized = True
            logger.info("PostgreSQL connection pool established")

            # Run migrations to add new columns
            await self._run_migrations()

        except Exception as e:
            logger.error(f"Failed to connect to PostgreSQL: {e}")
            raise

    async def _run_migrations(self) -> None:
        """Run database migrations to add new columns."""
        migrations = [
            # Run tracking
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS run_mode TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS run_id TEXT",
            # TrendRadar signal influence
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS signal_applied BOOLEAN DEFAULT FALSE",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS signal_direction TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS signal_topic TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS signal_sentiment TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS signal_strength REAL",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS signal_source_count INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS confidence_boost REAL DEFAULT 0",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS kelly_multiplier REAL DEFAULT 1.0",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS override_skip_triggered BOOLEAN DEFAULT FALSE",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS signal_reasoning TEXT",
            # Indexes
            "CREATE INDEX IF NOT EXISTS idx_decisions_run_mode ON betting_decisions(run_mode)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON betting_decisions(run_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_signal_direction ON betting_decisions(signal_direction)",
            # =========================================================================
            # Position Tracking for Stop-Loss / Take-Profit (Migration 003)
            # =========================================================================
            # Entry price tracking
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS order_id TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS filled_price_cents INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS filled_contracts INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS filled_timestamp TIMESTAMPTZ",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS fill_cost_dollars REAL",
            # Stop-loss / take-profit configuration
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS stop_loss_pct REAL DEFAULT 0.15",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS take_profit_pct REAL DEFAULT 0.30",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS trailing_stop_pct REAL",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS sl_tp_enabled BOOLEAN DEFAULT TRUE",
            # Exit tracking
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_order_id TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_price_cents INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_contracts INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_timestamp TIMESTAMPTZ",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_reason TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_pnl_dollars REAL",
            # Real-time tracking fields
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS current_price_cents INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS unrealized_pnl_dollars REAL",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS unrealized_pnl_pct REAL",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS high_water_mark_cents INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS last_price_update TIMESTAMPTZ",
            # Position tracking indexes
            "CREATE INDEX IF NOT EXISTS idx_decisions_order_id ON betting_decisions(order_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_exit_order_id ON betting_decisions(exit_order_id)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_exit_reason ON betting_decisions(exit_reason)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_sl_tp_enabled ON betting_decisions(sl_tp_enabled)",
            "CREATE INDEX IF NOT EXISTS idx_decisions_filled_timestamp ON betting_decisions(filled_timestamp)",
            # Position snapshots table
            """CREATE TABLE IF NOT EXISTS position_snapshots (
                id SERIAL PRIMARY KEY,
                decision_id TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                snapshot_timestamp TIMESTAMPTZ DEFAULT NOW(),
                contracts INTEGER NOT NULL,
                entry_price_cents INTEGER NOT NULL,
                current_price_cents INTEGER NOT NULL,
                unrealized_pnl_dollars REAL,
                unrealized_pnl_pct REAL,
                bid_cents INTEGER,
                ask_cents INTEGER,
                spread_cents INTEGER,
                volume_24h INTEGER,
                stop_loss_distance_pct REAL,
                take_profit_distance_pct REAL,
                UNIQUE(decision_id, snapshot_timestamp)
            )""",
            # Exit history table
            """CREATE TABLE IF NOT EXISTS exit_history (
                id SERIAL PRIMARY KEY,
                decision_id TEXT NOT NULL,
                market_ticker TEXT NOT NULL,
                exit_timestamp TIMESTAMPTZ DEFAULT NOW(),
                exit_reason TEXT NOT NULL,
                exit_order_id TEXT,
                contracts INTEGER NOT NULL,
                entry_price_cents INTEGER NOT NULL,
                exit_price_cents INTEGER NOT NULL,
                realized_pnl_dollars REAL NOT NULL,
                realized_pnl_pct REAL NOT NULL,
                trigger_threshold REAL,
                high_water_mark_cents INTEGER,
                slippage_cents INTEGER,
                execution_time_ms INTEGER,
                created_at TIMESTAMPTZ DEFAULT NOW()
            )""",
            # Position snapshots indexes
            "CREATE INDEX IF NOT EXISTS idx_snapshots_decision_id ON position_snapshots(decision_id)",
            "CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON position_snapshots(snapshot_timestamp)",
            "CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON position_snapshots(market_ticker)",
            # Exit history indexes
            "CREATE INDEX IF NOT EXISTS idx_exit_history_decision_id ON exit_history(decision_id)",
            "CREATE INDEX IF NOT EXISTS idx_exit_history_reason ON exit_history(exit_reason)",
            "CREATE INDEX IF NOT EXISTS idx_exit_history_timestamp ON exit_history(exit_timestamp)",
        ]

        for migration in migrations:
            try:
                await self._pool.execute(migration)
            except Exception as e:
                # Ignore errors for already existing columns/indexes
                if (
                    "already exists" not in str(e).lower()
                    and "duplicate" not in str(e).lower()
                ):
                    logger.warning(f"Migration warning: {e}")

        logger.info("Database migrations completed")

    async def close(self) -> None:
        """Close the database connection pool."""
        if self._pool is not None:
            await self._pool.close()
            self._pool = None
            self._initialized = False
            logger.info("PostgreSQL connection pool closed")

    async def execute(self, query: str, *args) -> str:
        """Execute a query and return status."""
        if not self._pool:
            await self.connect()
        return await self._pool.execute(query, *args)

    async def executemany(self, query: str, args_list: List[Tuple]) -> None:
        """Execute a query with multiple parameter sets."""
        if not self._pool:
            await self.connect()
        async with self._pool.acquire() as conn:
            await conn.executemany(query, args_list)

    async def fetchone(self, query: str, *args) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one row as dict."""
        if not self._pool:
            await self.connect()
        row = await self._pool.fetchrow(query, *args)
        return dict(row) if row else None

    async def fetchall(self, query: str, *args) -> List[Dict[str, Any]]:
        """Execute query and fetch all rows as list of dicts."""
        if not self._pool:
            await self.connect()
        rows = await self._pool.fetch(query, *args)
        return [dict(row) for row in rows]

    async def insert_decision(self, decision: Dict[str, Any]) -> int:
        """Insert a betting decision and return its ID."""
        columns = list(decision.keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
        column_names = ", ".join(columns)

        query = f"""
            INSERT INTO betting_decisions ({column_names})
            VALUES ({placeholders})
            ON CONFLICT (decision_id) DO UPDATE SET
                updated_at = CURRENT_TIMESTAMP
            RETURNING id
        """
        row = await self.fetchone(query, *decision.values())
        return row["id"] if row else 0

    async def insert_decisions_batch(self, decisions: List[Dict[str, Any]]) -> int:
        """Insert multiple decisions in a batch."""
        if not decisions:
            return 0

        # Get all unique columns
        all_columns = set()
        for d in decisions:
            all_columns.update(d.keys())
        columns = sorted(all_columns)

        column_names = ", ".join(columns)
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])

        query = f"""
            INSERT INTO betting_decisions ({column_names})
            VALUES ({placeholders})
            ON CONFLICT (decision_id) DO NOTHING
        """

        if not self._pool:
            await self.connect()

        async with self._pool.acquire() as conn:
            # Build parameter tuples
            inserted = 0
            for d in decisions:
                params = tuple(d.get(col) for col in columns)
                try:
                    await conn.execute(query, *params)
                    inserted += 1
                except Exception as e:
                    logger.error(
                        f"Error inserting decision {d.get('decision_id', 'unknown')}: {e}"
                    )
                    logger.debug(f"Columns: {columns}")
                    logger.debug(f"Params: {params}")

        logger.info(f"Successfully inserted {inserted}/{len(decisions)} decisions")
        return inserted

    async def get_pending_decisions(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get decisions awaiting outcome reconciliation."""
        query = """
            SELECT * FROM betting_decisions
            WHERE status = 'pending' AND action != 'skip'
            ORDER BY timestamp DESC
            LIMIT $1
        """
        return await self.fetchall(query, limit)

    async def update_decision_outcome(
        self, decision_id: str, outcome: str, payout: float, profit_loss: float
    ) -> None:
        """Update a decision with its outcome."""
        query = """
            UPDATE betting_decisions
            SET outcome = $1,
                payout_amount = $2,
                profit_loss = $3,
                outcome_timestamp = $4,
                status = 'settled'
            WHERE decision_id = $5
        """
        await self.execute(
            query, outcome, payout, profit_loss, datetime.utcnow(), decision_id
        )

    async def insert_market_snapshot(self, snapshot: Dict[str, Any]) -> int:
        """Insert a market snapshot."""
        columns = list(snapshot.keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
        column_names = ", ".join(columns)

        query = f"""
            INSERT INTO market_snapshots ({column_names})
            VALUES ({placeholders})
            ON CONFLICT (market_ticker, snapshot_timestamp) DO NOTHING
            RETURNING id
        """
        row = await self.fetchone(query, *snapshot.values())
        return row["id"] if row else 0

    async def insert_calibration_record(self, record: Dict[str, Any]) -> int:
        """Insert a calibration record."""
        columns = list(record.keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
        column_names = ", ".join(columns)

        query = f"""
            INSERT INTO calibration_records ({column_names})
            VALUES ({placeholders})
            ON CONFLICT (prediction_id) DO NOTHING
            RETURNING id
        """
        row = await self.fetchone(query, *record.values())
        return row["id"] if row else 0

    async def update_calibration_outcome(
        self, prediction_id: str, outcome: float, actual_payout: Optional[float] = None
    ) -> None:
        """Update calibration record with outcome."""
        query = """
            UPDATE calibration_records
            SET outcome = $1,
                actual_payout = $2,
                resolved_timestamp = $3
            WHERE prediction_id = $4
        """
        await self.execute(
            query, outcome, actual_payout, datetime.utcnow(), prediction_id
        )

    async def get_daily_performance(self, date: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific date."""
        query = "SELECT * FROM performance_daily WHERE date = $1"
        return await self.fetchone(query, date)

    async def upsert_daily_performance(self, metrics: Dict[str, Any]) -> None:
        """Insert or update daily performance metrics."""
        columns = list(metrics.keys())
        placeholders = ", ".join([f"${i+1}" for i in range(len(columns))])
        column_names = ", ".join(columns)
        updates = ", ".join(
            [f"{col} = EXCLUDED.{col}" for col in columns if col != "date"]
        )

        query = f"""
            INSERT INTO performance_daily ({column_names})
            VALUES ({placeholders})
            ON CONFLICT (date) DO UPDATE SET {updates}
        """
        await self.execute(query, *metrics.values())

    async def start_run(self, run_id: str, config: Dict[str, Any]) -> int:
        """Record the start of a bot run."""
        query = """
            INSERT INTO run_history (
                run_id, started_at, mode, environment,
                max_events, z_threshold, kelly_fraction, status
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, 'running')
            RETURNING id
        """
        row = await self.fetchone(
            query,
            run_id,
            datetime.utcnow(),
            config.get("mode", "dry_run"),
            config.get("environment", "demo"),
            config.get("max_events", 50),
            config.get("z_threshold", 1.5),
            config.get("kelly_fraction", 0.5),
        )
        return row["id"] if row else 0

    async def complete_run(
        self,
        run_id: str,
        events: int,
        markets: int,
        decisions: int,
        bets: int,
        wagered: float,
        status: str = "completed",
        error: Optional[str] = None,
    ) -> None:
        """Record the completion of a bot run."""
        query = """
            UPDATE run_history
            SET completed_at = $1,
                events_analyzed = $2,
                markets_analyzed = $3,
                decisions_made = $4,
                bets_placed = $5,
                total_wagered = $6,
                status = $7,
                error_message = $8
            WHERE run_id = $9
        """
        await self.execute(
            query,
            datetime.utcnow(),
            events,
            markets,
            decisions,
            bets,
            wagered,
            status,
            error,
            run_id,
        )

    async def cache_event(self, event: Dict[str, Any]) -> None:
        """Cache an event from Kalshi API."""
        query = """
            INSERT INTO events_cache (
                event_ticker, title, subtitle, category, status,
                mutually_exclusive, volume, volume_24h, liquidity,
                open_interest, strike_date, strike_period, total_markets,
                expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14)
            ON CONFLICT (event_ticker) DO UPDATE SET
                title = EXCLUDED.title,
                volume = EXCLUDED.volume,
                volume_24h = EXCLUDED.volume_24h,
                cached_at = CURRENT_TIMESTAMP,
                expires_at = EXCLUDED.expires_at
        """
        await self.execute(
            query,
            event.get("event_ticker"),
            event.get("title"),
            event.get("subtitle"),
            event.get("category"),
            event.get("status", "open"),
            event.get("mutually_exclusive", False),
            event.get("volume", 0),
            event.get("volume_24h", 0),
            event.get("liquidity", 0),
            event.get("open_interest", 0),
            event.get("strike_date"),
            event.get("strike_period"),
            event.get("total_markets", 0),
            datetime.utcnow().replace(hour=datetime.utcnow().hour + 1),  # 1 hour cache
        )

    async def cache_market(self, market: Dict[str, Any], event_ticker: str) -> None:
        """Cache a market from Kalshi API."""
        query = """
            INSERT INTO markets_cache (
                market_ticker, event_ticker, title, subtitle,
                yes_bid, yes_ask, no_bid, no_ask, last_price,
                volume, volume_24h, status, result,
                open_time, close_time, expiration_time, expires_at
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
            ON CONFLICT (market_ticker) DO UPDATE SET
                yes_bid = EXCLUDED.yes_bid,
                yes_ask = EXCLUDED.yes_ask,
                no_bid = EXCLUDED.no_bid,
                no_ask = EXCLUDED.no_ask,
                volume = EXCLUDED.volume,
                status = EXCLUDED.status,
                cached_at = CURRENT_TIMESTAMP,
                expires_at = EXCLUDED.expires_at
        """
        await self.execute(
            query,
            market.get("ticker"),
            event_ticker,
            market.get("title"),
            market.get("subtitle"),
            market.get("yes_bid"),
            market.get("yes_ask"),
            market.get("no_bid"),
            market.get("no_ask"),
            market.get("last_price"),
            market.get("volume", 0),
            market.get("volume_24h", 0),
            market.get("status"),
            market.get("result"),
            market.get("open_time"),
            market.get("close_time"),
            market.get("expiration_time"),
            datetime.utcnow().replace(
                minute=datetime.utcnow().minute + 15
            ),  # 15 min cache
        )

    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        stats = {}

        # Total decisions
        row = await self.fetchone("SELECT COUNT(*) as count FROM betting_decisions")
        stats["total_decisions"] = row["count"] if row else 0

        # Pending decisions
        row = await self.fetchone(
            "SELECT COUNT(*) as count FROM betting_decisions WHERE status = 'pending'"
        )
        stats["pending_decisions"] = row["count"] if row else 0

        # Settled decisions
        row = await self.fetchone(
            "SELECT COUNT(*) as count FROM betting_decisions WHERE status = 'settled'"
        )
        stats["settled_decisions"] = row["count"] if row else 0

        # Total P&L
        row = await self.fetchone(
            "SELECT SUM(profit_loss) as total FROM betting_decisions WHERE status = 'settled'"
        )
        stats["total_pnl"] = float(row["total"]) if row and row["total"] else 0.0

        # Win rate
        row = await self.fetchone(
            """
            SELECT
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                COUNT(*) as total
            FROM betting_decisions
            WHERE status = 'settled' AND action != 'skip'
        """
        )
        if row and row["total"] > 0:
            stats["win_rate"] = row["wins"] / row["total"]
        else:
            stats["win_rate"] = 0.0

        return stats


async def get_postgres_database(
    host: str = "",
    database: str = "neondb",
    user: str = "",
    password: str = "",
    port: int = 5432,
    ssl: str = "require",
) -> PostgresDatabase:
    """Get or create the singleton PostgreSQL database instance."""
    global _pg_database_instance

    async with _lock:
        if _pg_database_instance is None:
            _pg_database_instance = PostgresDatabase(
                host=host,
                database=database,
                user=user,
                password=password,
                port=port,
                ssl=ssl,
            )
            await _pg_database_instance.connect()
        return _pg_database_instance


async def close_postgres_database() -> None:
    """Close the singleton PostgreSQL database instance."""
    global _pg_database_instance

    async with _lock:
        if _pg_database_instance is not None:
            await _pg_database_instance.close()
            _pg_database_instance = None
