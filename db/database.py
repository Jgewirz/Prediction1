"""
Async SQLite database connection manager for Kalshi Deep Trading Bot.
"""

import asyncio
import aiosqlite
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from datetime import datetime
from loguru import logger

# Singleton instance
_database_instance: Optional["Database"] = None
_lock = asyncio.Lock()


class Database:
    """Async SQLite database connection manager."""

    def __init__(self, db_path: str = "trading_bot.db"):
        self.db_path = Path(db_path)
        self._connection: Optional[aiosqlite.Connection] = None
        self._initialized = False

    async def connect(self) -> None:
        """Connect to the database and run migrations."""
        if self._connection is not None:
            return

        logger.info(f"Connecting to SQLite database: {self.db_path}")

        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = await aiosqlite.connect(str(self.db_path))
        self._connection.row_factory = aiosqlite.Row

        # Enable foreign keys
        await self._connection.execute("PRAGMA foreign_keys = ON")

        # Run schema migrations
        await self._run_migrations()

        self._initialized = True
        logger.info("Database connection established and schema initialized")

    async def _run_migrations(self) -> None:
        """Run schema migrations from schema.sql."""
        schema_path = Path(__file__).parent / "schema.sql"

        if not schema_path.exists():
            logger.warning(f"Schema file not found: {schema_path}")
            return

        logger.info("Running database schema migrations...")

        with open(schema_path, 'r') as f:
            schema_sql = f.read()

        # Split by semicolons and execute each statement
        statements = [s.strip() for s in schema_sql.split(';') if s.strip()]

        for statement in statements:
            if statement and not statement.startswith('--'):
                try:
                    await self._connection.execute(statement)
                except Exception as e:
                    # Log but continue - some statements may already exist
                    if "already exists" not in str(e).lower():
                        logger.debug(f"Migration statement note: {e}")

        await self._connection.commit()
        logger.info("Schema migrations completed")

    async def close(self) -> None:
        """Close the database connection."""
        if self._connection is not None:
            await self._connection.close()
            self._connection = None
            self._initialized = False
            logger.info("Database connection closed")

    async def execute(self, query: str, params: Tuple = ()) -> aiosqlite.Cursor:
        """Execute a query and return cursor."""
        if not self._connection:
            await self.connect()
        return await self._connection.execute(query, params)

    async def executemany(self, query: str, params_list: List[Tuple]) -> aiosqlite.Cursor:
        """Execute a query with multiple parameter sets."""
        if not self._connection:
            await self.connect()
        return await self._connection.executemany(query, params_list)

    async def commit(self) -> None:
        """Commit the current transaction."""
        if self._connection:
            await self._connection.commit()

    async def fetchone(self, query: str, params: Tuple = ()) -> Optional[Dict[str, Any]]:
        """Execute query and fetch one row as dict."""
        cursor = await self.execute(query, params)
        row = await cursor.fetchone()
        return dict(row) if row else None

    async def fetchall(self, query: str, params: Tuple = ()) -> List[Dict[str, Any]]:
        """Execute query and fetch all rows as list of dicts."""
        cursor = await self.execute(query, params)
        rows = await cursor.fetchall()
        return [dict(row) for row in rows]

    async def insert_decision(self, decision: Dict[str, Any]) -> int:
        """Insert a betting decision and return its ID."""
        columns = list(decision.keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)

        query = f"INSERT OR REPLACE INTO betting_decisions ({column_names}) VALUES ({placeholders})"
        cursor = await self.execute(query, tuple(decision.values()))
        await self.commit()
        return cursor.lastrowid

    async def insert_decisions_batch(self, decisions: List[Dict[str, Any]]) -> int:
        """Insert multiple decisions in a batch."""
        if not decisions:
            return 0

        # Get all unique columns across all decisions
        all_columns = set()
        for d in decisions:
            all_columns.update(d.keys())
        columns = sorted(all_columns)

        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)

        query = f"INSERT OR REPLACE INTO betting_decisions ({column_names}) VALUES ({placeholders})"

        # Build parameter tuples, using None for missing columns
        params_list = []
        for d in decisions:
            params = tuple(d.get(col) for col in columns)
            params_list.append(params)

        await self.executemany(query, params_list)
        await self.commit()
        return len(decisions)

    async def get_pending_decisions(self, limit: int = 1000) -> List[Dict[str, Any]]:
        """Get decisions awaiting outcome reconciliation."""
        query = """
            SELECT * FROM betting_decisions
            WHERE status = 'pending' AND action != 'skip'
            ORDER BY timestamp DESC
            LIMIT ?
        """
        return await self.fetchall(query, (limit,))

    async def update_decision_outcome(
        self,
        decision_id: str,
        outcome: str,
        payout: float,
        profit_loss: float
    ) -> None:
        """Update a decision with its outcome."""
        query = """
            UPDATE betting_decisions
            SET outcome = ?,
                payout_amount = ?,
                profit_loss = ?,
                outcome_timestamp = ?,
                status = 'settled'
            WHERE decision_id = ?
        """
        await self.execute(query, (outcome, payout, profit_loss, datetime.utcnow().isoformat(), decision_id))
        await self.commit()

    async def insert_market_snapshot(self, snapshot: Dict[str, Any]) -> int:
        """Insert a market snapshot."""
        columns = list(snapshot.keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)

        query = f"INSERT OR REPLACE INTO market_snapshots ({column_names}) VALUES ({placeholders})"
        cursor = await self.execute(query, tuple(snapshot.values()))
        await self.commit()
        return cursor.lastrowid

    async def insert_calibration_record(self, record: Dict[str, Any]) -> int:
        """Insert a calibration record."""
        columns = list(record.keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)

        query = f"INSERT OR REPLACE INTO calibration_records ({column_names}) VALUES ({placeholders})"
        cursor = await self.execute(query, tuple(record.values()))
        await self.commit()
        return cursor.lastrowid

    async def update_calibration_outcome(
        self,
        prediction_id: str,
        outcome: float,
        actual_payout: Optional[float] = None
    ) -> None:
        """Update calibration record with outcome."""
        query = """
            UPDATE calibration_records
            SET outcome = ?,
                actual_payout = ?,
                resolved_timestamp = ?
            WHERE prediction_id = ?
        """
        await self.execute(query, (outcome, actual_payout, datetime.utcnow().isoformat(), prediction_id))
        await self.commit()

    async def get_daily_performance(self, date: str) -> Optional[Dict[str, Any]]:
        """Get performance metrics for a specific date."""
        query = "SELECT * FROM performance_daily WHERE date = ?"
        return await self.fetchone(query, (date,))

    async def upsert_daily_performance(self, metrics: Dict[str, Any]) -> None:
        """Insert or update daily performance metrics."""
        columns = list(metrics.keys())
        placeholders = ', '.join(['?' for _ in columns])
        column_names = ', '.join(columns)
        updates = ', '.join([f"{col} = excluded.{col}" for col in columns if col != 'date'])

        query = f"""
            INSERT INTO performance_daily ({column_names}) VALUES ({placeholders})
            ON CONFLICT(date) DO UPDATE SET {updates}
        """
        await self.execute(query, tuple(metrics.values()))
        await self.commit()

    async def start_run(self, run_id: str, config: Dict[str, Any]) -> int:
        """Record the start of a bot run."""
        query = """
            INSERT INTO run_history (
                run_id, started_at, mode, environment,
                max_events, z_threshold, kelly_fraction, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, 'running')
        """
        cursor = await self.execute(query, (
            run_id,
            datetime.utcnow().isoformat(),
            config.get('mode', 'dry_run'),
            config.get('environment', 'demo'),
            config.get('max_events', 50),
            config.get('z_threshold', 1.5),
            config.get('kelly_fraction', 0.5)
        ))
        await self.commit()
        return cursor.lastrowid

    async def complete_run(
        self,
        run_id: str,
        events: int,
        markets: int,
        decisions: int,
        bets: int,
        wagered: float,
        status: str = 'completed',
        error: Optional[str] = None
    ) -> None:
        """Record the completion of a bot run."""
        query = """
            UPDATE run_history
            SET completed_at = ?,
                events_analyzed = ?,
                markets_analyzed = ?,
                decisions_made = ?,
                bets_placed = ?,
                total_wagered = ?,
                status = ?,
                error_message = ?
            WHERE run_id = ?
        """
        await self.execute(query, (
            datetime.utcnow().isoformat(),
            events, markets, decisions, bets, wagered,
            status, error, run_id
        ))
        await self.commit()

    async def get_statistics(self) -> Dict[str, Any]:
        """Get overall database statistics."""
        stats = {}

        # Total decisions
        row = await self.fetchone("SELECT COUNT(*) as count FROM betting_decisions")
        stats['total_decisions'] = row['count'] if row else 0

        # Pending decisions
        row = await self.fetchone("SELECT COUNT(*) as count FROM betting_decisions WHERE status = 'pending'")
        stats['pending_decisions'] = row['count'] if row else 0

        # Settled decisions
        row = await self.fetchone("SELECT COUNT(*) as count FROM betting_decisions WHERE status = 'settled'")
        stats['settled_decisions'] = row['count'] if row else 0

        # Total P&L
        row = await self.fetchone("SELECT SUM(profit_loss) as total FROM betting_decisions WHERE status = 'settled'")
        stats['total_pnl'] = row['total'] if row and row['total'] else 0.0

        # Win rate
        row = await self.fetchone("""
            SELECT
                COUNT(CASE WHEN profit_loss > 0 THEN 1 END) as wins,
                COUNT(*) as total
            FROM betting_decisions
            WHERE status = 'settled' AND action != 'skip'
        """)
        if row and row['total'] > 0:
            stats['win_rate'] = row['wins'] / row['total']
        else:
            stats['win_rate'] = 0.0

        return stats


async def get_database(db_path: str = "trading_bot.db") -> Database:
    """Get or create the singleton database instance."""
    global _database_instance

    async with _lock:
        if _database_instance is None:
            _database_instance = Database(db_path)
            await _database_instance.connect()
        return _database_instance


async def close_database() -> None:
    """Close the singleton database instance."""
    global _database_instance

    async with _lock:
        if _database_instance is not None:
            await _database_instance.close()
            _database_instance = None
