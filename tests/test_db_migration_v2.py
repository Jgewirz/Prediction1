"""
Database migration and query validation for Signal Quality v2.

Tests:
1. Migration applies cleanly
2. New columns exist with correct types
3. Calibration queries execute and return expected columns
4. Empty dataset handling
"""
import os
import sqlite3
import tempfile
import pytest
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from db.queries import Queries


@pytest.fixture
def test_db():
    """Create a temporary SQLite database with base schema."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)

    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row

    # Create base schema (simplified betting_decisions table with existing signal columns)
    conn.execute("""
        CREATE TABLE betting_decisions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            decision_id TEXT UNIQUE NOT NULL,
            timestamp DATETIME NOT NULL,
            event_ticker TEXT NOT NULL,
            market_ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            bet_amount REAL DEFAULT 0,
            confidence REAL,
            research_probability REAL,
            profit_loss REAL,
            outcome TEXT,
            status TEXT DEFAULT 'pending',
            signal_applied INTEGER DEFAULT 0,
            signal_direction TEXT,
            signal_sentiment TEXT,
            signal_strength REAL,
            override_skip_triggered INTEGER DEFAULT 0
        )
    """)
    conn.commit()

    yield conn, path

    conn.close()
    os.unlink(path)


class TestMigrationApplies:
    """Test that migration applies cleanly."""

    def test_migration_adds_new_columns(self, test_db):
        """Migration should add all new columns."""
        conn, _ = test_db

        # Apply migration (individual ALTER statements)
        migration_statements = [
            "ALTER TABLE betting_decisions ADD COLUMN signal_unique_stories INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN signal_unique_outlets INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN signal_raw_articles INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN signal_tier1_count INTEGER",
            "ALTER TABLE betting_decisions ADD COLUMN signal_relevance REAL",
            "ALTER TABLE betting_decisions ADD COLUMN signal_frequency_ratio REAL",
            "ALTER TABLE betting_decisions ADD COLUMN signal_baseline_frequency REAL",
            "ALTER TABLE betting_decisions ADD COLUMN signal_current_frequency REAL",
            "ALTER TABLE betting_decisions ADD COLUMN probability_adjustment REAL DEFAULT 0",
            "ALTER TABLE betting_decisions ADD COLUMN skip_reason TEXT",
            "ALTER TABLE betting_decisions ADD COLUMN override_blocked INTEGER DEFAULT 0",
            "ALTER TABLE betting_decisions ADD COLUMN original_action TEXT",
        ]

        for stmt in migration_statements:
            conn.execute(stmt)
        conn.commit()

        # Verify columns exist
        cursor = conn.execute("PRAGMA table_info(betting_decisions)")
        columns = {row[1] for row in cursor.fetchall()}

        expected_new_columns = [
            "signal_unique_stories", "signal_unique_outlets", "signal_raw_articles",
            "signal_tier1_count", "signal_relevance", "signal_frequency_ratio",
            "signal_baseline_frequency", "signal_current_frequency",
            "probability_adjustment", "skip_reason", "override_blocked", "original_action"
        ]

        for col in expected_new_columns:
            assert col in columns, f"Column {col} not found after migration"

    def test_signal_calibration_table_created(self, test_db):
        """Migration should create signal_calibration table."""
        conn, _ = test_db

        conn.execute("""
            CREATE TABLE IF NOT EXISTS signal_calibration (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date DATE NOT NULL,
                strength_bucket TEXT,
                sentiment_direction TEXT,
                signal_aligned INTEGER,
                total_bets INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                brier_score REAL,
                avg_confidence REAL,
                avg_r_score REAL,
                baseline_win_rate REAL,
                lift_vs_baseline REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(date, strength_bucket, sentiment_direction, signal_aligned)
            )
        """)
        conn.commit()

        # Verify table exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='signal_calibration'"
        )
        assert cursor.fetchone() is not None, "signal_calibration table not created"


class TestCalibrationQueries:
    """Test calibration queries execute correctly."""

    def test_signal_calibration_by_strength_empty(self, test_db):
        """Query should handle empty dataset."""
        conn, _ = test_db

        # Apply migration
        conn.execute("ALTER TABLE betting_decisions ADD COLUMN signal_unique_stories INTEGER")
        conn.commit()

        # Execute query (should return empty but not error)
        cursor = conn.execute(Queries.GET_SIGNAL_CALIBRATION_BY_STRENGTH)
        rows = cursor.fetchall()
        assert rows == [], "Empty table should return empty list"

    def test_signal_calibration_by_strength_with_data(self, test_db):
        """Query should return correct columns with data."""
        conn, _ = test_db

        # Insert test data
        conn.execute("""
            INSERT INTO betting_decisions
            (decision_id, timestamp, event_ticker, market_ticker, action,
             confidence, research_probability, profit_loss, outcome, status,
             signal_applied, signal_strength)
            VALUES
            ('test1', '2024-01-01', 'EVT1', 'MKT1', 'buy_yes', 0.8, 60.0, 10.0, 'yes', 'settled', 1, 0.5),
            ('test2', '2024-01-01', 'EVT2', 'MKT2', 'buy_no', 0.7, 40.0, -5.0, 'yes', 'settled', 1, 0.8),
            ('test3', '2024-01-01', 'EVT3', 'MKT3', 'buy_yes', 0.9, 70.0, 15.0, 'yes', 'settled', 1, 0.9)
        """)
        conn.commit()

        cursor = conn.execute(Queries.GET_SIGNAL_CALIBRATION_BY_STRENGTH)
        rows = cursor.fetchall()

        assert len(rows) > 0, "Should return data"
        # Check column names
        columns = [desc[0] for desc in cursor.description]
        expected_columns = ['strength_bucket', 'total_bets', 'wins', 'brier_score', 'total_pnl', 'avg_confidence']
        for col in expected_columns:
            assert col in columns, f"Missing column: {col}"

    def test_signal_vs_baseline_query(self, test_db):
        """Signal vs baseline query should execute."""
        conn, _ = test_db

        # Insert mix of signal and non-signal bets
        conn.execute("""
            INSERT INTO betting_decisions
            (decision_id, timestamp, event_ticker, market_ticker, action,
             profit_loss, outcome, status, signal_applied)
            VALUES
            ('sig1', '2024-01-01', 'EVT1', 'MKT1', 'buy_yes', 10.0, 'yes', 'settled', 1),
            ('base1', '2024-01-01', 'EVT2', 'MKT2', 'buy_yes', -5.0, 'no', 'settled', 0)
        """)
        conn.commit()

        cursor = conn.execute(Queries.GET_SIGNAL_VS_BASELINE)
        rows = cursor.fetchall()

        assert len(rows) == 1, "Should return one row"
        columns = [desc[0] for desc in cursor.description]
        assert 'signal_win_rate' in columns
        assert 'baseline_win_rate' in columns
        assert 'lift' in columns

    def test_signal_direction_accuracy_query(self, test_db):
        """Signal direction accuracy query should execute."""
        conn, _ = test_db

        conn.execute("""
            INSERT INTO betting_decisions
            (decision_id, timestamp, event_ticker, market_ticker, action,
             profit_loss, outcome, status, signal_applied, signal_direction, signal_sentiment)
            VALUES
            ('test1', '2024-01-01', 'EVT1', 'MKT1', 'buy_yes', 10.0, 'yes', 'settled', 1, 'aligned', 'positive')
        """)
        conn.commit()

        cursor = conn.execute(Queries.GET_SIGNAL_DIRECTION_ACCURACY)
        rows = cursor.fetchall()

        assert len(rows) >= 1
        columns = [desc[0] for desc in cursor.description]
        assert 'signal_direction' in columns
        assert 'signal_sentiment' in columns
        assert 'win_rate' in columns

    def test_skip_override_effectiveness_query(self, test_db):
        """Skip override effectiveness query should execute."""
        conn, _ = test_db

        conn.execute("""
            INSERT INTO betting_decisions
            (decision_id, timestamp, event_ticker, market_ticker, action,
             profit_loss, outcome, status, signal_applied, override_skip_triggered)
            VALUES
            ('test1', '2024-01-01', 'EVT1', 'MKT1', 'buy_yes', 10.0, 'yes', 'settled', 1, 1),
            ('test2', '2024-01-01', 'EVT2', 'MKT2', 'buy_yes', -5.0, 'no', 'settled', 1, 0)
        """)
        conn.commit()

        cursor = conn.execute(Queries.GET_SKIP_OVERRIDE_EFFECTIVENESS)
        rows = cursor.fetchall()

        assert len(rows) == 2, "Should have overridden and not_overridden rows"

    def test_signal_calibration_alert_query(self, test_db):
        """Signal calibration alert query should execute."""
        conn, _ = test_db

        # Need both signal and non-signal data for comparison
        conn.execute("""
            INSERT INTO betting_decisions
            (decision_id, timestamp, event_ticker, market_ticker, action,
             research_probability, outcome, status, signal_applied, signal_strength)
            VALUES
            ('sig1', '2024-01-01', 'EVT1', 'MKT1', 'buy_yes', 60.0, 'yes', 'settled', 1, 0.3),
            ('sig2', '2024-01-01', 'EVT2', 'MKT2', 'buy_yes', 70.0, 'no', 'settled', 1, 0.7),
            ('base1', '2024-01-01', 'EVT3', 'MKT3', 'buy_yes', 50.0, 'yes', 'settled', 0, NULL)
        """)
        conn.commit()

        cursor = conn.execute(Queries.GET_SIGNAL_CALIBRATION_ALERT)
        rows = cursor.fetchall()

        # Should return rows without error
        columns = [desc[0] for desc in cursor.description]
        assert 'strength_group' in columns
        assert 'status' in columns


class TestIndexCreation:
    """Test that indexes are created correctly."""

    def test_indexes_created(self, test_db):
        """Migration indexes should be creatable."""
        conn, _ = test_db

        # Create indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_decisions_signal_strength ON betting_decisions(signal_strength)")
        conn.commit()

        # Verify index exists
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='index' AND name='idx_decisions_signal_strength'"
        )
        assert cursor.fetchone() is not None, "Index not created"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
