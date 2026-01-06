"""
Setup PostgreSQL tables in Neon database for Kalshi Deep Trading Bot.
"""

import psycopg2
from psycopg2 import sql

# Database credentials
DB_CONFIG = {
    "host": "ep-flat-silence-ahtaz4uz-pooler.c-3.us-east-1.aws.neon.tech",
    "database": "neondb",
    "user": "neondb_owner",
    "password": "npg_hjmprEz3wJ0B",
    "sslmode": "require",
}

# PostgreSQL Schema
SCHEMA_SQL = """
-- Drop existing tables if they exist (for clean setup)
DROP TABLE IF EXISTS calibration_records CASCADE;
DROP TABLE IF EXISTS performance_daily CASCADE;
DROP TABLE IF EXISTS market_snapshots CASCADE;
DROP TABLE IF EXISTS betting_decisions CASCADE;
DROP TABLE IF EXISTS run_history CASCADE;
DROP TABLE IF EXISTS events_cache CASCADE;
DROP TABLE IF EXISTS markets_cache CASCADE;

-- betting_decisions: Main decisions table with 40+ fields
CREATE TABLE betting_decisions (
    id SERIAL PRIMARY KEY,
    decision_id VARCHAR(255) UNIQUE NOT NULL,
    timestamp TIMESTAMP NOT NULL,

    -- Event/Market identifiers
    event_ticker VARCHAR(100) NOT NULL,
    event_title TEXT,
    market_ticker VARCHAR(100) NOT NULL,
    market_title TEXT,

    -- Decision fields
    action VARCHAR(20) NOT NULL,  -- buy_yes, buy_no, skip
    bet_amount DECIMAL(12, 2) NOT NULL DEFAULT 0,
    confidence DECIMAL(5, 4),
    reasoning TEXT,

    -- Research data (0-100 scale)
    research_probability DECIMAL(6, 2),
    research_reasoning TEXT,
    research_summary TEXT,
    raw_research TEXT,

    -- Market data (0-100 cents)
    market_yes_price DECIMAL(8, 2),
    market_no_price DECIMAL(8, 2),
    market_yes_mid DECIMAL(8, 2),
    market_no_mid DECIMAL(8, 2),

    -- Risk metrics (hedge-fund style)
    expected_return DECIMAL(10, 6),
    r_score DECIMAL(10, 6),
    kelly_fraction DECIMAL(10, 6),
    calc_market_prob DECIMAL(5, 4),  -- Normalized 0-1
    calc_research_prob DECIMAL(5, 4),  -- Normalized 0-1

    -- Hedging
    is_hedge BOOLEAN DEFAULT FALSE,
    hedge_for VARCHAR(100),

    -- Extended market data (from Kalshi API)
    market_yes_bid DECIMAL(8, 2),
    market_yes_ask DECIMAL(8, 2),
    market_no_bid DECIMAL(8, 2),
    market_no_ask DECIMAL(8, 2),
    market_volume DECIMAL(15, 2),
    market_volume_24h DECIMAL(15, 2),
    market_liquidity DECIMAL(15, 2),
    market_open_interest DECIMAL(15, 2),
    market_status VARCHAR(20),
    market_close_time TIMESTAMP,

    -- Outcome tracking (populated by reconciliation)
    outcome VARCHAR(10),  -- 'yes', 'no', NULL if pending
    outcome_timestamp TIMESTAMP,
    payout_amount DECIMAL(12, 2),
    profit_loss DECIMAL(12, 2),

    -- Status tracking
    status VARCHAR(20) DEFAULT 'pending',  -- pending, settled, cancelled, skipped

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- market_snapshots: Historical market state for reconciliation
CREATE TABLE market_snapshots (
    id SERIAL PRIMARY KEY,
    market_ticker VARCHAR(100) NOT NULL,
    snapshot_timestamp TIMESTAMP NOT NULL,

    -- Market state
    status VARCHAR(20),  -- open, closed, settled
    result VARCHAR(10),  -- yes, no (only for settled)
    settlement_value DECIMAL(10, 4),
    settlement_value_dollars DECIMAL(12, 2),

    -- Prices at snapshot time
    yes_bid DECIMAL(8, 2),
    yes_ask DECIMAL(8, 2),
    no_bid DECIMAL(8, 2),
    no_ask DECIMAL(8, 2),
    last_price DECIMAL(8, 2),

    -- Timing
    open_time TIMESTAMP,
    close_time TIMESTAMP,
    expiration_time TIMESTAMP,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(market_ticker, snapshot_timestamp)
);

-- performance_daily: Aggregated daily P&L
CREATE TABLE performance_daily (
    date DATE PRIMARY KEY,

    -- Bet counts
    total_bets INTEGER DEFAULT 0,
    winning_bets INTEGER DEFAULT 0,
    losing_bets INTEGER DEFAULT 0,
    pending_bets INTEGER DEFAULT 0,

    -- Amounts
    total_wagered DECIMAL(15, 2) DEFAULT 0,
    total_payout DECIMAL(15, 2) DEFAULT 0,
    net_profit_loss DECIMAL(15, 2) DEFAULT 0,

    -- Performance metrics
    roi_percent DECIMAL(10, 4),
    win_rate DECIMAL(5, 4),
    avg_bet_size DECIMAL(12, 2),
    avg_r_score DECIMAL(10, 6),

    -- Calibration metrics
    brier_score DECIMAL(10, 6),
    accuracy DECIMAL(5, 4),

    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- calibration_records: Prediction vs outcome tracking
CREATE TABLE calibration_records (
    id SERIAL PRIMARY KEY,
    prediction_id VARCHAR(255) UNIQUE NOT NULL,

    -- Market identifiers
    ticker VARCHAR(100) NOT NULL,
    event_ticker VARCHAR(100),

    -- Prediction data
    predicted_prob DECIMAL(6, 2),  -- 0-100 scale
    market_price DECIMAL(6, 2),  -- 0-100 scale
    confidence DECIMAL(5, 4),  -- 0-1 scale
    r_score DECIMAL(10, 6),
    action VARCHAR(20),  -- buy_yes, buy_no, skip
    reasoning TEXT,

    -- Timing
    timestamp TIMESTAMP NOT NULL,

    -- Outcome (filled when market resolves)
    outcome DECIMAL(3, 1),  -- 1.0 for YES, 0.0 for NO, NULL if pending
    resolved_timestamp TIMESTAMP,
    actual_payout DECIMAL(12, 2),

    -- Link to betting_decisions
    decision_id VARCHAR(255),

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (decision_id) REFERENCES betting_decisions(decision_id) ON DELETE SET NULL
);

-- run_history: Track bot execution runs
CREATE TABLE run_history (
    id SERIAL PRIMARY KEY,
    run_id VARCHAR(100) UNIQUE NOT NULL,
    started_at TIMESTAMP NOT NULL,
    completed_at TIMESTAMP,

    -- Configuration snapshot
    mode VARCHAR(20),  -- dry_run, live
    environment VARCHAR(20),  -- demo, production
    max_events INTEGER,
    z_threshold DECIMAL(5, 2),
    kelly_fraction DECIMAL(5, 2),

    -- Results
    events_analyzed INTEGER DEFAULT 0,
    markets_analyzed INTEGER DEFAULT 0,
    decisions_made INTEGER DEFAULT 0,
    bets_placed INTEGER DEFAULT 0,
    total_wagered DECIMAL(15, 2) DEFAULT 0,

    -- Status
    status VARCHAR(20) DEFAULT 'running',  -- running, completed, failed
    error_message TEXT,

    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- events_cache: Cache of Kalshi events
CREATE TABLE events_cache (
    id SERIAL PRIMARY KEY,
    event_ticker VARCHAR(100) UNIQUE NOT NULL,
    title TEXT,
    subtitle TEXT,
    category VARCHAR(50),
    status VARCHAR(20),
    mutually_exclusive BOOLEAN,

    -- Volume metrics
    volume BIGINT,
    volume_24h BIGINT,
    liquidity BIGINT,
    open_interest BIGINT,

    -- Timing
    strike_date TIMESTAMP,
    strike_period VARCHAR(50),

    -- Cache metadata
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,

    total_markets INTEGER DEFAULT 0
);

-- markets_cache: Cache of Kalshi markets
CREATE TABLE markets_cache (
    id SERIAL PRIMARY KEY,
    market_ticker VARCHAR(100) UNIQUE NOT NULL,
    event_ticker VARCHAR(100) NOT NULL,
    title TEXT,
    subtitle TEXT,

    -- Prices
    yes_bid DECIMAL(8, 2),
    yes_ask DECIMAL(8, 2),
    no_bid DECIMAL(8, 2),
    no_ask DECIMAL(8, 2),
    last_price DECIMAL(8, 2),

    -- Volume
    volume BIGINT,
    volume_24h BIGINT,

    -- Status
    status VARCHAR(20),
    result VARCHAR(10),

    -- Timing
    open_time TIMESTAMP,
    close_time TIMESTAMP,
    expiration_time TIMESTAMP,

    -- Cache metadata
    cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,

    FOREIGN KEY (event_ticker) REFERENCES events_cache(event_ticker) ON DELETE CASCADE
);

-- Indexes for efficient queries
CREATE INDEX idx_decisions_market_ticker ON betting_decisions(market_ticker);
CREATE INDEX idx_decisions_event_ticker ON betting_decisions(event_ticker);
CREATE INDEX idx_decisions_status ON betting_decisions(status);
CREATE INDEX idx_decisions_timestamp ON betting_decisions(timestamp);
CREATE INDEX idx_decisions_action ON betting_decisions(action);
CREATE INDEX idx_decisions_outcome ON betting_decisions(outcome);

CREATE INDEX idx_snapshots_ticker ON market_snapshots(market_ticker);
CREATE INDEX idx_snapshots_status ON market_snapshots(status);
CREATE INDEX idx_snapshots_timestamp ON market_snapshots(snapshot_timestamp);

CREATE INDEX idx_calibration_ticker ON calibration_records(ticker);
CREATE INDEX idx_calibration_outcome ON calibration_records(outcome);
CREATE INDEX idx_calibration_timestamp ON calibration_records(timestamp);

CREATE INDEX idx_run_history_status ON run_history(status);
CREATE INDEX idx_run_history_started ON run_history(started_at);

CREATE INDEX idx_events_cache_ticker ON events_cache(event_ticker);
CREATE INDEX idx_events_cache_category ON events_cache(category);
CREATE INDEX idx_events_cache_expires ON events_cache(expires_at);

CREATE INDEX idx_markets_cache_ticker ON markets_cache(market_ticker);
CREATE INDEX idx_markets_cache_event ON markets_cache(event_ticker);
CREATE INDEX idx_markets_cache_expires ON markets_cache(expires_at);

-- Create function to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Triggers for updated_at timestamps
CREATE TRIGGER update_decisions_timestamp
    BEFORE UPDATE ON betting_decisions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_calibration_timestamp
    BEFORE UPDATE ON calibration_records
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_performance_timestamp
    BEFORE UPDATE ON performance_daily
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries

-- View: Daily P&L summary
CREATE OR REPLACE VIEW v_daily_pnl AS
SELECT
    DATE(timestamp) as date,
    SUM(profit_loss) as daily_pnl,
    COUNT(*) as total_bets,
    SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
    SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
    SUM(bet_amount) as total_wagered,
    CASE WHEN SUM(bet_amount) > 0
         THEN SUM(profit_loss) / SUM(bet_amount) * 100
         ELSE 0 END as roi_percent
FROM betting_decisions
WHERE status = 'settled' AND action != 'skip'
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- View: Cumulative P&L
CREATE OR REPLACE VIEW v_cumulative_pnl AS
SELECT
    DATE(timestamp) as date,
    SUM(profit_loss) as daily_pnl,
    SUM(SUM(profit_loss)) OVER (ORDER BY DATE(timestamp)) as cumulative_pnl
FROM betting_decisions
WHERE status = 'settled' AND action != 'skip'
GROUP BY DATE(timestamp)
ORDER BY date;

-- View: R-score effectiveness
CREATE OR REPLACE VIEW v_rscore_effectiveness AS
SELECT
    CASE
        WHEN r_score < 1.0 THEN 'low (<1.0)'
        WHEN r_score < 1.5 THEN 'medium (1.0-1.5)'
        WHEN r_score < 2.0 THEN 'good (1.5-2.0)'
        ELSE 'high (>2.0)'
    END as r_score_bucket,
    COUNT(*) as count,
    AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
    SUM(profit_loss) as total_pnl,
    AVG(profit_loss) as avg_pnl
FROM betting_decisions
WHERE status = 'settled' AND action != 'skip' AND r_score IS NOT NULL
GROUP BY r_score_bucket
ORDER BY r_score_bucket;

-- View: Pending decisions
CREATE OR REPLACE VIEW v_pending_decisions AS
SELECT
    decision_id,
    market_ticker,
    event_ticker,
    action,
    bet_amount,
    confidence,
    r_score,
    timestamp,
    market_close_time
FROM betting_decisions
WHERE status = 'pending' AND action != 'skip'
ORDER BY timestamp DESC;

-- View: Overall statistics
CREATE OR REPLACE VIEW v_overall_stats AS
SELECT
    COUNT(*) as total_decisions,
    SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END) as pending_decisions,
    SUM(CASE WHEN status = 'settled' THEN 1 ELSE 0 END) as settled_decisions,
    SUM(profit_loss) as total_pnl,
    SUM(bet_amount) as total_wagered,
    AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
    AVG(r_score) as avg_r_score,
    AVG(confidence) as avg_confidence
FROM betting_decisions
WHERE action != 'skip';
"""


def setup_database():
    """Create all tables in the PostgreSQL database."""
    print("Connecting to Neon PostgreSQL database...")

    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = True
        cursor = conn.cursor()

        print("Connected successfully!")
        print("Creating tables and indexes...")

        # Execute the schema SQL
        cursor.execute(SCHEMA_SQL)

        print("\nDatabase setup complete!")
        print("\nTables created:")

        # List created tables
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema = 'public' AND table_type = 'BASE TABLE'
            ORDER BY table_name
        """
        )
        tables = cursor.fetchall()
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table[0]}")
            count = cursor.fetchone()[0]
            print(f"  - {table[0]}: {count} rows")

        print("\nViews created:")
        cursor.execute(
            """
            SELECT table_name
            FROM information_schema.views
            WHERE table_schema = 'public'
            ORDER BY table_name
        """
        )
        views = cursor.fetchall()
        for view in views:
            print(f"  - {view[0]}")

        cursor.close()
        conn.close()

        print("\nDatabase setup successful!")

    except Exception as e:
        print(f"Error setting up database: {e}")
        raise


if __name__ == "__main__":
    setup_database()
