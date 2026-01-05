-- SQLite Schema for Kalshi Deep Trading Bot
-- Historical tracking and outcome reconciliation

-- betting_decisions: Main decisions table with 40+ fields matching CSV export
CREATE TABLE IF NOT EXISTS betting_decisions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    decision_id TEXT UNIQUE NOT NULL,  -- UUID for deduplication
    timestamp DATETIME NOT NULL,

    -- Event/Market identifiers
    event_ticker TEXT NOT NULL,
    event_title TEXT,
    market_ticker TEXT NOT NULL,
    market_title TEXT,

    -- Decision fields
    action TEXT NOT NULL,  -- buy_yes, buy_no, skip
    bet_amount REAL NOT NULL DEFAULT 0,
    confidence REAL,
    reasoning TEXT,

    -- Research data (0-100 scale)
    research_probability REAL,
    research_reasoning TEXT,
    research_summary TEXT,
    raw_research TEXT,

    -- Market data (0-100 cents)
    market_yes_price REAL,
    market_no_price REAL,
    market_yes_mid REAL,
    market_no_mid REAL,

    -- Risk metrics (hedge-fund style)
    expected_return REAL,
    r_score REAL,
    kelly_fraction REAL,
    calc_market_prob REAL,  -- Normalized 0-1
    calc_research_prob REAL,  -- Normalized 0-1

    -- Hedging
    is_hedge INTEGER DEFAULT 0,
    hedge_for TEXT,

    -- Extended market data (from Kalshi API)
    market_yes_bid REAL,
    market_yes_ask REAL,
    market_no_bid REAL,
    market_no_ask REAL,
    market_volume REAL,
    market_volume_24h REAL,
    market_liquidity REAL,
    market_open_interest REAL,
    market_status TEXT,
    market_close_time DATETIME,

    -- Outcome tracking (populated by reconciliation)
    outcome TEXT,  -- 'yes', 'no', NULL if pending
    outcome_timestamp DATETIME,
    payout_amount REAL,
    profit_loss REAL,

    -- Status tracking
    status TEXT DEFAULT 'pending',  -- pending, settled, cancelled

    -- Run tracking
    run_mode TEXT,                          -- 'dry_run' or 'live'
    run_id TEXT,                            -- FK to run_history.run_id

    -- TrendRadar signal influence
    signal_applied INTEGER DEFAULT 0,       -- 1 if TrendRadar signal was applied
    signal_direction TEXT,                  -- 'aligned', 'conflicting', 'neutral'
    signal_topic TEXT,                      -- Trending topic matched
    signal_sentiment TEXT,                  -- 'positive', 'negative', 'neutral'
    signal_strength REAL,                   -- 0.0 to 1.0
    signal_source_count INTEGER,            -- Number of news sources
    confidence_boost REAL DEFAULT 0,        -- Raw boost amount (-0.15 to +0.30)
    kelly_multiplier REAL DEFAULT 1.0,      -- Position multiplier (0.8 to 1.25)
    override_skip_triggered INTEGER DEFAULT 0,  -- 1 if skip was overridden
    signal_reasoning TEXT,                  -- Explanation of signal impact

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- market_snapshots: Historical market state for reconciliation
CREATE TABLE IF NOT EXISTS market_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    market_ticker TEXT NOT NULL,
    snapshot_timestamp DATETIME NOT NULL,

    -- Market state
    status TEXT,  -- open, closed, settled
    result TEXT,  -- yes, no (only for settled)
    settlement_value REAL,
    settlement_value_dollars REAL,

    -- Prices at snapshot time
    yes_bid REAL,
    yes_ask REAL,
    no_bid REAL,
    no_ask REAL,
    last_price REAL,

    -- Timing
    open_time DATETIME,
    close_time DATETIME,
    expiration_time DATETIME,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(market_ticker, snapshot_timestamp)
);

-- performance_daily: Aggregated daily P&L
CREATE TABLE IF NOT EXISTS performance_daily (
    date DATE PRIMARY KEY,

    -- Bet counts
    total_bets INTEGER DEFAULT 0,
    winning_bets INTEGER DEFAULT 0,
    losing_bets INTEGER DEFAULT 0,
    pending_bets INTEGER DEFAULT 0,

    -- Amounts
    total_wagered REAL DEFAULT 0,
    total_payout REAL DEFAULT 0,
    net_profit_loss REAL DEFAULT 0,

    -- Performance metrics
    roi_percent REAL,
    win_rate REAL,
    avg_bet_size REAL,
    avg_r_score REAL,

    -- Calibration metrics
    brier_score REAL,
    accuracy REAL,

    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- calibration_records: Prediction vs outcome tracking (migrated from JSON)
CREATE TABLE IF NOT EXISTS calibration_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id TEXT UNIQUE NOT NULL,

    -- Market identifiers
    ticker TEXT NOT NULL,
    event_ticker TEXT,

    -- Prediction data
    predicted_prob REAL,  -- 0-100 scale
    market_price REAL,  -- 0-100 scale
    confidence REAL,  -- 0-1 scale
    r_score REAL,
    action TEXT,  -- buy_yes, buy_no, skip
    reasoning TEXT,

    -- Timing
    timestamp DATETIME NOT NULL,

    -- Outcome (filled when market resolves)
    outcome REAL,  -- 1.0 for YES, 0.0 for NO, NULL if pending
    resolved_timestamp DATETIME,
    actual_payout REAL,

    -- Link to betting_decisions
    decision_id TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,

    FOREIGN KEY (decision_id) REFERENCES betting_decisions(decision_id)
);

-- run_history: Track bot execution runs
CREATE TABLE IF NOT EXISTS run_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT UNIQUE NOT NULL,
    started_at DATETIME NOT NULL,
    completed_at DATETIME,

    -- Configuration snapshot
    mode TEXT,  -- dry_run, live
    environment TEXT,  -- demo, production
    max_events INTEGER,
    z_threshold REAL,
    kelly_fraction REAL,

    -- Results
    events_analyzed INTEGER DEFAULT 0,
    markets_analyzed INTEGER DEFAULT 0,
    decisions_made INTEGER DEFAULT 0,
    bets_placed INTEGER DEFAULT 0,
    total_wagered REAL DEFAULT 0,

    -- Status
    status TEXT DEFAULT 'running',  -- running, completed, failed
    error_message TEXT,

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_decisions_market_ticker ON betting_decisions(market_ticker);
CREATE INDEX IF NOT EXISTS idx_decisions_event_ticker ON betting_decisions(event_ticker);
CREATE INDEX IF NOT EXISTS idx_decisions_status ON betting_decisions(status);
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON betting_decisions(timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_action ON betting_decisions(action);
CREATE INDEX IF NOT EXISTS idx_decisions_run_mode ON betting_decisions(run_mode);
CREATE INDEX IF NOT EXISTS idx_decisions_run_id ON betting_decisions(run_id);
CREATE INDEX IF NOT EXISTS idx_decisions_signal_direction ON betting_decisions(signal_direction);

CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON market_snapshots(market_ticker);
CREATE INDEX IF NOT EXISTS idx_snapshots_status ON market_snapshots(status);

CREATE INDEX IF NOT EXISTS idx_calibration_ticker ON calibration_records(ticker);
CREATE INDEX IF NOT EXISTS idx_calibration_outcome ON calibration_records(outcome);
CREATE INDEX IF NOT EXISTS idx_calibration_timestamp ON calibration_records(timestamp);

CREATE INDEX IF NOT EXISTS idx_run_history_status ON run_history(status);

-- Triggers for updated_at timestamps
CREATE TRIGGER IF NOT EXISTS update_decisions_timestamp
    AFTER UPDATE ON betting_decisions
BEGIN
    UPDATE betting_decisions SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_calibration_timestamp
    AFTER UPDATE ON calibration_records
BEGIN
    UPDATE calibration_records SET updated_at = CURRENT_TIMESTAMP WHERE id = NEW.id;
END;

CREATE TRIGGER IF NOT EXISTS update_performance_timestamp
    AFTER UPDATE ON performance_daily
BEGIN
    UPDATE performance_daily SET updated_at = CURRENT_TIMESTAMP WHERE date = NEW.date;
END;
