-- Position Tracking Migration
-- Enables stop-loss/take-profit functionality
-- Run: uv run python setup_postgres.py --migrate

-- ============================================================================
-- PART 1: Add entry price tracking to betting_decisions
-- ============================================================================

-- Order execution tracking
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS order_id TEXT;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS filled_price_cents INTEGER;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS filled_contracts INTEGER;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS filled_timestamp TIMESTAMPTZ;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS fill_cost_dollars REAL;

-- Stop-loss/take-profit configuration per position
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS stop_loss_pct REAL DEFAULT 0.15;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS take_profit_pct REAL DEFAULT 0.30;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS trailing_stop_pct REAL;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS sl_tp_enabled BOOLEAN DEFAULT TRUE;

-- Exit tracking
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_order_id TEXT;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_price_cents INTEGER;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_contracts INTEGER;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_timestamp TIMESTAMPTZ;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_reason TEXT;  -- 'stop_loss', 'take_profit', 'trailing_stop', 'manual', 'expiry'
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS exit_pnl_dollars REAL;

-- Real-time tracking fields
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS current_price_cents INTEGER;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS unrealized_pnl_dollars REAL;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS unrealized_pnl_pct REAL;
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS high_water_mark_cents INTEGER;  -- For trailing stops
ALTER TABLE betting_decisions ADD COLUMN IF NOT EXISTS last_price_update TIMESTAMPTZ;

-- ============================================================================
-- PART 2: Position snapshots table for historical tracking
-- ============================================================================

CREATE TABLE IF NOT EXISTS position_snapshots (
    id SERIAL PRIMARY KEY,
    decision_id TEXT REFERENCES betting_decisions(decision_id) ON DELETE CASCADE,
    market_ticker TEXT NOT NULL,
    snapshot_timestamp TIMESTAMPTZ DEFAULT NOW(),

    -- Position state
    contracts INTEGER NOT NULL,
    side TEXT NOT NULL,  -- 'yes' or 'no'
    entry_price_cents INTEGER NOT NULL,
    current_price_cents INTEGER NOT NULL,

    -- P&L metrics
    unrealized_pnl_dollars REAL,
    unrealized_pnl_pct REAL,

    -- Market data
    bid_cents INTEGER,
    ask_cents INTEGER,
    spread_cents INTEGER,
    volume_24h INTEGER,

    -- Trigger proximity
    stop_loss_distance_pct REAL,
    take_profit_distance_pct REAL,

    -- Constraints
    UNIQUE(decision_id, snapshot_timestamp)
);

-- ============================================================================
-- PART 3: Exit events log
-- ============================================================================

CREATE TABLE IF NOT EXISTS exit_events (
    id SERIAL PRIMARY KEY,
    decision_id TEXT REFERENCES betting_decisions(decision_id) ON DELETE CASCADE,
    market_ticker TEXT NOT NULL,

    -- Trigger info
    trigger_type TEXT NOT NULL,  -- 'stop_loss', 'take_profit', 'trailing_stop', 'manual'
    trigger_timestamp TIMESTAMPTZ DEFAULT NOW(),
    trigger_price_cents INTEGER NOT NULL,

    -- Position info at trigger
    entry_price_cents INTEGER NOT NULL,
    contracts INTEGER NOT NULL,
    side TEXT NOT NULL,

    -- P&L at trigger
    unrealized_pnl_dollars REAL,
    unrealized_pnl_pct REAL,

    -- Execution result
    order_id TEXT,
    filled_price_cents INTEGER,
    filled_contracts INTEGER,
    execution_status TEXT,  -- 'pending', 'filled', 'partial', 'failed', 'cancelled'
    execution_timestamp TIMESTAMPTZ,

    -- Final P&L
    realized_pnl_dollars REAL,
    slippage_cents INTEGER,

    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- ============================================================================
-- PART 4: Indexes for efficient queries
-- ============================================================================

-- betting_decisions indexes
CREATE INDEX IF NOT EXISTS idx_decisions_order_id ON betting_decisions(order_id);
CREATE INDEX IF NOT EXISTS idx_decisions_sl_tp_enabled ON betting_decisions(sl_tp_enabled);
CREATE INDEX IF NOT EXISTS idx_decisions_exit_reason ON betting_decisions(exit_reason);
CREATE INDEX IF NOT EXISTS idx_decisions_filled_timestamp ON betting_decisions(filled_timestamp);
CREATE INDEX IF NOT EXISTS idx_decisions_exit_timestamp ON betting_decisions(exit_timestamp);

-- Position snapshots indexes
CREATE INDEX IF NOT EXISTS idx_snapshots_decision_id ON position_snapshots(decision_id);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON position_snapshots(snapshot_timestamp);
CREATE INDEX IF NOT EXISTS idx_snapshots_ticker ON position_snapshots(market_ticker);

-- Exit events indexes
CREATE INDEX IF NOT EXISTS idx_exit_events_decision_id ON exit_events(decision_id);
CREATE INDEX IF NOT EXISTS idx_exit_events_trigger_type ON exit_events(trigger_type);
CREATE INDEX IF NOT EXISTS idx_exit_events_timestamp ON exit_events(trigger_timestamp);

-- ============================================================================
-- PART 5: Views for monitoring dashboard
-- ============================================================================

-- Active positions view (positions without exits)
CREATE OR REPLACE VIEW v_active_positions AS
SELECT
    bd.decision_id,
    bd.market_ticker,
    bd.action,
    CASE WHEN bd.action = 'buy_yes' THEN 'yes' ELSE 'no' END as side,
    bd.filled_price_cents as entry_price,
    bd.filled_contracts as contracts,
    bd.filled_timestamp as entry_time,
    bd.current_price_cents,
    bd.unrealized_pnl_dollars,
    bd.unrealized_pnl_pct,
    bd.stop_loss_pct,
    bd.take_profit_pct,
    bd.trailing_stop_pct,
    bd.high_water_mark_cents,
    bd.last_price_update,
    -- Trigger distances
    CASE
        WHEN bd.filled_price_cents > 0 AND bd.current_price_cents > 0 THEN
            (bd.current_price_cents::REAL - bd.filled_price_cents::REAL) / bd.filled_price_cents::REAL + bd.stop_loss_pct
    END as stop_loss_distance,
    CASE
        WHEN bd.filled_price_cents > 0 AND bd.current_price_cents > 0 THEN
            bd.take_profit_pct - (bd.current_price_cents::REAL - bd.filled_price_cents::REAL) / bd.filled_price_cents::REAL
    END as take_profit_distance
FROM betting_decisions bd
WHERE bd.status = 'pending'
  AND bd.action IN ('buy_yes', 'buy_no')
  AND bd.filled_contracts > 0
  AND bd.sl_tp_enabled = TRUE
  AND bd.exit_order_id IS NULL;

-- Exit summary view
CREATE OR REPLACE VIEW v_exit_summary AS
SELECT
    exit_reason,
    COUNT(*) as count,
    SUM(exit_pnl_dollars) as total_pnl,
    AVG(exit_pnl_dollars) as avg_pnl,
    MIN(exit_pnl_dollars) as min_pnl,
    MAX(exit_pnl_dollars) as max_pnl
FROM betting_decisions
WHERE exit_reason IS NOT NULL
GROUP BY exit_reason;

-- Daily P&L from exits
CREATE OR REPLACE VIEW v_daily_exit_pnl AS
SELECT
    DATE(exit_timestamp) as exit_date,
    exit_reason,
    COUNT(*) as exits,
    SUM(exit_pnl_dollars) as pnl
FROM betting_decisions
WHERE exit_timestamp IS NOT NULL
GROUP BY DATE(exit_timestamp), exit_reason
ORDER BY exit_date DESC;

-- ============================================================================
-- PART 6: Helper functions
-- ============================================================================

-- Function to calculate P&L for a position
CREATE OR REPLACE FUNCTION calculate_position_pnl(
    entry_price INTEGER,
    current_price INTEGER,
    contracts INTEGER
) RETURNS TABLE(pnl_dollars REAL, pnl_pct REAL) AS $$
BEGIN
    RETURN QUERY SELECT
        (contracts * (current_price - entry_price)::REAL / 100.0) as pnl_dollars,
        CASE
            WHEN entry_price > 0 THEN (current_price::REAL - entry_price::REAL) / entry_price::REAL
            ELSE 0.0
        END as pnl_pct;
END;
$$ LANGUAGE plpgsql;

-- Function to check if stop-loss should trigger
CREATE OR REPLACE FUNCTION should_trigger_stop_loss(
    entry_price INTEGER,
    current_price INTEGER,
    stop_loss_pct REAL
) RETURNS BOOLEAN AS $$
BEGIN
    IF entry_price <= 0 THEN
        RETURN FALSE;
    END IF;

    RETURN (current_price::REAL - entry_price::REAL) / entry_price::REAL <= -stop_loss_pct;
END;
$$ LANGUAGE plpgsql;

-- Function to check if take-profit should trigger
CREATE OR REPLACE FUNCTION should_trigger_take_profit(
    entry_price INTEGER,
    current_price INTEGER,
    take_profit_pct REAL
) RETURNS BOOLEAN AS $$
BEGIN
    IF entry_price <= 0 THEN
        RETURN FALSE;
    END IF;

    RETURN (current_price::REAL - entry_price::REAL) / entry_price::REAL >= take_profit_pct;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- Migration complete
-- ============================================================================
