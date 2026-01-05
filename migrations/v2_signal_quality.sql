-- Migration: Signal Quality Enhancement (v2)
-- Adds columns for de-duplication, relevance, baseline tracking, skip reasons, and calibration

-- Phase 1: De-duplication metrics
ALTER TABLE betting_decisions ADD COLUMN signal_unique_stories INTEGER;
ALTER TABLE betting_decisions ADD COLUMN signal_unique_outlets INTEGER;
ALTER TABLE betting_decisions ADD COLUMN signal_raw_articles INTEGER;
ALTER TABLE betting_decisions ADD COLUMN signal_tier1_count INTEGER;

-- Phase 2: Relevance scoring
ALTER TABLE betting_decisions ADD COLUMN signal_relevance REAL;

-- Phase 3: Baseline frequency tracking
ALTER TABLE betting_decisions ADD COLUMN signal_frequency_ratio REAL;
ALTER TABLE betting_decisions ADD COLUMN signal_baseline_frequency REAL;
ALTER TABLE betting_decisions ADD COLUMN signal_current_frequency REAL;

-- Phase 6: Single-lever probability adjustment
ALTER TABLE betting_decisions ADD COLUMN probability_adjustment REAL DEFAULT 0;

-- Phase 7: Skip reason tracking
ALTER TABLE betting_decisions ADD COLUMN skip_reason TEXT;
ALTER TABLE betting_decisions ADD COLUMN override_blocked INTEGER DEFAULT 0;
ALTER TABLE betting_decisions ADD COLUMN original_action TEXT;

-- Phase 8: Signal calibration table
CREATE TABLE IF NOT EXISTS signal_calibration (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    date DATE NOT NULL,

    -- Bucketing dimensions
    strength_bucket TEXT,           -- '0-0.3', '0.3-0.6', '0.6-0.8', '0.8-1.0'
    sentiment_direction TEXT,       -- 'positive', 'negative', 'neutral'
    signal_aligned INTEGER,         -- 1 if signal was aligned with action

    -- Outcome metrics
    total_bets INTEGER DEFAULT 0,
    wins INTEGER DEFAULT 0,
    total_pnl REAL DEFAULT 0,

    -- Calibration metrics
    brier_score REAL,
    avg_confidence REAL,
    avg_r_score REAL,

    -- Comparison to baseline
    baseline_win_rate REAL,         -- Win rate of non-signal bets same period
    lift_vs_baseline REAL,          -- (signal_win_rate - baseline) / baseline

    created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(date, strength_bucket, sentiment_direction, signal_aligned)
);

-- Index for efficient calibration queries
CREATE INDEX IF NOT EXISTS idx_decisions_signal_strength ON betting_decisions(signal_strength);
CREATE INDEX IF NOT EXISTS idx_decisions_skip_reason ON betting_decisions(skip_reason);
CREATE INDEX IF NOT EXISTS idx_signal_calibration_date ON signal_calibration(date);
