"""
SQL query constants for Kalshi Deep Trading Bot database.
"""


class Queries:
    """Collection of SQL queries for common operations."""

    # ========== Decision Queries ==========

    GET_PENDING_DECISIONS = """
        SELECT * FROM betting_decisions
        WHERE status = 'pending' AND action != 'skip'
        ORDER BY timestamp DESC
    """

    GET_PENDING_BY_MARKET = """
        SELECT * FROM betting_decisions
        WHERE market_ticker = ? AND status = 'pending'
        ORDER BY timestamp DESC
    """

    GET_DECISIONS_BY_DATE_RANGE = """
        SELECT * FROM betting_decisions
        WHERE timestamp BETWEEN ? AND ?
        ORDER BY timestamp DESC
    """

    GET_DECISIONS_BY_EVENT = """
        SELECT * FROM betting_decisions
        WHERE event_ticker = ?
        ORDER BY timestamp DESC
    """

    # ========== P&L Queries ==========

    GET_TODAY_PNL = """
        SELECT COALESCE(SUM(profit_loss), 0) as daily_pnl
        FROM betting_decisions
        WHERE DATE(timestamp) = CURRENT_DATE
          AND status = 'settled'
          AND action != 'skip'
    """

    GET_DAILY_PNL = """
        SELECT
            DATE(timestamp) as date,
            SUM(profit_loss) as daily_pnl,
            COUNT(*) as total_bets,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(bet_amount) as total_wagered
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip'
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    """

    GET_CUMULATIVE_PNL = """
        SELECT
            DATE(timestamp) as date,
            SUM(profit_loss) as daily_pnl,
            SUM(SUM(profit_loss)) OVER (ORDER BY DATE(timestamp)) as cumulative_pnl
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip'
        GROUP BY DATE(timestamp)
        ORDER BY date
    """

    GET_TOTAL_PNL = """
        SELECT
            SUM(profit_loss) as total_pnl,
            SUM(bet_amount) as total_wagered,
            COUNT(*) as total_bets,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as winning_bets,
            SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losing_bets
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip'
    """

    # ========== Risk Metrics Queries ==========

    GET_R_SCORE_EFFECTIVENESS = """
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
        ORDER BY r_score_bucket
    """

    GET_CONFIDENCE_EFFECTIVENESS = """
        SELECT
            CASE
                WHEN confidence < 0.6 THEN 'low (<0.6)'
                WHEN confidence < 0.75 THEN 'medium (0.6-0.75)'
                WHEN confidence < 0.85 THEN 'good (0.75-0.85)'
                ELSE 'high (>0.85)'
            END as confidence_bucket,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip' AND confidence IS NOT NULL
        GROUP BY confidence_bucket
        ORDER BY confidence_bucket
    """

    GET_KELLY_EFFECTIVENESS = """
        SELECT
            CASE
                WHEN kelly_fraction < 0.1 THEN 'conservative (<10%)'
                WHEN kelly_fraction < 0.25 THEN 'moderate (10-25%)'
                WHEN kelly_fraction < 0.5 THEN 'aggressive (25-50%)'
                ELSE 'very aggressive (>50%)'
            END as kelly_bucket,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl,
            AVG(bet_amount) as avg_bet_size
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip' AND kelly_fraction IS NOT NULL
        GROUP BY kelly_bucket
    """

    # ========== Calibration Queries ==========

    GET_CALIBRATION_METRICS = """
        SELECT
            COUNT(*) as total_predictions,
            AVG((predicted_prob/100.0 - outcome) * (predicted_prob/100.0 - outcome)) as brier_score,
            AVG(CASE
                WHEN (predicted_prob > 50 AND outcome = 1) OR (predicted_prob < 50 AND outcome = 0)
                THEN 1.0 ELSE 0.0
            END) as directional_accuracy,
            AVG(predicted_prob/100.0) as avg_predicted,
            AVG(outcome) as avg_outcome
        FROM calibration_records
        WHERE outcome IS NOT NULL
    """

    GET_CALIBRATION_BY_BUCKET = """
        SELECT
            CASE
                WHEN predicted_prob < 20 THEN '0-20%'
                WHEN predicted_prob < 40 THEN '20-40%'
                WHEN predicted_prob < 60 THEN '40-60%'
                WHEN predicted_prob < 80 THEN '60-80%'
                ELSE '80-100%'
            END as probability_bucket,
            COUNT(*) as count,
            AVG(predicted_prob) as avg_predicted,
            AVG(outcome * 100) as avg_actual,
            ABS(AVG(predicted_prob) - AVG(outcome * 100)) as calibration_error
        FROM calibration_records
        WHERE outcome IS NOT NULL
        GROUP BY probability_bucket
        ORDER BY probability_bucket
    """

    GET_UNRESOLVED_CALIBRATION = """
        SELECT * FROM calibration_records
        WHERE outcome IS NULL
        ORDER BY timestamp DESC
    """

    # ========== Action Analysis Queries ==========

    GET_ACTION_PERFORMANCE = """
        SELECT
            action,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl,
            SUM(bet_amount) as total_wagered
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip'
        GROUP BY action
    """

    GET_HEDGE_PERFORMANCE = """
        SELECT
            is_hedge,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip'
        GROUP BY is_hedge
    """

    # ========== Market Snapshot Queries ==========

    GET_LATEST_SNAPSHOT = """
        SELECT * FROM market_snapshots
        WHERE market_ticker = ?
        ORDER BY snapshot_timestamp DESC
        LIMIT 1
    """

    GET_SETTLED_SNAPSHOTS = """
        SELECT DISTINCT market_ticker, result, settlement_value
        FROM market_snapshots
        WHERE status = 'settled'
        AND market_ticker IN ({})
    """

    # ========== Run History Queries ==========

    GET_RECENT_RUNS = """
        SELECT * FROM run_history
        ORDER BY started_at DESC
        LIMIT ?
    """

    GET_RUN_STATISTICS = """
        SELECT
            COUNT(*) as total_runs,
            SUM(events_analyzed) as total_events,
            SUM(decisions_made) as total_decisions,
            SUM(bets_placed) as total_bets,
            SUM(total_wagered) as total_wagered,
            AVG(JULIANDAY(completed_at) - JULIANDAY(started_at)) * 24 * 60 as avg_duration_minutes
        FROM run_history
        WHERE status = 'completed'
    """

    # ========== Signal Calibration Queries (Phase 8) ==========

    GET_SIGNAL_CALIBRATION_BY_STRENGTH = """
        SELECT
            CASE
                WHEN signal_strength < 0.3 THEN '0-0.3'
                WHEN signal_strength < 0.6 THEN '0.3-0.6'
                WHEN signal_strength < 0.8 THEN '0.6-0.8'
                ELSE '0.8-1.0'
            END as strength_bucket,
            COUNT(*) as total_bets,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            AVG(CASE WHEN outcome IS NOT NULL THEN
                (research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) *
                (research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END)
            END) as brier_score,
            SUM(profit_loss) as total_pnl,
            AVG(confidence) as avg_confidence
        FROM betting_decisions
        WHERE signal_applied = 1 AND outcome IS NOT NULL AND action != 'skip'
        GROUP BY strength_bucket
        ORDER BY strength_bucket
    """

    GET_SIGNAL_VS_BASELINE = """
        WITH signal_bets AS (
            SELECT
                AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                SUM(profit_loss) as pnl,
                COUNT(*) as count
            FROM betting_decisions
            WHERE signal_applied = 1 AND outcome IS NOT NULL AND action != 'skip'
        ),
        baseline_bets AS (
            SELECT
                AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
                SUM(profit_loss) as pnl,
                COUNT(*) as count
            FROM betting_decisions
            WHERE signal_applied = 0 AND outcome IS NOT NULL AND action != 'skip'
        )
        SELECT
            s.win_rate as signal_win_rate,
            s.pnl as signal_pnl,
            s.count as signal_count,
            b.win_rate as baseline_win_rate,
            b.pnl as baseline_pnl,
            b.count as baseline_count,
            CASE WHEN b.win_rate > 0 THEN (s.win_rate - b.win_rate) / b.win_rate ELSE 0 END as lift
        FROM signal_bets s, baseline_bets b
    """

    GET_SIGNAL_DIRECTION_ACCURACY = """
        SELECT
            signal_direction,
            signal_sentiment,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl
        FROM betting_decisions
        WHERE signal_applied = 1 AND outcome IS NOT NULL AND action != 'skip'
        GROUP BY signal_direction, signal_sentiment
        ORDER BY signal_direction, signal_sentiment
    """

    GET_SKIP_OVERRIDE_EFFECTIVENESS = """
        SELECT
            CASE WHEN override_skip_triggered = 1 THEN 'overridden' ELSE 'not_overridden' END as override_status,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl
        FROM betting_decisions
        WHERE signal_applied = 1 AND outcome IS NOT NULL AND action != 'skip'
        GROUP BY override_status
    """

    GET_SIGNAL_CALIBRATION_ALERT = """
        -- Returns signal strength buckets where Brier score is worse than baseline
        WITH signal_brier AS (
            SELECT
                CASE WHEN signal_strength < 0.5 THEN 'low' ELSE 'high' END as strength_group,
                AVG((research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) *
                    (research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END)) as brier
            FROM betting_decisions
            WHERE signal_applied = 1 AND outcome IS NOT NULL AND action != 'skip'
            GROUP BY strength_group
        ),
        baseline_brier AS (
            SELECT
                AVG((research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) *
                    (research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END)) as brier
            FROM betting_decisions
            WHERE signal_applied = 0 AND outcome IS NOT NULL AND action != 'skip'
        )
        SELECT
            s.strength_group,
            s.brier as signal_brier,
            b.brier as baseline_brier,
            CASE WHEN s.brier > b.brier * 1.1 THEN 'ALERT: Signal hurting predictions' ELSE 'OK' END as status
        FROM signal_brier s, baseline_brier b
    """

    # ========== Utility Queries ==========

    GET_UNIQUE_MARKETS = """
        SELECT DISTINCT market_ticker
        FROM betting_decisions
        WHERE status = 'pending'
    """

    # Get markets analyzed in the last N hours to avoid duplicate analysis
    GET_RECENTLY_ANALYZED_MARKETS = """
        SELECT DISTINCT market_ticker
        FROM betting_decisions
        WHERE timestamp > NOW() - INTERVAL '{hours} hours'
    """

    # SQLite version (for backwards compatibility)
    GET_RECENTLY_ANALYZED_MARKETS_SQLITE = """
        SELECT DISTINCT market_ticker
        FROM betting_decisions
        WHERE timestamp > datetime('now', '-{hours} hours')
    """

    GET_DECISION_COUNT_BY_STATUS = """
        SELECT status, COUNT(*) as count
        FROM betting_decisions
        GROUP BY status
    """

    CHECK_DECISION_EXISTS = """
        SELECT 1 FROM betting_decisions
        WHERE decision_id = ?
        LIMIT 1
    """

    # ========== ROI Calibration System Queries (Phase 1) ==========

    GET_CALIBRATION_DATA_FOR_CURVE = """
        SELECT
            research_probability as predicted_prob,
            CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END as outcome,
            confidence,
            r_score,
            action
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND outcome IS NOT NULL
          AND research_probability IS NOT NULL
        ORDER BY timestamp ASC
    """

    GET_PROBABILITY_BUCKET_ACCURACY = """
        SELECT
            CASE
                WHEN research_probability < 20 THEN '0-20%'
                WHEN research_probability < 40 THEN '20-40%'
                WHEN research_probability < 60 THEN '40-60%'
                WHEN research_probability < 80 THEN '60-80%'
                ELSE '80-100%'
            END as prob_bucket,
            COUNT(*) as count,
            AVG(research_probability) as avg_predicted,
            AVG(CASE WHEN outcome = 'yes' THEN 100.0 ELSE 0.0 END) as avg_actual,
            ABS(AVG(research_probability) - AVG(CASE WHEN outcome = 'yes' THEN 100.0 ELSE 0.0 END)) as calibration_error
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND outcome IS NOT NULL
          AND research_probability IS NOT NULL
        GROUP BY prob_bucket
        ORDER BY prob_bucket
    """

    GET_EDGE_DISTRIBUTION = """
        SELECT
            CASE
                WHEN ABS(research_probability - market_price * 100) < 5 THEN 'tiny (<5%)'
                WHEN ABS(research_probability - market_price * 100) < 10 THEN 'small (5-10%)'
                WHEN ABS(research_probability - market_price * 100) < 20 THEN 'medium (10-20%)'
                ELSE 'large (>20%)'
            END as edge_bucket,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND outcome IS NOT NULL
          AND research_probability IS NOT NULL
          AND market_price IS NOT NULL
        GROUP BY edge_bucket
        ORDER BY edge_bucket
    """

    GET_OVERCONFIDENCE_ANALYSIS = """
        SELECT
            AVG(research_probability / 100.0) as avg_predicted_prob,
            AVG(CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) as avg_actual_outcome,
            AVG(research_probability / 100.0) - AVG(CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) as overconfidence_bias,
            COUNT(*) as sample_size,
            AVG(confidence) as avg_confidence
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND outcome IS NOT NULL
          AND research_probability IS NOT NULL
    """

    GET_KELLY_SIZE_EFFECTIVENESS = """
        SELECT
            CASE
                WHEN bet_amount < 5 THEN 'micro (<$5)'
                WHEN bet_amount < 25 THEN 'small ($5-25)'
                WHEN bet_amount < 75 THEN 'medium ($25-75)'
                ELSE 'large (>$75)'
            END as size_bucket,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl,
            AVG(kelly_fraction) as avg_kelly
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
        GROUP BY size_bucket
        ORDER BY size_bucket
    """

    GET_RECENT_PERFORMANCE_TREND = """
        SELECT
            DATE(timestamp) as date,
            COUNT(*) as num_bets,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins,
            SUM(CASE WHEN profit_loss < 0 THEN 1 ELSE 0 END) as losses,
            SUM(profit_loss) as daily_pnl,
            AVG(r_score) as avg_r_score,
            AVG(confidence) as avg_confidence
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND timestamp >= date('now', '-30 days')
        GROUP BY DATE(timestamp)
        ORDER BY date DESC
    """

    GET_CALIBRATION_SUMMARY = """
        SELECT
            COUNT(*) as total_settled,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) as win_rate,
            SUM(profit_loss) as total_pnl,
            SUM(bet_amount) as total_wagered,
            CASE WHEN SUM(bet_amount) > 0
                THEN SUM(profit_loss) / SUM(bet_amount) * 100
                ELSE 0
            END as roi_pct,
            AVG(
                (research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) *
                (research_probability/100.0 - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END)
            ) as brier_score,
            AVG(r_score) as avg_r_score,
            AVG(confidence) as avg_confidence
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND outcome IS NOT NULL
    """
