"""
Dashboard API - FastAPI server for Kalshi Trading Bot Dashboard.
Provides endpoints for decisions, KPIs, WebSocket real-time updates, and serves the dashboard UI.
"""
import os
import sys
import asyncio
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Query, HTTPException, WebSocket, WebSocketDisconnect, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_config
from db import get_database, get_postgres_database
from dashboard.websocket import manager, handle_websocket, WebSocketMessage, MessageType

app = FastAPI(
    title="Kalshi Trading Bot Dashboard",
    description="Real-time dashboard for monitoring trading decisions and performance",
    version="2.0.0"
)

# CORS middleware for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global database connection
_db = None
_config = None


async def get_db():
    """Get or create database connection."""
    global _db, _config
    if _db is None:
        _config = load_config()
        if _config.database.db_type.lower() == "postgres":
            _db = await get_postgres_database(
                host=_config.database.pg_host,
                database=_config.database.pg_database,
                user=_config.database.pg_user,
                password=_config.database.pg_password,
                port=_config.database.pg_port,
                ssl=_config.database.pg_ssl
            )
        else:
            _db = await get_database(_config.database.db_path)
    return _db


# Mount static files
static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# === Response Models ===

class DecisionRow(BaseModel):
    decision_id: str
    timestamp: str
    event_ticker: str
    event_title: str
    market_ticker: str
    market_title: str
    action: str
    bet_amount: float
    confidence: float
    reasoning: str
    research_probability: Optional[float] = None
    calc_market_prob: Optional[float] = None
    r_score: Optional[float] = None
    kelly_fraction: Optional[float] = None
    expected_return: Optional[float] = None
    status: str
    is_hedge: bool = False
    # TrendRadar fields
    run_mode: Optional[str] = None
    signal_applied: bool = False
    signal_direction: Optional[str] = None
    signal_strength: Optional[float] = None


class KPISummary(BaseModel):
    period: str
    realized_pnl: float
    unrealized_exposure: float
    win_rate: float
    avg_edge: float
    avg_confidence: float
    avg_r_score: float
    total_decisions: int
    actionable_bets: int
    skip_count: int
    skip_rate: float
    # TrendRadar KPIs
    trendradar_aligned: int = 0
    trendradar_conflicts: int = 0
    override_count: int = 0


class StatusResponse(BaseModel):
    bot_running: bool
    last_run_at: Optional[str]
    last_run_mode: str
    trendradar_enabled: bool
    database_connected: bool
    total_decisions: int
    pending_decisions: int


# === Endpoints ===

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the main dashboard page."""
    html_path = Path(__file__).parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return HTMLResponse("<h1>Dashboard loading...</h1><p>Static files not found.</p>")


@app.get("/api/decisions")
async def get_decisions(
    page: int = Query(1, ge=1),
    per_page: int = Query(50, ge=1, le=200),
    sort_by: str = Query("timestamp"),
    sort_order: str = Query("desc"),
    action: Optional[str] = None,
    min_confidence: Optional[float] = None,
    status: Optional[str] = None,
    search: Optional[str] = None,
    signal_direction: Optional[str] = None,
    run_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Get paginated list of betting decisions with filters."""
    db = await get_db()

    # Build query
    conditions = []
    params = []
    param_idx = 1

    if action and action != "all":
        conditions.append(f"action = ${param_idx}")
        params.append(action)
        param_idx += 1

    if min_confidence is not None:
        conditions.append(f"confidence >= ${param_idx}")
        params.append(min_confidence)
        param_idx += 1

    if status and status != "all":
        conditions.append(f"status = ${param_idx}")
        params.append(status)
        param_idx += 1

    if search:
        conditions.append(f"(event_title ILIKE ${param_idx} OR market_title ILIKE ${param_idx})")
        params.append(f"%{search}%")
        param_idx += 1

    if signal_direction and signal_direction != "all":
        conditions.append(f"signal_direction = ${param_idx}")
        params.append(signal_direction)
        param_idx += 1

    if run_mode and run_mode != "all":
        conditions.append(f"run_mode = ${param_idx}")
        params.append(run_mode)
        param_idx += 1

    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Validate sort column
    valid_sorts = ["timestamp", "confidence", "r_score", "bet_amount", "action"]
    if sort_by not in valid_sorts:
        sort_by = "timestamp"

    order = "DESC" if sort_order.lower() == "desc" else "ASC"

    # Count total
    count_query = f"SELECT COUNT(*) as count FROM betting_decisions WHERE {where_clause}"
    count_result = await db.fetchone(count_query, *params)
    total = count_result["count"] if count_result else 0

    # Get paginated results
    offset = (page - 1) * per_page
    query = f"""
        SELECT
            decision_id, timestamp, event_ticker, event_title,
            market_ticker, market_title, action, bet_amount,
            confidence, reasoning, research_probability,
            calc_market_prob, r_score, kelly_fraction,
            expected_return, status, is_hedge,
            run_mode, signal_applied, signal_direction, signal_strength
        FROM betting_decisions
        WHERE {where_clause}
        ORDER BY {sort_by} {order}
        LIMIT ${param_idx} OFFSET ${param_idx + 1}
    """
    params.extend([per_page, offset])

    rows = await db.fetchall(query, *params)

    decisions = []
    for row in rows:
        decisions.append({
            "decision_id": row["decision_id"],
            "timestamp": str(row["timestamp"]) if row["timestamp"] else "",
            "event_ticker": row["event_ticker"] or "",
            "event_title": row["event_title"] or "",
            "market_ticker": row["market_ticker"] or "",
            "market_title": row["market_title"] or "",
            "action": row["action"] or "",
            "bet_amount": float(row["bet_amount"]) if row["bet_amount"] else 0,
            "confidence": float(row["confidence"]) if row["confidence"] else 0,
            "reasoning": row["reasoning"] or "",
            "research_probability": float(row["research_probability"]) if row["research_probability"] else None,
            "calc_market_prob": float(row["calc_market_prob"]) if row["calc_market_prob"] else None,
            "r_score": float(row["r_score"]) if row["r_score"] else None,
            "kelly_fraction": float(row["kelly_fraction"]) if row["kelly_fraction"] else None,
            "expected_return": float(row["expected_return"]) if row["expected_return"] else None,
            "status": row["status"] or "pending",
            "is_hedge": bool(row["is_hedge"]) if row["is_hedge"] else False,
            # TrendRadar fields
            "run_mode": row.get("run_mode"),
            "signal_applied": bool(row.get("signal_applied")) if row.get("signal_applied") else False,
            "signal_direction": row.get("signal_direction"),
            "signal_strength": float(row["signal_strength"]) if row.get("signal_strength") else None,
        })

    return {
        "decisions": decisions,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page > 0 else 1
    }


@app.get("/api/decisions/{decision_id}")
async def get_decision_detail(decision_id: str) -> Dict[str, Any]:
    """Get full detail for a single decision."""
    db = await get_db()

    query = "SELECT * FROM betting_decisions WHERE decision_id = $1"
    row = await db.fetchone(query, decision_id)

    if not row:
        raise HTTPException(status_code=404, detail="Decision not found")

    return dict(row)


@app.get("/api/kpis")
async def get_kpis(period: str = Query("7d")) -> KPISummary:
    """Get KPI summary for specified time window."""
    db = await get_db()

    # Calculate period start
    now = datetime.utcnow()
    if period == "today":
        period_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == "7d":
        period_start = now - timedelta(days=7)
    elif period == "30d":
        period_start = now - timedelta(days=30)
    else:
        period_start = datetime(2020, 1, 1)  # All time

    # Get aggregated metrics
    query = """
        SELECT
            COALESCE(SUM(CASE WHEN status = 'settled' THEN profit_loss ELSE 0 END), 0) as realized_pnl,
            COALESCE(SUM(CASE WHEN status = 'pending' AND action != 'skip' THEN bet_amount ELSE 0 END), 0) as unrealized_exposure,
            COUNT(CASE WHEN status = 'settled' AND profit_loss > 0 THEN 1 END) as wins,
            COUNT(CASE WHEN status = 'settled' THEN 1 END) as settled_count,
            COALESCE(AVG(CASE WHEN action != 'skip' THEN calc_research_prob - calc_market_prob END), 0) as avg_edge,
            COALESCE(AVG(CASE WHEN action != 'skip' THEN confidence END), 0) as avg_confidence,
            COALESCE(AVG(CASE WHEN action != 'skip' THEN r_score END), 0) as avg_r_score,
            COUNT(*) as total_decisions,
            COUNT(CASE WHEN action != 'skip' THEN 1 END) as actionable_bets,
            COUNT(CASE WHEN action = 'skip' THEN 1 END) as skip_count,
            COUNT(CASE WHEN signal_direction = 'aligned' THEN 1 END) as trendradar_aligned,
            COUNT(CASE WHEN signal_direction = 'conflicting' THEN 1 END) as trendradar_conflicts,
            COUNT(CASE WHEN override_skip_triggered = true THEN 1 END) as override_count
        FROM betting_decisions
        WHERE timestamp >= $1
    """

    row = await db.fetchone(query, period_start)

    if not row:
        return KPISummary(
            period=period,
            realized_pnl=0,
            unrealized_exposure=0,
            win_rate=0,
            avg_edge=0,
            avg_confidence=0,
            avg_r_score=0,
            total_decisions=0,
            actionable_bets=0,
            skip_count=0,
            skip_rate=0
        )

    total = row["total_decisions"] or 0
    settled = row["settled_count"] or 0
    wins = row["wins"] or 0
    skip_count = row["skip_count"] or 0

    return KPISummary(
        period=period,
        realized_pnl=float(row["realized_pnl"]) if row["realized_pnl"] else 0,
        unrealized_exposure=float(row["unrealized_exposure"]) if row["unrealized_exposure"] else 0,
        win_rate=(wins / settled * 100) if settled > 0 else 0,
        avg_edge=float(row["avg_edge"]) * 100 if row["avg_edge"] else 0,
        avg_confidence=float(row["avg_confidence"]) if row["avg_confidence"] else 0,
        avg_r_score=float(row["avg_r_score"]) if row["avg_r_score"] else 0,
        total_decisions=total,
        actionable_bets=row["actionable_bets"] or 0,
        skip_count=skip_count,
        skip_rate=(skip_count / total * 100) if total > 0 else 0,
        trendradar_aligned=row["trendradar_aligned"] or 0,
        trendradar_conflicts=row["trendradar_conflicts"] or 0,
        override_count=row["override_count"] or 0
    )


@app.get("/api/status")
async def get_status() -> StatusResponse:
    """Get current bot and system status."""
    db = await get_db()
    config = _config

    # Get latest run info
    run_query = """
        SELECT run_id, started_at, completed_at, mode, status
        FROM run_history
        ORDER BY started_at DESC
        LIMIT 1
    """
    latest_run = await db.fetchone(run_query)

    # Get decision counts
    stats = await db.get_statistics()

    return StatusResponse(
        bot_running=latest_run["status"] == "running" if latest_run else False,
        last_run_at=str(latest_run["started_at"]) if latest_run else None,
        last_run_mode=latest_run["mode"] if latest_run else "dry_run",
        trendradar_enabled=config.trendradar.enabled if config else False,
        database_connected=True,
        total_decisions=stats.get("total_decisions", 0),
        pending_decisions=stats.get("pending_decisions", 0)
    )


@app.on_event("startup")
async def startup():
    """Initialize database connection on startup."""
    db = await get_db()
    print("Dashboard API started - Database connected")

    # Start WebSocket heartbeat loop
    asyncio.create_task(manager.heartbeat_loop())
    print("WebSocket heartbeat loop started")


# === Learning Metrics ===

@app.get("/api/learning/calibration")
async def get_calibration_metrics() -> Dict[str, Any]:
    """
    Get calibration metrics showing predicted vs actual outcomes.
    Used for learning from past events and improving predictions.
    """
    db = await get_db()

    # Overall calibration summary
    summary_query = """
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
                (calc_research_prob - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) *
                (calc_research_prob - CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END)
            ) as brier_score,
            AVG(r_score) as avg_r_score,
            AVG(confidence) as avg_confidence,
            AVG(calc_research_prob) as avg_predicted_prob,
            AVG(CASE WHEN outcome = 'yes' THEN 1.0 ELSE 0.0 END) as avg_actual_outcome
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND outcome IS NOT NULL
    """
    summary = await db.fetchone(summary_query)

    # Calibration by probability bucket
    bucket_query = """
        SELECT
            CASE
                WHEN calc_research_prob < 0.2 THEN '0-20%'
                WHEN calc_research_prob < 0.4 THEN '20-40%'
                WHEN calc_research_prob < 0.6 THEN '40-60%'
                WHEN calc_research_prob < 0.8 THEN '60-80%'
                ELSE '80-100%'
            END as prob_bucket,
            COUNT(*) as count,
            AVG(calc_research_prob) * 100 as avg_predicted,
            AVG(CASE WHEN outcome = 'yes' THEN 100.0 ELSE 0.0 END) as avg_actual,
            ABS(AVG(calc_research_prob) * 100 - AVG(CASE WHEN outcome = 'yes' THEN 100.0 ELSE 0.0 END)) as calibration_error
        FROM betting_decisions
        WHERE status = 'settled'
          AND action != 'skip'
          AND outcome IS NOT NULL
        GROUP BY prob_bucket
        ORDER BY prob_bucket
    """
    buckets = await db.fetchall(bucket_query)

    # R-score effectiveness
    rscore_query = """
        SELECT
            CASE
                WHEN r_score < 1.0 THEN 'low (<1.0)'
                WHEN r_score < 1.5 THEN 'medium (1.0-1.5)'
                WHEN r_score < 2.0 THEN 'good (1.5-2.0)'
                ELSE 'high (>2.0)'
            END as r_score_bucket,
            COUNT(*) as count,
            AVG(CASE WHEN profit_loss > 0 THEN 1.0 ELSE 0.0 END) * 100 as win_rate,
            SUM(profit_loss) as total_pnl,
            AVG(profit_loss) as avg_pnl
        FROM betting_decisions
        WHERE status = 'settled' AND action != 'skip' AND r_score IS NOT NULL
        GROUP BY r_score_bucket
        ORDER BY r_score_bucket
    """
    rscore_metrics = await db.fetchall(rscore_query)

    # Calculate overconfidence bias
    overconfidence_bias = 0.0
    if summary and summary.get("avg_predicted_prob") and summary.get("avg_actual_outcome"):
        overconfidence_bias = (summary["avg_predicted_prob"] - summary["avg_actual_outcome"]) * 100

    return {
        "summary": {
            "total_settled": summary["total_settled"] if summary else 0,
            "win_rate": round(summary["win_rate"] * 100, 2) if summary and summary["win_rate"] else 0,
            "total_pnl": round(summary["total_pnl"], 2) if summary and summary["total_pnl"] else 0,
            "roi_pct": round(summary["roi_pct"], 2) if summary and summary["roi_pct"] else 0,
            "brier_score": round(summary["brier_score"], 4) if summary and summary["brier_score"] else 0,
            "avg_r_score": round(summary["avg_r_score"], 2) if summary and summary["avg_r_score"] else 0,
            "overconfidence_bias": round(overconfidence_bias, 2)
        },
        "calibration_by_bucket": [
            {
                "bucket": b["prob_bucket"],
                "count": b["count"],
                "avg_predicted": round(b["avg_predicted"], 1),
                "avg_actual": round(b["avg_actual"], 1),
                "calibration_error": round(b["calibration_error"], 1)
            }
            for b in buckets
        ] if buckets else [],
        "rscore_effectiveness": [
            {
                "bucket": r["r_score_bucket"],
                "count": r["count"],
                "win_rate": round(r["win_rate"], 1),
                "total_pnl": round(r["total_pnl"], 2) if r["total_pnl"] else 0,
                "avg_pnl": round(r["avg_pnl"], 2) if r["avg_pnl"] else 0
            }
            for r in rscore_metrics
        ] if rscore_metrics else [],
        "recommendations": _generate_learning_recommendations(summary, buckets, overconfidence_bias)
    }


def _generate_learning_recommendations(summary, buckets, overconfidence_bias):
    """Generate actionable recommendations based on calibration data."""
    recommendations = []

    if summary:
        # Check for overconfidence
        if overconfidence_bias > 5:
            recommendations.append({
                "type": "calibration",
                "severity": "warning",
                "message": f"System is overconfident by {overconfidence_bias:.1f}%. Consider reducing research probability estimates."
            })
        elif overconfidence_bias < -5:
            recommendations.append({
                "type": "calibration",
                "severity": "info",
                "message": f"System is underconfident by {abs(overconfidence_bias):.1f}%. Consider increasing research probability estimates."
            })

        # Check Brier score (lower is better, <0.25 is good)
        brier = summary.get("brier_score", 0)
        if brier and brier > 0.25:
            recommendations.append({
                "type": "accuracy",
                "severity": "warning",
                "message": f"Brier score ({brier:.3f}) indicates room for improvement. Target is <0.25."
            })

        # Check ROI
        roi = summary.get("roi_pct", 0)
        if roi and roi < 0:
            recommendations.append({
                "type": "profitability",
                "severity": "critical",
                "message": f"Negative ROI ({roi:.1f}%). Review position sizing and entry criteria."
            })

    # Check calibration by bucket
    if buckets:
        for bucket in buckets:
            if bucket["calibration_error"] > 15:
                recommendations.append({
                    "type": "bucket_calibration",
                    "severity": "warning",
                    "message": f"Large calibration error ({bucket['calibration_error']:.1f}%) in {bucket['prob_bucket']} bucket."
                })

    if not recommendations:
        recommendations.append({
            "type": "status",
            "severity": "success",
            "message": "System is well-calibrated. Continue monitoring."
        })

    return recommendations


# === Health Check ===

@app.get("/api/health")
async def health_check():
    """Health check endpoint for load balancers."""
    try:
        db = await get_db()
        return {
            "status": "healthy",
            "database": "connected",
            "timestamp": datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }


@app.get("/api/freshness")
async def get_data_freshness():
    """
    Check data freshness - helps verify 15-min schedule is working.

    Returns information about the most recent decision and run,
    and whether data is considered stale (>20 minutes old).
    """
    try:
        db = await get_db()

        # Get most recent decision
        last_decision = await db.fetchone(
            "SELECT timestamp, run_id, market_ticker, action FROM betting_decisions ORDER BY timestamp DESC LIMIT 1"
        )

        # Get most recent run
        last_run = await db.fetchone(
            "SELECT run_id, started_at, completed_at, status FROM run_history ORDER BY started_at DESC LIMIT 1"
        )

        now = datetime.utcnow()
        last_decision_age_seconds = None
        last_run_age_seconds = None

        if last_decision and last_decision.get('timestamp'):
            last_ts = last_decision['timestamp']
            if isinstance(last_ts, str):
                last_ts = datetime.fromisoformat(last_ts.replace('Z', '+00:00'))
            if last_ts.tzinfo is not None:
                last_ts = last_ts.replace(tzinfo=None)
            last_decision_age_seconds = (now - last_ts).total_seconds()

        if last_run and last_run.get('started_at'):
            run_ts = last_run['started_at']
            if isinstance(run_ts, str):
                run_ts = datetime.fromisoformat(run_ts.replace('Z', '+00:00'))
            if run_ts.tzinfo is not None:
                run_ts = run_ts.replace(tzinfo=None)
            last_run_age_seconds = (now - run_ts).total_seconds()

        # Stale if no decisions or >20 minutes since last decision
        is_stale = last_decision_age_seconds is None or last_decision_age_seconds > 20 * 60

        return {
            "status": "fresh" if not is_stale else "stale",
            "last_decision": {
                "timestamp": str(last_decision['timestamp']) if last_decision and last_decision.get('timestamp') else None,
                "age_seconds": round(last_decision_age_seconds, 1) if last_decision_age_seconds else None,
                "age_minutes": round(last_decision_age_seconds / 60, 1) if last_decision_age_seconds else None,
                "market_ticker": last_decision.get('market_ticker') if last_decision else None,
                "action": last_decision.get('action') if last_decision else None,
            } if last_decision else None,
            "last_run": {
                "run_id": last_run.get('run_id') if last_run else None,
                "started_at": str(last_run.get('started_at')) if last_run else None,
                "status": last_run.get('status') if last_run else None,
                "age_seconds": round(last_run_age_seconds, 1) if last_run_age_seconds else None,
            } if last_run else None,
            "is_stale": is_stale,
            "expected_interval_minutes": 15,
            "stale_threshold_minutes": 20,
            "checked_at": now.isoformat()
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e),
            "is_stale": True,
            "checked_at": datetime.utcnow().isoformat()
        }


# === WebSocket Endpoints ===

@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    """
    WebSocket endpoint for real-time updates.

    Client can send:
    - {"action": "subscribe", "topics": ["decisions", "kpis", "alerts"]}
    - {"action": "unsubscribe", "topics": ["alerts"]}
    - {"action": "ping"}

    Server sends:
    - {"type": "decision", "data": {...}, "timestamp": "..."}
    - {"type": "kpi_update", "data": {...}, "timestamp": "..."}
    - {"type": "alert", "data": {...}, "timestamp": "..."}
    - {"type": "heartbeat", "data": {...}, "timestamp": "..."}
    """
    await handle_websocket(websocket)


@app.get("/api/ws/stats")
async def get_websocket_stats() -> Dict[str, Any]:
    """Get WebSocket connection statistics."""
    return manager.get_stats()


# === Broadcast Helper Endpoints (for bot integration) ===

@app.post("/api/broadcast/decision")
async def broadcast_decision(decision: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast a new decision to all WebSocket subscribers.
    Called by the trading bot when a new decision is made.
    """
    sent = await manager.broadcast_decision(decision)
    return {"sent_to": sent, "message": "Decision broadcast"}


@app.post("/api/broadcast/kpis")
async def broadcast_kpis_update() -> Dict[str, Any]:
    """
    Calculate and broadcast updated KPIs to all subscribers.
    Called after decisions are saved.
    """
    # Get fresh KPIs
    db = await get_db()
    now = datetime.utcnow()
    period_start = now - timedelta(days=7)

    query = """
        SELECT
            COALESCE(SUM(CASE WHEN status = 'settled' THEN profit_loss ELSE 0 END), 0) as realized_pnl,
            COALESCE(SUM(CASE WHEN status = 'pending' AND action != 'skip' THEN bet_amount ELSE 0 END), 0) as unrealized_exposure,
            COUNT(CASE WHEN status = 'settled' AND profit_loss > 0 THEN 1 END) as wins,
            COUNT(CASE WHEN status = 'settled' THEN 1 END) as settled_count,
            COALESCE(AVG(CASE WHEN action != 'skip' THEN calc_research_prob - calc_market_prob END), 0) as avg_edge,
            COALESCE(AVG(CASE WHEN action != 'skip' THEN confidence END), 0) as avg_confidence,
            COALESCE(AVG(CASE WHEN action != 'skip' THEN r_score END), 0) as avg_r_score,
            COUNT(*) as total_decisions,
            COUNT(CASE WHEN action != 'skip' THEN 1 END) as actionable_bets,
            COUNT(CASE WHEN action = 'skip' THEN 1 END) as skip_count
        FROM betting_decisions
        WHERE timestamp >= $1
    """
    row = await db.fetchone(query, period_start)

    if row:
        kpis = {
            "period": "7d",
            "realized_pnl": float(row["realized_pnl"]) if row["realized_pnl"] else 0,
            "unrealized_exposure": float(row["unrealized_exposure"]) if row["unrealized_exposure"] else 0,
            "win_rate": (row["wins"] / row["settled_count"] * 100) if row["settled_count"] > 0 else 0,
            "avg_edge": float(row["avg_edge"]) * 100 if row["avg_edge"] else 0,
            "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else 0,
            "avg_r_score": float(row["avg_r_score"]) if row["avg_r_score"] else 0,
            "total_decisions": row["total_decisions"] or 0,
            "actionable_bets": row["actionable_bets"] or 0,
            "skip_count": row["skip_count"] or 0,
        }
    else:
        kpis = {"period": "7d", "error": "No data"}

    sent = await manager.broadcast_kpi_update(kpis)
    return {"sent_to": sent, "kpis": kpis}


@app.post("/api/broadcast/alert")
async def broadcast_alert(alert: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast an alert to all WebSocket subscribers.
    Called when important events occur (loss limits, errors, etc.)
    """
    sent = await manager.broadcast_alert(alert)
    return {"sent_to": sent, "message": "Alert broadcast"}


@app.post("/api/broadcast/status")
async def broadcast_status_update(status: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast bot status update to all subscribers.
    Called when bot starts/stops or changes state.
    """
    sent = await manager.broadcast_status(status)
    return {"sent_to": sent, "message": "Status broadcast"}


# === Time Series Chart Endpoints ===

@app.get("/api/charts/pnl-curve")
async def get_pnl_curve(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Get cumulative P&L curve data for charting."""
    db = await get_db()
    period_start = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            DATE(timestamp) as date,
            SUM(profit_loss) as daily_pnl,
            COUNT(*) as trades,
            SUM(CASE WHEN profit_loss > 0 THEN 1 ELSE 0 END) as wins
        FROM betting_decisions
        WHERE status = 'settled' AND timestamp >= $1
        GROUP BY DATE(timestamp)
        ORDER BY date
    """
    rows = await db.fetchall(query, period_start)

    # Calculate cumulative P&L
    cumulative = 0
    data = []
    for row in rows:
        cumulative += float(row["daily_pnl"]) if row["daily_pnl"] else 0
        data.append({
            "date": str(row["date"]),
            "daily_pnl": float(row["daily_pnl"]) if row["daily_pnl"] else 0,
            "cumulative_pnl": cumulative,
            "trades": row["trades"],
            "wins": row["wins"]
        })

    return {
        "period_days": days,
        "total_pnl": cumulative,
        "data": data
    }


@app.get("/api/charts/r-score-distribution")
async def get_rscore_distribution(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Get R-score distribution for histogram charting."""
    db = await get_db()
    period_start = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            r_score,
            action,
            profit_loss,
            status
        FROM betting_decisions
        WHERE timestamp >= $1 AND r_score IS NOT NULL AND action != 'skip'
        ORDER BY r_score
    """
    rows = await db.fetchall(query, period_start)

    # Create histogram buckets
    buckets = {}
    for row in rows:
        r = float(row["r_score"]) if row["r_score"] else 0
        bucket = round(r * 2) / 2  # Round to nearest 0.5
        bucket_key = f"{bucket:.1f}"

        if bucket_key not in buckets:
            buckets[bucket_key] = {"count": 0, "wins": 0, "total_pnl": 0}

        buckets[bucket_key]["count"] += 1
        if row["status"] == "settled":
            pnl = float(row["profit_loss"]) if row["profit_loss"] else 0
            buckets[bucket_key]["total_pnl"] += pnl
            if pnl > 0:
                buckets[bucket_key]["wins"] += 1

    # Convert to sorted list
    distribution = [
        {
            "r_score": float(k),
            "count": v["count"],
            "wins": v["wins"],
            "win_rate": v["wins"] / v["count"] * 100 if v["count"] > 0 else 0,
            "total_pnl": v["total_pnl"]
        }
        for k, v in sorted(buckets.items(), key=lambda x: float(x[0]))
    ]

    return {
        "period_days": days,
        "total_trades": len(rows),
        "distribution": distribution
    }


@app.get("/api/charts/daily-activity")
async def get_daily_activity(days: int = Query(30, ge=1, le=365)) -> Dict[str, Any]:
    """Get daily trading activity for charting."""
    db = await get_db()
    period_start = datetime.utcnow() - timedelta(days=days)

    query = """
        SELECT
            DATE(timestamp) as date,
            COUNT(*) as total_decisions,
            COUNT(CASE WHEN action = 'buy_yes' THEN 1 END) as buy_yes_count,
            COUNT(CASE WHEN action = 'buy_no' THEN 1 END) as buy_no_count,
            COUNT(CASE WHEN action = 'skip' THEN 1 END) as skip_count,
            COALESCE(SUM(bet_amount), 0) as total_wagered,
            COALESCE(AVG(confidence), 0) as avg_confidence,
            COALESCE(AVG(CASE WHEN action != 'skip' THEN r_score END), 0) as avg_r_score
        FROM betting_decisions
        WHERE timestamp >= $1
        GROUP BY DATE(timestamp)
        ORDER BY date
    """
    rows = await db.fetchall(query, period_start)

    data = []
    for row in rows:
        data.append({
            "date": str(row["date"]),
            "total_decisions": row["total_decisions"],
            "buy_yes": row["buy_yes_count"],
            "buy_no": row["buy_no_count"],
            "skipped": row["skip_count"],
            "total_wagered": float(row["total_wagered"]) if row["total_wagered"] else 0,
            "avg_confidence": float(row["avg_confidence"]) if row["avg_confidence"] else 0,
            "avg_r_score": float(row["avg_r_score"]) if row["avg_r_score"] else 0
        })

    return {
        "period_days": days,
        "data": data
    }


# === Run History Endpoint ===

@app.get("/api/runs")
async def get_run_history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
) -> Dict[str, Any]:
    """Get bot run history with pagination."""
    db = await get_db()

    # Count total
    count_result = await db.fetchone("SELECT COUNT(*) as count FROM run_history")
    total = count_result["count"] if count_result else 0

    # Get paginated results
    offset = (page - 1) * per_page
    query = """
        SELECT
            run_id, started_at, completed_at, mode, environment,
            max_events, z_threshold, kelly_fraction,
            events_analyzed, markets_analyzed, decisions_made,
            bets_placed, total_wagered, status, error_message
        FROM run_history
        ORDER BY started_at DESC
        LIMIT $1 OFFSET $2
    """
    rows = await db.fetchall(query, per_page, offset)

    runs = []
    for row in rows:
        started = row["started_at"]
        completed = row["completed_at"]
        duration = None
        if started and completed:
            try:
                duration = (completed - started).total_seconds()
            except Exception:
                pass

        runs.append({
            "run_id": row["run_id"],
            "started_at": str(started) if started else None,
            "completed_at": str(completed) if completed else None,
            "duration_seconds": duration,
            "mode": row["mode"],
            "environment": row["environment"],
            "status": row["status"],
            "events_analyzed": row["events_analyzed"],
            "markets_analyzed": row["markets_analyzed"],
            "decisions_made": row["decisions_made"],
            "bets_placed": row["bets_placed"],
            "total_wagered": float(row["total_wagered"]) if row["total_wagered"] else 0,
            "config": {
                "max_events": row["max_events"],
                "z_threshold": float(row["z_threshold"]) if row["z_threshold"] else None,
                "kelly_fraction": float(row["kelly_fraction"]) if row["kelly_fraction"] else None
            },
            "error_message": row["error_message"]
        })

    return {
        "runs": runs,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page if per_page > 0 else 1
    }


# === Position Monitoring Endpoints ===

# Global position monitor reference (set by bot when it starts)
_position_monitor = None
_kalshi_client = None


def set_position_monitor(monitor):
    """Set the position monitor instance (called by trading bot on startup)."""
    global _position_monitor
    _position_monitor = monitor


def set_kalshi_client(client):
    """Set the Kalshi client instance (called by trading bot on startup)."""
    global _kalshi_client
    _kalshi_client = client


class PositionResponse(BaseModel):
    decision_id: str
    ticker: str
    side: str
    contracts: int
    entry_price: int
    current_price: int
    pnl_dollars: float
    pnl_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    stop_loss_price: int
    take_profit_price: int
    trailing_stop_price: Optional[int] = None
    high_water_mark: int
    is_active: bool
    exit_pending: bool
    last_update: Optional[str] = None


class PositionSummary(BaseModel):
    count: int
    total_unrealized_pnl: float
    total_value: float
    monitor_running: bool
    stats: Dict[str, int]
    positions: List[Dict[str, Any]]


class TriggerUpdateRequest(BaseModel):
    stop_loss_pct: Optional[float] = None
    take_profit_pct: Optional[float] = None
    trailing_stop_pct: Optional[float] = None


@app.get("/api/positions/live")
async def get_live_positions() -> PositionSummary:
    """
    Get all monitored positions with real-time P&L.

    Returns positions being actively monitored for stop-loss/take-profit.
    """
    if _position_monitor is None:
        # No monitor running - try to get positions from database
        db = await get_db()

        query = """
            SELECT
                decision_id, market_ticker, action,
                filled_price_cents, filled_contracts,
                current_price_cents, unrealized_pnl_dollars, unrealized_pnl_pct,
                stop_loss_pct, take_profit_pct, trailing_stop_pct,
                high_water_mark_cents, last_price_update
            FROM betting_decisions
            WHERE status = 'pending'
              AND action IN ('buy_yes', 'buy_no')
              AND filled_contracts > 0
              AND sl_tp_enabled = TRUE
              AND exit_order_id IS NULL
        """

        try:
            rows = await db.fetchall(query)
            positions = []
            total_pnl = 0

            for row in rows:
                pnl = float(row.get("unrealized_pnl_dollars") or 0)
                total_pnl += pnl
                entry_price = row.get("filled_price_cents") or 0
                sl_pct = row.get("stop_loss_pct") or 0.15
                tp_pct = row.get("take_profit_pct") or 0.30

                positions.append({
                    "decision_id": row["decision_id"],
                    "ticker": row["market_ticker"],
                    "side": "yes" if row["action"] == "buy_yes" else "no",
                    "contracts": row.get("filled_contracts") or 0,
                    "entry_price": entry_price,
                    "current_price": row.get("current_price_cents") or entry_price,
                    "pnl_dollars": round(pnl, 2),
                    "pnl_pct": round(float(row.get("unrealized_pnl_pct") or 0) * 100, 2),
                    "stop_loss_pct": sl_pct,
                    "take_profit_pct": tp_pct,
                    "stop_loss_price": int(entry_price * (1 - sl_pct)),
                    "take_profit_price": int(entry_price * (1 + tp_pct)),
                    "trailing_stop_price": None,
                    "high_water_mark": row.get("high_water_mark_cents") or entry_price,
                    "is_active": True,
                    "exit_pending": False,
                    "last_update": str(row.get("last_price_update")) if row.get("last_price_update") else None,
                })

            return PositionSummary(
                count=len(positions),
                total_unrealized_pnl=round(total_pnl, 2),
                total_value=sum(p.get("contracts", 0) * p.get("current_price", 0) / 100 for p in positions),
                monitor_running=False,
                stats={"checks": 0, "triggers": 0, "successful_exits": 0, "failed_exits": 0},
                positions=positions
            )

        except Exception as e:
            return PositionSummary(
                count=0,
                total_unrealized_pnl=0,
                total_value=0,
                monitor_running=False,
                stats={"error": str(e)},
                positions=[]
            )

    # Monitor is running - get live data
    summary = _position_monitor.get_positions_summary()
    return PositionSummary(
        count=summary["count"],
        total_unrealized_pnl=summary["total_unrealized_pnl"],
        total_value=summary["total_value"],
        monitor_running=True,
        stats=summary["stats"],
        positions=summary["positions"]
    )


@app.get("/api/positions/near-triggers")
async def get_positions_near_triggers(threshold_pct: float = Query(0.05, ge=0.01, le=0.20)) -> Dict[str, Any]:
    """
    Get positions that are near stop-loss or take-profit triggers.

    Args:
        threshold_pct: Distance threshold (default 5%)
    """
    if _position_monitor is None:
        return {"count": 0, "positions": [], "monitor_running": False}

    near_positions = _position_monitor.get_positions_near_trigger(threshold_pct)

    return {
        "count": len(near_positions),
        "threshold_pct": threshold_pct,
        "monitor_running": True,
        "positions": [p.to_dict() for p in near_positions]
    }


@app.get("/api/positions/exits")
async def get_exit_history(
    days: int = Query(7, ge=1, le=90),
    exit_reason: Optional[str] = None
) -> Dict[str, Any]:
    """
    Get history of position exits (stop-loss, take-profit, manual).

    Args:
        days: Number of days to look back
        exit_reason: Filter by exit reason (stop_loss, take_profit, trailing_stop, manual)
    """
    db = await get_db()
    period_start = datetime.utcnow() - timedelta(days=days)

    conditions = ["exit_timestamp >= $1", "exit_order_id IS NOT NULL"]
    params = [period_start]
    param_idx = 2

    if exit_reason:
        conditions.append(f"exit_reason = ${param_idx}")
        params.append(exit_reason)

    query = f"""
        SELECT
            decision_id, market_ticker, action,
            filled_price_cents, exit_price_cents,
            filled_contracts, exit_contracts,
            exit_reason, exit_pnl_dollars,
            exit_timestamp, exit_order_id
        FROM betting_decisions
        WHERE {' AND '.join(conditions)}
        ORDER BY exit_timestamp DESC
        LIMIT 100
    """

    try:
        rows = await db.fetchall(query, *params)
    except Exception:
        rows = []

    exits = []
    for row in rows:
        exits.append({
            "decision_id": row["decision_id"],
            "ticker": row["market_ticker"],
            "side": "yes" if row["action"] == "buy_yes" else "no",
            "entry_price": row.get("filled_price_cents"),
            "exit_price": row.get("exit_price_cents"),
            "contracts": row.get("exit_contracts"),
            "exit_reason": row.get("exit_reason"),
            "pnl_dollars": float(row["exit_pnl_dollars"]) if row.get("exit_pnl_dollars") else 0,
            "exit_timestamp": str(row["exit_timestamp"]) if row.get("exit_timestamp") else None,
            "order_id": row.get("exit_order_id")
        })

    # Summary by exit reason
    summary_query = """
        SELECT
            exit_reason,
            COUNT(*) as count,
            SUM(exit_pnl_dollars) as total_pnl,
            AVG(exit_pnl_dollars) as avg_pnl
        FROM betting_decisions
        WHERE exit_timestamp >= $1 AND exit_order_id IS NOT NULL
        GROUP BY exit_reason
    """

    try:
        summary_rows = await db.fetchall(summary_query, period_start)
        summary = {
            row["exit_reason"]: {
                "count": row["count"],
                "total_pnl": round(float(row["total_pnl"]) if row["total_pnl"] else 0, 2),
                "avg_pnl": round(float(row["avg_pnl"]) if row["avg_pnl"] else 0, 2)
            }
            for row in summary_rows if row.get("exit_reason")
        }
    except Exception:
        summary = {}

    return {
        "period_days": days,
        "exit_reason_filter": exit_reason,
        "count": len(exits),
        "exits": exits,
        "summary_by_reason": summary
    }


@app.post("/api/positions/liquidate-all")
async def liquidate_all_positions() -> Dict[str, Any]:
    """
    Emergency: Liquidate ALL open positions.

    This is a destructive operation that closes all positions at market.
    """
    if _kalshi_client is None:
        raise HTTPException(
            status_code=503,
            detail="Kalshi client not available. Bot must be running."
        )

    result = await _kalshi_client.liquidate_all_positions()

    # Broadcast alert
    await manager.broadcast_alert({
        "type": "emergency_liquidation",
        "message": f"Emergency liquidation: {len(result.get('liquidated', []))} positions closed",
        "severity": "critical",
        "result": result
    })

    return result


@app.get("/api/positions/{decision_id}")
async def get_position_detail(decision_id: str) -> Dict[str, Any]:
    """Get detailed info for a specific position."""
    if _position_monitor:
        position = _position_monitor.get_position(decision_id)
        if position:
            return position.to_dict()

    # Fallback to database
    db = await get_db()
    query = """
        SELECT *
        FROM betting_decisions
        WHERE decision_id = $1
    """
    row = await db.fetchone(query, decision_id)

    if not row:
        raise HTTPException(status_code=404, detail="Position not found")

    return dict(row)


@app.put("/api/positions/{decision_id}/triggers")
async def update_position_triggers(
    decision_id: str,
    request: TriggerUpdateRequest
) -> Dict[str, Any]:
    """
    Update stop-loss/take-profit triggers for a position.

    Args:
        decision_id: The decision ID of the position
        request: Trigger update request with optional stop_loss_pct, take_profit_pct, trailing_stop_pct
    """
    if _position_monitor:
        success = await _position_monitor.update_triggers(
            decision_id,
            stop_loss_pct=request.stop_loss_pct,
            take_profit_pct=request.take_profit_pct,
            trailing_stop_pct=request.trailing_stop_pct
        )

        if success:
            position = _position_monitor.get_position(decision_id)
            return {
                "success": True,
                "position": position.to_dict() if position else None
            }
        else:
            raise HTTPException(status_code=404, detail="Position not found or not monitored")

    # Fallback: update directly in database
    db = await get_db()

    updates = []
    params = []
    param_idx = 1

    if request.stop_loss_pct is not None:
        updates.append(f"stop_loss_pct = ${param_idx}")
        params.append(request.stop_loss_pct)
        param_idx += 1

    if request.take_profit_pct is not None:
        updates.append(f"take_profit_pct = ${param_idx}")
        params.append(request.take_profit_pct)
        param_idx += 1

    if request.trailing_stop_pct is not None:
        updates.append(f"trailing_stop_pct = ${param_idx}")
        params.append(request.trailing_stop_pct)
        param_idx += 1

    if not updates:
        raise HTTPException(status_code=400, detail="No updates provided")

    params.append(decision_id)
    query = f"""
        UPDATE betting_decisions
        SET {', '.join(updates)}
        WHERE decision_id = ${param_idx}
        RETURNING decision_id, stop_loss_pct, take_profit_pct, trailing_stop_pct
    """

    try:
        row = await db.fetchone(query, *params)
        if not row:
            raise HTTPException(status_code=404, detail="Position not found")

        return {
            "success": True,
            "decision_id": row["decision_id"],
            "stop_loss_pct": row["stop_loss_pct"],
            "take_profit_pct": row["take_profit_pct"],
            "trailing_stop_pct": row["trailing_stop_pct"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/positions/{ticker}/liquidate")
async def liquidate_position(ticker: str) -> Dict[str, Any]:
    """
    Manually liquidate a specific position.

    Args:
        ticker: Market ticker to liquidate
    """
    if _kalshi_client is None:
        raise HTTPException(
            status_code=503,
            detail="Kalshi client not available. Bot must be running."
        )

    result = await _kalshi_client.liquidate_position(ticker)

    # Broadcast alert
    if result.get("success"):
        await manager.broadcast_alert({
            "type": "liquidation",
            "ticker": ticker,
            "message": f"Position liquidated: {ticker}",
            "severity": "warning"
        })

    return result


@app.post("/api/broadcast/position")
async def broadcast_position_update(position: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast a position update to all WebSocket subscribers.
    Called by the position monitor when prices update.
    """
    message = WebSocketMessage(
        type=MessageType.STATUS,  # Reuse STATUS for positions
        data={"position_update": position}
    )
    sent = await manager.broadcast(message)
    return {"sent_to": sent, "message": "Position update broadcast"}


@app.post("/api/broadcast/trigger")
async def broadcast_trigger_alert(trigger: Dict[str, Any]) -> Dict[str, Any]:
    """
    Broadcast a trigger alert to all WebSocket subscribers.
    Called when stop-loss or take-profit triggers.
    """
    await manager.broadcast_alert({
        "type": "trigger",
        "trigger_type": trigger.get("trigger_type"),
        "ticker": trigger.get("ticker"),
        "pnl_dollars": trigger.get("pnl_dollars"),
        "message": f"{trigger.get('trigger_type', 'TRIGGER').upper()}: {trigger.get('ticker')} "
                   f"(P&L: ${trigger.get('pnl_dollars', 0):.2f})",
        "severity": "warning" if trigger.get("trigger_type") == "stop_loss" else "info"
    })
    return {"success": True, "message": "Trigger alert broadcast"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
