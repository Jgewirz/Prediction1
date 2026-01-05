# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**Kalshi Deep Trading Bot** - An automated prediction market trading bot for Kalshi that uses:
- **OpenAI GPT-4o** for market research via Octagon Deep Research API
- **OpenAI GPT-5 Responses API** for structured betting decisions with Pydantic models
- **TrendRadar MCP** for real-time news sentiment signals (optional integration)
- **Kalshi API** for market data and order execution with RSA authentication
- **PostgreSQL (Neon) or SQLite** for historical tracking and automated outcome reconciliation

## Commands

```bash
# Install dependencies (uses uv package manager)
uv sync

# Run bot in dry-run mode (default - no real bets)
uv run trading-bot

# Run bot in live trading mode
uv run trading-bot --live

# Filter markets by expiration (hours from now)
uv run trading-bot --max-expiration-hours 6

# Reconcile outcomes and calculate P&L for settled markets
uv run trading-bot --reconcile

# Show performance statistics from database
uv run trading-bot --stats

# Migrate existing CSV/JSON data to SQLite
uv run trading-bot --migrate

# Analyze performance
uv run analyze_performance.py
uv run analyze_performance.py --db      # Load from database
uv run analyze_performance.py --pnl     # Show P&L summary

# Start the dashboard API (FastAPI on port 8000)
uv run python dashboard/api.py

# Install dev dependencies and run tests
uv sync --group dev
uv run pytest
uv run pytest tests/test_config_validation.py -v  # Run single test file

# Format and lint
uv run black .
uv run isort .
uv run flake8 .
```

## Architecture

### Bot Workflow (trading_bot.py)

The `SimpleTradingBot` class orchestrates a linear pipeline:

1. **Fetch Events** (`get_top_events`) - Gets top events by 24h volume from Kalshi
2. **Process Markets** (`get_markets_for_events`) - Uses top N markets per event by volume
3. **Filter Positions** (`filter_markets_by_positions`) - Skips events with existing positions
4. **Fetch TrendRadar Signals** - Gets news sentiment signals for events (if enabled)
5. **Research Events** (`research_events`) - Parallel batch calls to OpenAI GPT-4o
6. **Extract Probabilities** (`extract_probabilities`) - GPT-5 extracts structured probabilities
7. **Get Market Odds** (`get_market_odds`) - Fetches current bid/ask prices
8. **Generate Decisions** (`get_betting_decisions`) - GPT-5 generates betting decisions with signal influence
9. **Save to Database** (`save_decisions_to_db`) - Persists decisions to PostgreSQL/SQLite
10. **Place Bets** (`place_bets`) - Executes orders via Kalshi API

### Key Components

| File | Purpose |
|------|---------|
| `trading_bot.py` | Main bot with `SimpleTradingBot` class, CLI entry point |
| `kalshi_client.py` | `KalshiClient` - RSA-signed API calls, pagination, settlement data |
| `research_client.py` | `OctagonClient` - GPT-4o research via OpenAI SDK |
| `openai_utils.py` | Responses API helpers for GPT-5 structured outputs |
| `betting_models.py` | Pydantic models: `BettingDecision`, `MarketAnalysis`, `ProbabilityExtraction` |
| `config.py` | `BotConfig` with nested configs: `KalshiConfig`, `OpenAIConfig`, `DatabaseConfig`, `TrendRadarConfig` |
| `reconciliation.py` | `ReconciliationEngine` - Automated outcome tracking and P&L calculation |
| `trendradar_client.py` | `TrendRadarClient` - News sentiment signals with circuit breaker, retry, caching |
| `analyze_performance.py` | Performance analysis dashboard |
| `dashboard/api.py` | FastAPI server for real-time monitoring dashboard |
| `dashboard/websocket.py` | WebSocket connection manager for live updates |
| `dashboard/broadcaster.py` | Helper module for bot to push updates to dashboard |

### Database Module (db/)

| File | Purpose |
|------|---------|
| `db/database.py` | `Database` class - async SQLite connection manager |
| `db/postgres.py` | `PostgresDatabase` class - async PostgreSQL (Neon) connection |
| `db/schema.sql` | Schema: `betting_decisions`, `market_snapshots`, `performance_daily`, `calibration_records`, `run_history` |
| `db/queries.py` | `Queries` class - SQL constants for P&L, calibration, R-score analysis |

### Migrations (migrations/)

| File | Purpose |
|------|---------|
| `migrations/migrate_csv.py` | Migrate existing CSV files to SQLite |
| `migrations/migrate_json.py` | Migrate calibration JSON to SQLite |

### TrendRadar Integration

The bot optionally integrates with TrendRadar MCP server for news-based signal enhancement:

| Component | Purpose |
|-----------|---------|
| `TrendRadarClient` | HTTP client with circuit breaker, exponential backoff retry, TTL caching |
| `TrendingSignal` | Dataclass: topic, sentiment (positive/negative/neutral), strength, source_count |
| `calculate_signal_influence()` | Calculates confidence boost and Kelly multiplier from signals |
| `SignalCache` | Per-event signal caching with configurable TTL (default 5 min) |

Signal influence on decisions:
- **Aligned signals** (positive+buy_yes or negative+buy_no): Up to +30% confidence boost, 1.25x Kelly multiplier
- **Conflicting signals**: -15% confidence penalty, 0.8x Kelly multiplier
- **Skip override**: Strong signals (>0.8 strength, >5 sources) can override skip decisions

### Risk-Adjusted Trading System

Hedge-fund style metrics calculated in `calculate_risk_adjusted_metrics()`:
- **R-score (z-score)**: `(research_prob - market_price) / sqrt(p*(1-p))` - minimum threshold for betting
- **Kelly fraction**: `(p - y) / (1 - y)` - position sizing with fractional Kelly
- **Expected return**: `(p - y) / y`

Portfolio selection in `apply_portfolio_selection()` limits to top N positions by R-score.

### Authentication Flow (Kalshi)

RSA signature authentication:
1. `_get_headers()` creates timestamp + method + path message
2. `_sign_message()` signs with PSS padding and SHA256
3. Headers: `KALSHI-ACCESS-KEY`, `KALSHI-ACCESS-TIMESTAMP`, `KALSHI-ACCESS-SIGNATURE`

### Reconciliation Flow

1. `--reconcile` fetches pending decisions from database
2. Batch queries Kalshi API for settled markets
3. Calculates P&L: `buy_yes` wins if outcome=yes, `buy_no` wins if outcome=no
4. Updates `betting_decisions` with outcome, payout, profit_loss
5. Aggregates daily performance metrics

## Configuration

Copy `env_template.txt` to `.env`. Key settings:

```env
# Environment
KALSHI_USE_DEMO=true
KALSHI_API_KEY=...
KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"

# Research & Decision APIs
OCTAGON_API_KEY=...
OPENAI_API_KEY=...
OPENAI_MODEL=gpt-4o

# Database (PostgreSQL - Neon, recommended)
DB_TYPE=postgres
PGHOST=ep-flat-silence-ahtaz4uz-pooler.c-3.us-east-1.aws.neon.tech
PGDATABASE=neondb
PGUSER=neondb_owner
PGPASSWORD=...
PGSSLMODE=require

# Or SQLite (local)
# DB_TYPE=sqlite
# DB_PATH=trading_bot.db

ENABLE_DB=true
SAVE_TO_CSV=true
AUTO_RECONCILE=true

# TrendRadar News Intelligence
TRENDRADAR_ENABLED=true
TRENDRADAR_URL=http://localhost:3333
TRENDRADAR_TIMEOUT=10.0
TRENDRADAR_MAX_RETRIES=2
TRENDRADAR_CIRCUIT_THRESHOLD=3

# Risk Management
Z_THRESHOLD=1.5                   # Minimum R-score to place bet
KELLY_FRACTION=0.5                # Fractional Kelly (0.5 = half-Kelly)
BANKROLL=1000.0
MAX_BET_AMOUNT=25.0
MAX_PORTFOLIO_POSITIONS=10

# Daily Loss Limit (kill switch)
MAX_DAILY_LOSS=100.0
MAX_DAILY_LOSS_PCT=0.10
ENABLE_KILL_SWITCH=true

# Processing
MAX_EVENTS_TO_ANALYZE=50
RESEARCH_BATCH_SIZE=10
SKIP_EXISTING_POSITIONS=true
```

## Database Schema

```
betting_decisions     - All bet decisions with outcomes
├── decision_id       - Unique UUID
├── market_ticker     - Market identifier
├── action/amount     - Bet details (buy_yes, buy_no, skip)
├── r_score/kelly     - Risk metrics
├── outcome           - yes/no/NULL (pending)
├── profit_loss       - Calculated P&L
├── status            - pending/settled/skipped
└── signal_*          - TrendRadar signal fields (direction, strength, etc.)

market_snapshots      - Point-in-time market data
calibration_records   - Prediction vs outcome tracking
performance_daily     - Aggregated daily metrics
run_history          - Bot execution history with config snapshots
```

## API Documentation

- **Kalshi**: [docs.kalshi.com](https://docs.kalshi.com/) - RSA auth, market data, settlements
- **OpenAI Responses API**: [platform.openai.com](https://platform.openai.com/docs/api-reference) - structured outputs
- **Octagon**: [docs.octagon.ai](https://docs.octagon.ai/) - deep research endpoint

## Real-time Dashboard (v2.0)

The dashboard provides live monitoring with WebSocket streaming:

### Starting the Dashboard

```bash
# Start dashboard server
cd kalshi-deep-trading-bot
uv run python dashboard/api.py

# Access at http://localhost:8000
```

### Dashboard API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Dashboard UI |
| `/ws/live` | WS | WebSocket for real-time updates |
| `/api/decisions` | GET | Paginated decisions with filters |
| `/api/decisions/{id}` | GET | Single decision detail |
| `/api/kpis` | GET | KPI summary (P&L, win rate, etc.) |
| `/api/status` | GET | Bot status and connection info |
| `/api/charts/pnl-curve` | GET | Cumulative P&L data for charting |
| `/api/charts/r-score-distribution` | GET | R-score histogram data |
| `/api/charts/daily-activity` | GET | Daily trading activity |
| `/api/runs` | GET | Bot run history |
| `/api/ws/stats` | GET | WebSocket connection stats |

### Broadcasting from Trading Bot

```python
from dashboard.broadcaster import broadcast_decision, broadcast_kpi_update, broadcast_alert

# After saving a decision
await broadcast_decision({
    "decision_id": "...",
    "action": "buy_yes",
    "market_title": "...",
    "bet_amount": 25.0
})

# After a batch of decisions
await broadcast_kpi_update()

# On important events
await broadcast_alert("Daily loss limit reached!", severity="critical")
```

### WebSocket Message Types

- `decision` - New betting decision
- `kpi_update` - Updated KPI metrics
- `alert` - Alert notification
- `status` - Bot status change
- `heartbeat` - Connection health check (every 30s)

## Output

- **PostgreSQL/SQLite Database**: Primary storage for all decisions
- **CSV Backup**: `betting_decisions/betting_decisions_{timestamp}.csv`
- **Logs**: `bot_output.log`
- **Dashboard**: http://localhost:8000 (real-time WebSocket monitoring)
