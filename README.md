# Kalshi Deep Trading Bot

An automated trading bot for Kalshi prediction markets using AI-powered research and structured decision making.

![Trading Bot Flowchart](KalshiDeepTradingBot.png)

## Features

- **AI-Powered Research**: Uses OpenAI GPT-4o for market analysis and structured betting decisions
- **Real-time Dashboard**: Web-based monitoring with WebSocket updates
- **Risk Management**: Kelly criterion, R-score thresholds, daily loss limits, and kill switch
- **PostgreSQL Storage**: Persistent tracking with Neon cloud database
- **Position Monitoring**: Stop-loss and take-profit automation
- **Cloud Deployment**: Ready for Render with Docker support

## Quick Start

### 1. Install Dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install project dependencies
uv sync
```

### 2. Configure Environment

```bash
cp env_template.txt .env
# Edit .env with your API keys
```

Required API keys:
- **Kalshi API**: Get from [kalshi.com](https://docs.kalshi.com/getting_started/api_keys)
- **OpenAI API**: Get from [platform.openai.com](https://platform.openai.com/api-keys)

### 3. Run the Bot

```bash
# Dry run mode (no real bets)
uv run trading-bot

# Live trading mode
uv run trading-bot --live

# Filter by expiration time
uv run trading-bot --max-expiration-hours 6

# Reconcile settled bets
uv run trading-bot --reconcile

# Show performance stats
uv run trading-bot --stats
```

### 4. Start the Dashboard

```bash
uv run python dashboard/api.py
# Open http://localhost:8000
```

## Project Structure

```
kalshi-deep-trading-bot/
├── trading_bot.py           # Main bot with SimpleTradingBot class
├── continuous_bot.py        # Continuous trading loop for production
├── config.py                # Configuration management with Pydantic
├── cli.py                   # Command-line interface
│
├── kalshi_client.py         # Kalshi API client (RSA auth, orders, settlements)
├── research_client.py       # Research API client
├── openai_utils.py          # OpenAI Responses API helpers
├── betting_models.py        # Pydantic models for decisions
├── trendradar_client.py     # News sentiment integration (optional)
│
├── reconciliation.py        # Outcome tracking and P&L calculation
├── calibration_tracker.py   # Prediction calibration metrics
├── position_monitor.py      # Stop-loss/take-profit automation
│
├── dashboard/
│   ├── api.py               # FastAPI server (serves UI + REST API)
│   ├── websocket.py         # Real-time WebSocket updates
│   ├── broadcaster.py       # Event broadcasting to dashboard
│   └── static/
│       ├── index.html       # Dashboard UI
│       ├── app.js           # Dashboard JavaScript
│       └── style.css        # Dashboard styles
│
├── db/
│   ├── database.py          # SQLite connection manager
│   ├── postgres.py          # PostgreSQL (Neon) connection
│   ├── queries.py           # SQL query constants
│   └── schema.sql           # Database schema
│
├── migrations/
│   ├── migrate_csv.py       # CSV to database migration
│   └── migrate_json.py      # JSON to database migration
│
├── tests/
│   ├── test_config_validation.py
│   ├── test_trendradar_client.py
│   └── test_prompt_hygiene.py
│
├── Dockerfile               # Docker container config
├── render.yaml              # Render deployment blueprint
├── pyproject.toml           # Dependencies and project config
└── env_template.txt         # Environment variable template
```

## Configuration

Key settings in `.env`:

```env
# Kalshi API
KALSHI_API_KEY=your_key
KALSHI_PRIVATE_KEY="-----BEGIN RSA PRIVATE KEY-----\n...\n-----END RSA PRIVATE KEY-----"
KALSHI_USE_DEMO=true

# OpenAI
OPENAI_API_KEY=your_key
OPENAI_MODEL=gpt-4o

# Database (PostgreSQL)
DB_TYPE=postgres
PGHOST=your-neon-host.neon.tech
PGDATABASE=neondb
PGUSER=neondb_owner
PGPASSWORD=your_password
PGSSLMODE=require

# Risk Management
BANKROLL=1000.0
MAX_BET_AMOUNT=25.0
Z_THRESHOLD=1.5
KELLY_FRACTION=0.25
MAX_DAILY_LOSS=100.0
ENABLE_KILL_SWITCH=true
```

## Dashboard

The dashboard provides real-time monitoring at `http://localhost:8000`:

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Dashboard UI |
| `GET /api/decisions` | Paginated decisions with filters |
| `GET /api/kpis` | KPI summary (P&L, win rate) |
| `GET /api/status` | Bot status and connection info |
| `GET /api/health` | Health check for load balancers |
| `GET /api/charts/pnl-curve` | Cumulative P&L data |
| `GET /api/positions/live` | Active positions with real-time P&L |
| `WS /ws/live` | WebSocket for real-time updates |

## Deployment

### Render (Recommended)

1. Connect your GitHub repo to Render
2. Render auto-detects `render.yaml` and creates:
   - **kalshi-dashboard**: Web service (FastAPI + UI)
   - **kalshi-trader**: Background worker (continuous bot)
3. Add environment variables in Render dashboard

### Docker

```bash
docker build -t kalshi-bot .
docker run -p 8000:8000 --env-file .env kalshi-bot
```

## Architecture

The bot follows a 6-step workflow:

1. **Fetch Events** - Get top events by 24h volume from Kalshi
2. **Process Markets** - Select top N markets per event
3. **Research Events** - Parallel AI research on events
4. **Extract Probabilities** - GPT-4o structured probability extraction
5. **Generate Decisions** - Risk-adjusted betting decisions with Kelly sizing
6. **Execute Trades** - Place orders via Kalshi API

### Risk-Adjusted Trading

- **R-score (z-score)**: `(research_prob - market_price) / sqrt(p*(1-p))`
- **Kelly fraction**: Position sizing with fractional Kelly
- **Portfolio selection**: Top N positions by R-score

## Development

```bash
# Install dev dependencies
uv sync --group dev

# Run tests
uv run pytest

# Format code
uv run black .
uv run isort .

# Lint
uv run flake8 .
```

## Disclaimer

**This software is for educational and research purposes only. Trading prediction markets involves significant financial risk. You may lose some or all of your invested capital. Use at your own risk.**

## License

MIT License - See LICENSE file for details.
