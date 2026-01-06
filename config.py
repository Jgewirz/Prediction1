"""
Configuration management for the simple trading bot.
"""

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load environment variables
load_dotenv()


class KalshiConfig(BaseModel):
    """Kalshi API configuration."""

    api_key: str = Field(..., description="Kalshi API key")
    private_key: str = Field(..., description="Kalshi private key (PEM format)")
    use_demo: bool = Field(default=True, description="Use demo environment")

    @property
    def base_url(self) -> str:
        """Get the appropriate base URL based on environment."""
        if self.use_demo:
            return "https://demo-api.kalshi.co"
        return "https://api.elections.kalshi.com"

    @field_validator("private_key", mode="before")
    @classmethod
    def validate_private_key(cls, v: str) -> str:
        """Validate and format private key."""
        if not v or v == "your_kalshi_private_key_here":
            raise ValueError(
                "KALSHI_PRIVATE_KEY is required. Please set it in your .env file."
            )

        # Strip surrounding quotes (common when copying from env files to Render)
        v = v.strip()
        if (v.startswith('"') and v.endswith('"')) or (
            v.startswith("'") and v.endswith("'")
        ):
            v = v[1:-1]

        # Convert escaped newlines to actual newlines
        if "\\n" in v:
            v = v.replace("\\n", "\n")

        # If it looks like a file path, try to read it
        if not v.startswith("-----BEGIN") and (Path(v).exists() or v.endswith(".pem")):
            try:
                with open(v, "r") as f:
                    v = f.read()
            except Exception as e:
                raise ValueError(f"Could not read private key file '{v}': {e}")

        # Basic validation that it looks like a PEM key
        if not v.strip().startswith("-----BEGIN") or not v.strip().endswith("-----"):
            raise ValueError(
                "Private key must be in PEM format starting with '-----BEGIN' and ending with '-----'. "
                "Make sure to include \\n for line breaks in your .env file."
            )

        return v


class OctagonConfig(BaseModel):
    """Octagon Deep Research API configuration (optional - now uses OpenAI)."""

    api_key: str = Field(default="unused", description="Octagon API key (optional)")
    base_url: str = Field(
        default="https://api.octagon.ai", description="Octagon API base URL"
    )


class OpenAIConfig(BaseModel):
    """OpenAI API configuration."""

    api_key: str = Field(..., description="OpenAI API key")
    model: str = Field(default="gpt-4o", description="OpenAI model to use")

    @field_validator("api_key", mode="before")
    @classmethod
    def validate_api_key(cls, v: str) -> str:
        if not v or v == "your_openai_api_key_here":
            raise ValueError(
                "OPENAI_API_KEY is required. Please set it in your .env file."
            )
        return v


class DatabaseConfig(BaseModel):
    """Database configuration for historical tracking (SQLite or PostgreSQL)."""

    # Database type selection
    db_type: str = Field(
        default="postgres", description="Database type: 'sqlite' or 'postgres'"
    )

    # SQLite settings
    db_path: str = Field(
        default="trading_bot.db", description="SQLite database file path"
    )

    # PostgreSQL settings (Neon) - credentials must be provided via environment variables
    pg_host: str = Field(default="", description="PostgreSQL host (set via PGHOST)")
    pg_database: str = Field(default="neondb", description="PostgreSQL database name")
    pg_user: str = Field(default="", description="PostgreSQL username (set via PGUSER)")
    pg_password: str = Field(
        default="", description="PostgreSQL password (set via PGPASSWORD)"
    )
    pg_port: int = Field(default=5432, description="PostgreSQL port")
    pg_ssl: str = Field(default="require", description="PostgreSQL SSL mode")

    # Common settings
    enable_db: bool = Field(default=True, description="Enable database storage")
    save_to_csv: bool = Field(
        default=True, description="Also save to CSV files (backup)"
    )
    migrate_on_startup: bool = Field(
        default=True, description="Run schema migrations on startup"
    )
    auto_reconcile: bool = Field(
        default=True, description="Auto-reconcile outcomes after bot run"
    )


class SchedulerConfig(BaseModel):
    """Scheduler configuration for continuous operation."""

    trading_interval_minutes: int = Field(
        default=15, ge=1, le=60, description="Minutes between trading runs"
    )
    reconciliation_interval_minutes: int = Field(
        default=60, ge=15, le=240, description="Minutes between reconciliation"
    )
    health_check_interval_minutes: int = Field(
        default=5, ge=1, le=15, description="Minutes between health checks"
    )
    max_consecutive_failures: int = Field(
        default=5, ge=1, le=20, description="Max consecutive failures before pausing"
    )
    startup_delay_seconds: int = Field(
        default=10, ge=0, le=60, description="Delay before first run"
    )


class TrendRadarConfig(BaseModel):
    """TrendRadar news intelligence integration configuration."""

    enabled: bool = Field(
        default=True, description="Enable TrendRadar integration for news signals"
    )
    base_url: str = Field(
        default="http://localhost:3333", description="TrendRadar MCP server URL"
    )
    timeout: float = Field(
        default=10.0,
        description="Request timeout in seconds (reduced from 30s for faster failures)",
    )

    # Retry settings
    max_retries: int = Field(
        default=2, ge=0, le=5, description="Max retry attempts for failed requests"
    )
    retry_backoff_base: float = Field(
        default=1.0,
        ge=0.5,
        le=5.0,
        description="Base backoff seconds for exponential retry",
    )

    # Circuit breaker settings
    circuit_failure_threshold: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Consecutive failures before opening circuit",
    )
    circuit_reset_seconds: int = Field(
        default=300, ge=60, le=900, description="Seconds before circuit breaker resets"
    )

    # Signal influence settings
    max_confidence_boost: float = Field(
        default=0.30,
        ge=0.0,
        le=0.5,
        description="Maximum confidence boost from aligned signals",
    )
    strong_signal_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Threshold for strong signal classification",
    )
    min_source_count: int = Field(
        default=2, ge=1, description="Minimum news sources for valid signal"
    )

    # Position sizing adjustments
    aligned_signal_kelly_multiplier: float = Field(
        default=1.25,
        ge=1.0,
        le=2.0,
        description="Kelly multiplier for strong aligned signals",
    )

    # Skip override settings
    enable_skip_override: bool = Field(
        default=True, description="Allow strong signals to override skip decisions"
    )
    skip_override_min_strength: float = Field(
        default=0.8,
        ge=0.5,
        le=1.0,
        description="Minimum signal strength for skip override",
    )
    skip_override_min_sources: int = Field(
        default=5, ge=1, description="Minimum sources for skip override"
    )

    # Ablation mode for A/B testing ROI lift
    ablation_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Fraction of events to ignore signal influence for A/B testing (0=disabled)",
    )

    # Cache settings
    cache_ttl_seconds: float = Field(
        default=300.0, ge=60, le=1800, description="Signal cache TTL in seconds"
    )


class EnhancedContextConfig(BaseModel):
    """Enhanced market context configuration for improved AI research."""

    enabled: bool = Field(
        default=True, description="Enable enhanced context fetching (rules, settlement)"
    )
    event_cache_ttl_seconds: float = Field(
        default=300.0, ge=60, le=1800, description="Event context cache TTL in seconds"
    )
    fetch_settlement_rules: bool = Field(
        default=True, description="Fetch rules_primary and rules_secondary from API"
    )
    fetch_strike_thresholds: bool = Field(
        default=True, description="Fetch floor_strike, cap_strike, strike_type"
    )
    include_yes_no_descriptions: bool = Field(
        default=True, description="Include yes_sub_title and no_sub_title in prompts"
    )
    prioritize_high_signal_markets: bool = Field(
        default=True, description="Prioritize markets where news analysis adds value"
    )
    max_rules_chars: int = Field(
        default=500, ge=100, le=2000, description="Max chars for rules in prompt"
    )


class StopLossConfig(BaseModel):
    """Stop-loss and take-profit configuration for automated position exits."""

    enabled: bool = Field(
        default=True, description="Enable stop-loss/take-profit monitoring"
    )

    # Default thresholds (can be overridden per-position)
    default_stop_loss_pct: float = Field(
        default=0.15,
        ge=0.01,
        le=0.50,
        description="Default stop-loss percentage (0.15 = exit if down 15%)",
    )
    default_take_profit_pct: float = Field(
        default=0.30,
        ge=0.05,
        le=1.0,
        description="Default take-profit percentage (0.30 = exit if up 30%)",
    )

    # Trailing stop (optional)
    trailing_stop_enabled: bool = Field(
        default=False, description="Enable trailing stop-loss"
    )
    trailing_stop_pct: float = Field(
        default=0.10,
        ge=0.01,
        le=0.30,
        description="Trail stop this percentage below high water mark",
    )

    # Monitoring settings
    monitor_interval_seconds: int = Field(
        default=30, ge=5, le=300, description="Seconds between position checks"
    )
    price_staleness_seconds: int = Field(
        default=60, ge=10, le=300, description="Re-fetch price if older than this"
    )

    # Execution settings
    use_market_orders: bool = Field(
        default=False,
        description="Use market orders for exits (True) or limit at bid (False)",
    )
    slippage_tolerance_pct: float = Field(
        default=0.02,
        ge=0.0,
        le=0.10,
        description="Maximum slippage tolerance for limit orders",
    )
    retry_failed_exits: bool = Field(
        default=True, description="Retry failed exit orders"
    )
    max_exit_retries: int = Field(
        default=3, ge=1, le=10, description="Maximum retry attempts for exit orders"
    )

    # Risk controls
    max_simultaneous_exits: int = Field(
        default=5, ge=1, le=20, description="Maximum positions to exit simultaneously"
    )
    min_position_age_seconds: int = Field(
        default=300,
        ge=0,
        le=3600,
        description="Don't exit positions younger than this (anti-churn)",
    )

    # Alerting
    alert_on_trigger: bool = Field(
        default=True, description="Send alert when SL/TP triggers"
    )
    alert_on_near_trigger_pct: float = Field(
        default=0.05,
        ge=0.0,
        le=0.20,
        description="Alert when within this percentage of trigger",
    )


class EarlyEntryConfig(BaseModel):
    """Early entry event selection strategy - prioritize new, low-volume, unique opportunities.

    Designed to match kalshi.com/?live=new&liveEventType=unique behavior.
    """

    enabled: bool = Field(
        default=True, description="Enable early entry selection strategy"
    )

    # Market age filters
    max_market_age_hours: float = Field(
        default=72.0,
        ge=1.0,
        le=168.0,
        description="Maximum market age in hours (increased for more opportunities)",
    )
    min_time_remaining_hours: float = Field(
        default=12.0,
        ge=1.0,
        le=720.0,
        description="Minimum hours until market close (reduced for more opportunities)",
    )

    # Volume filters
    max_volume_24h: int = Field(
        default=50000,
        ge=100,
        le=1000000,
        description="Maximum 24h volume to consider (increased for liquidity)",
    )

    # NEW/UNIQUE event filters (like kalshi.com/?live=new&liveEventType=unique)
    favor_unique_events: bool = Field(
        default=True, description="Prioritize unique/non-recurring events over series"
    )
    favor_new_markets: bool = Field(
        default=True, description="Prioritize recently created markets"
    )
    new_market_hours: float = Field(
        default=48.0,
        ge=1.0,
        le=168.0,
        description="Markets created within this window are 'new'",
    )
    unique_event_bonus: float = Field(
        default=0.35,
        ge=0.0,
        le=0.5,
        description="Score bonus for unique (non-series) events",
    )
    new_market_bonus: float = Field(
        default=0.20,
        ge=0.0,
        le=0.5,
        description="Score bonus for newly created markets",
    )

    # Scoring weights (must sum to 1.0 for normalized scoring)
    recency_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for market recency (newer = higher score)",
    )
    low_volume_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for low volume (lower volume = higher score)",
    )
    time_remaining_weight: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Weight for time remaining (more time = higher score)",
    )
    category_weight: float = Field(
        default=0.3, ge=0.0, le=1.0, description="Weight for category alpha potential"
    )

    # Category filtering for ALPHA POTENTIAL
    # HIGH ALPHA: Politics, Economics, Crypto, Tech, Legal - research can find edge
    # LOW ALPHA: Weather - forecasts are too accurate, no information asymmetry
    exclude_categories: list = Field(
        default=["Weather"], description="Categories to EXCLUDE (low alpha potential)"
    )
    exclude_series_prefixes: list = Field(
        default=["KXLOWT", "KXHIGHT", "KXPRECIP", "KXSNOW", "KXWIND"],
        description="Series ticker prefixes to EXCLUDE (weather markets have no alpha)",
    )
    high_alpha_categories: list = Field(
        default=[
            "Politics",
            "Economics",
            "Financials",
            "Fed",
            "Crypto",
            "Tech",
            "Legal",
            "Congress",
            "Supreme Court",
            "Elections",
        ],
        description="Categories with HIGH alpha potential (research can find edge)",
    )
    high_alpha_series_prefixes: list = Field(
        default=[
            "KXFED",
            "KXINFL",
            "KXGDP",
            "KXBTC",
            "KXETH",
            "KXTRUMP",
            "KXBIDEN",
            "KXCONGRESS",
            "KXSCOTUS",
            "KXELECTION",
            "KXPRES",
            "KXSENATE",
            "KXHOUSE",
        ],
        description="Series ticker prefixes with HIGH alpha potential",
    )
    high_alpha_bonus: float = Field(
        default=0.40,
        ge=0.0,
        le=0.6,
        description="Score bonus for high-alpha categories",
    )


def _clean_env_value(value: str) -> str:
    """Clean environment variable value by removing inline comments."""
    # Split on '#' and take the first part, then strip whitespace
    return value.split("#")[0].strip()


class BotConfig(BaseSettings):
    """Main bot configuration."""

    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="allow"
    )

    # API configurations
    kalshi: KalshiConfig = Field(..., description="Kalshi configuration")
    octagon: OctagonConfig = Field(..., description="Octagon configuration")
    openai: OpenAIConfig = Field(..., description="OpenAI configuration")
    database: DatabaseConfig = Field(
        default_factory=DatabaseConfig, description="Database configuration"
    )
    trendradar: TrendRadarConfig = Field(
        default_factory=TrendRadarConfig,
        description="TrendRadar news intelligence configuration",
    )
    scheduler: SchedulerConfig = Field(
        default_factory=SchedulerConfig,
        description="Scheduler configuration for continuous operation",
    )
    stop_loss: StopLossConfig = Field(
        default_factory=StopLossConfig,
        description="Stop-loss and take-profit configuration",
    )
    early_entry: EarlyEntryConfig = Field(
        default_factory=EarlyEntryConfig,
        description="Early entry event selection strategy",
    )

    # Bot settings
    dry_run: bool = Field(
        default=True, description="Run in dry-run mode (overridden by CLI)"
    )
    max_bet_amount: float = Field(
        default=300.0,
        description="Maximum bet amount per market (increased for aggressive growth)",
    )
    max_events_to_analyze: int = Field(
        default=50, description="Number of top events to analyze by volume_24h"
    )
    research_batch_size: int = Field(
        default=10, description="Number of parallel deep research requests to batch"
    )
    research_timeout_seconds: int = Field(
        default=900, description="Per-event research timeout in seconds"
    )
    skip_existing_positions: bool = Field(
        default=True,
        description="Skip betting on markets where we already have positions",
    )
    minimum_time_remaining_hours: float = Field(
        default=1.0,
        description="Minimum hours remaining before event strike to consider it tradeable (only applied to events with strike_date)",
    )
    max_markets_per_event: int = Field(
        default=10,
        description="Maximum number of markets per event to analyze (selects top N markets by volume)",
    )
    # Legacy alpha threshold (deprecated - use R-score filtering instead)
    minimum_alpha_threshold: float = Field(
        default=2.0,
        description="DEPRECATED: Use z_threshold and enable_r_score_filtering instead",
    )

    # Risk-adjusted trading parameters (now the default system)
    z_threshold: float = Field(
        default=0.8,
        description="Minimum R-score (z-score) threshold for placing bets - lowered for more opportunities",
    )
    enable_r_score_filtering: bool = Field(
        default=True, description="DEPRECATED: R-score filtering is now always enabled"
    )

    # Kelly criterion and position sizing
    enable_kelly_sizing: bool = Field(
        default=True, description="Use Kelly criterion for position sizing"
    )
    kelly_fraction: float = Field(
        default=0.8,
        ge=0.1,
        le=1.5,
        description="Fraction of Kelly to use (0.8 = near-full Kelly for max growth)",
    )
    max_kelly_bet_fraction: float = Field(
        default=0.25,
        ge=0.01,
        le=0.5,
        description="Maximum fraction of bankroll per bet (25%)",
    )
    bankroll: float = Field(
        default=1000.0, description="Total bankroll for Kelly sizing calculations"
    )

    # Portfolio management
    max_portfolio_positions: int = Field(
        default=25, description="Maximum number of positions to hold simultaneously"
    )
    portfolio_selection_method: str = Field(
        default="top_r_scores",
        description="Method for portfolio selection: 'top_r_scores', 'diversified', or 'legacy'",
    )

    # Hedging settings
    enable_hedging: bool = Field(
        default=False,
        description="Disabled - hedging reduces ROI by ~25% on winning bets",
    )
    hedge_ratio: float = Field(
        default=0.25,
        ge=0,
        le=0.5,
        description="Default hedge ratio (0.25 = hedge 25% of main bet)",
    )
    min_confidence_for_hedging: float = Field(
        default=0.6,
        ge=0,
        le=1,
        description="Only hedge bets with confidence below this threshold",
    )
    max_hedge_amount: float = Field(
        default=50.0, description="Maximum hedge amount per bet"
    )

    # =========================================================================
    # RISK MANAGEMENT - Daily Loss Limit & Kill Switch (CRITICAL for live trading)
    # =========================================================================
    max_daily_loss: float = Field(
        default=100.0,
        ge=0,
        description="Maximum daily loss before kill switch triggers ($)",
    )
    max_daily_loss_pct: float = Field(
        default=0.10,
        ge=0,
        le=0.5,
        description="Maximum daily loss as fraction of bankroll (10%)",
    )
    enable_kill_switch: bool = Field(
        default=True, description="Enable automatic trading halt on daily loss limit"
    )
    max_total_exposure: float = Field(
        default=500.0,
        ge=0,
        description="Maximum total notional exposure across all positions",
    )

    # Live trading safety
    require_live_confirmation: bool = Field(
        default=True, description="Require explicit confirmation for live trading"
    )

    def __init__(self, **data):
        # Build nested configs from environment variables

        # Handle private key from file if specified
        private_key = os.getenv("KALSHI_PRIVATE_KEY", "")
        private_key_file = os.getenv("KALSHI_PRIVATE_KEY_FILE", "")

        if private_key_file and not private_key:
            private_key = private_key_file  # Will be processed by validator

        kalshi_config = KalshiConfig(
            api_key=os.getenv("KALSHI_API_KEY", ""),
            private_key=private_key,
            use_demo=os.getenv("KALSHI_USE_DEMO", "true").lower() == "true",
        )

        octagon_config = OctagonConfig(
            api_key=os.getenv("OCTAGON_API_KEY", ""),
            base_url=os.getenv("OCTAGON_BASE_URL", "https://api.octagon.ai"),
        )

        openai_config = OpenAIConfig(
            api_key=os.getenv("OPENAI_API_KEY", ""),
            model=os.getenv("OPENAI_MODEL", "gpt-4o"),
        )

        database_config = DatabaseConfig(
            db_type=os.getenv("DB_TYPE", "postgres"),  # Default to PostgreSQL
            db_path=os.getenv("DB_PATH", "trading_bot.db"),
            pg_host=os.getenv("PGHOST", ""),  # Required: set via PGHOST env var
            pg_database=os.getenv("PGDATABASE", "neondb"),
            pg_user=os.getenv("PGUSER", ""),  # Required: set via PGUSER env var
            pg_password=os.getenv(
                "PGPASSWORD", ""
            ),  # Required: set via PGPASSWORD env var
            pg_port=int(os.getenv("PGPORT", "5432")),
            pg_ssl=os.getenv("PGSSLMODE", "require"),
            enable_db=_clean_env_value(os.getenv("ENABLE_DB", "true")).lower()
            == "true",
            save_to_csv=_clean_env_value(os.getenv("SAVE_TO_CSV", "true")).lower()
            == "true",
            migrate_on_startup=_clean_env_value(
                os.getenv("DB_MIGRATE_ON_STARTUP", "true")
            ).lower()
            == "true",
            auto_reconcile=_clean_env_value(os.getenv("AUTO_RECONCILE", "true")).lower()
            == "true",
        )

        trendradar_config = TrendRadarConfig(
            enabled=_clean_env_value(os.getenv("TRENDRADAR_ENABLED", "true")).lower()
            == "true",
            base_url=os.getenv("TRENDRADAR_URL", "http://localhost:3333"),
            timeout=float(_clean_env_value(os.getenv("TRENDRADAR_TIMEOUT", "10.0"))),
            # Retry settings
            max_retries=int(_clean_env_value(os.getenv("TRENDRADAR_MAX_RETRIES", "2"))),
            retry_backoff_base=float(
                _clean_env_value(os.getenv("TRENDRADAR_RETRY_BACKOFF", "1.0"))
            ),
            # Circuit breaker settings
            circuit_failure_threshold=int(
                _clean_env_value(os.getenv("TRENDRADAR_CIRCUIT_THRESHOLD", "3"))
            ),
            circuit_reset_seconds=int(
                _clean_env_value(os.getenv("TRENDRADAR_CIRCUIT_RESET", "300"))
            ),
            # Signal influence settings
            max_confidence_boost=float(
                _clean_env_value(os.getenv("TRENDRADAR_MAX_BOOST", "0.30"))
            ),
            strong_signal_threshold=float(
                _clean_env_value(os.getenv("TRENDRADAR_STRONG_THRESHOLD", "0.7"))
            ),
            min_source_count=int(
                _clean_env_value(os.getenv("TRENDRADAR_MIN_SOURCES", "2"))
            ),
            aligned_signal_kelly_multiplier=float(
                _clean_env_value(os.getenv("TRENDRADAR_KELLY_MULTIPLIER", "1.25"))
            ),
            enable_skip_override=_clean_env_value(
                os.getenv("TRENDRADAR_SKIP_OVERRIDE", "true")
            ).lower()
            == "true",
            skip_override_min_strength=float(
                _clean_env_value(os.getenv("TRENDRADAR_SKIP_OVERRIDE_STRENGTH", "0.8"))
            ),
            skip_override_min_sources=int(
                _clean_env_value(os.getenv("TRENDRADAR_SKIP_OVERRIDE_SOURCES", "5"))
            ),
            # Ablation testing and caching
            ablation_rate=float(
                _clean_env_value(os.getenv("TRENDRADAR_ABLATION_RATE", "0.0"))
            ),
            cache_ttl_seconds=float(
                _clean_env_value(os.getenv("TRENDRADAR_CACHE_TTL", "300.0"))
            ),
        )

        scheduler_config = SchedulerConfig(
            trading_interval_minutes=int(
                _clean_env_value(os.getenv("SCHEDULER_TRADING_INTERVAL", "15"))
            ),
            reconciliation_interval_minutes=int(
                _clean_env_value(os.getenv("SCHEDULER_RECONCILIATION_INTERVAL", "60"))
            ),
            health_check_interval_minutes=int(
                _clean_env_value(os.getenv("SCHEDULER_HEALTH_CHECK_INTERVAL", "5"))
            ),
            max_consecutive_failures=int(
                _clean_env_value(os.getenv("SCHEDULER_MAX_FAILURES", "5"))
            ),
            startup_delay_seconds=int(
                _clean_env_value(os.getenv("SCHEDULER_STARTUP_DELAY", "10"))
            ),
        )

        stop_loss_config = StopLossConfig(
            enabled=_clean_env_value(os.getenv("SL_TP_ENABLED", "true")).lower()
            == "true",
            default_stop_loss_pct=float(
                _clean_env_value(os.getenv("DEFAULT_STOP_LOSS_PCT", "0.15"))
            ),
            default_take_profit_pct=float(
                _clean_env_value(os.getenv("DEFAULT_TAKE_PROFIT_PCT", "0.30"))
            ),
            trailing_stop_enabled=_clean_env_value(
                os.getenv("TRAILING_STOP_ENABLED", "false")
            ).lower()
            == "true",
            trailing_stop_pct=float(
                _clean_env_value(os.getenv("TRAILING_STOP_PCT", "0.10"))
            ),
            monitor_interval_seconds=int(
                _clean_env_value(os.getenv("MONITOR_INTERVAL_SECONDS", "30"))
            ),
            price_staleness_seconds=int(
                _clean_env_value(os.getenv("PRICE_STALENESS_SECONDS", "60"))
            ),
            use_market_orders=_clean_env_value(
                os.getenv("USE_MARKET_ORDERS", "false")
            ).lower()
            == "true",
            slippage_tolerance_pct=float(
                _clean_env_value(os.getenv("SLIPPAGE_TOLERANCE_PCT", "0.02"))
            ),
            retry_failed_exits=_clean_env_value(
                os.getenv("RETRY_FAILED_EXITS", "true")
            ).lower()
            == "true",
            max_exit_retries=int(_clean_env_value(os.getenv("MAX_EXIT_RETRIES", "3"))),
            max_simultaneous_exits=int(
                _clean_env_value(os.getenv("MAX_SIMULTANEOUS_EXITS", "5"))
            ),
            min_position_age_seconds=int(
                _clean_env_value(os.getenv("MIN_POSITION_AGE_SECONDS", "300"))
            ),
            alert_on_trigger=_clean_env_value(
                os.getenv("ALERT_ON_TRIGGER", "true")
            ).lower()
            == "true",
            alert_on_near_trigger_pct=float(
                _clean_env_value(os.getenv("ALERT_ON_NEAR_TRIGGER_PCT", "0.05"))
            ),
        )

        early_entry_config = EarlyEntryConfig(
            enabled=_clean_env_value(os.getenv("EARLY_ENTRY_ENABLED", "true")).lower()
            == "true",
            max_market_age_hours=float(
                _clean_env_value(os.getenv("EARLY_ENTRY_MAX_AGE_HOURS", "48.0"))
            ),
            min_time_remaining_hours=float(
                _clean_env_value(os.getenv("EARLY_ENTRY_MIN_TIME_HOURS", "24.0"))
            ),
            max_volume_24h=int(
                _clean_env_value(os.getenv("EARLY_ENTRY_MAX_VOLUME", "10000"))
            ),
            recency_weight=float(
                _clean_env_value(os.getenv("EARLY_ENTRY_RECENCY_WEIGHT", "0.4"))
            ),
            low_volume_weight=float(
                _clean_env_value(os.getenv("EARLY_ENTRY_VOLUME_WEIGHT", "0.3"))
            ),
            time_remaining_weight=float(
                _clean_env_value(os.getenv("EARLY_ENTRY_TIME_WEIGHT", "0.3"))
            ),
        )

        data.update(
            {
                "kalshi": kalshi_config,
                "octagon": octagon_config,
                "openai": openai_config,
                "database": database_config,
                "trendradar": trendradar_config,
                "scheduler": scheduler_config,
                "stop_loss": stop_loss_config,
                "early_entry": early_entry_config,
                "dry_run": True,  # Default to dry run, overridden by CLI
                "max_bet_amount": float(
                    _clean_env_value(os.getenv("MAX_BET_AMOUNT", "100.0"))
                ),
                "max_events_to_analyze": int(
                    _clean_env_value(os.getenv("MAX_EVENTS_TO_ANALYZE", "50"))
                ),
                "research_batch_size": int(
                    _clean_env_value(os.getenv("RESEARCH_BATCH_SIZE", "10"))
                ),
                "research_timeout_seconds": int(
                    _clean_env_value(os.getenv("RESEARCH_TIMEOUT_SECONDS", "900"))
                ),
                "skip_existing_positions": _clean_env_value(
                    os.getenv("SKIP_EXISTING_POSITIONS", "true")
                ).lower()
                == "true",
                "minimum_time_remaining_hours": float(
                    _clean_env_value(os.getenv("MINIMUM_TIME_REMAINING_HOURS", "1.0"))
                ),
                "max_markets_per_event": int(
                    _clean_env_value(os.getenv("MAX_MARKETS_PER_EVENT", "10"))
                ),
                "minimum_alpha_threshold": float(
                    _clean_env_value(os.getenv("MINIMUM_ALPHA_THRESHOLD", "2.0"))
                ),
                # Risk-adjusted trading parameters
                "z_threshold": float(_clean_env_value(os.getenv("Z_THRESHOLD", "1.5"))),
                "enable_r_score_filtering": _clean_env_value(
                    os.getenv("ENABLE_R_SCORE_FILTERING", "true")
                ).lower()
                == "true",
                # Kelly criterion and position sizing
                "enable_kelly_sizing": _clean_env_value(
                    os.getenv("ENABLE_KELLY_SIZING", "true")
                ).lower()
                == "true",
                "kelly_fraction": float(
                    _clean_env_value(os.getenv("KELLY_FRACTION", "0.5"))
                ),
                "max_kelly_bet_fraction": float(
                    _clean_env_value(os.getenv("MAX_KELLY_BET_FRACTION", "0.1"))
                ),
                "bankroll": float(_clean_env_value(os.getenv("BANKROLL", "1000.0"))),
                # Portfolio management
                "max_portfolio_positions": int(
                    _clean_env_value(os.getenv("MAX_PORTFOLIO_POSITIONS", "10"))
                ),
                "portfolio_selection_method": os.getenv(
                    "PORTFOLIO_SELECTION_METHOD", "top_r_scores"
                ),
                # Hedging settings
                "enable_hedging": _clean_env_value(
                    os.getenv("ENABLE_HEDGING", "true")
                ).lower()
                == "true",
                "hedge_ratio": float(
                    _clean_env_value(os.getenv("HEDGE_RATIO", "0.25"))
                ),
                "min_confidence_for_hedging": float(
                    _clean_env_value(os.getenv("MIN_CONFIDENCE_FOR_HEDGING", "0.6"))
                ),
                "max_hedge_amount": float(
                    _clean_env_value(os.getenv("MAX_HEDGE_AMOUNT", "50.0"))
                ),
                # Risk management - daily loss limit and kill switch
                "max_daily_loss": float(
                    _clean_env_value(os.getenv("MAX_DAILY_LOSS", "100.0"))
                ),
                "max_daily_loss_pct": float(
                    _clean_env_value(os.getenv("MAX_DAILY_LOSS_PCT", "0.10"))
                ),
                "enable_kill_switch": _clean_env_value(
                    os.getenv("ENABLE_KILL_SWITCH", "true")
                ).lower()
                == "true",
                "max_total_exposure": float(
                    _clean_env_value(os.getenv("MAX_TOTAL_EXPOSURE", "500.0"))
                ),
                "require_live_confirmation": _clean_env_value(
                    os.getenv("REQUIRE_LIVE_CONFIRMATION", "true")
                ).lower()
                == "true",
            }
        )

        super().__init__(**data)


def load_config() -> BotConfig:
    """Load and validate configuration."""
    return BotConfig()
