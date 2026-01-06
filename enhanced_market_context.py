"""
Enhanced Market Context Extraction for Higher Returns

This module enriches market data with full event context (rules, strike thresholds,
settlement sources) to provide GPT-4o with market-specific inputs for better
probability estimates.

Key improvements over base implementation:
1. Full event enrichment with rules_primary, rules_secondary, settlement_source
2. Strike thresholds (floor_strike, cap_strike, strike_type)
3. Caching layer for event data (reduces API calls)
4. Series-based filtering for relevant market categories
"""

import asyncio
import hashlib
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger


@dataclass
class EventContext:
    """Full event context for AI research prompts."""

    event_ticker: str
    title: str
    subtitle: str = ""
    category: str = ""

    # Settlement rules - CRITICAL for AI understanding
    rules_primary: str = ""
    rules_secondary: str = ""
    settlement_source: str = ""

    # Strike configuration
    strike_type: str = ""  # e.g., "greater", "less", "range"
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    strike_date: str = ""
    strike_period: str = ""

    # Market structure
    mutually_exclusive: bool = False
    series_ticker: str = ""

    # Metadata
    fetched_at: float = field(default_factory=time.time)


@dataclass
class MarketContext:
    """Full market context for AI research prompts."""

    ticker: str
    title: str
    subtitle: str = ""

    # Settlement specifics
    yes_sub_title: str = ""  # What YES means
    no_sub_title: str = ""   # What NO means

    # Strike thresholds for this specific market
    floor_strike: Optional[float] = None
    cap_strike: Optional[float] = None
    custom_strike: Optional[float] = None

    # Market rules (if different from event)
    rules_primary: str = ""
    rules_secondary: str = ""

    # Timing
    open_time: str = ""
    close_time: str = ""
    expiration_time: str = ""

    # Volume metrics
    volume: int = 0
    volume_24h: int = 0
    open_interest: int = 0
    liquidity: int = 0


class EventCache:
    """TTL-based cache for event details to reduce API calls."""

    def __init__(self, ttl_seconds: float = 300.0):
        self._cache: Dict[str, Tuple[EventContext, float]] = {}
        self._ttl = ttl_seconds

    def get(self, event_ticker: str) -> Optional[EventContext]:
        """Get cached event if not expired."""
        if event_ticker not in self._cache:
            return None

        context, cached_at = self._cache[event_ticker]
        if time.time() - cached_at > self._ttl:
            del self._cache[event_ticker]
            return None

        return context

    def set(self, event_ticker: str, context: EventContext):
        """Cache event context."""
        self._cache[event_ticker] = (context, time.time())

    def clear(self):
        """Clear all cached events."""
        self._cache.clear()

    def stats(self) -> Dict[str, Any]:
        """Return cache statistics."""
        now = time.time()
        valid_count = sum(
            1 for _, (_, cached_at) in self._cache.items()
            if now - cached_at <= self._ttl
        )
        return {
            "total_cached": len(self._cache),
            "valid_entries": valid_count,
            "ttl_seconds": self._ttl,
        }


# Global cache instance
_event_cache = EventCache(ttl_seconds=300.0)


def get_event_cache() -> EventCache:
    """Get the global event cache."""
    return _event_cache


async def fetch_event_details(
    client,  # KalshiClient instance
    event_ticker: str,
    use_cache: bool = True,
) -> Optional[EventContext]:
    """
    Fetch full event details including rules and settlement info.

    Args:
        client: KalshiClient instance
        event_ticker: Event ticker to fetch
        use_cache: Whether to use cached data if available

    Returns:
        EventContext with full details, or None on error
    """
    # Check cache first
    if use_cache:
        cached = _event_cache.get(event_ticker)
        if cached:
            logger.debug(f"Using cached event context for {event_ticker}")
            return cached

    try:
        # Fetch from API
        headers = await client._get_headers("GET", f"/trade-api/v2/events/{event_ticker}")
        response = await client.client.get(
            f"/trade-api/v2/events/{event_ticker}",
            headers=headers
        )
        response.raise_for_status()

        data = response.json()
        event = data.get("event", {})

        context = EventContext(
            event_ticker=event.get("event_ticker", event_ticker),
            title=event.get("title", ""),
            subtitle=event.get("sub_title", ""),
            category=event.get("category", ""),
            rules_primary=event.get("rules_primary", ""),
            rules_secondary=event.get("rules_secondary", ""),
            settlement_source=event.get("settlement_source_url", ""),
            strike_type=event.get("strike_type", ""),
            floor_strike=event.get("floor_strike"),
            cap_strike=event.get("cap_strike"),
            strike_date=event.get("strike_date", ""),
            strike_period=event.get("strike_period", ""),
            mutually_exclusive=event.get("mutually_exclusive", False),
            series_ticker=event.get("series_ticker", ""),
        )

        # Cache the result
        _event_cache.set(event_ticker, context)

        logger.info(f"Fetched event context for {event_ticker} (rules: {len(context.rules_primary)} chars)")
        return context

    except Exception as e:
        logger.error(f"Error fetching event details for {event_ticker}: {e}")
        return None


def extract_market_context(market: Dict[str, Any]) -> MarketContext:
    """
    Extract full market context from API response.

    Args:
        market: Market dict from Kalshi API

    Returns:
        MarketContext with full details
    """
    return MarketContext(
        ticker=market.get("ticker", ""),
        title=market.get("title", ""),
        subtitle=market.get("subtitle", ""),
        yes_sub_title=market.get("yes_sub_title", ""),
        no_sub_title=market.get("no_sub_title", ""),
        floor_strike=market.get("floor_strike"),
        cap_strike=market.get("cap_strike"),
        custom_strike=market.get("custom_strike"),
        rules_primary=market.get("rules_primary", ""),
        rules_secondary=market.get("rules_secondary", ""),
        open_time=market.get("open_time", ""),
        close_time=market.get("close_time", ""),
        expiration_time=market.get("expiration_time", ""),
        volume=market.get("volume", 0),
        volume_24h=market.get("volume_24h", 0),
        open_interest=market.get("open_interest", 0),
        liquidity=market.get("liquidity", 0),
    )


def format_event_context_for_prompt(event_ctx: EventContext) -> str:
    """
    Format event context into a string for AI prompts.

    Returns a structured string with all relevant event details
    that helps GPT-4o make market-specific predictions.
    """
    parts = [
        f"EVENT: {event_ctx.title}",
    ]

    if event_ctx.subtitle:
        parts.append(f"Description: {event_ctx.subtitle}")

    if event_ctx.category:
        parts.append(f"Category: {event_ctx.category}")

    # Strike configuration
    strike_info = []
    if event_ctx.strike_type:
        strike_info.append(f"Type: {event_ctx.strike_type}")
    if event_ctx.floor_strike is not None:
        strike_info.append(f"Floor: {event_ctx.floor_strike}")
    if event_ctx.cap_strike is not None:
        strike_info.append(f"Cap: {event_ctx.cap_strike}")
    if event_ctx.strike_date:
        strike_info.append(f"Strike Date: {event_ctx.strike_date}")

    if strike_info:
        parts.append(f"Strike Configuration: {', '.join(strike_info)}")

    # Settlement rules - CRITICAL for accurate predictions
    if event_ctx.rules_primary:
        # Truncate if too long but keep first 500 chars
        rules = event_ctx.rules_primary
        if len(rules) > 500:
            rules = rules[:500] + "..."
        parts.append(f"PRIMARY SETTLEMENT RULES: {rules}")

    if event_ctx.rules_secondary:
        rules = event_ctx.rules_secondary
        if len(rules) > 300:
            rules = rules[:300] + "..."
        parts.append(f"SECONDARY RULES: {rules}")

    if event_ctx.settlement_source:
        parts.append(f"Settlement Source: {event_ctx.settlement_source}")

    if event_ctx.mutually_exclusive:
        parts.append("NOTE: Markets are MUTUALLY EXCLUSIVE (probabilities should sum to ~100%)")

    return "\n".join(parts)


def format_market_context_for_prompt(market_ctx: MarketContext) -> str:
    """
    Format market context into a string for AI prompts.

    Returns a structured string with market-specific details
    including what YES/NO outcomes mean.
    """
    parts = [
        f"MARKET: {market_ctx.title} (Ticker: {market_ctx.ticker})",
    ]

    if market_ctx.subtitle:
        parts.append(f"  Details: {market_ctx.subtitle}")

    # YES/NO outcome descriptions - CRITICAL for understanding the bet
    if market_ctx.yes_sub_title:
        parts.append(f"  YES means: {market_ctx.yes_sub_title}")
    if market_ctx.no_sub_title:
        parts.append(f"  NO means: {market_ctx.no_sub_title}")

    # Strike thresholds for this specific market
    if market_ctx.custom_strike is not None:
        parts.append(f"  Strike threshold: {market_ctx.custom_strike}")
    elif market_ctx.floor_strike is not None or market_ctx.cap_strike is not None:
        threshold = f"Floor: {market_ctx.floor_strike}" if market_ctx.floor_strike else ""
        if market_ctx.cap_strike:
            threshold += f", Cap: {market_ctx.cap_strike}" if threshold else f"Cap: {market_ctx.cap_strike}"
        parts.append(f"  Strike range: {threshold}")

    # Market-specific rules if different from event
    if market_ctx.rules_primary:
        rules = market_ctx.rules_primary[:300] + "..." if len(market_ctx.rules_primary) > 300 else market_ctx.rules_primary
        parts.append(f"  Market Rules: {rules}")

    if market_ctx.close_time:
        parts.append(f"  Closes: {market_ctx.close_time}")

    return "\n".join(parts)


def build_enhanced_research_prompt(
    event_ctx: EventContext,
    markets: List[MarketContext],
    trending_context: str = "",
) -> str:
    """
    Build an enhanced research prompt with full context.

    This prompt includes:
    - Full event rules and settlement criteria
    - Per-market strike thresholds and YES/NO definitions
    - Settlement source for verification
    - Trending news context (if available)

    Args:
        event_ctx: Full event context
        markets: List of market contexts
        trending_context: Optional news signals

    Returns:
        Complete prompt string for GPT-4o research
    """
    sections = []

    # Event context
    sections.append("=" * 60)
    sections.append("EVENT INFORMATION")
    sections.append("=" * 60)
    sections.append(format_event_context_for_prompt(event_ctx))
    sections.append("")

    # Markets to analyze
    sections.append("=" * 60)
    sections.append(f"MARKETS TO ANALYZE ({len(markets)} total)")
    sections.append("=" * 60)

    for i, market in enumerate(markets, 1):
        sections.append(f"\n--- Market {i} ---")
        sections.append(format_market_context_for_prompt(market))

    sections.append("")

    # Trending context if available
    if trending_context:
        sections.append("=" * 60)
        sections.append("SUPPLEMENTARY NEWS SIGNALS (for context only)")
        sections.append("=" * 60)
        sections.append("""
IMPORTANT: The following headlines are provided as supplementary context only.
Do NOT treat them as authoritative. Headlines may be incomplete or biased.
""")
        sections.append(trending_context)
        sections.append("")

    # Analysis requirements
    sections.append("=" * 60)
    sections.append("ANALYSIS REQUIREMENTS")
    sections.append("=" * 60)
    sections.append("""
For EACH market listed above, provide:

1. **Market Ticker**: Always include the exact ticker (e.g., KXMARKET-123)

2. **Probability Estimate**: 0-100% for YES outcome
   - Use the settlement rules above to determine what triggers YES vs NO
   - Consider the strike thresholds for range-based markets

3. **Confidence Level**: 1-10 scale
   - Higher confidence for well-defined outcomes with clear data
   - Lower confidence for subjective or data-scarce predictions

4. **Key Reasoning**: 2-3 sentences covering:
   - What evidence supports/opposes this probability
   - How the settlement rules affect the outcome determination
   - Any time-sensitive factors given the close date

5. **Base Rate Check**: Compare to historical rates for similar events

FORMAT EXAMPLE:
"KXMARKET-123: 65% probability (confidence: 7/10)
Settlement triggers at [threshold from rules]. Current data suggests [evidence].
Base rate for similar events: ~55%. Adjusting +10% due to [specific factor]."
""")

    return "\n".join(sections)


# =============================================================================
# SERIES-BASED FILTERING FOR RELEVANT MARKETS
# =============================================================================

# Categories where news/sentiment analysis is most valuable
HIGH_SIGNAL_CATEGORIES = [
    "Economics",
    "Climate",
    "Politics",
    "Technology",
    "Finance",
    "Energy",
    "Cryptocurrency",
]

# Series tickers known to have high-quality, researchable markets
PREFERRED_SERIES = [
    # Economic indicators
    "KXJOBLESS",     # Jobless claims
    "KXCPI",         # Consumer Price Index
    "KXGDP",         # GDP
    "KXUNRATE",      # Unemployment rate
    "KXFED",         # Fed decisions

    # Weather/Climate
    "KXHIGHNY",      # NYC temperature
    "KXHIGHLA",      # LA temperature
    "KXHURRICANE",   # Hurricane events

    # Tech/Markets
    "KXBTC",         # Bitcoin price
    "KXETH",         # Ethereum price
    "KXSP500",       # S&P 500
]


def is_high_signal_market(event: Dict[str, Any]) -> bool:
    """
    Check if a market is in a high-signal category for news research.

    High-signal markets benefit most from news sentiment analysis.
    """
    category = event.get("category", "").lower()
    series_ticker = event.get("series_ticker", "")

    # Check category
    for cat in HIGH_SIGNAL_CATEGORIES:
        if cat.lower() in category:
            return True

    # Check preferred series
    if series_ticker in PREFERRED_SERIES:
        return True

    return False


def filter_markets_by_signal_value(
    events: List[Dict[str, Any]],
    prioritize_high_signal: bool = True,
) -> List[Dict[str, Any]]:
    """
    Filter and sort markets by their signal analysis value.

    Args:
        events: List of events from Kalshi API
        prioritize_high_signal: If True, sort high-signal markets first

    Returns:
        Sorted list with high-signal markets prioritized
    """
    if not prioritize_high_signal:
        return events

    high_signal = []
    low_signal = []

    for event in events:
        if is_high_signal_market(event):
            event["_high_signal"] = True
            high_signal.append(event)
        else:
            event["_high_signal"] = False
            low_signal.append(event)

    logger.info(
        f"Market filtering: {len(high_signal)} high-signal, {len(low_signal)} standard"
    )

    # Return high-signal first, then remaining by volume
    return high_signal + low_signal
