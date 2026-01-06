"""
TrendRadar MCP HTTP Client for the Kalshi Trading Bot.
Connects to TrendRadar's FastMCP 2.0 HTTP server to fetch trending signals.

Production hardening features:
- Retries with exponential backoff + jitter
- Circuit breaker (fail-open after N failures)
- Per-event signal caching with TTL
- Fail-closed on influence (neutral if TrendRadar fails)

Signal quality enhancements (v2):
- Story de-duplication via aggregate_news
- Outlet tiering (Tier-1 sources weighted higher)
- Relevance scoring with OpenAI embeddings
- Dynamic baseline frequency normalization
- Single-lever probability adjustment (no double-counting)
"""

import asyncio
import hashlib
import json
import math
import os
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger

# ============================================================================
# OFFLINE_MODE: Deterministic mode for testing without external services
# Set OFFLINE_MODE=1 to enable
# ============================================================================
OFFLINE_MODE = os.getenv("OFFLINE_MODE", "0").lower() in ("1", "true", "yes")

if OFFLINE_MODE:
    logger.warning(
        "OFFLINE_MODE enabled: Using mock responses for TrendRadar and embeddings"
    )


# ============================================================================
# Outlet Tiering - Weight news sources by credibility
# ============================================================================
class OutletTier(Enum):
    """News outlet credibility tiers."""

    TIER1 = 1  # Major wire services, financial papers
    TIER2 = 2  # Major news networks
    TIER3 = 3  # Other sources


# Tier-1: Major wire services and financial papers (highest weight)
TIER1_OUTLETS = {
    "reuters",
    "reuters-business",
    "reuters-politics",
    "bloomberg",
    "bloomberg-markets",
    "wsj",
    "wsj-markets",
    "wsj-world",
    "ap",
    "ap-news",
    "ap-news-top",
    "ap-news-politics",
    "nyt",
    "nyt-business",
    "nyt-politics",
    "ft",
    "financial-times",
}

# Tier-2: Major news networks (medium weight)
TIER2_OUTLETS = {
    "cnbc",
    "cnbc-top",
    "cnbc-world",
    "marketwatch",
    "marketwatch-top",
    "bbc",
    "bbc-news",
    "cnn",
    "cnn-business",
    "fox",
    "fox-business",
}

# Tier weights for volume calculation
TIER_WEIGHTS = {OutletTier.TIER1: 1.0, OutletTier.TIER2: 0.7, OutletTier.TIER3: 0.4}


def get_outlet_tier(outlet_id: str) -> OutletTier:
    """Get the tier for a news outlet."""
    outlet_lower = outlet_id.lower().replace(" ", "-")
    if outlet_lower in TIER1_OUTLETS:
        return OutletTier.TIER1
    elif outlet_lower in TIER2_OUTLETS:
        return OutletTier.TIER2
    return OutletTier.TIER3


# ============================================================================
# Sentiment Analysis Keywords (with negation handling)
# ============================================================================
POSITIVE_KEYWORDS = {
    "gain",
    "rise",
    "up",
    "surge",
    "rally",
    "boost",
    "win",
    "success",
    "growth",
    "positive",
    "strong",
    "good",
    "best",
    "record",
    "high",
    "soar",
    "jump",
    "climb",
    "advance",
    "improve",
    "profit",
    "beat",
}

NEGATIVE_KEYWORDS = {
    "fall",
    "drop",
    "down",
    "crash",
    "decline",
    "lose",
    "loss",
    "fail",
    "weak",
    "negative",
    "bad",
    "worst",
    "low",
    "cut",
    "fear",
    "risk",
    "plunge",
    "sink",
    "tumble",
    "retreat",
    "slump",
    "miss",
    "warning",
}

NEGATION_WORDS = {
    "not",
    "no",
    "never",
    "fails",
    "unlikely",
    "avoids",
    "denies",
    "without",
    "lack",
    "neither",
    "nor",
    "refuse",
    "reject",
}

UNCERTAINTY_WORDS = {
    "may",
    "could",
    "might",
    "reportedly",
    "signals",
    "weighs",
    "considers",
    "possible",
    "potential",
    "expected",
    "likely",
    "suggests",
    "appears",
}


# ============================================================================
# Circuit Breaker Configuration
# ============================================================================
@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""

    failure_threshold: int = 3  # Failures before opening circuit
    recovery_timeout: float = 300.0  # 5 minutes before attempting recovery
    half_open_max_calls: int = 1  # Calls allowed in half-open state


@dataclass
class CircuitBreakerState:
    """Mutable state for circuit breaker."""

    failure_count: int = 0
    last_failure_time: float = 0.0
    state: str = "closed"  # closed, open, half_open
    half_open_calls: int = 0

    def record_failure(self, config: CircuitBreakerConfig) -> None:
        """Record a failure and potentially open the circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= config.failure_threshold:
            self.state = "open"
            logger.warning(
                f"Circuit breaker OPENED after {self.failure_count} failures"
            )

    def record_success(self) -> None:
        """Record a success and reset failure count."""
        self.failure_count = 0
        self.state = "closed"
        self.half_open_calls = 0

    def should_allow_request(self, config: CircuitBreakerConfig) -> bool:
        """Check if a request should be allowed through."""
        if self.state == "closed":
            return True
        elif self.state == "open":
            # Check if recovery timeout has passed
            if time.time() - self.last_failure_time >= config.recovery_timeout:
                self.state = "half_open"
                self.half_open_calls = 0
                logger.info("Circuit breaker transitioning to HALF-OPEN state")
                return True
            return False
        else:  # half_open
            if self.half_open_calls < config.half_open_max_calls:
                self.half_open_calls += 1
                return True
            return False


# ============================================================================
# Signal Cache
# ============================================================================
@dataclass
class CacheEntry:
    """A cached signal result with TTL."""

    signals: List["TrendingSignal"]
    timestamp: float
    ttl: float = 300.0  # 5 minutes default

    def is_valid(self) -> bool:
        return time.time() - self.timestamp < self.ttl


class SignalCache:
    """TTL-based cache for event signals to reduce variance and API calls."""

    def __init__(self, default_ttl: float = 300.0):
        self._cache: Dict[str, CacheEntry] = {}
        self.default_ttl = default_ttl

    def _cache_key(self, event_title: str, event_category: str) -> str:
        """Generate a stable cache key."""
        content = f"{event_title}|{event_category}".lower()
        return hashlib.md5(content.encode()).hexdigest()[:16]

    def get(
        self, event_title: str, event_category: str
    ) -> Optional[List["TrendingSignal"]]:
        """Get cached signals if valid."""
        key = self._cache_key(event_title, event_category)
        entry = self._cache.get(key)
        if entry and entry.is_valid():
            logger.debug(f"Cache HIT for event: {event_title[:30]}...")
            return entry.signals
        return None

    def set(
        self, event_title: str, event_category: str, signals: List["TrendingSignal"]
    ) -> None:
        """Cache signals for an event."""
        key = self._cache_key(event_title, event_category)
        self._cache[key] = CacheEntry(
            signals=signals, timestamp=time.time(), ttl=self.default_ttl
        )

    def clear(self) -> None:
        """Clear all cached entries."""
        self._cache.clear()


@dataclass
class TrendingSignal:
    """A trending signal from TrendRadar with quality metrics."""

    topic: str
    sentiment: str  # "positive", "negative", "neutral"
    strength: float  # 0.0 to 1.0 (calculated from quality metrics)
    source_count: int  # DEPRECATED: use unique_story_count instead

    # Headlines and sources
    sample_headlines: List[str] = field(default_factory=list)
    platforms: List[str] = field(default_factory=list)

    # NEW: De-duplication metrics (Phase 1)
    raw_article_count: int = 0  # Before de-duplication
    unique_story_count: int = 0  # After de-duplication
    unique_outlet_count: int = 0  # Distinct news sources
    tier1_outlet_count: int = 0  # Reuters, Bloomberg, WSJ, AP, etc.
    tier2_outlet_count: int = 0  # CNBC, MarketWatch, etc.
    story_cluster_sizes: List[int] = field(default_factory=list)  # Size of each cluster

    # NEW: Quality metrics (Phases 2-4)
    relevance: float = 0.0  # How relevant headlines are to event (0-1)
    novelty: float = 1.0  # How new the news is (0-1, decays)
    sentiment_confidence: float = 0.5  # Confidence in sentiment direction (0-1)
    neutral_ratio: float = 0.0  # Fraction of headlines that are neutral

    # NEW: Baseline comparison (Phase 3)
    baseline_frequency: float = 0.0  # Normal daily mention count for this entity
    current_frequency: float = 0.0  # Current mention count
    frequency_ratio: float = 1.0  # current / baseline (>1.5 is significant)

    @property
    def is_strong(self) -> bool:
        """Check if this is a strong signal (>= 0.7 strength)."""
        return self.strength >= 0.7

    @property
    def weighted_source_count(self) -> float:
        """Get tier-weighted source count."""
        return (
            self.tier1_outlet_count * TIER_WEIGHTS[OutletTier.TIER1]
            + self.tier2_outlet_count * TIER_WEIGHTS[OutletTier.TIER2]
            + max(
                0,
                self.unique_outlet_count
                - self.tier1_outlet_count
                - self.tier2_outlet_count,
            )
            * TIER_WEIGHTS[OutletTier.TIER3]
        )

    @property
    def dedup_ratio(self) -> float:
        """Ratio of unique stories to raw articles (lower = more duplication)."""
        if self.raw_article_count == 0:
            return 1.0
        return self.unique_story_count / self.raw_article_count

    def matches_event(self, event_title: str, keywords: List[str] = None) -> bool:
        """Check if this signal matches an event by keyword overlap."""
        event_lower = event_title.lower()
        topic_lower = self.topic.lower()

        # Direct topic match
        if topic_lower in event_lower or event_lower in topic_lower:
            return True

        # Split topic into words and check each
        topic_words = topic_lower.split()
        for word in topic_words:
            if len(word) > 3 and word in event_lower:
                return True

        # Keyword matching
        if keywords:
            for kw in keywords:
                if kw.lower() in topic_lower:
                    return True

        return False


@dataclass
class SignalConfig:
    """Configuration for how trending signals influence decisions."""

    # Maximum probability adjustment from signals (single-lever)
    max_probability_adjustment: float = 0.15  # Up to 15% probability shift

    # DEPRECATED: confidence_boost replaced by probability_adjustment
    max_confidence_boost: float = 0.30  # Kept for backwards compatibility

    # Strong signal threshold
    strong_signal_threshold: float = 0.7

    # Minimum unique stories to consider signal valid (after de-dup)
    min_unique_stories: int = 2

    # DEPRECATED: use min_unique_stories instead
    min_source_count: int = 2

    # DEPRECATED: Kelly multiplier removed to prevent double-counting
    aligned_signal_kelly_multiplier: float = 1.0  # Always 1.0 now

    # Skip override: allow betting on otherwise-skipped markets if strong signal
    enable_skip_override: bool = True
    skip_override_min_strength: float = 0.8
    skip_override_min_sources: int = 5

    # Minimum relevance to apply signal influence
    min_relevance: float = 0.3

    # Baseline frequency threshold (signal only if 1.5x+ above baseline)
    baseline_ratio_threshold: float = 1.5


def calculate_signal_strength(
    unique_story_count: int,
    relevance: float,
    novelty: float,
    sentiment_confidence: float,
    neutral_ratio: float,
    tier1_count: int = 0,
) -> float:
    """
    Calculate signal strength with proper weighting (v2 formula).

    Components:
    - Volume: Saturating function of unique stories (not raw count)
    - Relevance: How related headlines are to event (0-1)
    - Novelty: How recent the news is (0-1, decays over time)
    - Sentiment confidence: How clear the sentiment direction is (0-1)
    - Neutral penalty: Discount when most headlines are neutral
    - Tier-1 bonus: Boost for having quality sources

    Args:
        unique_story_count: Number of unique stories after de-duplication
        relevance: Semantic relevance to event (0-1)
        novelty: Time decay factor (0-1)
        sentiment_confidence: Confidence in sentiment direction (0-1)
        neutral_ratio: Fraction of headlines that are neutral (0-1)
        tier1_count: Number of Tier-1 outlets covering the story

    Returns:
        Signal strength (0-1)
    """
    if unique_story_count == 0:
        return 0.0

    # Volume component (saturates at ~10 unique stories)
    # Using 1 - exp(-x/5) gives smooth saturation
    volume = 0.5 * (1 - math.exp(-unique_story_count / 5))

    # Tier-1 bonus (up to 20% boost for quality sources)
    tier1_bonus = 1.0 + min(tier1_count * 0.05, 0.2)

    # Clarity component (with sample-size damping)
    # Avoid high clarity from tiny samples: confidence * n / (n + k)
    sample_damping = (
        sentiment_confidence * unique_story_count / (unique_story_count + 4)
    )

    # Neutral penalty (if >50% of headlines are neutral, reduce strength)
    neutral_penalty = 1.0 - (max(0, neutral_ratio - 0.5) * 0.5)

    # Relevance gate (must have some relevance to event)
    relevance_factor = max(relevance, 0.1)  # Floor at 0.1 to not zero out

    # Final strength
    strength = (
        volume
        * tier1_bonus
        * relevance_factor
        * novelty
        * sample_damping
        * neutral_penalty
    )

    return min(max(strength, 0.0), 1.0)


def analyze_headline_sentiment(headline: str) -> Tuple[str, float]:
    """
    Analyze sentiment of a single headline with negation handling.

    Args:
        headline: News headline text

    Returns:
        Tuple of (direction, confidence) where:
        - direction: "positive", "negative", or "neutral"
        - confidence: 0.0-1.0 confidence in the direction
    """
    words = headline.lower().split()

    # Check for negation and uncertainty
    has_negation = any(word in NEGATION_WORDS for word in words)
    has_uncertainty = any(word in UNCERTAINTY_WORDS for word in words)

    # Count sentiment words
    positive = sum(1 for w in words if w in POSITIVE_KEYWORDS)
    negative = sum(1 for w in words if w in NEGATIVE_KEYWORDS)

    # Apply negation flip
    if has_negation:
        positive, negative = negative, positive

    # Calculate confidence (reduced by uncertainty)
    base_confidence = 1.0
    if has_uncertainty:
        base_confidence *= 0.6

    # Determine direction and confidence
    total = positive + negative
    if total == 0:
        return ("neutral", 0.5)

    if positive > negative:
        direction = "positive"
        clarity = (positive - negative) / total
    elif negative > positive:
        direction = "negative"
        clarity = (negative - positive) / total
    else:
        return ("neutral", 0.5)

    confidence = base_confidence * min(clarity + 0.5, 1.0)
    return (direction, confidence)


def calculate_signal_influence(
    signal: TrendingSignal,
    action: str,
    research_probability: float,
    config: SignalConfig = None,
) -> Dict[str, Any]:
    """
    Calculate how a trending signal influences a betting decision (v2).

    IMPORTANT: Uses SINGLE-LEVER approach - adjusts probability only.
    Kelly is derived from adjusted probability, not multiplied separately.
    This prevents double-counting of the same information.

    Args:
        signal: The trending signal with quality metrics
        action: "buy_yes" or "buy_no"
        research_probability: Original probability estimate (0-1)
        config: Signal configuration

    Returns:
        Dict with probability_adjustment, should_override_skip, signal_direction, etc.
    """
    if config is None:
        config = SignalConfig()

    # Use unique_story_count (after de-dup) instead of raw source_count
    effective_count = (
        signal.unique_story_count
        if signal.unique_story_count > 0
        else signal.source_count
    )

    # Determine if signal aligns with action
    # Positive sentiment -> supports YES
    # Negative sentiment -> supports NO
    signal_aligns = (signal.sentiment == "positive" and action == "buy_yes") or (
        signal.sentiment == "negative" and action == "buy_no"
    )

    signal_conflicts = (signal.sentiment == "positive" and action == "buy_no") or (
        signal.sentiment == "negative" and action == "buy_yes"
    )

    result = {
        # NEW: Single-lever probability adjustment
        "probability_adjustment": 0.0,
        # DEPRECATED: Kept for backwards compatibility, always 1.0 now
        "confidence_boost": 0.0,
        "kelly_multiplier": 1.0,
        # Override and direction
        "should_override_skip": False,
        "signal_direction": "neutral",
        "reasoning": "",
        # NEW: Quality metrics for logging
        "unique_stories": effective_count,
        "relevance": signal.relevance,
        "sentiment_confidence": signal.sentiment_confidence,
    }

    # Gate 1: Minimum story count
    if effective_count < config.min_unique_stories:
        result["reasoning"] = (
            f"Insufficient unique stories ({effective_count} < {config.min_unique_stories})"
        )
        return result

    # Gate 2: Minimum relevance (must be about the event)
    if signal.relevance < config.min_relevance:
        result["reasoning"] = (
            f"Low relevance ({signal.relevance:.2f} < {config.min_relevance})"
        )
        return result

    # Gate 3: Baseline frequency check (skip if normal volume for this entity)
    if (
        signal.frequency_ratio < config.baseline_ratio_threshold
        and signal.baseline_frequency > 0
    ):
        result["reasoning"] = (
            f"Normal volume ({signal.current_frequency:.0f} vs baseline {signal.baseline_frequency:.0f})"
        )
        return result

    # Calculate adjustment magnitude based on strength and relevance
    # Max adjustment is ±15% of probability
    adjustment_magnitude = (
        signal.strength * signal.relevance * config.max_probability_adjustment
    )

    if signal_aligns:
        result["signal_direction"] = "aligned"

        # Adjust probability towards certainty (1.0 for YES, 0.0 for NO)
        if action == "buy_yes":
            result["probability_adjustment"] = adjustment_magnitude
        else:  # buy_no
            result["probability_adjustment"] = -adjustment_magnitude

        # Check for skip override (strong aligned signal)
        if config.enable_skip_override:
            if (
                signal.strength >= config.skip_override_min_strength
                and effective_count >= config.skip_override_min_sources
            ):
                result["should_override_skip"] = True

        result["reasoning"] = (
            f"Signal aligned with {action}: {signal.sentiment} sentiment "
            f"(strength: {signal.strength:.2f}, unique_stories: {effective_count}, "
            f"relevance: {signal.relevance:.2f}, prob_adj: {result['probability_adjustment']:+.3f})"
        )

        # Backwards compatibility: also set confidence_boost
        result["confidence_boost"] = adjustment_magnitude

    elif signal_conflicts:
        result["signal_direction"] = "conflicting"

        # Reduce probability adjustment (pull back from certainty)
        penalty = adjustment_magnitude * 0.67  # 2/3 of aligned boost

        if action == "buy_yes":
            result["probability_adjustment"] = -penalty
        else:
            result["probability_adjustment"] = penalty

        result["reasoning"] = (
            f"Signal conflicts with {action}: {signal.sentiment} sentiment "
            f"(strength: {signal.strength:.2f}, unique_stories: {effective_count}, "
            f"relevance: {signal.relevance:.2f}, prob_adj: {result['probability_adjustment']:+.3f})"
        )

        # Backwards compatibility
        result["confidence_boost"] = -penalty

    else:
        result["reasoning"] = f"Neutral signal (sentiment: {signal.sentiment})"

    return result


class TrendRadarClient:
    """HTTP Client for TrendRadar MCP Server with production hardening.

    Features:
    - Retries with exponential backoff + jitter
    - Circuit breaker pattern
    - Per-event signal caching with TTL
    - Fail-closed: returns neutral influence on failure

    Signal Quality Enhancements (v2):
    - OpenAI embeddings for semantic relevance scoring
    - Dynamic baseline frequency tracking
    - Story de-duplication and outlet tiering
    """

    # Western financial RSS feed IDs to query
    WESTERN_FINANCIAL_FEEDS = [
        "reuters-business",
        "reuters-politics",
        "cnbc-top",
        "cnbc-world",
        "wsj-markets",
        "wsj-world",
        "ap-news-top",
        "ap-news-politics",
        "bloomberg-markets",
        "nyt-business",
        "nyt-politics",
        "marketwatch-top",
    ]

    # Default retry configuration (can be overridden via constructor)
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_BASE_BACKOFF = 1.0  # seconds
    DEFAULT_MAX_BACKOFF = 10.0  # seconds

    # OpenAI embedding model
    EMBEDDING_MODEL = "text-embedding-3-small"

    def __init__(
        self,
        base_url: str = "http://localhost:3333",
        timeout: float = 10.0,
        enabled: bool = True,
        cache_ttl: float = 300.0,  # 5 minutes
        max_retries: int = None,
        retry_backoff_base: float = None,
        circuit_failure_threshold: int = 3,
        circuit_reset_seconds: int = 300,
        openai_api_key: str = None,
        enable_embeddings: bool = True,
    ):
        self.base_url = base_url.rstrip("/")
        self.mcp_endpoint = f"{self.base_url}/mcp"
        self.timeout = timeout
        self.enabled = enabled
        self._client: Optional[httpx.AsyncClient] = None

        # MCP session management (FastMCP 2.0 HTTP transport)
        self._session_id: Optional[str] = None
        self._session_initialized: bool = False
        self._request_id: int = 0

        # Retry configuration (use provided values or defaults)
        self.MAX_RETRIES = (
            max_retries if max_retries is not None else self.DEFAULT_MAX_RETRIES
        )
        self.BASE_BACKOFF = (
            retry_backoff_base
            if retry_backoff_base is not None
            else self.DEFAULT_BASE_BACKOFF
        )
        self.MAX_BACKOFF = self.DEFAULT_MAX_BACKOFF

        # Circuit breaker
        self._cb_config = CircuitBreakerConfig(
            failure_threshold=circuit_failure_threshold,
            recovery_timeout=float(circuit_reset_seconds),
        )
        self._cb_state = CircuitBreakerState()

        # Signal cache
        self._cache = SignalCache(default_ttl=cache_ttl)

        # OpenAI embeddings for relevance scoring (Phase 2)
        self.openai_api_key = openai_api_key
        self.enable_embeddings = enable_embeddings and openai_api_key is not None
        self._embedding_cache: Dict[str, List[float]] = {}  # Cache embeddings

        # Baseline frequency tracking (Phase 3)
        self._baseline_frequencies: Dict[str, float] = {}  # entity -> 7-day avg
        self._frequency_history: Dict[str, List[Tuple[float, int]]] = (
            {}
        )  # entity -> [(timestamp, count)]

        # Metrics for observability
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "cache_hits": 0,
            "circuit_breaker_rejections": 0,
            "total_latency_ms": 0.0,
            "embedding_requests": 0,
            "embedding_cache_hits": 0,
        }

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create HTTP client with separate connect/read timeouts."""
        if self._client is None or self._client.is_closed:
            # Separate connect and read timeouts for better control
            timeout_config = httpx.Timeout(
                connect=5.0,  # Connection timeout
                read=self.timeout,  # Read timeout
                write=10.0,
                pool=5.0,
            )
            self._client = httpx.AsyncClient(timeout=timeout_config)
        return self._client

    async def close(self):
        """Close the HTTP client."""
        if self._client and not self._client.is_closed:
            await self._client.aclose()

    def get_metrics(self) -> Dict[str, Any]:
        """Return client metrics for observability."""
        metrics = self.metrics.copy()
        metrics["circuit_breaker_state"] = self._cb_state.state
        metrics["circuit_breaker_failures"] = self._cb_state.failure_count
        return metrics

    def clear_cache(self) -> None:
        """Clear the signal cache."""
        self._cache.clear()
        self._embedding_cache.clear()

    # =========================================================================
    # Phase 2: OpenAI Embeddings for Semantic Relevance
    # =========================================================================

    async def _get_embedding(self, text: str) -> Optional[List[float]]:
        """Get embedding for a text using OpenAI API."""
        if not self.enable_embeddings or not self.openai_api_key:
            return None

        # Check cache first
        cache_key = hashlib.md5(text.encode()).hexdigest()[:16]
        if cache_key in self._embedding_cache:
            self.metrics["embedding_cache_hits"] += 1
            return self._embedding_cache[cache_key]

        # OFFLINE_MODE: Return deterministic fake embedding based on text hash
        if OFFLINE_MODE:
            # Generate deterministic 1536-dim embedding from text hash (same as text-embedding-3-small)
            hash_bytes = hashlib.sha256(text.encode()).digest()
            # Use hash bytes to seed a pseudo-random number generator for determinism
            import struct

            seed = struct.unpack("<I", hash_bytes[:4])[0]
            rng = random.Random(seed)
            fake_embedding = [rng.gauss(0, 0.1) for _ in range(1536)]
            # Normalize to unit length
            norm = math.sqrt(sum(x * x for x in fake_embedding))
            fake_embedding = [x / norm for x in fake_embedding]
            self._embedding_cache[cache_key] = fake_embedding
            logger.debug(f"OFFLINE_MODE: Returning mock embedding for '{text[:50]}...'")
            return fake_embedding

        try:
            self.metrics["embedding_requests"] += 1
            client = await self._get_client()

            response = await client.post(
                "https://api.openai.com/v1/embeddings",
                headers={
                    "Authorization": f"Bearer {self.openai_api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.EMBEDDING_MODEL,
                    "input": text[:8000],  # Truncate to model limit
                },
            )
            response.raise_for_status()
            data = response.json()

            embedding = data["data"][0]["embedding"]
            self._embedding_cache[cache_key] = embedding
            return embedding

        except Exception as e:
            logger.debug(f"Embedding request failed: {e}")
            return None

    def _cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    async def calculate_relevance(
        self, event_title: str, headlines: List[str]
    ) -> float:
        """
        Calculate semantic relevance of headlines to event using embeddings.

        Falls back to keyword matching if embeddings not available.
        Returns: relevance score bounded to [0, 1]
        """
        if not headlines:
            return 0.0

        # Try embedding-based relevance first
        if self.enable_embeddings:
            event_embedding = await self._get_embedding(event_title)
            if event_embedding:
                # Get embeddings for top headlines
                similarities = []
                for headline in headlines[:5]:
                    headline_embedding = await self._get_embedding(headline)
                    if headline_embedding:
                        sim = self._cosine_similarity(
                            event_embedding, headline_embedding
                        )
                        # Clamp to [0, 1] - negative similarity means no relevance
                        similarities.append(max(0.0, sim))

                if similarities:
                    # Return max similarity (best match), bounded to [0, 1]
                    return min(max(similarities), 1.0)

        # Fallback: keyword-based relevance
        event_words = set(event_title.lower().split())
        max_overlap = 0.0

        for headline in headlines[:10]:
            headline_words = set(headline.lower().split())
            if len(headline_words) > 0:
                overlap = len(event_words & headline_words) / len(headline_words)
                max_overlap = max(max_overlap, overlap)

        return min(max_overlap * 1.5, 1.0)  # Scale up keyword overlap

    # =========================================================================
    # Phase 3: Dynamic Baseline Frequency Tracking
    # =========================================================================

    def update_baseline_frequency(self, entity: str, current_count: int) -> None:
        """Update the 7-day rolling baseline for an entity.

        IMPORTANT: Baseline is calculated from HISTORICAL data only (excluding current).
        This prevents data leakage where current observation inflates its own baseline.
        """
        entity_lower = entity.lower()
        now = time.time()

        # Initialize history if needed
        if entity_lower not in self._frequency_history:
            self._frequency_history[entity_lower] = []

        # Calculate baseline from HISTORICAL data BEFORE adding current observation
        # This prevents the current observation from inflating its own baseline
        cutoff = now - (7 * 24 * 60 * 60)
        historical = [
            (ts, count)
            for ts, count in self._frequency_history[entity_lower]
            if ts > cutoff
        ]

        # Update baseline from historical data (excluding current)
        if len(historical) >= 1:
            avg = sum(count for _, count in historical) / len(historical)
            self._baseline_frequencies[entity_lower] = avg

        # NOW add current observation for future baseline calculations
        self._frequency_history[entity_lower] = historical + [(now, current_count)]

    def get_frequency_ratio(
        self, entity: str, current_count: int
    ) -> Tuple[float, float, float]:
        """
        Get the frequency ratio for an entity vs its baseline.

        Returns:
            Tuple of (baseline, current, ratio)
        """
        entity_lower = entity.lower()
        baseline = self._baseline_frequencies.get(entity_lower, 5.0)  # Default baseline

        if baseline == 0:
            baseline = 1.0

        ratio = current_count / baseline
        return (baseline, float(current_count), ratio)

    def _next_request_id(self) -> int:
        """Get next request ID for JSON-RPC."""
        self._request_id += 1
        return self._request_id

    def _parse_sse_response(self, text: str) -> Optional[Dict[str, Any]]:
        """Parse Server-Sent Events response from FastMCP 2.0.

        Format:
            event: message
            data: {"jsonrpc": "2.0", "id": 1, "result": {...}}

        Returns the parsed JSON data or None if invalid.
        """
        for line in text.strip().split("\n"):
            if line.startswith("data: "):
                json_str = line[6:]  # Remove 'data: ' prefix
                try:
                    return json.loads(json_str)
                except json.JSONDecodeError:
                    continue
        return None

    async def _ensure_session(self) -> bool:
        """Initialize MCP session if not already done.

        FastMCP 2.0 HTTP transport requires:
        1. Initialize handshake to get session ID
        2. Send notifications/initialized to complete handshake
        3. Include mcp-session-id header in subsequent requests
        """
        if self._session_initialized and self._session_id:
            return True

        try:
            client = await self._get_client()

            # Step 1: MCP initialize handshake
            init_payload = {
                "jsonrpc": "2.0",
                "id": self._next_request_id(),
                "method": "initialize",
                "params": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {},
                    "clientInfo": {"name": "kalshi-trading-bot", "version": "1.0.0"},
                },
            }

            response = await client.post(
                self.mcp_endpoint,
                json=init_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                },
            )
            response.raise_for_status()

            # Extract session ID from response headers
            self._session_id = response.headers.get("mcp-session-id")
            if not self._session_id:
                logger.warning("TrendRadar: No session ID in initialize response")
                return False

            # Parse SSE response to verify success
            result = self._parse_sse_response(response.text)
            if not result or "result" not in result:
                logger.warning("TrendRadar: Invalid initialize response")
                return False

            # Step 2: Send notifications/initialized to complete handshake
            # Note: Notifications don't have an "id" field per JSON-RPC spec
            initialized_payload = {
                "jsonrpc": "2.0",
                "method": "notifications/initialized",
            }

            await client.post(
                self.mcp_endpoint,
                json=initialized_payload,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json, text/event-stream",
                    "mcp-session-id": self._session_id,
                },
            )

            self._session_initialized = True
            logger.info(f"TrendRadar session initialized: {self._session_id[:8]}...")
            return True

        except Exception as e:
            logger.warning(f"TrendRadar session initialization failed: {e}")
            return False

    async def _call_tool(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Call a TrendRadar MCP tool via HTTP with retry and circuit breaker.

        FastMCP 2.0 HTTP transport uses JSON-RPC style calls with session management.

        Production hardening:
        - Session initialization and management
        - Circuit breaker pattern (fail-fast when server is down)
        - Retries with exponential backoff + jitter
        - Metrics tracking for observability
        """
        # OFFLINE_MODE: Return deterministic mock responses without network calls
        if OFFLINE_MODE:
            logger.debug(f"OFFLINE_MODE: Mock response for {tool_name}")
            if tool_name in ("search_rss", "aggregate_news"):
                # Return deterministic mock news articles based on keyword hash
                keywords = arguments.get("keywords", arguments.get("keyword", "test"))
                hash_seed = (
                    int(hashlib.md5(keywords.encode()).hexdigest()[:8], 16) % 1000
                )
                mock_articles = [
                    {
                        "title": f"Mock headline about {keywords} - article {i+1}",
                        "description": f"This is a mock article about {keywords} for offline testing.",
                        "source": ["reuters", "bloomberg", "cnbc", "ap-news"][i % 4],
                        "link": f"https://mock.news/article/{hash_seed + i}",
                        "published": "2024-01-01T12:00:00Z",
                    }
                    for i in range(3)
                ]
                return {
                    "success": True,
                    "articles": mock_articles,
                    "total_count": len(mock_articles),
                    "clusters": [{"size": 3, "articles": mock_articles}],
                }
            elif tool_name == "analyze_sentiment":
                return {
                    "success": True,
                    "sentiment": "neutral",
                    "confidence": 0.5,
                    "analysis": "Mock sentiment analysis for offline testing",
                }
            else:
                return {"success": True, "mock": True}

        if not self.enabled:
            return {"success": False, "error": "TrendRadar client disabled"}

        # Check circuit breaker
        if not self._cb_state.should_allow_request(self._cb_config):
            self.metrics["circuit_breaker_rejections"] += 1
            logger.debug(f"Circuit breaker OPEN - rejecting request to {tool_name}")
            return {"success": False, "error": "Circuit breaker open"}

        # Ensure session is initialized
        if not await self._ensure_session():
            self.metrics["failed_requests"] += 1
            return {"success": False, "error": "Failed to initialize MCP session"}

        self.metrics["total_requests"] += 1
        start_time = time.time()
        last_error = None

        for attempt in range(self.MAX_RETRIES):
            try:
                client = await self._get_client()

                # FastMCP 2.0 HTTP endpoint format with session
                payload = {
                    "jsonrpc": "2.0",
                    "id": self._next_request_id(),
                    "method": "tools/call",
                    "params": {"name": tool_name, "arguments": arguments},
                }

                response = await client.post(
                    self.mcp_endpoint,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "Accept": "application/json, text/event-stream",
                        "mcp-session-id": self._session_id,
                    },
                )
                response.raise_for_status()

                # Parse SSE response format
                result = self._parse_sse_response(response.text)
                if not result:
                    # Try direct JSON parse as fallback
                    try:
                        result = response.json()
                    except json.JSONDecodeError:
                        last_error = "Failed to parse response"
                        continue

                # Parse the MCP response
                if "result" in result:
                    content = result["result"].get("content", [])
                    if content and isinstance(content, list):
                        # MCP returns content as list of TextContent
                        text_content = content[0].get("text", "{}")
                        parsed = json.loads(text_content)

                        # Success - record metrics and reset circuit breaker
                        self._cb_state.record_success()
                        self.metrics["successful_requests"] += 1
                        self.metrics["total_latency_ms"] += (
                            time.time() - start_time
                        ) * 1000
                        return parsed

                # Check for error in result
                if "error" in result:
                    error_msg = result["error"].get("message", "Unknown MCP error")
                    # Session expired - reset and retry
                    if "session" in error_msg.lower():
                        self._session_initialized = False
                        self._session_id = None
                        if await self._ensure_session():
                            continue  # Retry with new session
                    last_error = error_msg
                    break

                # Invalid response format - don't retry
                self.metrics["failed_requests"] += 1
                return {"success": False, "error": "Invalid MCP response format"}

            except httpx.HTTPStatusError as e:
                last_error = f"HTTP {e.response.status_code}"
                # Reset session on 4xx errors (might be session issue)
                if e.response.status_code in [401, 403, 406]:
                    self._session_initialized = False
                    self._session_id = None
                # Don't retry on most 4xx client errors
                if (
                    400 <= e.response.status_code < 500
                    and e.response.status_code != 406
                ):
                    break
            except httpx.ConnectError:
                last_error = "Connection failed"
            except httpx.TimeoutException:
                last_error = "Request timeout"
            except json.JSONDecodeError as e:
                last_error = f"Invalid JSON: {e}"
                # Don't retry on malformed responses
                break
            except Exception as e:
                last_error = str(e)

            # Exponential backoff with jitter before retry
            if attempt < self.MAX_RETRIES - 1:
                backoff = min(self.BASE_BACKOFF * (2**attempt), self.MAX_BACKOFF)
                jitter = random.uniform(0, backoff * 0.3)
                wait_time = backoff + jitter
                logger.debug(
                    f"TrendRadar retry {attempt + 1}/{self.MAX_RETRIES} after {wait_time:.2f}s: {last_error}"
                )
                await asyncio.sleep(wait_time)

        # All retries exhausted - record failure
        self._cb_state.record_failure(self._cb_config)
        self.metrics["failed_requests"] += 1
        self.metrics["total_latency_ms"] += (time.time() - start_time) * 1000
        logger.warning(
            f"TrendRadar {tool_name} failed after {self.MAX_RETRIES} attempts: {last_error}"
        )
        return {"success": False, "error": last_error or "Unknown error"}

    async def health_check(self) -> bool:
        """Check if TrendRadar server is running."""
        try:
            client = await self._get_client()
            # Try the MCP endpoint with a simple tool call
            response = await client.get(f"{self.base_url}/", timeout=5.0)
            return response.status_code in [200, 404]  # 404 is ok, means server is up
        except Exception:
            return False

    async def get_trending_topics(
        self, top_n: int = 20, mode: str = "current", extract_mode: str = "auto_extract"
    ) -> List[Dict[str, Any]]:
        """Get trending topics from TrendRadar.

        Args:
            top_n: Number of top topics to return
            mode: "current" for latest batch, "daily" for full day
            extract_mode: "auto_extract" for automatic keyword discovery
        """
        result = await self._call_tool(
            "get_trending_topics",
            {"top_n": top_n, "mode": mode, "extract_mode": extract_mode},
        )

        if result.get("success", True) and "error" not in result:
            return result.get("topics", [])
        return []

    async def search_news(
        self, query: str, include_rss: bool = True, limit: int = 50, rss_limit: int = 30
    ) -> Dict[str, Any]:
        """Search news across hot topics and RSS feeds.

        Args:
            query: Search keyword (event title, topic, etc.)
            include_rss: Include RSS feed results
            limit: Max hot topic results
            rss_limit: Max RSS results
        """
        result = await self._call_tool(
            "search_news",
            {
                "query": query,
                "search_mode": "keyword",
                "include_rss": include_rss,
                "limit": limit,
                "rss_limit": rss_limit,
                "include_url": False,
            },
        )

        return result

    async def analyze_sentiment(self, topic: str, limit: int = 30) -> Dict[str, Any]:
        """Analyze sentiment for a topic.

        Args:
            topic: Topic keyword to analyze
            limit: Max news items to analyze
        """
        result = await self._call_tool(
            "analyze_sentiment", {"topic": topic, "limit": limit, "include_url": False}
        )

        return result

    async def get_latest_rss(
        self, feeds: List[str] = None, limit: int = 50, include_summary: bool = True
    ) -> Dict[str, Any]:
        """Get latest RSS feed entries.

        Args:
            feeds: List of feed IDs to query (default: Western financial)
            limit: Max entries to return
            include_summary: Include article summaries
        """
        if feeds is None:
            feeds = self.WESTERN_FINANCIAL_FEEDS

        result = await self._call_tool(
            "get_latest_rss",
            {"feeds": feeds, "limit": limit, "include_summary": include_summary},
        )

        return result

    async def search_rss(
        self, keyword: str, feeds: List[str] = None, days: int = 3, limit: int = 30
    ) -> Dict[str, Any]:
        """Search RSS feeds for keyword.

        Args:
            keyword: Search term
            feeds: Feed IDs to search (default: Western financial)
            days: How many days back to search
            limit: Max results
        """
        if feeds is None:
            feeds = self.WESTERN_FINANCIAL_FEEDS

        result = await self._call_tool(
            "search_rss",
            {
                "keyword": keyword,
                "feeds": feeds,
                "days": days,
                "limit": limit,
                "include_summary": True,
            },
        )

        return result

    async def get_signals_for_event(
        self, event_title: str, event_category: str = "", keywords: List[str] = None
    ) -> List[TrendingSignal]:
        """Get trending signals relevant to a prediction market event.

        This is the main integration point - extracts signals for an event.

        Production hardening:
        - Caching: Results cached per event with TTL to reduce variance
        - Fail-closed: Returns empty list on failure (neutral influence)

        Args:
            event_title: The event title from Kalshi
            event_category: Event category (e.g., "Politics", "Finance")
            keywords: Additional keywords to search

        Returns:
            List of TrendingSignal objects matching the event
        """
        # Check cache first
        cached = self._cache.get(event_title, event_category)
        if cached is not None:
            self.metrics["cache_hits"] += 1
            return cached

        signals = []

        # Extract keywords from event title
        search_terms = self._extract_search_terms(event_title, keywords)

        if not search_terms:
            # Cache empty result to avoid repeated lookups
            self._cache.set(event_title, event_category, signals)
            return signals

        logger.debug(f"Searching TrendRadar for terms: {search_terms[:3]}")

        # Search each term (limit to top 3 to avoid too many API calls)
        for term in search_terms[:3]:
            try:
                # Search RSS for this term
                rss_result = await self.search_rss(keyword=term, days=2, limit=20)

                # Get sentiment analysis (may not be available for all terms)
                sentiment_result = await self.analyze_sentiment(topic=term, limit=20)

                # Parse into signal if we have data
                signal = self._parse_signal(
                    term=term,
                    rss_result=rss_result,
                    sentiment_result=sentiment_result,
                    event_title=event_title,  # Pass for relevance calculation
                )

                # Use unique_story_count (after de-dup) instead of raw source_count
                if signal and signal.unique_story_count >= 2:
                    # Phase 2: Calculate semantic relevance with embeddings
                    if self.enable_embeddings and signal.sample_headlines:
                        relevance = await self.calculate_relevance(
                            event_title=event_title, headlines=signal.sample_headlines
                        )
                        signal.relevance = relevance

                        # Recalculate strength with new relevance
                        signal.strength = calculate_signal_strength(
                            unique_story_count=signal.unique_story_count,
                            relevance=signal.relevance,
                            novelty=signal.novelty,
                            sentiment_confidence=signal.sentiment_confidence,
                            neutral_ratio=signal.neutral_ratio,
                            tier1_count=signal.tier1_outlet_count,
                        )

                    # Phase 3: Update baseline frequency tracking
                    self.update_baseline_frequency(term, signal.unique_story_count)
                    baseline, current, ratio = self.get_frequency_ratio(
                        term, signal.unique_story_count
                    )
                    signal.baseline_frequency = baseline
                    signal.current_frequency = current
                    signal.frequency_ratio = ratio

                    signals.append(signal)

            except Exception as e:
                logger.debug(f"Error searching for term '{term}': {e}")
                continue

        # Cache the result
        self._cache.set(event_title, event_category, signals)
        return signals

    def _extract_search_terms(
        self, event_title: str, additional_keywords: List[str] = None
    ) -> List[str]:
        """Extract meaningful search terms from event title."""
        # Common stopwords to filter
        stopwords = {
            "will",
            "the",
            "be",
            "a",
            "an",
            "of",
            "and",
            "or",
            "for",
            "to",
            "in",
            "on",
            "at",
            "by",
            "with",
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "been",
            "being",
            "this",
            "that",
            "which",
            "who",
            "whom",
            "what",
            "when",
            "where",
            "how",
            "why",
            "before",
            "after",
            "during",
            "yes",
            "no",
            "win",
            "winner",
            "election",
            "vote",
            "votes",
            "than",
            "more",
            "less",
            "most",
            "least",
            "any",
            "all",
            "some",
            "each",
        }

        terms = []

        # Get multi-word phrases first (proper nouns, entities)
        # Look for capitalized words in original title
        original_words = event_title.split()
        phrase = []
        for word in original_words:
            clean_word = word.strip(".,!?()[]{}\"'-:;")
            if (
                clean_word
                and clean_word[0].isupper()
                and clean_word.lower() not in stopwords
            ):
                phrase.append(clean_word)
            elif phrase:
                if len(phrase) >= 2:
                    terms.append(" ".join(phrase))
                elif len(phrase) == 1 and len(phrase[0]) > 3:
                    terms.append(phrase[0])
                phrase = []
        if phrase:
            if len(phrase) >= 2:
                terms.append(" ".join(phrase))
            elif len(phrase) == 1 and len(phrase[0]) > 3:
                terms.append(phrase[0])

        # Add single significant words
        words = event_title.lower().split()
        for word in words:
            clean = word.strip(".,!?()[]{}\"'-:;")
            if len(clean) > 3 and clean not in stopwords:
                terms.append(clean)

        # Add additional keywords
        if additional_keywords:
            terms.extend(additional_keywords)

        # Deduplicate while preserving order
        seen = set()
        unique_terms = []
        for term in terms:
            lower = term.lower()
            if lower not in seen:
                seen.add(lower)
                unique_terms.append(term)

        return unique_terms

    def _parse_signal(
        self,
        term: str,
        rss_result: Dict[str, Any],
        sentiment_result: Dict[str, Any],
        event_title: str = "",
    ) -> Optional[TrendingSignal]:
        """
        Parse TrendRadar results into a TrendingSignal with quality metrics (v2).

        Improvements:
        - De-duplication: Groups similar headlines into unique stories
        - Outlet tiering: Weights Tier-1 sources (Reuters, Bloomberg, etc.) higher
        - Sentiment: Uses negation-aware analysis
        - Strength: Uses new multi-factor formula
        """
        # Extract all items from RSS and news results
        rss_items = []
        if isinstance(rss_result, dict):
            rss_items = rss_result.get("rss", []) or rss_result.get("items", []) or []

        news_items = []
        if isinstance(sentiment_result, dict):
            news_items = (
                sentiment_result.get("news", [])
                or sentiment_result.get("news_sample", [])
                or []
            )

        # Combine all items with their sources
        all_items = []
        for item in rss_items:
            all_items.append(
                {
                    "title": item.get("title", ""),
                    "source": item.get("feed_name", "")
                    or item.get("feed", "")
                    or item.get("feed_id", ""),
                    "type": "rss",
                }
            )
        for item in news_items:
            all_items.append(
                {
                    "title": item.get("title", ""),
                    "source": item.get("platform", "") or item.get("source", ""),
                    "type": "news",
                }
            )

        raw_article_count = len(all_items)
        if raw_article_count == 0:
            return None

        # === DE-DUPLICATION: Group similar headlines ===
        # Use simple fuzzy matching to identify unique stories
        unique_stories = []
        story_clusters = []

        for item in all_items:
            title = item["title"].lower().strip()
            if not title:
                continue

            # Check if this title is similar to an existing story
            found_cluster = False
            for i, cluster in enumerate(story_clusters):
                # Simple similarity: check word overlap
                title_words = set(title.split())
                cluster_words = set(cluster["representative"].lower().split())
                if len(title_words) > 0 and len(cluster_words) > 0:
                    overlap = len(title_words & cluster_words) / min(
                        len(title_words), len(cluster_words)
                    )
                    if overlap > 0.6:  # 60% word overlap = same story
                        cluster["items"].append(item)
                        cluster["sources"].add(item["source"])
                        found_cluster = True
                        break

            if not found_cluster:
                # New unique story
                story_clusters.append(
                    {
                        "representative": item["title"],
                        "items": [item],
                        "sources": {item["source"]},
                    }
                )

        unique_story_count = len(story_clusters)
        story_cluster_sizes = [len(c["items"]) for c in story_clusters]

        # === OUTLET TIERING ===
        all_sources = set()
        tier1_sources = set()
        tier2_sources = set()

        for cluster in story_clusters:
            for source in cluster["sources"]:
                all_sources.add(source)
                tier = get_outlet_tier(source)
                if tier == OutletTier.TIER1:
                    tier1_sources.add(source)
                elif tier == OutletTier.TIER2:
                    tier2_sources.add(source)

        unique_outlet_count = len(all_sources)
        tier1_outlet_count = len(tier1_sources)
        tier2_outlet_count = len(tier2_sources)

        # === HEADLINES for analysis ===
        headlines = [c["representative"] for c in story_clusters[:10]]
        platforms = list(all_sources)

        # === SENTIMENT ANALYSIS (with negation handling) ===
        positive_votes = 0
        negative_votes = 0
        neutral_votes = 0
        total_confidence = 0.0

        for headline in headlines:
            direction, confidence = analyze_headline_sentiment(headline)
            total_confidence += confidence
            if direction == "positive":
                positive_votes += 1
            elif direction == "negative":
                negative_votes += 1
            else:
                neutral_votes += 1

        total_votes = positive_votes + negative_votes + neutral_votes
        if total_votes == 0:
            return None

        # Determine overall sentiment
        if positive_votes > negative_votes and positive_votes > neutral_votes:
            sentiment_dir = "positive"
        elif negative_votes > positive_votes and negative_votes > neutral_votes:
            sentiment_dir = "negative"
        else:
            sentiment_dir = "neutral"

        # Sentiment confidence = average confidence * clarity
        avg_confidence = total_confidence / total_votes if total_votes > 0 else 0.5
        clarity = abs(positive_votes - negative_votes) / max(
            positive_votes + negative_votes, 1
        )
        sentiment_confidence = avg_confidence * (0.5 + clarity * 0.5)

        # Neutral ratio
        neutral_ratio = neutral_votes / total_votes if total_votes > 0 else 0.0

        # === RELEVANCE (basic keyword matching - Phase 2 will add embeddings) ===
        relevance = 0.5  # Default relevance
        if event_title:
            event_words = set(event_title.lower().split())
            term_words = set(term.lower().split())
            # Check if term is in event title
            if term.lower() in event_title.lower():
                relevance = 0.8
            elif len(event_words & term_words) > 0:
                relevance = 0.6

        # === NOVELTY (default to 1.0 - Phase 3 will add time decay) ===
        novelty = 1.0

        # === CALCULATE STRENGTH using new formula ===
        strength = calculate_signal_strength(
            unique_story_count=unique_story_count,
            relevance=relevance,
            novelty=novelty,
            sentiment_confidence=sentiment_confidence,
            neutral_ratio=neutral_ratio,
            tier1_count=tier1_outlet_count,
        )

        return TrendingSignal(
            topic=term,
            sentiment=sentiment_dir,
            strength=strength,
            source_count=raw_article_count,  # DEPRECATED: kept for backwards compat
            # Headlines and sources
            sample_headlines=headlines[:5],
            platforms=platforms,
            # NEW: De-duplication metrics
            raw_article_count=raw_article_count,
            unique_story_count=unique_story_count,
            unique_outlet_count=unique_outlet_count,
            tier1_outlet_count=tier1_outlet_count,
            tier2_outlet_count=tier2_outlet_count,
            story_cluster_sizes=story_cluster_sizes,
            # NEW: Quality metrics
            relevance=relevance,
            novelty=novelty,
            sentiment_confidence=sentiment_confidence,
            neutral_ratio=neutral_ratio,
        )


def format_signals_for_research(signals: List[TrendingSignal]) -> str:
    """Format trending signals for inclusion in research prompt (v2)."""
    if not signals:
        return ""

    lines = []
    for sig in signals:
        emoji = (
            "+"
            if sig.sentiment == "positive"
            else "-" if sig.sentiment == "negative" else "~"
        )

        # Show de-duplication ratio if significant
        dedup_info = ""
        if sig.raw_article_count > sig.unique_story_count:
            dedup_info = f", {sig.unique_story_count}/{sig.raw_article_count} unique"

        # Show tier-1 sources if any
        tier1_info = ""
        if sig.tier1_outlet_count > 0:
            tier1_info = f", {sig.tier1_outlet_count} Tier-1"

        lines.append(
            f"[{emoji}] {sig.topic}: {sig.sentiment.upper()} sentiment "
            f"(strength: {sig.strength:.2f}, relevance: {sig.relevance:.2f}{dedup_info}{tier1_info})"
        )
        if sig.sample_headlines:
            for headline in sig.sample_headlines[:2]:
                lines.append(f"    - {headline[:100]}...")

    return "\n".join(lines)
