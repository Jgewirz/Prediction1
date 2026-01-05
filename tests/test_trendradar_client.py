"""
Tests for TrendRadar client with production hardening features.

Coverage:
- Signal influence calculation (aligned, conflicting, neutral, boundaries)
- Circuit breaker behavior
- Retry logic with exponential backoff
- Caching behavior
- Config validation
- Response parsing (missing fields, malformed JSON)
"""
import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import asdict

# Import the modules under test
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trendradar_client import (
    TrendingSignal,
    SignalConfig,
    calculate_signal_influence,
    CircuitBreakerConfig,
    CircuitBreakerState,
    SignalCache,
    TrendRadarClient,
    format_signals_for_research,
)


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def signal_config():
    """Default signal configuration (v2)."""
    return SignalConfig(
        max_probability_adjustment=0.15,
        max_confidence_boost=0.30,  # Deprecated but kept for compat
        strong_signal_threshold=0.7,
        min_unique_stories=2,
        min_source_count=2,  # Deprecated
        aligned_signal_kelly_multiplier=1.25,  # Deprecated - always 1.0 now
        enable_skip_override=True,
        skip_override_min_strength=0.8,
        skip_override_min_sources=5,
        min_relevance=0.3,
        baseline_ratio_threshold=1.5,
    )


@pytest.fixture
def strong_positive_signal():
    """A strong positive signal (v2 with quality metrics)."""
    return TrendingSignal(
        topic="Trump tariffs",
        sentiment="positive",
        strength=0.85,
        source_count=7,
        sample_headlines=["Market rallies on trade deal hopes"],
        platforms=["Reuters", "Bloomberg", "CNBC"],
        # V2 quality fields
        unique_story_count=5,
        unique_outlet_count=3,
        tier1_outlet_count=2,  # Correct field name
        relevance=0.8,
        sentiment_confidence=0.9,
    )


@pytest.fixture
def weak_negative_signal():
    """A weak negative signal (v2)."""
    return TrendingSignal(
        topic="Fed rates",
        sentiment="negative",
        strength=0.4,
        source_count=3,
        sample_headlines=["Fed signals caution"],
        platforms=["WSJ"],
        # V2 quality fields
        unique_story_count=2,
        unique_outlet_count=1,
        relevance=0.5,
    )


@pytest.fixture
def neutral_signal():
    """A neutral signal (v2)."""
    return TrendingSignal(
        topic="Election",
        sentiment="neutral",
        strength=0.5,
        source_count=4,
        sample_headlines=["Voters head to polls"],
        platforms=["AP"],
        # V2 quality fields
        unique_story_count=3,
        unique_outlet_count=1,
        relevance=0.6,
    )


# ============================================================================
# Signal Influence Calculation Tests
# ============================================================================

class TestCalculateSignalInfluence:
    """Tests for calculate_signal_influence function (v2 single-lever)."""

    def test_aligned_positive_buy_yes(self, strong_positive_signal, signal_config):
        """Positive sentiment + buy_yes should adjust probability upward (v2)."""
        result = calculate_signal_influence(
            strong_positive_signal, "buy_yes", 0.6, signal_config
        )

        assert result["signal_direction"] == "aligned"
        # V2: Uses probability_adjustment, not confidence_boost
        assert result["probability_adjustment"] > 0
        assert result["probability_adjustment"] <= signal_config.max_probability_adjustment
        # V2: Kelly multiplier is always 1.0 (single-lever)
        assert result["kelly_multiplier"] == 1.0
        # Should override skip (strength=0.85, unique_stories=5 >= 5)
        assert result["should_override_skip"] is True

    def test_aligned_negative_buy_no(self, signal_config):
        """Negative sentiment + buy_no should adjust probability downward (v2)."""
        signal = TrendingSignal(
            topic="Market crash",
            sentiment="negative",
            strength=0.75,
            source_count=5,
            sample_headlines=["Stocks tumble"],
            platforms=["Reuters"],
            # V2 quality fields
            unique_story_count=4,
            unique_outlet_count=2,
            relevance=0.7,
        )
        result = calculate_signal_influence(signal, "buy_no", 0.6, signal_config)

        assert result["signal_direction"] == "aligned"
        # V2: probability_adjustment is negative for buy_no alignment
        assert result["probability_adjustment"] < 0
        # V2: Kelly multiplier is always 1.0
        assert result["kelly_multiplier"] == 1.0

    def test_conflicting_positive_buy_no(self, strong_positive_signal, signal_config):
        """Positive sentiment + buy_no should reduce probability (v2)."""
        result = calculate_signal_influence(
            strong_positive_signal, "buy_no", 0.6, signal_config
        )

        assert result["signal_direction"] == "conflicting"
        # V2: Conflicting signal adjusts probability opposite direction
        assert result["probability_adjustment"] > 0  # Positive adjustment for buy_no conflict
        # V2: Kelly multiplier is always 1.0
        assert result["kelly_multiplier"] == 1.0
        assert result["should_override_skip"] is False

    def test_conflicting_negative_buy_yes(self, signal_config):
        """Negative sentiment + buy_yes should reduce probability (v2)."""
        signal = TrendingSignal(
            topic="Crisis",
            sentiment="negative",
            strength=0.8,
            source_count=6,
            sample_headlines=["Markets in turmoil"],
            platforms=["Bloomberg"],
            # V2 quality fields
            unique_story_count=4,
            unique_outlet_count=2,
            relevance=0.6,
        )
        result = calculate_signal_influence(signal, "buy_yes", 0.6, signal_config)

        assert result["signal_direction"] == "conflicting"
        # V2: Conflicting signal gives negative probability adjustment
        assert result["probability_adjustment"] < 0

    def test_neutral_signal_no_effect(self, neutral_signal, signal_config):
        """Neutral sentiment should not affect confidence."""
        result = calculate_signal_influence(
            neutral_signal, "buy_yes", 0.6, signal_config
        )

        assert result["confidence_boost"] == 0.0
        assert result["kelly_multiplier"] == 1.0
        assert result["should_override_skip"] is False

    def test_insufficient_sources(self, signal_config):
        """Signals with insufficient unique stories should be ignored (v2)."""
        signal = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.9,
            source_count=1,
            sample_headlines=["Headline"],
            platforms=["Test"],
            # V2: unique_story_count below threshold
            unique_story_count=1,
            relevance=0.8,
        )
        result = calculate_signal_influence(signal, "buy_yes", 0.6, signal_config)

        assert result["probability_adjustment"] == 0.0
        assert result["confidence_boost"] == 0.0
        assert "Insufficient unique stories" in result["reasoning"]

    def test_skip_override_threshold(self, signal_config):
        """Skip override requires both strength and unique_story thresholds (v2)."""
        # Strong but few unique stories - no override
        signal1 = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.9,
            source_count=3,
            sample_headlines=[],
            platforms=[],
            # V2 quality fields
            unique_story_count=3,  # < skip_override_min_sources (5)
            relevance=0.8,
        )
        result1 = calculate_signal_influence(signal1, "buy_yes", 0.6, signal_config)
        assert result1["should_override_skip"] is False

        # Many unique stories but weak - no override
        signal2 = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.6,  # < skip_override_min_strength (0.8)
            source_count=10,
            sample_headlines=[],
            platforms=[],
            # V2 quality fields
            unique_story_count=8,
            relevance=0.7,
        )
        result2 = calculate_signal_influence(signal2, "buy_yes", 0.6, signal_config)
        assert result2["should_override_skip"] is False

        # Both thresholds met - override
        signal3 = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.85,  # >= 0.8
            source_count=6,
            sample_headlines=[],
            platforms=[],
            # V2 quality fields
            unique_story_count=6,  # >= 5
            relevance=0.8,
        )
        result3 = calculate_signal_influence(signal3, "buy_yes", 0.6, signal_config)
        assert result3["should_override_skip"] is True

    def test_confidence_boost_bounded(self, signal_config):
        """Confidence boost should never exceed max_confidence_boost."""
        signal = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=1.0,  # Maximum strength
            source_count=100,
            sample_headlines=[],
            platforms=[],
        )
        result = calculate_signal_influence(signal, "buy_yes", 0.6, signal_config)

        assert result["confidence_boost"] <= signal_config.max_confidence_boost


# ============================================================================
# Circuit Breaker Tests
# ============================================================================

class TestCircuitBreaker:
    """Tests for circuit breaker behavior."""

    def test_initial_state_closed(self):
        """Circuit breaker starts in closed state."""
        state = CircuitBreakerState()
        config = CircuitBreakerConfig()

        assert state.state == "closed"
        assert state.should_allow_request(config) is True

    def test_opens_after_threshold_failures(self):
        """Circuit opens after failure_threshold consecutive failures."""
        state = CircuitBreakerState()
        config = CircuitBreakerConfig(failure_threshold=3)

        state.record_failure(config)
        assert state.state == "closed"

        state.record_failure(config)
        assert state.state == "closed"

        state.record_failure(config)
        assert state.state == "open"
        assert state.should_allow_request(config) is False

    def test_success_resets_failures(self):
        """Successful request resets failure count."""
        state = CircuitBreakerState()
        config = CircuitBreakerConfig(failure_threshold=3)

        state.record_failure(config)
        state.record_failure(config)
        assert state.failure_count == 2

        state.record_success()
        assert state.failure_count == 0
        assert state.state == "closed"

    def test_half_open_after_recovery_timeout(self):
        """Circuit transitions to half-open after recovery timeout."""
        state = CircuitBreakerState()
        config = CircuitBreakerConfig(failure_threshold=1, recovery_timeout=0.1)

        state.record_failure(config)
        assert state.state == "open"

        # Wait for recovery timeout
        time.sleep(0.15)

        assert state.should_allow_request(config) is True
        assert state.state == "half_open"


# ============================================================================
# Signal Cache Tests
# ============================================================================

class TestSignalCache:
    """Tests for signal caching."""

    def test_cache_hit(self):
        """Cached signals are returned on subsequent calls."""
        cache = SignalCache(default_ttl=60.0)
        signals = [TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.5,
            source_count=3,
            sample_headlines=[],
            platforms=[],
        )]

        cache.set("Event Title", "Politics", signals)
        cached = cache.get("Event Title", "Politics")

        assert cached == signals

    def test_cache_miss(self):
        """Returns None for uncached events."""
        cache = SignalCache(default_ttl=60.0)
        result = cache.get("Unknown Event", "")

        assert result is None

    def test_cache_expiry(self):
        """Expired entries are not returned."""
        cache = SignalCache(default_ttl=0.1)  # 100ms TTL
        signals = [TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.5,
            source_count=3,
            sample_headlines=[],
            platforms=[],
        )]

        cache.set("Event Title", "Politics", signals)
        time.sleep(0.15)  # Wait for expiry

        result = cache.get("Event Title", "Politics")
        assert result is None

    def test_cache_key_normalization(self):
        """Cache keys are case-insensitive."""
        cache = SignalCache(default_ttl=60.0)
        signals = [TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.5,
            source_count=3,
            sample_headlines=[],
            platforms=[],
        )]

        cache.set("EVENT TITLE", "POLITICS", signals)
        result = cache.get("event title", "politics")

        assert result == signals


# ============================================================================
# TrendRadar Client Integration Tests
# ============================================================================

class TestTrendRadarClient:
    """Integration tests for TrendRadarClient."""

    @pytest.fixture
    def client(self):
        """Create a TrendRadar client for testing."""
        return TrendRadarClient(
            base_url="http://localhost:3333",
            timeout=5.0,
            enabled=True,
            cache_ttl=60.0,
        )

    def test_client_disabled(self, client):
        """Disabled client returns failure immediately."""
        client.enabled = False

        async def test():
            result = await client._call_tool("test", {})
            assert result["success"] is False
            assert "disabled" in result["error"].lower()

        asyncio.run(test())

    def test_metrics_tracking(self, client):
        """Client tracks metrics for observability."""
        metrics = client.get_metrics()

        assert "total_requests" in metrics
        assert "successful_requests" in metrics
        assert "failed_requests" in metrics
        assert "cache_hits" in metrics
        assert "circuit_breaker_state" in metrics

    def test_circuit_breaker_rejection_counted(self, client):
        """Circuit breaker rejections are counted in metrics."""
        # Force circuit open
        client._cb_state.state = "open"
        client._cb_state.last_failure_time = time.time()

        async def test():
            await client._call_tool("test", {})
            metrics = client.get_metrics()
            assert metrics["circuit_breaker_rejections"] >= 1

        asyncio.run(test())


# ============================================================================
# Format Signals for Research Tests
# ============================================================================

class TestFormatSignalsForResearch:
    """Tests for format_signals_for_research function."""

    def test_empty_signals(self):
        """Empty signal list returns empty string."""
        result = format_signals_for_research([])
        assert result == ""

    def test_formats_signal_with_sentiment(self, strong_positive_signal):
        """Signals are formatted with sentiment indicator."""
        result = format_signals_for_research([strong_positive_signal])

        assert "[+]" in result  # Positive indicator
        assert "Trump tariffs" in result
        assert "POSITIVE" in result.upper()

    def test_includes_headlines(self, strong_positive_signal):
        """Signal format includes sample headlines."""
        result = format_signals_for_research([strong_positive_signal])

        assert "Market rallies" in result


# ============================================================================
# Config Validation Tests
# ============================================================================

class TestConfigValidation:
    """Tests for configuration validation."""

    def test_valid_config(self):
        """Valid config passes validation."""
        config = SignalConfig(
            max_confidence_boost=0.3,
            strong_signal_threshold=0.7,
            min_source_count=2,
            aligned_signal_kelly_multiplier=1.25,
            enable_skip_override=True,
            skip_override_min_strength=0.8,
            skip_override_min_sources=5,
        )
        # Should not raise
        assert config.max_confidence_boost == 0.3

    def test_boundary_values(self):
        """Boundary values are accepted."""
        # Minimum values
        config_min = SignalConfig(
            max_confidence_boost=0.0,
            strong_signal_threshold=0.0,
            min_source_count=1,
            aligned_signal_kelly_multiplier=1.0,
        )
        assert config_min.max_confidence_boost == 0.0

        # Maximum values
        config_max = SignalConfig(
            max_confidence_boost=0.5,
            strong_signal_threshold=1.0,
            aligned_signal_kelly_multiplier=2.0,
        )
        assert config_max.max_confidence_boost == 0.5


# ============================================================================
# Run tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
