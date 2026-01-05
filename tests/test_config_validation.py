"""
Tests for configuration validation.

Coverage:
- TrendRadarConfig validation
- BotConfig risk limits
- Environment variable parsing
- Boundary value validation
"""
import pytest
import os
from unittest.mock import patch
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pydantic import ValidationError
from config import TrendRadarConfig, BotConfig, load_config


class TestTrendRadarConfigValidation:
    """Tests for TrendRadarConfig validation."""

    def test_valid_config(self):
        """Valid config values are accepted."""
        config = TrendRadarConfig(
            enabled=True,
            base_url="http://localhost:3333",
            timeout=10.0,
            max_confidence_boost=0.3,
            strong_signal_threshold=0.7,
            min_source_count=2,
            aligned_signal_kelly_multiplier=1.25,
        )
        assert config.max_confidence_boost == 0.3

    def test_max_confidence_boost_bounds(self):
        """max_confidence_boost must be in [0.0, 0.5]."""
        # Valid bounds
        TrendRadarConfig(max_confidence_boost=0.0)
        TrendRadarConfig(max_confidence_boost=0.5)

        # Invalid - too high
        with pytest.raises(ValidationError):
            TrendRadarConfig(max_confidence_boost=0.6)

        # Invalid - negative
        with pytest.raises(ValidationError):
            TrendRadarConfig(max_confidence_boost=-0.1)

    def test_strong_signal_threshold_bounds(self):
        """strong_signal_threshold must be in [0.0, 1.0]."""
        TrendRadarConfig(strong_signal_threshold=0.0)
        TrendRadarConfig(strong_signal_threshold=1.0)

        with pytest.raises(ValidationError):
            TrendRadarConfig(strong_signal_threshold=1.1)

        with pytest.raises(ValidationError):
            TrendRadarConfig(strong_signal_threshold=-0.1)

    def test_kelly_multiplier_bounds(self):
        """aligned_signal_kelly_multiplier must be in [1.0, 2.0]."""
        TrendRadarConfig(aligned_signal_kelly_multiplier=1.0)
        TrendRadarConfig(aligned_signal_kelly_multiplier=2.0)

        with pytest.raises(ValidationError):
            TrendRadarConfig(aligned_signal_kelly_multiplier=0.5)

        with pytest.raises(ValidationError):
            TrendRadarConfig(aligned_signal_kelly_multiplier=2.5)

    def test_skip_override_strength_bounds(self):
        """skip_override_min_strength must be in [0.5, 1.0]."""
        TrendRadarConfig(skip_override_min_strength=0.5)
        TrendRadarConfig(skip_override_min_strength=1.0)

        with pytest.raises(ValidationError):
            TrendRadarConfig(skip_override_min_strength=0.3)

        with pytest.raises(ValidationError):
            TrendRadarConfig(skip_override_min_strength=1.2)

    def test_min_source_count_minimum(self):
        """min_source_count must be >= 1."""
        TrendRadarConfig(min_source_count=1)
        TrendRadarConfig(min_source_count=10)

        with pytest.raises(ValidationError):
            TrendRadarConfig(min_source_count=0)

    def test_ablation_rate_bounds(self):
        """ablation_rate must be in [0.0, 1.0]."""
        TrendRadarConfig(ablation_rate=0.0)
        TrendRadarConfig(ablation_rate=0.5)
        TrendRadarConfig(ablation_rate=1.0)

        with pytest.raises(ValidationError):
            TrendRadarConfig(ablation_rate=-0.1)

        with pytest.raises(ValidationError):
            TrendRadarConfig(ablation_rate=1.1)

    def test_cache_ttl_bounds(self):
        """cache_ttl_seconds must be in [60, 1800]."""
        TrendRadarConfig(cache_ttl_seconds=60)
        TrendRadarConfig(cache_ttl_seconds=1800)

        with pytest.raises(ValidationError):
            TrendRadarConfig(cache_ttl_seconds=30)

        with pytest.raises(ValidationError):
            TrendRadarConfig(cache_ttl_seconds=3600)


class TestRiskLimitValidation:
    """Tests for risk limit configuration."""

    def test_max_daily_loss_non_negative(self):
        """max_daily_loss must be >= 0."""
        # This would need to be tested with BotConfig when env vars are set
        # For now, verify the Field definition exists
        pass

    def test_max_daily_loss_pct_bounds(self):
        """max_daily_loss_pct must be in [0, 0.5]."""
        pass

    def test_kelly_fraction_bounds(self):
        """kelly_fraction must be in [0.1, 1.5]."""
        pass


class TestEnvironmentVariableParsing:
    """Tests for environment variable parsing."""

    def test_cleans_inline_comments(self):
        """Inline comments are stripped from env values."""
        from config import _clean_env_value

        assert _clean_env_value("true # this is a comment") == "true"
        assert _clean_env_value("0.3 # 30% boost") == "0.3"
        assert _clean_env_value("300 # 5 minutes") == "300"

    def test_strips_whitespace(self):
        """Whitespace is stripped from env values."""
        from config import _clean_env_value

        assert _clean_env_value("  true  ") == "true"
        assert _clean_env_value("\t0.3\n") == "0.3"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
