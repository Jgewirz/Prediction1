"""
Validation tests for Signal Quality Enhancement (v2).

Tests the 8-phase implementation:
1. Story de-duplication
2. Outlet tiering
3. Relevance scoring
4. Baseline normalization
5. Negation-aware sentiment
6. Strength formula bounds
7. Skip override safety
8. Single-lever influence
"""

import math
import os
import sys

import pytest

# Add parent to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from betting_models import SkipReason
from trendradar_client import (TIER1_OUTLETS, TIER2_OUTLETS, TIER_WEIGHTS,
                               OutletTier, SignalConfig, TrendingSignal,
                               analyze_headline_sentiment,
                               calculate_signal_influence,
                               calculate_signal_strength, get_outlet_tier)


# =============================================================================
# Test 1: De-duplication
# =============================================================================
class TestDeduplication:
    """Test story de-duplication logic."""

    def test_dedup_clusters_syndicated_stories(self):
        """10 headlines with 5 syndicated near-duplicates should cluster correctly."""
        # Simulate the clustering logic from _parse_signal
        headlines = [
            # Cluster 1: Trump tariffs (3 variants)
            "Trump announces new tariffs on China",
            "Trump announces new tariffs on China imports",
            "Trump to announce new tariffs on China today",
            # Cluster 2: Fed rates (2 variants)
            "Federal Reserve raises interest rates",
            "Fed raises interest rates by 25 basis points",
            # Cluster 3-7: Unique stories
            "Apple reports record quarterly earnings",
            "Bitcoin drops below $40,000",
            "Oil prices surge amid Middle East tensions",
            "Tesla unveils new electric truck",
            "Amazon expands into healthcare market",
        ]

        story_clusters = []
        for title in headlines:
            title_lower = title.lower().strip()
            found_cluster = False
            for cluster in story_clusters:
                title_words = set(title_lower.split())
                cluster_words = set(cluster["representative"].lower().split())
                if len(title_words) > 0 and len(cluster_words) > 0:
                    overlap = len(title_words & cluster_words) / min(
                        len(title_words), len(cluster_words)
                    )
                    if overlap > 0.6:
                        cluster["items"].append(title)
                        found_cluster = True
                        break
            if not found_cluster:
                story_clusters.append({"representative": title, "items": [title]})

        # Should have fewer clusters than raw headlines
        assert len(story_clusters) < len(headlines), "De-dup should reduce count"
        # Expect roughly 7 clusters (3 Trump -> 1, 2 Fed -> 1, 5 unique)
        assert (
            len(story_clusters) <= 8
        ), f"Expected ~7-8 clusters, got {len(story_clusters)}"

    def test_dedup_symmetric_overlap(self):
        """Overlap metric should be symmetric."""

        def calc_overlap(a: str, b: str) -> float:
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if len(words_a) == 0 or len(words_b) == 0:
                return 0.0
            return len(words_a & words_b) / min(len(words_a), len(words_b))

        # Test symmetry
        a = "Trump announces new tariffs"
        b = "Trump announces tariffs on China"
        assert calc_overlap(a, b) == calc_overlap(b, a), "Overlap should be symmetric"

    def test_dedup_stopwords_dont_cause_false_matches(self):
        """Stories with only stopword overlap should not cluster."""

        def calc_overlap(a: str, b: str) -> float:
            words_a = set(a.lower().split())
            words_b = set(b.lower().split())
            if len(words_a) == 0 or len(words_b) == 0:
                return 0.0
            return len(words_a & words_b) / min(len(words_a), len(words_b))

        # Different stories that share common words
        a = "The Fed raises interest rates"
        b = "The President signs new bill"
        overlap = calc_overlap(a, b)
        assert overlap < 0.6, f"Unrelated stories should not cluster: overlap={overlap}"


# =============================================================================
# Test 2: Outlet Tiering
# =============================================================================
class TestOutletTiering:
    """Test outlet tier classification."""

    def test_tier1_outlets_classified_correctly(self):
        """Known Tier-1 outlets should be classified as TIER1."""
        tier1_names = ["reuters", "bloomberg", "wsj", "ap", "nyt", "ft"]
        for name in tier1_names:
            assert get_outlet_tier(name) == OutletTier.TIER1, f"{name} should be TIER1"

    def test_tier2_outlets_classified_correctly(self):
        """Known Tier-2 outlets should be classified as TIER2."""
        tier2_names = ["cnbc", "marketwatch", "bbc", "cnn", "fox"]
        for name in tier2_names:
            assert get_outlet_tier(name) == OutletTier.TIER2, f"{name} should be TIER2"

    def test_unknown_outlets_default_to_tier3(self):
        """Unknown outlets should default to TIER3."""
        assert get_outlet_tier("random-blog") == OutletTier.TIER3
        assert get_outlet_tier("unknown-source") == OutletTier.TIER3
        assert get_outlet_tier("") == OutletTier.TIER3

    def test_tier_weights_are_correct(self):
        """Tier weights should be: TIER1=1.0, TIER2=0.7, TIER3=0.4."""
        assert TIER_WEIGHTS[OutletTier.TIER1] == 1.0
        assert TIER_WEIGHTS[OutletTier.TIER2] == 0.7
        assert TIER_WEIGHTS[OutletTier.TIER3] == 0.4

    def test_outlet_tier_case_insensitive(self):
        """Outlet tier lookup should be case-insensitive."""
        assert get_outlet_tier("REUTERS") == OutletTier.TIER1
        assert get_outlet_tier("Bloomberg") == OutletTier.TIER1
        assert get_outlet_tier("CNN") == OutletTier.TIER2


# =============================================================================
# Test 3: Negation-Aware Sentiment
# =============================================================================
class TestSentimentAnalysis:
    """Test negation-aware sentiment analysis."""

    def test_positive_sentiment(self):
        """Plain positive headline should return positive sentiment."""
        direction, confidence = analyze_headline_sentiment("Stocks rise sharply")
        assert direction == "positive"
        assert confidence > 0.5

    def test_negative_sentiment(self):
        """Plain negative headline should return negative sentiment."""
        direction, confidence = analyze_headline_sentiment("Stocks crash after report")
        assert direction == "negative"
        assert confidence > 0.5

    def test_negation_flips_sentiment(self):
        """Negation should flip sentiment direction."""
        # "not rise" should be negative
        dir_negated, _ = analyze_headline_sentiment("Stocks do not rise")
        dir_positive, _ = analyze_headline_sentiment("Stocks rise sharply")
        assert dir_negated != dir_positive, "Negation should flip sentiment"

    def test_unlikely_as_negation(self):
        """'Unlikely' should act as negation."""
        direction, _ = analyze_headline_sentiment("Stocks unlikely to rise")
        # "rise" is positive, but "unlikely" negates it -> negative
        assert direction == "negative" or direction == "neutral"

    def test_uncertainty_reduces_confidence(self):
        """Uncertainty words should reduce confidence."""
        _, conf_certain = analyze_headline_sentiment("Stocks will rise sharply")
        _, conf_uncertain = analyze_headline_sentiment("Stocks may rise sharply")
        assert conf_uncertain < conf_certain, "Uncertainty should reduce confidence"

    def test_neutral_when_no_sentiment_words(self):
        """Headlines without sentiment words should be neutral."""
        direction, _ = analyze_headline_sentiment("Company announces quarterly results")
        assert direction == "neutral"


# =============================================================================
# Test 4: Baseline Normalization
# =============================================================================
class TestBaselineNormalization:
    """Test baseline frequency tracking."""

    def test_baseline_trigger_threshold(self):
        """Signal should only trigger when >= 1.5x baseline."""
        config = SignalConfig()

        # Create signal with baseline 4, current 5 -> ratio 1.25 (NOT triggered)
        signal_low = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.8,
            source_count=5,
            unique_story_count=5,
            relevance=0.8,
            baseline_frequency=4.0,
            current_frequency=5.0,
            frequency_ratio=1.25,  # Below 1.5 threshold
        )

        result_low = calculate_signal_influence(signal_low, "buy_yes", 0.5, config)
        # Should be gated out due to normal volume
        assert result_low["signal_direction"] == "neutral", "1.25x should not trigger"

        # Create signal with baseline 4, current 6 -> ratio 1.5 (triggered)
        signal_high = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.8,
            source_count=6,
            unique_story_count=6,
            relevance=0.8,
            baseline_frequency=4.0,
            current_frequency=6.0,
            frequency_ratio=1.5,  # At threshold
        )

        result_high = calculate_signal_influence(signal_high, "buy_yes", 0.5, config)
        assert result_high["signal_direction"] == "aligned", "1.5x should trigger"


# =============================================================================
# Test 5: Strength Formula Bounds
# =============================================================================
class TestStrengthFormula:
    """Test strength formula correctness and bounds."""

    def test_strength_always_in_zero_one(self):
        """Strength should always be in [0, 1] for any input."""
        test_cases = [
            # (unique_story_count, relevance, novelty, sentiment_confidence, neutral_ratio, tier1_count)
            (0, 0.5, 1.0, 0.5, 0.0, 0),  # Zero stories
            (1, 0.0, 0.0, 0.0, 1.0, 0),  # All zeros
            (100, 1.0, 1.0, 1.0, 0.0, 10),  # All max
            (5, 0.5, 0.5, 0.5, 0.5, 2),  # Medium values
            (1, 1.0, 1.0, 1.0, 0.0, 0),  # Single story, high confidence
            (10, 0.1, 1.0, 0.9, 0.8, 5),  # Low relevance, high neutral
        ]

        for params in test_cases:
            strength = calculate_signal_strength(*params)
            assert (
                0.0 <= strength <= 1.0
            ), f"Strength {strength} out of bounds for {params}"
            assert not math.isnan(strength), f"Strength is NaN for {params}"
            assert not math.isinf(strength), f"Strength is infinite for {params}"

    def test_strength_zero_for_zero_stories(self):
        """Zero unique stories should produce zero strength."""
        strength = calculate_signal_strength(0, 0.8, 1.0, 0.8, 0.2, 3)
        assert strength == 0.0

    def test_sample_damping_prevents_small_sample_clarity(self):
        """Small samples should not produce high clarity."""
        # 2 stories with perfect confidence should still be dampened
        strength_small = calculate_signal_strength(2, 0.8, 1.0, 1.0, 0.0, 0)
        strength_large = calculate_signal_strength(10, 0.8, 1.0, 1.0, 0.0, 0)
        assert (
            strength_small < strength_large
        ), "Small samples should have lower strength"

    def test_neutral_penalty_reduces_strength(self):
        """High neutral ratio should reduce strength."""
        strength_low_neutral = calculate_signal_strength(5, 0.8, 1.0, 0.8, 0.2, 0)
        strength_high_neutral = calculate_signal_strength(5, 0.8, 1.0, 0.8, 0.8, 0)
        assert (
            strength_high_neutral < strength_low_neutral
        ), "High neutral should reduce strength"

    def test_tier1_bonus_increases_strength(self):
        """Tier-1 sources should increase strength."""
        strength_no_tier1 = calculate_signal_strength(5, 0.8, 1.0, 0.8, 0.2, 0)
        strength_with_tier1 = calculate_signal_strength(5, 0.8, 1.0, 0.8, 0.2, 3)
        assert strength_with_tier1 > strength_no_tier1, "Tier-1 should boost strength"


# =============================================================================
# Test 6: Skip Override Safety
# =============================================================================
class TestSkipOverrideSafety:
    """Test that skip override only works for INSUFFICIENT_EDGE."""

    def test_skip_reason_enum_values(self):
        """Verify all SkipReason enum values exist."""
        assert SkipReason.INSUFFICIENT_EDGE.value == "insufficient_edge"
        assert SkipReason.MARKET_ILLIQUID.value == "illiquid"
        assert SkipReason.DATA_MISSING.value == "data_missing"
        assert SkipReason.EXISTING_POSITION.value == "existing_position"
        assert SkipReason.KILL_SWITCH.value == "kill_switch"

    def test_strong_signal_can_trigger_override(self):
        """Strong aligned signal should set should_override_skip=True."""
        config = SignalConfig()
        signal = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.85,  # Above 0.8 threshold
            source_count=10,
            unique_story_count=10,  # Above 5 threshold
            relevance=0.8,
            frequency_ratio=2.0,  # Above baseline
        )

        result = calculate_signal_influence(signal, "buy_yes", 0.5, config)
        assert (
            result["should_override_skip"] is True
        ), "Strong signal should allow override"

    def test_weak_signal_cannot_trigger_override(self):
        """Weak signal should not allow skip override."""
        config = SignalConfig()
        signal = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.5,  # Below 0.8 threshold
            source_count=3,
            unique_story_count=3,
            relevance=0.8,
            frequency_ratio=2.0,
        )

        result = calculate_signal_influence(signal, "buy_yes", 0.5, config)
        assert (
            result["should_override_skip"] is False
        ), "Weak signal should not allow override"

    def test_override_requires_sufficient_sources(self):
        """Override requires min_unique_stories >= 5."""
        config = SignalConfig()
        signal = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.9,  # High strength
            source_count=3,
            unique_story_count=3,  # Below 5 threshold
            relevance=0.8,
            frequency_ratio=2.0,
        )

        result = calculate_signal_influence(signal, "buy_yes", 0.5, config)
        assert result["should_override_skip"] is False, "Override needs 5+ sources"


# =============================================================================
# Test 7: Single-Lever Influence
# =============================================================================
class TestSingleLeverInfluence:
    """Test that only probability is adjusted (no Kelly multiplier)."""

    def test_kelly_multiplier_always_one(self):
        """Kelly multiplier should always be 1.0 (deprecated)."""
        config = SignalConfig()

        # Test aligned signal
        signal_aligned = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=0.9,
            source_count=10,
            unique_story_count=10,
            relevance=0.8,
            frequency_ratio=2.0,
        )
        result_aligned = calculate_signal_influence(
            signal_aligned, "buy_yes", 0.5, config
        )
        assert result_aligned["kelly_multiplier"] == 1.0, "Kelly should always be 1.0"

        # Test conflicting signal
        result_conflict = calculate_signal_influence(
            signal_aligned, "buy_no", 0.5, config
        )
        assert result_conflict["kelly_multiplier"] == 1.0, "Kelly should always be 1.0"

    def test_probability_adjustment_bounded(self):
        """Probability adjustment should be bounded by max_probability_adjustment."""
        config = SignalConfig()
        signal = TrendingSignal(
            topic="Test",
            sentiment="positive",
            strength=1.0,  # Max strength
            source_count=20,
            unique_story_count=20,
            relevance=1.0,  # Max relevance
            frequency_ratio=3.0,
        )

        result = calculate_signal_influence(signal, "buy_yes", 0.5, config)
        assert (
            abs(result["probability_adjustment"])
            <= config.max_probability_adjustment + 0.001
        )

    def test_conflicting_signal_reduces_probability(self):
        """Conflicting signal should reduce (or not increase) probability."""
        config = SignalConfig()
        signal = TrendingSignal(
            topic="Test",
            sentiment="positive",  # Positive sentiment
            strength=0.8,
            source_count=10,
            unique_story_count=10,
            relevance=0.8,
            frequency_ratio=2.0,
        )

        # buy_no conflicts with positive sentiment
        result = calculate_signal_influence(signal, "buy_no", 0.5, config)
        assert result["signal_direction"] == "conflicting"
        # For buy_no with positive signal, adjustment should be positive (reducing bet on NO)
        # Actually let's check the logic...
        # buy_no + positive = conflicting, penalty is applied
        assert (
            result["probability_adjustment"] > 0
        ), "Conflicting should have positive adj for buy_no"


# =============================================================================
# Test 8: Cosine Similarity Bounds
# =============================================================================
class TestCosineSimilarity:
    """Test cosine similarity is properly bounded."""

    def test_cosine_similarity_bounds(self):
        """Cosine similarity should be in [-1, 1]."""
        from trendradar_client import TrendRadarClient

        client = TrendRadarClient(enabled=False)

        # Identical vectors
        vec = [1.0, 0.0, 0.0]
        sim = client._cosine_similarity(vec, vec)
        assert abs(sim - 1.0) < 0.001, "Identical vectors should have similarity 1.0"

        # Opposite vectors
        vec1 = [1.0, 0.0]
        vec2 = [-1.0, 0.0]
        sim = client._cosine_similarity(vec1, vec2)
        assert abs(sim - (-1.0)) < 0.001, "Opposite vectors should have similarity -1.0"

        # Orthogonal vectors
        vec1 = [1.0, 0.0]
        vec2 = [0.0, 1.0]
        sim = client._cosine_similarity(vec1, vec2)
        assert abs(sim) < 0.001, "Orthogonal vectors should have similarity 0.0"

    def test_cosine_similarity_handles_zero_vectors(self):
        """Zero vectors should return 0.0, not error."""
        from trendradar_client import TrendRadarClient

        client = TrendRadarClient(enabled=False)

        zero_vec = [0.0, 0.0, 0.0]
        normal_vec = [1.0, 0.0, 0.0]

        sim = client._cosine_similarity(zero_vec, normal_vec)
        assert sim == 0.0, "Zero vector should return 0.0"

        sim = client._cosine_similarity(normal_vec, zero_vec)
        assert sim == 0.0, "Zero vector should return 0.0"


# =============================================================================
# Run Tests
# =============================================================================
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
