"""
Pydantic models for structured betting decisions.

Includes:
- BettingDecision: Single market decision with risk metrics
- MarketAnalysis: Aggregate analysis results
- SkipReason: Enum for why a bet was skipped (Phase 7)
"""
from enum import Enum
from typing import List, Optional, Literal
from pydantic import BaseModel, Field


# =============================================================================
# Phase 7: Skip Reason Tracking for Safe Override
# =============================================================================
class SkipReason(str, Enum):
    """
    Reasons why a bet was skipped.

    CRITICAL: Signal override should ONLY apply to INSUFFICIENT_EDGE skips.
    Other skip reasons indicate structural problems that signals cannot fix.
    """
    INSUFFICIENT_EDGE = "insufficient_edge"    # R-score below threshold
    MARKET_ILLIQUID = "illiquid"               # Wide bid-ask spread
    DATA_MISSING = "data_missing"              # No price data available
    EXISTING_POSITION = "existing_position"    # Already have position
    KILL_SWITCH = "kill_switch"                # Daily loss limit hit
    MANUAL_EXCLUDE = "manual_exclude"          # User-configured exclusion
    LOW_CONFIDENCE = "low_confidence"          # GPT confidence too low
    MARKET_CLOSED = "market_closed"            # Market already settled
    BELOW_MIN_BET = "below_min_bet"            # Kelly bet below minimum


class MarketProbability(BaseModel):
    """Structured probability data for a single market."""
    ticker: str = Field(..., description="The market ticker symbol")
    title: str = Field(..., description="Human-readable market title")
    research_probability: float = Field(..., ge=0, le=100, description="Research predicted probability (0-100)")
    reasoning: str = Field(..., description="Brief reasoning for the probability estimate")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in the probability estimate (0-1)")


class ProbabilityExtraction(BaseModel):
    """Structured extraction of probabilities from research."""
    markets: List[MarketProbability] = Field(..., description="List of market probabilities")
    overall_summary: str = Field(..., description="Overall research summary and key insights")


class BettingDecision(BaseModel):
    """A single betting decision for a market."""
    ticker: str = Field(..., description="The market ticker symbol")
    action: Literal["buy_yes", "buy_no", "skip"] = Field(..., description="Action to take")
    confidence: float = Field(..., ge=0, le=1, description="Confidence in decision (0-1)")
    amount: float = Field(..., ge=0, description="Amount to bet in dollars")
    reasoning: str = Field(..., description="Brief reasoning for the decision")
    
    # Human-readable names for display
    event_name: Optional[str] = Field(None, description="Human-readable event name")
    market_name: Optional[str] = Field(None, description="Human-readable market name")
    
    # Hedging fields
    is_hedge: bool = Field(False, description="Whether this is a hedge bet")
    hedge_for: Optional[str] = Field(None, description="Ticker of the main bet this hedges")
    hedge_ratio: Optional[float] = Field(None, ge=0, le=1, description="Proportion of main bet this hedges")
    
    # Risk-adjusted metrics (hedge-fund style)
    expected_return: Optional[float] = Field(None, description="Expected return on capital E[R] = (p-y)/y")
    r_score: Optional[float] = Field(None, description="Risk-adjusted edge: (p-y)/sqrt(p*(1-p)) - the z-score")
    kelly_fraction: Optional[float] = Field(None, description="Optimal Kelly fraction for position sizing")
    market_price: Optional[float] = Field(None, description="Market price used for calculations (0-1)")
    research_probability: Optional[float] = Field(None, description="Research probability used for calculations (0-1)")

    # Run tracking
    run_mode: Optional[str] = Field(None, description="'dry_run' or 'live'")
    run_id: Optional[str] = Field(None, description="Unique run identifier")

    # TrendRadar signal influence
    signal_applied: bool = Field(False, description="Whether a TrendRadar signal was applied")
    signal_direction: Optional[str] = Field(None, description="'aligned', 'conflicting', or 'neutral'")
    signal_topic: Optional[str] = Field(None, description="Trending topic matched")
    signal_sentiment: Optional[str] = Field(None, description="'positive', 'negative', or 'neutral'")
    signal_strength: Optional[float] = Field(None, ge=0, le=1, description="Signal strength 0.0 to 1.0")
    signal_source_count: Optional[int] = Field(None, ge=0, description="Number of news sources")
    confidence_boost: float = Field(0.0, description="Confidence adjustment from signal")
    kelly_multiplier: float = Field(1.0, description="Position size multiplier from signal")
    override_skip_triggered: bool = Field(False, description="Whether skip was overridden by signal")
    signal_reasoning: Optional[str] = Field(None, description="Explanation of signal impact")

    # NEW v2: Enhanced signal metrics
    signal_unique_stories: Optional[int] = Field(None, description="Unique stories after de-dup")
    signal_unique_outlets: Optional[int] = Field(None, description="Unique news outlets")
    signal_tier1_count: Optional[int] = Field(None, description="Tier-1 outlet count")
    signal_relevance: Optional[float] = Field(None, description="Semantic relevance to event")
    signal_frequency_ratio: Optional[float] = Field(None, description="Current vs baseline frequency")
    probability_adjustment: float = Field(0.0, description="Probability shift from signal (single-lever)")

    # Phase 7: Skip reason tracking for safe override
    skip_reason: Optional[str] = Field(None, description="Why the bet was skipped (SkipReason enum value)")
    override_blocked: bool = Field(False, description="True if signal tried to override but was blocked")
    original_action: Optional[str] = Field(None, description="Original action before signal override")


class MarketAnalysis(BaseModel):
    """Analysis results for all markets."""
    decisions: List[BettingDecision] = Field(..., description="List of betting decisions")
    total_recommended_bet: float = Field(..., description="Total amount recommended to bet")
    high_confidence_bets: int = Field(..., description="Number of high confidence bets (>0.7)")
    summary: str = Field(..., description="Overall market summary and strategy") 