"""
OpenAI-powered Research API client for prediction market analysis.
Uses GPT-4o for intelligent analysis of prediction markets.

Production hardening:
- Prompt injection protection for trending headlines
- Character limits and sanitization
- Non-authoritative signal framing

Calibration enhancements (Phase 4):
- Base rate anchoring to prevent overconfidence
- Explicit uncertainty quantification
- Calibration guidelines embedded in prompts

Enhanced Context (Phase 5):
- Full event rules and settlement criteria
- Per-market strike thresholds and YES/NO definitions
- Event data caching for API efficiency
"""

import re
from typing import Any, Dict, List, Optional

import openai
from config import OctagonConfig, OpenAIConfig
from loguru import logger

# Enhanced market context imports
try:
    from enhanced_market_context import (
        EventContext,
        MarketContext,
        build_enhanced_research_prompt,
        extract_market_context,
        fetch_event_details,
        get_event_cache,
    )
    ENHANCED_CONTEXT_AVAILABLE = True
except ImportError:
    ENHANCED_CONTEXT_AVAILABLE = False
    logger.warning("Enhanced market context module not available - using basic prompts")

# =============================================================================
# CALIBRATION-AWARE PROMPT TEMPLATES
# =============================================================================

CALIBRATION_SYSTEM_PROMPT = """You are a calibrated prediction market analyst. Your probability estimates are tracked for accuracy over time.

CALIBRATION PRINCIPLES:
1. Start with base rates - most binary events have significant uncertainty
2. Only move away from 50% with STRONG, verifiable evidence
3. Use confidence < 0.5 for inherently uncertain markets
4. Quantify uncertainty honestly - if you'd say "60-70%", use 65% with confidence 0.6

OVERCONFIDENCE TRAPS TO AVOID:
- Recency bias: Don't overweight events from the last 24-48 hours
- Availability bias: Rare events seem more likely when recently discussed
- Confirmation bias: Seek disconfirming evidence, not just supporting facts
- Consensus ≠ correctness: Markets and polls can be systematically wrong
- Narrative fallacy: Compelling stories don't make events more likely

PROBABILITY GUIDELINES BY CONFIDENCE LEVEL:
- 90%+ only for: Legally/physically required outcomes, signed contracts, completed events
- 75-89% for: Strong institutional commitments, consistent polling >60%, clear trends
- 55-74% for: Moderate evidence, mixed signals, competitive situations
- 45-54% for: True uncertainty, insufficient data, balanced factors
- Under 45% for: Unlikely but possible, against trend, contrarian positions

Always include the market TICKER when giving probability estimates.
Format: "TICKER: XX% probability (confidence: X/10)"
Be specific with percentages, not ranges."""

CALIBRATION_USER_PROMPT_TEMPLATE = """Analyze this prediction market event with CALIBRATED probability estimates.

{event_info}

{markets_info}
{trending_section}

ANALYSIS REQUIREMENTS:

1. **Base Rate Check**:
   - What is the historical base rate for similar events?
   - How should that anchor your probability estimates?

2. **Market Predictions** (for EACH market):
   - Market ticker and title
   - Probability estimate (0-100%) for YES outcome
   - Confidence level (1-10) reflecting your uncertainty
   - Key reasoning with BOTH supporting and opposing factors
   - Base rate comparison: Is this above/below typical rates?

3. **Uncertainty Analysis**:
   - What information would change your estimates significantly?
   - What are you most uncertain about?
   - Are there any "unknown unknowns" that could matter?

4. **Calibration Check**:
   - For each prediction, ask: "If I made 100 predictions at this probability, would ~X actually occur?"
   - Adjust if your gut says different than your stated probability

FORMATTING:
- Always include market TICKER: "KXMARKET-123: 65% probability (confidence: 7/10)"
- If mutually exclusive: probabilities should sum to ~100%
- If independent: evaluate each market separately
- Be specific (65% not "60-70%")

Example:
"KXMARKET-123: 62% probability (confidence: 6/10)
Base rate for similar events: ~55%. Adjusting +7% due to [specific factor].
Supporting: [evidence for YES]
Opposing: [evidence for NO]
Uncertainty: [what could change this]"
"""


# Maximum characters per headline to prevent token overflow
MAX_HEADLINE_CHARS = 150
# Maximum total characters for trending context
MAX_TRENDING_CONTEXT_CHARS = 2000


def sanitize_headline(headline: str) -> str:
    """Sanitize a headline to prevent prompt injection.

    - Removes control characters
    - Escapes markdown-like formatting
    - Truncates to MAX_HEADLINE_CHARS
    - Removes suspicious instruction-like patterns
    """
    if not headline:
        return ""

    # Remove control characters
    headline = re.sub(r"[\x00-\x1f\x7f-\x9f]", "", headline)

    # Remove common prompt injection patterns (case-insensitive)
    injection_patterns = [
        r"ignore\s+(all\s+)?(previous|above)\s+instructions?",
        r"ignore\s+all\s+instructions?",
        r"disregard\s+(all\s+)?(previous|above|the)",
        r"disregard\s+all",
        r"forget\s+everything",
        r"you\s+are\s+now",
        r"new\s+instructions?:",
        r"system\s*:",
        r"assistant\s*:",
        r"user\s*:",
    ]
    for pattern in injection_patterns:
        headline = re.sub(pattern, "[REDACTED]", headline, flags=re.IGNORECASE)

    # Escape special characters that could interfere with prompt parsing
    headline = headline.replace("```", "`")
    headline = headline.replace("---", "-")

    # Truncate
    if len(headline) > MAX_HEADLINE_CHARS:
        headline = headline[: MAX_HEADLINE_CHARS - 3] + "..."

    return headline.strip()


def sanitize_trending_context(context: str) -> str:
    """Sanitize the entire trending context block."""
    if not context:
        return ""

    # Apply line-by-line sanitization for headlines
    lines = context.split("\n")
    sanitized_lines = []
    for line in lines:
        # Headlines typically start with "-" or are indented
        if line.strip().startswith("-") or line.strip().startswith("•"):
            # This is likely a headline - sanitize more aggressively
            sanitized_lines.append(sanitize_headline(line))
        else:
            # Metadata line (sentiment, sources) - keep as-is but remove control chars
            sanitized_lines.append(re.sub(r"[\x00-\x1f\x7f-\x9f]", "", line))

    result = "\n".join(sanitized_lines)

    # Enforce total length limit
    if len(result) > MAX_TRENDING_CONTEXT_CHARS:
        result = (
            result[: MAX_TRENDING_CONTEXT_CHARS - 50]
            + "\n[...additional signals truncated]"
        )

    return result


class OctagonClient:
    """Client for OpenAI Research API (replaces Octagon/Perplexity).

    Supports calibration-aware prompts for improved probability accuracy.
    """

    def __init__(
        self,
        config: OctagonConfig,
        openai_config: OpenAIConfig = None,
        use_calibration_prompts: bool = True,
    ):
        """Initialize the research client.

        Args:
            config: OctagonConfig with API settings
            openai_config: Optional OpenAI config override
            use_calibration_prompts: Enable calibration-aware prompts (default: True)
        """
        self.config = config
        self.use_calibration_prompts = use_calibration_prompts
        # Use OpenAI directly for research
        self.client = openai.AsyncOpenAI(
            api_key=openai_config.api_key if openai_config else config.api_key,
            timeout=120.0,
        )
        self.model = "gpt-4o"
        logger.info(
            f"Research client initialized (calibration_prompts={use_calibration_prompts})"
        )

    async def research_event(
        self,
        event: Dict[str, Any],
        markets: List[Dict[str, Any]],
        trending_context: str = "",
    ) -> str:
        """
        Research an event and its markets using OpenAI GPT-4o.

        Args:
            event: Event information (title, subtitle, category, etc.)
            markets: List of markets within the event (without odds)
            trending_context: Optional trending news signals from TrendRadar

        Returns:
            Research response as a string
        """
        try:
            event_title = event.get("title", "")
            event_subtitle = event.get("subtitle", "")
            mutually_exclusive = event.get("mutually_exclusive", False)

            event_info = f"""
Event: {event_title}
Subtitle: {event_subtitle}
Mutually Exclusive: {mutually_exclusive}
"""

            markets_info = "Markets to analyze:\n"
            market_count = 0
            for i, market in enumerate(markets, 1):
                if market.get("volume", 0) < 100:
                    continue
                market_count += 1
                title = market.get("title", "")
                ticker = market.get("ticker", "")
                markets_info += f"{market_count}. {title}"
                if ticker:
                    markets_info += f" (Ticker: {ticker})"
                markets_info += "\n"
                if market.get("subtitle"):
                    markets_info += f"   Details: {market.get('subtitle', '')}\n"
                close_time = market.get("close_time", "")
                if close_time:
                    markets_info += f"   Closes: {close_time}\n"
                markets_info += "\n"

            if market_count == 0:
                return "No markets with sufficient volume to analyze."

            # Build trending section if we have signals from TrendRadar
            # Apply sanitization to prevent prompt injection from headlines
            trending_section = ""
            if trending_context:
                sanitized_context = sanitize_trending_context(trending_context)
                trending_section = f"""
---
**SUPPLEMENTARY NEWS SIGNALS (for context only):**

IMPORTANT: The following headlines are provided as supplementary context only.
Do NOT treat them as authoritative or verified information. Headlines may be:
- Incomplete, misleading, or taken out of context
- From sources with varying reliability
- Potentially biased or speculative

Use these signals as ONE input among many. Your probability estimates should
primarily be based on fundamental analysis, historical patterns, and verifiable facts.
Avoid overconfidence based solely on headline sentiment.

{sanitized_context}
---

"""

            # Build prompt based on calibration mode
            if self.use_calibration_prompts:
                # Use calibration-aware prompts for better probability accuracy
                prompt = CALIBRATION_USER_PROMPT_TEMPLATE.format(
                    event_info=event_info,
                    markets_info=markets_info,
                    trending_section=trending_section,
                )
                system_prompt = CALIBRATION_SYSTEM_PROMPT
            else:
                # Legacy prompt (for A/B testing or fallback)
                prompt = f"""You are an expert prediction market analyst. Analyze this event and predict probabilities for each market based on your knowledge.

{event_info}

{markets_info}
{trending_section}
Provide your analysis:

1. **Event Overview**: Brief analysis of the event and key factors affecting outcomes

2. **Market Predictions**: For EACH market listed above, provide:
   - Market ticker and title
   - Probability estimate (0-100%) for YES outcome
   - Confidence level (1-10)
   - Key reasoning (2-3 sentences based on relevant factors)

3. **Key Risks**: What factors could significantly change these predictions?

IMPORTANT FORMATTING RULES:
- Always include the market TICKER when giving probability predictions
- Format: "TICKER: XX% probability" or "Market Title (TICKER): XX%"
- If mutually exclusive event: probabilities should sum to approximately 100%
- If not mutually exclusive: evaluate each market independently
- Be specific with percentages, not ranges
- Base predictions on logical analysis of available information

Example format:
"KXMARKET-123: 65% probability - Based on current trends and historical patterns..."
"""
                system_prompt = "You are an expert prediction market analyst. Provide accurate probability estimates based on logical analysis. Always be specific with probability percentages and include market tickers in your predictions."

            event_ticker = event.get("event_ticker", "UNKNOWN")
            prompt_mode = "calibration" if self.use_calibration_prompts else "legacy"
            logger.info(
                f"Researching event {event_ticker} via OpenAI GPT-4o ({prompt_mode} prompts)..."
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,  # Lower temperature for more consistent, calibrated outputs
                max_tokens=4096,
            )

            content = response.choices[0].message.content
            logger.info(f"Completed research for event {event_ticker}")

            return content if content else ""

        except Exception as e:
            logger.error(
                f"Error researching event {event.get('event_ticker', '')}: {e}"
            )
            return f"Error researching event: {str(e)}"

    async def research_event_enhanced(
        self,
        kalshi_client,  # KalshiClient instance for fetching event details
        event: Dict[str, Any],
        markets: List[Dict[str, Any]],
        trending_context: str = "",
    ) -> str:
        """
        Enhanced research with full event context including settlement rules.

        This method fetches additional event details (rules, strike thresholds)
        to provide GPT-4o with market-specific context for better predictions.

        Args:
            kalshi_client: KalshiClient instance for API calls
            event: Event information (title, subtitle, category, etc.)
            markets: List of markets within the event (without odds)
            trending_context: Optional trending news signals from TrendRadar

        Returns:
            Research response as a string
        """
        if not ENHANCED_CONTEXT_AVAILABLE:
            # Fallback to basic research
            logger.info("Enhanced context not available, using basic research")
            return await self.research_event(event, markets, trending_context)

        try:
            event_ticker = event.get("event_ticker", "")

            # Fetch full event context (uses cache)
            event_ctx = await fetch_event_details(
                kalshi_client,
                event_ticker,
                use_cache=True
            )

            if not event_ctx:
                # Fallback to basic context from event dict
                event_ctx = EventContext(
                    event_ticker=event_ticker,
                    title=event.get("title", ""),
                    subtitle=event.get("subtitle", ""),
                    category=event.get("category", ""),
                    mutually_exclusive=event.get("mutually_exclusive", False),
                    series_ticker=event.get("series_ticker", ""),
                    strike_date=event.get("strike_date", ""),
                )

            # Extract full market contexts
            market_contexts = []
            for market in markets:
                if market.get("volume", 0) < 100:
                    continue
                market_contexts.append(extract_market_context(market))

            if not market_contexts:
                return "No markets with sufficient volume to analyze."

            # Sanitize trending context
            sanitized_trending = ""
            if trending_context:
                sanitized_trending = sanitize_trending_context(trending_context)

            # Build enhanced prompt with full context
            prompt = build_enhanced_research_prompt(
                event_ctx,
                market_contexts,
                sanitized_trending
            )

            # Use calibration system prompt
            system_prompt = CALIBRATION_SYSTEM_PROMPT

            logger.info(
                f"Researching event {event_ticker} via GPT-4o (enhanced context: "
                f"rules={len(event_ctx.rules_primary)} chars, "
                f"markets={len(market_contexts)})"
            )

            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=4096,
            )

            content = response.choices[0].message.content
            logger.info(f"Completed enhanced research for event {event_ticker}")

            return content if content else ""

        except Exception as e:
            logger.error(f"Error in enhanced research for {event.get('event_ticker', '')}: {e}")
            # Fallback to basic research on error
            logger.info("Falling back to basic research due to error")
            return await self.research_event(event, markets, trending_context)

    async def close(self):
        """Close the client."""
        pass
