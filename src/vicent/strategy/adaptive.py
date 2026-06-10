"""Adaptive strategy selector — profit in ANY market condition.

The core idea behind market-neutral futures trading:
  Different market modes need OPPOSITE strategies.

  TREND mode (ADX > 25):
    Price moves in one direction with conviction.
    → TREND-FOLLOWING: go LONG on uptrends, SHORT on downtrends.
    → Buying breakouts works, fading extremes loses.

  RANGE mode (ADX < 20):
    Price oscillates between support/resistance, no clear direction.
    → MEAN-REVERSION: fade the extremes.
    → SHORT when price hits BB upper / RSI overbought.
    → LONG when price hits BB lower / RSI oversold.
    → Buying breakouts loses (false breakouts), fading works.

  TRANSITION mode (ADX 20-25):
    Unclear. Trade smaller, require higher conviction.

Plus FUNDING RATE CAPTURE (perps-specific edge):
  Perps charge funding every 8h. When funding is very positive,
  longs pay shorts → shorting earns funding on top of price moves.
  When very negative, longs get paid → longing earns funding.
  This is a market-neutral income stream independent of price direction.

This module returns a StrategyMode + a directional bias adjustment that
the scoring layer applies. The result: the agent makes money whether
the market trends up, trends down, OR chops sideways.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from vicent.signals.indicators import IntradayIndicators

log = structlog.get_logger(__name__)


class StrategyMode(str, Enum):
    TREND_FOLLOW   = "trend_follow"    # ADX high — ride the trend
    MEAN_REVERT    = "mean_revert"     # ADX low — fade extremes
    TRANSITION     = "transition"      # unclear — cautious
    NO_TRADE       = "no_trade"        # conditions too poor


@dataclass
class AdaptiveDecision:
    mode: StrategyMode
    directional_bias: float    # -1 (short) to +1 (long), 0 = neutral
    confidence_mult: float     # multiplier on final confidence
    funding_bias: float        # -1 to +1, funding-rate-driven directional tilt
    reason: str


def select_strategy(
    intraday: "IntradayIndicators | None",
    funding_rate: float = 0.0,
) -> AdaptiveDecision:
    """Decide which strategy mode applies and compute directional bias.

    Returns AdaptiveDecision used by the scoring layer to:
      1. Pick trend-follow vs mean-revert sub-scoring
      2. Apply funding-rate directional tilt
      3. Adjust final confidence
    """
    if intraday is None or not intraday.has_enough_data:
        return AdaptiveDecision(
            mode=StrategyMode.NO_TRADE,
            directional_bias=0.0,
            confidence_mult=1.0,
            funding_bias=0.0,
            reason="insufficient_data",
        )

    adx = intraday.adx if intraday.adx is not None else 0.0
    bb_pos = intraday.bb_position
    rsi = intraday.rsi
    roc_short = intraday.roc_short

    # ------------------------------------------------------------------ #
    # Funding rate bias (perps-specific, market-neutral edge)             #
    # ------------------------------------------------------------------ #
    # Positive funding = longs pay shorts → SHORT earns funding income
    # Negative funding = shorts pay longs → LONG earns funding income
    funding_bias = 0.0
    if funding_rate > 0.03:
        funding_bias = -0.3   # strongly positive funding → lean SHORT
    elif funding_rate > 0.01:
        funding_bias = -0.15
    elif funding_rate < -0.03:
        funding_bias = 0.3    # strongly negative funding → lean LONG
    elif funding_rate < -0.01:
        funding_bias = 0.15

    # ------------------------------------------------------------------ #
    # Mode selection based on ADX                                         #
    # ------------------------------------------------------------------ #

    # === TREND MODE: ADX > 25 → follow direction ===
    if adx > 25:
        # Direction comes from ROC + BB position in trend context
        bias = 0.0
        if roc_short > 0.3:
            bias = 0.6   # upward momentum in a trend → LONG
        elif roc_short < -0.3:
            bias = -0.6  # downward momentum in a trend → SHORT
        # BB position confirms: in strong trend, riding the band is OK
        if bb_pos is not None:
            if bb_pos > 0.7 and bias > 0:
                bias = min(1.0, bias + 0.2)   # riding upper band in uptrend
            elif bb_pos < 0.3 and bias < 0:
                bias = max(-1.0, bias - 0.2)  # riding lower band in downtrend

        conf_mult = 1.2 if adx > 40 else 1.1   # stronger trend = more confidence
        return AdaptiveDecision(
            mode=StrategyMode.TREND_FOLLOW,
            directional_bias=round(bias, 3),
            confidence_mult=conf_mult,
            funding_bias=funding_bias,
            reason=f"trend_follow ADX={adx:.0f}",
        )

    # === RANGE MODE: ADX < 20 → fade extremes (mean reversion) ===
    if adx < 20:
        bias = 0.0
        # Fade BB extremes: price at upper band → SHORT, lower band → LONG
        if bb_pos is not None:
            if bb_pos > 0.85:
                bias = -0.7   # at upper band in a range → SHORT (expect pullback)
            elif bb_pos > 0.70:
                bias = -0.4
            elif bb_pos < 0.15:
                bias = 0.7    # at lower band in a range → LONG (expect bounce)
            elif bb_pos < 0.30:
                bias = 0.4
        # RSI extremes reinforce mean-reversion
        if rsi is not None:
            if rsi > 70 and bias <= 0:
                bias = min(bias - 0.2, -0.3)   # overbought → stronger SHORT
            elif rsi < 30 and bias >= 0:
                bias = max(bias + 0.2, 0.3)    # oversold → stronger LONG

        # In range, only trade when at a clear extreme (bias is meaningful)
        if abs(bias) < 0.3:
            return AdaptiveDecision(
                mode=StrategyMode.MEAN_REVERT,
                directional_bias=0.0,
                confidence_mult=0.6,
                funding_bias=funding_bias,
                reason=f"range_no_extreme ADX={adx:.0f} bb={bb_pos}",
            )

        return AdaptiveDecision(
            mode=StrategyMode.MEAN_REVERT,
            directional_bias=round(bias, 3),
            confidence_mult=0.95,   # mean-reversion slightly less confident than trend
            funding_bias=funding_bias,
            reason=f"mean_revert ADX={adx:.0f} fade_extreme bb={bb_pos}",
        )

    # === TRANSITION MODE: ADX 20-25 → cautious ===
    # Weak trend forming or trend dying. Require strong momentum confirmation.
    bias = 0.0
    if roc_short > 0.6:
        bias = 0.4
    elif roc_short < -0.6:
        bias = -0.4
    return AdaptiveDecision(
        mode=StrategyMode.TRANSITION,
        directional_bias=round(bias, 3),
        confidence_mult=0.75,   # reduce size in unclear conditions
        funding_bias=funding_bias,
        reason=f"transition ADX={adx:.0f}",
    )


def apply_adaptive_to_composite(
    base_composite: float,
    adaptive: AdaptiveDecision,
) -> float:
    """Blend the base TA composite with the adaptive directional bias.

    In TREND mode: composite and bias usually agree → reinforce.
    In RANGE mode: bias may OPPOSE composite (composite says 'up' from momentum
      but we're at upper band so we fade → SHORT). The adaptive bias wins
      because mean-reversion is the correct play in a range.

    Weighting:
      TREND:      70% base composite + 30% bias (trend-following dominates)
      MEAN_REVERT: 40% base composite + 60% bias (fade dominates)
      TRANSITION: 60% base + 40% bias
      NO_TRADE:   return base composite (fall back to pure TA)

    Note: funding_bias is a perps-only concept (funding rate payments on
    perpetual futures). In spot-only BSC mode it is always 0.0 and has
    no effect, but the field is retained for forward compatibility.
    """
    if adaptive.mode == StrategyMode.NO_TRADE:
        return base_composite   # fall back to pure TA when no intraday data

    if adaptive.mode == StrategyMode.TREND_FOLLOW:
        blended = base_composite * 0.70 + adaptive.directional_bias * 0.30
    elif adaptive.mode == StrategyMode.MEAN_REVERT:
        # Mean-reversion: the fade signal dominates
        blended = base_composite * 0.40 + adaptive.directional_bias * 0.60
    else:  # TRANSITION
        blended = base_composite * 0.60 + adaptive.directional_bias * 0.40

    # funding_bias is always 0.0 in spot mode (no perpetual funding).
    # Kept for structural completeness; no-op when 0.
    blended += adaptive.funding_bias * 0.15

    # Apply confidence multiplier
    blended *= adaptive.confidence_mult

    return max(-1.0, min(1.0, blended))
