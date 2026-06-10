"""Macro regime detector — reads global CMC metrics and derivatives data.

Returns a RegimeScore that other strategies use as a filter.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import structlog

log = structlog.get_logger(__name__)


class Regime(str, Enum):
    BULL = "bull"       # Risk-on, momentum longs preferred
    NEUTRAL = "neutral" # No clear edge, trade smaller
    BEAR = "bear"       # Risk-off, reduce exposure / hold stables


@dataclass
class RegimeScore:
    regime: Regime
    fear_greed: int        # 0-100
    btc_dominance: float   # %
    funding_rate: float    # avg perpetual funding rate
    open_interest_change: float  # 24h OI change %
    confidence: float      # 0-1


def detect_regime(
    global_metrics: dict[str, Any],
    derivatives: dict[str, Any],
) -> RegimeScore:
    """Compute macro regime from raw CMC data dicts."""

    # --- Fear & Greed ---
    fg = _extract_fear_greed(global_metrics)

    # --- BTC dominance ---
    btc_dom = _safe_float(global_metrics, ["data", "btc_dominance"], default=50.0)

    # --- Derivatives ---
    avg_funding = _extract_funding_rate(derivatives)
    oi_change = _safe_float(derivatives, ["data", "open_interest_24h_pct_change"], default=0.0)

    # --- Score → regime ---
    score = 0.0

    # Fear & Greed contribution (weight 0.40)
    if fg >= 65:
        score += 0.40  # Greed → bull
    elif fg >= 45:
        score += 0.20  # Neutral
    else:
        score -= 0.10  # Fear → bear

    # Funding rate contribution (weight 0.30)
    # Very positive funding = crowded longs = slightly bearish signal
    # Negative funding = shorts paying = bullish for spot
    if avg_funding < 0:
        score += 0.30  # shorts paying, bullish
    elif avg_funding < 0.01:
        score += 0.15  # mildly positive
    elif avg_funding < 0.03:
        score += 0.00  # neutral
    else:
        score -= 0.10  # very crowded longs, caution

    # OI change contribution (weight 0.15)
    if oi_change > 5:
        score += 0.10  # rising OI + price up = strong bull
    elif oi_change < -10:
        score -= 0.10  # liquidation cascade risk

    # BTC dominance (weight 0.15)
    # Falling BTC dom = alt season = more opportunity
    if btc_dom < 45:
        score += 0.15
    elif btc_dom < 55:
        score += 0.05
    else:
        score -= 0.05  # BTC dominance high = flight to safety

    # Map to regime
    if score >= 0.40:
        regime = Regime.BULL
    elif score >= 0.10:
        regime = Regime.NEUTRAL
    else:
        regime = Regime.BEAR

    confidence = min(1.0, max(0.0, (score + 0.5)))

    log.info(
        "regime_detected",
        regime=regime.value,
        fear_greed=fg,
        btc_dom=btc_dom,
        avg_funding=avg_funding,
        oi_change=oi_change,
        score=round(score, 3),
    )

    return RegimeScore(
        regime=regime,
        fear_greed=fg,
        btc_dominance=btc_dom,
        funding_rate=avg_funding,
        open_interest_change=oi_change,
        confidence=confidence,
    )


# ---- helpers ----------------------------------------------------------------

def _extract_fear_greed(data: dict[str, Any]) -> int:
    try:
        # New CMC MCP format: data.sentiment.fear_greed.current.index
        fg = data.get("sentiment", {}).get("fear_greed", {}).get("current", {})
        if fg:
            return int(fg.get("index", fg.get("value", 50)))
        # Legacy format
        value = data["data"]["fear_and_greed_index"]["value"]
        return int(value)
    except (KeyError, TypeError, ValueError):
        try:
            return int(data["data"]["fear_and_greed_value"])
        except (KeyError, TypeError, ValueError):
            log.warning("fear_greed_extract_failed", fallback=50)
            return 50  # neutral default — logged so it's visible


def _extract_funding_rate(derivatives: dict[str, Any]) -> float:
    try:
        # New CMC MCP format
        data = derivatives
        # Try nested data key
        if "data" in derivatives:
            data = derivatives["data"]
        rates = data.get("funding_rate")
        if rates is not None:
            if isinstance(rates, list):
                return sum(r.get("rate", 0.0) for r in rates) / max(len(rates), 1)
            return float(rates)
        return 0.0   # neutral default — never bias when data unavailable
    except (KeyError, TypeError):
        return 0.0   # neutral — not 0.01 which biases toward BEAR


def _safe_float(data: dict, path: list[str], default: float = 0.0) -> float:
    try:
        node = data
        for key in path:
            node = node[key]
        return float(node)
    except (KeyError, TypeError, ValueError):
        return default
