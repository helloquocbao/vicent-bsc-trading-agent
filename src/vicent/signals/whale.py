"""Whale / holder distribution signal — parses CMC get_crypto_metrics.

What we're looking for:
  - Whale accumulation: large-wallet count growing + whale % of supply increasing
  - Retail FOMO: trader addresses (hold < 30d) growing fast → late-cycle
  - HODLer conviction: long-term holder % high → strong hands, less sell pressure
  - Whale distribution: whale % very high + trader % high → smart money exiting

Output: WhalePressure with score [-1, +1]
  +1 = whales accumulating, HODLers strong, traders low (early bull)
   0 = neutral / data unavailable
  -1 = whales distributing, traders piling in (late cycle / dump risk)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog

log = structlog.get_logger(__name__)


@dataclass
class WhalePressure:
    """Whale and holder-distribution signal for a token."""
    symbol: str
    score: float           # -1.0 to +1.0
    confidence: float      # 0.0 to 1.0 (data completeness)
    whale_pct: float       # % supply held by large wallets
    trader_pct: float      # % held by short-term traders (<30d)
    hodler_pct: float      # % held by long-term holders (>1y)
    address_growth: float  # 24h new address growth %
    data_available: bool   # False if CMC returned empty metrics


def score_whale(symbol: str, metrics: dict[str, Any]) -> WhalePressure:
    """Score whale/holder pressure from CMC crypto_metrics data."""

    if not metrics:
        return _neutral(symbol, data_available=False)

    data = metrics.get("data", metrics)
    if not data:
        return _neutral(symbol, data_available=False)

    # --- Extract holder distribution ---
    # CMC get_crypto_metrics returns holder_breakdown or similar
    holder_data = (
        data.get("holder_breakdown")
        or data.get("holders")
        or data.get("holder_distribution")
        or {}
    )

    whale_pct = _pct(holder_data, [
        "whale_percent", "large_holder_pct", "whales_pct",
        "top_10_percent", "percent_in_top_10",
    ])
    trader_pct = _pct(holder_data, [
        "trader_percent", "short_term_holder_pct", "traders_pct",
    ])
    hodler_pct = _pct(holder_data, [
        "hodler_percent", "long_term_holder_pct", "hodlers_pct",
        "holder_1y_pct",
    ])

    # Address growth
    addr_data = data.get("addresses", data.get("address_metrics", {}))
    addr_growth = _float(addr_data, [
        "growth_24h", "new_addresses_24h_pct", "address_growth_pct",
    ])

    # --- Score calculation ---
    if whale_pct == 0 and trader_pct == 0 and hodler_pct == 0:
        return _neutral(symbol, data_available=True)

    score = 0.0
    data_points = 0

    # 1. HODLer conviction (weight 0.35)
    # High long-term holders = strong hands, bullish
    if hodler_pct > 0:
        if hodler_pct > 50:
            score += 0.35
        elif hodler_pct > 30:
            score += 0.20
        elif hodler_pct > 15:
            score += 0.05
        data_points += 1

    # 2. Whale behaviour (weight 0.35)
    # Moderate whale concentration = institutional confidence (bullish)
    # Very high whale % + high trader % = dump setup (bearish)
    if whale_pct > 0:
        if whale_pct > 60 and trader_pct > 30:
            score -= 0.35  # pump & dump risk
        elif whale_pct > 40:
            score += 0.10  # some concentration, neutral-ish
        elif 20 < whale_pct <= 40:
            score += 0.25  # healthy institutional participation
        elif whale_pct <= 20:
            score += 0.15  # well distributed, good sign
        data_points += 1

    # 3. Trader % (weight 0.20)
    # High short-term trader participation = FOMO = late cycle = caution
    if trader_pct > 0:
        if trader_pct > 50:
            score -= 0.20  # hot money dominant, sell risk
        elif trader_pct > 30:
            score -= 0.10
        elif trader_pct < 15:
            score += 0.10  # low trader activity = early or steady
        data_points += 1

    # 4. Address growth (weight 0.10)
    # Moderate new address growth = healthy adoption
    # Very high growth = FOMO peak
    if addr_growth != 0:
        if 2 < addr_growth <= 10:
            score += 0.10  # healthy growth
        elif addr_growth > 10:
            score -= 0.05  # possible FOMO peak
        elif addr_growth < 0:
            score -= 0.05  # contraction
        data_points += 1

    # Normalise and compute confidence from data completeness
    confidence = round(min(1.0, data_points / 3.0 * 0.8), 3)

    log.info(
        "whale_scored",
        symbol=symbol,
        score=round(score, 3),
        confidence=confidence,
        whale_pct=whale_pct,
        trader_pct=trader_pct,
        hodler_pct=hodler_pct,
        addr_growth=addr_growth,
    )

    return WhalePressure(
        symbol=symbol,
        score=round(max(-1.0, min(1.0, score)), 3),
        confidence=confidence,
        whale_pct=whale_pct,
        trader_pct=trader_pct,
        hodler_pct=hodler_pct,
        address_growth=addr_growth,
        data_available=True,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _neutral(symbol: str, data_available: bool) -> WhalePressure:
    return WhalePressure(
        symbol=symbol, score=0.0, confidence=0.0,
        whale_pct=0.0, trader_pct=0.0, hodler_pct=0.0,
        address_growth=0.0, data_available=data_available,
    )


def _pct(data: dict, keys: list[str]) -> float:
    """Try multiple key names, return first found as a 0-100 float."""
    for k in keys:
        v = data.get(k)
        if v is not None:
            try:
                f = float(v)
                # Normalise: some APIs return 0-1, others 0-100
                if f <= 1.0 and f > 0:
                    return f * 100
                return f
            except (TypeError, ValueError):
                continue
    return 0.0


def _float(data: dict, keys: list[str]) -> float:
    for k in keys:
        v = data.get(k)
        if v is not None:
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return 0.0
