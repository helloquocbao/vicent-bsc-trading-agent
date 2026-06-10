"""Ensemble strategy — combines regime filter + token signals into a TradeDecision.

Three sub-strategies vote. A trade fires when:
  - Regime is BULL and confidence >= 0.50, OR
  - Regime is NEUTRAL and confidence >= 0.65 (higher bar), OR
  - Regime is BEAR → FLAT only (hold stables, minimum required trades only)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from vicent.signals.regime import Regime, RegimeScore
from vicent.signals.signals import Direction, TokenSignal

if TYPE_CHECKING:
    pass

log = structlog.get_logger(__name__)


@dataclass
class TradeDecision:
    should_trade: bool
    symbol: str
    direction: Direction
    confidence: float
    regime: Regime
    reason: str


def decide(
    signal: TokenSignal,
    regime: RegimeScore,
    min_confidence: float = 0.50,
) -> TradeDecision:
    """Core decision function — combine regime + signal → trade/no-trade.

    min_confidence is set dynamically by the reflexion layer.
    """

    # --- Regime gate ---
    # Futures-only mode: BEAR regime allows SHORT perps (profit from downtrend)
    # Only hard-block if signal is LONG during BEAR (don't fight the trend)
    if regime.regime == Regime.BEAR:
        if signal.direction == Direction.LONG:
            return TradeDecision(
                should_trade=False,
                symbol=signal.symbol,
                direction=Direction.FLAT,
                confidence=0.0,
                regime=regime.regime,
                reason="bear_regime_blocks_long",
            )
        # SHORT signals pass through in BEAR — this is the most profitable case
        if signal.direction == Direction.SHORT and signal.confidence >= min_confidence:
            adj_confidence = min(1.0, signal.confidence * 1.20)  # 20% boost for shorting in bear
            return TradeDecision(
                should_trade=True,
                symbol=signal.symbol,
                direction=Direction.SHORT,
                confidence=adj_confidence,
                regime=regime.regime,
                reason="bear_regime_short_boost",
            )
        return TradeDecision(
            should_trade=False,
            symbol=signal.symbol,
            direction=Direction.FLAT,
            confidence=0.0,
            regime=regime.regime,
            reason="bear_regime_flat",
        )

    # Minimum confidence threshold varies by regime AND reflexion feedback
    base_min = min_confidence
    if regime.regime == Regime.NEUTRAL:
        base_min = max(min_confidence, 0.55)  # futures-only: lower neutral bar

    if signal.direction == Direction.FLAT or signal.confidence < base_min:
        return TradeDecision(
            should_trade=False,
            symbol=signal.symbol,
            direction=Direction.FLAT,
            confidence=signal.confidence,
            regime=regime.regime,
            reason=f"low_signal: conf={signal.confidence:.2f} < {base_min:.2f}",
        )

    # --- Regime boost/penalty on confidence ---
    adj_confidence = signal.confidence
    if regime.regime == Regime.BULL and signal.direction == Direction.LONG:
        adj_confidence = min(1.0, signal.confidence * 1.10)  # 10% boost in bull
    elif regime.regime == Regime.NEUTRAL:
        adj_confidence = signal.confidence * 0.90  # 10% penalty in neutral

    # --- Predictive signal quality boost / penalty ---
    # Cap tổng boost để tránh compound vượt cap dẫn đến mất differentiation
    if signal.prediction is not None:
        quality = signal.prediction.entry_quality
        if quality == "A":
            adj_confidence = min(0.95, adj_confidence * 1.10)   # giảm từ 1.15 → 1.10
        elif quality == "B":
            adj_confidence = min(0.95, adj_confidence * 1.04)
        elif quality == "C":
            adj_confidence = adj_confidence * 0.92

    # --- Narrative alignment adjustments ---
    # Extreme greed + LONG = contrarian caution
    if regime.fear_greed > 80 and signal.direction == Direction.LONG:
        adj_confidence *= 0.85
        log.debug("fg_caution_applied", fg=regime.fear_greed)

    # Extreme oversold + LONG = strong conviction boost
    if signal.rsi_1h is not None and signal.rsi_1h < 30 and signal.direction == Direction.LONG:
        adj_confidence = min(0.95, adj_confidence * 1.10)   # giảm từ 1.15 → 1.10

    # Final cap — tránh saturate ở 1.0 (mất khả năng phân biệt giữa các signal mạnh)
    adj_confidence = min(0.99, adj_confidence)

    log.info(
        "trade_decided",
        symbol=signal.symbol,
        direction=signal.direction.value,
        raw_conf=round(signal.confidence, 3),
        adj_conf=round(adj_confidence, 3),
        regime=regime.regime.value,
    )

    return TradeDecision(
        should_trade=True,
        symbol=signal.symbol,
        direction=signal.direction,
        confidence=adj_confidence,
        regime=regime.regime,
        reason="signal_pass",
    )


def rank_decisions(decisions: list[TradeDecision]) -> list[TradeDecision]:
    """Return trade decisions sorted by adjusted confidence, best first."""
    tradeable = [d for d in decisions if d.should_trade]
    return sorted(tradeable, key=lambda d: d.confidence, reverse=True)
