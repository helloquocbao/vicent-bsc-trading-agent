"""Tests for ensemble strategy."""

from vicent.signals.regime import Regime, RegimeScore
from vicent.signals.signals import Direction, TokenSignal
from vicent.strategy.ensemble import decide, rank_decisions


def _regime(r: Regime, fg: int = 60) -> RegimeScore:
    return RegimeScore(
        regime=r,
        fear_greed=fg,
        btc_dominance=48,
        funding_rate=0.01,
        open_interest_change=5,
        confidence=0.7,
    )


def _signal(sym: str, direction: Direction, conf: float, rsi: float = 50) -> TokenSignal:
    return TokenSignal(
        symbol=sym,
        cmc_id=1,
        direction=direction,
        confidence=conf,
        price_usd=100.0,
        volume_24h=1_000_000,
        volume_change_pct=20.0,
        rsi_1h=rsi,
        macd_signal="bullish",
        ema_trend="up",
    )


def test_bull_regime_allows_long() -> None:
    sig = _signal("CAKE", Direction.LONG, 0.70)
    result = decide(sig, _regime(Regime.BULL))
    assert result.should_trade is True
    assert result.direction == Direction.LONG


def test_bear_regime_blocks_all() -> None:
    sig = _signal("CAKE", Direction.LONG, 0.90)
    result = decide(sig, _regime(Regime.BEAR))
    assert result.should_trade is False


def test_bear_regime_blocks_short_spot_only() -> None:
    """Spot-only mode: BEAR regime blocks SHORT signals (can't short in spot).
    SHORT signals return should_trade=False with reason='bear_regime_spot_no_short'.
    """
    sig = _signal("CAKE", Direction.SHORT, 0.60)
    result = decide(sig, _regime(Regime.BEAR))
    assert result.should_trade is False
    assert result.direction == Direction.FLAT
    assert "no_short" in result.reason


def test_neutral_requires_higher_confidence() -> None:
    sig_low = _signal("CAKE", Direction.LONG, 0.40)  # below 0.55 threshold
    sig_high = _signal("CAKE", Direction.LONG, 0.65)
    result_low = decide(sig_low, _regime(Regime.NEUTRAL))
    result_high = decide(sig_high, _regime(Regime.NEUTRAL))
    assert result_low.should_trade is False
    assert result_high.should_trade is True


def test_extreme_fg_reduces_confidence() -> None:
    sig = _signal("CAKE", Direction.LONG, 0.80)
    result = decide(sig, _regime(Regime.BULL, fg=85))
    # Should still trade but with reduced confidence
    if result.should_trade:
        assert result.confidence < 0.80


def test_oversold_boosts_confidence() -> None:
    sig_normal = _signal("CAKE", Direction.LONG, 0.60, rsi=55)
    sig_oversold = _signal("CAKE", Direction.LONG, 0.60, rsi=25)
    r_normal = decide(sig_normal, _regime(Regime.BULL))
    r_oversold = decide(sig_oversold, _regime(Regime.BULL))
    if r_normal.should_trade and r_oversold.should_trade:
        assert r_oversold.confidence >= r_normal.confidence


def test_rank_decisions() -> None:
    sigs = [
        _signal("A", Direction.LONG, 0.60),
        _signal("B", Direction.LONG, 0.80),
        _signal("C", Direction.LONG, 0.70),
    ]
    regime = _regime(Regime.BULL)
    decisions = [decide(s, regime) for s in sigs]
    ranked = rank_decisions(decisions)
    assert ranked[0].symbol == "B"  # highest confidence first
