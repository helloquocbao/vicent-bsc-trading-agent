"""Tests for the adaptive market-neutral strategy selector."""

import pytest

from vicent.signals.indicators import IntradayIndicators
from vicent.strategy.adaptive import (
    StrategyMode,
    apply_adaptive_to_composite,
    select_strategy,
)


def _intraday(
    adx: float | None = 30.0,
    bb_position: float | None = 0.5,
    rsi: float | None = 50.0,
    roc_short: float = 0.0,
    has_enough: bool = True,
) -> IntradayIndicators:
    return IntradayIndicators(
        rsi=rsi, stoch_rsi=50.0, macd_signal="neutral", macd_histogram=0.0,
        ema_trend="flat", roc_short=roc_short, roc_medium=0.0,
        bb_upper=110, bb_lower=90, bb_mid=100, bb_position=bb_position, bb_squeeze=False,
        atr=2.0, atr_pct=2.0, adx=adx, trend_strong=(adx or 0) > 25,
        vwap=100, vwap_position="near", n_bars=50, has_enough_data=has_enough,
    )


# ---- Mode selection ---------------------------------------------------------

def test_no_data_returns_no_trade() -> None:
    d = select_strategy(None)
    assert d.mode == StrategyMode.NO_TRADE


def test_insufficient_data_no_trade() -> None:
    d = select_strategy(_intraday(has_enough=False))
    assert d.mode == StrategyMode.NO_TRADE


def test_high_adx_is_trend_follow() -> None:
    d = select_strategy(_intraday(adx=35, roc_short=0.5))
    assert d.mode == StrategyMode.TREND_FOLLOW


def test_low_adx_is_mean_revert() -> None:
    d = select_strategy(_intraday(adx=12, bb_position=0.9, rsi=75))
    assert d.mode == StrategyMode.MEAN_REVERT


def test_mid_adx_is_transition() -> None:
    d = select_strategy(_intraday(adx=22, roc_short=0.7))
    assert d.mode == StrategyMode.TRANSITION


# ---- Trend-follow direction -------------------------------------------------

def test_trend_follow_long_on_up_momentum() -> None:
    d = select_strategy(_intraday(adx=35, roc_short=1.0, bb_position=0.75))
    assert d.directional_bias > 0   # LONG bias


def test_trend_follow_short_on_down_momentum() -> None:
    d = select_strategy(_intraday(adx=35, roc_short=-1.0, bb_position=0.25))
    assert d.directional_bias < 0   # SHORT bias


# ---- Mean-reversion fades extremes ------------------------------------------

def test_mean_revert_shorts_at_upper_band() -> None:
    # Range market, price at upper band → should SHORT (fade)
    d = select_strategy(_intraday(adx=12, bb_position=0.90, rsi=72))
    assert d.mode == StrategyMode.MEAN_REVERT
    assert d.directional_bias < 0   # SHORT (fade the top)


def test_mean_revert_longs_at_lower_band() -> None:
    # Range market, price at lower band → should LONG (fade)
    d = select_strategy(_intraday(adx=12, bb_position=0.10, rsi=28))
    assert d.mode == StrategyMode.MEAN_REVERT
    assert d.directional_bias > 0   # LONG (fade the bottom)


def test_mean_revert_no_trade_at_mid_range() -> None:
    # Range market but price in the middle → no clear fade
    d = select_strategy(_intraday(adx=12, bb_position=0.5, rsi=50))
    assert d.mode == StrategyMode.MEAN_REVERT
    assert abs(d.directional_bias) < 0.3


# ---- Funding rate capture ---------------------------------------------------

def test_positive_funding_leans_short() -> None:
    d = select_strategy(_intraday(adx=30, roc_short=0.5), funding_rate=0.05)
    assert d.funding_bias < 0   # high positive funding → SHORT earns funding


def test_negative_funding_leans_long() -> None:
    d = select_strategy(_intraday(adx=30, roc_short=0.5), funding_rate=-0.05)
    assert d.funding_bias > 0   # negative funding → LONG earns funding


def test_neutral_funding_no_bias() -> None:
    d = select_strategy(_intraday(adx=30), funding_rate=0.005)
    assert d.funding_bias == 0.0


# ---- Composite blending -----------------------------------------------------

def test_apply_trend_follow_reinforces() -> None:
    from vicent.strategy.adaptive import AdaptiveDecision
    adaptive = AdaptiveDecision(
        mode=StrategyMode.TREND_FOLLOW, directional_bias=0.6,
        confidence_mult=1.1, funding_bias=0.0, reason="test",
    )
    # base composite positive + bias positive → reinforced
    result = apply_adaptive_to_composite(0.3, adaptive)
    assert result > 0.3


def test_apply_mean_revert_flips_composite() -> None:
    from vicent.strategy.adaptive import AdaptiveDecision
    # Composite says LONG (momentum up) but we're at upper band → fade to SHORT
    adaptive = AdaptiveDecision(
        mode=StrategyMode.MEAN_REVERT, directional_bias=-0.7,
        confidence_mult=0.95, funding_bias=0.0, reason="fade_top",
    )
    result = apply_adaptive_to_composite(0.3, adaptive)
    # Mean-revert bias (0.6 weight) should overpower base composite (0.4 weight)
    assert result < 0   # flipped to SHORT


def test_apply_no_trade_returns_base() -> None:
    from vicent.strategy.adaptive import AdaptiveDecision
    adaptive = AdaptiveDecision(
        mode=StrategyMode.NO_TRADE, directional_bias=0.0,
        confidence_mult=1.0, funding_bias=0.0, reason="no_data",
    )
    assert apply_adaptive_to_composite(0.42, adaptive) == 0.42


def test_composite_bounded() -> None:
    from vicent.strategy.adaptive import AdaptiveDecision
    adaptive = AdaptiveDecision(
        mode=StrategyMode.TREND_FOLLOW, directional_bias=1.0,
        confidence_mult=1.2, funding_bias=1.0, reason="extreme",
    )
    result = apply_adaptive_to_composite(1.0, adaptive)
    assert -1.0 <= result <= 1.0
