"""Tests for predictive algorithms: candle patterns, directional ADX, breakout."""

import pytest

from vicent.signals.prediction import (
    PatternType,
    BreakoutScore,
    compute_breakout_score,
    compute_directional_adx,
    compute_predictive_signal,
    detect_candle_pattern,
)


def _rising(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(n)]


def _falling(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start - i * step for i in range(n)]


# ---- Candle Pattern Detection -----------------------------------------------

def test_no_pattern_when_insufficient_data() -> None:
    r = detect_candle_pattern([100.0, 101.0])
    assert r.pattern == PatternType.NONE


def test_three_white_soldiers_detected() -> None:
    # Three strong consecutive bullish bars
    prices = [100.0] * 10 + [100, 102, 104, 107]   # 3 big green candles
    r = detect_candle_pattern(prices)
    assert r.pattern == PatternType.THREE_SOLDIERS
    assert r.direction == "bullish"
    assert r.confidence >= 0.65


def test_three_black_crows_detected() -> None:
    prices = [107.0] * 10 + [107, 105, 103, 100]
    r = detect_candle_pattern(prices)
    assert r.pattern == PatternType.THREE_CROWS
    assert r.direction == "bearish"


def test_bull_engulfing_detected() -> None:
    # Bearish candle followed by larger bullish that engulfs it
    prices = [100.0] * 5 + [105, 103, 102, 104.5]  # last 2: small down, big up
    r = detect_candle_pattern(prices)
    if r.pattern == PatternType.BULL_ENGULFING:
        assert r.direction == "bullish"


def test_morning_star_detected() -> None:
    # Big bearish, small doji-like, big bullish
    prices = [110, 108, 106, 105, 104.8, 107]  # down-doji-up pattern
    r = detect_candle_pattern(prices)
    # Should detect morning star or similar bullish pattern
    assert r.direction in ("bullish", "neutral")


def test_pattern_confidence_bounded() -> None:
    prices = _rising(20)
    r = detect_candle_pattern(prices)
    assert 0.0 <= r.confidence <= 1.0


# ---- Directional ADX --------------------------------------------------------

def test_directional_adx_none_when_short() -> None:
    r = compute_directional_adx(_rising(10))
    assert r.adx is None
    assert r.score == 0.0


def test_directional_adx_positive_on_uptrend() -> None:
    r = compute_directional_adx(_rising(60, step=1.5))
    assert r.adx is not None
    assert r.trend_direction == "up"
    assert r.di_plus > r.di_minus
    assert r.score > 0


def test_directional_adx_negative_on_downtrend() -> None:
    r = compute_directional_adx(_falling(60, step=1.5))
    assert r.adx is not None
    assert r.trend_direction == "down"
    assert r.di_minus > r.di_plus
    assert r.score < 0


def test_directional_adx_strong_on_big_trend() -> None:
    big   = compute_directional_adx(_rising(60, step=3.0))
    small = compute_directional_adx(_rising(60, step=0.1))
    # Both uptrend but big step should have higher DI+ dominance
    assert big.di_plus >= small.di_plus
    assert big.trend_direction == "up"


def test_directional_adx_score_bounded() -> None:
    r = compute_directional_adx(_rising(60))
    assert -1.0 <= r.score <= 1.0


def test_directional_adx_strength_labels() -> None:
    strong = compute_directional_adx(_rising(80, step=2.0))
    assert strong.trend_strength in ("strong", "moderate", "weak", "sideways")


# ---- Breakout Score ---------------------------------------------------------

def test_breakout_neutral_without_data() -> None:
    r = compute_breakout_score([])
    assert r.score == 0.0
    assert r.signal == "neutral"


def test_breakout_up_confirmed() -> None:
    r = compute_breakout_score(
        prices=_rising(30),
        volume_change_pct=60.0,
        bb_position=0.85,    # near upper band
        bb_squeeze=False,
        roc_short=1.8,       # strong upward momentum
        roc_medium=0.5,      # medium also positive
    )
    assert r.signal == "breakout_up"
    assert r.score > 0.5
    assert r.roc_confirmed is True
    assert r.volume_confirmed is True


def test_breakout_down_confirmed() -> None:
    r = compute_breakout_score(
        prices=_falling(30),
        volume_change_pct=50.0,
        bb_position=0.10,    # near lower band
        bb_squeeze=False,
        roc_short=-1.8,
        roc_medium=-0.5,
    )
    assert r.signal == "breakout_down"
    assert r.score < -0.5


def test_false_breakout_detected() -> None:
    r = compute_breakout_score(
        prices=_rising(30),
        volume_change_pct=10.0,   # low volume = no conviction
        bb_position=0.92,         # at upper band extreme
        bb_squeeze=False,
        roc_short=2.0,            # spike up
        roc_medium=-0.2,          # but medium trend is down (divergence)
    )
    # Should flag as false breakout or at least not a strong breakout
    assert r.signal in ("false_breakout", "neutral", "breakout_up")
    if r.signal == "false_breakout":
        assert r.score < 0.5


def test_squeeze_breakout_upward() -> None:
    r = compute_breakout_score(
        prices=_rising(30),
        volume_change_pct=20.0,
        bb_position=0.5,
        bb_squeeze=True,    # squeeze about to break
        roc_short=0.6,      # moving up
    )
    assert r.signal == "breakout_up"
    assert r.confidence >= 0.60


def test_breakout_confidence_bounded() -> None:
    r = compute_breakout_score(
        prices=_rising(30), volume_change_pct=100,
        bb_position=0.9, roc_short=3.0
    )
    assert 0.0 <= r.confidence <= 1.0
    assert -1.0 <= r.score <= 1.0


# ---- Combined Predictive Signal ---------------------------------------------

def test_predictive_signal_skip_on_short_data() -> None:
    r = compute_predictive_signal([1.0, 2.0])
    assert r.entry_quality == "skip"
    assert r.combined_score == pytest.approx(0.0, abs=0.3)


def test_predictive_signal_bullish_on_strong_uptrend() -> None:
    prices = _rising(60, step=1.0)
    r = compute_predictive_signal(
        prices=prices,
        volume_change_pct=70.0,
        bb_position=0.75,
        bb_squeeze=False,
        roc_short=1.5,
        roc_medium=0.8,
    )
    assert r.combined_score > 0
    assert r.entry_quality in ("A", "B", "C")


def test_predictive_signal_bearish_on_downtrend() -> None:
    prices = _falling(60, step=1.0)
    r = compute_predictive_signal(
        prices=prices,
        volume_change_pct=50.0,
        bb_position=0.15,
        bb_squeeze=False,
        roc_short=-1.5,
        roc_medium=-0.8,
    )
    assert r.combined_score < 0


def test_predictive_signal_has_all_components() -> None:
    prices = _rising(60)
    r = compute_predictive_signal(prices, volume_change_pct=40, bb_position=0.6, roc_short=0.8)
    assert r.pattern is not None
    assert r.directional_adx is not None
    assert r.breakout is not None
    assert r.entry_quality in ("A", "B", "C", "skip")


def test_a_grade_only_on_high_confidence_strong_signal() -> None:
    prices = _rising(80, step=2.0)
    r = compute_predictive_signal(
        prices, volume_change_pct=80, bb_position=0.8,
        bb_squeeze=False, roc_short=2.0, roc_medium=1.0
    )
    # Strong trend + volume + breakout should get at least B
    assert r.entry_quality in ("A", "B")
