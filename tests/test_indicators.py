"""Tests for intraday indicators computed from price series."""

import pytest

from vicent.signals.indicators import (
    adx,
    atr_from_closes,
    bollinger_bands,
    compute_intraday,
    ema,
    ema_trend,
    macd,
    roc,
    rsi,
    stochastic_rsi,
    vwap_from_prices,
)


def _rising(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start + i * step for i in range(n)]


def _falling(n: int, start: float = 100.0, step: float = 1.0) -> list[float]:
    return [start - i * step for i in range(n)]


def _flat(n: int, val: float = 100.0) -> list[float]:
    return [val] * n


# ---- EMA --------------------------------------------------------------------

def test_ema_none_when_too_short() -> None:
    assert ema([1, 2, 3], 10) is None


def test_ema_rising_series() -> None:
    assert ema(_rising(30), 9) > 100


# ---- RSI --------------------------------------------------------------------

def test_rsi_all_gains_near_100() -> None:
    assert rsi(_rising(30), 14) > 90


def test_rsi_all_losses_near_0() -> None:
    assert rsi(_falling(30), 14) < 10


def test_rsi_none_when_short() -> None:
    assert rsi([1, 2, 3], 14) is None


# ---- Stochastic RSI ---------------------------------------------------------

def test_stoch_rsi_returns_value() -> None:
    """Stochastic RSI returns a valid 0-100 value when enough data."""
    import random; random.seed(42)
    # Oscillating prices create meaningful StochRSI variance
    prices = [100 + 5 * (1 if i % 6 < 3 else -1) + random.uniform(-0.5, 0.5)
              for i in range(50)]
    val = stochastic_rsi(prices)
    assert val is not None
    assert 0.0 <= val <= 100.0


def test_stoch_rsi_none_when_short() -> None:
    assert stochastic_rsi([1, 2, 3]) is None


def test_stoch_rsi_at_low_end_after_drop() -> None:
    """After a sharp drop following a high, StochRSI should be low."""
    import random; random.seed(7)
    # High period then crash
    high = [100 + i * 2 + random.uniform(-0.2, 0.2) for i in range(30)]
    crash = [high[-1] - i * 5 + random.uniform(-0.2, 0.2) for i in range(20)]
    prices = high + crash
    val = stochastic_rsi(prices)
    assert val is not None and val < 50  # should be in lower half after drop


def test_stoch_rsi_none_when_short() -> None:
    assert stochastic_rsi([1, 2, 3]) is None


# ---- MACD -------------------------------------------------------------------

def test_macd_bullish_on_uptrend() -> None:
    label, _ = macd(_rising(50))
    assert label == "bullish"


def test_macd_bearish_on_downtrend() -> None:
    label, _ = macd(_falling(50))
    assert label == "bearish"


def test_macd_neutral_when_short() -> None:
    label, _ = macd([1, 2, 3])
    assert label == "neutral"


# ---- ROC --------------------------------------------------------------------

def test_roc_positive_on_rise() -> None:
    assert roc([100, 101, 102, 105], 3) == pytest.approx(5.0, abs=0.1)


def test_roc_negative_on_drop() -> None:
    assert roc([100, 99, 98, 95], 3) == pytest.approx(-5.0, abs=0.1)


def test_roc_zero_when_short() -> None:
    assert roc([100], 3) == 0.0


# ---- EMA trend --------------------------------------------------------------

def test_ema_trend_up() -> None:
    assert ema_trend(_rising(40)) == "up"


def test_ema_trend_down() -> None:
    assert ema_trend(_falling(40)) == "down"


# ---- Bollinger Bands --------------------------------------------------------

def test_bb_returns_none_when_short() -> None:
    u, m, l, pos, sq = bollinger_bands([1, 2, 3])
    assert u is None


def test_bb_upper_above_lower() -> None:
    import random; random.seed(42)
    prices = [100 + random.uniform(-2, 2) for _ in range(25)]
    u, m, l, pos, sq = bollinger_bands(prices)
    assert u > l


def test_bb_position_at_lower_when_price_low() -> None:
    prices = [100.0] * 19 + [90.0]
    _, _, _, pos, _ = bollinger_bands(prices)
    assert pos is not None and pos < 0.3


def test_bb_position_at_upper_when_price_high() -> None:
    prices = [100.0] * 19 + [115.0]
    _, _, _, pos, _ = bollinger_bands(prices)
    assert pos is not None and pos > 0.7


def test_bb_squeeze_on_flat_prices() -> None:
    _, _, _, _, sq = bollinger_bands(_flat(25))
    assert sq is True


# ---- ATR --------------------------------------------------------------------

def test_atr_none_when_short() -> None:
    assert atr_from_closes([1, 2, 3]) is None


def test_atr_higher_on_volatile() -> None:
    import random; random.seed(0)
    calm     = [100 + random.uniform(-0.1, 0.1) for _ in range(30)]
    volatile = [100 + random.uniform(-3.0, 3.0) for _ in range(30)]
    assert atr_from_closes(volatile) > atr_from_closes(calm)


def test_atr_positive() -> None:
    assert atr_from_closes(_rising(30, step=0.5)) > 0


# ---- ADX --------------------------------------------------------------------

def test_adx_none_when_short() -> None:
    assert adx(_rising(10)) is None


def test_adx_higher_on_trend_vs_sideways() -> None:
    import random; random.seed(1)
    sideways = [100 + random.uniform(-0.3, 0.3) for _ in range(60)]
    trend_val = adx(_rising(60, step=2.0))
    side_val  = adx(sideways)
    assert trend_val is not None and side_val is not None
    assert trend_val > side_val


# ---- VWAP -------------------------------------------------------------------

def test_vwap_no_volume_equals_mean() -> None:
    assert vwap_from_prices([100.0, 102.0, 104.0]) == pytest.approx(102.0)


def test_vwap_with_volume_weighted() -> None:
    result = vwap_from_prices([100.0, 200.0], [9.0, 1.0])
    assert result is not None and result < 120


def test_vwap_empty() -> None:
    assert vwap_from_prices([]) is None


# ---- compute_intraday -------------------------------------------------------

def test_compute_intraday_not_enough_data() -> None:
    r = compute_intraday(_rising(10))
    assert r.has_enough_data is False
    assert r.stoch_rsi is None
    assert r.bb_position is None


def test_compute_intraday_full_uptrend() -> None:
    r = compute_intraday(_rising(60, step=0.5))
    assert r.has_enough_data is True
    assert r.rsi is not None and r.rsi > 60
    assert r.macd_signal == "bullish"
    assert r.ema_trend == "up"
    assert r.roc_short >= 0
    assert r.bb_position is not None
    assert r.atr is not None and r.atr > 0
    assert r.vwap_position in ("above", "below", "near")


def test_compute_intraday_downtrend() -> None:
    r = compute_intraday(_falling(60))
    assert r.macd_signal == "bearish"
    assert r.ema_trend == "down"
    assert r.roc_short <= 0


def test_compute_intraday_empty() -> None:
    r = compute_intraday([])
    assert r.n_bars == 0
    assert r.has_enough_data is False
    assert r.rsi is None
    assert r.stoch_rsi is None
