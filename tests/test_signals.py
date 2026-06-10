"""Tests for per-token signal scoring."""

from vicent.signals.signals import Direction, score_token


def _make_quotes(cmc_id: int, price: float, pct_1h: float, pct_24h: float) -> dict:
    return {
        "data": {
            str(cmc_id): {
                "quote": {
                    "USD": {
                        "price": price,
                        "percent_change_1h": pct_1h,
                        "percent_change_24h": pct_24h,
                        "percent_change_7d": pct_24h * 2,
                        "volume_24h": 5_000_000,
                        "volume_30d": 100_000_000,
                    }
                }
            }
        }
    }


def _make_ta(rsi: float, macd_bull: bool, ema20: float, ema50: float) -> dict:
    macd_line = 10.0 if macd_bull else -10.0
    return {
        "data": {
            "1h": {
                "indicators": {
                    "rsi": rsi,
                    "macd": {"macd_line": macd_line, "signal_line": 0.0},
                    "ema_20": ema20,
                    "ema_50": ema50,
                }
            }
        }
    }


def test_strong_long_signal() -> None:
    quotes = _make_quotes(1, price=100.0, pct_1h=2.5, pct_24h=5.0)
    ta = _make_ta(rsi=40, macd_bull=True, ema20=102, ema50=98)
    sig = score_token("TEST", 1, quotes, ta)
    assert sig.direction == Direction.LONG
    assert sig.confidence >= 0.50


def test_overbought_reduces_long_confidence() -> None:
    quotes = _make_quotes(1, price=100.0, pct_1h=1.0, pct_24h=2.0)
    ta = _make_ta(rsi=80, macd_bull=True, ema20=101, ema50=98)
    sig = score_token("TEST", 1, quotes, ta)
    # RSI 80 = overbought; confidence should be lower or FLAT
    if sig.direction == Direction.LONG:
        assert sig.confidence < 0.75


def test_flat_signal_on_no_data() -> None:
    sig = score_token("EMPTY", 999, {}, {})
    assert sig.direction == Direction.FLAT
    assert sig.confidence == 0.0


def test_volume_surge_detection() -> None:
    quotes = {
        "data": {
            "1": {
                "quote": {
                    "USD": {
                        "price": 50.0,
                        "percent_change_1h": 1.5,
                        "percent_change_24h": 4.0,
                        "percent_change_7d": 8.0,
                        "volume_24h": 30_000_000,   # 3x normal
                        "volume_30d": 300_000_000,  # avg daily = 10M
                    }
                }
            }
        }
    }
    ta = _make_ta(rsi=50, macd_bull=True, ema20=51, ema50=49)
    sig = score_token("TEST", 1, quotes, ta)
    assert sig.volume_change_pct > 50  # should detect the surge
