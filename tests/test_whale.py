"""Tests for whale / holder distribution signal."""

import pytest

from vicent.signals.whale import WhalePressure, score_whale


def _metrics(
    whale_pct: float = 0.0,
    trader_pct: float = 0.0,
    hodler_pct: float = 0.0,
    addr_growth: float = 0.0,
) -> dict:
    return {
        "data": {
            "holder_breakdown": {
                "whale_percent": whale_pct,
                "trader_percent": trader_pct,
                "hodler_percent": hodler_pct,
            },
            "addresses": {
                "growth_24h": addr_growth,
            },
        }
    }


# ---- Neutral / missing data -------------------------------------------

def test_empty_metrics_returns_neutral() -> None:
    result = score_whale("CAKE", {})
    assert result.score == 0.0
    assert result.data_available is False


def test_none_values_return_neutral() -> None:
    # Empty data dict — no holder fields available → data_available=False, score=0
    result = score_whale("CAKE", {"data": {}})
    assert result.score == 0.0
    assert result.confidence == 0.0


# ---- Accumulation signals (bullish) ------------------------------------

def test_strong_hodler_base_is_bullish() -> None:
    m = _metrics(hodler_pct=60, whale_pct=25, trader_pct=10)
    result = score_whale("CAKE", m)
    assert result.score > 0.2


def test_healthy_whale_distribution_is_positive() -> None:
    m = _metrics(hodler_pct=40, whale_pct=30, trader_pct=20)
    result = score_whale("CAKE", m)
    assert result.score > 0.0


def test_low_trader_pct_is_bullish() -> None:
    m = _metrics(hodler_pct=50, whale_pct=30, trader_pct=8)
    result = score_whale("CAKE", m)
    assert result.score > 0.15


def test_healthy_address_growth_adds_score() -> None:
    m = _metrics(hodler_pct=40, whale_pct=25, trader_pct=15, addr_growth=5.0)
    result_growth = score_whale("CAKE", m)
    m2 = _metrics(hodler_pct=40, whale_pct=25, trader_pct=15, addr_growth=0.0)
    result_no_growth = score_whale("CAKE", m2)
    assert result_growth.score >= result_no_growth.score


# ---- Distribution signals (bearish) -----------------------------------

def test_whale_dump_with_retail_fomo_is_bearish() -> None:
    # Whales at 70%, traders at 40% = classic distribution into FOMO
    m = _metrics(whale_pct=70, trader_pct=40, hodler_pct=10)
    result = score_whale("CAKE", m)
    assert result.score < 0.0


def test_high_trader_pct_is_bearish() -> None:
    m = _metrics(whale_pct=20, trader_pct=55, hodler_pct=15)
    result = score_whale("CAKE", m)
    assert result.score < 0.0


def test_fomo_address_growth_is_cautious() -> None:
    m = _metrics(hodler_pct=20, whale_pct=20, trader_pct=30, addr_growth=15.0)
    result = score_whale("CAKE", m)
    # Very high growth = possible FOMO peak — shouldn't be strongly bullish
    assert result.score < 0.3


# ---- Score bounds -------------------------------------------------------

def test_score_bounded() -> None:
    m = _metrics(whale_pct=80, trader_pct=70, hodler_pct=5, addr_growth=20)
    result = score_whale("CAKE", m)
    assert -1.0 <= result.score <= 1.0


def test_confidence_bounded() -> None:
    m = _metrics(whale_pct=30, trader_pct=20, hodler_pct=40, addr_growth=3)
    result = score_whale("CAKE", m)
    assert 0.0 <= result.confidence <= 1.0


# ---- Integration with score_token --------------------------------------

def test_news_whale_signals_integrate_into_token_signal() -> None:
    """Verify score_token produces a valid signal when news + whale provided."""
    from datetime import datetime, timedelta, timezone

    from vicent.signals.signals import Direction, score_token

    quotes = {
        "data": {
            "1": {
                "quote": {
                    "USD": {
                        "price": 3.5,
                        "percent_change_1h": 1.5,
                        "percent_change_24h": 4.0,
                        "percent_change_7d": 8.0,
                        "volume_24h": 5_000_000,
                        "volume_30d": 100_000_000,
                    }
                }
            }
        }
    }
    ta = {
        "data": {
            "1h": {
                "indicators": {
                    "rsi": 45,
                    "macd": {"macd_line": 5.0, "signal_line": 2.0},
                    "ema_20": 3.6,
                    "ema_50": 3.4,
                }
            }
        }
    }
    ts = (datetime.now(timezone.utc) - timedelta(minutes=20)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    articles = [{"title": "CAKE listing announced", "published_at": ts}]
    whale_data = _metrics(hodler_pct=45, whale_pct=28, trader_pct=18)

    sig = score_token("CAKE", 1, quotes, ta, news_articles=articles, whale_metrics=whale_data)

    assert sig.direction in (Direction.LONG, Direction.FLAT)
    assert 0.0 <= sig.confidence <= 1.0
    assert sig.news is not None
    assert sig.whale is not None
    assert "news" in sig.sub_scores
    assert "whale" in sig.sub_scores


def test_strong_negative_news_forces_flat() -> None:
    """News emergency override: hack news should force FLAT even on bullish TA."""
    from datetime import datetime, timedelta, timezone

    from vicent.signals.signals import Direction, score_token

    quotes = {
        "data": {
            "1": {
                "quote": {
                    "USD": {
                        "price": 10.0,
                        "percent_change_1h": 2.0,
                        "percent_change_24h": 5.0,
                        "percent_change_7d": 12.0,
                        "volume_24h": 10_000_000,
                    }
                }
            }
        }
    }
    ta = {}  # no TA — pure news test
    ts = (datetime.now(timezone.utc) - timedelta(minutes=5)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    articles = [
        {"title": "MASSIVE EXPLOIT: $50M drained from protocol hack", "published_at": ts},
        {"title": "Protocol emergency shutdown after attack", "published_at": ts},
    ]

    sig = score_token("TEST", 1, quotes, ta, news_articles=articles)
    # Should be FLAT due to news emergency override
    assert sig.direction == Direction.FLAT
    assert sig.confidence == 0.0
