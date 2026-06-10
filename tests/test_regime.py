"""Tests for macro regime detector."""

from vicent.signals.regime import Regime, detect_regime


def _make_global(fg: int, btc_dom: float) -> dict:
    return {
        "data": {
            "fear_and_greed_index": {"value": fg},
            "btc_dominance": btc_dom,
        }
    }


def _make_derivatives(funding: float, oi_change: float) -> dict:
    return {
        "data": {
            "funding_rate": funding,
            "open_interest_24h_pct_change": oi_change,
        }
    }


def test_bull_regime() -> None:
    g = _make_global(fg=75, btc_dom=42)
    d = _make_derivatives(funding=-0.01, oi_change=10)
    result = detect_regime(g, d)
    assert result.regime == Regime.BULL
    assert result.fear_greed == 75


def test_bear_regime() -> None:
    g = _make_global(fg=15, btc_dom=60)
    d = _make_derivatives(funding=0.05, oi_change=-15)
    result = detect_regime(g, d)
    assert result.regime == Regime.BEAR


def test_neutral_regime() -> None:
    g = _make_global(fg=50, btc_dom=52)
    d = _make_derivatives(funding=0.01, oi_change=2)
    result = detect_regime(g, d)
    assert result.regime in (Regime.NEUTRAL, Regime.BULL)


def test_confidence_range() -> None:
    g = _make_global(fg=60, btc_dom=48)
    d = _make_derivatives(funding=0.005, oi_change=3)
    result = detect_regime(g, d)
    assert 0.0 <= result.confidence <= 1.0
