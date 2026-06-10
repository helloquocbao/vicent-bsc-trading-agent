"""Tests for risk manager."""

import pytest

from vicent.state.portfolio import Portfolio
from vicent.strategy.risk import RiskManager


def _portfolio(nav: float, peak: float) -> Portfolio:
    p = Portfolio(initial_capital_usd=peak)
    p._peak_nav = peak
    p._cash_usd = nav
    return p


def test_normal_conditions_allow_trade() -> None:
    rm = RiskManager()
    p = _portfolio(nav=100, peak=100)
    size = rm.compute_size(confidence=0.75, portfolio=p, token_volume_24h=1_000_000)
    assert size.allowed is True
    assert size.nav_fraction > 0


def test_low_confidence_blocks_trade() -> None:
    rm = RiskManager()
    p = _portfolio(nav=100, peak=100)
    size = rm.compute_size(confidence=0.30, portfolio=p, token_volume_24h=1_000_000)
    assert size.allowed is False


def test_drawdown_cap_triggers_hard_stop() -> None:
    rm = RiskManager()
    # NAV = 79, peak = 100 → drawdown = 21% > 20% cap
    p = _portfolio(nav=79, peak=100)
    ok, reason = rm.check_hard_stops(p)
    assert ok is False
    assert "drawdown" in reason.lower()


def test_near_cap_reduces_size() -> None:
    rm = RiskManager()
    # 15% drawdown → should reduce size
    p = _portfolio(nav=85, peak=100)
    size_normal = rm.compute_size(confidence=0.80, portfolio=_portfolio(nav=100, peak=100), token_volume_24h=1_000_000)
    size_near_cap = rm.compute_size(confidence=0.80, portfolio=p, token_volume_24h=1_000_000)
    assert size_near_cap.nav_fraction <= size_normal.nav_fraction


def test_low_liquidity_blocks_trade() -> None:
    rm = RiskManager()
    p = _portfolio(nav=100, peak=100)
    size = rm.compute_size(confidence=0.80, portfolio=p, token_volume_24h=10_000)  # below 250k
    assert size.allowed is False
    assert "liquidity" in size.reason


def test_confidence_tiers() -> None:
    rm = RiskManager()
    p = _portfolio(nav=100, peak=100)
    high = rm.compute_size(0.85, p, 1_000_000).nav_fraction
    mid = rm.compute_size(0.70, p, 1_000_000).nav_fraction
    low = rm.compute_size(0.55, p, 1_000_000).nav_fraction
    assert high >= mid >= low
