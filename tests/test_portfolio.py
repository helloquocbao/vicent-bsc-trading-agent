"""Tests for portfolio and ledger."""

from vicent.state.portfolio import Portfolio


def test_initial_state() -> None:
    p = Portfolio(1000.0)
    assert p.nav_usd() == 1000.0
    assert p.current_drawdown() == 0.0
    assert p.initial_capital == 1000.0


def test_open_and_close_position() -> None:
    p = Portfolio(1000.0)
    p.open_position("CAKE", quantity=10.0, price_usd=5.0)  # costs $50
    assert p._cash_usd == 950.0
    assert p.has_position("CAKE")

    p.update_prices({"CAKE": 6.0})
    assert p.nav_usd() == pytest.approx(1010.0, rel=0.01)

    proceeds = p.close_position("CAKE", price_usd=6.0)
    assert proceeds == pytest.approx(60.0, rel=0.01)
    assert not p.has_position("CAKE")


def test_drawdown_calculation() -> None:
    p = Portfolio(1000.0)
    p.open_position("CAKE", 100.0, 10.0)  # costs $1000 (full NAV)
    p._peak_nav = 1000.0
    p.update_prices({"CAKE": 8.0})   # -20% on position
    dd = p.current_drawdown()
    assert dd == pytest.approx(0.20, rel=0.05)


def test_peak_tracking() -> None:
    p = Portfolio(1000.0)
    p.open_position("CAKE", 10.0, 10.0)
    p.update_prices({"CAKE": 15.0})  # +50%, peak = 1050
    peak_1 = p._peak_nav
    p.update_prices({"CAKE": 12.0})  # down from peak
    assert p._peak_nav == peak_1  # peak should not decrease
    assert p.current_drawdown() > 0


def test_position_peak_price_tracking() -> None:
    """Position should track its own high-water mark for trailing stops."""
    p = Portfolio(1000.0)
    p.open_position("CAKE", 100.0, 5.0, atr_pct=0.05)
    pos = p.get_position("CAKE")
    assert pos is not None
    assert pos.peak_price_usd == pytest.approx(5.0)

    p.update_prices({"CAKE": 6.5})
    assert pos.peak_price_usd == pytest.approx(6.5)

    # Pullback — peak should NOT decrease
    p.update_prices({"CAKE": 5.5})
    assert pos.peak_price_usd == pytest.approx(6.5)


def test_position_peak_pnl_pct() -> None:
    p = Portfolio(1000.0)
    p.open_position("CAKE", 100.0, 4.0, atr_pct=0.05)
    pos = p.get_position("CAKE")
    assert pos is not None
    p.update_prices({"CAKE": 5.0})  # +25%
    assert pos.peak_pnl_pct == pytest.approx(0.25, abs=0.01)
    p.update_prices({"CAKE": 4.5})  # -10% from peak
    assert pos.peak_pnl_pct == pytest.approx(0.25, abs=0.01)


def test_position_atr_persisted() -> None:
    p = Portfolio(1000.0)
    p.open_position("INJ", 10.0, 25.0, atr_pct=0.09)
    pos = p.get_position("INJ")
    assert pos is not None
    assert pos.entry_atr_pct == pytest.approx(0.09)


def test_portfolio_serialization_roundtrip() -> None:
    p = Portfolio(2000.0)
    p._cash_usd = 1500.0
    p._peak_nav = 2100.0
    p.open_position("BTCB", quantity=2.0, price_usd=250.0, atr_pct=0.04)
    pos = p.get_position("BTCB")
    assert pos is not None
    pos.peak_price_usd = 280.0
    pos.scaled_out_t1 = True

    # Serialize to dict/json and reconstruct
    serialized = p.to_dict()
    reconstructed = Portfolio.from_snapshot_dict(serialized)

    assert reconstructed.initial_capital == 2000.0
    assert reconstructed._cash_usd == 1000.0
    assert reconstructed._peak_nav == 2100.0
    
    pos_recon = reconstructed.get_position("BTCB")
    assert pos_recon is not None
    assert pos_recon.quantity == 2.0
    assert pos_recon.avg_cost_usd == 250.0
    assert pos_recon.peak_price_usd == 280.0
    assert pos_recon.entry_atr_pct == 0.04
    assert pos_recon.scaled_out_t1 is True
    assert pos_recon.scaled_out_t2 is False


import pytest

