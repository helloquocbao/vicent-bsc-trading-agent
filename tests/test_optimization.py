"""Tests for the PnL optimization layer."""

import pytest

from vicent.state.portfolio import Portfolio
from vicent.strategy.optimization import (
    ExitAction,
    Sector,
    compute_volatility,
    correlation_check,
    evaluate_exit,
    get_sector,
    sector_exposure,
    should_skip_for_fees,
    volatility_aware_targets,
)


# ---- compute_volatility -------------------------------------------------

def test_volatility_calm_token() -> None:
    # Small moves → low ATR
    atr = compute_volatility(pct_1h=0.2, pct_24h=1.5, pct_7d=3.0)
    assert atr < 0.05


def test_volatility_volatile_token() -> None:
    atr = compute_volatility(pct_1h=2.0, pct_24h=8.0, pct_7d=15.0)
    assert atr >= 0.08


def test_volatility_clamped_low() -> None:
    atr = compute_volatility(pct_1h=0.0, pct_24h=0.0, pct_7d=0.0)
    assert atr >= 0.01  # never below 1%


def test_volatility_clamped_high() -> None:
    atr = compute_volatility(pct_1h=10.0, pct_24h=80.0, pct_7d=200.0)
    assert atr <= 0.25  # never above 25%


# ---- volatility_aware_targets ------------------------------------------

def test_targets_scale_with_volatility() -> None:
    calm = volatility_aware_targets(0.02)
    volatile = volatility_aware_targets(0.10)
    # Volatile token should have wider stops AND wider TPs
    assert volatile.stop_loss_pct < calm.stop_loss_pct
    assert volatile.take_profit_t1_pct > calm.take_profit_t1_pct
    assert volatile.take_profit_t2_pct > calm.take_profit_t2_pct
    assert volatile.trail_pct > calm.trail_pct


def test_targets_minimum_floor() -> None:
    # Very low ATR shouldn't produce absurdly tight stops
    targets = volatility_aware_targets(0.005)
    assert targets.stop_loss_pct <= -0.05
    assert targets.take_profit_t1_pct >= 0.05


# ---- should_skip_for_fees ----------------------------------------------

def test_skip_when_edge_below_floor() -> None:
    # Tiny ATR + low confidence = not enough edge.
    # With HL cost defaulting to 0.10% round-trip, use a higher custom cost to test
    # the skip logic (e.g. high fee / high slippage market conditions).
    skip, reason = should_skip_for_fees(
        expected_move_pct=0.005, confidence=0.55, custom_cost_pct=0.008
    )
    assert skip is True
    assert "edge_below" in reason


def test_skip_uses_bsc_cost_by_default() -> None:
    # Default is now BSC PancakeSwap (~0.55% round-trip).
    # 0.5% ATR is NOT enough to cover 0.55% BSC fee — should skip.
    skip, _ = should_skip_for_fees(expected_move_pct=0.005, confidence=0.55)
    assert skip is True   # 0.5% ATR insufficient for BSC fees

    # 3% ATR at high confidence should pass BSC fee filter
    skip2, _ = should_skip_for_fees(expected_move_pct=0.03, confidence=0.80)
    assert skip2 is False  # 3% ATR easily covers 0.55% BSC fees


def test_allow_when_edge_above_floor() -> None:
    # Healthy ATR + high confidence
    skip, _ = should_skip_for_fees(expected_move_pct=0.05, confidence=0.80)
    assert skip is False


def test_higher_confidence_lowers_edge_bar() -> None:
    # Same ATR, different confidence — higher conf should be more permissive
    skip_low, _ = should_skip_for_fees(expected_move_pct=0.015, confidence=0.50)
    skip_high, _ = should_skip_for_fees(expected_move_pct=0.015, confidence=0.95)
    # At least the high-confidence case shouldn't be MORE blocked than low
    if skip_high is True:
        assert skip_low is True


# ---- Sector / correlation guard ----------------------------------------

def test_sector_lookup() -> None:
    assert get_sector("CAKE") == Sector.DEFI
    assert get_sector("AVAX") == Sector.L1
    assert get_sector("PENGU") == Sector.MEME
    assert get_sector("UNKNOWN") == Sector.OTHER


def test_correlation_allows_first_defi_position() -> None:
    p = Portfolio(1000.0)
    ok, reason = correlation_check(p, "CAKE", new_trade_usd=200.0)
    assert ok is True


def test_correlation_blocks_overconcentration() -> None:
    p = Portfolio(1000.0)
    # Open two DeFi positions = 30% (300/1000)
    p.open_position("CAKE", quantity=100, price_usd=1.5)   # $150
    p.open_position("AAVE", quantity=10, price_usd=15.0)   # $150
    # Try to add a third DeFi worth $200 → would push to 50% > 40% cap
    ok, reason = correlation_check(p, "UNI", new_trade_usd=200.0)
    assert ok is False
    assert "sector_cap" in reason


def test_correlation_other_sectors_unblocked() -> None:
    p = Portfolio(1000.0)
    # Heavy DeFi position
    p.open_position("CAKE", quantity=100, price_usd=3.5)
    # New L1 position should be allowed regardless
    ok, _ = correlation_check(p, "AVAX", new_trade_usd=200.0)
    assert ok is True


def test_sector_exposure_calculation() -> None:
    p = Portfolio(1000.0)
    p.open_position("CAKE", quantity=100, price_usd=2.0)   # $200 DeFi
    p.open_position("AVAX", quantity=10, price_usd=30.0)   # $300 L1
    exp = sector_exposure(p)
    assert exp[Sector.DEFI] == pytest.approx(200.0)
    assert exp[Sector.L1] == pytest.approx(300.0)


# ---- evaluate_exit / scaled exits --------------------------------------

def test_exit_stop_loss_takes_priority() -> None:
    targets = volatility_aware_targets(0.05)
    plan = evaluate_exit(
        pnl_pct=-0.20,         # massive loss
        peak_pnl_pct=0.10,     # was up earlier
        targets=targets,
        scaled_out_t1=True,    # already scaled out — irrelevant
        scaled_out_t2=True,
    )
    assert plan.action == ExitAction.STOP_LOSS
    assert plan.fraction == 1.0


def test_exit_signal_reversal_full_close() -> None:
    targets = volatility_aware_targets(0.05)
    plan = evaluate_exit(
        pnl_pct=0.05, peak_pnl_pct=0.07, targets=targets,
        scaled_out_t1=False, scaled_out_t2=False, signal_reversed=True,
    )
    assert plan.action == ExitAction.SIGNAL_REVERSAL
    assert plan.fraction == 1.0


def test_exit_tp1_partial_sale() -> None:
    targets = volatility_aware_targets(0.05)
    # ATR=5% → TP1 ~6%
    plan = evaluate_exit(
        pnl_pct=0.07, peak_pnl_pct=0.07, targets=targets,
        scaled_out_t1=False, scaled_out_t2=False,
    )
    assert plan.action == ExitAction.SCALE_OUT_T1
    assert plan.fraction < 0.5  # partial


def test_exit_tp2_after_tp1() -> None:
    targets = volatility_aware_targets(0.05)
    # ATR=5% → TP2 ~11%
    plan = evaluate_exit(
        pnl_pct=0.13, peak_pnl_pct=0.13, targets=targets,
        scaled_out_t1=True, scaled_out_t2=False,
    )
    assert plan.action == ExitAction.SCALE_OUT_T2


def test_exit_trailing_stop_engages_after_both_tps() -> None:
    targets = volatility_aware_targets(0.05)  # trail ~4%
    # Was up 20%, now down to 14% → drawdown 6% > trail
    plan = evaluate_exit(
        pnl_pct=0.14, peak_pnl_pct=0.20, targets=targets,
        scaled_out_t1=True, scaled_out_t2=True,
    )
    assert plan.action == ExitAction.TRAIL_EXIT


def test_exit_holds_when_no_threshold_hit() -> None:
    targets = volatility_aware_targets(0.05)
    plan = evaluate_exit(
        pnl_pct=0.02, peak_pnl_pct=0.02, targets=targets,
        scaled_out_t1=False, scaled_out_t2=False,
    )
    assert plan.action == ExitAction.HOLD
    assert plan.fraction == 0.0


def test_exit_trail_does_not_fire_before_both_tps() -> None:
    targets = volatility_aware_targets(0.05)
    # Big drawdown from peak but TPs not yet taken → no trail exit
    plan = evaluate_exit(
        pnl_pct=0.05, peak_pnl_pct=0.20, targets=targets,
        scaled_out_t1=False, scaled_out_t2=False,
    )
    assert plan.action != ExitAction.TRAIL_EXIT
