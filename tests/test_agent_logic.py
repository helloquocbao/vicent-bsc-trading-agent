"""Tests for agent-level logic: reflexion, day reset, min-trade, position limits."""

import pytest

from vicent.agent import VICENTAgent
from vicent.state.portfolio import Portfolio


# ---- Reflexion ----------------------------------------------------------

def test_reflexion_tightens_on_losing_streak() -> None:
    agent = VICENTAgent(initial_capital_usd=100.0)
    agent._closed_trade_pnls = [-0.05, -0.06, -0.04, -0.07, -0.03]  # all losses
    initial = agent._min_confidence
    agent._reflexion_step()
    assert agent._min_confidence > initial, "Threshold should tighten after losses"


def test_reflexion_loosens_on_winning_streak() -> None:
    agent = VICENTAgent(initial_capital_usd=100.0)
    agent._closed_trade_pnls = [0.05, 0.08, 0.06, 0.07, 0.04]  # all wins
    agent._min_confidence = 0.65  # start elevated
    agent._reflexion_step()
    assert agent._min_confidence < 0.65, "Threshold should loosen after wins"


def test_reflexion_no_change_below_3_trades() -> None:
    agent = VICENTAgent(initial_capital_usd=100.0)
    agent._closed_trade_pnls = [-0.10, -0.10]  # only 2 — not enough
    initial = agent._min_confidence
    agent._reflexion_step()
    assert agent._min_confidence == initial


def test_reflexion_threshold_bounded() -> None:
    agent = VICENTAgent(initial_capital_usd=100.0)
    # Many losses — threshold should cap at 0.75
    agent._min_confidence = 0.73
    agent._closed_trade_pnls = [-0.10] * 10
    for _ in range(10):
        agent._reflexion_step()
    assert agent._min_confidence <= 0.75


# ---- Volatility-aware exit targets (replaces fixed constants) ----------

def test_volatility_targets_calm_market() -> None:
    from vicent.strategy.optimization import volatility_aware_targets
    targets = volatility_aware_targets(atr_pct=0.02)  # 2% ATR = calm
    assert targets.stop_loss_pct == pytest.approx(-0.05, abs=0.001)
    assert targets.take_profit_t1_pct >= 0.05


def test_volatility_targets_volatile_token() -> None:
    from vicent.strategy.optimization import volatility_aware_targets
    targets = volatility_aware_targets(atr_pct=0.10)  # 10% ATR = volatile
    # Wider stop, wider TPs
    assert targets.stop_loss_pct < -0.10
    assert targets.take_profit_t2_pct > 0.15


def test_volatility_targets_capped() -> None:
    from vicent.strategy.optimization import volatility_aware_targets
    targets = volatility_aware_targets(atr_pct=0.30)  # extreme — should cap
    assert targets.stop_loss_pct >= -0.18
    assert targets.take_profit_t2_pct <= 0.40
    assert targets.trail_pct <= 0.10


# ---- Day boundary reset -------------------------------------------------

def test_day_reset_resets_daily_drawdown() -> None:
    p = Portfolio(1000.0)
    # Simulate a loss that raises daily drawdown
    p._cash_usd = 950.0
    p._day_start_nav = 1000.0
    assert p.daily_drawdown() == pytest.approx(0.05)

    # Reset at day boundary
    p.reset_day_start()
    assert p.daily_drawdown() == 0.0


# ---- Risk — min trade enforcement ---------------------------------------

def test_force_min_trade_triggers_with_4h_remaining() -> None:
    from vicent.strategy.risk import RiskManager
    rm = RiskManager()
    # 0 trades today with 4h left → should force
    assert rm.should_force_min_trade(trades_today=0, hours_remaining=4) is True


def test_force_min_trade_does_not_trigger_early() -> None:
    from vicent.strategy.risk import RiskManager
    rm = RiskManager()
    # 0 trades but 20h remaining — don't force yet
    assert rm.should_force_min_trade(trades_today=0, hours_remaining=20) is False


def test_force_min_trade_satisfied() -> None:
    from vicent.strategy.risk import RiskManager
    rm = RiskManager()
    # 2 trades already done — no need to force
    assert rm.should_force_min_trade(trades_today=2, hours_remaining=2) is False


# ---- Ensemble dynamic threshold -----------------------------------------

def test_ensemble_respects_min_confidence_override() -> None:
    from vicent.signals.regime import Regime, RegimeScore
    from vicent.signals.signals import Direction, TokenSignal
    from vicent.strategy.ensemble import decide

    regime = RegimeScore(
        regime=Regime.BULL,
        fear_greed=60,
        btc_dominance=48,
        funding_rate=0.01,
        open_interest_change=5,
        confidence=0.7,
    )
    sig = TokenSignal(
        symbol="CAKE",
        cmc_id=1,
        direction=Direction.LONG,
        confidence=0.55,  # above default 0.50
        price_usd=3.0,
        volume_24h=1_000_000,
        volume_change_pct=20.0,
        rsi_1h=45,
        macd_signal="bullish",
        ema_trend="up",
    )

    # With raised threshold (reflexion tightened to 0.70), this should not trade
    result_blocked = decide(sig, regime, min_confidence=0.70)
    assert result_blocked.should_trade is False

    # With default threshold (0.50), this should trade
    result_allowed = decide(sig, regime, min_confidence=0.50)
    assert result_allowed.should_trade is True


@pytest.mark.asyncio
async def test_agent_restores_portfolio_on_startup(tmp_path, monkeypatch) -> None:
    # 1. Override the database path and mode to a temp file/paper
    db_file = tmp_path / "test_restore_agent.db"
    monkeypatch.setenv("VICENT_DB_PATH", str(db_file))
    monkeypatch.setenv("VICENT_MODE", "paper")
    
    # Reload settings to pick up the env var
    from vicent.config import get_settings, Mode
    settings = get_settings()
    settings.vicent_db_path = str(db_file)
    settings.vicent_mode = Mode.PAPER
    
    from vicent.state.ledger import init_db, record_snapshot
    init_db()
    
    # 2. Create a dummy portfolio snapshot
    p = Portfolio(1500.0)
    p._cash_usd = 1200.0
    p.open_position("ETH", quantity=1.0, price_usd=300.0)
    
    record_snapshot(
        nav_usd=p.nav_usd(),
        peak_nav=p._peak_nav,
        drawdown=p.current_drawdown(),
        positions=p.to_json(),
    )
    
    # 3. Initialize agent
    agent = VICENTAgent(initial_capital_usd=1000.0)
    agent._running = False
    
    class DummyStopException(Exception):
        pass
        
    import unittest.mock as mock
    with mock.patch("vicent.agent.CMCClient") as mock_cmc:
        mock_cmc.return_value.__aenter__.side_effect = DummyStopException("stop")
        try:
            await agent.run()
        except DummyStopException:
            pass
            
    # Verify that the agent portfolio was successfully restored from the database snapshot
    assert agent.portfolio.initial_capital == 1500.0
    assert agent.portfolio._cash_usd == 900.0
    assert agent.portfolio.has_position("ETH")
    assert agent.portfolio.get_position("ETH").quantity == 1.0

