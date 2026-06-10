"""PnL optimization layer — volatility-aware stops, fee filter, correlation guard.

This module is the difference between a working agent and a profitable one.

Five tools:
  1. compute_volatility       — ATR-style proxy from CMC % changes
  2. volatility_aware_targets — per-position SL/TP/trail based on the token's vol
  3. should_skip_for_fees     — kill trades whose edge can't beat round-trip costs
  4. SectorMap / sector_check — cap exposure to any one correlated sector
  5. ScaledExit               — three-tier profit-taking (lock-in + ride trail)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from vicent.state.portfolio import Portfolio

log = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# 1. Volatility proxy
# ---------------------------------------------------------------------------

def compute_volatility(
    pct_1h: float,
    pct_24h: float,
    pct_7d: float,
) -> float:
    """Estimate daily ATR-equivalent (as a fraction, e.g. 0.05 = 5%).

    We use the maximum of three candidates:
      • |pct_24h| as direct daily move
      • |pct_1h| * sqrt(24) as scaled hourly volatility
      • |pct_7d| / sqrt(7) as scaled weekly volatility
    Capped to [1%, 25%] — anything outside is a data error or untradeable.
    """
    import math

    candidates = [
        abs(pct_24h) / 100.0,
        abs(pct_1h) / 100.0 * math.sqrt(24),
        abs(pct_7d) / 100.0 / math.sqrt(7),
    ]
    atr = max(candidates) if candidates else 0.04
    return max(0.01, min(0.25, atr))


# ---------------------------------------------------------------------------
# 2. Volatility-aware exit targets
# ---------------------------------------------------------------------------

@dataclass
class ExitTargets:
    stop_loss_pct: float       # negative (e.g. -0.08)
    take_profit_t1_pct: float  # first scale-out (sell 1/3)
    take_profit_t2_pct: float  # second scale-out (sell 1/3)
    trail_pct: float           # trailing stop fraction below peak for last 1/3


def volatility_aware_targets(atr_pct: float) -> ExitTargets:
    """Per-position SL/TP scaled by the token's recent volatility.

    Calm market (ATR ~2%):  SL=-5%, TP1=+6%, TP2=+12%, trail 4%
    Normal (ATR ~5%):       SL=-8%, TP1=+10%, TP2=+18%, trail 6%
    Volatile (ATR ~10%):    SL=-13%, TP1=+15%, TP2=+25%, trail 9%

    Reasoning: stop must be wide enough to avoid noise stop-outs, TP must
    be far enough that R:R justifies the trade after fees.
    """
    sl = -max(0.05, 1.6 * atr_pct)
    tp1 = max(0.05, 1.2 * atr_pct)
    tp2 = max(0.10, 2.2 * atr_pct)
    trail = max(0.03, 0.85 * atr_pct)

    # Cap absolute extremes
    sl = max(sl, -0.18)        # never stop wider than -18%
    tp2 = min(tp2, 0.40)       # take profit hard at +40%
    trail = min(trail, 0.10)

    return ExitTargets(
        stop_loss_pct=sl,
        take_profit_t1_pct=tp1,
        take_profit_t2_pct=tp2,
        trail_pct=trail,
    )


# ---------------------------------------------------------------------------
# 3. Fee/slippage edge filter
# ---------------------------------------------------------------------------

# BSC / PancakeSwap V3 fee: 0.25% per side = 0.50% round-trip
# Plus estimated 0.05% slippage for small orders → ~0.55% total cost floor
# Note: PancakeSwap V3 has tiers (0.01%, 0.05%, 0.25%, 1%) — 0.25% is the
# most common tier for mid-cap BEP-20 tokens. BNB/stablecoin pairs use 0.05%.
_ROUND_TRIP_COST_PCT = 0.0055  # 0.55% — calibrated for BSC PancakeSwap V3


def should_skip_for_fees(
    expected_move_pct: float,
    confidence: float,
    custom_cost_pct: float | None = None,
) -> tuple[bool, str]:
    """Return (skip, reason). Trade is skipped if expected edge < cost floor.

    expected_move_pct is the ATR proxy — the "typical" move we expect to
    capture. We require the captured move (= ATR * confidence_factor)
    to exceed round-trip cost by at least 1.5x.

    Calibrated for BSC PancakeSwap V3 (0.25% fee each side + ~0.05% slippage
    = 0.55% round-trip). Pass custom_cost_pct to override for other venues.
    """
    cost = custom_cost_pct if custom_cost_pct is not None else _ROUND_TRIP_COST_PCT
    expected_capture = expected_move_pct * (0.4 + 0.6 * confidence)
    required = cost * 1.5

    if expected_capture < required:
        return True, (
            f"edge_below_floor: expected_capture={expected_capture:.4f} "
            f"< required={required:.4f}"
        )
    return False, "ok"


# ---------------------------------------------------------------------------
# 4. Sector / correlation guard
# ---------------------------------------------------------------------------

class Sector(str, Enum):
    DEFI = "defi"
    L1 = "l1"
    L2_INFRA = "l2_infra"
    MEME = "meme"
    AI = "ai"
    GAMING = "gaming"
    OTHER = "other"


SECTOR_MAP: dict[str, Sector] = {
    # Majors
    "BTC": Sector.L1, "ETH": Sector.L1, "BNB": Sector.L1,
    "SOL": Sector.L1, "XRP": Sector.L1, "ADA": Sector.L1,
    "DOGE": Sector.MEME, "SHIB": Sector.MEME,
    "LTC": Sector.L1, "TRX": Sector.L1, "TON": Sector.L1,

    # DeFi protocols
    "CAKE": Sector.DEFI, "AAVE": Sector.DEFI, "UNI": Sector.DEFI,
    "COMP": Sector.DEFI, "SNX": Sector.DEFI, "SUSHI": Sector.DEFI,
    "1INCH": Sector.DEFI, "PENDLE": Sector.DEFI, "LDO": Sector.DEFI,
    "RAY": Sector.DEFI, "CRV": Sector.DEFI, "BAL": Sector.DEFI,
    "RUNE": Sector.DEFI, "GMT": Sector.DEFI,

    # Layer 1 chains
    "AVAX": Sector.L1, "DOT": Sector.L1, "ATOM": Sector.L1,
    "INJ": Sector.L1, "ZIL": Sector.L1, "FIL": Sector.L1,
    "NEAR": Sector.L1, "FTM": Sector.L1, "EGLD": Sector.L1,
    "KAVA": Sector.L1,

    # Layer 2 / Infra
    "LINK": Sector.L2_INFRA, "ZRO": Sector.L2_INFRA, "STG": Sector.L2_INFRA,
    "ARB": Sector.L2_INFRA, "OP": Sector.L2_INFRA, "MATIC": Sector.L2_INFRA,

    # Memes
    "FLOKI": Sector.MEME, "PENGU": Sector.MEME, "BONK": Sector.MEME,

    # Gaming
    "AXS": Sector.GAMING, "APE": Sector.GAMING,
    "SAND": Sector.GAMING, "MANA": Sector.GAMING,

    # AI
    "FET": Sector.AI, "PEAQ": Sector.AI,
}

# Hard cap: max NAV % allocated to any single sector
_MAX_SECTOR_EXPOSURE = 0.40


def get_sector(symbol: str) -> Sector:
    return SECTOR_MAP.get(symbol.upper(), Sector.OTHER)


def sector_exposure(portfolio: "Portfolio") -> dict[Sector, float]:
    """Return USD value held in each sector."""
    exposure: dict[Sector, float] = {}
    for sym, pos in portfolio._positions.items():
        sector = get_sector(sym)
        exposure[sector] = exposure.get(sector, 0.0) + pos.value_usd
    return exposure


def correlation_check(
    portfolio: "Portfolio",
    new_symbol: str,
    new_trade_usd: float,
) -> tuple[bool, str]:
    """Return (allowed, reason). Block if adding this trade would push
    sector exposure above the cap.
    """
    nav = portfolio.nav_usd()
    if nav <= 0:
        return True, "ok"

    sector = get_sector(new_symbol)
    if sector == Sector.OTHER:
        # Uncategorized tokens have no per-sector cap (they're naturally diverse)
        return True, "ok"

    exposure = sector_exposure(portfolio)
    current = exposure.get(sector, 0.0)
    projected = (current + new_trade_usd) / nav

    if projected > _MAX_SECTOR_EXPOSURE:
        return False, (
            f"sector_cap: {sector.value} would be {projected:.0%} > {_MAX_SECTOR_EXPOSURE:.0%}"
        )
    return True, "ok"


# ---------------------------------------------------------------------------
# 5. Scaled exit decisions
# ---------------------------------------------------------------------------

class ExitAction(str, Enum):
    HOLD = "hold"
    SCALE_OUT_T1 = "scale_out_t1"   # sell 1/3 at TP1
    SCALE_OUT_T2 = "scale_out_t2"   # sell 1/3 at TP2
    TRAIL_EXIT = "trail_exit"        # final 1/3 hit trailing stop
    STOP_LOSS = "stop_loss"
    SIGNAL_REVERSAL = "signal_reversal"


@dataclass
class ExitPlan:
    action: ExitAction
    fraction: float           # how much of remaining position to sell (0-1)
    reason: str
    realized_pnl_pct: float   # PnL on the chunk being sold


def evaluate_exit(
    pnl_pct: float,
    peak_pnl_pct: float,
    targets: ExitTargets,
    scaled_out_t1: bool,
    scaled_out_t2: bool,
    signal_reversed: bool = False,
) -> ExitPlan:
    """Decide whether to exit fully, partially, or hold.

    Order of checks (highest priority first):
      1. Stop-loss — exit 100% if breached
      2. Signal reversal — exit 100% if reversed strongly
      3. Trailing stop on remaining 1/3 (only if t1 + t2 already done)
      4. TP2 — sell 50% of remainder (=1/3 of original)
      5. TP1 — sell 33% of original
      6. Hold
    """
    # Hard stops first
    if pnl_pct <= targets.stop_loss_pct:
        return ExitPlan(ExitAction.STOP_LOSS, 1.0,
                        f"stop_loss_hit:{pnl_pct:.2%}", pnl_pct)

    if signal_reversed:
        return ExitPlan(ExitAction.SIGNAL_REVERSAL, 1.0,
                        "signal_reversed", pnl_pct)

    # Trailing stop on remaining slice
    if scaled_out_t1 and scaled_out_t2:
        # Final tier — exit if drawdown from peak exceeds trail_pct
        drawdown_from_peak = peak_pnl_pct - pnl_pct
        if drawdown_from_peak >= targets.trail_pct and pnl_pct > 0:
            return ExitPlan(ExitAction.TRAIL_EXIT, 1.0,
                            f"trail_hit:peak={peak_pnl_pct:.2%}_now={pnl_pct:.2%}",
                            pnl_pct)

    # Scale-out T2 (1/3 of original = 50% of remaining after T1)
    if not scaled_out_t2 and pnl_pct >= targets.take_profit_t2_pct:
        return ExitPlan(ExitAction.SCALE_OUT_T2, 0.50,
                        f"tp2_hit:{pnl_pct:.2%}", pnl_pct)

    # Scale-out T1 (1/3 of original)
    if not scaled_out_t1 and pnl_pct >= targets.take_profit_t1_pct:
        return ExitPlan(ExitAction.SCALE_OUT_T1, 0.34,
                        f"tp1_hit:{pnl_pct:.2%}", pnl_pct)

    return ExitPlan(ExitAction.HOLD, 0.0, "hold", pnl_pct)
