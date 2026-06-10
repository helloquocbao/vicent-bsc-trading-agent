"""Risk manager — position sizing and hard-stop enforcement.

This is the most important module. A single breach of the 30% drawdown
cap disqualifies the agent regardless of PnL. We set our internal cap
at 20% with tiered position sizing.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import structlog

from vicent.config import get_settings

if TYPE_CHECKING:
    from vicent.state.portfolio import Portfolio

log = structlog.get_logger(__name__)


@dataclass
class SizeDecision:
    allowed: bool
    nav_fraction: float       # fraction of NAV to trade (0-1)
    reason: str


class RiskManager:
    """Enforces hard stops and computes position sizes."""

    def __init__(self) -> None:
        cfg = get_settings()
        self.max_drawdown = cfg.risk_max_drawdown_pct        # 0.20
        self.daily_loss_cap = cfg.risk_daily_loss_pct        # 0.05
        self.per_trade_max = cfg.risk_per_trade_nav_pct      # 0.20
        self.min_liquidity = cfg.risk_min_liquidity_usd      # 250k

    # ------------------------------------------------------------------ #
    # Hard stops — these HALT the agent entirely                          #
    # ------------------------------------------------------------------ #

    def check_hard_stops(self, portfolio: "Portfolio") -> tuple[bool, str]:
        """Return (ok, reason). If not ok, agent must stop trading."""
        dd = portfolio.current_drawdown()
        if dd >= self.max_drawdown:
            return False, f"HARD_STOP: total drawdown {dd:.1%} >= {self.max_drawdown:.1%}"

        daily_dd = portfolio.daily_drawdown()
        if daily_dd >= self.daily_loss_cap:
            return False, f"DAILY_STOP: daily loss {daily_dd:.1%} >= {self.daily_loss_cap:.1%}"

        return True, "ok"

    # ------------------------------------------------------------------ #
    # Position sizing — Kelly-inspired tiered model                       #
    # ------------------------------------------------------------------ #

    def compute_size(
        self,
        confidence: float,
        portfolio: "Portfolio",
        token_volume_24h: float,
    ) -> SizeDecision:
        """
        Tiered position sizing:
        - confidence >= 0.80 → 20% NAV (max)
        - confidence >= 0.65 → 15% NAV
        - confidence >= 0.50 → 10% NAV
        - confidence <  0.50 → no trade
        """
        ok, reason = self.check_hard_stops(portfolio)
        if not ok:
            return SizeDecision(allowed=False, nav_fraction=0.0, reason=reason)

        # Liquidity gate
        if token_volume_24h < self.min_liquidity:
            return SizeDecision(
                allowed=False,
                nav_fraction=0.0,
                reason=f"insufficient_liquidity: {token_volume_24h:.0f} < {self.min_liquidity:.0f}",
            )

        # Drawdown-aware scaling: reduce size as we approach the cap
        dd = portfolio.current_drawdown()
        dd_remaining = self.max_drawdown - dd
        if dd_remaining < 0.05:
            # Very close to cap — minimal size only
            base_fraction = 0.05
        elif dd_remaining < 0.10:
            base_fraction = min(self.per_trade_max, 0.10)
        else:
            base_fraction = self.per_trade_max

        # Confidence tiers (align với _PERPS_MIN_CONFIDENCE = 0.55)
        if confidence >= 0.80:
            fraction = base_fraction
        elif confidence >= 0.65:
            fraction = base_fraction * 0.75
        elif confidence >= 0.55:
            fraction = base_fraction * 0.50
        else:
            return SizeDecision(
                allowed=False,
                nav_fraction=0.0,
                reason=f"low_confidence: {confidence:.2f} < 0.55",
            )

        # Don't let single trade exceed 25% of daily 24h volume (market impact)
        nav = portfolio.nav_usd()
        trade_usd = nav * fraction
        if token_volume_24h > 0 and trade_usd > token_volume_24h * 0.25:
            fraction = (token_volume_24h * 0.25) / nav
            fraction = min(fraction, base_fraction)

        log.info(
            "position_sized",
            confidence=confidence,
            nav_fraction=round(fraction, 4),
            trade_usd=round(nav * fraction, 2),
            drawdown=round(dd, 4),
        )

        return SizeDecision(
            allowed=True,
            nav_fraction=fraction,
            reason="ok",
        )

    # ------------------------------------------------------------------ #
    # Day-level trade count check                                          #
    # ------------------------------------------------------------------ #

    def should_force_min_trade(self, trades_today: int, hours_remaining: int) -> bool:
        """True when we must trade to meet the 1-trade/day minimum."""
        cfg = get_settings()
        if trades_today < cfg.vicent_min_trades_per_day and hours_remaining <= 4:
            log.warning("force_min_trade", trades_today=trades_today, hours_remaining=hours_remaining)
            return True
        return False
