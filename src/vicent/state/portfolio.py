"""Portfolio tracker — computes NAV, drawdown, and open positions.

In paper mode: simulates holdings in memory.
In live mode: NAV is synced from Hyperliquid account equity on startup.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

from vicent.config import get_settings

log = structlog.get_logger(__name__)


@dataclass
class Position:
    symbol: str
    quantity: float
    avg_cost_usd: float
    current_price_usd: float = 0.0
    # PnL optimization tracking
    peak_price_usd: float = 0.0          # high-water mark since entry
    entry_atr_pct: float = 0.05           # volatility at entry — used for SL/TP
    scaled_out_t1: bool = False           # did we already take TP1?
    scaled_out_t2: bool = False           # did we already take TP2?

    @property
    def value_usd(self) -> float:
        return self.quantity * self.current_price_usd

    @property
    def unrealized_pnl_usd(self) -> float:
        return (self.current_price_usd - self.avg_cost_usd) * self.quantity

    @property
    def unrealized_pnl_pct(self) -> float:
        if self.avg_cost_usd <= 0:
            return 0.0
        return (self.current_price_usd - self.avg_cost_usd) / self.avg_cost_usd

    @property
    def peak_pnl_pct(self) -> float:
        """PnL at the high-water mark — used for trailing stop calculation."""
        if self.avg_cost_usd <= 0:
            return 0.0
        peak = max(self.peak_price_usd, self.current_price_usd, self.avg_cost_usd)
        return (peak - self.avg_cost_usd) / self.avg_cost_usd


class Portfolio:
    """Tracks NAV, positions, and drawdown metrics."""

    def __init__(self, initial_capital_usd: float) -> None:
        self.initial_capital = initial_capital_usd   # public — agents read this
        self._peak_nav = initial_capital_usd
        self._cash_usd = initial_capital_usd   # stablecoin balance
        self._positions: dict[str, Position] = {}
        self._day_start_nav = initial_capital_usd
        self._last_updated = datetime.now(timezone.utc)
        self._min_confidence_override: float | None = None  # set by reflexion
        self._perps_unrealized_pnl = 0.0   # injected by agent each iteration

    # ---- NAV -----------------------------------------------------------------

    def nav_usd(self) -> float:
        positions_value = sum(p.value_usd for p in self._positions.values())
        return self._cash_usd + positions_value + self._perps_unrealized_pnl

    def update_perps_pnl(self, unrealized_pnl_usd: float) -> None:
        """Inject unrealized PnL from perps positions so NAV/drawdown reflect them.

        Called by agent each iteration after sync with HL. Without this, perps
        losses don't affect NAV and hard_stops never trigger.
        """
        self._perps_unrealized_pnl = unrealized_pnl_usd
        nav = self.nav_usd()
        if nav > self._peak_nav:
            self._peak_nav = nav

    def update_prices(self, prices: dict[str, float]) -> None:
        """Update current prices for all held positions and track peak watermarks."""
        for sym, pos in self._positions.items():
            if sym in prices:
                pos.current_price_usd = prices[sym]
                # Track peak for trailing stop calculation
                if prices[sym] > pos.peak_price_usd:
                    pos.peak_price_usd = prices[sym]
        nav = self.nav_usd()
        if nav > self._peak_nav:
            self._peak_nav = nav
        self._last_updated = datetime.now(timezone.utc)

    # ---- Drawdown ------------------------------------------------------------

    def current_drawdown(self) -> float:
        """Peak-to-current drawdown as a positive fraction (0.10 = -10%)."""
        if self._peak_nav <= 0:
            return 0.0
        nav = self.nav_usd()
        return max(0.0, (self._peak_nav - nav) / self._peak_nav)

    def daily_drawdown(self) -> float:
        """Loss since day start as a positive fraction."""
        if self._day_start_nav <= 0:
            return 0.0
        nav = self.nav_usd()
        return max(0.0, (self._day_start_nav - nav) / self._day_start_nav)

    def reset_day_start(self) -> None:
        self._day_start_nav = self.nav_usd()

    # ---- Position management -------------------------------------------------

    def open_position(
        self,
        symbol: str,
        quantity: float,
        price_usd: float,
        atr_pct: float = 0.05,
    ) -> None:
        cost = quantity * price_usd
        if cost > self._cash_usd:
            log.warning("insufficient_cash", symbol=symbol, cost=cost, cash=self._cash_usd)
            cost = self._cash_usd
            quantity = cost / price_usd if price_usd > 0 else 0

        self._cash_usd -= cost
        if symbol in self._positions:
            existing = self._positions[symbol]
            total_qty = existing.quantity + quantity
            total_cost = existing.quantity * existing.avg_cost_usd + quantity * price_usd
            existing.avg_cost_usd = total_cost / total_qty if total_qty > 0 else price_usd
            existing.quantity = total_qty
            existing.current_price_usd = price_usd
            # Don't reset peak / scale-out state on average-up
            existing.peak_price_usd = max(existing.peak_price_usd, price_usd)
        else:
            self._positions[symbol] = Position(
                symbol=symbol,
                quantity=quantity,
                avg_cost_usd=price_usd,
                current_price_usd=price_usd,
                peak_price_usd=price_usd,
                entry_atr_pct=atr_pct,
            )
        log.info(
            "position_opened",
            symbol=symbol,
            qty=quantity,
            price=price_usd,
            atr_pct=round(atr_pct, 4),
            cash=round(self._cash_usd, 2),
        )

    def close_position(self, symbol: str, price_usd: float, fraction: float = 1.0) -> float:
        """Close `fraction` of a position. Returns USD received."""
        if symbol not in self._positions:
            log.warning("no_position_to_close", symbol=symbol)
            return 0.0
        pos = self._positions[symbol]
        qty_to_sell = pos.quantity * fraction
        proceeds = qty_to_sell * price_usd
        self._cash_usd += proceeds
        if fraction >= 1.0:
            del self._positions[symbol]
        else:
            pos.quantity -= qty_to_sell
        log.info("position_closed", symbol=symbol, qty=qty_to_sell, price=price_usd, proceeds=proceeds)
        return proceeds

    def has_position(self, symbol: str) -> bool:
        return symbol in self._positions and self._positions[symbol].quantity > 0

    def get_position(self, symbol: str) -> Position | None:
        return self._positions.get(symbol)

    # ---- Serialization -------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        nav = self.nav_usd()
        return {
            "nav_usd": nav,
            "cash_usd": self._cash_usd,
            "peak_nav": self._peak_nav,
            "drawdown": self.current_drawdown(),
            "daily_drawdown": self.daily_drawdown(),
            "positions": {
                sym: {
                    "qty": pos.quantity,
                    "avg_cost": pos.avg_cost_usd,
                    "current_price": pos.current_price_usd,
                    "value_usd": pos.value_usd,
                    "pnl_pct": pos.unrealized_pnl_pct,
                    "peak_price_usd": pos.peak_price_usd,
                    "entry_atr_pct": pos.entry_atr_pct,
                    "scaled_out_t1": pos.scaled_out_t1,
                    "scaled_out_t2": pos.scaled_out_t2,
                }
                for sym, pos in self._positions.items()
            },
            "initial_capital": self.initial_capital,
            "total_return_pct": (nav - self.initial_capital) / self.initial_capital if self.initial_capital > 0 else 0,
            "last_updated": self._last_updated.isoformat(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_snapshot_dict(cls, snapshot_data: dict[str, Any]) -> "Portfolio":
        """Reconstruct a Portfolio object from a snapshot dictionary."""
        initial_capital = snapshot_data.get("initial_capital", 100.0)
        p = cls(initial_capital)
        p._cash_usd = snapshot_data.get("cash_usd", initial_capital)
        p._peak_nav = snapshot_data.get("peak_nav", initial_capital)
        
        # Restore positions
        positions_data = snapshot_data.get("positions", {})
        if isinstance(positions_data, str):
            try:
                positions_data = json.loads(positions_data)
            except Exception:
                positions_data = {}

        for sym, details in positions_data.items():
            qty = details.get("qty", 0.0)
            if qty <= 0:
                continue
            avg_cost = details.get("avg_cost", 0.0)
            current_price = details.get("current_price", avg_cost)
            peak_price = details.get("peak_price_usd", current_price)
            entry_atr = details.get("entry_atr_pct", 0.05)
            t1 = details.get("scaled_out_t1", False)
            t2 = details.get("scaled_out_t2", False)

            p._positions[sym] = Position(
                symbol=sym,
                quantity=qty,
                avg_cost_usd=avg_cost,
                current_price_usd=current_price,
                peak_price_usd=peak_price,
                entry_atr_pct=entry_atr,
                scaled_out_t1=t1,
                scaled_out_t2=t2,
            )
        
        if "last_updated" in snapshot_data:
            try:
                p._last_updated = datetime.fromisoformat(snapshot_data["last_updated"])
            except Exception:
                pass
        return p

    @classmethod
    def from_live_equity(cls, equity_usd: float) -> "Portfolio":
        """Reconstruct portfolio from live equity (e.g. Hyperliquid account value)."""
        p = cls(equity_usd)
        p._cash_usd = equity_usd
        return p

