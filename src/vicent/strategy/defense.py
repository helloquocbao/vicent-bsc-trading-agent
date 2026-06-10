"""Market defense layer — protects capital when market conditions deteriorate.

Evaluates real-time market health based on:
  1. Market breadth      — percentage of bearish tokens
  2. Volatility spike    — Average ATR increase
  3. BTC crash guard     — BTC short-term price crashes (lead indicator)
  4. Momentum cascade    — multiple tokens experiencing negative ROC

Output: DefensePosture determining whether to:
  - Halt new trade entries
  - Scale down trade sizes
  - Raise confidence thresholds
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

import structlog

if TYPE_CHECKING:
    from vicent.signals.signals import TokenSignal

log = structlog.get_logger(__name__)


class DefenseLevel(str, Enum):
    NORMAL    = "normal"      # Market stable — normal operations
    CAUTION   = "caution"     # Signs of weakness — reduce size, tighten thresholds
    DEFENSIVE = "defensive"   # High risk — trade higher confidence entries with smaller sizes
    HALT      = "halt"        # Danger — halt all new entries completely


@dataclass
class DefensePosture:
    level: DefenseLevel
    size_multiplier: float        # multiplies trade size (0.0 - 1.0)
    min_confidence_add: float     # added to confidence threshold
    allow_new_entries: bool       # allows new trades
    max_positions_override: int | None  # limits max concurrent trades
    reason: str
    # Details for debugging
    bearish_breadth: float        # % bearish tokens (0-1)
    avg_atr_pct: float            # average ATR percentage
    btc_roc_short: float          # BTC short-term ROC
    cascade_count: int            # number of dumping tokens


# Configuration thresholds
_BREADTH_CAUTION   = 0.60   # 60% tokens bearish → caution
_BREADTH_DEFENSIVE = 0.75   # 75% bearish → defensive
_BREADTH_HALT      = 0.90   # 90% bearish → halt

_ATR_SPIKE_CAUTION   = 0.06   # average ATR > 6% → high volatility
_ATR_SPIKE_DEFENSIVE = 0.10   # > 10% → very high volatility

_BTC_CRASH_CAUTION   = -2.0   # BTC short-term ROC < -2% → caution
_BTC_CRASH_DEFENSIVE = -4.0   # < -4% → defensive
_BTC_CRASH_HALT      = -7.0   # < -7% → halt (flash crash)

_CASCADE_ROC_THRESHOLD = -2.5   # token considered dumping when ROC < -2.5%
_CASCADE_COUNT_DEFENSIVE = 5    # >=5 tokens dumping → defensive


def assess_market_health(
    signals: list["TokenSignal"],
    fear_greed: int = 50,
    btc_pct_1h: float = 0.0,
) -> DefensePosture:
    """Evaluate market health from all token signals.

    btc_pct_1h: BTC 1h % change from global_metrics — used as fallback when
    BTC is not in the signals list (e.g. CMC quota exceeded for BTC).

    Adjusts trade sizing, confidence thresholds, and decides if entries are allowed.
    """
    if not signals:
        return DefensePosture(
            level=DefenseLevel.NORMAL,
            size_multiplier=1.0,
            min_confidence_add=0.0,
            allow_new_entries=True,
            max_positions_override=None,
            reason="no_signals",
            bearish_breadth=0.0, avg_atr_pct=0.0,
            btc_roc_short=0.0, cascade_count=0,
        )

    from vicent.signals.signals import Direction

    # --- 1. Market breadth: % bearish tokens ---
    n = len(signals)
    bearish = sum(1 for s in signals if s.direction == Direction.SHORT)
    # Weak tokens with negative ROC also counted
    weak = sum(1 for s in signals if s.roc_short < -0.5)
    breadth = max(bearish, weak) / n if n > 0 else 0.0

    # --- 2. Volatility spike: Average ATR ---
    atrs = [s.intraday.atr_pct / 100 for s in signals
            if s.intraday and s.intraday.atr_pct]
    avg_atr = sum(atrs) / len(atrs) if atrs else 0.0

    # --- 3. BTC crash guard ---
    btc_roc = 0.0
    btc_sig = next((s for s in signals if s.symbol == "BTC"), None)
    if btc_sig:
        btc_roc = btc_sig.roc_short
    elif btc_pct_1h != 0.0:
        # Fallback: use the 1h % change from CMC global metrics when BTC
        # was not scored this iteration (quota exhausted, API timeout, etc.)
        btc_roc = btc_pct_1h
        log.debug("btc_crash_guard_fallback", source="global_metrics_1h", value=btc_roc)

    # --- 4. Momentum cascade: count of dumping tokens ---
    cascade = sum(1 for s in signals if s.roc_short < _CASCADE_ROC_THRESHOLD)

    # ====== Aggregate → DefenseLevel ======
    level = DefenseLevel.NORMAL
    reasons: list[str] = []

    # BTC crash check (strongest lead indicator)
    if btc_roc <= _BTC_CRASH_HALT:
        level = DefenseLevel.HALT
        reasons.append(f"BTC flash crash {btc_roc:.1f}%")
    elif btc_roc <= _BTC_CRASH_DEFENSIVE:
        level = max(level, DefenseLevel.DEFENSIVE, key=_level_rank)
        reasons.append(f"BTC down heavily {btc_roc:.1f}%")
    elif btc_roc <= _BTC_CRASH_CAUTION:
        level = max(level, DefenseLevel.CAUTION, key=_level_rank)
        reasons.append(f"BTC weak {btc_roc:.1f}%")

    # Breadth check
    if breadth >= _BREADTH_HALT:
        level = max(level, DefenseLevel.HALT, key=_level_rank)
        reasons.append(f"{breadth:.0%} tokens dumping")
    elif breadth >= _BREADTH_DEFENSIVE:
        level = max(level, DefenseLevel.DEFENSIVE, key=_level_rank)
        reasons.append(f"{breadth:.0%} tokens bearish")
    elif breadth >= _BREADTH_CAUTION:
        level = max(level, DefenseLevel.CAUTION, key=_level_rank)
        reasons.append(f"{breadth:.0%} tokens weak")

    # Volatility spike check
    if avg_atr >= _ATR_SPIKE_DEFENSIVE:
        level = max(level, DefenseLevel.DEFENSIVE, key=_level_rank)
        reasons.append(f"extreme ATR volatility {avg_atr:.1%}")
    elif avg_atr >= _ATR_SPIKE_CAUTION:
        level = max(level, DefenseLevel.CAUTION, key=_level_rank)
        reasons.append(f"high ATR volatility {avg_atr:.1%}")

    # Cascade check
    if cascade >= _CASCADE_COUNT_DEFENSIVE:
        level = max(level, DefenseLevel.DEFENSIVE, key=_level_rank)
        reasons.append(f"{cascade} tokens momentum cascade")

    # Extreme fear (F&G < 15)
    if fear_greed < 15:
        level = max(level, DefenseLevel.CAUTION, key=_level_rank)
        reasons.append(f"extremely low F&G {fear_greed}")

    # ====== Map level → posture ======
    posture_map = {
        DefenseLevel.NORMAL: dict(
            size_multiplier=1.0, min_confidence_add=0.0,
            allow_new_entries=True, max_positions_override=None,
        ),
        DefenseLevel.CAUTION: dict(
            size_multiplier=0.70, min_confidence_add=0.05,
            allow_new_entries=True, max_positions_override=2,
        ),
        DefenseLevel.DEFENSIVE: dict(
            size_multiplier=0.40, min_confidence_add=0.12,
            allow_new_entries=True, max_positions_override=1,
        ),
        DefenseLevel.HALT: dict(
            size_multiplier=0.0, min_confidence_add=0.50,
            allow_new_entries=False, max_positions_override=0,
        ),
    }
    p = posture_map[level]
    reason = "; ".join(reasons) if reasons else "market stable"

    posture = DefensePosture(
        level=level,
        size_multiplier=p["size_multiplier"],
        min_confidence_add=p["min_confidence_add"],
        allow_new_entries=p["allow_new_entries"],
        max_positions_override=p["max_positions_override"],
        reason=reason,
        bearish_breadth=round(breadth, 3),
        avg_atr_pct=round(avg_atr, 4),
        btc_roc_short=round(btc_roc, 2),
        cascade_count=cascade,
    )

    if level != DefenseLevel.NORMAL:
        log.warning(
            "defense_posture",
            level=level.value,
            size_mult=posture.size_multiplier,
            breadth=f"{breadth:.0%}",
            avg_atr=f"{avg_atr:.1%}",
            btc_roc=f"{btc_roc:.1f}%",
            cascade=cascade,
            reason=reason,
        )

    return posture


def _level_rank(level: DefenseLevel) -> int:
    """Severity order for max()."""
    return {
        DefenseLevel.NORMAL: 0,
        DefenseLevel.CAUTION: 1,
        DefenseLevel.DEFENSIVE: 2,
        DefenseLevel.HALT: 3,
    }[level]
