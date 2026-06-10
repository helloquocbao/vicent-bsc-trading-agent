"""Pre-trade market sentiment analyzer.

This runs ONCE per iteration, RIGHT BEFORE any new order is opened.
It synthesizes 6 dimensions of market mood into a single SentimentReading
that gates, sizes, and adjusts every trade.

Design philosophy:
  - Regime (BULL/NEUTRAL/BEAR) tells us the BACKGROUND — computed once per hour
  - Sentiment tells us the MOMENT — "is right now a good time to pull the trigger?"
  - A trade can have a great signal AND good regime but still be in a bad moment
    (e.g. Fear & Greed just spiked to 90 in the last 30 min = entering at peak FOMO)

6 dimensions (each scored -1 to +1):
  1. Fear & Greed level + momentum (direction matters, not just level)
  2. Market cap TA trend (total crypto market cap EMA crossover)
  3. Narrative heat (are trending themes aligned with the token's sector?)
  4. Derivatives pressure (funding rate + OI combined)
  5. Macro event risk (upcoming Fed meeting / regulatory deadline?)
  6. Token-specific news freshness (bonus from news.py result)

Output: SentimentReading with:
  - verdict: ENTER | WAIT | BLOCK
  - multiplier: 0.0–1.2  (applied to position size at entry)
  - score: -1 to +1
  - summary: human-readable explanation for logging / DoraHacks submission
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

import structlog

if TYPE_CHECKING:
    from vicent.signals.news import NewsSentiment

log = structlog.get_logger(__name__)


class SentimentVerdict(str, Enum):
    ENTER = "enter"    # conditions are good — proceed at normal or boosted size
    WAIT  = "wait"     # mixed signals — skip this bar, wait for next loop
    BLOCK = "block"    # conditions are bad — do NOT open new positions this loop


@dataclass
class SentimentReading:
    verdict: SentimentVerdict
    score: float                    # -1.0 to +1.0 composite
    multiplier: float               # applied to nav_fraction (0.5–1.2)
    fear_greed: int                 # raw CMC value 0-100
    fg_momentum: str                # "rising" | "falling" | "stable"
    dominant_narrative: str         # top trending theme name
    narrative_aligned: bool         # does it align with the token we want to trade?
    funding_pressure: str           # "crowded_long" | "crowded_short" | "balanced"
    macro_event_imminent: bool      # big macro event in <24h?
    macro_event_name: str           # description of the event
    dimension_scores: dict[str, float] = field(default_factory=dict)
    summary: str = ""


# ---------------------------------------------------------------------------
# Previous Fear & Greed — tracked across iterations for momentum
# ---------------------------------------------------------------------------
_prev_fear_greed: int | None = None
_prev_fg_iter: int = 0


def analyze_sentiment(
    global_metrics: dict[str, Any],
    derivatives: dict[str, Any],
    marketcap_ta: dict[str, Any],
    narratives: list[dict[str, Any]],
    upcoming_events: list[dict[str, Any]],
    target_symbol: str = "",
    token_news: "NewsSentiment | None" = None,
    iteration: int = 0,
) -> SentimentReading:
    """Compute the pre-trade sentiment reading from all available data.

    All inputs are the raw CMC MCP responses — already fetched in the
    iteration step, so this function is pure computation (no I/O).
    """
    global _prev_fear_greed, _prev_fg_iter

    dim: dict[str, float] = {}

    # ------------------------------------------------------------------ #
    # 1. Fear & Greed + momentum                                          #
    # ------------------------------------------------------------------ #
    fg = _extract_fear_greed(global_metrics)
    fg_momentum = _compute_fg_momentum(fg, iteration)
    _prev_fear_greed = fg
    _prev_fg_iter = iteration

    # Score the F&G level itself
    if fg < 20:
        fg_score = 0.70    # extreme fear = contrarian BUY signal
    elif fg < 35:
        fg_score = 0.40    # fear = mild bullish
    elif fg < 50:
        fg_score = 0.10    # mild fear → slight positive
    elif fg < 65:
        fg_score = 0.10    # mild greed → neutral-positive
    elif fg < 80:
        fg_score = -0.10   # greed → slight caution
    else:
        fg_score = -0.40   # extreme greed = overbought / FOMO peak

    # Momentum adjustment: a RISING F&G (FOMO building) is more dangerous
    # than a stable high reading. A falling F&G after extreme high = sell signal.
    if fg_momentum == "rising" and fg > 70:
        fg_score -= 0.20   # accelerating greed = near top
    elif fg_momentum == "falling" and fg < 30:
        fg_score += 0.20   # capitulation easing = bottoming signal
    elif fg_momentum == "rising" and fg < 50:
        fg_score += 0.10   # market recovering from fear

    dim["fear_greed"] = max(-1.0, min(1.0, fg_score))

    # ------------------------------------------------------------------ #
    # 2. Market cap TA trend                                              #
    # ------------------------------------------------------------------ #
    mcap_score = _score_marketcap_ta(marketcap_ta)
    dim["marketcap_ta"] = mcap_score

    # ------------------------------------------------------------------ #
    # 3. Narrative heat + alignment                                       #
    # ------------------------------------------------------------------ #
    dominant_narrative, narrative_score, narrative_aligned = _score_narratives(
        narratives, target_symbol
    )
    dim["narrative"] = narrative_score

    # ------------------------------------------------------------------ #
    # 4. Derivatives pressure                                             #
    # ------------------------------------------------------------------ #
    funding_pressure, deriv_score = _score_derivatives(derivatives)
    dim["derivatives"] = deriv_score

    # ------------------------------------------------------------------ #
    # 5. Macro event risk                                                 #
    # ------------------------------------------------------------------ #
    macro_imminent, macro_name, macro_score = _score_macro_events(upcoming_events)
    dim["macro_event"] = macro_score

    # ------------------------------------------------------------------ #
    # 6. Token news freshness bonus                                       #
    # ------------------------------------------------------------------ #
    news_score = 0.0
    if token_news is not None and token_news.confidence > 0:
        # Amplify (don't duplicate) — news.py already scored it; here we use
        # it as a tiebreaker signal weighted lightly
        news_score = token_news.score * min(0.5, token_news.confidence)
    dim["news_freshness"] = news_score

    # ------------------------------------------------------------------ #
    # Weighted composite                                                  #
    # ------------------------------------------------------------------ #
    weights = {
        "fear_greed":   0.30,
        "marketcap_ta": 0.20,
        "narrative":    0.15,
        "derivatives":  0.20,
        "macro_event":  0.10,
        "news_freshness": 0.05,
    }
    composite = sum(dim.get(k, 0.0) * w for k, w in weights.items())
    composite = max(-1.0, min(1.0, composite))

    # ------------------------------------------------------------------ #
    # Verdict + multiplier                                                #
    # ------------------------------------------------------------------ #
    verdict, multiplier = _verdict_from_score(composite, macro_imminent, fg)

    summary = _build_summary(
        composite, verdict, fg, fg_momentum,
        dominant_narrative, narrative_aligned,
        funding_pressure, macro_imminent, macro_name,
    )

    log.info(
        "sentiment_analyzed",
        symbol=target_symbol or "global",
        verdict=verdict.value,
        score=round(composite, 3),
        multiplier=round(multiplier, 2),
        fg=fg,
        fg_momentum=fg_momentum,
        narrative=dominant_narrative,
        aligned=narrative_aligned,
        funding=funding_pressure,
        macro_event=macro_imminent,
        dimensions={k: round(v, 3) for k, v in dim.items()},
    )

    return SentimentReading(
        verdict=verdict,
        score=composite,
        multiplier=multiplier,
        fear_greed=fg,
        fg_momentum=fg_momentum,
        dominant_narrative=dominant_narrative,
        narrative_aligned=narrative_aligned,
        funding_pressure=funding_pressure,
        macro_event_imminent=macro_imminent,
        macro_event_name=macro_name,
        dimension_scores=dim,
        summary=summary,
    )


# ---------------------------------------------------------------------------
# Verdict mapping
# ---------------------------------------------------------------------------

def _verdict_from_score(
    score: float,
    macro_imminent: bool,
    fear_greed: int,
) -> tuple[SentimentVerdict, float]:
    """Map composite score to a verdict + position size multiplier.

    Hard BLOCK conditions (regardless of score):
      - Macro event imminent: massive uncertainty, hold cash
      - Extreme greed (>88): statistically near-peak, very dangerous to enter
    """
    # Hard blocks
    if macro_imminent:
        return SentimentVerdict.BLOCK, 0.0
    if fear_greed >= 88:
        return SentimentVerdict.BLOCK, 0.0

    # Score-based
    if score >= 0.30:
        # Strong positive sentiment → boost size slightly
        multiplier = min(1.20, 0.90 + score * 1.0)
        return SentimentVerdict.ENTER, round(multiplier, 2)
    elif score >= 0.05:
        # Mildly positive → enter at normal size
        return SentimentVerdict.ENTER, 1.00
    elif score >= -0.10:
        # Neutral → enter at reduced size (cautious)
        return SentimentVerdict.ENTER, 0.75
    elif score >= -0.30:
        # Mildly negative → wait for better entry
        return SentimentVerdict.WAIT, 0.0
    else:
        # Clearly negative sentiment → block new entries
        return SentimentVerdict.BLOCK, 0.0


# ---------------------------------------------------------------------------
# Dimension scorers
# ---------------------------------------------------------------------------

def _extract_fear_greed(data: dict[str, Any]) -> int:
    try:
        # New CMC MCP format: data.sentiment.fear_greed.current.index
        fg = data.get("sentiment", {}).get("fear_greed", {}).get("current", {})
        if fg:
            return int(fg.get("index", fg.get("value", 50)))
        # Legacy format
        value = data["data"]["fear_and_greed_index"]["value"]
        return int(value)
    except (KeyError, TypeError):
        try:
            return int(data["data"]["fear_and_greed_value"])
        except (KeyError, TypeError):
            return 50


def _compute_fg_momentum(current_fg: int, iteration: int) -> str:
    """Compare current F&G to previous iteration's value."""
    global _prev_fear_greed, _prev_fg_iter
    if _prev_fear_greed is None or (iteration - _prev_fg_iter) > 5:
        return "stable"
    delta = current_fg - _prev_fear_greed
    if delta >= 5:
        return "rising"
    elif delta <= -5:
        return "falling"
    return "stable"


def _score_marketcap_ta(ta: dict[str, Any]) -> float:
    """Score total market cap TA: EMA crossover + MACD + RSI."""
    try:
        data = ta.get("data", ta)
        score = 0.0

        # EMA crossover
        for timeframe in ["1d", "day_1", "4h"]:
            if timeframe in data:
                ind = data[timeframe].get("indicators", data[timeframe])
                ema20 = float(ind.get("ema_20", ind.get("ema20", 0)) or 0)
                ema50 = float(ind.get("ema_50", ind.get("ema50", 0)) or 0)
                if ema20 > 0 and ema50 > 0:
                    diff = (ema20 - ema50) / ema50
                    if diff > 0.01:
                        score += 0.40   # market cap EMA bullish
                    elif diff < -0.01:
                        score -= 0.40
                    # RSI of total market cap
                    rsi = float(ind.get("rsi", 50) or 50)
                    if rsi < 40:
                        score += 0.30
                    elif rsi > 70:
                        score -= 0.30
                    break  # stop at first valid timeframe

        return max(-1.0, min(1.0, score))
    except Exception:
        return 0.0


# Map token symbols/sectors to CMC narrative keywords
_SECTOR_NARRATIVE_KEYWORDS: dict[str, list[str]] = {
    "CAKE": ["defi", "dex", "yield", "pancake"],
    "AAVE": ["defi", "lending", "aave"],
    "UNI":  ["defi", "dex", "uniswap"],
    "INJ":  ["layer1", "cosmos", "injective"],
    "AVAX": ["layer1", "avalanche"],
    "ATOM": ["cosmos", "ibc", "layer1"],
    "DOT":  ["polkadot", "parachain", "layer1"],
    "LINK": ["oracle", "chainlink", "infrastructure"],
    "FET":  ["ai", "artificial intelligence", "agent"],
    "PENDLE": ["defi", "yield", "pendle"],
    "LDO":  ["staking", "liquid staking", "lido"],
    "AXS":  ["gaming", "nft", "axie"],
    "FLOKI": ["meme", "dog", "floki"],
    "BONK": ["meme", "solana"],
}


def _score_narratives(
    narratives: list[dict[str, Any]],
    target_symbol: str,
) -> tuple[str, float, bool]:
    """Return (dominant_narrative_name, score, aligned_with_target).

    Score:
      - Hot narrative is in our token's sector  → +0.5 (tailwind)
      - No hot narrative for our sector         →  0.0 (neutral)
      - Hot narrative is competitor/opposite     → -0.2 (headwind, but minor)
    """
    if not narratives:
        return "none", 0.0, False

    # Rank by market cap change or volume (use whatever CMC provides)
    def _narrative_heat(n: dict) -> float:
        return abs(float(n.get("market_cap_change_24h", n.get("change_24h", 0)) or 0))

    sorted_n = sorted(narratives, key=_narrative_heat, reverse=True)
    dominant = sorted_n[0]
    dom_name = str(dominant.get("name", dominant.get("title", "unknown"))).lower()
    dom_change = float(dominant.get("market_cap_change_24h", dominant.get("change_24h", 0)) or 0)

    # Check if dominant narrative is positive or negative
    if dom_change > 0:
        narrative_base_score = min(0.5, dom_change / 20.0)   # cap at +0.5
    else:
        narrative_base_score = max(-0.5, dom_change / 20.0)  # floor at -0.5

    # Check alignment with target symbol
    aligned = False
    symbol_keywords = _SECTOR_NARRATIVE_KEYWORDS.get(target_symbol.upper(), [])
    for kw in symbol_keywords:
        if kw in dom_name:
            aligned = True
            break

    if aligned and dom_change > 0:
        score = narrative_base_score * 1.3   # aligned tailwind: amplify
    elif aligned and dom_change < 0:
        score = narrative_base_score * 1.3   # aligned headwind: amplify too
    else:
        score = narrative_base_score * 0.3   # not our sector: minor influence

    return dom_name[:40], round(max(-1.0, min(1.0, score)), 3), aligned


def _score_derivatives(
    derivatives: dict[str, Any],
) -> tuple[str, float]:
    """Return (pressure_label, score).

    Funding rate analysis:
      - Strongly negative (-0.03%+):  shorts dominant = spot bullish = +0.4
      - Mildly negative:              balanced = +0.2
      - Neutral (±0.01%):             balanced = 0.0
      - Mildly positive:              longs slightly crowded = -0.1
      - Strongly positive (>0.03%):   crowded longs = liquidation risk = -0.4

    OI change amplifies:
      - Rising OI + positive price = adding longs = danger of cascade
      - Falling OI = deleveraging = price move is cleaner
    """
    try:
        data = derivatives.get("data", derivatives)

        # Funding rate
        funding = 0.0  # default neutral — không bias khi không có data
        raw = data.get("funding_rate")
        if raw is not None:
            if isinstance(raw, list):
                rates = [float(r.get("rate", 0)) for r in raw]
                funding = sum(rates) / max(len(rates), 1)
            else:
                funding = float(raw)

        # OI change
        oi_change = float(data.get("open_interest_24h_pct_change", 0) or 0)

        # Score from funding
        if funding < -0.02:
            score = 0.40
            label = "crowded_short"
        elif funding < 0:
            score = 0.20
            label = "balanced_lean_short"
        elif funding < 0.01:
            score = 0.05
            label = "balanced"
        elif funding < 0.03:
            score = -0.15
            label = "balanced_lean_long"
        else:
            score = -0.40
            label = "crowded_long"

        # OI modifier
        if oi_change > 10 and funding > 0.02:
            score -= 0.15   # longs being added in crowded market = dangerous
        elif oi_change < -10:
            score += 0.10   # leverage being washed out = cleaner moves

        return label, round(max(-1.0, min(1.0, score)), 3)
    except Exception:
        return "unknown", 0.0


def _score_macro_events(
    events: list[dict[str, Any]],
) -> tuple[bool, str, float]:
    """Return (imminent, event_name, score).

    Checks upcoming macro events for ones within 24 hours.
    High-impact events (Fed, SEC, ETF decisions) = BLOCK.
    Medium events = slight negative (uncertainty).
    """
    HIGH_IMPACT_KEYWORDS = [
        "fed", "fomc", "interest rate", "sec decision", "etf",
        "regulatory", "cpi", "inflation", "ban", "enforcement",
    ]
    MEDIUM_IMPACT_KEYWORDS = [
        "conference", "summit", "report", "data release", "expiry",
        "options", "futures settlement",
    ]

    if not events:
        return False, "", 0.0

    for event in events[:10]:
        name = str(event.get("name", event.get("title", "")) or "").lower()
        hours_until = _hours_until(
            event.get("date", event.get("scheduled_at", ""))
        )

        if hours_until is None:
            continue

        # Within 24 hours
        if hours_until <= 24:
            for kw in HIGH_IMPACT_KEYWORDS:
                if kw in name:
                    log.warning("macro_event_imminent", name=name, hours=hours_until)
                    return True, name[:50], -1.0

        # Within 48 hours — medium impact
        if hours_until <= 48:
            for kw in MEDIUM_IMPACT_KEYWORDS:
                if kw in name:
                    return False, name[:50], -0.20

    return False, "", 0.0


def _hours_until(date_str: str) -> float | None:
    """Parse a date string and return hours from now, or None if unparseable."""
    if not date_str:
        return None
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc)
    for fmt in (
        "%Y-%m-%dT%H:%M:%S.%fZ",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%d",
    ):
        try:
            dt = datetime.strptime(date_str, fmt)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            delta = (dt - now).total_seconds() / 3600
            return delta
        except ValueError:
            continue
    return None


def _build_summary(
    score: float,
    verdict: SentimentVerdict,
    fg: int,
    fg_momentum: str,
    narrative: str,
    narrative_aligned: bool,
    funding: str,
    macro_imminent: bool,
    macro_name: str,
) -> str:
    """Build a compact human-readable summary for logging and submission."""
    parts = [
        f"verdict={verdict.value.upper()}",
        f"score={score:+.2f}",
        f"F&G={fg}({fg_momentum})",
        f"funding={funding}",
    ]
    if narrative_aligned:
        parts.append(f"narrative_tailwind={narrative[:20]}")
    if macro_imminent:
        parts.append(f"MACRO_BLOCK={macro_name[:30]}")
    return " | ".join(parts)
