"""News sentiment signal — parses CMC latest news into a scored signal.

Design choices:
  - Time-decay: news from the past 30min scores 100%, 2h = 60%, 6h = 20%, older = 0%
  - Keyword scoring: tiered positive/negative/caution keyword sets
  - Dedup: same story repeated across outlets counts once
  - Output: NewsSentiment with score in [-1, +1] and freshest headline text
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import structlog

log = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword dictionaries — order matters (checked top-down, first match wins)
# ---------------------------------------------------------------------------

# Strong positive signals (score += 0.8)
_STRONG_POSITIVE = [
    "partnership", "launch", "mainnet", "listing", "upgrade", "milestone",
    "all-time high", "ath", "record", "adoption", "integration", "bullish",
    "breakout", "surge", "rally", "token burn", "buyback", "institutional",
    "etf approved", "regulated", "compliant",
]

# Mild positive signals (score += 0.4)
_MILD_POSITIVE = [
    "growth", "increase", "expand", "new feature", "testnet", "grant",
    "funding", "investment", "audit", "positive", "development", "update",
    "improvement", "community", "ecosystem",
]

# Strong negative signals (score -= 0.8)
_STRONG_NEGATIVE = [
    "hack", "exploit", "breach", "rug", "scam", "fraud", "lawsuit", "ban",
    "suspend", "emergency", "vulnerability", "stolen", "drain", "bankruptcy",
    "insolvency", "attack", "shut down", "delisted", "delist", "sec",
    "investigation", "arrested", "ponzi", "collapse",
]

# Mild negative signals (score -= 0.4)
_MILD_NEGATIVE = [
    "delay", "concern", "warning", "risk", "decline", "drop", "fall",
    "bearish", "correction", "selloff", "sell-off", "outflow", "controversy",
    "criticism", "issue", "problem", "bug",
]

# Caution signals — reduce confidence without strong direction (score *= 0.8)
_CAUTION = [
    "uncertainty", "volatile", "mixed", "unclear", "debate", "regulatory",
    "pending", "review", "scrutiny",
]


@dataclass
class NewsSentiment:
    """Sentiment result for a single token from its recent news."""
    symbol: str
    score: float                 # -1.0 to +1.0
    confidence: float            # 0.0 to 1.0 (how many fresh + relevant articles)
    article_count: int           # raw number of articles parsed
    fresh_count: int             # articles within 2h
    top_headline: str            # most relevant recent headline
    keywords_hit: list[str] = field(default_factory=list)


def score_news(symbol: str, articles: list[dict[str, Any]]) -> NewsSentiment:
    """Score a list of CMC news articles for a token.

    Each article is weighted by recency. Final score is the weighted mean
    of individual article scores, capped to [-1, +1].
    """
    if not articles:
        return NewsSentiment(
            symbol=symbol, score=0.0, confidence=0.0,
            article_count=0, fresh_count=0, top_headline="",
        )

    now = datetime.now(timezone.utc)
    weighted_scores: list[tuple[float, float]] = []  # (score, weight)
    fresh_count = 0
    top_headline = ""
    top_weight = 0.0
    all_keywords: list[str] = []
    seen_titles: set[str] = set()

    for article in articles[:20]:  # cap at 20 — avoid stale noise
        title = str(article.get("title", "") or article.get("name", ""))
        description = str(article.get("description", "") or article.get("text", ""))
        pub_str = article.get("published_at", article.get("date", ""))

        if not title:
            continue

        # Dedup by normalised title
        norm_title = re.sub(r"\s+", " ", title.lower().strip())
        if norm_title in seen_titles:
            continue
        seen_titles.add(norm_title)

        text = (title + " " + description).lower()

        # --- Recency weight ---
        weight = _recency_weight(pub_str, now)
        if weight <= 0:
            continue
        if weight >= 0.6:
            fresh_count += 1

        # --- Keyword scoring ---
        article_score = 0.0
        caution = False
        hits: list[str] = []

        for kw in _STRONG_POSITIVE:
            if kw in text:
                article_score += 0.8
                hits.append(f"+{kw}")
                break

        for kw in _MILD_POSITIVE:
            if kw in text:
                article_score += 0.4
                hits.append(f"+{kw}")
                break

        for kw in _STRONG_NEGATIVE:
            if kw in text:
                article_score -= 0.8
                hits.append(f"-{kw}")
                break

        for kw in _MILD_NEGATIVE:
            if kw in text:
                article_score -= 0.4
                hits.append(f"-{kw}")
                break

        for kw in _CAUTION:
            if kw in text:
                caution = True
                hits.append(f"?{kw}")
                break

        if caution:
            article_score *= 0.8

        # Clamp individual article score
        article_score = max(-1.0, min(1.0, article_score))
        weighted_scores.append((article_score, weight))
        all_keywords.extend(hits)

        if weight > top_weight:
            top_weight = weight
            top_headline = title

    if not weighted_scores:
        return NewsSentiment(
            symbol=symbol, score=0.0, confidence=0.0,
            article_count=len(articles), fresh_count=0, top_headline="",
        )

    # Weighted mean
    total_weight = sum(w for _, w in weighted_scores)
    if total_weight <= 0:
        composite = 0.0
    else:
        composite = sum(s * w for s, w in weighted_scores) / total_weight

    # Confidence: how many fresh articles we had + how strong the signal
    raw_conf = min(1.0, fresh_count / 3.0) * min(1.0, abs(composite) + 0.2)
    confidence = round(max(0.0, raw_conf), 3)

    log.info(
        "news_scored",
        symbol=symbol,
        score=round(composite, 3),
        confidence=confidence,
        articles=len(weighted_scores),
        fresh=fresh_count,
        top_headline=top_headline[:60],
    )

    return NewsSentiment(
        symbol=symbol,
        score=round(max(-1.0, min(1.0, composite)), 3),
        confidence=confidence,
        article_count=len(articles),
        fresh_count=fresh_count,
        top_headline=top_headline,
        keywords_hit=list(dict.fromkeys(all_keywords))[:6],  # dedup, keep order
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _recency_weight(pub_str: str, now: datetime) -> float:
    """Convert a publish timestamp to a 0-1 weight based on age.

    >6h old  → 0.0  (ignored)
    2-6h old → 0.2  (background signal)
    30min-2h → 0.6  (relevant)
    <30min   → 1.0  (breaking — full weight)
    """
    if not pub_str:
        return 0.0   # no timestamp = unknown age = don't trust

    try:
        # Try multiple formats CMC might return
        for fmt in (
            "%Y-%m-%dT%H:%M:%S.%fZ",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ):
            try:
                dt = datetime.strptime(pub_str, fmt)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                break
            except ValueError:
                continue
        else:
            return 0.0   # unknown format = don't trust

        age_minutes = (now - dt).total_seconds() / 60

        if age_minutes < 30:
            return 1.0
        elif age_minutes < 120:
            return 0.6
        elif age_minutes < 360:
            return 0.2
        else:
            return 0.0

    except Exception:
        return 0.0   # parse error = treat as stale
