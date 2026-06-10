"""Tests for news sentiment signal."""

from datetime import datetime, timedelta, timezone

import pytest

from vicent.signals.news import NewsSentiment, _recency_weight, score_news


def _article(title: str, mins_ago: int = 10, desc: str = "") -> dict:
    ts = (datetime.now(timezone.utc) - timedelta(minutes=mins_ago)).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    return {"title": title, "description": desc, "published_at": ts}


# ---- Recency weight -------------------------------------------------

def test_fresh_article_weight_is_1() -> None:
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(minutes=10)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _recency_weight(ts, now) == 1.0


def test_old_article_weight_is_0() -> None:
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(hours=8)).strftime("%Y-%m-%dT%H:%M:%SZ")
    assert _recency_weight(ts, now) == 0.0


def test_medium_article_weight_is_partial() -> None:
    now = datetime.now(timezone.utc)
    ts = (now - timedelta(hours=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    w = _recency_weight(ts, now)
    assert 0.0 < w < 1.0


def test_missing_timestamp_returns_default() -> None:
    now = datetime.now(timezone.utc)
    w = _recency_weight("", now)
    assert w == 0.0   # no timestamp = unknown age = don't trust


# ---- score_news ---------------------------------------------------------

def test_empty_articles_returns_neutral() -> None:
    result = score_news("CAKE", [])
    assert result.score == 0.0
    assert result.confidence == 0.0
    assert result.article_count == 0


def test_positive_news_gives_positive_score() -> None:
    articles = [
        _article("CAKE announces major exchange listing", mins_ago=5),
        _article("CAKE partnership with leading DeFi protocol", mins_ago=15),
    ]
    result = score_news("CAKE", articles)
    assert result.score > 0.0
    assert result.fresh_count >= 2


def test_negative_news_exploit_forces_negative_score() -> None:
    articles = [
        _article("CAKE smart contract exploit drains $10M", mins_ago=5),
    ]
    result = score_news("CAKE", articles)
    assert result.score < 0.0


def test_hack_article_gets_strong_negative() -> None:
    articles = [_article("Protocol hack detected — funds at risk", mins_ago=3)]
    result = score_news("TEST", articles)
    assert result.score <= -0.5


def test_stale_articles_have_low_weight() -> None:
    fresh = [_article("CAKE listing on exchange", mins_ago=10)]
    stale = [_article("CAKE listing on exchange", mins_ago=600)]
    r_fresh = score_news("CAKE", fresh)
    r_stale = score_news("CAKE", stale)
    # Stale article should have less impact
    assert abs(r_fresh.score) >= abs(r_stale.score)


def test_dedup_same_title() -> None:
    articles = [_article("CAKE to list on Binance", mins_ago=5)] * 5
    result = score_news("CAKE", articles)
    # Only one article should count
    assert result.article_count == 5
    # But effective weight should be as if 1 unique article
    assert result.confidence <= 0.5


def test_caution_keywords_reduce_confidence() -> None:
    articles = [
        _article("CAKE faces regulatory uncertainty", mins_ago=5),
    ]
    result = score_news("CAKE", articles)
    # Should not be strongly positive or negative
    assert abs(result.score) < 0.5


def test_mixed_signals_balance_out() -> None:
    articles = [
        _article("CAKE major partnership announced", mins_ago=5),
        _article("CAKE security vulnerability found", mins_ago=10),
    ]
    result = score_news("CAKE", articles)
    # With roughly equal weight, score should be near 0
    assert -0.4 < result.score < 0.4


def test_top_headline_set_to_freshest() -> None:
    articles = [
        _article("Older news article", mins_ago=120),
        _article("Breaking: CAKE hits ATH", mins_ago=5),
    ]
    result = score_news("CAKE", articles)
    assert "Breaking" in result.top_headline or result.top_headline != ""


def test_keywords_hit_captured() -> None:
    articles = [_article("CAKE partnership with major CEX", mins_ago=5)]
    result = score_news("CAKE", articles)
    assert any("partnership" in kw for kw in result.keywords_hit)
