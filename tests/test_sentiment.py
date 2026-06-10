"""Tests for the pre-trade market sentiment analyzer."""

import pytest

from vicent.signals.sentiment import (
    SentimentVerdict,
    _compute_fg_momentum,
    _hours_until,
    _score_derivatives,
    _score_macro_events,
    _score_marketcap_ta,
    _score_narratives,
    analyze_sentiment,
)


# ---- helpers ----------------------------------------------------------------

def _global(fg: int) -> dict:
    return {"data": {"fear_and_greed_index": {"value": fg}}}


def _deriv(funding: float, oi_change: float = 0.0) -> dict:
    return {"data": {"funding_rate": funding, "open_interest_24h_pct_change": oi_change}}


def _mcap_ta(rsi: float = 50.0, ema20: float = 100.0, ema50: float = 98.0) -> dict:
    return {
        "data": {
            "1d": {
                "indicators": {
                    "rsi": rsi,
                    "ema_20": ema20,
                    "ema_50": ema50,
                }
            }
        }
    }


def _narrative(name: str, change_pct: float) -> dict:
    return {"name": name, "market_cap_change_24h": change_pct}


def _event(name: str, hours_from_now: float) -> dict:
    from datetime import datetime, timedelta, timezone
    dt = datetime.now(timezone.utc) + timedelta(hours=hours_from_now)
    return {"name": name, "date": dt.strftime("%Y-%m-%dT%H:%M:%SZ")}


# ---- _hours_until -----------------------------------------------------------

def test_hours_until_future() -> None:
    from datetime import datetime, timedelta, timezone
    future = (datetime.now(timezone.utc) + timedelta(hours=6)).strftime("%Y-%m-%dT%H:%M:%SZ")
    h = _hours_until(future)
    assert h is not None
    assert 5.5 < h < 6.5


def test_hours_until_past() -> None:
    from datetime import datetime, timedelta, timezone
    past = (datetime.now(timezone.utc) - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
    h = _hours_until(past)
    assert h is not None
    assert h < 0


def test_hours_until_empty() -> None:
    assert _hours_until("") is None


def test_hours_until_bad_string() -> None:
    assert _hours_until("not-a-date") is None


# ---- Fear & Greed momentum --------------------------------------------------

def test_fg_momentum_rising() -> None:
    import vicent.signals.sentiment as s
    s._prev_fear_greed = 50
    s._prev_fg_iter = 1
    result = _compute_fg_momentum(60, iteration=2)
    assert result == "rising"


def test_fg_momentum_falling() -> None:
    import vicent.signals.sentiment as s
    s._prev_fear_greed = 70
    s._prev_fg_iter = 1
    result = _compute_fg_momentum(60, iteration=2)
    assert result == "falling"


def test_fg_momentum_stable() -> None:
    import vicent.signals.sentiment as s
    s._prev_fear_greed = 55
    s._prev_fg_iter = 1
    result = _compute_fg_momentum(57, iteration=2)
    assert result == "stable"


def test_fg_momentum_stale_prev_returns_stable() -> None:
    import vicent.signals.sentiment as s
    s._prev_fear_greed = 50
    s._prev_fg_iter = 1
    # iteration very far ahead → stale
    result = _compute_fg_momentum(80, iteration=20)
    assert result == "stable"


# ---- _score_derivatives -----------------------------------------------------

def test_crowded_long_is_negative() -> None:
    label, score = _score_derivatives(_deriv(funding=0.05))
    assert score < 0
    assert label == "crowded_long"


def test_crowded_short_is_positive() -> None:
    label, score = _score_derivatives(_deriv(funding=-0.03))
    assert score > 0
    assert label == "crowded_short"


def test_balanced_funding_near_zero() -> None:
    _, score = _score_derivatives(_deriv(funding=0.005))
    assert -0.2 < score < 0.2


def test_crowded_long_plus_rising_oi_more_negative() -> None:
    _, score_plain = _score_derivatives(_deriv(funding=0.04, oi_change=0))
    _, score_rising = _score_derivatives(_deriv(funding=0.04, oi_change=15))
    assert score_rising <= score_plain


def test_falling_oi_slightly_bullish() -> None:
    _, score_normal = _score_derivatives(_deriv(funding=0.01, oi_change=0))
    _, score_falling = _score_derivatives(_deriv(funding=0.01, oi_change=-15))
    assert score_falling >= score_normal


# ---- _score_macro_events ----------------------------------------------------

def test_fed_event_within_24h_blocks() -> None:
    imminent, name, score = _score_macro_events([_event("FOMC Fed Rate Decision", hours_from_now=10)])
    assert imminent is True
    assert score == -1.0


def test_high_impact_event_outside_24h_does_not_block() -> None:
    imminent, _, _ = _score_macro_events([_event("FOMC Fed Rate Decision", hours_from_now=30)])
    assert imminent is False


def test_no_events_returns_neutral() -> None:
    imminent, name, score = _score_macro_events([])
    assert imminent is False
    assert score == 0.0


def test_options_expiry_within_48h_reduces_score() -> None:
    _, _, score = _score_macro_events([_event("BTC options expiry", hours_from_now=36)])
    assert score < 0.0


def test_unrelated_event_ignored() -> None:
    imminent, _, score = _score_macro_events([_event("Blockchain gaming summit", hours_from_now=5)])
    assert imminent is False
    # summit is medium impact at most
    assert score >= -0.25


# ---- _score_narratives ------------------------------------------------------

def test_aligned_bullish_narrative_boosts_score() -> None:
    narratives = [_narrative("defi yield farming boom", change_pct=15.0)]
    name, score, aligned = _score_narratives(narratives, target_symbol="CAKE")
    assert aligned is True
    assert score > 0.0


def test_aligned_bearish_narrative_penalises() -> None:
    narratives = [_narrative("defi protocol failures", change_pct=-12.0)]
    name, score, aligned = _score_narratives(narratives, target_symbol="CAKE")
    assert aligned is True
    assert score < 0.0


def test_unaligned_narrative_has_smaller_impact() -> None:
    narratives = [_narrative("meme coin season", change_pct=20.0)]
    name, score_unaligned, aligned = _score_narratives(narratives, target_symbol="CAKE")
    assert aligned is False
    _, score_aligned, _ = _score_narratives(
        [_narrative("defi boom", change_pct=20.0)], target_symbol="CAKE"
    )
    assert score_aligned > score_unaligned


def test_empty_narratives_neutral() -> None:
    name, score, aligned = _score_narratives([], "CAKE")
    assert score == 0.0
    assert aligned is False


# ---- Full analyze_sentiment -------------------------------------------------

def test_extreme_greed_blocks_entry() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=92),
        derivatives=_deriv(funding=0.04),
        marketcap_ta=_mcap_ta(),
        narratives=[],
        upcoming_events=[],
    )
    assert reading.verdict == SentimentVerdict.BLOCK
    assert reading.multiplier == 0.0


def test_macro_event_imminent_blocks_entry() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=55),
        derivatives=_deriv(funding=0.01),
        marketcap_ta=_mcap_ta(),
        narratives=[],
        upcoming_events=[_event("FOMC rate decision", hours_from_now=8)],
    )
    assert reading.verdict == SentimentVerdict.BLOCK
    assert reading.macro_event_imminent is True


def test_bullish_conditions_enter_with_boost() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=25),      # fear = contrarian bullish
        derivatives=_deriv(funding=-0.02),  # shorts paying = bullish
        marketcap_ta=_mcap_ta(rsi=38, ema20=102, ema50=98),  # bullish TA
        narratives=[_narrative("defi surge", change_pct=10.0)],
        upcoming_events=[],
        target_symbol="CAKE",
    )
    assert reading.verdict == SentimentVerdict.ENTER
    assert reading.multiplier >= 1.0


def test_negative_sentiment_blocks_or_waits() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=78),
        derivatives=_deriv(funding=0.05, oi_change=15),
        marketcap_ta=_mcap_ta(rsi=72, ema20=95, ema50=100),  # bearish TA
        narratives=[_narrative("defi hack wave", change_pct=-15)],
        upcoming_events=[],
        target_symbol="CAKE",
    )
    assert reading.verdict in (SentimentVerdict.WAIT, SentimentVerdict.BLOCK)


def test_rising_fg_in_greed_zone_penalises() -> None:
    import vicent.signals.sentiment as s
    s._prev_fear_greed = 70
    s._prev_fg_iter = 1

    reading = analyze_sentiment(
        global_metrics=_global(fg=78),   # +8 from last = "rising" in greed zone
        derivatives=_deriv(funding=0.01),
        marketcap_ta=_mcap_ta(),
        narratives=[],
        upcoming_events=[],
        iteration=2,
    )
    # Rising F&G in greed zone should reduce score
    assert reading.score < 0.3


def test_falling_fg_from_extreme_fear_is_bullish() -> None:
    import vicent.signals.sentiment as s
    s._prev_fear_greed = 20
    s._prev_fg_iter = 1

    reading = analyze_sentiment(
        global_metrics=_global(fg=10),   # still low = falling = bottoming
        derivatives=_deriv(funding=-0.01),
        marketcap_ta=_mcap_ta(rsi=35, ema20=100, ema50=99),
        narratives=[],
        upcoming_events=[],
        iteration=2,
    )
    assert reading.score > 0.0


def test_multiplier_bounded() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=20),
        derivatives=_deriv(funding=-0.03),
        marketcap_ta=_mcap_ta(rsi=30, ema20=105, ema50=95),
        narratives=[_narrative("defi boom", change_pct=20)],
        upcoming_events=[],
        target_symbol="CAKE",
    )
    assert 0.0 <= reading.multiplier <= 1.20


def test_dimension_scores_all_present() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=50),
        derivatives=_deriv(funding=0.01),
        marketcap_ta=_mcap_ta(),
        narratives=[],
        upcoming_events=[],
    )
    for dim in ("fear_greed", "marketcap_ta", "narrative", "derivatives", "macro_event", "news_freshness"):
        assert dim in reading.dimension_scores


def test_neutral_entry_has_reduced_multiplier() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=50),
        derivatives=_deriv(funding=0.01),
        marketcap_ta=_mcap_ta(rsi=50, ema20=100, ema50=100),
        narratives=[],
        upcoming_events=[],
    )
    # Near-zero score → cautious ENTER at 0.75x
    if reading.verdict == SentimentVerdict.ENTER:
        assert reading.multiplier <= 1.0


def test_summary_string_nonempty() -> None:
    reading = analyze_sentiment(
        global_metrics=_global(fg=55),
        derivatives=_deriv(funding=0.01),
        marketcap_ta=_mcap_ta(),
        narratives=[],
        upcoming_events=[],
    )
    assert len(reading.summary) > 10
    assert "verdict=" in reading.summary
