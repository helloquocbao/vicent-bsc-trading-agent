"""Per-token signal scoring.

Takes CMC technical analysis + quotes data and produces a TokenSignal
with a normalized confidence score and direction (LONG / SHORT / FLAT).

Signal weight breakdown (total = 100%):
  momentum  30%  — price action 1h + 24h
  rsi       20%  — RSI overbought/oversold
  macd      15%  — MACD crossover direction
  ema       10%  — EMA20 vs EMA50 trend
  volume    10%  — volume surge vs 7d avg
  news      10%  — freshness-weighted news sentiment  ← NEW
  whale      5%  — holder distribution pressure       ← NEW
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from vicent.signals.indicators import IntradayIndicators
from vicent.signals.news import NewsSentiment, score_news
from vicent.signals.prediction import PredictiveSignal, compute_predictive_signal
from vicent.signals.whale import WhalePressure, score_whale

log = structlog.get_logger(__name__)


class Direction(str, Enum):
    LONG = "long"
    SHORT = "short"
    FLAT = "flat"


@dataclass
class TokenSignal:
    symbol: str
    cmc_id: int
    direction: Direction
    confidence: float          # 0.0 – 1.0
    price_usd: float
    volume_24h: float
    volume_change_pct: float | None   # vs 7-day avg, None if data unavailable
    rsi_1h: float | None
    macd_signal: str | None    # "bullish" | "bearish" | "neutral"
    ema_trend: str | None      # "up" | "down" | "flat"
    volatility_pct: float = 0.05  # ATR-style daily volatility proxy
    news: NewsSentiment | None = None    # freshness-weighted news score
    whale: WhalePressure | None = None   # holder distribution pressure
    intraday: IntradayIndicators | None = None  # self-computed intraday TA
    prediction: PredictiveSignal | None = None  # pattern/ADX/breakout prediction
    roc_short: float = 0.0               # short-term momentum %
    using_intraday: bool = False         # True if intraday TA was used
    sub_scores: dict[str, float] = field(default_factory=dict)


def score_token(
    symbol: str,
    cmc_id: int,
    quotes: dict[str, Any],
    ta: dict[str, Any],
    news_articles: list[dict[str, Any]] | None = None,
    whale_metrics: dict[str, Any] | None = None,
    intraday: IntradayIndicators | None = None,
    price_series: list[float] | None = None,
    ohlcv_candles: list[dict] | None = None,   # real HL OHLCV — for accurate pattern detection
    funding_rate: float = 0.0,
) -> TokenSignal:
    """Combine quote + TA + news + whale + intraday data into a TokenSignal.

    When `intraday` has enough data, its RSI/MACD/EMA OVERRIDE the daily CMC
    values (they match the trading timeframe). Otherwise we fall back to the
    daily CMC TA. Short-term ROC is added as a momentum trigger.
    """

    # --- Extract quote data ---
    quote_data = _extract_quote(quotes, cmc_id)
    price = quote_data.get("price", 0.0)
    volume_24h = quote_data.get("volume_24h", 0.0)
    pct_1h = quote_data.get("percent_change_1h", 0.0)
    pct_24h = quote_data.get("percent_change_24h", 0.0)
    pct_7d = quote_data.get("percent_change_7d", 0.0)

    # --- Volatility proxy (ATR-style daily) ---
    from vicent.strategy.optimization import compute_volatility
    volatility_pct = compute_volatility(pct_1h, pct_24h, pct_7d)

    # --- Volume surge ---
    volume_change_pct = _estimate_volume_surge(quotes, cmc_id)

    # --- TA signals: prefer intraday when available, else daily CMC ---
    using_intraday = intraday is not None and intraday.has_enough_data
    roc_short = intraday.roc_short if intraday is not None else 0.0

    if using_intraday:
        rsi = intraday.rsi
        macd_signal = intraday.macd_signal
        ema_trend = intraday.ema_trend
    else:
        rsi = _extract_rsi(ta)
        macd_signal = _extract_macd_signal(ta)
        ema_trend = _extract_ema_trend(ta)

    # --- News sentiment ---
    news_result: NewsSentiment | None = None
    if news_articles is not None:
        news_result = score_news(symbol, news_articles)

    # --- Whale / holder pressure ---
    whale_result: WhalePressure | None = None
    if whale_metrics is not None:
        whale_result = score_whale(symbol, whale_metrics)

    # --- Predictive algorithms (pattern + directional ADX + breakout) ---
    prediction_result: PredictiveSignal | None = None
    if intraday is not None and intraday.has_enough_data and price_series is not None and len(price_series) >= 26:
        # Pass real OHLCV candles if available — otherwise falls back to close-only estimate
        prediction_result = compute_predictive_signal(
            prices=price_series,
            volume_change_pct=volume_change_pct,
            bb_position=intraday.bb_position,
            bb_squeeze=intraday.bb_squeeze,
            roc_short=intraday.roc_short,
            roc_medium=intraday.roc_medium,
            ohlcv_candles=ohlcv_candles,
        )

    # --- Score each sub-signal ---
    scores: dict[str, float] = {}

    # 1. Momentum (25%) — blend daily % change + intraday ROC
    mom = 0.0
    if pct_1h > 1.0:
        mom += 0.4
    elif pct_1h > 0.3:
        mom += 0.2
    elif pct_1h < -1.0:
        mom -= 0.4
    elif pct_1h < -0.3:
        mom -= 0.2

    if pct_24h > 3.0:
        mom += 0.25
    elif pct_24h > 1.0:
        mom += 0.12
    elif pct_24h < -3.0:
        mom -= 0.25
    elif pct_24h < -1.0:
        mom -= 0.12
    scores["momentum"] = max(-1.0, min(1.0, mom))

    # 2. Short-term ROC momentum trigger (15%) — intraday entry signal
    roc_score = 0.0
    if using_intraday:
        if roc_short > 1.5:   roc_score = 0.9
        elif roc_short > 0.6: roc_score = 0.5
        elif roc_short > 0.2: roc_score = 0.2
        elif roc_short < -1.5: roc_score = -0.9
        elif roc_short < -0.6: roc_score = -0.5
        elif roc_short < -0.2: roc_score = -0.2
    scores["roc"] = roc_score

    # 3. RSI (15%)
    rsi_score = 0.0
    if rsi is not None:
        if rsi < 30:   rsi_score = 0.8
        elif rsi < 45: rsi_score = 0.4
        elif rsi < 55: rsi_score = 0.0
        elif rsi < 70: rsi_score = -0.3
        else:          rsi_score = -0.7
    scores["rsi"] = rsi_score

    # 4. Stochastic RSI (8%) — faster overbought/oversold than plain RSI
    stoch_score = 0.0
    if using_intraday and intraday is not None and intraday.stoch_rsi is not None:
        sr = intraday.stoch_rsi
        if sr < 20:    stoch_score = 0.8   # oversold → strong buy signal
        elif sr < 35:  stoch_score = 0.4
        elif sr < 65:  stoch_score = 0.0
        elif sr < 80:  stoch_score = -0.3
        else:          stoch_score = -0.7  # overbought
    scores["stoch_rsi"] = stoch_score

    # 5. MACD (12%)
    macd_score = 0.7 if macd_signal == "bullish" else (-0.7 if macd_signal == "bearish" else 0.0)
    scores["macd"] = macd_score

    # 6. Bollinger Bands (10%) — breakout detection + squeeze
    bb_score = 0.0
    if using_intraday and intraday is not None and intraday.bb_position is not None:
        pos = intraday.bb_position
        if intraday.bb_squeeze:
            bb_score = 0.0  # squeeze: wait for direction, no signal yet
        elif pos < 0.1:
            bb_score = 0.7  # near lower band = oversold, bounce likely
        elif pos < 0.25:
            bb_score = 0.35
        elif pos > 0.9:
            bb_score = -0.7  # near upper band = overbought
        elif pos > 0.75:
            bb_score = -0.35
        else:
            bb_score = 0.0  # mid-band = neutral
    scores["bb"] = bb_score

    # 7. ADX trend strength (8%) — filter out sideways noise
    # ADX doesn't give direction, it amplifies or dampens other signals
    adx_multiplier = 1.0
    if using_intraday and intraday is not None and intraday.adx is not None:
        if intraday.adx > 40:
            adx_multiplier = 1.3   # strong trend — boost directional signals
        elif intraday.adx > 25:
            adx_multiplier = 1.1
        elif intraday.adx < 15:
            adx_multiplier = 0.5   # sideways — heavily discount directional signals
        elif intraday.adx < 20:
            adx_multiplier = 0.7
    # ADX score = 0 (it's a multiplier, not a direction signal)
    scores["adx"] = 0.0

    # 8. VWAP position (7%) — key intraday support/resistance
    vwap_score = 0.0
    if using_intraday and intraday is not None:
        if intraday.vwap_position == "above":
            vwap_score = 0.4   # price above VWAP = bullish
        elif intraday.vwap_position == "below":
            vwap_score = -0.4  # price below VWAP = bearish
    scores["vwap"] = vwap_score

    # 9. EMA trend (5%)
    ema_score = 0.6 if ema_trend == "up" else (-0.6 if ema_trend == "down" else 0.0)
    scores["ema"] = ema_score

    # 10. Volume surge (3%)
    vol_score = 0.0
    has_volume = volume_change_pct is not None
    if has_volume:
        if volume_change_pct > 100:   vol_score = 0.8
        elif volume_change_pct > 50:  vol_score = 0.4
        elif volume_change_pct < -40: vol_score = -0.3
    scores["volume"] = vol_score

    # 11. News sentiment (5%)
    news_score = 0.0
    if news_result is not None and news_result.confidence > 0:
        news_score = news_result.score * news_result.confidence
    scores["news"] = news_score

    # 12. Whale pressure (2%)
    whale_score = 0.0
    if whale_result is not None and whale_result.confidence > 0:
        whale_score = whale_result.score * whale_result.confidence
    scores["whale"] = whale_score

    # 13. Predictive signal (10% when available) — pattern + directional ADX + breakout
    pred_score = 0.0
    has_prediction = prediction_result is not None and prediction_result.entry_quality != "skip"
    if has_prediction:
        pred_score = prediction_result.combined_score * prediction_result.combined_confidence
    scores["prediction"] = pred_score

    # --- Weighted composite ---
    has_news  = news_result  is not None and news_result.confidence  > 0
    has_whale = whale_result is not None and whale_result.confidence > 0
    has_roc   = using_intraday

    weights: dict[str, float] = {
        "momentum":   0.20,
        "roc":        0.13 if has_roc else 0.0,
        "rsi":        0.12,
        "stoch_rsi":  0.07 if has_roc else 0.0,
        "macd":       0.10,
        "bb":         0.08 if has_roc else 0.0,
        "adx":        0.00,   # multiplier only
        "vwap":       0.06 if has_roc else 0.0,
        "ema":        0.05,
        "volume":     0.03 if has_volume else 0.0,
        "news":       0.05 if has_news else 0.0,
        "whale":      0.02 if has_whale else 0.0,
        "prediction": 0.10 if has_prediction else 0.0,  # ← new
    }

    # Redistribute zero-weight slots back to momentum + rsi
    assigned = sum(weights.values())
    if assigned < 1.0:
        gap = 1.0 - assigned
        weights["momentum"] += gap * 0.55
        weights["rsi"]      += gap * 0.45

    # Apply ADX multiplier to directional sub-scores
    directional_keys = ["roc", "rsi", "stoch_rsi", "macd", "bb", "vwap", "ema", "momentum", "prediction"]
    composite = 0.0
    for k, w in weights.items():
        s = scores.get(k, 0.0)
        if k in directional_keys:
            s *= adx_multiplier
        composite += s * w

    # --- ADAPTIVE STRATEGY: trend-follow vs mean-revert + funding capture ---
    # This is the key to profiting in ANY market direction.
    # In sideways markets, this FLIPS the composite to fade extremes
    # instead of just dampening it.
    from vicent.strategy.adaptive import apply_adaptive_to_composite, select_strategy
    adaptive = select_strategy(intraday, funding_rate=funding_rate)
    if adaptive.mode.value != "no_trade":
        composite = apply_adaptive_to_composite(composite, adaptive)
        scores["adaptive_mode"] = adaptive.directional_bias

    # --- News emergency override ---
    # A strong negative news signal (hack/exploit/scam) forces FLAT or SHORT
    # regardless of TA, because price hasn't reacted yet
    if news_result is not None and news_result.score <= -0.6 and news_result.confidence >= 0.5:
        log.warning(
            "news_emergency_override",
            symbol=symbol,
            news_score=news_result.score,
            headline=news_result.top_headline[:60],
        )
        return TokenSignal(
            symbol=symbol, cmc_id=cmc_id,
            direction=Direction.FLAT,
            confidence=0.0,
            price_usd=price, volume_24h=volume_24h,
            volume_change_pct=volume_change_pct,
            rsi_1h=rsi, macd_signal=macd_signal, ema_trend=ema_trend,
            volatility_pct=volatility_pct,
            news=news_result, whale=whale_result,
            intraday=intraday, prediction=prediction_result,
            roc_short=roc_short, using_intraday=using_intraday,
            sub_scores=scores,
        )

    # --- Whale emergency override ---
    # Whales distributing heavily + traders FOMO-ing in → force FLAT
    if (whale_result is not None
            and whale_result.score <= -0.5
            and whale_result.confidence >= 0.5
            and scores["momentum"] > 0):   # price still up = distribution into strength
        log.warning(
            "whale_distribution_override",
            symbol=symbol,
            whale_score=whale_result.score,
            trader_pct=whale_result.trader_pct,
        )
        composite = min(composite, 0.0)

    # Map composite → direction + confidence
    if composite >= 0.25:
        direction = Direction.LONG
        confidence = min(1.0, (composite - 0.25) / 0.75 * 0.8 + 0.4)
    elif composite <= -0.25:
        direction = Direction.SHORT
        confidence = min(1.0, (abs(composite) - 0.25) / 0.75 * 0.8 + 0.4)
    else:
        direction = Direction.FLAT
        confidence = 0.0

    log.info(
        "token_scored",
        symbol=symbol,
        direction=direction.value,
        confidence=round(confidence, 3),
        composite=round(composite, 3),
        rsi=round(rsi, 1) if rsi else None,
        stoch_rsi=round(intraday.stoch_rsi, 1) if (using_intraday and intraday and intraday.stoch_rsi) else None,
        macd=macd_signal,
        ema=ema_trend,
        bb_pos=round(intraday.bb_position, 2) if (using_intraday and intraday and intraday.bb_position is not None) else None,
        adx=round(intraday.adx, 1) if (using_intraday and intraday and intraday.adx) else None,
        vwap=intraday.vwap_position if (using_intraday and intraday) else None,
        roc_short=round(roc_short, 3) if has_roc else None,
        intraday=using_intraday,
        adaptive_mode=adaptive.mode.value if using_intraday else None,
        news_score=round(news_score, 3) if has_news else None,
        pred_score=round(pred_score, 3) if has_prediction else None,
        pred_quality=prediction_result.entry_quality if prediction_result else None,
        pred_pattern=prediction_result.pattern.pattern.value if prediction_result else None,
    )

    return TokenSignal(
        symbol=symbol,
        cmc_id=cmc_id,
        direction=direction,
        confidence=confidence,
        price_usd=price,
        volume_24h=volume_24h,
        volume_change_pct=volume_change_pct,
        rsi_1h=rsi,
        macd_signal=macd_signal,
        ema_trend=ema_trend,
        volatility_pct=volatility_pct,
        news=news_result,
        whale=whale_result,
        intraday=intraday,
        prediction=prediction_result,
        roc_short=roc_short,
        using_intraday=using_intraday,
        sub_scores=scores,
    )


# ---- helpers ----------------------------------------------------------------

def _extract_quote(quotes: dict | list, cmc_id: int) -> dict[str, Any]:
    """Navigate CMC quotes response to the price/volume fields for a given ID.

    Handles three formats from CMC MCP:
    - Columnar: {"headers": [...], "rows": [[val1, val2, ...], ...]}  ← batch
    - List:     [{id, price, percent_change_1h, ...}]                 ← single
    - Legacy:   {"data": {"7186": {"quote": {"USD": {...}}}}}
    """
    try:
        # Columnar batch format: {"headers": [...], "rows": [...]}
        if isinstance(quotes, dict) and "headers" in quotes and "rows" in quotes:
            headers = quotes["headers"]
            id_idx = next((i for i, h in enumerate(headers) if h == "id"), None)
            if id_idx is not None:
                for row in quotes["rows"]:
                    if str(row[id_idx]) == str(cmc_id):
                        item = dict(zip(headers, row))
                        return {
                            "price": float(item.get("price") or 0),
                            "volume_24h": float(item.get("volume_24h") or 0),
                            "percent_change_1h": float(item.get("percent_change_1h") or 0),
                            "percent_change_24h": float(item.get("percent_change_24h") or 0),
                            "percent_change_7d": float(item.get("percent_change_7d") or 0),
                            "volume_change_24h": float(item.get("volume_change_24h") or 0),
                            "market_cap": float(item.get("market_cap") or 0),
                        }
            return {}

        # Flat list format (single-token response)
        if isinstance(quotes, list):
            for item in quotes:
                if str(item.get("id", "")) == str(cmc_id):
                    return {
                        "price": float(item.get("price") or 0),
                        "volume_24h": float(item.get("volume_24h") or 0),
                        "percent_change_1h": float(item.get("percent_change_1h") or 0),
                        "percent_change_24h": float(item.get("percent_change_24h") or 0),
                        "percent_change_7d": float(item.get("percent_change_7d") or 0),
                        "volume_change_24h": float(item.get("volume_change_24h") or 0),
                        "market_cap": float(item.get("market_cap") or 0),
                    }
            return {}

        # Legacy nested format
        data = quotes.get("data", quotes)
        item = data.get(str(cmc_id)) or data.get(cmc_id, {})
        return item.get("quote", {}).get("USD", {})

    except (AttributeError, KeyError, TypeError, ValueError):
        return {}


def _extract_rsi(ta: dict) -> float | None:
    """Extract RSI from CMC TA response.
    CMC returns: {"rsi": {"rsi7": "41.55", "rsi14": "40.43", "rsi21": "41.39"}}
    """
    try:
        # New CMC flat format
        rsi_block = ta.get("rsi")
        if isinstance(rsi_block, dict):
            for key in ("rsi14", "rsi7", "rsi21"):
                v = rsi_block.get(key)
                if v is not None:
                    return float(v)
        # Legacy nested format
        data = ta.get("data", ta)
        for key in ["1h", "hour_1", "1H"]:
            if key in data:
                indicators = data[key].get("indicators", data[key])
                rsi = indicators.get("rsi")
                if rsi is not None:
                    return float(rsi)
        return float(data.get("rsi", data.get("RSI")))
    except (KeyError, TypeError, ValueError):
        return None


def _extract_macd_signal(ta: dict) -> str:
    """Return 'bullish', 'bearish', or 'neutral' from MACD data.
    CMC returns: {"macd": {"macdLine": "-0.06", "signalLine": "-0.04", "histogram": "-0.01"}}
    """
    try:
        # New CMC flat format
        macd_block = ta.get("macd")
        if isinstance(macd_block, dict):
            line = float(macd_block.get("macdLine", 0) or 0)
            signal = float(macd_block.get("signalLine", 0) or 0)
            if line > signal:
                return "bullish"
            elif line < signal:
                return "bearish"
            return "neutral"
        # Legacy nested format
        data = ta.get("data", ta)
        for key in ["1h", "hour_1", "1H"]:
            if key in data:
                indicators = data[key].get("indicators", data[key])
                macd = indicators.get("macd", {})
                if isinstance(macd, dict):
                    line = float(macd.get("macd_line", macd.get("value", 0)))
                    signal = float(macd.get("signal_line", macd.get("signal", 0)))
                    return "bullish" if line > signal else ("bearish" if line < signal else "neutral")
        macd = data.get("macd", {})
        if isinstance(macd, dict):
            line = float(macd.get("macd_line", 0))
            signal = float(macd.get("signal_line", 0))
            return "bullish" if line > signal else ("bearish" if line < signal else "neutral")
    except (KeyError, TypeError, ValueError):
        pass
    return "neutral"


def _extract_ema_trend(ta: dict) -> str:
    """Return 'up', 'down', or 'flat' based on EMA7 vs EMA30.
    CMC returns: {"moving_averages": {"exponential_moving_average_7_day": "1.27", ...}}
    """
    try:
        # New CMC flat format
        ma_block = ta.get("moving_averages")
        if isinstance(ma_block, dict):
            ema7  = float(ma_block.get("exponential_moving_average_7_day", 0) or 0)
            ema30 = float(ma_block.get("exponential_moving_average_30_day", 0) or 0)
            if ema7 > 0 and ema30 > 0:
                diff_pct = (ema7 - ema30) / ema30 * 100
                if diff_pct > 0.5:
                    return "up"
                elif diff_pct < -0.5:
                    return "down"
                return "flat"
        # Legacy nested format
        data = ta.get("data", ta)
        for key in ["1h", "hour_1", "1H"]:
            if key in data:
                indicators = data[key].get("indicators", data[key])
                ema20 = float(indicators.get("ema_20", indicators.get("ema20", 0)))
                ema50 = float(indicators.get("ema_50", indicators.get("ema50", 0)))
                if ema20 > 0 and ema50 > 0:
                    diff_pct = (ema20 - ema50) / ema50 * 100
                    return "up" if diff_pct > 0.5 else ("down" if diff_pct < -0.5 else "flat")
        ema20 = float(data.get("ema_20", data.get("ema20", 0)))
        ema50 = float(data.get("ema_50", data.get("ema50", 0)))
        if ema20 > 0 and ema50 > 0:
            diff_pct = (ema20 - ema50) / ema50 * 100
            return "up" if diff_pct > 0.5 else ("down" if diff_pct < -0.5 else "flat")
    except (KeyError, TypeError, ValueError):
        pass
    return "flat"


def _estimate_volume_surge(quotes: dict | list, cmc_id: int) -> float | None:
    """Estimate volume surge % vs average using volume_change_24h from CMC.
    Returns None when data is unavailable — callers must handle None."""
    try:
        quote = _extract_quote(quotes, cmc_id)
        vol_change = quote.get("volume_change_24h")
        if vol_change is not None:
            return float(vol_change)
        vol_24h = float(quote.get("volume_24h") or 0)
        vol_30d = float(quote.get("volume_30d") or 0)
        if vol_30d > 0:
            avg_daily = vol_30d / 30
            if avg_daily > 0:
                return (vol_24h - avg_daily) / avg_daily * 100
        return None   # no data — signal disabled
    except (KeyError, TypeError, ValueError):
        return None
