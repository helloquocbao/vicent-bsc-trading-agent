"""Intraday technical indicators computed from self-collected price history.

The core problem this solves:
  CMC MCP returns DAILY RSI/MACD/EMA. The agent loops every 5 minutes.
  Using daily indicators on a 5-minute loop means signals barely change
  between iterations — the agent is "blind" to intraday moves.

Solution:
  Store a price snapshot every loop into SQLite. Once we have enough points,
  compute REAL intraday indicators from our own rolling window.

Implemented (pure-Python, no pandas in hot path):
  - EMA (exponential moving average)
  - RSI (Wilder's smoothing)
  - MACD (12/26/9)
  - ROC (rate of change — short-term momentum trigger)
  - Bollinger Bands (20-period, 2σ) — breakout detection
  - ATR (Average True Range) — real intraday volatility for position sizing
  - ADX (Average Directional Index) — trend strength, filters out sideways noise
  - VWAP (Volume-Weighted Average Price) — key intraday support/resistance
  - Stochastic RSI — faster overbought/oversold detection
"""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class IntradayIndicators:
    """Computed intraday indicators for one token."""
    rsi: float | None              # 0-100
    stoch_rsi: float | None        # 0-100 — Stochastic RSI (faster signal)
    macd_signal: str               # "bullish" | "bearish" | "neutral"
    macd_histogram: float
    ema_trend: str                 # "up" | "down" | "flat"
    roc_short: float               # % change over ~15 min
    roc_medium: float              # % change over ~1 hour
    # Bollinger Bands
    bb_upper: float | None         # upper band
    bb_lower: float | None         # lower band
    bb_mid: float | None           # 20-period SMA (mid band)
    bb_position: float | None      # 0=lower band, 1=upper band, 0.5=mid
    bb_squeeze: bool               # True when bands are very tight (breakout imminent)
    # ATR (real intraday)
    atr: float | None              # Average True Range in price units
    atr_pct: float                 # ATR as % of current price
    # ADX (trend strength)
    adx: float | None              # 0-100. >25 = trending, <20 = sideways
    trend_strong: bool             # True when ADX > 25
    # VWAP
    vwap: float | None             # Volume-weighted avg price
    vwap_position: str             # "above" | "below" | "near"
    # Multi-timeframe (15m, 1h) — None nếu không có data
    tf_15m_rsi: float | None = None       # RSI trên 15m
    tf_15m_macd: str = "neutral"          # MACD trên 15m
    tf_15m_trend: str = "flat"            # EMA trend 15m
    tf_1h_rsi: float | None = None        # RSI trên 1h
    tf_1h_macd: str = "neutral"           # MACD trên 1h
    tf_1h_trend: str = "flat"             # EMA trend 1h
    tf_1h_adx: float | None = None        # ADX 1h — medium-term trend strength
    tf_alignment: str = "none"            # "bullish" | "bearish" | "mixed" | "none"
    # Meta
    n_bars: int = 0
    has_enough_data: bool = False


_MIN_BARS_RSI = 15
_MIN_BARS_MACD = 26
_MIN_BARS_BB = 20
_MIN_BARS_ADX = 28       # needs 14+14 periods
_MIN_BARS_RELIABLE = 26


# ---------------------------------------------------------------------------
# EMA / SMA primitives
# ---------------------------------------------------------------------------

def ema(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    k = 2 / (period + 1)
    sma = sum(values[:period]) / period
    v = sma
    for price in values[period:]:
        v = price * k + v * (1 - k)
    return v


def ema_series(values: list[float], period: int) -> list[float]:
    if len(values) < period:
        return []
    k = 2 / (period + 1)
    sma = sum(values[:period]) / period
    out = [sma]
    for price in values[period:]:
        out.append(price * k + out[-1] * (1 - k))
    return out


def sma(values: list[float], period: int) -> float | None:
    if len(values) < period:
        return None
    return sum(values[-period:]) / period


def sma_series(values: list[float], period: int) -> list[float]:
    return [
        sum(values[i - period:i]) / period
        for i in range(period, len(values) + 1)
    ]


# ---------------------------------------------------------------------------
# RSI (Wilder's smoothing)
# ---------------------------------------------------------------------------

def rsi(values: list[float], period: int = 14) -> float | None:
    if len(values) < period + 1:
        return None
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = values[i] - values[i - 1]
        if d >= 0:
            gains += d
        else:
            losses -= d
    avg_gain = gains / period
    avg_loss = losses / period
    for i in range(period + 1, len(values)):
        d = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
    if avg_loss == 0:
        # avg_gain > 0 → pure uptrend → 100
        # avg_gain = 0 → completely flat/stale → 50 (neutral, not overbought)
        return 100.0 if avg_gain > 0 else 50.0
    return 100.0 - (100.0 / (1 + avg_gain / avg_loss))


def rsi_series(values: list[float], period: int = 14) -> list[float]:
    """Return full RSI series — needed for Stochastic RSI."""
    if len(values) < period + 1:
        return []
    out = []
    gains = losses = 0.0
    for i in range(1, period + 1):
        d = values[i] - values[i - 1]
        if d >= 0: gains += d
        else: losses -= d
    avg_gain = gains / period
    avg_loss = losses / period
    if avg_loss == 0:
        out.append(100.0 if avg_gain > 0 else 50.0)
    else:
        out.append(100.0 - 100.0 / (1 + avg_gain / avg_loss))
    for i in range(period + 1, len(values)):
        d = values[i] - values[i - 1]
        avg_gain = (avg_gain * (period - 1) + max(d, 0)) / period
        avg_loss = (avg_loss * (period - 1) + max(-d, 0)) / period
        if avg_loss == 0:
            out.append(100.0 if avg_gain > 0 else 50.0)
        else:
            out.append(100.0 - 100.0 / (1 + avg_gain / avg_loss))
    return out


# ---------------------------------------------------------------------------
# Stochastic RSI  — %K of RSI values
# ---------------------------------------------------------------------------

def stochastic_rsi(values: list[float], rsi_period: int = 14, stoch_period: int = 14) -> float | None:
    """Stochastic RSI = (RSI - min_RSI) / (max_RSI - min_RSI) × 100."""
    rsi_vals = rsi_series(values, rsi_period)
    if len(rsi_vals) < stoch_period:
        return None
    window = rsi_vals[-stoch_period:]
    lo, hi = min(window), max(window)
    if hi == lo:
        return 50.0
    return (rsi_vals[-1] - lo) / (hi - lo) * 100


# ---------------------------------------------------------------------------
# MACD
# ---------------------------------------------------------------------------

def macd(values: list[float], fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[str, float]:
    if len(values) < slow + signal:
        return "neutral", 0.0
    ef = ema_series(values, fast)
    es = ema_series(values, slow)
    if not ef or not es:
        return "neutral", 0.0
    offset = len(ef) - len(es)
    macd_line = [ef[i + offset] - es[i] for i in range(len(es))]
    if len(macd_line) < signal:
        return "neutral", 0.0
    sig_line = ema_series(macd_line, signal)
    if not sig_line:
        return "neutral", 0.0
    mv, sv = macd_line[-1], sig_line[-1]
    hist = mv - sv
    # Direction is determined by crossover (MACD line vs Signal line),
    # NOT by whether MACD line is above/below zero.
    # A rising MACD crossing above signal = bullish regardless of absolute value.
    if hist > 0:
        return "bullish", hist
    if hist < 0:
        return "bearish", hist
    return "neutral", hist


# ---------------------------------------------------------------------------
# ROC
# ---------------------------------------------------------------------------

def roc(values: list[float], periods: int) -> float:
    if len(values) <= periods:
        return 0.0
    old = values[-periods - 1]
    if old <= 0:
        return 0.0
    return (values[-1] - old) / old * 100


# ---------------------------------------------------------------------------
# EMA trend
# ---------------------------------------------------------------------------

def ema_trend(values: list[float], short: int = 9, long: int = 21) -> str:
    es = ema(values, short)
    el = ema(values, long)
    if es is None or el is None or el == 0:
        return "flat"
    diff = (es - el) / el * 100
    if diff > 0.15: return "up"
    if diff < -0.15: return "down"
    return "flat"


# ---------------------------------------------------------------------------
# Bollinger Bands (20-period, 2σ)
# ---------------------------------------------------------------------------

def bollinger_bands(
    values: list[float], period: int = 20, n_std: float = 2.0
) -> tuple[float | None, float | None, float | None, float | None, bool]:
    """Return (upper, mid, lower, position 0-1, squeeze).

    position = (price - lower) / (upper - lower):
      0.0 = at lower band (oversold)
      1.0 = at upper band (overbought)
      0.5 = at mid (neutral)

    squeeze = bands are within 1% of mid (breakout imminent).
    """
    if len(values) < period:
        return None, None, None, None, False
    window = values[-period:]
    mid = sum(window) / period
    variance = sum((p - mid) ** 2 for p in window) / period
    std = math.sqrt(variance)
    upper = mid + n_std * std
    lower = mid - n_std * std
    price = values[-1]
    band_width = upper - lower
    if band_width <= 0:
        return upper, mid, lower, 0.5, True
    position = (price - lower) / band_width
    squeeze = (band_width / mid) < 0.01
    return upper, mid, lower, position, squeeze


# ---------------------------------------------------------------------------
# ATR (Average True Range) — needs high/low; we approximate with close prices
# ---------------------------------------------------------------------------

def atr_from_closes(values: list[float], period: int = 14) -> float | None:
    """ATR approximated from close prices only (no high/low available).

    True Range approximation: |close[i] - close[i-1]|
    This underestimates ATR vs full OHLCV but is proportional and usable.
    """
    if len(values) < period + 1:
        return None
    tr_values = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]
    # EMA of true ranges
    k = 2 / (period + 1)
    atr_val = sum(tr_values[:period]) / period  # seed with SMA
    for tr in tr_values[period:]:
        atr_val = tr * k + atr_val * (1 - k)
    return atr_val


def atr_from_ohlc(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    period: int = 14,
) -> float | None:
    """ATR chính xác từ OHLC thật — True Range = max(H-L, |H-Cprev|, |L-Cprev|)."""
    if len(closes) < period + 1:
        return None
    tr_values = []
    for i in range(1, len(closes)):
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_values.append(max(hl, hc, lc))
    k = 2 / (period + 1)
    atr_val = sum(tr_values[:period]) / period
    for tr in tr_values[period:]:
        atr_val = tr * k + atr_val * (1 - k)
    return atr_val


# ---------------------------------------------------------------------------
# ADX (Average Directional Index) — trend strength 0-100
# ---------------------------------------------------------------------------

def adx(values: list[float], period: int = 14) -> float | None:
    """ADX from close-price series only.

    Approximation: uses absolute ROC as directional movement proxy.
    ADX > 25 = trending, ADX > 50 = strong trend, < 20 = sideways.
    """
    if len(values) < period * 2 + 1:
        return None

    # Compute +DM, -DM from closes
    dm_plus  = [max(values[i] - values[i - 1], 0) for i in range(1, len(values))]
    dm_minus = [max(values[i - 1] - values[i], 0) for i in range(1, len(values))]
    tr_vals  = [abs(values[i] - values[i - 1]) for i in range(1, len(values))]

    def smooth(arr: list[float], p: int) -> list[float]:
        if len(arr) < p:
            return []
        k = 1 / p
        val = sum(arr[:p])
        out = [val]
        for x in arr[p:]:
            val = val - val / p + x
            out.append(val)
        return out

    atr_s  = smooth(tr_vals,  period)
    dmp_s  = smooth(dm_plus,  period)
    dmn_s  = smooth(dm_minus, period)
    if not atr_s:
        return None

    n = min(len(atr_s), len(dmp_s), len(dmn_s))
    di_plus  = [100 * dmp_s[i] / atr_s[i] if atr_s[i] > 0 else 0 for i in range(n)]
    di_minus = [100 * dmn_s[i] / atr_s[i] if atr_s[i] > 0 else 0 for i in range(n)]

    dx_vals = []
    for i in range(n):
        denom = di_plus[i] + di_minus[i]
        if denom == 0:
            dx_vals.append(0.0)
        else:
            dx_vals.append(100 * abs(di_plus[i] - di_minus[i]) / denom)

    if len(dx_vals) < period:
        return None
    adx_val = sum(dx_vals[-period:]) / period
    return adx_val


def adx_from_ohlc(
    closes: list[float],
    highs: list[float],
    lows: list[float],
    period: int = 14,
) -> float | None:
    """ADX chính xác từ OHLC thật."""
    if len(closes) < period * 2 + 1:
        return None

    dm_plus, dm_minus, tr_vals = [], [], []
    for i in range(1, len(closes)):
        up   = highs[i] - highs[i - 1]
        down = lows[i - 1] - lows[i]
        dm_plus.append(up if up > down and up > 0 else 0.0)
        dm_minus.append(down if down > up and down > 0 else 0.0)
        hl = highs[i] - lows[i]
        hc = abs(highs[i] - closes[i - 1])
        lc = abs(lows[i] - closes[i - 1])
        tr_vals.append(max(hl, hc, lc))

    def smooth(arr: list[float], p: int) -> list[float]:
        if len(arr) < p:
            return []
        val = sum(arr[:p])
        out = [val]
        for x in arr[p:]:
            val = val - val / p + x
            out.append(val)
        return out

    atr_s = smooth(tr_vals,  period)
    dmp_s = smooth(dm_plus,  period)
    dmn_s = smooth(dm_minus, period)
    if not atr_s:
        return None

    n = min(len(atr_s), len(dmp_s), len(dmn_s))
    dx_vals = []
    for i in range(n):
        if atr_s[i] == 0:
            dx_vals.append(0.0)
            continue
        di_p = 100 * dmp_s[i] / atr_s[i]
        di_n = 100 * dmn_s[i] / atr_s[i]
        denom = di_p + di_n
        dx_vals.append(100 * abs(di_p - di_n) / denom if denom > 0 else 0.0)

    if len(dx_vals) < period:
        return None
    return sum(dx_vals[-period:]) / period


# ---------------------------------------------------------------------------
# VWAP (Volume-Weighted Average Price)
# ---------------------------------------------------------------------------

def vwap_from_prices(
    prices: list[float],
    volumes: list[float] | None = None,
) -> float | None:
    """VWAP approximation.

    Without volume data, falls back to simple arithmetic mean (less accurate
    but directionally correct for support/resistance estimation).
    """
    if not prices:
        return None
    if volumes and len(volumes) == len(prices):
        total_vol = sum(volumes)
        if total_vol <= 0:
            return sum(prices) / len(prices)
        return sum(p * v for p, v in zip(prices, volumes)) / total_vol
    # No volume: use SMA as VWAP approximation
    return sum(prices) / len(prices)


# ---------------------------------------------------------------------------
# Master compute function
# ---------------------------------------------------------------------------

def compute_intraday(
    prices: list[float],
    volumes: list[float] | None = None,
    highs: list[float] | None = None,
    lows: list[float] | None = None,
    candles_15m: list[dict] | None = None,   # HL candles 15m cho multi-tf
    candles_1h: list[dict] | None = None,    # HL candles 1h cho multi-tf
) -> IntradayIndicators:
    """Compute all intraday indicators từ OHLCV series.

    prices = close prices (required, từ 5m candles)
    volumes, highs, lows = optional — từ HL candles cho ATR/VWAP chính xác hơn
    candles_15m, candles_1h = optional — cho multi-timeframe alignment
    """
    n = len(prices)
    has_enough = n >= _MIN_BARS_RELIABLE

    # ---- Core TA (5m) ----
    rsi_val    = rsi(prices, 14) if n >= _MIN_BARS_RSI else None
    stoch_rsi_val = stochastic_rsi(prices) if n >= 29 else None
    macd_lbl, macd_hist = macd(prices) if n >= _MIN_BARS_MACD else ("neutral", 0.0)
    trend      = ema_trend(prices) if n >= 21 else "flat"
    roc_short  = roc(prices, 3)  if n >= 4  else 0.0
    roc_medium = roc(prices, 12) if n >= 13 else 0.0

    # ---- Bollinger Bands ----
    bb_u, bb_m, bb_l, bb_pos, bb_sq = bollinger_bands(prices) if n >= _MIN_BARS_BB else (None, None, None, None, False)

    # ---- ATR — dùng high/low thật nếu có, fallback close-only ----
    if highs and lows and len(highs) == n and len(lows) == n:
        atr_val = atr_from_ohlc(prices, highs, lows)
    else:
        atr_val = atr_from_closes(prices) if n >= 15 else None
    atr_pct = (atr_val / prices[-1] * 100) if (atr_val and prices[-1] > 0) else 0.0

    # ---- ADX ----
    if highs and lows and len(highs) == n and len(lows) == n:
        adx_val = adx_from_ohlc(prices, highs, lows)
    else:
        adx_val = adx(prices) if n >= _MIN_BARS_ADX else None
    trend_strong = (adx_val is not None and adx_val > 25)

    # ---- VWAP — dùng volume thật nếu có ----
    vwap_val = vwap_from_prices(prices, volumes)
    vwap_pos = "near"
    if vwap_val and prices:
        diff_pct = (prices[-1] - vwap_val) / vwap_val * 100
        if diff_pct > 0.5:
            vwap_pos = "above"
        elif diff_pct < -0.5:
            vwap_pos = "below"

    # ---- Multi-timeframe (15m) ----
    tf_15m_rsi: float | None = None
    tf_15m_macd = "neutral"
    tf_15m_trend = "flat"
    if candles_15m and len(candles_15m) >= 26:
        c15 = [float(c.get("c", c.get("close", 0))) for c in candles_15m]
        tf_15m_rsi = rsi(c15, 14) if len(c15) >= 15 else None
        tf_15m_macd, _ = macd(c15) if len(c15) >= 35 else ("neutral", 0.0)
        tf_15m_trend = ema_trend(c15) if len(c15) >= 21 else "flat"

    # ---- Multi-timeframe (1h) ----
    tf_1h_rsi: float | None = None
    tf_1h_macd = "neutral"
    tf_1h_trend = "flat"
    tf_1h_adx: float | None = None
    if candles_1h and len(candles_1h) >= 26:
        c1h = [float(c.get("c", c.get("close", 0))) for c in candles_1h]
        h1h = [float(c.get("h", c.get("high", 0))) for c in candles_1h]
        l1h = [float(c.get("l", c.get("low", 0))) for c in candles_1h]
        tf_1h_rsi = rsi(c1h, 14) if len(c1h) >= 15 else None
        tf_1h_macd, _ = macd(c1h) if len(c1h) >= 35 else ("neutral", 0.0)
        tf_1h_trend = ema_trend(c1h) if len(c1h) >= 21 else "flat"
        tf_1h_adx = adx_from_ohlc(c1h, h1h, l1h) if len(c1h) >= _MIN_BARS_ADX else None

    # ---- Timeframe alignment score ----
    # Chỉ vào lệnh khi nhiều timeframe đồng thuận — giảm false signals
    tf_signals = []
    if trend != "flat":       tf_signals.append(1 if trend == "up" else -1)
    if tf_15m_trend != "flat": tf_signals.append(1 if tf_15m_trend == "up" else -1)
    if tf_1h_trend != "flat":  tf_signals.append(1 if tf_1h_trend == "up" else -1)
    if tf_15m_macd != "neutral": tf_signals.append(1 if tf_15m_macd == "bullish" else -1)
    if tf_1h_macd != "neutral":  tf_signals.append(1 if tf_1h_macd == "bullish" else -1)

    if len(tf_signals) >= 3:
        alignment_sum = sum(tf_signals)
        if alignment_sum >= 2:
            tf_alignment = "bullish"
        elif alignment_sum <= -2:
            tf_alignment = "bearish"
        else:
            tf_alignment = "mixed"
    else:
        tf_alignment = "none"

    return IntradayIndicators(
        rsi=rsi_val,
        stoch_rsi=stoch_rsi_val,
        macd_signal=macd_lbl,
        macd_histogram=macd_hist,
        ema_trend=trend,
        roc_short=roc_short,
        roc_medium=roc_medium,
        bb_upper=bb_u,
        bb_lower=bb_l,
        bb_mid=bb_m,
        bb_position=bb_pos,
        bb_squeeze=bb_sq,
        atr=atr_val,
        atr_pct=atr_pct,
        adx=adx_val,
        trend_strong=trend_strong,
        vwap=vwap_val,
        vwap_position=vwap_pos,
        tf_15m_rsi=tf_15m_rsi,
        tf_15m_macd=tf_15m_macd,
        tf_15m_trend=tf_15m_trend,
        tf_1h_rsi=tf_1h_rsi,
        tf_1h_macd=tf_1h_macd,
        tf_1h_trend=tf_1h_trend,
        tf_1h_adx=tf_1h_adx,
        tf_alignment=tf_alignment,
        n_bars=n,
        has_enough_data=has_enough,
    )
