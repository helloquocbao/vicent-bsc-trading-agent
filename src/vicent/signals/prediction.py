"""Predictive algorithms to improve trade win rate.

Three complementary approaches:

1. CandlePattern  — detects reversal/continuation patterns from price series.
   Each pattern has a historically-validated directional bias and confidence.
   Patterns: Hammer, Shooting Star, Engulfing (bull/bear), Doji,
             Morning Star, Evening Star, Three White Soldiers, Three Black Crows,
             Tweezer Bottom/Top, Harami.

2. DirectionalADX — extracts DI+ and DI- from the ADX calculation to know
   not just whether a trend is strong, but which direction.
   Combines with existing ADX to give a cleaner entry signal.

3. BreakoutScore  — cross-references ROC momentum with Bollinger Band position
   and volume surge to distinguish genuine breakouts from fakeouts.
   A high ROC near the BB lower band + volume surge = high-probability breakout.
   A high ROC near the BB upper band = likely fade (false breakout / exhaustion).

All are pure-Python, computed from the self-collected price series.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

class PatternType(str, Enum):
    HAMMER            = "hammer"
    SHOOTING_STAR     = "shooting_star"
    BULL_ENGULFING    = "bull_engulfing"
    BEAR_ENGULFING    = "bear_engulfing"
    DOJI              = "doji"
    MORNING_STAR      = "morning_star"
    EVENING_STAR      = "evening_star"
    THREE_SOLDIERS    = "three_white_soldiers"
    THREE_CROWS       = "three_black_crows"
    TWEEZER_BOTTOM    = "tweezer_bottom"
    TWEEZER_TOP       = "tweezer_top"
    HARAMI_BULL       = "harami_bull"
    HARAMI_BEAR       = "harami_bear"
    NONE              = "none"


@dataclass
class CandlePattern:
    pattern: PatternType
    direction: str       # "bullish" | "bearish" | "neutral"
    confidence: float    # 0.0–1.0 (historical win rate of pattern)
    description: str


@dataclass
class DirectionalADX:
    adx: float | None     # 0-100 trend strength
    di_plus: float        # +DI: upward directional movement
    di_minus: float       # -DI: downward directional movement
    trend_direction: str  # "up" | "down" | "unclear"
    trend_strength: str   # "strong" | "moderate" | "weak" | "sideways"
    score: float          # -1 to +1 (combines strength + direction)


@dataclass
class BreakoutScore:
    score: float          # -1 to +1
    signal: str           # "breakout_up" | "breakout_down" | "false_breakout" | "neutral"
    confidence: float
    roc_confirmed: bool   # momentum confirms direction
    volume_confirmed: bool
    bb_confirmed: bool    # BB position supports the move


@dataclass
class TrendProjection:
    """Dự đoán hướng giá tương lai bằng linear regression trên giá gần đây.

    slope    — độ dốc đường hồi quy (price units / bar)
    r_squared — độ khớp (0-1). Cao = trend rõ ràng, dự đoán được.
                Thấp = giá choppy/nhiễu, KHÔNG nên tin dự đoán.
    projected_move_pct — % thay đổi dự kiến trong N bar tới
    """
    slope: float
    r_squared: float
    projected_move_pct: float    # % move dự kiến forward
    direction: str               # "up" | "down" | "flat"
    reliable: bool               # True khi r_squared đủ cao
    score: float                 # -1 to +1


@dataclass
class PredictiveSignal:
    """Combined output of all predictive algorithms."""
    pattern: CandlePattern
    directional_adx: DirectionalADX
    breakout: BreakoutScore
    projection: "TrendProjection | None" = None
    combined_score: float = 0.0      # -1 to +1
    combined_confidence: float = 0.0  # 0–1
    entry_quality: str = "skip"       # "A" | "B" | "C" | "skip"


# ---------------------------------------------------------------------------
# 1. Candle Pattern Recognition
# ---------------------------------------------------------------------------
# We simulate OHLCV from close prices only using:
#   open  ≈ previous close
#   close ≈ current close
#   high  ≈ max(open, close) × 1 + small noise proxy from ATR
#   low   ≈ min(open, close) × 1 - small noise proxy from ATR
# This gives approximate candle bodies and shadows.

def _make_candles(prices: list[float]) -> list[dict]:
    """Convert close-price series to approximate OHLC candles."""
    if len(prices) < 2:
        return []
    candles = []
    for i in range(1, len(prices)):
        o = prices[i - 1]   # open = previous close
        c = prices[i]       # close = current close
        body = abs(c - o)
        # Approximate shadow size = 30% of body (we have no real high/low)
        shadow = body * 0.3
        h = max(o, c) + shadow
        l = min(o, c) - shadow
        candles.append({
            "open": o, "high": h, "low": l, "close": c,
            "body": body,
            "upper_shadow": h - max(o, c),
            "lower_shadow": min(o, c) - l,
            "bullish": c > o,
        })
    return candles


def detect_candle_pattern(prices: list[float]) -> CandlePattern:
    """Detect the most recent candlestick pattern from price series.

    Checks the last 3 candles (enough for 3-bar patterns).
    Returns the highest-confidence pattern found, or NONE.
    """
    if len(prices) < 4:
        return CandlePattern(PatternType.NONE, "neutral", 0.0, "Not enough data")

    candles = _make_candles(prices)
    if len(candles) < 3:
        return CandlePattern(PatternType.NONE, "neutral", 0.0, "Not enough candles")

    c1 = candles[-3]   # 3 bars ago
    c2 = candles[-2]   # 2 bars ago
    c3 = candles[-1]   # most recent bar

    # Average body size for relative comparisons
    avg_body = sum(c["body"] for c in candles[-10:]) / min(10, len(candles))
    if avg_body <= 0:
        avg_body = abs(prices[-1] - prices[-2]) + 1e-10

    # ---- 3-bar patterns (highest priority) ----

    # Morning Star (bullish reversal): big bearish, small body, big bullish
    if (not c1["bullish"] and c1["body"] > avg_body * 0.8
            and c2["body"] < avg_body * 0.4
            and c3["bullish"] and c3["body"] > avg_body * 0.8
            and c3["close"] > (c1["open"] + c1["close"]) / 2):
        return CandlePattern(PatternType.MORNING_STAR, "bullish", 0.72,
                             "Morning Star — bullish reversal after downtrend")

    # Evening Star (bearish reversal): big bullish, small body, big bearish
    if (c1["bullish"] and c1["body"] > avg_body * 0.8
            and c2["body"] < avg_body * 0.4
            and not c3["bullish"] and c3["body"] > avg_body * 0.8
            and c3["close"] < (c1["open"] + c1["close"]) / 2):
        return CandlePattern(PatternType.EVENING_STAR, "bearish", 0.72,
                             "Evening Star — bearish reversal after uptrend")

    # Three White Soldiers (strong bull continuation): 3 consecutive bullish with increasing closes
    if (c1["bullish"] and c2["bullish"] and c3["bullish"]
            and c2["close"] > c1["close"] and c3["close"] > c2["close"]
            and c1["body"] > avg_body * 0.5
            and c2["body"] > avg_body * 0.5
            and c3["body"] > avg_body * 0.5):
        return CandlePattern(PatternType.THREE_SOLDIERS, "bullish", 0.68,
                             "Three White Soldiers — strong bullish continuation")

    # Three Black Crows (strong bear continuation): 3 consecutive bearish with decreasing closes
    if (not c1["bullish"] and not c2["bullish"] and not c3["bullish"]
            and c2["close"] < c1["close"] and c3["close"] < c2["close"]
            and c1["body"] > avg_body * 0.5
            and c2["body"] > avg_body * 0.5
            and c3["body"] > avg_body * 0.5):
        return CandlePattern(PatternType.THREE_CROWS, "bearish", 0.68,
                             "Three Black Crows — strong bearish continuation")

    # ---- 2-bar patterns ----

    # Bullish Engulfing: bearish followed by larger bullish that engulfs it
    if (not c2["bullish"] and c3["bullish"]
            and c3["open"] < c2["close"]
            and c3["close"] > c2["open"]
            and c3["body"] > c2["body"] * 1.2):
        return CandlePattern(PatternType.BULL_ENGULFING, "bullish", 0.65,
                             "Bullish Engulfing — buyer momentum takes over")

    # Bearish Engulfing
    if (c2["bullish"] and not c3["bullish"]
            and c3["open"] > c2["close"]
            and c3["close"] < c2["open"]
            and c3["body"] > c2["body"] * 1.2):
        return CandlePattern(PatternType.BEAR_ENGULFING, "bearish", 0.65,
                             "Bearish Engulfing — seller momentum takes over")

    # Bullish Harami: large bearish, small bullish inside previous body
    if (not c2["bullish"] and c3["bullish"]
            and c3["open"] > c2["close"]
            and c3["close"] < c2["open"]
            and c3["body"] < c2["body"] * 0.5):
        return CandlePattern(PatternType.HARAMI_BULL, "bullish", 0.55,
                             "Bullish Harami — momentum slowing, possible reversal")

    # Bearish Harami
    if (c2["bullish"] and not c3["bullish"]
            and c3["open"] < c2["close"]
            and c3["close"] > c2["open"]
            and c3["body"] < c2["body"] * 0.5):
        return CandlePattern(PatternType.HARAMI_BEAR, "bearish", 0.55,
                             "Bearish Harami — momentum slowing, possible reversal")

    # Tweezer Bottom: two candles with similar lows (support test)
    low_diff = abs(c2["low"] - c3["low"]) / max(c2["low"], 1e-10)
    if low_diff < 0.002 and not c2["bullish"] and c3["bullish"]:
        return CandlePattern(PatternType.TWEEZER_BOTTOM, "bullish", 0.60,
                             "Tweezer Bottom — double-tested support, bullish reversal")

    # Tweezer Top: two candles with similar highs (resistance test)
    high_diff = abs(c2["high"] - c3["high"]) / max(c2["high"], 1e-10)
    if high_diff < 0.002 and c2["bullish"] and not c3["bullish"]:
        return CandlePattern(PatternType.TWEEZER_TOP, "bearish", 0.60,
                             "Tweezer Top — double-tested resistance, bearish reversal")

    # ---- 1-bar patterns ----

    # Hammer: small body at top, long lower shadow (bullish reversal at bottom)
    lower_shadow_ratio = c3["lower_shadow"] / max(c3["body"], 1e-10)
    upper_shadow_ratio = c3["upper_shadow"] / max(c3["body"], 1e-10)
    if lower_shadow_ratio > 2.0 and upper_shadow_ratio < 0.5 and c3["body"] > 0:
        direction = "bullish" if c3["bullish"] else "bullish"  # hammer = bullish regardless
        return CandlePattern(PatternType.HAMMER, direction, 0.62,
                             f"Hammer — rejection of lower prices, potential reversal up")

    # Shooting Star: small body at bottom, long upper shadow (bearish reversal at top)
    if upper_shadow_ratio > 2.0 and lower_shadow_ratio < 0.5 and c3["body"] > 0:
        return CandlePattern(PatternType.SHOOTING_STAR, "bearish", 0.62,
                             "Shooting Star — rejection of higher prices, potential reversal down")

    # Doji: very small body vs shadows
    if c3["body"] < avg_body * 0.1:
        return CandlePattern(PatternType.DOJI, "neutral", 0.40,
                             "Doji — indecision, wait for next bar confirmation")

    return CandlePattern(PatternType.NONE, "neutral", 0.0, "No pattern detected")


# ---------------------------------------------------------------------------
# 2. Directional ADX (DI+ and DI-)
# ---------------------------------------------------------------------------

def compute_directional_adx(values: list[float], period: int = 14) -> DirectionalADX:
    """Compute ADX with directional components DI+ and DI-.

    DI+ > DI- → uptrend. DI- > DI+ → downtrend.
    The gap between DI+ and DI- indicates conviction.

    Returns combined score:
      score = (DI+ - DI-) / (DI+ + DI- + ε) × (ADX / 50)
      Range: approximately -1 to +1
      - Positive = bullish trend
      - Negative = bearish trend
      - Near 0 = unclear or sideways
    """
    if len(values) < period * 2 + 1:
        return DirectionalADX(None, 0.0, 0.0, "unclear", "sideways", 0.0)

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

    atr_s = smooth(tr_vals, period)
    dmp_s = smooth(dm_plus, period)
    dmn_s = smooth(dm_minus, period)
    if not atr_s or not dmp_s or not dmn_s:
        return DirectionalADX(None, 0.0, 0.0, "unclear", "sideways", 0.0)

    n = min(len(atr_s), len(dmp_s), len(dmn_s))
    di_plus_series  = [100 * dmp_s[i] / atr_s[i] if atr_s[i] > 0 else 0 for i in range(n)]
    di_minus_series = [100 * dmn_s[i] / atr_s[i] if atr_s[i] > 0 else 0 for i in range(n)]

    dx_vals = []
    for i in range(n):
        denom = di_plus_series[i] + di_minus_series[i]
        dx_vals.append(100 * abs(di_plus_series[i] - di_minus_series[i]) / denom if denom > 0 else 0)

    if len(dx_vals) < period:
        return DirectionalADX(None, 0.0, 0.0, "unclear", "sideways", 0.0)

    adx_val = sum(dx_vals[-period:]) / period
    di_plus  = di_plus_series[-1]
    di_minus = di_minus_series[-1]

    # Direction
    if abs(di_plus - di_minus) < 5:
        direction = "unclear"
    elif di_plus > di_minus:
        direction = "up"
    else:
        direction = "down"

    # Strength
    if adx_val > 40:
        strength = "strong"
    elif adx_val > 25:
        strength = "moderate"
    elif adx_val > 15:
        strength = "weak"
    else:
        strength = "sideways"

    # Combined directional score
    di_diff = di_plus - di_minus
    di_sum  = di_plus + di_minus + 1e-10
    adx_factor = min(1.0, adx_val / 50.0)   # 0 at ADX=0, 1 at ADX=50+
    score = (di_diff / di_sum) * adx_factor  # [-1, +1]

    return DirectionalADX(
        adx=round(adx_val, 1),
        di_plus=round(di_plus, 1),
        di_minus=round(di_minus, 1),
        trend_direction=direction,
        trend_strength=strength,
        score=round(max(-1.0, min(1.0, score)), 3),
    )


# ---------------------------------------------------------------------------
# 3. Breakout Confirmation
# ---------------------------------------------------------------------------

def compute_breakout_score(
    prices: list[float],
    volume_change_pct: float = 0.0,
    bb_position: float | None = None,
    bb_squeeze: bool = False,
    roc_short: float = 0.0,
    roc_medium: float = 0.0,
) -> BreakoutScore:
    """Classify the current move as genuine breakout, false breakout, or noise.

    Logic:
      Genuine breakout UP:
        - Price near or above BB upper (bb_position > 0.8)
        - ROC_short > threshold (momentum confirms)
        - Volume surge (volume_change_pct > 30%)
        - OR: just escaped BB squeeze (bb_squeeze was True, now moving)

      Genuine breakout DOWN:
        - Price near or below BB lower (bb_position < 0.2)
        - ROC_short < -threshold
        - Volume surge

      False breakout (fade signal):
        - High ROC but price near BB extreme AND medium ROC diverges
        - Price spiked quickly into resistance/support with no volume

    Returns score [-1, +1]:
      +1 = confirmed breakout UP (enter LONG)
      -1 = confirmed breakout DOWN (enter SHORT / exit LONG)
       0 = noise / unclear
    """
    if bb_position is None or len(prices) < 4:
        return BreakoutScore(0.0, "neutral", 0.0, False, False, False)

    roc_confirmed    = abs(roc_short) > 0.5
    volume_confirmed = volume_change_pct > 30
    upward_move      = roc_short > 0
    downward_move    = roc_short < 0

    # --- BB position confirmation ---
    # near upper band = potential upward breakout or overbought
    # near lower band = potential downward breakout or oversold
    bb_supports_up   = bb_position > 0.65
    bb_supports_down = bb_position < 0.35

    # Squeeze breakout: high potential, direction determined by ROC
    if bb_squeeze and abs(roc_short) > 0.3:
        direction = "breakout_up" if roc_short > 0 else "breakout_down"
        score = 0.7 if roc_short > 0 else -0.7
        return BreakoutScore(
            score=score, signal=direction,
            confidence=0.65,
            roc_confirmed=True,
            volume_confirmed=volume_confirmed,
            bb_confirmed=True,
        )

    # Confirmed upward breakout
    if upward_move and roc_confirmed and bb_supports_up:
        # Check for false breakout: medium ROC much weaker than short ROC
        # (spike without follow-through)
        if roc_medium < 0 and roc_short > 1.5:
            return BreakoutScore(
                score=0.2, signal="false_breakout",
                confidence=0.55,
                roc_confirmed=True,
                volume_confirmed=volume_confirmed,
                bb_confirmed=bb_supports_up,
            )
        score = 0.6
        if volume_confirmed: score += 0.2
        if roc_short > 1.5:  score += 0.1
        return BreakoutScore(
            score=min(1.0, score), signal="breakout_up",
            confidence=min(0.85, 0.55 + (0.1 if volume_confirmed else 0) + 0.05),
            roc_confirmed=True,
            volume_confirmed=volume_confirmed,
            bb_confirmed=bb_supports_up,
        )

    # Confirmed downward breakout
    if downward_move and roc_confirmed and bb_supports_down:
        if roc_medium > 0 and roc_short < -1.5:
            return BreakoutScore(
                score=-0.2, signal="false_breakout",
                confidence=0.55,
                roc_confirmed=True,
                volume_confirmed=volume_confirmed,
                bb_confirmed=bb_supports_down,
            )
        score = -0.6
        if volume_confirmed: score -= 0.2
        if roc_short < -1.5: score -= 0.1
        return BreakoutScore(
            score=max(-1.0, score), signal="breakout_down",
            confidence=min(0.85, 0.55 + (0.1 if volume_confirmed else 0)),
            roc_confirmed=True,
            volume_confirmed=volume_confirmed,
            bb_confirmed=bb_supports_down,
        )

    # Overbought without volume = potential fade (reversal back down)
    if bb_position > 0.85 and not volume_confirmed and roc_short > 0:
        return BreakoutScore(
            score=-0.3, signal="false_breakout",
            confidence=0.50,
            roc_confirmed=False,
            volume_confirmed=False,
            bb_confirmed=True,
        )

    return BreakoutScore(0.0, "neutral", 0.0, roc_confirmed, volume_confirmed, False)


# ---------------------------------------------------------------------------
# 4. Trend Projection — linear regression forward forecast
# ---------------------------------------------------------------------------

def compute_trend_projection(
    prices: list[float],
    lookback: int = 20,
    project_bars: int = 3,
) -> TrendProjection:
    """Dự đoán hướng giá forward bằng least-squares linear regression.

    Ý tưởng: fit đường thẳng qua `lookback` giá gần nhất.
      - slope cho biết hướng + tốc độ
      - R² cho biết giá có ĐI THEO đường thẳng không (trend rõ) hay
        nhiễu loạn (choppy). R² thấp → dự đoán KHÔNG đáng tin → giảm score.

    Đây là filter quan trọng: chỉ vào lệnh khi xu hướng đủ "sạch" để dự đoán.
    Thị trường sideways/choppy có R² thấp → tránh được nhiều lệnh xấu.
    """
    if len(prices) < lookback:
        lookback = len(prices)
    if lookback < 5:
        return TrendProjection(0.0, 0.0, 0.0, "flat", False, 0.0)

    window = prices[-lookback:]
    n = len(window)
    xs = list(range(n))
    mean_x = sum(xs) / n
    mean_y = sum(window) / n

    # Least squares slope + intercept
    cov = sum((xs[i] - mean_x) * (window[i] - mean_y) for i in range(n))
    var_x = sum((xs[i] - mean_x) ** 2 for i in range(n))
    if var_x == 0:
        return TrendProjection(0.0, 0.0, 0.0, "flat", False, 0.0)
    slope = cov / var_x
    intercept = mean_y - slope * mean_x

    # R² (coefficient of determination)
    ss_tot = sum((window[i] - mean_y) ** 2 for i in range(n))
    ss_res = sum((window[i] - (slope * xs[i] + intercept)) ** 2 for i in range(n))
    r_squared = 1.0 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))

    # Forward projection
    current = window[-1]
    projected_price = slope * (n - 1 + project_bars) + intercept
    projected_move_pct = (projected_price - current) / current * 100 if current > 0 else 0.0

    # Direction
    if projected_move_pct > 0.3:
        direction = "up"
    elif projected_move_pct < -0.3:
        direction = "down"
    else:
        direction = "flat"

    # Reliable khi R² đủ cao (trend rõ ràng, không nhiễu)
    reliable = r_squared >= 0.55

    # Score: hướng × độ tin (R²) × magnitude (capped)
    raw_dir = 1.0 if projected_move_pct > 0 else (-1.0 if projected_move_pct < 0 else 0.0)
    magnitude = min(1.0, abs(projected_move_pct) / 3.0)   # 3% move → full magnitude
    score = raw_dir * r_squared * magnitude
    score = max(-1.0, min(1.0, score))

    return TrendProjection(
        slope=round(slope, 6),
        r_squared=round(r_squared, 3),
        projected_move_pct=round(projected_move_pct, 3),
        direction=direction,
        reliable=reliable,
        score=round(score, 3),
    )


# ---------------------------------------------------------------------------
# Master: combine all predictive signals
# ---------------------------------------------------------------------------

def compute_predictive_signal(
    prices: list[float],
    volume_change_pct: float = 0.0,
    bb_position: float | None = None,
    bb_squeeze: bool = False,
    roc_short: float = 0.0,
    roc_medium: float = 0.0,
    ohlcv_candles: list[dict] | None = None,   # real HL OHLCV {o,h,l,c,v} — preferred over close-only
) -> PredictiveSignal:
    """Compute combined predictive signal from all three algorithms."""
    # Use real OHLCV candles for pattern detection if available (much more accurate)
    if ohlcv_candles and len(ohlcv_candles) >= 4:
        pattern = _detect_pattern_from_ohlcv(ohlcv_candles)
    else:
        pattern = detect_candle_pattern(prices)

    dir_adx = compute_directional_adx(prices)
    breakout = compute_breakout_score(
        prices, volume_change_pct if volume_change_pct is not None else 0.0,
        bb_position, bb_squeeze, roc_short, roc_medium
    )
    projection = compute_trend_projection(prices)

    pattern_score = 0.0
    if pattern.direction == "bullish":
        pattern_score = pattern.confidence
    elif pattern.direction == "bearish":
        pattern_score = -pattern.confidence

    # Combine 4 signals. Trend projection được cân nặng vì nó là forward-looking
    # và đã tự điều chỉnh theo R² (độ tin cậy).
    combined = (
        dir_adx.score      * 0.32 +
        breakout.score     * 0.28 +
        pattern_score      * 0.20 +
        projection.score   * 0.20
    )
    combined = max(-1.0, min(1.0, combined))

    # --- Coherence filter: các tín hiệu có ĐỒNG THUẬN về hướng không? ---
    # Nếu projection (forward) MÂU THUẪN mạnh với composite → giảm độ tin.
    # Đây là cách tránh vào lệnh khi tín hiệu hiện tại và xu hướng dự báo đánh nhau.
    if projection.reliable and abs(projection.score) > 0.15:
        proj_dir = 1 if projection.score > 0 else -1
        comp_dir = 1 if combined > 0 else -1
        if proj_dir != comp_dir:
            combined *= 0.5   # phạt nặng khi forward projection ngược hướng

    confidences = [c for c in [
        abs(dir_adx.score),
        breakout.confidence if breakout.signal != "neutral" else 0.0,
        pattern.confidence  if pattern.pattern != PatternType.NONE else 0.0,
        projection.r_squared if projection.reliable else 0.0,
    ] if c > 0]
    combined_confidence = sum(confidences) / max(len(confidences), 1) if confidences else 0.0

    if combined_confidence >= 0.70 and abs(combined) >= 0.40:
        quality = "A"
    elif combined_confidence >= 0.55 and abs(combined) >= 0.25:
        quality = "B"
    elif abs(combined) >= 0.15:
        quality = "C"
    else:
        quality = "skip"

    return PredictiveSignal(
        pattern=pattern,
        directional_adx=dir_adx,
        breakout=breakout,
        projection=projection,
        combined_score=round(combined, 3),
        combined_confidence=round(combined_confidence, 3),
        entry_quality=quality,
    )


def _detect_pattern_from_ohlcv(candles: list[dict]) -> CandlePattern:
    """Pattern detection using real OHLCV candles from Hyperliquid.

    Each candle: {o: open, h: high, l: low, c: close, v: volume}
    Far more accurate than close-only approximation.
    """
    if len(candles) < 3:
        return CandlePattern(PatternType.NONE, "neutral", 0.0, "Not enough candles")

    def _candle(d: dict) -> dict:
        o = float(d.get("o", d.get("open", 0)))
        h = float(d.get("h", d.get("high", o)))
        l = float(d.get("l", d.get("low", o)))
        c = float(d.get("c", d.get("close", 0)))
        body = abs(c - o)
        return {
            "open": o, "high": h, "low": l, "close": c,
            "body": body,
            "upper_shadow": h - max(o, c),
            "lower_shadow": min(o, c) - l,
            "bullish": c >= o,
        }

    recent = [_candle(c) for c in candles[-10:]]
    c1, c2, c3 = recent[-3], recent[-2], recent[-1]
    avg_body = sum(c["body"] for c in recent) / len(recent) or 1e-10

    # Morning Star
    if (not c1["bullish"] and c1["body"] > avg_body * 0.8
            and c2["body"] < avg_body * 0.4
            and c3["bullish"] and c3["body"] > avg_body * 0.8
            and c3["close"] > (c1["open"] + c1["close"]) / 2):
        return CandlePattern(PatternType.MORNING_STAR, "bullish", 0.72, "Morning Star")

    # Evening Star
    if (c1["bullish"] and c1["body"] > avg_body * 0.8
            and c2["body"] < avg_body * 0.4
            and not c3["bullish"] and c3["body"] > avg_body * 0.8
            and c3["close"] < (c1["open"] + c1["close"]) / 2):
        return CandlePattern(PatternType.EVENING_STAR, "bearish", 0.72, "Evening Star")

    # Pre-compute shadows of last candle for pattern checks
    upper_sh = c3["upper_shadow"]
    lower_sh = c3["lower_shadow"]
    body = c3["body"]

    # Hammer (bullish reversal): long lower shadow, tiny upper shadow
    if body > 0 and lower_sh >= body * 2 and upper_sh <= body * 0.3:
        return CandlePattern(PatternType.HAMMER, "bullish", 0.60,
                             "Hammer — potential reversal up")

    # Shooting Star (bearish reversal): long upper shadow, tiny lower shadow
    if body > 0 and upper_sh >= body * 2 and lower_sh <= body * 0.3:
        return CandlePattern(PatternType.SHOOTING_STAR, "bearish", 0.60,
                             "Shooting Star — potential reversal down")

    # Bullish Engulfing
    if (not c2["bullish"] and c3["bullish"]
            and c3["open"] < c2["close"]
            and c3["close"] > c2["open"]
            and c3["body"] > c2["body"] * 1.2):
        return CandlePattern(PatternType.BULL_ENGULFING, "bullish", 0.65, "Bullish Engulfing")

    # Bearish Engulfing
    if (c2["bullish"] and not c3["bullish"]
            and c3["open"] > c2["close"]
            and c3["close"] < c2["open"]
            and c3["body"] > c2["body"] * 1.2):
        return CandlePattern(PatternType.BEAR_ENGULFING, "bearish", 0.65, "Bearish Engulfing")

    # Doji
    if c3["body"] < avg_body * 0.1 and avg_body > 0:
        return CandlePattern(PatternType.DOJI, "neutral", 0.40, "Doji — indecision")

    return CandlePattern(PatternType.NONE, "neutral", 0.0, "No pattern")
