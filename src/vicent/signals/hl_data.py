"""Hyperliquid market data source — free, no API key required.

Provides micro-layer data to complement CMC's macro-layer signals:
  - Mark prices (exact execution prices on HL)
  - Funding rates (earn income by being on the right side)
  - Open interest (leverage in the market)
  - Recent liquidations (confirm breakout strength)
  - 24h volume per coin (liquidity check before entry)

All data comes from https://api.hyperliquid.xyz/info
No authentication needed. Rate limit: ~1200 req/min (shared).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

import httpx
import structlog

log = structlog.get_logger(__name__)

_BASE = "https://api.hyperliquid.xyz"
_TIMEOUT = httpx.Timeout(15.0)


@dataclass
class HLCoinData:
    """All Hyperliquid data for one coin."""
    coin: str
    mark_price: float           # exact HL mark price
    mid_price: float            # mid between best bid and ask
    funding_rate: float         # current 8h funding rate (negative = longs earn)
    funding_rate_annualized: float  # funding × 3 × 365 for comparison
    open_interest_usd: float    # total OI in USD
    volume_24h_usd: float       # 24h trading volume
    # Derived signals
    funding_bias: str           # "earn_long" | "earn_short" | "neutral"
    funding_score: float        # -1 to +1 (positive = lean long)
    oi_score: float             # 0 to 1 (high OI = crowded, be careful)


@dataclass
class HLMarketState:
    """Snapshot of all Hyperliquid market data."""
    coins: dict[str, HLCoinData] = field(default_factory=dict)
    # Aggregate market stats
    total_oi_usd: float = 0.0
    total_volume_24h: float = 0.0
    avg_funding_rate: float = 0.0
    market_bias: str = "neutral"   # "long_crowded" | "short_crowded" | "neutral"
    # Recent liquidations (last fetch)
    recent_liq_longs: float = 0.0   # USD liquidated longs
    recent_liq_shorts: float = 0.0  # USD liquidated shorts


class HyperliquidDataClient:
    """Async HTTP client for Hyperliquid public API."""

    def __init__(self) -> None:
        self._http: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "HyperliquidDataClient":
        self._http = httpx.AsyncClient(
            base_url=_BASE,
            timeout=_TIMEOUT,
            headers={"Content-Type": "application/json"},
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()

    async def _post(self, payload: dict[str, Any]) -> Any:
        assert self._http is not None
        try:
            resp = await self._http.post("/info", json=payload)
            resp.raise_for_status()
            return resp.json()
        except httpx.TimeoutException:
            log.warning("hl_data_timeout", payload_type=payload.get("type"))
            return None
        except Exception as e:
            log.warning("hl_data_error", error=str(e), payload_type=payload.get("type"))
            return None

    # ------------------------------------------------------------------ #
    # Raw data fetchers                                                   #
    # ------------------------------------------------------------------ #

    async def get_all_mids(self) -> dict[str, float]:
        """Get mid prices for all coins. O(1) single call."""
        data = await self._post({"type": "allMids"})
        if not data:
            return {}
        return {coin: float(price) for coin, price in data.items()}

    async def get_meta_and_asset_ctxs(self) -> tuple[list, list]:
        """Get market metadata + per-asset context (funding, OI, volume).
        Returns (universe, assetCtxs) — one call for all coins.
        """
        data = await self._post({"type": "metaAndAssetCtxs"})
        if not data or not isinstance(data, list) or len(data) < 2:
            return [], []
        return data[0].get("universe", []), data[1]

    async def get_funding_history(self, coin: str, start_time: int, end_time: int | None = None) -> list:
        """Get historical funding rates for a coin."""
        payload: dict[str, Any] = {
            "type": "fundingHistory",
            "coin": coin,
            "startTime": start_time,
        }
        if end_time:
            payload["endTime"] = end_time
        data = await self._post(payload)
        return data or []

    async def get_candles(
        self,
        coin: str,
        interval: str = "5m",
        lookback_bars: int = 300,
    ) -> list[dict[str, Any]]:
        """Lấy OHLCV candles từ Hyperliquid.

        interval: "1m" | "3m" | "5m" | "15m" | "1h" | "4h" | "1d"
        Trả về list candles: [{t, o, h, l, c, v}, ...] oldest→newest
        """
        import time
        # Tính startTime dựa trên lookback
        interval_ms = {
            "1m": 60_000, "3m": 180_000, "5m": 300_000,
            "15m": 900_000, "1h": 3_600_000, "4h": 14_400_000, "1d": 86_400_000,
        }.get(interval, 300_000)
        now_ms = int(time.time() * 1000)
        start_ms = now_ms - interval_ms * (lookback_bars + 5)  # +5 buffer

        data = await self._post({
            "type": "candleSnapshot",
            "req": {
                "coin": coin,
                "interval": interval,
                "startTime": start_ms,
                "endTime": now_ms,
            }
        })
        if not data or not isinstance(data, list):
            return []

        result = []
        for c in data:
            try:
                result.append({
                    "t": int(c.get("t", 0)),           # timestamp ms
                    "o": float(c.get("o", 0) or 0),    # open
                    "h": float(c.get("h", 0) or 0),    # high
                    "l": float(c.get("l", 0) or 0),    # low
                    "c": float(c.get("c", 0) or 0),    # close
                    "v": float(c.get("v", 0) or 0),    # volume
                })
            except (TypeError, ValueError):
                continue
        return result

    async def get_candles_batch(
        self,
        coins: list[str],
        interval: str = "5m",
        lookback_bars: int = 300,
    ) -> dict[str, list[dict[str, Any]]]:
        """Lấy candles cho nhiều coins song song."""
        tasks = [
            asyncio.create_task(self.get_candles(coin, interval, lookback_bars))
            for coin in coins
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out: dict[str, list[dict[str, Any]]] = {}
        for coin, res in zip(coins, results):
            if isinstance(res, list):
                out[coin] = res
        return out

    async def get_candles_multi_tf(
        self,
        coins: list[str],
        intervals: list[tuple[str, int]] | None = None,
    ) -> dict[str, dict[str, list[dict[str, Any]]]]:
        """Lấy candles multi-timeframe cho nhiều coins — tất cả song song.

        intervals: list of (interval_str, lookback_bars)
          Mặc định: 5m/500 bars (41h), 15m/200 bars (50h), 1h/168 bars (1 tuần)

        Trả về: {coin: {interval: [candles]}}
        HL không rate-limit → có thể gọi tất cả đồng thời.
        """
        if intervals is None:
            intervals = [
                ("5m",  500),   # 41h — intraday entry timing
                ("15m", 200),   # 50h — short-term trend confirmation
                ("1h",  168),   # 7d  — medium-term direction
            ]

        # Tạo tất cả tasks cùng lúc — HL xử lý parallel tốt
        tasks: list[asyncio.Task] = []
        task_keys: list[tuple[str, str]] = []   # (coin, interval)
        for coin in coins:
            for interval, bars in intervals:
                tasks.append(
                    asyncio.create_task(self.get_candles(coin, interval, bars))
                )
                task_keys.append((coin, interval))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        out: dict[str, dict[str, list[dict[str, Any]]]] = {c: {} for c in coins}
        for (coin, interval), res in zip(task_keys, results):
            if isinstance(res, list) and res:
                out[coin][interval] = res

        return out

    # ------------------------------------------------------------------ #
    # Derived market state                                                #
    # ------------------------------------------------------------------ #

    async def get_market_state(self, coins: list[str]) -> HLMarketState:
        """Fetch all relevant HL data in 2 concurrent calls and build HLMarketState."""

        # Parallel: mids + metaAndAssetCtxs
        mids_task = asyncio.create_task(self.get_all_mids())
        meta_task = asyncio.create_task(self.get_meta_and_asset_ctxs())

        mids, (universe, asset_ctxs) = await asyncio.gather(mids_task, meta_task)

        state = HLMarketState()
        if not universe or not asset_ctxs:
            log.warning("hl_data_empty_meta")
            return state

        # Build coin name → index map
        coin_idx = {info["name"]: i for i, info in enumerate(universe)}

        for coin in coins:
            if coin not in coin_idx:
                continue
            i = coin_idx[coin]
            if i >= len(asset_ctxs):
                continue

            ctx = asset_ctxs[i]
            mid_price = mids.get(coin, 0.0)
            mark_price = float(ctx.get("markPx", mid_price) or mid_price)
            funding = float(ctx.get("funding", 0.0) or 0.0)
            oi_usd = float(ctx.get("openInterest", 0.0) or 0.0) * mark_price
            vol_24h = float(ctx.get("dayNtlVlm", 0.0) or 0.0)

            # Funding interpretation
            # HL funding: positive = longs pay shorts, negative = shorts pay longs
            funding_annualized = funding * 3 * 365   # 3 times/day × 365
            if funding < -0.0001:
                funding_bias = "earn_long"   # shorts paying longs → LONG is rewarded
                funding_score = min(0.5, abs(funding) * 5000)
            elif funding > 0.0001:
                funding_bias = "earn_short"  # longs paying shorts → SHORT is rewarded
                funding_score = -min(0.5, funding * 5000)
            else:
                funding_bias = "neutral"
                funding_score = 0.0

            # OI score: very high OI relative to volume = crowded market (caution)
            if vol_24h > 0:
                oi_ratio = oi_usd / vol_24h
                oi_score = min(1.0, oi_ratio / 3.0)   # >3x = crowded
            else:
                oi_score = 0.0

            state.coins[coin] = HLCoinData(
                coin=coin,
                mark_price=mark_price,
                mid_price=mid_price,
                funding_rate=funding,
                funding_rate_annualized=funding_annualized,
                open_interest_usd=oi_usd,
                volume_24h_usd=vol_24h,
                funding_bias=funding_bias,
                funding_score=funding_score,
                oi_score=oi_score,
            )
            state.total_oi_usd += oi_usd
            state.total_volume_24h += vol_24h

        # Aggregate funding bias
        if state.coins:
            avg_funding = sum(c.funding_rate for c in state.coins.values()) / len(state.coins)
            state.avg_funding_rate = avg_funding
            if avg_funding > 0.0002:
                state.market_bias = "long_crowded"    # most longs paying
            elif avg_funding < -0.0002:
                state.market_bias = "short_crowded"   # most shorts paying
            else:
                state.market_bias = "neutral"

        log.debug(
            "hl_market_state",
            coins=len(state.coins),
            total_oi=f"${state.total_oi_usd/1e9:.1f}B",
            avg_funding=f"{state.avg_funding_rate:.5f}",
            bias=state.market_bias,
        )
        return state


def compute_hl_signal_boost(coin_data: HLCoinData | None) -> float:
    """Compute a signal adjustment from HL data. Range: -0.3 to +0.3.

    Used to boost or penalise the CMC composite score based on HL-specific data.

    Positive → lean LONG (favorable funding, not crowded)
    Negative → lean SHORT or reduce confidence
    """
    if coin_data is None:
        return 0.0

    boost = 0.0

    # Funding rate contribution (most important for perps)
    # Earning funding by being on right side = free edge
    boost += coin_data.funding_score * 0.6   # max ±0.3

    # OI: if very crowded (oi_score > 0.7), reduce confidence
    if coin_data.oi_score > 0.7:
        boost -= 0.1   # crowded market = reversal risk

    return max(-0.3, min(0.3, boost))
