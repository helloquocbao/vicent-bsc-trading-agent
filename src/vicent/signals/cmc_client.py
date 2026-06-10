"""CMC MCP HTTP client — wraps the 12 MCP tools as typed Python calls.

The CMC MCP server is REST-based: POST to https://mcp.coinmarketcap.com/mcp
with the JSON-RPC-style body and the API key header.

Retry policy: up to 3 attempts with exponential backoff on transient errors.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import structlog
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from vicent.config import get_settings

log = structlog.get_logger(__name__)

_TIMEOUT = httpx.Timeout(connect=10.0, read=30.0, write=10.0, pool=5.0)

# Errors worth retrying (transient)
_RETRYABLE = (httpx.TimeoutException, httpx.ConnectError, httpx.RemoteProtocolError)


class CMCClient:
    """Async client for the CoinMarketCap MCP endpoint with automatic key rotation."""

    def __init__(self) -> None:
        cfg = get_settings()
        self._base = cfg.cmc_mcp_url
        self._keys = cfg.cmc_keys or [""]
        self._key_idx = 0
        self._exhausted_keys: set[int] = set()   # indices đã hết credit
        self._http: httpx.AsyncClient | None = None

    @property
    def _current_key(self) -> str:
        return self._keys[self._key_idx]

    def _build_headers(self) -> dict[str, str]:
        return {
            "X-CMC-MCP-API-KEY": self._current_key,
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
            "Accept-Encoding": "gzip, deflate, br",
        }

    def _rotate_key(self) -> bool:
        """Switch to next available key. Returns False if all exhausted."""
        self._exhausted_keys.add(self._key_idx)
        for i in range(len(self._keys)):
            if i not in self._exhausted_keys:
                old = self._key_idx
                self._key_idx = i
                log.warning(
                    "cmc_key_rotated",
                    from_idx=old,
                    to_idx=i,
                    exhausted=len(self._exhausted_keys),
                    total=len(self._keys),
                )
                # Update http client headers
                if self._http is not None:
                    self._http.headers.update(self._build_headers())
                return True
        log.error("cmc_all_keys_exhausted", count=len(self._keys))
        return False

    async def __aenter__(self) -> "CMCClient":
        self._http = httpx.AsyncClient(
            headers=self._build_headers(),
            timeout=_TIMEOUT,
            follow_redirects=True,
        )
        return self

    async def __aexit__(self, *_: Any) -> None:
        if self._http:
            await self._http.aclose()

    async def _call(self, tool: str, arguments: dict[str, Any]) -> dict[str, Any]:
        """Call one MCP tool with retry. Returns the parsed JSON result."""
        assert self._http is not None, "Use CMCClient as an async context manager"
        body = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {"name": tool, "arguments": arguments},
        }

        last_error: Exception | None = None
        for attempt in range(3):
            try:
                resp = await self._http.post(self._base, json=body)
                resp.raise_for_status()

                content_type = resp.headers.get("content-type", "")

                # SSE / event-stream response — parse the data: line
                if "text/event-stream" in content_type:
                    data = _parse_sse(resp.text)
                else:
                    data = resp.json()

                # MCP wraps result in data.result.content[0].text (JSON string)
                content = data.get("result", {}).get("content", [])
                if content and content[0].get("type") == "text":
                    return json.loads(content[0]["text"])
                return data.get("result", data)

            except httpx.HTTPStatusError as e:
                status = e.response.status_code
                log.warning("cmc_http_error", tool=tool, status=status, attempt=attempt + 1, key_idx=self._key_idx)

                # 401 = invalid key, 402 = payment required (out of credits), 403 = forbidden
                # → try rotating to next key
                if status in (401, 402, 403):
                    if self._rotate_key():
                        continue   # retry with new key
                    raise   # all keys exhausted

                if status == 429:
                    # Rate-limited — back off, also try rotating
                    import asyncio
                    if len(self._keys) > 1 and self._rotate_key():
                        continue
                    await asyncio.sleep(2 ** (attempt + 2))
                    last_error = e
                    continue
                raise  # Non-retryable HTTP errors propagate immediately

            except _RETRYABLE as e:
                import asyncio
                wait = 2 ** attempt
                log.warning("cmc_transient_error", tool=tool, error=str(e), retry_in=wait)
                await asyncio.sleep(wait)
                last_error = e
                continue

            except json.JSONDecodeError as e:
                log.error("cmc_json_decode_error", tool=tool, error=str(e))
                raise

        log.error("cmc_all_retries_failed", tool=tool)
        raise last_error or RuntimeError(f"CMC tool {tool!r} failed after 3 attempts")

    # ------------------------------------------------------------------ #
    # Public methods — one per CMC MCP tool                               #
    # ------------------------------------------------------------------ #

    async def get_global_metrics(self) -> dict[str, Any]:
        """Fear & Greed, total market cap, BTC dominance, ETF flows."""
        return await self._call("get_global_metrics_latest", {})

    async def get_derivatives_metrics(self) -> dict[str, Any]:
        """Open interest, funding rates, liquidations."""
        return await self._call("get_global_crypto_derivatives_metrics", {})

    async def search_cryptos(self, query: str) -> list[dict[str, Any]]:
        """Search by name/symbol → get CMC numeric IDs."""
        result = await self._call("search_cryptos", {"query": query})
        return result if isinstance(result, list) else result.get("data", [])

    async def get_quotes(self, ids: list[int]) -> dict[str, Any]:
        """Price, market cap, volume, % changes for one or many coins."""
        return await self._call(
            "get_crypto_quotes_latest",
            {"id": ",".join(str(i) for i in ids)},
        )

    async def get_technical_analysis(self, cmc_id: int) -> dict[str, Any]:
        """RSI, MACD, EMA20/50/200, Fibonacci, pivot points."""
        return await self._call("get_crypto_technical_analysis", {"id": cmc_id})

    async def get_crypto_info(self, cmc_id: int) -> dict[str, Any]:
        """Static metadata — description, website, tags."""
        return await self._call("get_crypto_info", {"id": cmc_id})

    async def get_crypto_metrics(self, cmc_id: int) -> dict[str, Any]:
        """Holder distribution, whale vs retail, HODLer breakdown."""
        return await self._call("get_crypto_metrics", {"id": cmc_id})

    async def get_latest_news(self, cmc_id: int) -> list[dict[str, Any]]:
        """Recent news headlines for a coin."""
        result = await self._call("get_crypto_latest_news", {"id": cmc_id})
        return result if isinstance(result, list) else result.get("data", [])

    async def get_marketcap_ta(self) -> dict[str, Any]:
        """TA indicators for total crypto market cap."""
        return await self._call("get_crypto_marketcap_technical_analysis", {})

    async def get_trending_narratives(self) -> list[dict[str, Any]]:
        """Hot narratives with performance and top coins."""
        result = await self._call("trending_crypto_narratives", {})
        return result if isinstance(result, list) else result.get("data", [])

    async def get_upcoming_events(self) -> list[dict[str, Any]]:
        """Fed meetings, regulatory deadlines, major announcements."""
        result = await self._call("get_upcoming_macro_events", {})
        return result if isinstance(result, list) else result.get("data", [])


def _parse_sse(text: str) -> dict[str, Any]:
    """Parse SSE (text/event-stream) response — extract the JSON data line."""
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:"):
            payload = line[len("data:"):].strip()
            if payload and payload != "[DONE]":
                try:
                    return json.loads(payload)
                except json.JSONDecodeError:
                    continue
    # Fallback — try parsing entire text as JSON
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return {}


# Convenience: resolve symbol → CMC ID (cached in-process)
_id_cache: dict[str, int] = {}


async def resolve_cmc_id(client: CMCClient, symbol: str) -> int | None:
    """Resolve a token symbol to its CMC numeric ID. Result is cached."""
    if symbol in _id_cache:
        return _id_cache[symbol]
    try:
        results = await client.search_cryptos(symbol)
    except Exception as e:
        log.warning("resolve_cmc_id_failed", symbol=symbol, error=str(e))
        return None
    for item in results:
        if item.get("symbol", "").upper() == symbol.upper():
            _id_cache[symbol] = int(item["id"])
            return _id_cache[symbol]
    return None
