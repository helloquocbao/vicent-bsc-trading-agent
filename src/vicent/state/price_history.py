"""Price history store — stores historical OHLCV data from CMC quotes.

Allows historical calculations (ATR, VWAP, BB) to remain accurate over a rolling window.

Stored in the same SQLite DB as the trade ledger.
Old data is pruned automatically to keep the rolling window bounded.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

def _get_db_path() -> Path:
    from vicent.config import get_settings
    return Path(get_settings().vicent_db_path)

_MAX_BARS_PER_SYMBOL = 300   # 300 bars × 5m = ~25h history


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_price_history() -> None:
    """Create tables if they don't exist. Migration-safe."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS price_history (
                id      INTEGER PRIMARY KEY AUTOINCREMENT,
                ts      TEXT    NOT NULL,
                symbol  TEXT    NOT NULL,
                price   REAL    NOT NULL,
                high    REAL    DEFAULT 0,
                low     REAL    DEFAULT 0,
                volume  REAL    DEFAULT 0,
                open    REAL    DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_price_symbol_ts
                ON price_history (symbol, id);
        """)
        # Migration: add OHLCV columns if database only contains close price
        for col, default in [("high", "0"), ("low", "0"), ("volume", "0"), ("open", "0")]:
            try:
                conn.execute(f"ALTER TABLE price_history ADD COLUMN {col} REAL DEFAULT {default}")
                conn.commit()
            except Exception:
                pass  # column already exists
    log.info("price_history_initialized")


def record_candles(candles_by_symbol: dict[str, list[dict]]) -> None:
    """Record OHLCV candles to history.

    candles_by_symbol: {symbol: [{t, o, h, l, c, v}, ...]}
    Only record the latest completed bar.
    Use timestamps to avoid duplicate entries.
    """
    if not candles_by_symbol:
        return

    ts_now = datetime.now(timezone.utc).isoformat()
    rows = []
    for symbol, candles in candles_by_symbol.items():
        if not candles:
            continue
        # Lấy bar gần nhất đã đóng (bar cuối thường chưa đóng → lấy bar -2)
        bar = candles[-2] if len(candles) >= 2 else candles[-1]
        close = bar.get("c", 0.0)
        if close <= 0:
            continue
        rows.append((
            ts_now,
            symbol,
            close,
            bar.get("h", close),
            bar.get("l", close),
            bar.get("v", 0.0),
            bar.get("o", close),
        ))

    if not rows:
        return
    try:
        with _connect() as conn:
            conn.executemany(
                """INSERT INTO price_history (ts, symbol, price, high, low, volume, open)
                   VALUES (?,?,?,?,?,?,?)""",
                rows,
            )
        _prune([r[1] for r in rows])
        log.debug("candles_recorded", count=len(rows))
    except Exception as e:
        log.error("record_candles_failed", error=str(e))


def record_prices(prices: dict[str, float]) -> None:
    """Fallback: write close price only.
    Kept for backward compatibility.
    """
    if not prices:
        return
    ts = datetime.now(timezone.utc).isoformat()
    rows = [
        (ts, sym, price, price, price, 0.0, price)
        for sym, price in prices.items() if price > 0
    ]
    if not rows:
        return
    try:
        with _connect() as conn:
            conn.executemany(
                """INSERT INTO price_history (ts, symbol, price, high, low, volume, open)
                   VALUES (?,?,?,?,?,?,?)""",
                rows,
            )
        _prune(prices.keys())
    except Exception as e:
        log.error("record_prices_failed", error=str(e))


def get_price_series(symbol: str, limit: int = 300) -> list[float]:
    """Return close prices oldest to newest (backward compatibility for indicators.py)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT price FROM price_history WHERE symbol=? ORDER BY id DESC LIMIT ?",
            (symbol, limit),
        ).fetchall()
    return [float(r["price"]) for r in reversed(rows)]


def get_ohlcv_series(symbol: str, limit: int = 300) -> dict[str, list[float]]:
    """Return full OHLCV series oldest to newest.

    Return: {"open": [...], "high": [...], "low": [...], "close": [...], "volume": [...]}
    """
    with _connect() as conn:
        rows = conn.execute(
            """SELECT open, high, low, price as close, volume
               FROM price_history WHERE symbol=? ORDER BY id DESC LIMIT ?""",
            (symbol, limit),
        ).fetchall()
    rows = list(reversed(rows))
    return {
        "open":   [float(r["open"])   for r in rows],
        "high":   [float(r["high"])   for r in rows],
        "low":    [float(r["low"])    for r in rows],
        "close":  [float(r["close"])  for r in rows],
        "volume": [float(r["volume"]) for r in rows],
    }


def get_bar_count(symbol: str) -> int:
    with _connect() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS c FROM price_history WHERE symbol=?", (symbol,)
        ).fetchone()
    return int(row["c"]) if row else 0


def _prune(symbols: Any) -> None:
    with _connect() as conn:
        for sym in symbols:
            conn.execute(
                """DELETE FROM price_history WHERE symbol=? AND id NOT IN (
                    SELECT id FROM price_history WHERE symbol=? ORDER BY id DESC LIMIT ?
                )""",
                (sym, sym, _MAX_BARS_PER_SYMBOL),
            )
