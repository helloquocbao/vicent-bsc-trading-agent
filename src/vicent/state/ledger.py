"""SQLite trade ledger — persists all trades for PnL calculation and audit."""

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


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(_get_db_path(), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def init_db() -> None:
    """Create tables if they don't exist."""
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS trades (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                symbol      TEXT    NOT NULL,
                direction   TEXT    NOT NULL,   -- 'buy' | 'sell'
                amount_usd  REAL    NOT NULL,
                price_usd   REAL    NOT NULL,
                quantity    REAL    NOT NULL,
                tx_hash     TEXT,
                slippage    REAL,
                fee_usd     REAL,
                mode        TEXT    NOT NULL,   -- 'paper' | 'live'
                confidence  REAL,
                regime      TEXT,
                status      TEXT    DEFAULT 'ok'
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                ts          TEXT    NOT NULL,
                nav_usd     REAL    NOT NULL,
                peak_nav    REAL    NOT NULL,
                drawdown    REAL    NOT NULL,
                positions   TEXT                -- JSON blob
            );

            CREATE TABLE IF NOT EXISTS daily_summary (
                date        TEXT    PRIMARY KEY,
                start_nav   REAL,
                end_nav     REAL,
                pnl_usd     REAL,
                pnl_pct     REAL,
                trade_count INTEGER
            );

            CREATE TABLE IF NOT EXISTS iteration_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                ts              TEXT    NOT NULL,
                iteration       INTEGER NOT NULL,
                regime          TEXT,
                fear_greed      INTEGER,
                nav_usd         REAL,
                total_return_pct REAL,
                drawdown_pct    REAL,
                tradeable_count INTEGER,
                top_symbol      TEXT,
                top_confidence  REAL,
                action          TEXT,   -- 'traded' | 'skipped' | 'blocked'
                action_symbol   TEXT,
                action_reason   TEXT,
                intraday_ready  INTEGER,  -- 0/1
                calls_used      INTEGER
            );

            CREATE TABLE IF NOT EXISTS paper_perps (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol      TEXT    NOT NULL,
                direction   TEXT    NOT NULL,
                collateral_usd REAL NOT NULL,
                size_usd    REAL    NOT NULL,
                size_coin   REAL    DEFAULT 0,
                leverage    REAL    NOT NULL,
                entry_price REAL    NOT NULL,
                liq_price   REAL    NOT NULL,
                peak_pnl_pct REAL   DEFAULT 0,
                open        INTEGER DEFAULT 1,
                ts_open     TEXT    NOT NULL
            );
        """)
        # Migration: add size_coin column if missing (for existing databases)
        try:
            conn.execute("ALTER TABLE paper_perps ADD COLUMN size_coin REAL DEFAULT 0")
            conn.commit()
        except Exception:
            pass  # column already exists
    log.info("ledger_initialized", path=str(_get_db_path()))


def record_trade(
    symbol: str,
    direction: str,
    amount_usd: float,
    price_usd: float,
    quantity: float,
    mode: str,
    tx_hash: str | None = None,
    slippage: float | None = None,
    fee_usd: float | None = None,
    confidence: float | None = None,
    regime: str | None = None,
) -> int:
    """Insert a trade record and return its row id."""
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        cur = conn.execute(
            """
            INSERT INTO trades
              (ts, symbol, direction, amount_usd, price_usd, quantity,
               tx_hash, slippage, fee_usd, mode, confidence, regime)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (ts, symbol, direction, amount_usd, price_usd, quantity,
             tx_hash, slippage, fee_usd, mode, confidence, regime),
        )
        trade_id = cur.lastrowid
    log.info("trade_recorded", id=trade_id, symbol=symbol, direction=direction, usd=amount_usd)
    return trade_id  # type: ignore[return-value]


def record_snapshot(nav_usd: float, peak_nav: float, drawdown: float, positions: str) -> None:
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO snapshots (ts, nav_usd, peak_nav, drawdown, positions) VALUES (?,?,?,?,?)",
            (ts, nav_usd, peak_nav, drawdown, positions),
        )


def get_trades_today() -> list[dict[str, Any]]:
    today = datetime.now(timezone.utc).date().isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trades WHERE ts LIKE ? ORDER BY ts DESC",
            (f"{today}%",),
        ).fetchall()
    return [dict(r) for r in rows]


def get_all_trades(limit: int = 200) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM trades ORDER BY ts DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_latest_snapshot() -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT * FROM snapshots ORDER BY ts DESC LIMIT 1"
        ).fetchone()
    return dict(row) if row else None


def record_iteration_log(
    iteration: int,
    regime: str,
    fear_greed: int,
    nav_usd: float,
    total_return_pct: float,
    drawdown_pct: float,
    tradeable_count: int,
    top_symbol: str,
    top_confidence: float,
    action: str,
    action_symbol: str,
    action_reason: str,
    intraday_ready: bool,
    calls_used: int,
) -> None:
    """Record one iteration summary — trade or no-trade with reason."""
    ts = datetime.now(timezone.utc).isoformat()
    with _connect() as conn:
        conn.execute(
            """
            INSERT INTO iteration_log
              (ts, iteration, regime, fear_greed, nav_usd, total_return_pct,
               drawdown_pct, tradeable_count, top_symbol, top_confidence,
               action, action_symbol, action_reason, intraday_ready, calls_used)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (ts, iteration, regime, fear_greed, nav_usd, total_return_pct,
             drawdown_pct, tradeable_count, top_symbol, top_confidence,
             action, action_symbol, action_reason, int(intraday_ready), calls_used),
        )


def get_iteration_logs(limit: int = 50) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM iteration_log ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]
