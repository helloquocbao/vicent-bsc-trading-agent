"""VICENT monitoring server — real-time dashboard + heartbeat.

Endpoints:
  GET /          → HTML dashboard (auto-refresh every 10s)
  GET /health    → JSON health check (for monitoring tools)
  GET /heartbeat → JSON agent heartbeat (last_seen, iteration, status)
  GET /portfolio → JSON portfolio snapshot
  GET /trades    → JSON recent trades
  GET /config    → JSON agent config

Heartbeat system:
  - Agent writes timestamp each iteration to the vicent_heartbeat.json file
  - Dashboard reads that file and displays: ALIVE / STALE / DEAD
  - ALIVE  = last_seen < 2× interval
  - STALE  = last_seen 2-5× interval (might be fetching data slowly)
  - DEAD   = last_seen > 5× interval or file does not exist
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse

from vicent.config import get_settings
from vicent.state.ledger import get_all_trades, get_iteration_logs, get_latest_snapshot, get_trades_today

_HEARTBEAT_FILE = Path("vicent_heartbeat.json")


def write_heartbeat(iteration: int, interval_sec: int, status: str = "running",
                    defense_level: str = "", defense_reason: str = "") -> None:
    """Called by agent every iteration to record it's alive. Atomic write."""
    import os
    data = {
        "last_seen": datetime.now(timezone.utc).isoformat(),
        "iteration": iteration,
        "interval_sec": interval_sec,
        "status": status,
        "defense_level": defense_level,
        "defense_reason": defense_reason,
    }
    tmp = _HEARTBEAT_FILE.with_suffix(".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, _HEARTBEAT_FILE)


def read_heartbeat() -> dict[str, Any]:
    """Read heartbeat file and compute alive/stale/dead status."""
    if not _HEARTBEAT_FILE.exists():
        return {"status": "dead", "reason": "no_heartbeat_file", "iteration": 0}

    try:
        data = json.loads(_HEARTBEAT_FILE.read_text())
        last_seen_str = data.get("last_seen", "")
        interval = int(data.get("interval_sec", 300))
        iteration = int(data.get("iteration", 0))

        last_seen = datetime.fromisoformat(last_seen_str)
        if last_seen.tzinfo is None:
            last_seen = last_seen.replace(tzinfo=timezone.utc)

        age_sec = (datetime.now(timezone.utc) - last_seen).total_seconds()

        if age_sec < interval * 2:
            alive = "alive"
        elif age_sec < interval * 5:
            alive = "stale"
        else:
            alive = "dead"

        return {
            "status": alive,
            "last_seen": last_seen_str,
            "age_seconds": round(age_sec),
            "iteration": iteration,
            "interval_sec": interval,
            "defense_level": data.get("defense_level", ""),
            "defense_reason": data.get("defense_reason", ""),
        }
    except Exception as e:
        return {"status": "dead", "reason": str(e), "iteration": 0}


@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    yield


app = FastAPI(
    title="VICENT Agent Monitor",
    description="Real-time monitoring dashboard for VICENT BSC trading agent",
    version="0.2.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

def _to_vn_time(utc_str: str) -> str:
    """Convert UTC ISO string to Vietnam time (UTC+7) for display."""
    if not utc_str or utc_str == "N/A":
        return utc_str
    try:
        from datetime import datetime, timezone, timedelta
        VN = timezone(timedelta(hours=7))
        dt = datetime.fromisoformat(utc_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(VN).strftime("%d/%m %H:%M")
    except Exception:
        return utc_str[:16]


def _fmt_usd(v: float, decimals: int = 2) -> str:
    """Format USD — automatically choose decimal places appropriate for magnitude."""
    if v == 0:
        return "$0.00"
    abs_v = abs(v)
    if abs_v >= 1_000_000:
        return f"${v/1_000_000:,.2f}M"
    if abs_v >= 1_000:
        return f"${v:,.2f}"
    if abs_v >= 100:
        return f"${v:.2f}"
    if abs_v >= 1:
        return f"${v:.4f}"
    return f"${v:.6f}"


def _fmt_price(v: float) -> str:
    """Format token price — display sufficient significant digits."""
    if v == 0:
        return "$0"
    abs_v = abs(v)
    if abs_v >= 10_000:
        return f"${v:,.0f}"
    if abs_v >= 1_000:
        return f"${v:,.1f}"
    if abs_v >= 100:
        return f"${v:,.2f}"
    if abs_v >= 1:
        return f"${v:.4f}"
    if abs_v >= 0.01:
        return f"${v:.5f}"
    return f"${v:.8f}"


def _fmt_pct(v: float, sign: bool = True) -> str:
    """Format percentage."""
    if sign:
        return f"{v:+.2f}%"
    return f"{v:.2f}%"


def _fmt_conf(v: float) -> str:
    """Format confidence 0-1 → xx%."""
    if not v:
        return "—"
    return f"{v*100:.0f}%"


@app.get("/", response_class=HTMLResponse)
async def dashboard() -> HTMLResponse:
    cfg = get_settings()
    snapshot = get_latest_snapshot()
    hb = read_heartbeat()
    trades_today = get_trades_today()
    recent_trades = get_all_trades(limit=10)
    iter_logs = get_iteration_logs(limit=30)

    nav = snapshot["nav_usd"] if snapshot else 0.0
    initial = 0.0
    drawdown = 0.0
    total_return = 0.0
    positions_html = "<tr><td colspan='4' style='text-align:center;color:#888'>No positions open</td></tr>"

    if snapshot:
        try:
            pos_data = json.loads(snapshot.get("positions", "{}"))
            initial = pos_data.get("initial_capital", nav)
            drawdown = pos_data.get("drawdown", 0.0) * 100
            total_return = pos_data.get("total_return_pct", 0.0) * 100
            positions = pos_data.get("positions", {})
            if positions:
                rows = []
                for sym, p in positions.items():
                    pnl_color = "#00c853" if p.get("pnl_pct", 0) >= 0 else "#ff1744"
                    rows.append(f"""
                    <tr>
                        <td><b>{sym}</b></td>
                        <td>${p.get('current_price', 0):.4f}</td>
                        <td>${p.get('value_usd', 0):.2f}</td>
                        <td style="color:{pnl_color}">{p.get('pnl_pct', 0)*100:+.2f}%</td>
                    </tr>""")
                positions_html = "".join(rows)
        except Exception:
            pass

    # Heartbeat color
    hb_status = hb.get("status", "dead")
    hb_color = {"alive": "#00c853", "stale": "#ffd600", "dead": "#ff1744"}.get(hb_status, "#ff1744")
    hb_text = {"alive": "🟢 RUNNING", "stale": "🟡 STALE / LAG", "dead": "🔴 NO RESPONSE"}.get(hb_status, "🔴 NO RESPONSE")
    last_seen = _to_vn_time(hb.get("last_seen", "N/A"))
    age = hb.get("age_seconds", "?")
    iteration = hb.get("iteration", 0)

    # Defense posture badge
    defense_level = hb.get("defense_level", "") or "normal"
    defense_reason = hb.get("defense_reason", "") or "market stable"
    _defense_styles = {
        "normal":    ("🛡️ NORMAL", "#00c853"),
        "caution":   ("⚠️ CAUTION", "#ffd600"),
        "defensive": ("🟠 DEFENSIVE", "#ff6d00"),
        "halt":      ("🔴 HALT ORDER ENTRY", "#ff1744"),
    }
    defense_text, defense_color = _defense_styles.get(defense_level, _defense_styles["normal"])

    # Trades rows
    trades_rows = ""
    for t in recent_trades:
        sym = t.get("symbol", "")
        direction = t.get("direction", "")
        regime = t.get("regime", "") or ""
        # Determine order type based on direction
        dir_label = "▲ BUY" if direction == "buy" else "▼ SELL"
        dir_color = "#00c853" if direction == "buy" else "#ff1744"
        trades_rows += f"""
        <tr>
            <td>{_to_vn_time(t.get('ts',''))}</td>
            <td><b>{sym}</b></td>
            <td style="color:{dir_color};font-weight:600">{dir_label}</td>
            <td>{_fmt_usd(t.get('amount_usd', 0))}</td>
            <td>{_fmt_price(t.get('price_usd', 0))}</td>
            <td>{_fmt_conf(t.get('confidence', 0))}</td>
            <td style="font-size:11px;color:#aaa">{(t.get('tx_hash','') or '')[:10] or 'paper'}</td>
        </tr>"""

    # Open positions — read from trades ledger + calculate unrealized PnL from live prices
    perps_positions_html = ""
    total_unrealized_pnl = 0.0
    total_unrealized_pct = 0.0
    try:
        # Fetch live prices from price_history (latest bar)
        from vicent.state.price_history import get_ohlcv_series
        live_prices: dict[str, float] = {}
        try:
            from vicent.strategy.tokens import get_watchlist
            for tok in get_watchlist(max_tier=1):
                ohlcv = get_ohlcv_series(tok.symbol, limit=1)
                if ohlcv["close"]:
                    live_prices[tok.symbol] = ohlcv["close"][-1]
        except Exception:
            pass

        all_trades = get_all_trades(limit=200)
        perp_buys: dict[str, dict] = {}
        for t in reversed(all_trades):  # oldest first
            sym = t.get("symbol", "")
            if t.get("direction") == "buy":
                perp_buys[sym] = t
            elif t.get("direction") == "sell" and sym in perp_buys:
                del perp_buys[sym]

        if perp_buys:
            rows = []
            total_collateral = 0.0
            for sym, t in perp_buys.items():
                collateral = float(t.get("amount_usd", 0))
                entry_price = float(t.get("price_usd", 0))
                confidence = float(t.get("confidence") or 0)
                total_collateral += collateral
                dir_label = "▲ BUY"
                dir_color = "#00c853"

                # Stop loss distance or liquidation proxy using standard target calculations
                # In spot swaps, we simulate a stop loss at 5% below entry price.
                liq = entry_price * 0.95

                # Unrealized PnL — use live price if available
                current_price = live_prices.get(sym, entry_price)
                if entry_price > 0 and current_price > 0:
                    price_move = (current_price - entry_price) / entry_price
                    upnl_pct = price_move
                    upnl_usd = collateral * upnl_pct
                else:
                    upnl_pct = 0.0
                    upnl_usd = 0.0

                total_unrealized_pnl += upnl_usd
                upnl_color = "#00c853" if upnl_pct >= 0 else "#ff1744"
                upnl_icon  = "▲" if upnl_pct > 0 else ("▼" if upnl_pct < 0 else "—")

                # Distance to stop loss (%)
                if entry_price > 0:
                    liq_dist = abs(current_price - liq) / current_price * 100
                    liq_warn = "color:#ff1744;font-weight:700" if liq_dist < 2 else "color:#ffd600"
                else:
                    liq_dist = 100.0
                    liq_warn = "color:#ffd600"

                rows.append(f"""
                <tr>
                    <td><b>{sym}</b></td>
                    <td style="color:{dir_color}">{dir_label}</td>
                    <td>
                        {_fmt_usd(collateral)} cost
                    </td>
                    <td>{_fmt_price(entry_price)}</td>
                    <td>{_fmt_price(current_price)}</td>
                    <td style="color:{upnl_color};font-weight:700">
                        {upnl_icon} {_fmt_usd(upnl_usd)}<br>
                        <small>{_fmt_pct(upnl_pct * 100)}</small>
                    </td>
                    <td style="{liq_warn}">{_fmt_price(liq)}<br><small>({liq_dist:.1f}% dist)</small></td>
                    <td style="color:#8b949e">{_fmt_conf(confidence)}</td>
                    <td>{_to_vn_time(t.get('ts',''))}</td>
                </tr>""")

            perps_positions_html = "".join(rows)
            deployed_usd = total_collateral
            deployed_pct = (deployed_usd / nav * 100) if nav > 0 else 0
            reserve_pct = max(0, 100 - deployed_pct)
            total_unrealized_pct = (total_unrealized_pnl / nav * 100) if nav > 0 else 0
        else:
            perps_positions_html = "<tr><td colspan='9' style='text-align:center;color:#888'>No open positions</td></tr>"
            deployed_usd = 0.0
            deployed_pct = 0.0
            reserve_pct = 100.0
        
    except Exception as e:
        perps_positions_html = f"<tr><td colspan='9' style='color:#ff1744'>Error: {e}</td></tr>"
        deployed_usd = 0.0
        deployed_pct = 0.0
        reserve_pct = 100.0
        total_unrealized_pnl = 0.0
        total_unrealized_pct = 0.0

    # Equity = NAV (already includes unrealized PnL from latest snapshots)
    equity = nav
    equity_color = "#00c853" if total_unrealized_pnl >= 0 else "#ff1744"

    # Total unrealized PnL for table header
    total_upnl_color = "#00c853" if total_unrealized_pnl >= 0 else "#ff1744"
    total_upnl_icon  = "▲" if total_unrealized_pnl > 0 else ("▼" if total_unrealized_pnl < 0 else "—")

    # ── Recent Closed Trades ─────────────────────────────────────────
    closed_rows = ""
    total_realized_pnl = 0.0
    try:
        all_trades_full = get_all_trades(limit=500)
        open_map: dict[str, dict] = {}
        closed_list: list[dict] = []

        # Taker fee estimate: 0.10% each side
        BSC_TAKER_FEE = 0.0010

        for t in reversed(all_trades_full):  # oldest first
            sym = t.get("symbol", "")
            if t.get("direction") == "buy":
                open_map[sym] = t
            elif t.get("direction") == "sell" and sym in open_map:
                buy_t  = open_map.pop(sym)
                sell_t = t
                entry  = float(buy_t.get("price_usd", 0))
                exit_p = float(sell_t.get("price_usd", 0))
                collat = float(buy_t.get("amount_usd", 0))

                # PnL gross (before fee)
                if entry > 0 and collat > 0:
                    move = (exit_p - entry) / entry
                    pnl_gross_pct = move
                    pnl_gross_usd = collat * pnl_gross_pct
                else:
                    pnl_gross_pct = 0.0
                    pnl_gross_usd = float(sell_t.get("amount_usd", 0)) - collat

                # Realized fee: prefer ledger values
                fee_open  = float(buy_t.get("fee_usd") or 0)
                fee_close = float(sell_t.get("fee_usd") or 0)
                if fee_open == 0 and fee_close == 0:
                    fee_open  = collat * BSC_TAKER_FEE
                    fee_close = collat * BSC_TAKER_FEE
                total_fee = fee_open + fee_close

                # PnL net (after fees)
                pnl_net_usd = pnl_gross_usd - total_fee
                pnl_net_pct = pnl_net_usd / collat if collat > 0 else 0.0

                closed_list.append({
                    "sym": sym,
                    "entry": entry,
                    "exit": exit_p,
                    "collat": collat,
                    "pnl_gross_usd": pnl_gross_usd,
                    "pnl_gross_pct": pnl_gross_pct,
                    "fee_usd": total_fee,
                    "pnl_net_usd": pnl_net_usd,
                    "pnl_net_pct": pnl_net_pct,
                    "conf": float(buy_t.get("confidence") or 0),
                    "ts_open":  buy_t.get("ts", ""),
                    "ts_close": sell_t.get("ts", ""),
                    "reason": sell_t.get("regime", "") or "—",
                })

        # Sort newest-first, slice last 15
        closed_list.sort(key=lambda x: x["ts_close"], reverse=True)
        closed_list = closed_list[:15]

        for c in closed_list:
            total_realized_pnl += c["pnl_net_usd"]
            net_pos  = c["pnl_net_usd"] >= 0
            pnl_color = "#00c853" if net_pos else "#ff1744"
            pnl_icon  = "✅" if net_pos else "❌"
            dir_label = "▲ BUY"
            dir_color = "#00c853"

            # Hold duration
            try:
                from datetime import datetime
                t1 = datetime.fromisoformat(c["ts_open"].replace("Z","+00:00"))
                t2 = datetime.fromisoformat(c["ts_close"].replace("Z","+00:00"))
                hold_min = int((t2 - t1).total_seconds() / 60)
                hold_str = f"{hold_min}m" if hold_min < 60 else f"{hold_min//60}h{hold_min%60}m"
            except Exception:
                hold_str = "—"

            # Map exit reasons to English descriptions
            _reason_map = {
                "hl_close_signal_reversed": "↩️ Signal reversed",
                "hl_close_stop_loss":       "🛑 Stop loss",
                "hl_close_take_profit":     "🎯 Take profit",
                "hl_close_trail_stop":      "📌 Trailing stop",
                "hl_close_liquidated":      "💀 Liquidated",
                "hl_realtime_close":        "⚡ Realtime close",
                "hl_realtime_close_liquidation": "💀 Realtime liquidation",
                "forced_minimum_perp":      "⚠️ Min trade limit",
                "stop_loss":                "🛑 Stop loss",
                "take_profit":              "🎯 Take profit",
                "trail_stop":               "📌 Trailing stop",
                "liquidated":               "💀 Liquidated",
                "signal_reversed":          "↩️ Signal reversed",
            }
            raw_reason = c["reason"] or ""
            reason_label = next(
                (v for k, v in _reason_map.items() if k in raw_reason),
                raw_reason[:28] or "—"
            )

            closed_rows += f"""
            <tr>
                <td style="color:#8b949e;white-space:nowrap">{_to_vn_time(c['ts_close'])}</td>
                <td><b>{c['sym']}</b></td>
                <td style="color:{dir_color}">{dir_label}</td>
                <td style="color:#8b949e">{_fmt_usd(c['collat'])}</td>
                <td>{_fmt_price(c['entry'])}</td>
                <td>{_fmt_price(c['exit'])}</td>
                <td style="color:{pnl_color}">
                    {pnl_icon} {_fmt_usd(c['pnl_gross_usd'])}<br>
                    <small style="font-weight:400">{_fmt_pct(c['pnl_gross_pct']*100)}</small>
                </td>
                <td style="color:#ff6d00;font-size:12px">-{_fmt_usd(c['fee_usd'])}</td>
                <td style="color:{pnl_color};font-weight:700">
                    {pnl_icon} {_fmt_usd(c['pnl_net_usd'])}<br>
                    <small style="font-weight:400">{_fmt_pct(c['pnl_net_pct']*100)}</small>
                </td>
                <td style="color:#8b949e;font-size:11px">{hold_str}</td>
                <td style="color:#484f58;font-size:11px">{reason_label}</td>
            </tr>"""

        if not closed_rows:
            closed_rows = "<tr><td colspan='11' style='text-align:center;color:#888;padding:16px'>No closed trades yet</td></tr>"

    except Exception as e:
        closed_rows = f"<tr><td colspan='11' style='color:#ff1744'>Error: {e}</td></tr>"
        total_realized_pnl = 0.0

    realized_color = "#00c853" if total_realized_pnl >= 0 else "#ff1744"

    # Reflexion summary
    try:
        from vicent.strategy.reflexion import ReflexionEngine
        _rfx = ReflexionEngine()
        rfx_summary = _rfx.get_summary()
        rfx_autopsies = _rfx.get_recent_autopsies(limit=8)
    except Exception:
        rfx_summary = {}
        rfx_autopsies = []

    # Build reflexion signal bias table
    penalized = rfx_summary.get("penalized_signals", {})
    boosted = rfx_summary.get("boosted_signals", {})
    banned = rfx_summary.get("banned_signals", [])
    cooled = rfx_summary.get("cooled_symbols", [])
    rfx_win_rate = rfx_summary.get("win_rate", 0.5)
    rfx_total = rfx_summary.get("total_autopsies", 0)

    bias_rows = ""
    all_bias = {**{k: v for k, v in penalized.items()}, **{k: v for k, v in boosted.items()}}
    for sig, bias in sorted(all_bias.items(), key=lambda x: x[1]):
        color = "#ff1744" if bias < 0 else "#00c853"
        icon = "📉" if bias < -0.1 else ("⚠️" if bias < 0 else "📈")
        banned_note = " 🚫BAN" if sig in banned else ""
        pct = f"{bias*100:+.0f}%"
        bias_rows += f"<tr><td>{icon} {sig}</td><td style='color:{color};font-weight:700'>{pct}{banned_note}</td></tr>"
    if not bias_rows:
        bias_rows = "<tr><td colspan='2' style='color:#8b949e;text-align:center'>No bias data yet (needs 1 closed trade)</td></tr>"

    autopsy_rows = ""
    _cause_map = {
        "profitable_trade":   "✅ Profitable",
        "wrong_direction":    "❌ Wrong direction",
        "regime_mismatch":    "🌊 Regime mismatch",
        "signal_conflict":    "⚡ Conflicting signals",
        "stop_hunted":        "🎯 Stop hunted",
        "good_trade_bad_luck":"🎲 Bad luck",
    }
    for a in rfx_autopsies:
        pnl = a.get("pnl_pct", 0)
        losing = a.get("losing", 0)
        pnl_color = "#ff1744" if losing else "#00c853"
        icon = "❌" if losing else "✅"
        cause = _cause_map.get(a.get("loss_cause",""), a.get("loss_cause","") or "—")
        lesson = a.get("lesson", "") or "—"
        autopsy_rows += f"""
        <tr>
            <td>{_to_vn_time(a.get('ts',''))}</td>
            <td><b>{a.get('symbol','')}</b></td>
            <td style="color:#8b949e">{a.get('direction','').upper()}</td>
            <td style="color:{pnl_color};font-weight:700">{icon} {_fmt_pct(pnl*100)}</td>
            <td style="color:#ffd600;font-size:11px">{cause}</td>
            <td style="font-size:11px;color:#ccc">{lesson}</td>
        </tr>"""
    if not autopsy_rows:
        autopsy_rows = "<tr><td colspan='6' style='color:#8b949e;text-align:center'>No trades analyzed yet</td></tr>"

    rfx_wr_color = "#00c853" if rfx_win_rate >= 0.5 else "#ff1744"
    cooled_str = ", ".join(cooled) if cooled else "None"

    iter_rows = ""
    for r in iter_logs:
        action = r.get("action", "")
        action_colors = {"traded": "#00c853", "blocked": "#ff6d00", "skipped": "#8b949e"}
        action_icons  = {"traded": "✅ TRADE", "blocked": "🚫 BLOCK", "skipped": "⏭ SKIP"}
        ac = action_colors.get(action, "#8b949e")
        ai = action_icons.get(action, action)
        intra = "✅" if r.get("intraday_ready") else "⏳"
        regime_color = {"bull": "#00c853", "neutral": "#ffd600", "bear": "#ff1744"}.get(
            r.get("regime", ""), "#8b949e"
        )
        ret = r.get("total_return_pct", 0.0)
        ret_color = "#00c853" if ret >= 0 else "#ff1744"
        row_id = r.get("id", r.get("iteration", 0))
        reason = r.get("action_reason", "") or "—"
        action_sym = r.get("action_symbol", "") or "—"
        top_conf = r.get("top_confidence", 0) or 0
        n_tradeable = r.get("tradeable_count", 0)
        calls = r.get("calls_used", 0)

        # Human-readable summary
        if action == "traded":
            summary = f"Opened spot trade for <b>{action_sym}</b> with confidence <b>{top_conf:.0%}</b>. {reason}"
        elif action == "blocked":
            summary = f"Signal for <b>{action_sym}</b> was blocked: {reason}"
        else:
            summary = reason

        detail_items = [
            ("💰 NAV", _fmt_usd(r.get('nav_usd', 0))),
            ("📉 Drawdown", _fmt_pct(r.get('drawdown_pct', 0), sign=False)),
            ("🎯 Eligible signals", f"{n_tradeable} tokens"),
            ("🔝 Best token", f"{r.get('top_symbol','—') or '—'} ({_fmt_conf(top_conf)})" if top_conf else "—"),
            ("📡 API calls this iteration", str(calls)),
            ("📊 Intraday TA", "✅ Active" if r.get("intraday_ready") else "⏳ Insufficient data"),
            ("🌍 Regime", r.get("regime", "?").upper()),
            ("😨 Fear & Greed", str(r.get("fear_greed", "?"))),
        ]
        detail_html = "".join(
            f'<div style="background:#21262d;border-radius:6px;padding:8px 12px">'
            f'<div style="color:#8b949e;font-size:10px;margin-bottom:2px">{lbl}</div>'
            f'<div style="font-size:13px;font-weight:600">{val}</div></div>'
            for lbl, val in detail_items
        )

        iter_rows += f"""
        <tr onclick="toggleDetail({row_id})" style="cursor:pointer" onmouseover="this.style.background='#1c2128'" onmouseout="this.style.background=''">
            <td style="color:#8b949e">
                <span id="icon-{row_id}" style="font-size:14px;color:#f0b90b;margin-right:6px">＋</span>#{r.get('iteration','')}
            </td>
            <td style="color:#8b949e;white-space:nowrap">{_to_vn_time(r.get('ts',''))}</td>
            <td style="color:{regime_color}">{r.get('regime','').upper()}</td>
            <td>{r.get('fear_greed','')}</td>
            <td style="color:{ac};font-weight:600">{ai}</td>
            <td><b>{r.get('action_symbol','') or '–'}</b></td>
            <td style="font-size:11px;color:#aaa">{reason[:45]}</td>
            <td style="color:{ret_color}">{_fmt_pct(ret)}</td>
            <td>{intra} {calls}</td>
        </tr>
        <tr id="detail-{row_id}" style="display:none">
            <td colspan="9" style="background:#0d1117;padding:0;border-top:none">
                <div style="margin:0 8px 8px 28px;border-left:3px solid #f0b90b;padding:12px 16px;border-radius:0 8px 8px 0;background:#161b22">
                    <p style="font-size:13px;color:#e6edf3;margin-bottom:12px">📋 <b>Summary of iteration #{r.get('iteration','')}:</b> {summary}</p>
                    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:8px">
                        {detail_html}
                    </div>
                </div>
            </td>
        </tr>"""

    if not iter_rows:
        iter_rows = "<tr><td colspan='9' style='text-align:center;color:#888;padding:16px'>No logs available</td></tr>"

    return_color = "#00c853" if total_return >= 0 else "#ff1744"
    network_badge = f"TWAK {cfg.twak_chain.upper()}"
    network_color = "#f0b90b"
    mode_badge = "LIVE 💸" if cfg.is_live else "PAPER 📄"
    mode_color = "#ff1744" if cfg.is_live else "#00b0ff"

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta http-equiv="refresh" content="10">
    <title>VICENT Dashboard</title>
    <style>
        * {{ box-sizing: border-box; margin: 0; padding: 0; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0d1117; color: #e6edf3; min-height: 100vh; }}
        .header {{ background: #161b22; border-bottom: 1px solid #30363d; padding: 16px 24px; display: flex; align-items: center; gap: 16px; }}
        .logo {{ font-size: 22px; font-weight: 700; color: #f0b90b; letter-spacing: 2px; }}
        .badge {{ padding: 4px 10px; border-radius: 20px; font-size: 12px; font-weight: 600; }}
        .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; padding: 24px; }}
        .card {{ background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 20px; }}
        .card-title {{ font-size: 12px; color: #8b949e; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
        .card-value {{ font-size: 28px; font-weight: 700; }}
        .card-sub {{ font-size: 12px; color: #8b949e; margin-top: 4px; }}
        .section {{ padding: 0 24px 24px; }}
        .section-title {{ font-size: 14px; font-weight: 600; color: #8b949e; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 1px; }}
        table {{ width: 100%; border-collapse: collapse; background: #161b22; border: 1px solid #30363d; border-radius: 10px; overflow: hidden; }}
        th {{ background: #21262d; color: #8b949e; font-size: 12px; font-weight: 600; padding: 10px 14px; text-align: left; text-transform: uppercase; }}
        td {{ padding: 10px 14px; font-size: 13px; border-top: 1px solid #21262d; }}
        tr:hover td {{ background: #1c2128; }}
        .heartbeat-bar {{ margin: 0 24px 24px; background: #161b22; border: 1px solid #30363d; border-radius: 10px; padding: 16px 20px; display: flex; align-items: center; gap: 20px; flex-wrap: wrap; }}
        .hb-status {{ font-size: 16px; font-weight: 700; color: {hb_color}; }}
        .hb-detail {{ font-size: 12px; color: #8b949e; }}
        .refresh-note {{ text-align: center; color: #484f58; font-size: 11px; padding: 12px; }}
    </style>
</head>
<body>
    <div class="header">
        <span class="logo">⚡ VICENT</span>
        <span class="badge" style="background:{mode_color}22; color:{mode_color}">{mode_badge}</span>
        <span class="badge" style="background:{network_color}22; color:{network_color}">{network_badge}</span>
        <span style="margin-left:auto; font-size:12px; color:#484f58">Auto-refresh every 10s</span>
    </div>

    <!-- Heartbeat bar -->
    <div class="heartbeat-bar">
        <div class="hb-status">{hb_text}</div>
        <div class="hb-detail">Iteration: <b>#{iteration}</b></div>
        <div class="hb-detail">Last Seen: <b>{last_seen} (ICT)</b></div>
        <div class="hb-detail">Age: <b>{age} seconds ago</b></div>
        <div class="hb-detail">Interval: <b>{hb.get('interval_sec', '?')}s</b></div>
    </div>

    <!-- Defense posture bar -->
    <div class="heartbeat-bar" style="border-color:{defense_color}55">
        <div style="font-size:15px;font-weight:700;color:{defense_color}">{defense_text}</div>
        <div class="hb-detail" style="flex:1">Market Guard: <b>{defense_reason}</b></div>
    </div>

    <!-- Stats cards -->
    <div class="grid">
        <div class="card">
            <div class="card-title">💰 Initial Capital</div>
            <div class="card-value">{_fmt_usd(initial)}</div>
            <div class="card-sub">Current NAV: {_fmt_usd(nav)}</div>
        </div>
        <div class="card">
            <div class="card-title">📈 Realized Return</div>
            <div class="card-value" style="color:{return_color}">{_fmt_pct(total_return)}</div>
            <div class="card-sub">{_fmt_usd(nav - initial)} vs initial capital</div>
        </div>
        <div class="card">
            <div class="card-title">⚡ Equity (NAV + Unrealized PnL)</div>
            <div class="card-value" style="color:{equity_color}">{_fmt_usd(equity)}</div>
            <div class="card-sub">
                Net value if closed now: <b style="color:{equity_color}">{_fmt_usd(total_unrealized_pnl)} ({_fmt_pct(total_unrealized_pct)})</b>
            </div>
        </div>
        <div class="card">
            <div class="card-title">📊 Capital Deployed / Reserve</div>
            <div style="display:flex;gap:12px;align-items:flex-end;margin-top:4px">
                <div>
                    <div style="font-size:11px;color:#8b949e;margin-bottom:2px">Deployed</div>
                    <div class="card-value" style="font-size:22px;color:#f0b90b">{_fmt_usd(deployed_usd)}</div>
                    <div style="font-size:11px;color:#8b949e">{_fmt_pct(deployed_pct, sign=False)} NAV</div>
                </div>
                <div style="color:#30363d;font-size:20px;padding-bottom:4px">/</div>
                <div>
                    <div style="font-size:11px;color:#8b949e;margin-bottom:2px">Reserve</div>
                    <div class="card-value" style="font-size:22px">{_fmt_usd(nav * reserve_pct / 100)}</div>
                    <div style="font-size:11px;color:#8b949e">{_fmt_pct(reserve_pct, sign=False)} NAV</div>
                </div>
            </div>
        </div>
    </div>

    <!-- Open positions -->
    <div class="section">
        <div class="section-title" style="display:flex;align-items:center;gap:16px">
            🔥 Open Token Positions
            <span style="font-size:13px;font-weight:400;color:{total_upnl_color}">
                Total Unrealized PnL: <b>{total_upnl_icon} {_fmt_usd(total_unrealized_pnl)} ({_fmt_pct(total_unrealized_pct)})</b>
            </span>
        </div>
        <table>
            <thead><tr>
                <th>Token</th><th>Direction</th><th>Trade Cost</th>
                <th>Entry Price</th><th>Current Price</th>
                <th>Unrealized PnL</th>
                <th>Stop Loss (Est.)</th><th>Confidence</th><th>Entry Time</th>
            </tr></thead>
            <tbody>{perps_positions_html}</tbody>
        </table>
    </div>

    <!-- Recent closed positions -->
    <div class="section">
        <div class="section-title" style="display:flex;align-items:center;gap:16px">
            📊 Closed Trades (Last 15)
            <span style="font-size:13px;font-weight:400;color:{realized_color}">
                Total Realized PnL: <b>{_fmt_usd(total_realized_pnl)}</b>
            </span>
        </div>
        <table>
            <thead><tr>
                <th>Close Time</th>
                <th>Token</th>
                <th>Direction</th>
                <th>Trade Capital</th>
                <th>Entry Price</th>
                <th>Exit Price</th>
                <th>Gross PnL</th>
                <th>Fees</th>
                <th>Net PnL</th>
                <th>Hold</th>
                <th>Exit Reason</th>
            </tr></thead>
            <tbody>{closed_rows}</tbody>
        </table>
    </div>

    <!-- Recent trades -->
    <div class="section">
        <div class="section-title">🕐 Recent Operations (Last 10)</div>
        <table>
            <thead><tr><th>Time</th><th>Token</th><th>Type</th><th>USD</th><th>Price</th><th>Confidence</th><th>TxHash</th></tr></thead>
            <tbody>{trades_rows}</tbody>
        </table>
    </div>

    <!-- Reflexion Engine -->
    <div class="section">
        <div class="section-title">🧠 Reflexion Engine — Self-learning from losses</div>
        <div style="display:grid;grid-template-columns:1fr 2fr;gap:16px">
            <div>
                <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px;margin-bottom:16px">
                    <div style="font-size:12px;color:#8b949e;margin-bottom:8px">📊 Overview</div>
                    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
                        <div style="background:#0d1117;padding:8px;border-radius:6px">
                            <div style="font-size:10px;color:#8b949e">Analyzed Trades</div>
                            <div style="font-size:20px;font-weight:700;color:#f0b90b">{rfx_total}</div>
                        </div>
                        <div style="background:#0d1117;padding:8px;border-radius:6px">
                            <div style="font-size:10px;color:#8b949e">Actual Win Rate</div>
                            <div style="font-size:20px;font-weight:700;color:{rfx_wr_color}">{rfx_win_rate*100:.0f}%</div>
                        </div>
                        <div style="background:#0d1117;padding:8px;border-radius:6px;grid-column:1/-1">
                            <div style="font-size:10px;color:#8b949e">Cooldown Tokens (Temporary)</div>
                            <div style="font-size:13px;font-weight:600;color:#ff6d00">{cooled_str}</div>
                        </div>
                    </div>
                </div>
                <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px">
                    <div style="font-size:12px;color:#8b949e;margin-bottom:8px">⚖️ Signal Bias (Learned)</div>
                    <table style="background:transparent;border:none">
                        <thead><tr><th style="background:transparent;font-size:11px">Signal</th><th style="background:transparent;font-size:11px">Adjustment</th></tr></thead>
                        <tbody>{bias_rows}</tbody>
                    </table>
                </div>
            </div>
            <div style="background:#161b22;border:1px solid #30363d;border-radius:10px;padding:16px">
                <div style="font-size:12px;color:#8b949e;margin-bottom:8px">🔬 Recent Trade Autopsies</div>
                <table style="background:transparent;border:none">
                    <thead><tr>
                        <th style="background:transparent;font-size:11px">Time</th>
                        <th style="background:transparent;font-size:11px">Token</th>
                        <th style="background:transparent;font-size:11px">Direction</th>
                        <th style="background:transparent;font-size:11px">PnL</th>
                        <th style="background:transparent;font-size:11px">Cause</th>
                        <th style="background:transparent;font-size:11px">Lesson Learned</th>
                    </tr></thead>
                    <tbody>{autopsy_rows}</tbody>
                </table>
            </div>
        </div>
    </div>

    <!-- Iteration log -->
    <div class="section">
        <div class="section-title">📋 Iteration History (Last 30)</div>
        <table>
            <thead><tr><th>Iteration</th><th>Time</th><th>Regime</th><th>F&G</th><th>Action</th><th>Token</th><th>Reason</th><th>Return</th><th>Intra/Calls</th></tr></thead>
            <tbody>{iter_rows}</tbody>
        </table>
    </div>

    <div class="refresh-note">Page auto-refreshes every 10 seconds · Timezone: ICT (UTC+7) · VICENT v0.2.0</div>
<script>
    function toggleDetail(id) {{
        var detail = document.getElementById('detail-' + id);
        var icon   = document.getElementById('icon-' + id);
        if (!detail) return;
        var open = detail.style.display === 'none' || detail.style.display === '';
        detail.style.display = open ? 'table-row' : 'none';
        icon.textContent = open ? '－' : '＋';
    }}
</script>
</body>
</html>"""
    return HTMLResponse(html)


# ---------------------------------------------------------------------------
# JSON API endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, Any]:
    hb = read_heartbeat()
    return {
        "status": hb.get("status", "dead"),
        "agent": "vicent",
        "iteration": hb.get("iteration", 0),
        "last_seen": hb.get("last_seen"),
        "age_seconds": hb.get("age_seconds"),
    }


@app.get("/heartbeat")
async def heartbeat() -> dict[str, Any]:
    return read_heartbeat()


@app.get("/portfolio")
async def portfolio() -> JSONResponse:
    snapshot = get_latest_snapshot()
    if not snapshot:
        return JSONResponse({"error": "no_snapshot_yet"}, status_code=404)
    return JSONResponse(snapshot)


@app.get("/trades")
async def trades(limit: int = 50) -> list[dict[str, Any]]:
    return get_all_trades(limit=limit)


@app.get("/trades/today")
async def trades_today_endpoint() -> dict[str, Any]:
    today_trades = get_trades_today()
    return {"count": len(today_trades), "trades": today_trades}


@app.get("/iteration-logs")
async def iteration_logs(limit: int = 50) -> list[dict[str, Any]]:
    return get_iteration_logs(limit=limit)


@app.get("/config")
async def config() -> dict[str, Any]:
    cfg = get_settings()
    return {
        "mode": cfg.vicent_mode.value,
        "twak_chain": cfg.twak_chain,
        "twak_enabled": cfg.twak_enabled,
        "strategy": cfg.vicent_strategy.value,
        "risk": {
            "max_drawdown_pct": cfg.risk_max_drawdown_pct,
            "daily_loss_pct": cfg.risk_daily_loss_pct,
            "per_trade_nav_pct": cfg.risk_per_trade_nav_pct,
            "max_slippage_bps": cfg.risk_max_slippage_bps,
        },
    }
