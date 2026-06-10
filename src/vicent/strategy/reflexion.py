"""ReflexionEngine — Agent tự phân tích nguyên nhân thua và điều chỉnh.

Sau mỗi lệnh đóng (close), engine:
  1. Ghi autopsy: lý do thua/thắng, signal nào sai, regime lúc đó là gì
  2. Phân tích pattern: signal nào hay sai nhất (false positive/negative)
  3. Điều chỉnh signal_bias: giảm trọng số signal hay sai, tăng signal hay đúng
  4. Điều chỉnh confidence threshold theo từng regime
  5. Cấm trade tạm thời nếu 1 signal liên tục sai trên 1 token cụ thể

Nguyên tắc:
  - Không fake data, không mock — chỉ học từ lệnh thật đã đóng
  - signal_bias lưu vào SQLite → sống sót qua restart
  - Xem được lịch sử học qua dashboard
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

log = structlog.get_logger(__name__)

def _get_db_path() -> Path:
    from vicent.config import get_settings
    return Path(get_settings().vicent_db_path)

# Giới hạn điều chỉnh bias
_BIAS_MIN = -0.4   # giảm tối đa 40% trọng số 1 signal
_BIAS_MAX = +0.3   # tăng tối đa 30% trọng số 1 signal
_BIAS_STEP_LOSE = -0.06   # mỗi lần thua giảm bias
_BIAS_STEP_WIN  = +0.03   # mỗi lần thắng tăng bias (học chậm hơn)

# Sau bao nhiêu lần thua liên tiếp thì cấm tạm thời signal đó
_BAN_THRESHOLD = 3


@dataclass
class TradeAutopsy:
    """Phân tích 1 lệnh đã đóng."""
    trade_id: int
    symbol: str
    direction: str           # "long" | "short"
    regime: str
    pnl_pct: float           # % lãi/lỗ thực tế (có đòn bẩy)
    confidence_at_entry: float
    sub_scores_json: str     # JSON của sub_scores lúc vào lệnh
    # Chẩn đoán
    losing: bool
    loss_cause: str          # "wrong_direction" | "stop_hunted" | "regime_mismatch" | "signal_conflict" | "good_trade_bad_luck"
    signals_that_failed: list[str]   # signal nào predict sai
    signals_that_worked: list[str]   # signal nào predict đúng
    lesson: str              # câu tóm tắt cho người đọc


@dataclass
class SignalBias:
    """Bias hiện tại của 1 signal, lưu vào DB."""
    signal_name: str
    bias: float              # -0.4 đến +0.3, nhân vào weight
    consecutive_losses: int  # thua liên tiếp
    total_trades: int
    win_trades: int
    banned_until: str | None  # ISO timestamp, None = không bị cấm

    @property
    def win_rate(self) -> float:
        if self.total_trades == 0:
            return 0.5
        return self.win_trades / self.total_trades

    @property
    def is_banned(self) -> bool:
        if self.banned_until is None:
            return False
        now = datetime.now(timezone.utc).isoformat()
        return now < self.banned_until


@dataclass
class ReflexionState:
    """Toàn bộ trạng thái học hiện tại."""
    signal_biases: dict[str, SignalBias] = field(default_factory=dict)
    regime_thresholds: dict[str, float] = field(default_factory=dict)
    # symbol-level blocks: {symbol: banned_until_iso}
    symbol_cooldowns: dict[str, str] = field(default_factory=dict)
    total_autopsies: int = 0
    total_wins: int = 0
    total_losses: int = 0

    @property
    def overall_win_rate(self) -> float:
        total = self.total_wins + self.total_losses
        return self.total_wins / total if total > 0 else 0.5


class ReflexionEngine:
    """Engine học từ lệnh thua, điều chỉnh signal weight + threshold."""

    # Tên tất cả signal đang dùng
    ALL_SIGNALS = [
        "momentum", "roc", "rsi", "stoch_rsi", "macd",
        "bb", "vwap", "ema", "volume", "news",
        "whale", "prediction", "hl_funding",
    ]

    def __init__(self) -> None:
        self._init_db()
        self.state = self._load_state()

    # ------------------------------------------------------------------ #
    # Public API — gọi từ agent                                          #
    # ------------------------------------------------------------------ #

    def process_closed_trade(
        self,
        symbol: str,
        direction: str,
        pnl_pct: float,          # % PnL có đòn bẩy (ví dụ: -0.15 = thua 15%)
        confidence_at_entry: float,
        sub_scores: dict[str, float],  # sub_scores từ TokenSignal lúc vào lệnh
        regime: str,
    ) -> TradeAutopsy:
        """Phân tích 1 lệnh vừa đóng và cập nhật state."""

        losing = pnl_pct < 0.0    # bất kỳ PnL âm = coi là thua (đã bao gồm phí)
        winning = pnl_pct > 0.0

        # --- Chẩn đoán nguyên nhân thua ---
        loss_cause, failed, worked = self._diagnose(
            direction, pnl_pct, sub_scores, regime, losing
        )

        lesson = self._generate_lesson(
            symbol, direction, pnl_pct, loss_cause, failed, worked, regime
        )

        # --- Lưu autopsy vào DB ---
        trade_id = self._save_autopsy(
            symbol=symbol,
            direction=direction,
            regime=regime,
            pnl_pct=pnl_pct,
            confidence=confidence_at_entry,
            sub_scores_json=json.dumps(sub_scores),
            losing=losing,
            loss_cause=loss_cause,
            signals_failed=failed,
            signals_worked=worked,
            lesson=lesson,
        )

        autopsy = TradeAutopsy(
            trade_id=trade_id,
            symbol=symbol,
            direction=direction,
            regime=regime,
            pnl_pct=pnl_pct,
            confidence_at_entry=confidence_at_entry,
            sub_scores_json=json.dumps(sub_scores),
            losing=losing,
            loss_cause=loss_cause,
            signals_that_failed=failed,
            signals_that_worked=worked,
            lesson=lesson,
        )

        # --- Cập nhật bias ---
        if losing:
            self.state.total_losses += 1
            self._penalize_signals(failed, worked)
            self._maybe_cooldown_symbol(symbol, sub_scores)
        elif winning:
            self.state.total_wins += 1
            self._reward_signals(failed, worked)

        self.state.total_autopsies += 1
        self._save_state()

        log.warning(
            "reflexion_autopsy",
            symbol=symbol,
            direction=direction,
            pnl_pct=f"{pnl_pct:.2%}",
            losing=losing,
            cause=loss_cause,
            failed=failed,
            lesson=lesson,
        ) if losing else log.info(
            "reflexion_autopsy",
            symbol=symbol,
            direction=direction,
            pnl_pct=f"{pnl_pct:.2%}",
            winning=winning,
            worked=worked,
        )

        return autopsy

    def get_adjusted_weights(self, base_weights: dict[str, float]) -> dict[str, float]:
        """Trả về weight đã điều chỉnh theo bias học được.

        sub_scores là giá trị có dấu (âm/dương), không phải weight thuần dương.
        Bias điều chỉnh *magnitude* (độ lớn) của score — không được đổi dấu.

        Ví dụ:
          w = +0.3  (bullish signal), bias = -0.06  → magnitude giảm → 0.3 * 0.94 = +0.282
          w = -0.4  (bearish signal), bias = -0.06  → magnitude giảm → -0.4 * 0.94 = -0.376
          w = +0.2  (sai direction), bias = -0.40   → 0.2 * 0.60 = +0.12  (giảm nhiều)
        """
        adjusted = {}
        for signal, w in base_weights.items():
            bias_obj = self.state.signal_biases.get(signal)
            if bias_obj and bias_obj.is_banned:
                adjusted[signal] = 0.0  # cấm hoàn toàn — signal này bị silence
                log.debug("signal_banned", signal=signal)
            elif bias_obj and bias_obj.bias != 0.0:
                # Giảm/tăng magnitude, giữ nguyên dấu
                multiplier = max(0.0, 1.0 + bias_obj.bias)  # không âm
                adjusted[signal] = w * multiplier            # giữ dấu gốc
            else:
                adjusted[signal] = w
        return adjusted

    def get_regime_threshold(self, regime: str, base: float) -> float:
        """Trả về confidence threshold điều chỉnh theo lịch sử regime."""
        return self.state.regime_thresholds.get(regime, base)

    def is_symbol_cooling_down(self, symbol: str) -> bool:
        """True nếu symbol đang trong cooldown (thua nhiều quá)."""
        deadline = self.state.symbol_cooldowns.get(symbol)
        if not deadline:
            return False
        cooling = datetime.now(timezone.utc).isoformat() < deadline
        if not cooling:
            # Hết cooldown → xóa
            del self.state.symbol_cooldowns[symbol]
            self._save_state()
        return cooling

    def get_summary(self) -> dict[str, Any]:
        """Tóm tắt trạng thái học để hiển thị trên dashboard."""
        penalized = {
            k: round(v.bias, 3)
            for k, v in self.state.signal_biases.items()
            if v.bias < -0.05
        }
        boosted = {
            k: round(v.bias, 3)
            for k, v in self.state.signal_biases.items()
            if v.bias > 0.05
        }
        banned_signals = [k for k, v in self.state.signal_biases.items() if v.is_banned]
        return {
            "total_autopsies": self.state.total_autopsies,
            "win_rate": round(self.state.overall_win_rate, 3),
            "penalized_signals": penalized,
            "boosted_signals": boosted,
            "banned_signals": banned_signals,
            "cooled_symbols": list(self.state.symbol_cooldowns.keys()),
            "regime_thresholds": self.state.regime_thresholds,
        }

    # ------------------------------------------------------------------ #
    # Chẩn đoán nguyên nhân                                              #
    # ------------------------------------------------------------------ #

    def _diagnose(
        self,
        direction: str,
        pnl_pct: float,
        sub_scores: dict[str, float],
        regime: str,
        losing: bool,
    ) -> tuple[str, list[str], list[str]]:
        """Xác định signal nào sai, signal nào đúng.

        Quan trọng: chỉ count signal có magnitude đủ lớn (|score| >= 0.15)
        để tránh noise và signal trung tính bị label nhầm thành "failed".
        Threshold 0.15 vì sub_scores thường ở [-0.9, +0.9] sau ADX boost.
        """
        is_long = direction in ("long", "LONG")
        SIGNIFICANT = 0.15

        failed: list[str] = []
        worked: list[str] = []

        for sig, score in sub_scores.items():
            if sig in ("adaptive_mode", "adx"):
                continue
            if abs(score) < SIGNIFICANT:
                continue
            if is_long:
                if score > 0:
                    worked.append(sig)
                else:
                    failed.append(sig)
            else:
                if score < 0:
                    worked.append(sig)
                else:
                    failed.append(sig)

        # Xác định nguyên nhân chính
        if not losing:
            return "profitable_trade", failed, worked

        # Regime mismatch: vào LONG trong BEAR hoặc SHORT trong BULL
        if "bear" in regime.lower() and is_long:
            return "regime_mismatch", failed, worked
        if "bull" in regime.lower() and not is_long:
            return "regime_mismatch", failed, worked

        # Signal conflict: nhiều signal mâu thuẫn nhau
        if len(failed) >= 3 and len(worked) >= 2:
            return "signal_conflict", failed, worked

        # Stop hunted: thua nhỏ (< 10% collateral với 5x = < 2% giá)
        if -0.10 <= pnl_pct < 0.0:
            return "stop_hunted", failed, worked

        # Wrong direction: signal chỉ sai hướng
        if len(failed) > len(worked):
            return "wrong_direction", failed, worked

        return "good_trade_bad_luck", failed, worked

    def _generate_lesson(
        self,
        symbol: str,
        direction: str,
        pnl_pct: float,
        cause: str,
        failed: list[str],
        worked: list[str],
        regime: str,
    ) -> str:
        if cause == "profitable_trade":
            return f"{symbol} {direction} +{pnl_pct:.1%}: {', '.join(worked[:3])} đúng hướng"
        if cause == "regime_mismatch":
            return f"{symbol}: vào LONG khi regime={regime} — không trade ngược trend nữa"
        if cause == "signal_conflict":
            f_str = ", ".join(failed[:2])
            w_str = ", ".join(worked[:2])
            return f"{symbol}: tín hiệu mâu thuẫn ({f_str} sai vs {w_str} đúng) — cần đồng thuận cao hơn"
        if cause == "stop_hunted":
            return f"{symbol}: bị stop hunt tại {pnl_pct:.1%} — cân nhắc nới SL hoặc giảm đòn bẩy"
        if cause == "wrong_direction":
            f_str = ", ".join(failed[:3])
            return f"{symbol}: {f_str} predict sai hướng — giảm trọng số các signal này"
        return f"{symbol} {direction} {pnl_pct:.1%}: thua do xui, không điều chỉnh gì"

    # ------------------------------------------------------------------ #
    # Điều chỉnh bias                                                     #
    # ------------------------------------------------------------------ #

    def _penalize_signals(self, failed: list[str], worked: list[str]) -> None:
        """Giảm bias signal sai, tăng nhẹ signal đúng."""
        for sig in failed:
            b = self.state.signal_biases.setdefault(
                sig, SignalBias(sig, 0.0, 0, 0, 0, None)
            )
            b.bias = max(_BIAS_MIN, b.bias + _BIAS_STEP_LOSE)
            b.consecutive_losses += 1
            b.total_trades += 1
            # Ban nếu thua liên tiếp quá nhiều
            if b.consecutive_losses >= _BAN_THRESHOLD:
                from datetime import timedelta
                ban_hours = 6 * b.consecutive_losses
                b.banned_until = (
                    datetime.now(timezone.utc)
                    + timedelta(hours=ban_hours)
                ).isoformat()
                log.warning(
                    "signal_banned",
                    signal=sig,
                    consecutive_losses=b.consecutive_losses,
                    ban_hours=ban_hours,
                )

        for sig in worked:
            b = self.state.signal_biases.setdefault(
                sig, SignalBias(sig, 0.0, 0, 0, 0, None)
            )
            # Reset consecutive losses khi signal thắng
            b.consecutive_losses = max(0, b.consecutive_losses - 1)
            b.total_trades += 1

    def _reward_signals(self, failed: list[str], worked: list[str]) -> None:
        """Tăng bias signal đúng sau lệnh thắng."""
        for sig in worked:
            b = self.state.signal_biases.setdefault(
                sig, SignalBias(sig, 0.0, 0, 0, 0, None)
            )
            b.bias = min(_BIAS_MAX, b.bias + _BIAS_STEP_WIN)
            b.consecutive_losses = 0
            b.total_trades += 1
            b.win_trades += 1
            # Unban nếu đang bị cấm mà lại thắng
            if b.is_banned:
                b.banned_until = None
                log.info("signal_unbanned", signal=sig)

        for sig in failed:
            b = self.state.signal_biases.setdefault(
                sig, SignalBias(sig, 0.0, 0, 0, 0, None)
            )
            b.total_trades += 1

    def _maybe_cooldown_symbol(self, symbol: str, sub_scores: dict) -> None:
        """Cấm trade 1 symbol nếu thua nhiều quá."""
        # Đếm số lần thua gần đây với symbol này
        recent_losses = self._count_recent_losses(symbol, hours=24)
        if recent_losses >= 2:
            from datetime import timedelta
            cooldown_hours = 4 * recent_losses
            deadline = (
                datetime.now(timezone.utc) + timedelta(hours=cooldown_hours)
            ).isoformat()
            self.state.symbol_cooldowns[symbol] = deadline
            log.warning(
                "symbol_cooldown",
                symbol=symbol,
                recent_losses=recent_losses,
                cooldown_hours=cooldown_hours,
            )

    # ------------------------------------------------------------------ #
    # Database                                                            #
    # ------------------------------------------------------------------ #

    def _init_db(self) -> None:
        with sqlite3.connect(_get_db_path()) as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trade_autopsy (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts              TEXT    NOT NULL,
                    symbol          TEXT    NOT NULL,
                    direction       TEXT    NOT NULL,
                    regime          TEXT,
                    pnl_pct         REAL    NOT NULL,
                    confidence      REAL,
                    sub_scores_json TEXT,
                    losing          INTEGER NOT NULL,
                    loss_cause      TEXT,
                    signals_failed  TEXT,   -- JSON list
                    signals_worked  TEXT,   -- JSON list
                    lesson          TEXT
                );

                CREATE TABLE IF NOT EXISTS reflexion_state (
                    key     TEXT PRIMARY KEY,
                    value   TEXT NOT NULL
                );
            """)

    def _save_autopsy(
        self,
        symbol: str,
        direction: str,
        regime: str,
        pnl_pct: float,
        confidence: float,
        sub_scores_json: str,
        losing: bool,
        loss_cause: str,
        signals_failed: list[str],
        signals_worked: list[str],
        lesson: str,
    ) -> int:
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(_get_db_path()) as conn:
            cur = conn.execute(
                """INSERT INTO trade_autopsy
                   (ts, symbol, direction, regime, pnl_pct, confidence,
                    sub_scores_json, losing, loss_cause,
                    signals_failed, signals_worked, lesson)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
                (ts, symbol, direction, regime, pnl_pct, confidence,
                 sub_scores_json, int(losing), loss_cause,
                 json.dumps(signals_failed), json.dumps(signals_worked), lesson),
            )
            return cur.lastrowid or 0

    def _save_state(self) -> None:
        biases_data = {
            k: {
                "bias": v.bias,
                "consecutive_losses": v.consecutive_losses,
                "total_trades": v.total_trades,
                "win_trades": v.win_trades,
                "banned_until": v.banned_until,
            }
            for k, v in self.state.signal_biases.items()
        }
        payload = {
            "signal_biases": biases_data,
            "regime_thresholds": self.state.regime_thresholds,
            "symbol_cooldowns": self.state.symbol_cooldowns,
            "total_autopsies": self.state.total_autopsies,
            "total_wins": self.state.total_wins,
            "total_losses": self.state.total_losses,
        }
        with sqlite3.connect(_get_db_path()) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO reflexion_state (key, value) VALUES ('main', ?)",
                (json.dumps(payload),),
            )

    def _load_state(self) -> ReflexionState:
        try:
            with sqlite3.connect(_get_db_path()) as conn:
                row = conn.execute(
                    "SELECT value FROM reflexion_state WHERE key='main'"
                ).fetchone()
            if not row:
                return ReflexionState()
            data = json.loads(row[0])
            state = ReflexionState(
                regime_thresholds=data.get("regime_thresholds", {}),
                symbol_cooldowns=data.get("symbol_cooldowns", {}),
                total_autopsies=data.get("total_autopsies", 0),
                total_wins=data.get("total_wins", 0),
                total_losses=data.get("total_losses", 0),
            )
            for sig, bd in data.get("signal_biases", {}).items():
                state.signal_biases[sig] = SignalBias(
                    signal_name=sig,
                    bias=bd["bias"],
                    consecutive_losses=bd["consecutive_losses"],
                    total_trades=bd["total_trades"],
                    win_trades=bd["win_trades"],
                    banned_until=bd.get("banned_until"),
                )
            return state
        except Exception as e:
            log.warning("reflexion_load_failed", error=str(e))
            return ReflexionState()

    def _count_recent_losses(self, symbol: str, hours: int = 24) -> int:
        try:
            with sqlite3.connect(_get_db_path()) as conn:
                rows = conn.execute(
                    """SELECT COUNT(*) FROM trade_autopsy
                       WHERE symbol=? AND losing=1
                       AND ts > datetime('now', ? || ' hours')""",
                    (symbol, f"-{hours}"),
                ).fetchone()
                return rows[0] if rows else 0
        except Exception:
            return 0

    def get_recent_autopsies(self, limit: int = 10) -> list[dict[str, Any]]:
        """Lấy các autopsy gần nhất để hiển thị trên dashboard."""
        try:
            with sqlite3.connect(_get_db_path()) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    """SELECT ts, symbol, direction, pnl_pct, losing,
                              loss_cause, lesson
                       FROM trade_autopsy ORDER BY id DESC LIMIT ?""",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception:
            return []
