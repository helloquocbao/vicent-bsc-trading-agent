"""VICENT main agent loop.

State machine:
  INIT → SYNC_PORTFOLIO → FETCH_MARKET → SCORE_TOKENS → DECIDE
       → MANAGE_POSITIONS → OPEN_TRADES(spot) → SNAPSHOT → REFLECT → SLEEP

Key rules enforced here:
  - Hard drawdown stop at 20% (competition cap is 30%)
  - Daily loss stop at 5%
  - Day-boundary reset at midnight UTC
  - Minimum 1 trade/day (checked with 4h remaining)
  - Volatility-aware stops/TPs (per-token via ATR proxy)
  - Three-tier scaled exits (lock-in 1/3 + 1/3 + ride trail)
  - Fee-aware edge filter (skip trades that can't beat round-trip cost)
  - Sector exposure cap (40% max in any one sector)
  - TWAK-driven spot swaps on BNB Smart Chain
  - Reflexion: closed-trade PnL feeds back to confidence threshold
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timezone
from typing import Any

import structlog

from vicent.config import get_settings
from vicent.execution.twak import TWAKExecutor
from vicent.signals.call_scheduler import CallScheduler
from vicent.signals.cmc_client import CMCClient, resolve_cmc_id
from vicent.signals.indicators import compute_intraday
from vicent.signals.regime import detect_regime
from vicent.signals.sentiment import SentimentVerdict, analyze_sentiment
from vicent.signals.signals import Direction, TokenSignal, score_token
from vicent.state.ledger import (
    get_trades_today,
    init_db,
    record_iteration_log,
    record_snapshot,
    record_trade,
)
from vicent.state.price_history import (
    get_ohlcv_series,
    init_price_history,
    record_prices,
)
from vicent.state.portfolio import Portfolio
from vicent.strategy.ensemble import TradeDecision, decide, rank_decisions
from vicent.strategy.reflexion import ReflexionEngine
from vicent.strategy.optimization import (
    ExitAction,
    correlation_check,
    evaluate_exit,
    should_skip_for_fees,
    volatility_aware_targets,
)
from vicent.strategy.risk import RiskManager
from vicent.strategy.tokens import get_watchlist, is_stable
from vicent.server import write_heartbeat

log = structlog.get_logger(__name__)

# ============================================================
# SPOT-ONLY MODE — capital allocation model
# ============================================================
_SPOT_MAX_POSITIONS = 3      # Maximum 3 concurrent token holdings


class VICENTAgent:
    """Main autonomous spot trading agent for BNB Smart Chain using TWAK."""

    def __init__(self, initial_capital_usd: float = 100.0) -> None:
        self.cfg = get_settings()
        self.portfolio = Portfolio(initial_capital_usd)
        self.risk = RiskManager()
        
        # TWAK Executor
        self.twak = TWAKExecutor() if get_settings().twak_enabled else None
        self._position_lock = asyncio.Lock()
        self._running = False
        self._iteration = 0
        self._last_day: date | None = None
        
        # Reflexion state
        self._min_confidence: float = 0.50
        self._closed_trade_pnls: list[float] = []
        # _entry_meta stores {symbol: {sub_scores, confidence, regime, direction}}
        # so _execute_sell can pass it to ReflexionEngine.process_closed_trade()
        self._entry_meta: dict[str, dict] = {}
        # Cached market-wide data (refreshed every iteration)
        self._marketcap_ta: dict = {}
        self._narratives: list = []
        self._upcoming_events: list = []
        self._derivatives: dict = {}
        self._cached_token_ta: dict[str, dict] = {}
        self._cached_whale: dict[str, dict] = {}
        self._cached_news: dict[str, list] = {}
        self._scheduler = CallScheduler()
        
        self._reflexion = ReflexionEngine()
        self._defense_posture = None

    async def run(self) -> None:
        """Run the agent loop indefinitely until stopped."""
        log.info(
            "agent_starting",
            mode=self.cfg.vicent_mode.value,
            strategy=self.cfg.vicent_strategy.value,
            chain=self.cfg.twak_chain,
            interval_sec=self.cfg.vicent_loop_interval_sec,
        )
        init_db()
        init_price_history()

        # Check API budget warning at startup
        budget_warn = self._scheduler.check_budget_warning(self.cfg.vicent_loop_interval_sec)
        if budget_warn:
            log.warning("api_budget_warning", message=budget_warn)

        # Sync live portfolio on startup
        if self.cfg.is_live:
            await self._sync_live_portfolio()

        self._running = True

        async with CMCClient() as cmc:
            while self._running:
                try:
                    await self._iteration_step(cmc)
                except Exception as e:
                    log.error("iteration_error", error=str(e), iteration=self._iteration)
                    await asyncio.sleep(60)
                    continue

                log.info("sleeping", seconds=self.cfg.vicent_loop_interval_sec)
                await asyncio.sleep(self.cfg.vicent_loop_interval_sec)

    def stop(self) -> None:
        self._running = False

    async def _iteration_step(self, cmc: CMCClient) -> None:
        self._iteration += 1
        now = datetime.now(timezone.utc)
        today = now.date()

        # --- Day boundary reset ---
        if self._last_day is None or today != self._last_day:
            self.portfolio.reset_day_start()
            self._last_day = today
            log.info("day_reset", date=str(today))

        log.info("iteration_start", n=self._iteration, ts=now.isoformat())

        # --- Heartbeat: status server check ---
        write_heartbeat(self._iteration, self.cfg.vicent_loop_interval_sec, status="running")
        self._scheduler.tick(self._iteration)

        # --- Live mode: sync equity and portfolio status ---
        if self.cfg.is_live and self.twak and self._iteration % 4 == 0:
            await self._sync_equity_from_twak()

        n_tokens = len(get_watchlist(max_tier=1))
        est_calls = self._scheduler.estimate_calls_this_iteration(n_tokens)
        log.info("scheduler", iteration=self._iteration, est_calls=est_calls, tokens=n_tokens)

        # --- Hard stops ---
        ok, reason = self.risk.check_hard_stops(self.portfolio)
        if not ok:
            log.warning("hard_stop_triggered", reason=reason)
            self._running = False
            return

        # --- Fetch Market Context: global metrics ---
        try:
            global_metrics = await cmc.get_global_metrics()
        except Exception as e:
            log.error("global_metrics_failed", error=str(e))
            return

        # --- Fetch Scheduled Metrics ---
        if self._scheduler.should_fetch_derivatives():
            result = await cmc.get_derivatives_metrics()
            if not isinstance(result, Exception):
                self._derivatives = result

        if self._scheduler.should_fetch_marketcap_ta():
            result = await cmc.get_marketcap_ta()
            if not isinstance(result, Exception):
                self._marketcap_ta = result

        if self._scheduler.should_fetch_narratives():
            result = await cmc.get_trending_narratives()
            if not isinstance(result, Exception):
                self._narratives = result if isinstance(result, list) else []

        if self._scheduler.should_fetch_events():
            result = await cmc.get_upcoming_events()
            if not isinstance(result, Exception):
                self._upcoming_events = result if isinstance(result, list) else []

        # --- Detect Regime ---
        regime = detect_regime(global_metrics, self._derivatives)
        log.info(
            "regime",
            value=regime.regime.value,
            fear_greed=regime.fear_greed,
            confidence=round(regime.confidence, 2),
        )

        # --- Fetch & Score Watchlist Tokens ---
        watchlist = get_watchlist(max_tier=1)
        signals = await self._fetch_and_score_tokens(cmc, watchlist)
        if not signals:
            log.info("no_signals_this_iteration")
            return

        # --- Update prices ---
        prices = {sig.symbol: sig.price_usd for sig in signals if sig.price_usd > 0}
        self.portfolio.update_prices(prices)

        # --- Ensemble decisions ---
        decisions = [decide(sig, regime, min_confidence=self._min_confidence) for sig in signals]
        ranked = rank_decisions(decisions)

        # --- DEFENSE LAYER ---
        from vicent.strategy.defense import assess_market_health
        posture = assess_market_health(signals, fear_greed=regime.fear_greed)
        self._defense_posture = posture
        effective_min_conf = self._min_confidence + posture.min_confidence_add
        effective_max_pos = (
            posture.max_positions_override
            if posture.max_positions_override is not None
            else _SPOT_MAX_POSITIONS
        )

        log.info(
            "decisions",
            tradeable=len(ranked),
            top=ranked[0].symbol if ranked else "none",
            min_conf_threshold=round(self._min_confidence, 2),
            defense=posture.level.value,
            eff_min_conf=round(effective_min_conf, 2),
            eff_max_pos=effective_max_pos,
        )

        # --- Open trades ---
        _last_action = "skipped"
        _last_action_symbol = ""
        _last_action_reason = ""

        # DEFENSE: skip entries if market is too dangerous
        if not posture.allow_new_entries:
            log.warning("defense_halt_entries", reason=posture.reason)
            ranked = []
            _last_action = "blocked"
            _last_action_reason = f"defense_halt:{posture.reason[:60]}"

        for decision in ranked:
            if len(self.portfolio._positions) >= effective_max_pos:
                break

            # Only trade if bullish
            if decision.direction != Direction.LONG:
                continue

            if decision.confidence < effective_min_conf:
                _last_action = "blocked"
                _last_action_symbol = decision.symbol
                _last_action_reason = f"low_conf:{decision.confidence:.2f}<{effective_min_conf:.2f}"
                continue

            # Check duplication
            if self.portfolio.has_position(decision.symbol):
                continue

            token_sig = next((s for s in signals if s.symbol == decision.symbol), None)
            sentiment = analyze_sentiment(
                global_metrics=global_metrics,
                derivatives=self._derivatives,
                marketcap_ta=self._marketcap_ta,
                narratives=self._narratives,
                upcoming_events=self._upcoming_events,
                target_symbol=decision.symbol,
                token_news=token_sig.news if token_sig else None,
                iteration=self._iteration,
            )

            if sentiment.verdict in (SentimentVerdict.BLOCK, SentimentVerdict.WAIT):
                _last_action = "blocked"
                _last_action_symbol = decision.symbol
                _last_action_reason = f"sentiment_{sentiment.verdict.value}:{sentiment.summary[:40]}"
                continue

            combined_multiplier = sentiment.multiplier * posture.size_multiplier

            # Execute SPOT BUY
            success = await self._execute_buy(decision, prices, signals, combined_multiplier)
            if success:
                _last_action = "traded"
                _last_action_symbol = decision.symbol
                _last_action_reason = f"conf:{decision.confidence:.2f} regime:{decision.regime.value}"
                break  # one new trade per iteration max (re-evaluate next loop)

        # --- Minimum trade enforcement ---
        await self._force_minimum_spot_trade(signals, prices)

        # --- Manage open spot positions (SL, TP, trailing stop) ---
        async with self._position_lock:
            await self._manage_open_positions(signals, prices)

        # --- Snapshot ---
        nav = self.portfolio.nav_usd()
        record_snapshot(
            nav_usd=nav,
            peak_nav=self.portfolio._peak_nav,
            drawdown=self.portfolio.current_drawdown(),
            positions=self.portfolio.to_json(),
        )

        # --- Iteration log ---
        total_return = (nav - self.portfolio.initial_capital) / self.portfolio.initial_capital
        log.info(
            "iteration_end",
            nav=round(nav, 2),
            peak_nav=round(self.portfolio._peak_nav, 2),
            drawdown=f"{self.portfolio.current_drawdown():.2%}",
            total_return=f"{total_return:.2%}",
            positions=len(self.portfolio._positions),
        )
        intraday_ready = any(s.using_intraday for s in signals) if signals else False
        top_sig = ranked[0] if ranked else None
        record_iteration_log(
            iteration=self._iteration,
            regime=regime.regime.value,
            fear_greed=regime.fear_greed,
            nav_usd=nav,
            total_return_pct=total_return,
            drawdown_pct=self.portfolio.current_drawdown(),
            tradeable_count=len(ranked),
            top_symbol=top_sig.symbol if top_sig else "none",
            top_confidence=top_sig.confidence if top_sig else 0.0,
            action=_last_action,
            action_symbol=_last_action_symbol,
            action_reason=_last_action_reason,
            intraday_ready=intraday_ready,
            calls_used=est_calls,
        )

        # Reflexion step
        self._reflexion_step()

    async def _sync_live_portfolio(self) -> None:
        """On startup in live mode, sync real account balance from TWAK."""
        if self.twak is None:
            return
        try:
            equity = await self.twak.get_equity()
            if equity > 0:
                self.portfolio = Portfolio.from_live_equity(equity)
                log.info("live_portfolio_synced", total_usd=round(equity, 2))
            else:
                raise ValueError("TWAK returned 0 balance")
        except Exception as e:
            log.error("live_portfolio_sync_failed", error=str(e))
            self._running = False
            raise RuntimeError(f"Cannot sync live portfolio: {e}")

    async def _sync_equity_from_twak(self) -> None:
        """Periodically sync live portfolio equity."""
        if self.twak is None:
            return
        try:
            equity = await self.twak.get_equity()
            if equity <= 0:
                return

            current_nav = self.portfolio.nav_usd()
            change_pct = abs(equity - current_nav) / max(current_nav, 1) * 100

            if change_pct > 5.0:
                delta = equity - current_nav
                log.warning("equity_sync_drift", local_nav=current_nav, real_equity=equity, delta=delta)
                self.portfolio._cash_usd = max(0.0, equity - sum(p.value_usd for p in self.portfolio._positions.values()))
                if equity > self.portfolio.initial_capital:
                    self.portfolio.initial_capital = equity
                    self.portfolio._peak_nav = max(self.portfolio._peak_nav, equity)
        except Exception as e:
            log.warning("equity_sync_failed", error=str(e))

    async def _fetch_and_score_tokens(self, cmc: CMCClient, watchlist: list) -> list[TokenSignal]:
        """Fetch quotes and technical analysis from CoinMarketCap API and score tokens."""
        symbols = [t.symbol for t in watchlist]
        
        # 1. Resolve IDs
        id_results = await asyncio.gather(
            *[resolve_cmc_id(cmc, sym) for sym in symbols],
            return_exceptions=True,
        )
        symbol_to_id = {
            sym: cid for sym, cid in zip(symbols, id_results)
            if isinstance(cid, int)
        }
        
        if not symbol_to_id:
            return []

        # 2. Get Quotes
        try:
            quotes = await cmc.get_quotes(list(symbol_to_id.values()))
        except Exception as e:
            log.error("quotes_fetch_failed", error=str(e))
            return []

        # Extract latest prices & record to history
        prices_map = {}
        for sym, cid in symbol_to_id.items():
            from vicent.signals.signals import _extract_quote
            q = _extract_quote(quotes, cid)
            if q and "price" in q:
                prices_map[sym] = q["price"]

        record_prices(prices_map)

        # 3. Fetch Technical Analysis & news/whale if scheduled
        ta_results = {}
        if self._scheduler.should_fetch_token_ta():
            # Parallel query to avoid bottleneck
            ta_tasks = {sym: cmc.get_technical_analysis(cid) for sym, cid in symbol_to_id.items()}
            keys = list(ta_tasks.keys())
            results = await asyncio.gather(*ta_tasks.values(), return_exceptions=True)
            for sym, res in zip(keys, results):
                if not isinstance(res, Exception):
                    ta_results[sym] = res
                    self._cached_token_ta[sym] = res
        
        news_map = {}
        if self._scheduler.should_fetch_news():
            news_tasks = {sym: cmc.get_latest_news(cid) for sym, cid in symbol_to_id.items()}
            keys = list(news_tasks.keys())
            results = await asyncio.gather(*news_tasks.values(), return_exceptions=True)
            for sym, res in zip(keys, results):
                if not isinstance(res, Exception) and isinstance(res, list):
                    news_map[sym] = res
                    self._cached_news[sym] = res

        whale_map: dict[str, dict] = {}
        if self._scheduler.should_fetch_whale():
            whale_tasks = {sym: cmc.get_crypto_metrics(cid) for sym, cid in symbol_to_id.items()}
            keys = list(whale_tasks.keys())
            results = await asyncio.gather(*whale_tasks.values(), return_exceptions=True)
            for sym, res in zip(keys, results):
                if not isinstance(res, Exception) and isinstance(res, dict):
                    whale_map[sym] = res
                    self._cached_whale[sym] = res

        # 4. Score each token
        scored: list[TokenSignal] = []
        for sym, cid in symbol_to_id.items():
            price = prices_map.get(sym, 0.0)
            if price <= 0:
                continue

            ta    = ta_results.get(sym) or self._cached_token_ta.get(sym, {})
            news  = news_map.get(sym)   or self._cached_news.get(sym)
            whale = whale_map.get(sym)  or self._cached_whale.get(sym)

            # Intraday indicator calculation based on SQLite OHLCV history.
            # record_prices() writes high=low=close=price so ATR is close-only.
            # record_candles() (when available) writes real OHLC bars for better accuracy.
            ohlcv = get_ohlcv_series(sym)
            closes  = ohlcv["close"]
            highs   = ohlcv["high"]
            lows    = ohlcv["low"]
            volumes = ohlcv["volume"]

            has_real_hl = (
                highs and lows
                and any(h > 0 for h in highs)
                and any(l > 0 for l in lows)
                # real OHLC bars have high != close on at least some bars
                and any(abs(highs[i] - closes[i]) > 1e-9 for i in range(len(closes)))
            )

            intraday = None
            if closes and len(closes) >= 14:
                intraday = compute_intraday(
                    prices=closes,
                    volumes=volumes if any(v > 0 for v in volumes) else None,
                    highs=highs   if has_real_hl else None,
                    lows=lows     if has_real_hl else None,
                )

            # Build OHLCV candle list for pattern detection when real data available
            ohlcv_candles = None
            if has_real_hl and len(closes) >= 4:
                ohlcv_candles = [
                    {
                        "o": ohlcv["open"][i],
                        "h": highs[i],
                        "l": lows[i],
                        "c": closes[i],
                        "v": volumes[i],
                    }
                    for i in range(len(closes))
                ]

            try:
                sig = score_token(
                    symbol=sym,
                    cmc_id=cid,
                    quotes=quotes,
                    ta=ta,
                    news_articles=news,
                    whale_metrics=whale,
                    intraday=intraday,
                    price_series=closes,
                    ohlcv_candles=ohlcv_candles,
                )
                scored.append(sig)
            except Exception as e:
                log.warning("score_error", symbol=sym, error=str(e))

        return scored

    async def _execute_buy(
        self,
        decision: TradeDecision,
        prices: dict[str, float],
        signals: list[TokenSignal],
        sentiment_multiplier: float = 1.0,
    ) -> bool:
        symbol = decision.symbol
        price = prices.get(symbol, 0.0)
        if price <= 0:
            return False

        sig = next((s for s in signals if s.symbol == symbol), None)
        if sig is None:
            return False

        skip, fee_reason = should_skip_for_fees(
            expected_move_pct=sig.volatility_pct,
            confidence=decision.confidence,
        )
        if skip:
            return False

        size = self.risk.compute_size(decision.confidence, self.portfolio, sig.volume_24h)
        if not size.allowed:
            return False

        nav_fraction = round(size.nav_fraction * sentiment_multiplier, 4)
        # Cap at per-trade max but do NOT apply a percentage floor — doing so
        # defeats the defense layer (0.4x) and drawdown-aware scaling from risk.py.
        # Instead use a $5 USD minimum to skip dust trades only.
        nav_fraction = min(nav_fraction, self.cfg.risk_per_trade_nav_pct)
        trade_usd = self.portfolio.nav_usd() * nav_fraction
        if trade_usd < 5.0:
            log.debug("trade_skipped_dust", symbol=symbol, trade_usd=round(trade_usd, 2))
            return False

        ok, corr_reason = correlation_check(self.portfolio, symbol, trade_usd)
        if not ok:
            return False

        # On-chain execution
        tx_hash = None
        if self.cfg.is_live and self.twak:
            res = await self.twak.swap(
                from_token="USDT",
                to_token=symbol,
                amount_usd=trade_usd,
                slippage=self.cfg.slippage_pct * 100,
            )
            if not res.success:
                log.error("onchain_buy_failed", symbol=symbol, error=res.error)
                return False
            tx_hash = res.tx_hash
            if res.fill_price:
                price = res.fill_price

        qty = trade_usd / price
        self.portfolio.open_position(symbol, qty, price, atr_pct=sig.volatility_pct)
        record_trade(
            symbol=symbol,
            direction="buy",
            amount_usd=trade_usd,
            price_usd=price,
            quantity=qty,
            mode=self.cfg.vicent_mode.value,
            tx_hash=tx_hash,
            confidence=decision.confidence,
            regime=decision.regime.value,
        )
        # Store entry metadata for Reflexion autopsy on close
        self._entry_meta[symbol] = {
            "sub_scores": sig.sub_scores,
            "confidence": decision.confidence,
            "regime": decision.regime.value,
            "direction": "long",
        }
        return True

    async def _execute_sell(
        self,
        symbol: str,
        price: float,
        reason: str,
        fraction: float = 1.0,
        confidence: float = 0.0,
    ) -> bool:
        pos = self.portfolio.get_position(symbol)
        if pos is None or pos.quantity <= 0:
            return False

        qty = pos.quantity * fraction
        tx_hash = None

        if self.cfg.is_live and self.twak:
            res = await self.twak.swap(
                from_token=symbol,
                to_token="USDT",
                amount=qty,
                slippage=self.cfg.slippage_pct * 100,
            )
            if not res.success:
                log.error("onchain_sell_failed", symbol=symbol, error=res.error)
                return False
            tx_hash = res.tx_hash
            if res.fill_price:
                price = res.fill_price

        entry_price = pos.avg_cost_usd
        entry_meta = self._entry_meta.get(symbol, {})
        proceeds = self.portfolio.close_position(symbol, price, fraction=fraction)
        pnl_pct = (price - entry_price) / entry_price if entry_price > 0 else 0.0

        record_trade(
            symbol=symbol,
            direction="sell",
            amount_usd=proceeds,
            price_usd=price,
            quantity=qty,
            mode=self.cfg.vicent_mode.value,
            tx_hash=tx_hash,
            confidence=confidence,
            regime=None,
        )

        if fraction >= 1.0:
            self._closed_trade_pnls.append(pnl_pct)
            if len(self._closed_trade_pnls) > 10:
                self._closed_trade_pnls.pop(0)
            # --- Reflexion autopsy: learn from this closed trade ---
            if entry_meta:
                try:
                    self._reflexion.process_closed_trade(
                        symbol=symbol,
                        direction=entry_meta.get("direction", "long"),
                        pnl_pct=pnl_pct,
                        confidence_at_entry=entry_meta.get("confidence", confidence),
                        sub_scores=entry_meta.get("sub_scores", {}),
                        regime=entry_meta.get("regime", "unknown"),
                    )
                except Exception as rfx_err:
                    log.warning("reflexion_autopsy_error", symbol=symbol, error=str(rfx_err))
            # Clean up entry metadata after full close
            self._entry_meta.pop(symbol, None)

        log.info("sell_executed", symbol=symbol, reason=reason, fraction=fraction, pnl_pct=f"{pnl_pct:.2%}", proceeds=proceeds)
        return True

    async def _manage_open_positions(self, signals: list[TokenSignal], prices: dict[str, float]) -> None:
        """Check all open spot positions for exit conditions."""
        sig_map = {s.symbol: s for s in signals}

        for symbol, pos in list(self.portfolio._positions.items()):
            price = prices.get(symbol, pos.current_price_usd)
            if price <= 0 or pos.avg_cost_usd <= 0:
                continue

            pnl_pct = pos.unrealized_pnl_pct
            peak_pnl_pct = pos.peak_pnl_pct
            targets = volatility_aware_targets(pos.entry_atr_pct)

            sig = sig_map.get(symbol)
            signal_reversed = (
                sig is not None
                and sig.direction == Direction.SHORT
                and sig.confidence >= 0.60
            )

            plan = evaluate_exit(
                pnl_pct=pnl_pct,
                peak_pnl_pct=peak_pnl_pct,
                targets=targets,
                scaled_out_t1=pos.scaled_out_t1,
                scaled_out_t2=pos.scaled_out_t2,
                signal_reversed=signal_reversed,
            )

            if plan.action == ExitAction.HOLD:
                continue

            log.info("exit_planned", symbol=symbol, action=plan.action.value, fraction=plan.fraction, pnl_pct=f"{pnl_pct:.2%}")

            sold = await self._execute_sell(
                symbol=symbol,
                price=price,
                reason=plan.reason,
                fraction=plan.fraction,
                confidence=sig.confidence if sig else 0.0,
            )

            if sold and plan.fraction < 1.0:
                still = self.portfolio.get_position(symbol)
                if still is not None:
                    if plan.action == ExitAction.SCALE_OUT_T1:
                        still.scaled_out_t1 = True
                    elif plan.action == ExitAction.SCALE_OUT_T2:
                        still.scaled_out_t2 = True

    async def _force_minimum_spot_trade(self, signals: list[TokenSignal], prices: dict[str, float]) -> None:
        """Ensure we satisfy the daily trade requirement by making a small trade if needed."""
        # Calculate daily trades from db
        trades_today = get_trades_today()
        # Calculate fractional hours remaining until UTC midnight
        now = datetime.now(timezone.utc)
        hours_remaining = 24 - now.hour - now.minute / 60.0

        if not self.risk.should_force_min_trade(len(trades_today), hours_remaining):
            return
        log.warning("forcing_minimum_spot_trade")
        eligible = [
            s for s in sorted(signals, key=lambda x: x.confidence, reverse=True)
            if s.confidence >= 0.45 and prices.get(s.symbol, 0) > 0 and not is_stable(s.symbol)
        ]

        if not eligible:
            log.warning("force_minimum_no_eligible_token")
            return

        sig = eligible[0]
        price = prices[sig.symbol]
        
        # If we have a position already, we sell it. If not, we buy a tiny amount ($5)
        if self.portfolio.has_position(sig.symbol):
            await self._execute_sell(sig.symbol, price, reason="forced_minimum_sell", fraction=1.0)
        else:
            # Fake a decision to buy $5 worth of the token
            tx_hash = None
            amount_usd = 5.0
            if self.cfg.is_live and self.twak:
                res = await self.twak.swap(
                    from_token="USDT",
                    to_token=sig.symbol,
                    amount_usd=amount_usd,
                )
                if res.success:
                    tx_hash = res.tx_hash
                    if res.fill_price:
                        price = res.fill_price
                else:
                    return

            qty = amount_usd / price
            self.portfolio.open_position(sig.symbol, qty, price, atr_pct=sig.volatility_pct)
            record_trade(
                symbol=sig.symbol,
                direction="buy",
                amount_usd=amount_usd,
                price_usd=price,
                quantity=qty,
                mode=self.cfg.vicent_mode.value,
                tx_hash=tx_hash,
                confidence=sig.confidence,
                regime="forced_minimum_buy",
            )
            # Store entry metadata for Reflexion autopsy
            self._entry_meta[sig.symbol] = {
                "sub_scores": sig.sub_scores,
                "confidence": sig.confidence,
                "regime": "forced_minimum_buy",
                "direction": "long",
            }
            log.info("forced_minimum_spot_buy_executed", symbol=sig.symbol, amount=amount_usd)

    def _reflexion_step(self) -> None:
        """Adjust confidence threshold based on recent trade performance.

        This is called every iteration. Heavy per-trade learning (signal bias,
        autopsy, symbol cooldown) happens inside _execute_sell via
        self._reflexion.process_closed_trade(). This function handles the
        global confidence threshold adjustment only.
        """
        if len(self._closed_trade_pnls) < 3:
            return

        recent = self._closed_trade_pnls[-5:]
        avg_pnl = sum(recent) / len(recent)
        win_rate = sum(1 for p in recent if p > 0) / len(recent)

        if win_rate < 0.40 or avg_pnl < -0.03:
            self._min_confidence = min(0.75, self._min_confidence + 0.03)
        elif win_rate >= 0.65 and avg_pnl > 0.02:
            self._min_confidence = max(0.45, self._min_confidence - 0.02)

        log.info(
            "reflexion_threshold_updated",
            min_confidence=round(self._min_confidence, 3),
            recent_win_rate=round(win_rate, 2),
            avg_pnl=round(avg_pnl, 4),
            total_autopsies=self._reflexion.state.total_autopsies,
            overall_win_rate=round(self._reflexion.state.overall_win_rate, 3),
        )
