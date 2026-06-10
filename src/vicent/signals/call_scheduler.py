"""CMC API call scheduler — stay within Free tier (15,000 credits/month).

New architecture: Spot Swaps on BSC, CMC is primary indicator & price source.

  CMC EVERY iteration (5 min):
    - get_global_metrics_latest ← Fear & Greed, market cap, regime
    - get_crypto_quotes_latest ← Price & Volume updates

  CMC EVERY 4 iterations (20 min):
    - get_global_crypto_derivatives_metrics  ← funding, OI total market
    - trending_crypto_narratives             ← hot narrative themes
    - get_upcoming_macro_events              ← Fed, SEC, macro events
    - get_crypto_marketcap_technical_analysis ← TA total market cap

  CMC EVERY 12 iterations (60 min):
    - get_crypto_latest_news × 23  ← news articles

Budget CMC: ~3 calls/iteration average → ~864 calls/day → safe for Free tier.
"""

from __future__ import annotations


class CallScheduler:
    """Manages frequency of CMC API calls."""

    def __init__(self) -> None:
        self._iteration = 0

    def tick(self, iteration: int) -> None:
        self._iteration = iteration

    # ------------------------------------------------------------------
    # CMC: Always (mỗi vòng 5 phút)
    # ------------------------------------------------------------------

    def should_fetch_global_metrics(self) -> bool:
        """Fear & Greed + global market cap — cần fresh mỗi vòng."""
        return True

    # ------------------------------------------------------------------
    # CMC: Every 4 iterations (20 min)
    # ------------------------------------------------------------------

    def should_fetch_derivatives(self) -> bool:
        """Global derivatives metrics from CMC (for reference)."""
        return self._iteration <= 1 or self._iteration % 4 == 0

    def should_fetch_marketcap_ta(self) -> bool:
        """TA of the total market capitalization - changes slowly."""
        return self._iteration % 4 == 0

    def should_fetch_narratives(self) -> bool:
        """Trending crypto narratives - changes slowly."""
        return self._iteration % 4 == 1  # offset

    def should_fetch_events(self) -> bool:
        """Macro events - changes very slowly."""
        return self._iteration % 4 == 2  # offset

    # ------------------------------------------------------------------
    # CMC: Every 12 iterations (60 min)
    # ------------------------------------------------------------------

    def should_fetch_news(self) -> bool:
        """Per-token news - slow changes, high call cost."""
        return self._iteration % 12 == 0

    # ------------------------------------------------------------------
    # CMC Data Sources
    # ------------------------------------------------------------------

    def should_fetch_quotes(self) -> bool:
        """Token prices - fetch quotes from CMC."""
        return True

    def should_fetch_token_ta(self) -> bool:
        """TA per-token - fetch indicators from CMC."""
        return self._iteration % 2 == 0  # Fetch every 10 min to save budget

    def should_fetch_whale(self) -> bool:
        """Whale data."""
        return self._iteration % 4 == 0

    # ------------------------------------------------------------------
    # Budget accounting
    # ------------------------------------------------------------------

    def estimate_calls_this_iteration(self, n_tokens: int = 23) -> int:
        """Estimate the number of CMC API calls in this loop iteration."""
        calls = 1  # global_metrics (always)
        if self.should_fetch_derivatives():    calls += 1
        if self.should_fetch_marketcap_ta():   calls += 1
        if self.should_fetch_narratives():     calls += 1
        if self.should_fetch_events():         calls += 1
        if self.should_fetch_news():           calls += n_tokens
        return calls

    def estimate_daily_calls(self, interval_sec: int) -> int:
        """Estimate expected daily CMC calls."""
        iters = 86400 // interval_sec
        # Average over 12-iteration cycle
        avg = (
            1 * 12 +      # global_metrics per iteration
            1 * 3 +       # derivatives every 4 iterations
            1 * 3 +       # marketcap_ta every 4 iterations
            1 * 3 +       # narratives every 4 iterations
            1 * 3 +       # events every 4 iterations
            23 * 1        # news every 12 iterations
        ) / 12
        return round(avg * iters)

    def check_budget_warning(self, interval_sec: int, monthly_limit: int = 15000) -> str | None:
        daily = self.estimate_daily_calls(interval_sec)
        monthly = daily * 30
        if monthly > monthly_limit:
            days = monthly_limit / daily
            return (
                f"API BUDGET WARNING: ~{daily} calls/day × 30 = {monthly:,} "
                f"exceeds {monthly_limit:,} free tier limit. "
                f"Credits will exhaust in ~{days:.1f} days."
            )
        return None
