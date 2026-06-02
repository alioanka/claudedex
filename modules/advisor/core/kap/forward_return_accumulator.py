"""
Forward Return Accumulator -- nightly job to fill T+N returns.

ADVICE-ONLY. This module computes forward price returns for stored KAP
disclosures. No trading signals, no order execution.

Purpose
-------
For each kap_disclosures row, after T+1, T+3, T+5, T+10, T+30 TRADING DAYS
have elapsed since the disclosure, fetch the ticker's BIST closing price for
that day and compute the return vs the anchor price.

Intraday-Anchor Convention
--------------------------
T+0 anchor = NEXT TRADING SESSION OPEN after the disclosure's Istanbul-time
timestamp (UTC+3).

Rationale:
  - A disclosure at 17:50 Istanbul time is AFTER market close (BIST closes
    18:00). The next actionable price is the NEXT DAY's open.
  - A disclosure at 09:05 Istanbul time is during market hours, but using
    the intraday price between 09:05 and 09:30 as T+0 would be look-ahead
    bias for any strategy that receives the disclosure at the same moment
    the market opens. For consistency, we use the NEXT SESSION open for ALL
    disclosures regardless of intraday timing.
  - This means return_1d = (close at T+1 session) / (open at T+1 session) - 1
    where T+1 session = the session immediately after the anchor session.

    Wait -- more precisely:
      anchor_price = open price of the FIRST FULL TRADING SESSION after
                     the disclosure (next-session open).
      return_Nd   = (close price at T+N trading sessions after anchor session)
                    / anchor_price - 1

Price data source
-----------------
Reuses BISTAnalyzer's fetch helpers:
  1. borsapy (AVAILABLE, ~15min delayed, TradingView-backed)
  2. yfinance (.IS suffix, DEGRADED fallback)

Both are called via _fetch_ohlcv_sync (blocking, run in executor).

BIST trading calendar
---------------------
Turkey's official stock exchange holidays are NOT embedded here (too fragile
to hardcode). Trading-day counting uses a simple calendar approximation:
  - weekdays (Mon-Fri) are assumed to be trading days
  - Saturday / Sunday are excluded

For precise T+N, actual price data availability is the ground truth:
if the price for day D is missing (NaN / empty), we look forward up to
3 extra calendar days. If still not found, the return stays NULL until
the next accumulator run.

Scheduling
----------
This is a daily async task, typically run at 06:00 Istanbul time (03:00 UTC)
after overnight data updates settle. It is wired into AdvisorApplication's
run loop ONLY if advisor_kap_enabled=true.

Wire-in (in main_advisor.py, inside AdvisorApplication.run()):
    if config.get("advisor_kap_enabled") == "true":
        from modules.advisor.core.kap.forward_return_accumulator import (
            ReturnAccumulator
        )
        accumulator = ReturnAccumulator(config=config, db_pool=pool)
        asyncio.create_task(accumulator.run_daily())

Fail-soft: any single ticker/disclosure failure is logged and skipped.
The job never raises; it logs a WARNING and continues.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

from modules.advisor.core.kap.kap_store import (
    get_disclosures_needing_returns,
    upsert_return_row,
)

logger = logging.getLogger("advisor.kap.accumulator")

# Trading day approximation: max lookforward days when a session price is missing
_MAX_LOOKFORWARD = 3

# Return windows in trading days
_DEFAULT_WINDOWS = [1, 3, 5, 10, 30]


# ---------------------------------------------------------------------------
# Price fetch helpers (reuse bist.py patterns exactly)
# ---------------------------------------------------------------------------

def _fetch_ohlcv_sync(ticker: str, start: date, end: date) -> Optional[object]:
    """
    Fetch BIST daily OHLCV as a DataFrame for ticker between start..end.
    Tries borsapy first (AVAILABLE), falls back to yfinance (DEGRADED).
    Returns None on any failure.
    Runs in executor (sync).

    This mirrors BISTAnalyzer._fetch_borsapy/_fetch_yfinance exactly --
    it reuses the same libraries rather than re-implementing price fetching.
    """
    import pandas as pd

    # Extend end by a few days to ensure we have enough rows for T+30
    fetch_end = end + timedelta(days=5)

    # --- borsapy (AVAILABLE) ---
    try:
        from borsapy import Client as BorsapyClient
        bare = ticker.upper().replace(".IS", "")
        # borsapy period string based on date range
        days = (fetch_end - start).days
        if days <= 30:
            period = "1mo"
        elif days <= 90:
            period = "3mo"
        elif days <= 180:
            period = "6mo"
        elif days <= 365:
            period = "1y"
        else:
            period = "2y"
        client = BorsapyClient()
        df = client.get_history(bare, period=period, interval="1d")
        if df is not None and not df.empty:
            df = df.copy()
            df.columns = [str(c).lower() for c in df.columns]
            df = df.sort_index()
            # Filter to requested range
            try:
                df = df.loc[str(start):str(fetch_end)]
            except Exception:
                pass
            if not df.empty and "close" in df.columns:
                return df
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("[accumulator] borsapy fetch %r failed: %s", ticker, exc)

    # --- yfinance fallback (DEGRADED) ---
    try:
        import yfinance as yf
        yf_symbol = ticker.upper()
        if not yf_symbol.endswith(".IS"):
            yf_symbol += ".IS"
        t = yf.Ticker(yf_symbol)
        df = t.history(
            start=start.strftime("%Y-%m-%d"),
            end=fetch_end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )
        if df is not None and not df.empty:
            df = df.copy()
            df.columns = [str(c).lower() for c in df.columns]
            df = df.sort_index()
            return df
    except ImportError:
        pass
    except Exception as exc:
        logger.debug("[accumulator] yfinance fetch %r failed: %s", ticker, exc)

    return None


def _detect_price_source() -> str:
    """Return 'borsapy' if importable, else 'yfinance'."""
    try:
        import borsapy  # noqa: F401
        return "borsapy"
    except ImportError:
        return "yfinance"


# ---------------------------------------------------------------------------
# Trading day helpers
# ---------------------------------------------------------------------------

def _next_trading_day(dt: datetime) -> date:
    """
    Return the next calendar weekday (Mon-Fri) after a given datetime.
    Istanbul timezone (UTC+3) is used to determine 'after market close'.
    BIST closes at 18:00 Istanbul time.

    Convention: if the disclosure is BEFORE 18:00 Istanbul time on a weekday,
    the anchor is the SAME day (but we use next-session open consistently,
    so we still advance one trading day to avoid intraday look-ahead).
    Result: always the NEXT trading day regardless of intraday time.
    """
    istanbul_tz = timezone(timedelta(hours=3))
    local_dt = dt.astimezone(istanbul_tz)
    d = local_dt.date() + timedelta(days=1)
    # Skip weekends
    while d.weekday() >= 5:  # 5=Sat, 6=Sun
        d += timedelta(days=1)
    return d


def _add_trading_days(start_date: date, n: int) -> date:
    """Add n trading days (Mon-Fri) to start_date."""
    d = start_date
    added = 0
    while added < n:
        d += timedelta(days=1)
        if d.weekday() < 5:  # Mon-Fri
            added += 1
    return d


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------

def _compute_returns_sync(ticker: str, disclosed_at: datetime,
                           windows: List[int]) -> Tuple[Optional[float], Optional[date],
                                                        Dict[str, Optional[float]], str]:
    """
    Synchronous: compute forward returns for a single (disclosure, ticker).

    Returns (anchor_price, anchor_date, returns_dict, price_source).
    returns_dict: {'1d': float|None, '3d': float|None, ...}
    anchor_price: None if we can't get price data yet.
    """
    anchor_trade_date = _next_trading_day(disclosed_at)
    today = datetime.now(timezone.utc).date()

    max_window = max(windows) if windows else 30
    fetch_start = anchor_trade_date - timedelta(days=2)  # buffer for data
    fetch_end   = _add_trading_days(anchor_trade_date, max_window + _MAX_LOOKFORWARD)

    if fetch_end > today:
        fetch_end = today

    df = _fetch_ohlcv_sync(ticker, fetch_start, fetch_end)
    if df is None or df.empty:
        return None, anchor_trade_date, {}, _detect_price_source()

    price_source = _detect_price_source()

    # Find anchor price: open on anchor_trade_date (or next available trading day)
    def _get_open(target: date) -> Optional[float]:
        for offset in range(_MAX_LOOKFORWARD + 1):
            d = target + timedelta(days=offset)
            d_str = str(d)
            try:
                row = df.loc[d_str] if d_str in df.index.astype(str).tolist() else None
                if row is not None:
                    v = row.get("open") if hasattr(row, "get") else getattr(row, "open", None)
                    if v is not None and v == v and v > 0:  # not NaN
                        return float(v)
            except Exception:
                pass
            # Also try index-based lookup
            try:
                mask = df.index.date == d
                sub = df[mask]
                if not sub.empty and "open" in sub.columns:
                    v = float(sub["open"].iloc[0])
                    if v > 0:
                        return v
            except Exception:
                pass
        return None

    def _get_close(target: date) -> Optional[float]:
        for offset in range(_MAX_LOOKFORWARD + 1):
            d = target + timedelta(days=offset)
            try:
                mask = df.index.date == d
                sub = df[mask]
                if not sub.empty and "close" in sub.columns:
                    v = float(sub["close"].iloc[0])
                    if v > 0:
                        return v
            except Exception:
                pass
        return None

    anchor_price = _get_open(anchor_trade_date)
    if anchor_price is None or anchor_price <= 0:
        return None, anchor_trade_date, {}, price_source

    returns: Dict[str, Optional[float]] = {}
    for w in windows:
        target_close_date = _add_trading_days(anchor_trade_date, w)
        if target_close_date > today:
            returns[str(w) + "d"] = None  # window not yet matured
            continue
        close = _get_close(target_close_date)
        if close is None:
            returns[str(w) + "d"] = None
        else:
            returns[str(w) + "d"] = round((close - anchor_price) / anchor_price, 6)

    return anchor_price, anchor_trade_date, returns, price_source


# ---------------------------------------------------------------------------
# ReturnAccumulator class
# ---------------------------------------------------------------------------

class ReturnAccumulator:
    """
    Daily job: fill forward returns for matured disclosure windows.

    Usage:
        accumulator = ReturnAccumulator(config=config, db_pool=pool)
        asyncio.create_task(accumulator.run_daily())

    Runs at 06:00 Istanbul time (03:00 UTC) each day.
    On first start, runs immediately then schedules daily.
    """

    def __init__(self, config: dict, db_pool=None):
        self.config = config
        self.db_pool = db_pool

    @property
    def enabled(self) -> bool:
        return str(self.config.get("advisor_kap_enabled", "false")).lower() == "true"

    def _get_windows(self) -> List[int]:
        raw = self.config.get("kap_return_windows", "1,3,5,10,30")
        try:
            return [int(x.strip()) for x in raw.split(",") if x.strip()]
        except ValueError:
            return _DEFAULT_WINDOWS

    async def run_daily(self) -> None:
        """
        Run the accumulator loop:
        - Run once immediately on startup.
        - Then sleep until 06:00 Istanbul (03:00 UTC) and run daily.
        Stops cleanly on CancelledError.
        """
        if not self.enabled:
            logger.info("[accumulator] advisor_kap_enabled=false -- accumulator not running.")
            return

        logger.info("[accumulator] Starting forward return accumulator.")
        try:
            # Run immediately on startup
            await self._run_once()
            while True:
                await self._sleep_until_next_run()
                await self._run_once()
        except asyncio.CancelledError:
            logger.info("[accumulator] Stopped (cancelled).")

    async def _sleep_until_next_run(self) -> None:
        """Sleep until 03:00 UTC (06:00 Istanbul) tomorrow."""
        now = datetime.now(timezone.utc)
        target = now.replace(hour=3, minute=0, second=0, microsecond=0)
        if target <= now:
            target += timedelta(days=1)
        sleep_s = (target - now).total_seconds()
        logger.info("[accumulator] Next run in %.1fh (at %s UTC).",
                    sleep_s / 3600, target.strftime("%Y-%m-%d %H:%M"))
        await asyncio.sleep(sleep_s)

    async def _run_once(self) -> None:
        """Process all disclosures needing returns."""
        if not self.enabled or self.db_pool is None:
            return

        windows = self._get_windows()
        logger.info("[accumulator] Run starting (windows=%s).", windows)

        try:
            pending = await get_disclosures_needing_returns(
                self.db_pool, max_age_days=max(windows) + 10
            )
        except Exception as exc:
            logger.warning("[accumulator] Could not query pending disclosures: %s", exc)
            return

        logger.info("[accumulator] %d disclosures to process.", len(pending))
        loop = asyncio.get_event_loop()
        processed = 0
        errors = 0

        for row in pending:
            disc_id   = row.get("disclosure_id", "")
            ticker    = row.get("ticker")
            disc_time = row.get("disclosed_at")

            if not ticker or not disc_time:
                continue

            try:
                anchor_price, anchor_date, returns, price_src = \
                    await loop.run_in_executor(
                        None, _compute_returns_sync, ticker, disc_time, windows
                    )
                # Build returns dict with string keys matching DB column suffix
                # e.g. {'1d': 0.015, '3d': None, ...}
                await upsert_return_row(
                    self.db_pool,
                    disclosure_id=disc_id,
                    ticker=ticker,
                    anchor_price=anchor_price,
                    anchor_date=anchor_date,
                    returns=returns,
                    price_source=price_src,
                )
                processed += 1
            except Exception as exc:
                logger.warning("[accumulator] Failed for %s/%s: %s",
                               disc_id, ticker, exc)
                errors += 1

        logger.info("[accumulator] Run complete: %d processed, %d errors.",
                    processed, errors)
