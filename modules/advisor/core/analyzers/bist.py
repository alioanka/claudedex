"""
BISTAnalyzer — Borsa Istanbul equities analyzer.

Data source priority (2026-06-04):

  0. Fonoloji API (OPTIONAL, PREFERRED when ADVISOR_FONOLOJI_API_KEY present) —
     AVAILABLE. https://fonoloji.com/v1 via the shared FonolojiClient
     (X-API-Key). GET /stocks/{ticker}/chart?period=1d|5d|1mo|3mo|6mo|1y|5y
     returns a TRY-denominated price series; OHLC synthesised from close where
     only close is provided. Preferred over the DEGRADED yfinance .IS path when
     a key is present. Enable: advisor_bist_data_source='fonoloji' OR just set
     the key (auto-preferred when the source is unset/''). FAIL-SOFT: no key /
     error => unchanged (borsapy/yfinance). Cached daily by the client.

  1. borsapy (PRIMARY) — AVAILABLE
     TradingView-backed library providing BIST stocks/indices with
     ~15-min delayed data (same as TradingView free tier).
     Install: pip install borsapy
     Import-guarded: if not installed, falls through to yfinance fallback.
     advisor_bist_data_source='borsapy'  — explicit
     advisor_bist_data_source=''         — auto-detect: borsapy tried first
     Watchlist: watchlist_bist='THYAO,EREGL,GARAN,AKBNK' (bare tickers)

  2. yfinance (.IS suffix) — DEGRADED fallback
     Incomplete coverage (mid/small caps often missing), ~15-20 min delayed.
     Falls back automatically when borsapy is unavailable/returning no data.
     advisor_bist_data_source='yfinance'  — explicit
     Watchlist: watchlist_bist='THYAO.IS,EREGL.IS,GARAN.IS,AKBNK.IS'

  3. Matriks / Rasyonet / IsYatirim — STUBBED (paid, Wave-23+)
     Full BIST coverage, real-time data. Requires ADVISOR_BIST_API_KEY
     in Secure Credentials. Returns NOT_CONFIGURED until wired.
     advisor_bist_data_source='matriks' + ADVISOR_BIST_API_KEY set

data_source_status:
  borsapy importable (any config incl. empty)   → AVAILABLE
  borsapy missing, yfinance importable           → DEGRADED
  matriks + key set                              → AVAILABLE (stub, NOT_CONFIGURED result)
  nothing importable                             → NOT_CONFIGURED

Technical indicators (identical suite to us_equities):
  SMA20/50, RSI-14, Bollinger Bands, volume ratio.

LLM rationale: shared rationale_helper.build_rationale().

Import guard: borsapy absence → NEVER crashes the advice loop.
              Falls through gracefully to yfinance DEGRADED path.

ADVICE-ONLY. analyze() never raises.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Optional

from modules.advisor.core.base_analyzer import BaseAnalyzer
from modules.advisor.core.models import (
    AdviceResult,
    DataSourceStatus,
    Direction,
    Horizon,
    Market,
)
from modules.advisor.core.rationale_helper import build_rationale
from modules.advisor.core.analyzers.levels import (
    horizon_levels,
    signal_confidence,
)

logger = logging.getLogger("advisor.analyzer.bist")

_LOOKBACK: dict = {
    Horizon.SHORT: 30,
    Horizon.MID: 180,
    Horizon.LONG: 730,
}

_NOT_CONFIGURED_NOTE = (
    "BIST data source not configured. "
    "Recommended (free, ~15min delayed): "
    "pip install borsapy  then set advisor_bist_data_source='borsapy'. "
    "Watchlist: watchlist_bist='THYAO,EREGL,GARAN,AKBNK' (bare tickers). "
    "Degraded free fallback: set advisor_bist_data_source='yfinance'; "
    "watchlist uses .IS suffix (e.g. THYAO.IS). "
    "Full real-time coverage: set advisor_bist_data_source='matriks' and "
    "ADVISOR_BIST_API_KEY in Secure Credentials."
)

_MATRIKS_STUB_NOTE = (
    "Matriks BIST integration is configured (key present) but the API "
    "client is not yet wired (Wave-23 planned). "
    "Interim: remove advisor_bist_data_source='matriks' to fall back to "
    "borsapy (AVAILABLE, free) or yfinance (DEGRADED, free). "
    "To wire Matriks: implement _fetch_matriks() in bist.py."
)


from modules.advisor.core.data.fonoloji_client import (
    FonolojiClient,
    resolve_api_key as _fonoloji_api_key,
)
from modules.advisor.core.data.activation import (
    prefer_fonoloji_bist as _prefer_fonoloji,
)


def _flag(config: dict, key: str, default: bool) -> bool:
    """Read a boolean advisor_config flag tolerating bool or string values."""
    val = (config or {}).get(key, default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def _fonoloji_period_for_lookback(lookback_days: int) -> str:
    """Map advisor lookback (days) to a Fonoloji /stocks chart period token
    (1d|5d|1mo|3mo|6mo|1y|5y)."""
    if lookback_days <= 5:
        return "5d"
    if lookback_days <= 31:
        return "1mo"
    if lookback_days <= 95:
        return "3mo"
    if lookback_days <= 190:
        return "6mo"
    if lookback_days <= 380:
        return "1y"
    return "5y"


def _borsapy_available() -> bool:
    """Return True if borsapy can be imported (does NOT test connectivity)."""
    try:
        import borsapy  # noqa: F401
        return True
    except ImportError:
        return False


def _yfinance_available() -> bool:
    """Return True if yfinance can be imported."""
    try:
        import yfinance  # noqa: F401
        return True
    except ImportError:
        return False


class BISTAnalyzer(BaseAnalyzer):
    """
    Borsa Istanbul analyzer.

    Primary   : borsapy (TradingView-backed, ~15min delayed)   -> AVAILABLE
    Fallback  : yfinance .IS suffix (incomplete coverage, free)  -> DEGRADED
    Paid stub : Matriks (key required, not yet wired)            -> NOT_CONFIGURED result
    None      : NOT_CONFIGURED
    """

    market = Market.BIST

    def __init__(self, config: dict, db_pool=None):
        super().__init__(config, db_pool)
        self.last_price: dict = {}

    # ------------------------------------------------------------------
    # BaseAnalyzer contract
    # ------------------------------------------------------------------

    def data_source_status(self) -> DataSourceStatus:
        source = self.config.get("advisor_bist_data_source", "")

        # Fonoloji (AVAILABLE) — explicit 'fonoloji', legacy '', or AUTO-PREFER
        # (activation.prefer_fonoloji_bist): when a key is present and
        # advisor_fonoloji_auto_prefer is on (default true), Fonoloji is tried
        # FIRST even over a stale 'yfinance'/'borsapy' value. Explicit
        # 'matriks' opt-out wins. No key => fail-through to borsapy/yfinance.
        if _prefer_fonoloji(self.config):
            return DataSourceStatus.AVAILABLE

        # Paid path: key required
        if source == "matriks":
            import os
            api_key = (
                self.config.get("advisor_bist_api_key")
                or os.getenv("ADVISOR_BIST_API_KEY", "")
            )
            return DataSourceStatus.AVAILABLE if api_key else DataSourceStatus.NOT_CONFIGURED

        # borsapy: explicit OR auto-detect (empty config)
        if source in ("borsapy", "") and _borsapy_available():
            return DataSourceStatus.AVAILABLE

        # yfinance: explicit OR auto-detect fallback
        if source in ("yfinance", "") and _yfinance_available():
            return DataSourceStatus.DEGRADED

        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce an AdviceResult for a BIST equity.
        Never raises -- _error_result / _not_configured on any failure.
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)

        source = self.config.get("advisor_bist_data_source", "")

        # Paid path: key present but not wired yet
        if source == "matriks":
            return self._not_configured(symbol, horizon, _MATRIKS_STUB_NOTE)

        try:
            return await self._analyze_internal(symbol, horizon, status)
        except Exception as exc:
            return self._error_result(symbol, horizon, exc)

    # ------------------------------------------------------------------
    # Internal: try borsapy then fall through to yfinance
    # ------------------------------------------------------------------

    async def _analyze_internal(
        self, symbol: str, horizon: Horizon, status: DataSourceStatus
    ) -> AdviceResult:
        import asyncio

        configured_source = self.config.get("advisor_bist_data_source", "")
        signals = None
        actual_status = DataSourceStatus.NOT_CONFIGURED
        note = ""
        data_source_tag = ""

        # --- Fonoloji (AVAILABLE) — explicit 'fonoloji', legacy '', or
        # AUTO-PREFER (activation.prefer_fonoloji_bist; even over a stale
        # 'yfinance'/'borsapy' value). FAIL-SOFT: no data => fall through to
        # borsapy/yfinance below. Preferred over the DEGRADED yfinance .IS path.
        if _prefer_fonoloji(self.config):
            loop = asyncio.get_event_loop()
            signals = await loop.run_in_executor(
                None, self._fetch_fonoloji, symbol, horizon
            )
            if signals is not None:
                actual_status = DataSourceStatus.AVAILABLE
                data_source_tag = "fonoloji"
                note = (
                    "BIST data via Fonoloji API (fonoloji.com/v1). "
                    "TRY-denominated; cached daily."
                )
            elif configured_source == "fonoloji":
                self.logger.warning(
                    "[bist] Fonoloji returned no data for %s; falling back to "
                    "borsapy/yfinance.", symbol,
                )

        # --- borsapy (AVAILABLE) ---
        if signals is None and _borsapy_available() and configured_source in ("borsapy", ""):
            loop = asyncio.get_event_loop()
            signals = await loop.run_in_executor(
                None, self._fetch_borsapy, symbol, horizon
            )
            if signals is not None:
                actual_status = DataSourceStatus.AVAILABLE
                data_source_tag = "borsapy"
                note = (
                    "BIST data via borsapy (TradingView feed, ~15min delayed). "
                    "For paid real-time: set advisor_bist_data_source='matriks' "
                    "and ADVISOR_BIST_API_KEY in Secure Credentials."
                )
            else:
                self.logger.warning(
                    "[bist] borsapy returned no data for %s. "
                    "Falling back to yfinance (.IS suffix).",
                    symbol,
                )

        # --- yfinance fallback (DEGRADED) ---
        if signals is None and _yfinance_available():
            loop = asyncio.get_event_loop()
            signals = await loop.run_in_executor(
                None, self._fetch_yfinance, symbol, horizon
            )
            if signals is not None:
                actual_status = DataSourceStatus.DEGRADED
                data_source_tag = "yfinance"
                note = (
                    "BIST data via yfinance (degraded): partial coverage, "
                    "~15min delay. "
                    "For better coverage: pip install borsapy (free, ~15min). "
                    "For full real-time: set advisor_bist_data_source='matriks' "
                    "and ADVISOR_BIST_API_KEY in Secure Credentials."
                )

        if signals is None:
            return self._error_result(
                symbol, horizon,
                Exception(
                    f"All BIST data sources failed for '{symbol}'. "
                    "borsapy not installed or returned no data, yfinance also "
                    "failed. Check symbol format (borsapy: bare ticker e.g. "
                    "'THYAO'; yfinance: .IS suffix e.g. 'THYAO.IS') and "
                    "network connectivity."
                )
            )

        # --- Broker analyst-consensus vote (Fonoloji, FREE within quota) ---
        # A genuine directional signal not available from yfinance. Folded in as
        # ONE additional vote alongside the technical SMA/RSI/BB votes — it does
        # NOT dominate. FAIL-SOFT: no key / disabled / 404 / bad shape => the
        # technical signal is unchanged (vote=0). The raw summary is stashed in
        # extra['analyst_consensus'] for the dashboard.
        analyst_vote = 0
        analyst_summary = None
        if (
            _fonoloji_api_key(self.config)
            and _flag(self.config, "advisor_bist_use_analyst_recommendations", True)
        ):
            analyst_vote, analyst_summary = await self._fetch_analyst_consensus(
                symbol, signals.get("close")
            )

        direction = _signals_to_direction(signals, analyst_vote)
        confidence = signal_confidence(signals, horizon, config=self.config)
        entry_low, entry_high, target, stop = horizon_levels(
            signals, direction, horizon, config=self.config, price_decimals=2
        )

        self.last_price[symbol] = signals.get("close", 0)

        rationale = await build_rationale(
            symbol=symbol,
            market=self.market,
            horizon=horizon,
            signals=signals,
            direction=direction,
            config=self.config,
            caller_logger=self.logger,
        )

        model_id = self.config.get("advisor_anthropic_model", "claude-opus-4-5")
        extra = {"klines_df": signals.get("df"), "data_source": data_source_tag}
        if analyst_summary is not None:
            extra["analyst_consensus"] = analyst_summary

        # Light/optional AI market digest (default off). The shared client caches
        # it 6h so the per-symbol call is effectively one real fetch per cycle.
        # Stashed in extra['market_digest'] for a later dashboard hook. FAIL-SOFT:
        # 404 (not warmed) / disabled / no key => omitted.
        if (
            _fonoloji_api_key(self.config)
            and _flag(self.config, "advisor_fonoloji_market_digest_enabled", False)
        ):
            digest = await self._fetch_market_digest()
            if digest is not None:
                extra["market_digest"] = digest

        return AdviceResult(
            market=self.market,
            symbol=symbol,
            horizon=horizon,
            direction=direction,
            entry_low=entry_low,
            entry_high=entry_high,
            target_price=target,
            stop_price=stop,
            confidence=confidence,
            rationale=rationale,
            model_id=model_id,
            data_source_status=actual_status,
            data_source_note=note,
            sim_enabled=str(self.config.get("sim_default_enabled", "false")).lower() == "true",
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra=extra,
        )

    # ------------------------------------------------------------------
    # Fonoloji path (AVAILABLE, optional — preferred when a key is present)
    # ------------------------------------------------------------------

    def _fetch_fonoloji(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Fetch a BIST price series via the shared FonolojiClient. Runs in executor.

        Verified contract: GET /stocks/{ticker}/chart?period=... -> a price
        series (points with date + price/close; mapped defensively). Fonoloji
        gives close/price only, so OHLC is synthesised from close (same approach
        as the NAV path), which is sufficient for the SMA/RSI/BB suite. Volume,
        if present, is used; otherwise vol_ratio is neutral.

        FAIL-SOFT: returns None on missing key / HTTP error / unusable shape so
        the caller falls through to borsapy/yfinance.
        """
        client = FonolojiClient(self.config)
        if not client.enabled:
            return None

        lookback = _LOOKBACK[horizon]
        period = _fonoloji_period_for_lookback(lookback + 60)
        payload = client.stock_chart(symbol, period=period)
        df = _fonoloji_chart_to_df(payload)
        if df is None or len(df) < 20:
            logger.debug(
                "[bist] Fonoloji chart for %r (period=%s) yielded no usable "
                "series.", symbol, period,
            )
            return None
        return _compute_technicals_ohlcv(df, lookback)

    # ------------------------------------------------------------------
    # Broker analyst-consensus (Fonoloji /stocks/{ticker}/recommendations)
    # ------------------------------------------------------------------

    async def _fetch_analyst_consensus(self, symbol: str, current_price):
        """
        Fetch broker recommendations (AL/TUT/SAT + target price) and reduce them
        to ONE directional vote in {-1, 0, +1} plus a raw summary dict for the
        dashboard. Runs the sync client call in an executor. FAIL-SOFT: returns
        (0, None) on no key / 404 / error / unrecognised shape so the BIST
        technical signal is unchanged.
        """
        import asyncio

        def _call():
            try:
                client = FonolojiClient(self.config)
                if not client.enabled:
                    return None
                return client.stock_recommendations(symbol)
            except Exception as exc:  # never break the advice loop
                logger.debug(
                    "[bist] Fonoloji recommendations fetch failed for %s: %s",
                    symbol, exc,
                )
                return None

        loop = asyncio.get_event_loop()
        payload = await loop.run_in_executor(None, _call)
        if payload is None:
            return 0, None
        try:
            return _consensus_vote(payload, current_price)
        except Exception as exc:
            logger.debug(
                "[bist] consensus-vote parse failed for %s: %s", symbol, exc
            )
            return 0, None

    async def _fetch_market_digest(self) -> Optional[dict]:
        """
        Fetch the FREE Fonoloji AI daily market digest (GET /market/digest).
        Runs the sync client call in an executor; the client owns the 6h TTL
        cache so this is one real fetch per cycle. FAIL-SOFT: None on no key /
        404 (not yet warmed) / error — never blocks the cycle.
        """
        import asyncio

        def _call():
            try:
                client = FonolojiClient(self.config)
                if not client.enabled:
                    return None
                return client.market_digest()
            except Exception as exc:
                logger.debug("[bist] Fonoloji market-digest fetch failed: %s", exc)
                return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, _call)

    # ------------------------------------------------------------------
    # borsapy primary path (AVAILABLE, ~15min delayed, TradingView-backed)
    # ------------------------------------------------------------------

    def _fetch_borsapy(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Fetch BIST OHLCV data via borsapy (TradingView-backed, ~15min delayed).
        Runs in executor (sync).

        Symbol format: bare ticker without .IS suffix (e.g. 'THYAO', 'GARAN').
        borsapy also accepts .IS-suffixed tickers; both are normalised here.

        borsapy API (v0.x, as of 2026-06-02):
          from borsapy import Client
          client = Client()
          df = client.get_history(symbol, period='1y', interval='1d')
          # DataFrame columns: open, high, low, close, volume (lowercase)

        Import guard: if borsapy is not installed, returns None immediately
        (caller falls through to yfinance). This is the sole place the
        ImportError is caught -- the advice loop NEVER crashes.

        Returns a signals dict with the same keys as us_equities (template).
        Returns None on any failure (network, no data, wrong columns).
        """
        try:
            from borsapy import Client as BorsapyClient  # import guard
        except ImportError:
            logger.debug(
                "[bist] borsapy not installed -- falling back to yfinance. "
                "To enable full BIST coverage: pip install borsapy "
                "(rebuild Docker image if containerised)."
            )
            return None

        try:
            import pandas as pd
        except ImportError:
            return None

        lookback = _LOOKBACK[horizon]
        # Map lookback days to borsapy period strings
        if lookback <= 30:
            period = "1mo"
        elif lookback <= 90:
            period = "3mo"
        elif lookback <= 180:
            period = "6mo"
        else:
            period = "2y"

        # Normalise: borsapy expects bare ticker (strip .IS suffix if present)
        bare_symbol = symbol.upper()
        if bare_symbol.endswith(".IS"):
            bare_symbol = bare_symbol[:-3]

        try:
            client = BorsapyClient()
            df = client.get_history(bare_symbol, period=period, interval="1d")
        except Exception as exc:
            logger.warning(
                "[bist] borsapy.get_history(%r, period=%r) failed: %s. "
                "Falling back to yfinance.",
                bare_symbol, period, exc,
            )
            return None

        if df is None or df.empty or len(df) < 20:
            logger.debug(
                "[bist] borsapy insufficient data for %r (rows=%d). Trying yfinance.",
                bare_symbol, 0 if df is None else len(df),
            )
            return None

        # Normalise column names to lowercase (borsapy may vary case)
        df = df.copy()
        df.columns = [str(c).lower() for c in df.columns]

        required = {"open", "high", "low", "close"}
        if not required.issubset(set(df.columns)):
            if "close" not in df.columns:
                logger.warning(
                    "[bist] borsapy missing 'close' column for %r. Columns: %s",
                    bare_symbol, list(df.columns),
                )
                return None
            # Synthesise missing OHLC columns from close (workable for SMA/RSI/BB)
            for col in required - set(df.columns):
                df[col] = df["close"]

        if "volume" not in df.columns:
            df["volume"] = 0.0

        # Ensure index is sorted ascending (TradingView may return descending)
        df = df.sort_index()

        return _compute_technicals_ohlcv(df, lookback)

    # ------------------------------------------------------------------
    # yfinance fallback path (DEGRADED)
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Fetch BIST stock data via yfinance (.IS suffix required).

        Auto-appends .IS if missing (BIST convention in yfinance).
        Runs in executor (sync).
        """
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance not installed: pip install yfinance")

        # Auto-append .IS if missing
        yf_symbol = symbol.upper()
        if not yf_symbol.endswith(".IS"):
            yf_symbol = yf_symbol + ".IS"

        lookback = _LOOKBACK[horizon]
        end = datetime.utcnow()
        start = end - timedelta(days=lookback + 60)

        ticker = yf.Ticker(yf_symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )

        if df is None or df.empty or len(df) < 20:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
        return _compute_technicals_ohlcv(df, lookback)

    # ------------------------------------------------------------------
    # Matriks paid path -- STUBBED (Wave-23 wiring needed)
    # ------------------------------------------------------------------

    def _fetch_matriks(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        STUB: Matriks / Rasyonet REST API client.

        To wire (Wave-23):
          1. Set advisor_bist_data_source='matriks' in advisor_config.
          2. Set ADVISOR_BIST_API_KEY in Secure Credentials panel.
          3. Obtain Matriks API docs from matriks.com.
          4. Implement GET /history/bars?symbol=...&apikey=...
          5. Map response to signals dict (same shape as _fetch_yfinance).
          6. Update data_source_status() for the matriks path.
        """
        raise NotImplementedError(
            "Matriks API client not yet implemented. "
            "See _fetch_matriks() docstring for wiring instructions."
        )


# ---------------------------------------------------------------------------
# Technical computation (shared by borsapy + yfinance paths)
# Mirrors us_equities._fetch_and_compute signal shape exactly.
# ---------------------------------------------------------------------------

def _compute_technicals_ohlcv(df, lookback: int) -> Optional[dict]:
    """
    Compute SMA20/50, RSI-14, Bollinger Bands, vol_ratio from OHLCV DataFrame.
    Returns a signals dict with the same keys as the us_equities template.
    """
    close = df["close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(20).mean()
    sma_signal = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1

    rsi_val = _rsi(close, 14).iloc[-1]
    if rsi_val > 70:
        rsi_signal = -1   # overbought
    elif rsi_val < 30:
        rsi_signal = 1    # oversold
    else:
        rsi_signal = 0

    bb_mid = sma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    last_close = close.iloc[-1]

    if last_close > bb_upper.iloc[-1]:
        bb_signal = -1   # above upper band -- overextended
    elif last_close < bb_lower.iloc[-1]:
        bb_signal = 1    # below lower band -- potential reversal
    else:
        bb_pos = (last_close - bb_lower.iloc[-1]) / (
            bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10
        )
        bb_signal = 1 if bb_pos > 0.5 else -1

    # Volume ratio (may be zero for some BIST feeds; default to 1.0)
    vol_col = df.get("volume", None)
    if vol_col is not None and vol_col.sum() > 0:
        avg_vol = vol_col.rolling(20).mean().iloc[-1]
        last_vol = vol_col.iloc[-1]
        vol_ratio = last_vol / max(avg_vol, 1)
    else:
        vol_ratio = 1.0  # no volume -- neutral

    # ATR-14 as a fraction of close — volatility unit for horizon-aware levels.
    atr_pct = _atr_pct(df["high"], df["low"], df["close"], 14)

    return {
        "close": last_close,
        "sma20": sma20.iloc[-1],
        "sma50": sma50.iloc[-1],
        "rsi": rsi_val,
        "bb_upper": bb_upper.iloc[-1],
        "bb_lower": bb_lower.iloc[-1],
        "bb_mid": bb_mid.iloc[-1],
        "sma_signal": sma_signal,
        "rsi_signal": rsi_signal,
        "bb_signal": bb_signal,
        "vol_ratio": vol_ratio,
        "atr_pct": atr_pct,
        "df": df[["open", "high", "low", "close", "volume"]].tail(lookback)
        if "volume" in df.columns
        else df[["open", "high", "low", "close"]].tail(lookback),
    }


# ---------------------------------------------------------------------------
# Signal helpers (pure functions -- no side effects)
# ---------------------------------------------------------------------------

def _rsi(series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _atr_pct(high, low, close, period: int = 14) -> Optional[float]:
    """ATR-`period` as a fraction of last close. None on insufficient/NaN data."""
    try:
        import math as _math
        prev_close = close.shift(1)
        tr = (
            (high - low).abs()
            .combine((high - prev_close).abs(), max)
            .combine((low - prev_close).abs(), max)
        )
        atr = tr.rolling(period).mean().iloc[-1]
        last_close = float(close.iloc[-1])
        if atr is None or _math.isnan(atr) or last_close <= 0:
            return None
        return float(atr) / last_close
    except Exception:
        return None


def _signals_to_direction(signals: dict, analyst_vote: int = 0) -> Direction:
    """Majority vote across SMA, RSI, BB signals + an optional broker
    analyst-consensus vote (AL/TUT/SAT). The analyst vote is ONE additional
    input (weight 1, same as each technical vote) so it cannot dominate the
    three technical votes; on a 0 vote (no data / TUT) behaviour is unchanged."""
    votes = (
        signals["sma_signal"]
        + signals["rsi_signal"]
        + signals["bb_signal"]
        + int(analyst_vote)
    )
    if votes > 0:
        return Direction.LONG
    elif votes < 0:
        return Direction.SHORT
    return Direction.NEUTRAL


# Turkish broker-rating tokens -> signed vote. AL=buy, TUT=hold, SAT=sell.
# English aliases tolerated defensively (the live JSON shape is operator-
# confirmable). Anything unrecognised => no vote (skipped from the tally).
_RATING_MAP = {
    "AL": 1, "GUCLU AL": 1, "GÜÇLÜ AL": 1, "TOPLA": 1, "EKLE": 1, "ENDEKS USTU": 1,
    "BUY": 1, "STRONG BUY": 1, "OUTPERFORM": 1, "OVERWEIGHT": 1, "ACCUMULATE": 1,
    "TUT": 0, "NOTR": 0, "NÖTR": 0, "HOLD": 0, "NEUTRAL": 0, "MARKET PERFORM": 0,
    "SAT": -1, "GUCLU SAT": -1, "GÜÇLÜ SAT": -1, "AZALT": -1, "ENDEKS ALTI": -1,
    "SELL": -1, "STRONG SELL": -1, "UNDERPERFORM": -1, "UNDERWEIGHT": -1, "REDUCE": -1,
}


def _consensus_vote(payload, current_price):
    """
    Reduce a Fonoloji /stocks/{ticker}/recommendations payload to ONE directional
    vote in {-1, 0, +1} and a raw summary dict for the dashboard.

    DEFENSIVE shape handling (confirm exact keys on the first live call):
      - payload may be a top-level list of recs, or a dict wrapping the list
        under recommendations/data/results/items/brokers/analysts.
      - each rec's rating is read from rating/recommendation/oneri/tavsiye/
        rec/signal/grade; its target from target_price/target/hedef/
        price_target/target_value.

    Vote rule (count-weighted, target-upside as a tie/confirmation influence):
      net = sum(AL=+1, TUT=0, SAT=-1) over recognised recs.
      If net != 0  -> sign(net).
      If net == 0 (or no ratings) but a mean target price exists and
        current_price is usable, vote on target upside:
          upside >= +5%  -> +1 ;  <= -5% -> -1 ; else 0.
    Returns (vote:int, summary:dict). summary always includes the AL/TUT/SAT
    counts + n + (optional) mean_target + implied_upside_pct + vote.
    """
    records = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for k in ("recommendations", "data", "results", "items",
                  "brokers", "analysts", "list", "rows"):
            v = payload.get(k)
            if isinstance(v, list) and v:
                records = v
                break
        if records is None:
            # A single flattened summary dict (e.g. {al:3, tut:1, sat:0}).
            counts = {}
            for token, field in (("AL", "al"), ("TUT", "tut"), ("SAT", "sat"),
                                  ("AL", "buy"), ("TUT", "hold"), ("SAT", "sell")):
                if field in payload:
                    try:
                        counts[token] = counts.get(token, 0) + int(payload[field] or 0)
                    except (TypeError, ValueError):
                        pass
            if counts:
                records = (
                    [{"rating": "AL"}] * counts.get("AL", 0)
                    + [{"rating": "TUT"}] * counts.get("TUT", 0)
                    + [{"rating": "SAT"}] * counts.get("SAT", 0)
                )
    if not records or not isinstance(records, list):
        return 0, None

    n_al = n_tut = n_sat = 0
    targets = []
    for r in records:
        if not isinstance(r, dict):
            continue
        rating = None
        for rk in ("rating", "recommendation", "oneri", "öneri", "tavsiye",
                   "rec", "signal", "grade", "action"):
            if r.get(rk) is not None:
                rating = str(r.get(rk)).strip().upper()
                break
        if rating is not None and rating in _RATING_MAP:
            v = _RATING_MAP[rating]
            if v > 0:
                n_al += 1
            elif v < 0:
                n_sat += 1
            else:
                n_tut += 1
        for tk in ("target_price", "target", "hedef", "hedef_fiyat",
                   "price_target", "target_value", "fiyat_hedefi"):
            if r.get(tk) is not None:
                try:
                    tv = float(r.get(tk))
                    if tv > 0:
                        targets.append(tv)
                    break
                except (TypeError, ValueError):
                    continue

    n = n_al + n_tut + n_sat
    net = n_al - n_sat

    mean_target = sum(targets) / len(targets) if targets else None
    upside_pct = None
    try:
        cp = float(current_price) if current_price is not None else None
        if mean_target and cp and cp > 0:
            upside_pct = (mean_target / cp - 1.0) * 100.0
    except (TypeError, ValueError):
        upside_pct = None

    if net > 0:
        vote = 1
    elif net < 0:
        vote = -1
    elif upside_pct is not None:
        vote = 1 if upside_pct >= 5.0 else (-1 if upside_pct <= -5.0 else 0)
    else:
        vote = 0

    summary = {
        "al": n_al, "tut": n_tut, "sat": n_sat, "n": n,
        "net": net, "vote": vote,
        "mean_target": round(mean_target, 4) if mean_target is not None else None,
        "implied_upside_pct": round(upside_pct, 2) if upside_pct is not None else None,
        "source": "fonoloji_recommendations",
        "disclaimer": "Broker consensus — one input, ADVICE-ONLY.",
    }
    if n == 0 and mean_target is None:
        return 0, None
    return vote, summary


def _fonoloji_chart_to_df(payload):
    """
    Turn a Fonoloji /stocks/{ticker}/chart payload into an OHLCV DataFrame.

    The exact JSON shape is operator-confirmable at runtime, so this is
    DEFENSIVE: accepts a top-level list, or a dict wrapping the series under
    points/data/prices/chart/series/result. Each row maps:
      date|tarih|timestamp        -> date (ISO or epoch-ms)
      close|price|fiyat|last      -> close
      open|high|low (if present)  -> OHLC; else synthesised from close
      volume|hacim (if present)   -> volume; else 0
    Returns a sorted OHLCV DataFrame indexed by date, or None. Pure + offline.
    """
    try:
        import pandas as pd
    except ImportError:
        return None
    if payload is None:
        return None

    records = None
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict):
        for k in ("points", "data", "prices", "chart", "series",
                  "result", "results", "candles", "rows", "items"):
            v = payload.get(k)
            if isinstance(v, list) and v:
                records = v
                break
            if isinstance(v, dict):
                for k2 in ("points", "data", "prices", "series", "candles"):
                    v2 = v.get(k2)
                    if isinstance(v2, list) and v2:
                        records = v2
                        break
            if records:
                break
    if not records or not isinstance(records, list):
        return None
    records = [r for r in records if isinstance(r, dict)]
    if not records:
        return None

    try:
        df = pd.DataFrame(records)
    except Exception:
        return None
    df.columns = [str(c).lower() for c in df.columns]

    col_map = {}
    for c in df.columns:
        if c in ("close", "price", "fiyat", "last", "kapanis"):
            col_map.setdefault("close", c)
        elif c in ("open", "acilis"):
            col_map.setdefault("open", c)
        elif c in ("high", "yuksek", "max"):
            col_map.setdefault("high", c)
        elif c in ("low", "dusuk", "min"):
            col_map.setdefault("low", c)
        elif c in ("volume", "hacim", "vol"):
            col_map.setdefault("volume", c)
        elif c in ("date", "tarih", "datetime", "timestamp", "time"):
            col_map.setdefault("date", c)

    if "close" not in col_map or "date" not in col_map:
        return None

    out = pd.DataFrame()
    raw_date = df[col_map["date"]]
    num = pd.to_numeric(raw_date, errors="coerce")
    if num.notna().mean() > 0.5 and float(num.dropna().abs().max() or 0) > 1e11:
        out["date"] = pd.to_datetime(num, unit="ms", errors="coerce")
    else:
        sample = raw_date.astype(str).str.strip()
        is_iso = sample.str.match(r"^\d{4}-\d{2}-\d{2}").mean() > 0.5
        out["date"] = pd.to_datetime(raw_date, dayfirst=not is_iso, errors="coerce")

    out["close"] = pd.to_numeric(df[col_map["close"]], errors="coerce")
    for ohlc in ("open", "high", "low"):
        if ohlc in col_map:
            out[ohlc] = pd.to_numeric(df[col_map[ohlc]], errors="coerce")
    out["volume"] = (
        pd.to_numeric(df[col_map["volume"]], errors="coerce")
        if "volume" in col_map else 0.0
    )

    out = out.dropna(subset=["date", "close"]).set_index("date").sort_index()
    if out.empty:
        return None
    # Synthesise any missing OHLC from close (workable for SMA/RSI/BB).
    for ohlc in ("open", "high", "low"):
        if ohlc not in out.columns:
            out[ohlc] = out["close"]
        else:
            out[ohlc] = out[ohlc].fillna(out["close"])
    return out[["open", "high", "low", "close", "volume"]]


# ---------------------------------------------------------------------------
# Guarded self-test (no network) — Fonoloji chart mapping + technicals.
# Run: python -m modules.advisor.core.analyzers.bist
# ---------------------------------------------------------------------------

def _self_test() -> int:
    try:
        import pandas as pd  # noqa: F401
    except ImportError:
        print("SKIP: pandas not installed")
        return 0
    import datetime as _dt

    dates = [(_dt.date(2026, 1, 1) + _dt.timedelta(days=i)).isoformat()
             for i in range(40)]
    prices = [100.0 + 0.5 * i for i in range(40)]
    failures = 0

    cases = {
        "close-only": [{"date": d, "close": p} for d, p in zip(dates, prices)],
        "price-tr": {"points": [{"tarih": d, "fiyat": p, "hacim": 1000 + i}
                                for i, (d, p) in enumerate(zip(dates, prices))]},
        "full-ohlcv": {"data": [{"date": d, "open": p, "high": p + 1,
                                 "low": p - 1, "close": p, "volume": 5000}
                                for d, p in zip(dates, prices)]},
    }
    for name, payload in cases.items():
        df = _fonoloji_chart_to_df(payload)
        if df is None or len(df) < 20 or "close" not in df.columns:
            print(f"FAIL[{name}]: chart->df produced no usable series")
            failures += 1
            continue
        sig = _compute_technicals_ohlcv(df, 30)
        ok = sig is not None and abs(sig["close"] - prices[-1]) < 1e-6
        print(f"{'OK' if ok else 'FAIL'}[{name}]: rows={len(df)} "
              f"close={sig['close'] if sig else None}")
        failures += 0 if ok else 1

    for bad in (None, {}, [], {"data": []}, [1, 2], {"x": 1}):
        if _fonoloji_chart_to_df(bad) is not None:
            print(f"FAIL: bad chart payload {bad!r} did not return None")
            failures += 1

    # --- analyst-consensus vote mapping (AL/TUT/SAT -> +1/0/-1) ---
    cons_cases = [
        # (payload, current_price, expected_vote)
        ([{"rating": "AL"}, {"rating": "AL"}, {"rating": "SAT"}], 100.0, 1),
        ([{"recommendation": "SAT"}, {"recommendation": "SAT"}], 100.0, -1),
        ([{"rating": "TUT"}, {"rating": "TUT"}], 100.0, 0),
        # net==0 -> fall back to target upside (+10% => +1)
        ([{"rating": "AL", "target_price": 110.0},
          {"rating": "SAT", "target": 110.0}], 100.0, 1),
        # wrapped + Turkish keys
        ({"recommendations": [{"oneri": "GUCLU AL"}, {"oneri": "AL"}]}, None, 1),
        # flattened counts dict
        ({"al": 1, "tut": 0, "sat": 3}, 100.0, -1),
    ]
    for payload, cp, expected in cons_cases:
        vote, summ = _consensus_vote(payload, cp)
        ok = vote == expected and summ is not None
        print(f"{'OK' if ok else 'FAIL'}[consensus {expected:+d}]: vote={vote} "
              f"summary={summ}")
        failures += 0 if ok else 1
    # No usable data => (0, None), fail-soft.
    for bad in (None, {}, [], {"data": []}, [1, 2], {"foo": "bar"},
                [{"broker": "X"}]):
        v, s = _consensus_vote(bad, 100.0)
        if v != 0 or s is not None:
            print(f"FAIL: bad consensus payload {bad!r} -> ({v}, {s})")
            failures += 1

    print("SELF-TEST", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
