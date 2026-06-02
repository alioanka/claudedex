"""
BISTAnalyzer — Borsa Istanbul equities analyzer.

Data source priority (2026-06-02):

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

        # --- borsapy (AVAILABLE) ---
        if _borsapy_available() and configured_source in ("borsapy", ""):
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

        direction = _signals_to_direction(signals)
        confidence = _compute_confidence(signals)
        entry_low, entry_high = _entry_range(signals)
        target = _target_price(signals, direction)
        stop = _stop_price(signals, direction)

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


def _signals_to_direction(signals: dict) -> Direction:
    """Majority vote across SMA, RSI, BB signals."""
    votes = (
        signals["sma_signal"]
        + signals["rsi_signal"]
        + signals["bb_signal"]
    )
    if votes > 0:
        return Direction.LONG
    elif votes < 0:
        return Direction.SHORT
    return Direction.NEUTRAL


def _compute_confidence(signals: dict) -> float:
    """Confidence in [0.0, 1.0] -- signal agreement + volume confirmation."""
    abs_vote = abs(
        signals["sma_signal"] + signals["rsi_signal"] + signals["bb_signal"]
    )
    base_conf = abs_vote / 3.0
    vol_boost = 0.10 if signals.get("vol_ratio", 1.0) > 1.5 else 0.0
    return min(base_conf + vol_boost, 1.0)


def _entry_range(signals: dict) -> tuple:
    """+-0.5% band around last close (TRY prices, 2 decimal places)."""
    close = signals["close"]
    return round(close * 0.995, 2), round(close * 1.005, 2)


def _target_price(signals: dict, direction: Direction) -> Optional[float]:
    if direction == Direction.LONG:
        return round(max(signals["bb_upper"], signals["close"] * 1.05), 2)
    elif direction == Direction.SHORT:
        return round(min(signals["bb_lower"], signals["close"] * 0.95), 2)
    return None


def _stop_price(signals: dict, direction: Direction) -> Optional[float]:
    close = signals["close"]
    if direction == Direction.LONG:
        return round(close * 0.97, 2)
    elif direction == Direction.SHORT:
        return round(close * 1.03, 2)
    return None
