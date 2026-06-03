"""
FXAnalyzer — FX pairs and metals (XAU/USD, XAG/USD) analyzer.

Default data source: yfinance (free, no key). Covers major FX pairs
via "EURUSD=X" format and metals via "GC=F" (gold futures) / "SI=F"
(silver futures). Data may be delayed ~15 min for futures contracts.

Optional upgrade: Alpha Vantage (free tier 25 req/day). Set
  advisor_fx_data_source='alphavantage' in advisor_config DB AND
  ADVISOR_FX_ALPHAVANTAGE_KEY in Secure Credentials (alphavantage.co).

Watchlist format (watchlist_fx config key):
  yfinance    : "EURUSD=X,GBPUSD=X,XAUUSD=X,XAGUSD=X,GC=F,SI=F"
  alphavantage: "EUR/USD,GBP/USD,XAU/USD,XAG/USD"

Alpha Vantage metal mapping (handled internally):
  XAU/USD → function=FX_DAILY from_symbol=XAU to_symbol=USD
  XAG/USD → function=FX_DAILY from_symbol=XAG to_symbol=USD

Data source status:
  yfinance     → AVAILABLE (yfinance installed)
  alphavantage → AVAILABLE (key present) or NOT_CONFIGURED (key missing)
  neither set  → NOT_CONFIGURED

Technical indicators (identical suite to us_equities):
  SMA20/50, RSI-14, Bollinger Bands, volume ratio.
  Volume may be zero for spot FX pairs (yfinance); handled gracefully.

LLM rationale: shared rationale_helper.build_rationale().

ADVICE-ONLY. analyze() never raises.
"""

from __future__ import annotations

import logging
import os
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

logger = logging.getLogger("advisor.analyzer.fx")

_LOOKBACK: dict = {
    Horizon.SHORT: 30,
    Horizon.MID: 180,
    Horizon.LONG: 730,
}

_NOT_CONFIGURED_NOTE = (
    "FX data source not configured. "
    "Set advisor_fx_data_source='yfinance' in advisor_config (no key needed; "
    "covers major FX pairs via EURUSD=X and metals via GC=F / SI=F). "
    "Optional paid upgrade: set advisor_fx_data_source='alphavantage' and add "
    "ADVISOR_FX_ALPHAVANTAGE_KEY to Secure Credentials (free signup at alphavantage.co). "
    "Example watchlist: watchlist_fx='EURUSD=X,GBPUSD=X,GC=F,SI=F'."
)

# Alpha Vantage: max 25 requests/day on free tier
_AV_RATE_LIMIT_MSG = (
    "[fx] Alpha Vantage free tier: 25 requests/day. "
    "Consider reducing watchlist_fx or upgrading to paid tier."
)


class FXAnalyzer(BaseAnalyzer):
    """
    FX pairs and metals (XAU/USD, XAG/USD) analyzer.

    Default data source: yfinance (free, no key).
    Optional: Alpha Vantage (ADVISOR_FX_ALPHAVANTAGE_KEY).
    """

    market = Market.FX

    def __init__(self, config: dict, db_pool=None):
        super().__init__(config, db_pool)
        self.last_price: dict = {}
        self._av_warned = False   # rate limit advisory shown once per run

    # ------------------------------------------------------------------
    # BaseAnalyzer contract
    # ------------------------------------------------------------------

    def data_source_status(self) -> DataSourceStatus:
        source = self.config.get("advisor_fx_data_source", "yfinance")
        if source == "yfinance":
            try:
                import yfinance  # noqa: F401
                return DataSourceStatus.AVAILABLE
            except ImportError:
                return DataSourceStatus.NOT_CONFIGURED
        if source == "alphavantage":
            key = (
                self.config.get("advisor_fx_alphavantage_key")
                or os.getenv("ADVISOR_FX_ALPHAVANTAGE_KEY", "")
            )
            return DataSourceStatus.AVAILABLE if key else DataSourceStatus.NOT_CONFIGURED
        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce an AdviceResult for an FX pair or metal symbol.
        Never raises — _error_result on any failure.
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)
        try:
            return await self._analyze_internal(symbol, horizon, status)
        except Exception as exc:
            return self._error_result(symbol, horizon, exc)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _analyze_internal(
        self, symbol: str, horizon: Horizon, status: DataSourceStatus
    ) -> AdviceResult:
        import asyncio

        source = self.config.get("advisor_fx_data_source", "yfinance")

        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_and_compute, symbol, horizon, source
        )

        if signals is None:
            return self._error_result(
                symbol, horizon,
                Exception(
                    f"FX data source '{source}' returned no data for {symbol}"
                )
            )

        direction = _signals_to_direction(signals)
        confidence = signal_confidence(signals, horizon, config=self.config)
        entry_low, entry_high, target, stop = horizon_levels(
            signals, direction, horizon, config=self.config, price_decimals=6
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
        extra = {"klines_df": signals.get("df"), "data_source": source}

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
            data_source_status=status,
            sim_enabled=str(self.config.get("sim_default_enabled", "false")).lower() == "true",
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra=extra,
        )

    def _fetch_and_compute(
        self, symbol: str, horizon: Horizon, source: str
    ) -> Optional[dict]:
        """Dispatch to the correct data backend. Sync, runs in executor."""
        if source == "yfinance":
            return self._fetch_yfinance(symbol, horizon)
        if source == "alphavantage":
            return self._fetch_alphavantage(symbol, horizon)
        return None

    # ------------------------------------------------------------------
    # yfinance path (AVAILABLE — default)
    # ------------------------------------------------------------------

    def _fetch_yfinance(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Fetch via yfinance. Supports:
          - FX pairs: EURUSD=X, GBPUSD=X, XAUUSD=X, XAGUSD=X
          - Metal futures: GC=F (gold), SI=F (silver)
        Volume is often zero for spot FX; vol_ratio handled gracefully.
        """
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance not installed: pip install yfinance")

        lookback = _LOOKBACK[horizon]
        end = datetime.utcnow()
        start = end - timedelta(days=lookback + 60)

        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start.strftime("%Y-%m-%d"),
            end=end.strftime("%Y-%m-%d"),
            interval="1d",
            auto_adjust=True,
        )

        if df is None or df.empty or len(df) < 20:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(columns=str.lower)
        return _compute_technicals(df, lookback)

    # ------------------------------------------------------------------
    # Alpha Vantage path (AVAILABLE when key set — optional upgrade)
    # ------------------------------------------------------------------

    def _fetch_alphavantage(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Fetch via Alpha Vantage FX_DAILY endpoint.
        Free tier: 25 req/day, 500/month.
        Symbol format: "EUR/USD", "XAU/USD" etc.
        Operator upgrade: set advisor_fx_data_source='alphavantage' and
        ADVISOR_FX_ALPHAVANTAGE_KEY in Secure Credentials.
        """
        try:
            import requests
            import pandas as pd
        except ImportError:
            raise RuntimeError("requests/pandas not installed")

        key = (
            self.config.get("advisor_fx_alphavantage_key")
            or os.getenv("ADVISOR_FX_ALPHAVANTAGE_KEY", "")
        )
        if not key:
            raise RuntimeError("ADVISOR_FX_ALPHAVANTAGE_KEY not set")

        if not self._av_warned:
            self.logger.info(_AV_RATE_LIMIT_MSG)
            self._av_warned = True

        # Normalise "EUR/USD" → from_symbol=EUR, to_symbol=USD
        if "/" in symbol:
            from_sym, to_sym = symbol.split("/", 1)
        else:
            from_sym, to_sym = symbol, "USD"

        url = "https://www.alphavantage.co/query"
        params = {
            "function": "FX_DAILY",
            "from_symbol": from_sym.strip().upper(),
            "to_symbol": to_sym.strip().upper(),
            "outputsize": "full",
            "apikey": key,
        }
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()

        ts_key = "Time Series FX (Daily)"
        if ts_key not in data:
            err = data.get("Note") or data.get("Information") or str(data)[:200]
            raise RuntimeError(f"Alpha Vantage response error: {err}")

        records = []
        for date_str, vals in data[ts_key].items():
            records.append({
                "date": date_str,
                "open": float(vals["1. open"]),
                "high": float(vals["2. high"]),
                "low": float(vals["3. low"]),
                "close": float(vals["4. close"]),
                "volume": 0.0,  # FX has no volume in AV FX_DAILY
            })

        df = pd.DataFrame(records)
        df["date"] = pd.to_datetime(df["date"])
        df = df.set_index("date").sort_index()

        lookback = _LOOKBACK[horizon]
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=lookback + 60)
        df = df[df.index >= cutoff]

        if len(df) < 20:
            return None

        return _compute_technicals(df, lookback)


# ---------------------------------------------------------------------------
# Shared technical computation (both yfinance and AV paths)
# ---------------------------------------------------------------------------

def _compute_technicals(df, lookback: int) -> Optional[dict]:
    """
    Compute SMA20/50, RSI-14, BB, vol_ratio from an OHLCV DataFrame.
    Volume-ratio gracefully set to 1.0 if volume column is all zeros
    (common for spot FX pairs).
    """
    close = df["close"]

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(20).mean()
    sma_signal = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1

    rsi_val = _rsi(close, 14).iloc[-1]
    if rsi_val > 70:
        rsi_signal = -1
    elif rsi_val < 30:
        rsi_signal = 1
    else:
        rsi_signal = 0

    bb_mid = sma20
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    last_close = close.iloc[-1]

    if last_close > bb_upper.iloc[-1]:
        bb_signal = -1
    elif last_close < bb_lower.iloc[-1]:
        bb_signal = 1
    else:
        bb_pos = (last_close - bb_lower.iloc[-1]) / (
            bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10
        )
        bb_signal = 1 if bb_pos > 0.5 else -1

    # Volume: FX spot pairs often have volume=0; default to neutral ratio
    vol_col = df.get("volume", None)
    if vol_col is not None and vol_col.sum() > 0:
        avg_vol = vol_col.rolling(20).mean().iloc[-1]
        last_vol = vol_col.iloc[-1]
        vol_ratio = last_vol / max(avg_vol, 1)
    else:
        vol_ratio = 1.0  # no volume data — neutral

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
        "df": df[["open", "high", "low", "close"]].tail(lookback),
    }


# ---------------------------------------------------------------------------
# Signal helpers
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


def _signals_to_direction(signals: dict) -> Direction:
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
