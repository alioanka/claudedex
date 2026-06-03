"""
USEquitiesAnalyzer — reference implementation using yfinance (free, no key).

Covers NASDAQ / NYSE equities. Data source: yfinance (Yahoo Finance).
This is the template analyzer — the others (BIST, FX, MIDAS_FUNDS) mirror
this structure but their data sources require operator setup.

Data source: FREE (yfinance). No API key required.
Installed via: pip install yfinance

LLM rationale: calls Anthropic API via the shared rationale_helper module.
Model id is loaded from advisor_config.advisor_anthropic_model (default
claude-opus-4-5). Loud WARNING on 404; never silent-fallback — rule-based
text is produced instead.

Prompt injection: strip non-printable chars + truncate before LLM call
(handled inside rationale_helper._sanitize).

ADVICE-ONLY: This analyzer produces no orders. The rationale field is
the only LLM output; it is stored in advisor_advice.rationale (read-only).
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

logger = logging.getLogger("advisor.analyzer.us_equities")

# Horizon → lookback window for technical analysis
_LOOKBACK: dict = {
    Horizon.SHORT: 30,    # 30 calendar days of daily bars
    Horizon.MID: 180,     # 6 months
    Horizon.LONG: 730,    # 2 years
}


class USEquitiesAnalyzer(BaseAnalyzer):
    """
    US Equities analyzer (NASDAQ / NYSE).

    Data source : yfinance — free, no key required.
    LLM         : Anthropic API (advisor's own key — advisor_anthropic_api_key
                  in advisor_config or ADVISOR_ANTHROPIC_API_KEY env var).

    Technical signals computed (all on daily bars):
      - 20-day / 50-day SMA crossover (trend bias)
      - RSI-14 (momentum + overbought/oversold)
      - 20-day Bollinger Bands (volatility)
      - Average Volume vs 20-day average (volume confirmation)

    LLM rationale: the computed signal summary is sent to Claude for a
    natural-language explanation. The LLM does NOT make the directional
    decision — technicals drive direction; the LLM explains it.
    """

    market = Market.US_EQUITIES

    def __init__(self, config: dict, db_pool=None):
        super().__init__(config, db_pool)
        self._anthropic_client = None
        self.last_price: dict = {}

    # ------------------------------------------------------------------
    # BaseAnalyzer contract
    # ------------------------------------------------------------------

    def data_source_status(self) -> DataSourceStatus:
        """yfinance is always available (no key); return AVAILABLE."""
        try:
            import yfinance  # noqa: F401
            return DataSourceStatus.AVAILABLE
        except ImportError:
            return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce an AdviceResult for a US equity symbol.
        """
        try:
            return await self._analyze_internal(symbol, horizon)
        except Exception as exc:
            return self._error_result(symbol, horizon, exc)

    # ------------------------------------------------------------------
    # Internal implementation
    # ------------------------------------------------------------------

    async def _analyze_internal(self, symbol: str, horizon: Horizon) -> AdviceResult:
        import asyncio

        # yfinance is sync; run in thread to avoid blocking event loop
        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_and_compute, symbol, horizon
        )

        if signals is None:
            return self._error_result(
                symbol, horizon,
                Exception(f"yfinance returned no data for {symbol}")
            )

        direction = _signals_to_direction(signals)
        confidence = signal_confidence(signals, horizon, config=self.config)
        entry_low, entry_high, target, stop = horizon_levels(
            signals, direction, horizon, config=self.config, price_decimals=4
        )

        # Cache last price for mark-to-market
        self.last_price[symbol] = signals.get("close", 0)

        # LLM rationale — shared helper; fail-soft rule-based on any failure
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

        # Expose klines_df in extra for Kronos overlay
        extra = {"klines_df": signals.get("df")}

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
            data_source_status=DataSourceStatus.AVAILABLE,
            sim_enabled=str(self.config.get("sim_default_enabled", "false")).lower() == "true",
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra=extra,
        )

    def _fetch_and_compute(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Fetch OHLCV from yfinance and compute technical signals.
        Runs in executor (sync).
        """
        try:
            import yfinance as yf
            import pandas as pd
        except ImportError:
            raise RuntimeError("yfinance not installed: pip install yfinance")

        lookback_days = _LOOKBACK[horizon]
        end = datetime.utcnow()
        start = end - timedelta(days=lookback_days + 60)  # +60 for MA warmup

        ticker = yf.Ticker(symbol)
        df = ticker.history(start=start.strftime("%Y-%m-%d"),
                            end=end.strftime("%Y-%m-%d"),
                            interval="1d",
                            auto_adjust=True)

        if df is None or df.empty or len(df) < 20:
            return None

        df = df[["Open", "High", "Low", "Close", "Volume"]].rename(
            columns=str.lower
        )

        close = df["close"]

        # SMA crossover
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(20).mean()
        sma_signal = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1

        # RSI-14
        rsi = _rsi(close, 14)
        rsi_val = rsi.iloc[-1]
        if rsi_val > 70:
            rsi_signal = -1  # overbought → bearish bias
        elif rsi_val < 30:
            rsi_signal = 1   # oversold → bullish bias
        else:
            rsi_signal = 0

        # Bollinger Bands (20-day, 2σ)
        bb_mid = sma20
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        last_close = close.iloc[-1]
        if last_close > bb_upper.iloc[-1]:
            bb_signal = -1   # above upper band → overextended
        elif last_close < bb_lower.iloc[-1]:
            bb_signal = 1    # below lower band → potential reversal
        else:
            # Position within band: > 50% = mildly bullish
            bb_pos = (last_close - bb_lower.iloc[-1]) / (
                bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10
            )
            bb_signal = 1 if bb_pos > 0.5 else -1

        # Volume confirmation
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        last_vol = df["volume"].iloc[-1]
        vol_ratio = last_vol / max(avg_vol, 1)

        # ATR-14 as a fraction of close — volatility unit for horizon-aware levels.
        atr_pct = _atr_pct(df["high"], df["low"], close, 14)

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
            "df": df[["open", "high", "low", "close", "volume"]].tail(lookback_days),
        }


# ---------------------------------------------------------------------------
# Signal helpers (pure functions — no side effects)
# ---------------------------------------------------------------------------

def _rsi(series, period: int = 14):
    """Compute RSI from a pandas Series."""
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


