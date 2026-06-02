"""
CryptoAnalyzer — crypto market analyzer using ccxt public REST (free, no key).

Data source: ccxt with public REST endpoints. No API key required for
read-only OHLCV. Supported exchanges: any ccxt exchange that exposes
fetch_ohlcv — default is Binance.

Operator config (advisor_config DB):
  advisor_crypto_exchange   : exchange id for ccxt (default "binance")
  watchlist_crypto          : comma-sep ccxt pairs, e.g. "BTC/USDT,ETH/USDT"

Data source status: AVAILABLE (ccxt public, no key).

Technical indicators (identical suite to us_equities reference impl):
  - SMA20 / SMA50 crossover
  - RSI-14
  - Bollinger Bands (20-day, 2-sigma)
  - Volume ratio vs 20-day average

LLM rationale: shared rationale_helper.build_rationale() — same Anthropic
key / model resolution as all other analyzers.

ADVICE-ONLY. No orders placed. analyze() never raises.
"""

from __future__ import annotations

import logging
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

logger = logging.getLogger("advisor.analyzer.crypto")

# Horizon → number of daily bars to request from ccxt
# ccxt fetch_ohlcv limit parameter; request extra for MA warmup
_LOOKBACK_BARS: dict = {
    Horizon.SHORT: 90,    # 30 day window + 60 warmup
    Horizon.MID: 240,     # 6-month window + 60 warmup
    Horizon.LONG: 790,    # 2-year window + 60 warmup (many exchanges cap at 1000)
}

# How many bars to keep for Kronos after technicals computed
_RETURN_BARS: dict = {
    Horizon.SHORT: 30,
    Horizon.MID: 180,
    Horizon.LONG: 730,
}


class CryptoAnalyzer(BaseAnalyzer):
    """
    Crypto market analyzer using ccxt public OHLCV.

    Data source : ccxt public REST — free, no API key required.
    LLM         : Anthropic API via shared rationale_helper.
    """

    market = Market.CRYPTO

    def __init__(self, config: dict, db_pool=None):
        super().__init__(config, db_pool)
        self.last_price: dict = {}
        self._exchange_id: str = config.get("advisor_crypto_exchange", "binance")

    # ------------------------------------------------------------------
    # BaseAnalyzer contract
    # ------------------------------------------------------------------

    def data_source_status(self) -> DataSourceStatus:
        """
        ccxt is always available (no key required for public endpoints).
        Returns NOT_CONFIGURED only if ccxt is not installed.
        """
        try:
            import ccxt  # noqa: F401
            return DataSourceStatus.AVAILABLE
        except ImportError:
            return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce an AdviceResult for a crypto pair (e.g. "BTC/USDT").
        Never raises — _error_result is returned on any failure.
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

        exchange_id = self.config.get("advisor_crypto_exchange", self._exchange_id)

        # ccxt is sync — run in executor
        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_and_compute, symbol, horizon, exchange_id
        )

        if signals is None:
            return self._error_result(
                symbol, horizon,
                Exception(
                    f"ccxt ({exchange_id}) returned no OHLCV data for {symbol}"
                )
            )

        direction = _signals_to_direction(signals)
        confidence = _compute_confidence(signals)
        entry_low, entry_high = _entry_range(signals)
        target = _target_price(signals, direction)
        stop = _stop_price(signals, direction)

        # Cache last price for mark-to-market
        self.last_price[symbol] = signals.get("close", 0)

        # LLM rationale — shared helper; rule-based fallback on any failure
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
        extra = {"klines_df": signals.get("df"), "exchange": exchange_id}

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
            sim_enabled=bool(self.config.get("sim_default_enabled", "false") == "true"),
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra=extra,
        )

    def _fetch_and_compute(
        self,
        symbol: str,
        horizon: Horizon,
        exchange_id: str,
    ) -> Optional[dict]:
        """
        Fetch daily OHLCV from ccxt and compute technical signals.
        Runs in executor (sync).

        ccxt notes:
        - Uses public REST only; no API key needed for fetch_ohlcv.
        - Some exchanges cap fetch_ohlcv limit at 500 or 1000; we request
          min(requested_bars, 1000) and accept fewer on long horizon.
        - Partial last bar (current open candle) is stripped if volume < 30%
          of the 5-bar average — avoids look-ahead into the current session.
        """
        try:
            import ccxt
            import pandas as pd
        except ImportError:
            raise RuntimeError("ccxt/pandas not installed: pip install ccxt pandas")

        limit = min(_LOOKBACK_BARS[horizon], 1000)

        # Instantiate exchange — public endpoints only
        try:
            exchange_cls = getattr(ccxt, exchange_id)
        except AttributeError:
            self.logger.warning(
                f"[crypto] Unknown ccxt exchange '{exchange_id}', falling back to binance"
            )
            exchange_cls = ccxt.binance

        exchange = exchange_cls({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })

        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe="1d", limit=limit)
        except Exception as exc:
            raise RuntimeError(
                f"ccxt fetch_ohlcv failed for {symbol} on {exchange_id}: {exc}"
            )
        finally:
            try:
                exchange.close()
            except Exception:
                pass

        if not ohlcv or len(ohlcv) < 20:
            return None

        df = pd.DataFrame(
            ohlcv,
            columns=["timestamp", "open", "high", "low", "close", "volume"],
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.set_index("timestamp").sort_index()

        # Drop last bar if it's the current partial candle (volume << typical)
        if len(df) >= 7:
            avg_vol_check = df["volume"].iloc[-6:-1].mean()
            if avg_vol_check > 0 and df["volume"].iloc[-1] < avg_vol_check * 0.3:
                df = df.iloc[:-1]

        if len(df) < 20:
            return None

        close = df["close"]

        # SMA crossover
        sma20 = close.rolling(20).mean()
        sma50 = close.rolling(50).mean() if len(close) >= 50 else close.rolling(20).mean()
        sma_signal = 1 if sma20.iloc[-1] > sma50.iloc[-1] else -1

        # RSI-14
        rsi_series = _rsi(close, 14)
        rsi_val = rsi_series.iloc[-1]
        if rsi_val > 70:
            rsi_signal = -1   # overbought
        elif rsi_val < 30:
            rsi_signal = 1    # oversold
        else:
            rsi_signal = 0

        # Bollinger Bands (20-day, 2-sigma)
        bb_mid = sma20
        bb_std = close.rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        last_close = close.iloc[-1]
        if last_close > bb_upper.iloc[-1]:
            bb_signal = -1    # overextended above band
        elif last_close < bb_lower.iloc[-1]:
            bb_signal = 1     # below lower band — potential reversal
        else:
            bb_pos = (last_close - bb_lower.iloc[-1]) / (
                bb_upper.iloc[-1] - bb_lower.iloc[-1] + 1e-10
            )
            bb_signal = 1 if bb_pos > 0.5 else -1

        # Volume confirmation
        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        last_vol = df["volume"].iloc[-1]
        vol_ratio = last_vol / max(avg_vol, 1)

        return_bars = _RETURN_BARS[horizon]
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
            "df": df[["open", "high", "low", "close", "volume"]].tail(return_bars),
        }


# ---------------------------------------------------------------------------
# Signal helpers (pure functions — identical to us_equities for consistency)
# ---------------------------------------------------------------------------

def _rsi(series, period: int = 14):
    """Compute RSI from a pandas Series."""
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
    """Confidence in [0.0, 1.0] based on signal agreement + volume."""
    abs_vote = abs(
        signals["sma_signal"] + signals["rsi_signal"] + signals["bb_signal"]
    )
    base_conf = abs_vote / 3.0
    vol_boost = 0.10 if signals.get("vol_ratio", 1.0) > 1.5 else 0.0
    return min(base_conf + vol_boost, 1.0)


def _entry_range(signals: dict) -> tuple:
    """Entry range = +/-0.5% around last close."""
    close = signals["close"]
    return round(close * 0.995, 8), round(close * 1.005, 8)


def _target_price(signals: dict, direction: Direction) -> Optional[float]:
    """
    Primary price target.
    LONG : max(BB upper, close * 1.05).
    SHORT: min(BB lower, close * 0.95).
    """
    if direction == Direction.LONG:
        return round(
            max(signals["bb_upper"], signals["close"] * 1.05), 8
        )
    elif direction == Direction.SHORT:
        return round(
            min(signals["bb_lower"], signals["close"] * 0.95), 8
        )
    return None


def _stop_price(signals: dict, direction: Direction) -> Optional[float]:
    """
    Suggested stop-loss.
    LONG : -3% from close.
    SHORT: +3% from close.
    """
    close = signals["close"]
    if direction == Direction.LONG:
        return round(close * 0.97, 8)
    elif direction == Direction.SHORT:
        return round(close * 1.03, 8)
    return None
