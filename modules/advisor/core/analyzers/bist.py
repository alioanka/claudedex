"""
BISTAnalyzer — Borsa Istanbul equities analyzer.

Data source options (as of 2026-06-02):

  1. yfinance (.IS suffix) — DEGRADED
     Some BIST stocks are available (e.g. "THYAO.IS"), but:
     - Coverage is incomplete (mid/small caps often missing).
     - Data may be delayed 15-20 minutes.
     - Mutual funds are NOT covered.
     This path returns data_source_status=DEGRADED.
     Activate: set advisor_bist_data_source='yfinance' in advisor_config.
     Watchlist: watchlist_bist='THYAO.IS,EREGL.IS,GARAN.IS,AKBNK.IS'

  2. Matriks / Rasyonet / IsYatirim — AVAILABLE (paid)
     Full BIST coverage, real-time data, professional grade.
     Activate: set advisor_bist_data_source='matriks' in advisor_config AND
     provide ADVISOR_BIST_API_KEY in Secure Credentials.
     This path is STUBBED — operator must supply key and the integration
     must be wired once the API contract is confirmed.

Default behavior:
  advisor_bist_data_source not set → NOT_CONFIGURED (returns instructions).
  advisor_bist_data_source='yfinance' → DEGRADED (limited data, free).
  advisor_bist_data_source='matriks' + key set → AVAILABLE (stub, needs wiring).
  advisor_bist_data_source='matriks', key missing → NOT_CONFIGURED.

Technical indicators (identical suite to us_equities):
  SMA20/50, RSI-14, Bollinger Bands, volume ratio.

LLM rationale: shared rationale_helper.build_rationale().

ADVICE-ONLY. analyze() never raises.

Dashboard note (Wave-21):
  data_source_status=DEGRADED is shown to the operator as an amber badge
  indicating partial data coverage. Full BIST coverage requires the Matriks
  paid integration and ADVISOR_BIST_API_KEY.
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
    "Set advisor_bist_data_source='yfinance' in advisor_config for degraded "
    "free coverage (incomplete, delayed, .IS suffix tickers). "
    "For production: set advisor_bist_data_source='matriks' and provide "
    "ADVISOR_BIST_API_KEY in Secure Credentials (Settings > Credentials). "
    "Paid providers: Matriks (matriks.com), Rasyonet, IsYatirim. "
    "Watchlist key: watchlist_bist='THYAO.IS,EREGL.IS,GARAN.IS'."
)

_MATRIKS_STUB_NOTE = (
    "Matriks BIST integration is configured (key present) but the API "
    "client is not yet wired (Wave-21 stub). "
    "Technical contact: set ADVISOR_BIST_API_KEY and open a Wave-22 task "
    "to wire the Matriks REST client in bist.py::_fetch_matriks()."
)


class BISTAnalyzer(BaseAnalyzer):
    """
    Borsa Istanbul analyzer.

    yfinance degraded path: DEGRADED (incomplete coverage, free).
    Matriks paid path     : AVAILABLE when key set, but STUBBED until wired.
    No config             : NOT_CONFIGURED.
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
        if source == "yfinance":
            try:
                import yfinance  # noqa: F401
                return DataSourceStatus.DEGRADED  # partial coverage, delays
            except ImportError:
                return DataSourceStatus.NOT_CONFIGURED
        if source == "matriks":
            import os
            api_key = (
                self.config.get("advisor_bist_api_key")
                or os.getenv("ADVISOR_BIST_API_KEY", "")
            )
            return DataSourceStatus.AVAILABLE if api_key else DataSourceStatus.NOT_CONFIGURED
        return DataSourceStatus.NOT_CONFIGURED

    async def analyze(self, symbol: str, horizon: Horizon) -> AdviceResult:
        """
        Produce an AdviceResult for a BIST equity.
        Never raises — _error_result on any failure.
        """
        status = self.data_source_status()
        if status == DataSourceStatus.NOT_CONFIGURED:
            return self._not_configured(symbol, horizon, _NOT_CONFIGURED_NOTE)

        source = self.config.get("advisor_bist_data_source", "")

        # Matriks path: key is present but integration is stubbed
        if source == "matriks":
            return self._not_configured(symbol, horizon, _MATRIKS_STUB_NOTE)

        try:
            return await self._analyze_internal(symbol, horizon, status)
        except Exception as exc:
            return self._error_result(symbol, horizon, exc)

    # ------------------------------------------------------------------
    # Internal implementation (yfinance degraded path)
    # ------------------------------------------------------------------

    async def _analyze_internal(
        self, symbol: str, horizon: Horizon, status: DataSourceStatus
    ) -> AdviceResult:
        import asyncio

        loop = asyncio.get_event_loop()
        signals = await loop.run_in_executor(
            None, self._fetch_yfinance, symbol, horizon
        )

        if signals is None:
            return self._error_result(
                symbol, horizon,
                Exception(
                    f"yfinance returned no BIST data for {symbol}. "
                    "Check that symbol ends with .IS (e.g. THYAO.IS). "
                    "Mid/small caps may not be covered."
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

        # data_source_note reminds the operator of coverage limitations
        note = (
            "BIST data via yfinance (degraded): partial coverage, ~15min delay. "
            "For full coverage set advisor_bist_data_source='matriks' and "
            "ADVISOR_BIST_API_KEY in Secure Credentials."
        )

        extra = {"klines_df": signals.get("df"), "data_source": "yfinance"}

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
            data_source_status=DataSourceStatus.DEGRADED,
            data_source_note=note,
            sim_enabled=bool(self.config.get("sim_default_enabled", "false") == "true"),
            sim_amount_usd=float(self.config.get("sim_default_amount_usd", 1000)),
            extra=extra,
        )

    def _fetch_yfinance(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        Fetch BIST stock data via yfinance.

        Symbol format: "THYAO.IS" — the .IS suffix is required.
        If the operator's watchlist omits .IS, append it automatically.
        Runs in executor (sync).
        """
        try:
            import yfinance as yf
        except ImportError:
            raise RuntimeError("yfinance not installed: pip install yfinance")

        # Auto-append .IS if missing (BIST convention in yfinance)
        yf_symbol = symbol if symbol.endswith(".IS") else f"{symbol}.IS"

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

        avg_vol = df["volume"].rolling(20).mean().iloc[-1]
        last_vol = df["volume"].iloc[-1]
        vol_ratio = last_vol / max(avg_vol, 1)

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
            "df": df[["open", "high", "low", "close", "volume"]].tail(lookback),
        }

    # ------------------------------------------------------------------
    # Matriks paid path — STUBBED (Wave-22 wiring needed)
    # ------------------------------------------------------------------

    def _fetch_matriks(self, symbol: str, horizon: Horizon) -> Optional[dict]:
        """
        STUB: Matriks / Rasyonet REST API client.

        To activate:
          1. Set advisor_bist_data_source='matriks' in advisor_config.
          2. Set ADVISOR_BIST_API_KEY in Secure Credentials panel.
          3. Obtain Matriks API documentation from matriks.com.
          4. Implement HTTP client call (likely GET /history/bars?symbol=...&apikey=...).
          5. Map response to the same signals dict format as _fetch_yfinance.
          6. Return DataSourceStatus.AVAILABLE in data_source_status().

        Until wired, bist.py returns NOT_CONFIGURED for the matriks source.
        """
        raise NotImplementedError(
            "Matriks API client not yet implemented. "
            "See bist.py::_fetch_matriks() docstring for wiring instructions."
        )


# ---------------------------------------------------------------------------
# Signal helpers
# ---------------------------------------------------------------------------

def _rsi(series, period: int = 14):
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


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


def _compute_confidence(signals: dict) -> float:
    abs_vote = abs(
        signals["sma_signal"] + signals["rsi_signal"] + signals["bb_signal"]
    )
    base_conf = abs_vote / 3.0
    vol_boost = 0.10 if signals.get("vol_ratio", 1.0) > 1.5 else 0.0
    return min(base_conf + vol_boost, 1.0)


def _entry_range(signals: dict) -> tuple:
    close = signals["close"]
    return round(close * 0.995, 2), round(close * 1.005, 2)  # TRY prices, 2 decimals


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
