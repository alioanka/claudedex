"""
vbt_backtest.py -- vectorbt-based backtesting helper for the advisor module.

Purpose
-------
Provides fast, indicator-aware backtesting of advisor signals using vectorbt
(pip install vectorbt). This is a LIGHTWEIGHT OPTIONAL complement to the
existing portfolio_engine.backtest() method.

When to use which:
  portfolio_engine.backtest()  -- Simple replay of a single past advice
                                  against free yfinance prices. No vectorbt
                                  required. Good for: "did this specific signal
                                  work out?". Runs without extra deps.

  vbt_backtest.run_strategy()  -- Full vectorbt backtest over a date range
                                  with SMA/RSI/BB signals, parameterised
                                  strategy, equity curve, Sharpe, max
                                  drawdown. Requires vectorbt. Good for:
                                  "how would this strategy have performed
                                  across 2y of data?".

Import guard
------------
vectorbt is listed in requirements.txt as optional (heavy dep: ~500MB with
numpy/pandas/scipy). The advisor module runs perfectly without it. If
vectorbt is not installed:
  - run_strategy() returns a BacktestUnavailableResult with a clear message.
  - The advice loop never crashes.

Supported data sources
----------------------
  yfinance   -- US equities, BIST (.IS), FX (default, no key needed)
  borsapy    -- BIST via TradingView feed (import-guarded like bist.py)
  ccxt       -- Crypto OHLCV (requires ccxt, pass exchange name)

ADVICE-ONLY. This module reads market data and returns backtest statistics.
It NEVER executes trades.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("advisor.vbt_backtest")


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class VbtBacktestResult:
    """
    Result of a vectorbt backtest run.

    Fields
    ------
    symbol          : Ticker string (e.g. 'THYAO', 'BTC/USDT', 'AAPL').
    strategy        : Strategy name (e.g. 'sma_crossover', 'rsi_bb').
    start_date      : Backtest start date.
    end_date        : Backtest end date (or last data date).
    total_return_pct: Total strategy return (%) over the period.
    sharpe_ratio    : Annualised Sharpe ratio (assumes 0% risk-free rate).
    max_drawdown_pct: Maximum peak-to-trough drawdown (%).
    total_trades    : Number of round-trip trades executed.
    win_rate_pct    : Percentage of winning trades.
    equity_curve    : List of cumulative portfolio values (daily).
    trade_log       : List of dict per trade (entry/exit date, price, pnl).
    data_source     : Which source provided price data.
    available       : True (vectorbt ran), False (lib unavailable).
    error_note      : Human-readable note if available=False.
    """
    symbol: str
    strategy: str
    start_date: str
    end_date: str
    total_return_pct: float = 0.0
    sharpe_ratio: float = 0.0
    max_drawdown_pct: float = 0.0
    total_trades: int = 0
    win_rate_pct: float = 0.0
    equity_curve: List[float] = field(default_factory=list)
    trade_log: List[Dict[str, Any]] = field(default_factory=list)
    data_source: str = ""
    available: bool = True
    error_note: str = ""


@dataclass
class BacktestUnavailableResult(VbtBacktestResult):
    """Returned when vectorbt is not installed or data fetch fails."""
    available: bool = False


# ---------------------------------------------------------------------------
# Import guard
# ---------------------------------------------------------------------------

def _vectorbt_available() -> bool:
    """Return True if vectorbt can be imported (does NOT test functionality)."""
    try:
        import vectorbt as vbt  # noqa: F401
        return True
    except ImportError:
        return False


_VECTORBT_UNAVAILABLE_NOTE = (
    "vectorbt is not installed -- backtest unavailable. "
    "To enable: pip install vectorbt (rebuild Docker image if containerised). "
    "Note: vectorbt is a heavy optional dependency (~500 MB with deps). "
    "The advisor module runs correctly without it. "
    "The simple portfolio_engine.backtest() remains available without vectorbt."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def run_strategy(
    symbol: str,
    strategy: str = "sma_crossover",
    *,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    lookback_days: int = 365,
    data_source: str = "yfinance",
    exchange: str = "binance",  # for ccxt crypto source
    fast_period: int = 20,
    slow_period: int = 50,
    rsi_period: int = 14,
    rsi_upper: float = 70.0,
    rsi_lower: float = 30.0,
    bb_period: int = 20,
    bb_std: float = 2.0,
    initial_cash: float = 10_000.0,
    fees: float = 0.001,  # 0.1% per trade (Midas/exchange fee estimate)
) -> VbtBacktestResult:
    """
    Run a vectorbt backtest for a given symbol and strategy.

    Parameters
    ----------
    symbol       : Ticker string. Format depends on data_source:
                     yfinance: 'AAPL', 'THYAO.IS', 'EURUSD=X'
                     borsapy : 'THYAO', 'GARAN' (bare, no .IS)
                     ccxt    : 'BTC/USDT', 'ETH/USDT'
    strategy     : One of:
                     'sma_crossover' -- buy fast>slow, sell fast<slow
                     'rsi'           -- buy RSI<lower, sell RSI>upper
                     'bb'            -- buy price<lower_band, sell price>upper_band
                     'sma_rsi'       -- combined SMA + RSI signals
    start_date   : ISO date string 'YYYY-MM-DD'. Defaults to lookback_days ago.
    end_date     : ISO date string 'YYYY-MM-DD'. Defaults to today.
    lookback_days: Days of history to fetch (ignored if start_date is set).
    data_source  : 'yfinance' | 'borsapy' | 'ccxt'
    exchange     : ccxt exchange name (only for data_source='ccxt').
    fast_period  : SMA fast window (sma_crossover, sma_rsi strategies).
    slow_period  : SMA slow window.
    rsi_period   : RSI lookback period.
    rsi_upper    : RSI overbought threshold (sell signal).
    rsi_lower    : RSI oversold threshold (buy signal).
    bb_period    : Bollinger Band period.
    bb_std       : Bollinger Band standard deviation multiplier.
    initial_cash : Starting portfolio value in USD.
    fees         : Round-trip fee rate (fraction, e.g. 0.001 = 0.1%).

    Returns
    -------
    VbtBacktestResult on success.
    BacktestUnavailableResult if vectorbt is not installed or data fails.

    Never raises -- exceptions are caught and returned as BacktestUnavailableResult.
    """
    if not _vectorbt_available():
        return BacktestUnavailableResult(
            symbol=symbol,
            strategy=strategy,
            start_date=start_date or "",
            end_date=end_date or "",
            error_note=_VECTORBT_UNAVAILABLE_NOTE,
        )

    try:
        return _run_strategy_internal(
            symbol=symbol,
            strategy=strategy,
            start_date=start_date,
            end_date=end_date,
            lookback_days=lookback_days,
            data_source=data_source,
            exchange=exchange,
            fast_period=fast_period,
            slow_period=slow_period,
            rsi_period=rsi_period,
            rsi_upper=rsi_upper,
            rsi_lower=rsi_lower,
            bb_period=bb_period,
            bb_std=bb_std,
            initial_cash=initial_cash,
            fees=fees,
        )
    except Exception as exc:
        logger.error(
            "[vbt_backtest] run_strategy(%r, strategy=%r) failed: %s",
            symbol, strategy, exc, exc_info=True,
        )
        return BacktestUnavailableResult(
            symbol=symbol,
            strategy=strategy,
            start_date=start_date or "",
            end_date=end_date or "",
            error_note=f"Backtest error: {exc!s:.400}",
        )


# ---------------------------------------------------------------------------
# Internal implementation
# ---------------------------------------------------------------------------

def _run_strategy_internal(
    symbol: str,
    strategy: str,
    *,
    start_date: Optional[str],
    end_date: Optional[str],
    lookback_days: int,
    data_source: str,
    exchange: str,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_upper: float,
    rsi_lower: float,
    bb_period: int,
    bb_std: float,
    initial_cash: float,
    fees: float,
) -> VbtBacktestResult:
    """
    Core backtest logic. Called only when vectorbt is available.
    """
    import vectorbt as vbt  # confirmed importable by caller

    # --- Resolve date range ---
    end_dt = (
        datetime.strptime(end_date, "%Y-%m-%d")
        if end_date
        else datetime.utcnow()
    )
    start_dt = (
        datetime.strptime(start_date, "%Y-%m-%d")
        if start_date
        else end_dt - timedelta(days=lookback_days)
    )
    start_str = start_dt.strftime("%Y-%m-%d")
    end_str = end_dt.strftime("%Y-%m-%d")

    # --- Fetch OHLCV ---
    close = _fetch_close_series(
        symbol=symbol,
        data_source=data_source,
        exchange=exchange,
        start_str=start_str,
        end_str=end_str,
    )

    if close is None or close.empty or len(close) < 30:
        return BacktestUnavailableResult(
            symbol=symbol,
            strategy=strategy,
            start_date=start_str,
            end_date=end_str,
            data_source=data_source,
            error_note=(
                f"Insufficient price data for {symbol!r} from {data_source!r} "
                f"({start_str} to {end_str}). "
                f"Got {0 if close is None else len(close)} rows, need 30+."
            ),
        )

    # --- Build entry/exit signals ---
    entries, exits = _build_signals(
        close=close,
        strategy=strategy,
        fast_period=fast_period,
        slow_period=slow_period,
        rsi_period=rsi_period,
        rsi_upper=rsi_upper,
        rsi_lower=rsi_lower,
        bb_period=bb_period,
        bb_std=bb_std,
    )

    if entries is None or exits is None:
        return BacktestUnavailableResult(
            symbol=symbol,
            strategy=strategy,
            start_date=start_str,
            end_date=end_str,
            data_source=data_source,
            error_note=f"Unknown strategy '{strategy}'. "
                       "Valid: sma_crossover, rsi, bb, sma_rsi.",
        )

    # --- Run vectorbt portfolio ---
    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        init_cash=initial_cash,
        fees=fees,
        freq="D",
    )

    # --- Extract stats ---
    stats = pf.stats()
    total_return = float(pf.total_return() * 100)
    sharpe = _safe_float(stats, "Sharpe Ratio", 0.0)
    max_dd = _safe_float(stats, "Max Drawdown [%]", 0.0)
    n_trades = int(_safe_float(stats, "Total Trades", 0))
    win_rate = float(_safe_float(stats, "Win Rate [%]", 0.0))

    # Equity curve (cumulative portfolio value, daily)
    try:
        equity = pf.value().tolist()
    except Exception:
        equity = []

    # Trade log (simplified)
    trade_log: List[Dict[str, Any]] = []
    try:
        trades_df = pf.trades.records_readable
        for _, row in trades_df.iterrows():
            trade_log.append({
                "entry_date": str(row.get("Entry Timestamp", "")),
                "exit_date": str(row.get("Exit Timestamp", "")),
                "entry_price": float(row.get("Avg Entry Price", 0)),
                "exit_price": float(row.get("Avg Exit Price", 0)),
                "pnl_pct": float(row.get("Return [%]", 0)),
                "status": str(row.get("Status", "")),
            })
    except Exception:
        pass  # trade log is best-effort

    return VbtBacktestResult(
        symbol=symbol,
        strategy=strategy,
        start_date=start_str,
        end_date=end_str,
        total_return_pct=round(total_return, 2),
        sharpe_ratio=round(sharpe, 3),
        max_drawdown_pct=round(abs(max_dd), 2),
        total_trades=n_trades,
        win_rate_pct=round(win_rate, 1),
        equity_curve=equity,
        trade_log=trade_log,
        data_source=data_source,
        available=True,
    )


def _fetch_close_series(
    symbol: str,
    data_source: str,
    exchange: str,
    start_str: str,
    end_str: str,
):
    """
    Fetch a close price Series for the given symbol and data source.
    Returns a pandas Series indexed by date, or None on failure.
    """
    if data_source == "yfinance":
        return _fetch_yfinance_close(symbol, start_str, end_str)
    if data_source == "borsapy":
        return _fetch_borsapy_close(symbol, start_str, end_str)
    if data_source == "ccxt":
        return _fetch_ccxt_close(symbol, exchange, start_str, end_str)

    # Default fallback: yfinance
    logger.warning(
        "[vbt_backtest] Unknown data_source %r -- falling back to yfinance.",
        data_source,
    )
    return _fetch_yfinance_close(symbol, start_str, end_str)


def _fetch_yfinance_close(symbol: str, start_str: str, end_str: str):
    """Fetch close prices via yfinance. Runs synchronously."""
    try:
        import yfinance as yf
        import pandas as pd
    except ImportError:
        logger.warning("[vbt_backtest] yfinance not installed.")
        return None

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(
            start=start_str, end=end_str, interval="1d", auto_adjust=True
        )
        if df is None or df.empty:
            return None
        return df["Close"].rename(symbol)
    except Exception as exc:
        logger.warning(
            "[vbt_backtest] yfinance fetch for %r failed: %s", symbol, exc
        )
        return None


def _fetch_borsapy_close(symbol: str, start_str: str, end_str: str):
    """
    Fetch close prices via borsapy (TradingView-backed, ~15min delayed).
    Import-guarded: returns None if borsapy not installed.
    """
    try:
        from borsapy import Client as BorsapyClient
        import pandas as pd
    except ImportError:
        logger.debug(
            "[vbt_backtest] borsapy not installed. "
            "pip install borsapy or use data_source='yfinance'."
        )
        return None

    # Strip .IS suffix for borsapy
    bare = symbol.upper()
    if bare.endswith(".IS"):
        bare = bare[:-3]

    # Estimate period from date range
    try:
        delta = datetime.strptime(end_str, "%Y-%m-%d") - datetime.strptime(
            start_str, "%Y-%m-%d"
        )
        days = delta.days
    except Exception:
        days = 365

    if days <= 30:
        period = "1mo"
    elif days <= 90:
        period = "3mo"
    elif days <= 180:
        period = "6mo"
    else:
        period = "2y"

    try:
        client = BorsapyClient()
        df = client.get_history(bare, period=period, interval="1d")
        if df is None or df.empty:
            return None
        df.columns = [c.lower() for c in df.columns]
        if "close" not in df.columns:
            return None
        # Filter to requested date range
        close = df["close"]
        close.index = pd.to_datetime(close.index)
        close = close[
            (close.index >= pd.Timestamp(start_str))
            & (close.index <= pd.Timestamp(end_str))
        ]
        return close.rename(symbol)
    except Exception as exc:
        logger.warning(
            "[vbt_backtest] borsapy fetch for %r failed: %s", bare, exc
        )
        return None


def _fetch_ccxt_close(symbol: str, exchange: str, start_str: str, end_str: str):
    """
    Fetch close prices via ccxt (crypto OHLCV).
    Import-guarded: returns None if ccxt not installed.
    Runs synchronously (ccxt sync API).
    """
    try:
        import ccxt
        import pandas as pd
    except ImportError:
        logger.warning(
            "[vbt_backtest] ccxt not installed. "
            "pip install ccxt for crypto backtesting."
        )
        return None

    try:
        ex = getattr(ccxt, exchange)({
            "enableRateLimit": True,
        })
        since_ms = int(
            datetime.strptime(start_str, "%Y-%m-%d").timestamp() * 1000
        )
        ohlcv = ex.fetch_ohlcv(symbol, timeframe="1d", since=since_ms, limit=1000)
        if not ohlcv:
            return None
        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms")
        df = df.set_index("date")
        close = df["close"]
        close = close[close.index <= pd.Timestamp(end_str)]
        return close.rename(symbol)
    except Exception as exc:
        logger.warning(
            "[vbt_backtest] ccxt fetch for %r on %r failed: %s",
            symbol, exchange, exc,
        )
        return None


def _build_signals(
    close,
    strategy: str,
    fast_period: int,
    slow_period: int,
    rsi_period: int,
    rsi_upper: float,
    rsi_lower: float,
    bb_period: int,
    bb_std: float,
):
    """
    Build entry/exit boolean Series for the chosen strategy.
    Returns (entries, exits) or (None, None) if strategy unknown.
    """
    if strategy == "sma_crossover":
        return _sma_crossover_signals(close, fast_period, slow_period)
    if strategy == "rsi":
        return _rsi_signals(close, rsi_period, rsi_upper, rsi_lower)
    if strategy == "bb":
        return _bb_signals(close, bb_period, bb_std)
    if strategy == "sma_rsi":
        return _sma_rsi_combined(
            close, fast_period, slow_period,
            rsi_period, rsi_upper, rsi_lower,
        )
    return None, None


def _sma_crossover_signals(close, fast: int, slow: int):
    """Buy when fast SMA crosses above slow SMA; sell when it crosses below."""
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    entries = (sma_fast > sma_slow) & (sma_fast.shift(1) <= sma_slow.shift(1))
    exits = (sma_fast < sma_slow) & (sma_fast.shift(1) >= sma_slow.shift(1))
    return entries.fillna(False), exits.fillna(False)


def _rsi_signals(close, period: int, upper: float, lower: float):
    """Buy on RSI < lower; exit on RSI > upper."""
    rsi = _compute_rsi(close, period)
    entries = rsi < lower
    exits = rsi > upper
    return entries.fillna(False), exits.fillna(False)


def _bb_signals(close, period: int, n_std: float):
    """Buy when price crosses below lower BB; sell when it crosses above upper BB."""
    sma = close.rolling(period).mean()
    std = close.rolling(period).std()
    lower_band = sma - n_std * std
    upper_band = sma + n_std * std
    entries = close < lower_band
    exits = close > upper_band
    return entries.fillna(False), exits.fillna(False)


def _sma_rsi_combined(
    close, fast: int, slow: int, rsi_period: int, upper: float, lower: float
):
    """
    Combined SMA + RSI strategy.
    Entry: fast SMA > slow SMA AND RSI < lower (oversold in uptrend).
    Exit : fast SMA < slow SMA OR RSI > upper (overbought).
    """
    sma_fast = close.rolling(fast).mean()
    sma_slow = close.rolling(slow).mean()
    rsi = _compute_rsi(close, rsi_period)

    entries = (sma_fast > sma_slow) & (rsi < lower)
    exits = (sma_fast < sma_slow) | (rsi > upper)
    return entries.fillna(False), exits.fillna(False)


def _compute_rsi(series, period: int = 14):
    """RSI computation (pure pandas, no TA-Lib dependency)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def _safe_float(stats, key: str, default: float) -> float:
    """Safely extract a float from vectorbt stats dict/Series."""
    try:
        val = stats[key]
        if val is None:
            return default
        return float(val)
    except (KeyError, TypeError, ValueError):
        return default
