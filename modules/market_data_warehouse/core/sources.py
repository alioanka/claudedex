"""Fail-soft fetchers for FREE public market-data endpoints.

Only sources the bot already touches elsewhere (regime_allocator and the
futures module hit the same hosts): Binance spot klines, Binance USD-M funding
history, Bybit v5 linear klines + funding history. No API keys, public REST
only. Every fetcher returns a (possibly empty) list and NEVER raises — a dead
source is skipped, the ingest tick continues.

All raw rows are pushed through core.normalizer for validation + canonical
(source, symbol, timeframe, ts) keys before they reach the DB.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional

from modules.market_data_warehouse.core.normalizer import (
    Candle,
    SeriesPoint,
    dedup_candles,
    dedup_series,
    make_candle,
    make_series_point,
    normalize_timeframe,
    to_exchange_symbol,
)

logger = logging.getLogger("market_data_warehouse")

_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
_BINANCE_FUNDING = "https://fapi.binance.com/fapi/v1/fundingRate"
_BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
_BYBIT_FUNDING = "https://api.bybit.com/v5/market/funding/history"

_HTTP_TIMEOUT_S = 15

# Canonical timeframe -> per-venue interval spelling.
_BINANCE_INTERVALS = {"1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
                      "1h": "1h", "4h": "4h", "1d": "1d"}
_BYBIT_INTERVALS = {"1m": "1", "5m": "5", "15m": "15", "30m": "30",
                    "1h": "60", "4h": "240", "1d": "D"}


async def _get_json(session, url: str, params: dict) -> Optional[object]:
    """One GET -> parsed JSON, fail-soft to None (timeouts, 4xx/5xx, junk)."""
    try:
        async with session.get(url, params=params,
                               timeout=_HTTP_TIMEOUT_S) as resp:
            if resp.status != 200:
                logger.debug("source GET %s -> HTTP %d", url, resp.status)
                return None
            return await resp.json(content_type=None)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        logger.debug("source GET %s fail-soft: %s", url, exc)
        return None


# ───────────────────────────── candles ──────────────────────────────────────

async def fetch_binance_klines(session, symbol: str, timeframe: str,
                               limit: int = 200,
                               start_ms: Optional[int] = None) -> List[Candle]:
    """Binance spot klines (oldest-first arrays)."""
    tf = normalize_timeframe(timeframe)
    ex_sym = to_exchange_symbol(symbol)
    interval = _BINANCE_INTERVALS.get(tf or "")
    if ex_sym is None or interval is None:
        return []
    params = {"symbol": ex_sym, "interval": interval,
              "limit": max(1, min(int(limit), 1000))}
    if start_ms:
        params["startTime"] = int(start_ms)
    data = await _get_json(session, _BINANCE_KLINES, params)
    if not isinstance(data, list):
        return []
    out = []
    for row in data:
        if not isinstance(row, (list, tuple)) or len(row) < 8:
            continue
        # [openTimeMs, open, high, low, close, volume, closeTimeMs, quoteVol, ...]
        out.append(make_candle("binance", symbol, tf, row[0],
                               row[1], row[2], row[3], row[4], row[5],
                               quote_volume=row[7]))
    return dedup_candles(out)


async def fetch_bybit_klines(session, symbol: str, timeframe: str,
                             limit: int = 200,
                             start_ms: Optional[int] = None) -> List[Candle]:
    """Bybit v5 linear klines (newest-first arrays)."""
    tf = normalize_timeframe(timeframe)
    ex_sym = to_exchange_symbol(symbol)
    interval = _BYBIT_INTERVALS.get(tf or "")
    if ex_sym is None or interval is None:
        return []
    params = {"category": "linear", "symbol": ex_sym, "interval": interval,
              "limit": max(1, min(int(limit), 1000))}
    if start_ms:
        params["start"] = int(start_ms)
    data = await _get_json(session, _BYBIT_KLINES, params)
    rows = (((data or {}).get("result") or {}).get("list")
            if isinstance(data, dict) else None)
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, (list, tuple)) or len(row) < 7:
            continue
        # [startMs, open, high, low, close, volume, turnover]
        out.append(make_candle("bybit", symbol, tf, row[0],
                               row[1], row[2], row[3], row[4], row[5],
                               quote_volume=row[6]))
    return dedup_candles(out)


# ───────────────────────────── funding ──────────────────────────────────────

async def fetch_binance_funding(session, symbol: str, limit: int = 200,
                                start_ms: Optional[int] = None) -> List[SeriesPoint]:
    """Binance USD-M perp realized funding-rate history."""
    ex_sym = to_exchange_symbol(symbol)
    if ex_sym is None:
        return []
    params = {"symbol": ex_sym, "limit": max(1, min(int(limit), 1000))}
    if start_ms:
        params["startTime"] = int(start_ms)
    data = await _get_json(session, _BINANCE_FUNDING, params)
    if not isinstance(data, list):
        return []
    out = []
    for row in data:
        if not isinstance(row, dict):
            continue
        out.append(make_series_point("binance", symbol, "funding_rate",
                                     row.get("fundingTime"),
                                     row.get("fundingRate")))
    return dedup_series(out)


async def fetch_bybit_funding(session, symbol: str, limit: int = 200,
                              start_ms: Optional[int] = None) -> List[SeriesPoint]:
    """Bybit v5 linear perp funding-rate history (newest-first)."""
    ex_sym = to_exchange_symbol(symbol)
    if ex_sym is None:
        return []
    params = {"category": "linear", "symbol": ex_sym,
              "limit": max(1, min(int(limit), 200))}
    if start_ms:
        params["startTime"] = int(start_ms)
    data = await _get_json(session, _BYBIT_FUNDING, params)
    rows = (((data or {}).get("result") or {}).get("list")
            if isinstance(data, dict) else None)
    if not isinstance(rows, list):
        return []
    out = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        out.append(make_series_point("bybit", symbol, "funding_rate",
                                     row.get("fundingRateTimestamp"),
                                     row.get("fundingRate")))
    return dedup_series(out)


# Registry the engine iterates — adding a source is one entry + one fetcher.
CANDLE_FETCHERS = {
    "binance": fetch_binance_klines,
    "bybit": fetch_bybit_klines,
}
FUNDING_FETCHERS = {
    "binance": fetch_binance_funding,
    "bybit": fetch_bybit_funding,
}
