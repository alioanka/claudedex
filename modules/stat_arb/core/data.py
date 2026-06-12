"""Free price-data layer for stat_arb. Fail-soft everywhere, no keys.

Source order per symbol:
  1. market_data_warehouse reader API (if the module/tables exist) — ONE
     consistent stored history, never mixed venues within a series.
  2. Bybit v5 public linear klines (free REST, no key).
  3. Binance USD-M public klines (free REST, no key).

Returns CLOSED-bar closes only, oldest-first. The last (in-progress) kline is
dropped from the public-REST paths so the engine never sees a forming bar
(look-ahead control). Any failure returns [] — the engine skips the symbol.
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import List, Optional

import aiohttp

logger = logging.getLogger("StatArbModule.Data")

_BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
_BINANCE_FAPI_KLINE_URL = "https://fapi.binance.com/fapi/v1/klines"

# timeframe -> (bybit interval, binance interval, seconds per bar)
_TF = {
    "15m": ("15", "15m", 900),
    "30m": ("30", "30m", 1800),
    "1h": ("60", "1h", 3600),
    "4h": ("240", "4h", 14400),
    "1d": ("D", "1d", 86400),
}


def compact_symbol(symbol: str) -> str:
    """'BTC/USDT' -> 'BTCUSDT' (exchange spelling)."""
    return symbol.replace("/", "").replace("-", "").upper().strip()


def bars_to_seconds(timeframe: str, bars: float) -> Optional[float]:
    tf = _TF.get(timeframe)
    return tf[2] * bars if tf else None


async def _warehouse_closes(db_pool, symbol: str, timeframe: str,
                            limit: int) -> List[float]:
    """Warehouse reader path. Empty on any miss; requires fresh-enough data."""
    if db_pool is None:
        return []
    try:
        from modules.market_data_warehouse import reader
    except Exception:
        return []
    try:
        candles = await reader.get_candles(db_pool, symbol, timeframe,
                                           limit=limit)
    except Exception as exc:
        logger.debug("warehouse read failed for %s: %s", symbol, exc)
        return []
    if len(candles) < limit:
        return []
    sec = _TF.get(timeframe, (None, None, 3600))[2]
    if time.time() - candles[-1]["ts"] > 3 * sec:
        return []  # stale tail: warehouse ingest is behind; use REST instead
    return [c["close"] for c in candles]


async def _bybit_closes(session: aiohttp.ClientSession, symbol: str,
                        timeframe: str, limit: int) -> List[float]:
    tf = _TF.get(timeframe)
    if tf is None:
        return []
    params = {"category": "linear", "symbol": compact_symbol(symbol),
              "interval": tf[0], "limit": min(limit + 1, 1000)}
    try:
        async with session.get(_BYBIT_KLINE_URL, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                return []
            body = await r.json(content_type=None)
        rows = (body or {}).get("result", {}).get("list") or []
        # Bybit returns newest-first: [startTime, open, high, low, close, ...]
        closes = [float(row[4]) for row in reversed(rows)]
        return closes[:-1][-limit:] if len(closes) > 1 else []  # drop forming bar
    except Exception as exc:
        logger.debug("bybit klines failed for %s: %s", symbol, exc)
        return []


async def _binance_closes(session: aiohttp.ClientSession, symbol: str,
                          timeframe: str, limit: int) -> List[float]:
    tf = _TF.get(timeframe)
    if tf is None:
        return []
    params = {"symbol": compact_symbol(symbol), "interval": tf[1],
              "limit": min(limit + 1, 1000)}
    try:
        async with session.get(_BINANCE_FAPI_KLINE_URL, params=params,
                               timeout=aiohttp.ClientTimeout(total=10)) as r:
            if r.status != 200:
                return []
            rows = await r.json(content_type=None)
        closes = [float(row[4]) for row in (rows or [])]  # oldest-first
        return closes[:-1][-limit:] if len(closes) > 1 else []  # drop forming bar
    except Exception as exc:
        logger.debug("binance klines failed for %s: %s", symbol, exc)
        return []


class PriceFeed:
    """Per-cycle close-series fetcher with one shared session + request spacing."""

    def __init__(self, db_pool, *, request_spacing_s: float = 0.25):
        self.db_pool = db_pool
        self.request_spacing_s = max(0.0, float(request_spacing_s))
        self._session: Optional[aiohttp.ClientSession] = None
        self.last_source: dict = {}   # symbol -> 'warehouse'|'bybit'|'binance'

    async def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def get_closes(self, symbol: str, timeframe: str,
                         limit: int) -> List[float]:
        """Closed-bar closes, oldest-first, exactly `limit` long or []."""
        closes = await _warehouse_closes(self.db_pool, symbol, timeframe, limit)
        if len(closes) >= limit:
            self.last_source[symbol] = "warehouse"
            return closes[-limit:]
        session = await self._ensure_session()
        await asyncio.sleep(self.request_spacing_s)  # politeness on free APIs
        closes = await _bybit_closes(session, symbol, timeframe, limit)
        if len(closes) >= limit:
            self.last_source[symbol] = "bybit"
            return closes[-limit:]
        await asyncio.sleep(self.request_spacing_s)
        closes = await _binance_closes(session, symbol, timeframe, limit)
        if len(closes) >= limit:
            self.last_source[symbol] = "binance"
            return closes[-limit:]
        self.last_source[symbol] = "none"
        return []
