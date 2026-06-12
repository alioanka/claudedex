"""Read accessor for the market-data warehouse — THE consumer API.

Other modules (backtest_replay, regime_allocator, param_tuner, options_vol,
advisor/ML retraining) import THIS instead of re-fetching from exchanges or
querying the tables directly, so the schema can evolve behind one surface.

Every function takes an asyncpg connection OR pool as its first argument
(both expose fetch/fetchrow/fetchval) and is read-only + fail-soft: any DB
error returns an empty result, never raises into the caller's loop.

    from modules.market_data_warehouse import reader

    candles = await reader.get_candles(pool, 'BTC/USDT', '1h', limit=500)
    funding = await reader.get_funding(pool, 'BTC/USDT', limit=100)
    px      = await reader.get_latest_close(pool, 'SOL/USDT')
    cov     = await reader.get_coverage(pool, 'ETH/USDT', '1m')

Symbols accept any spelling ('BTCUSDT', 'btc/usdt', ...) — they are pushed
through the same normalizer the writer uses. Rows come back oldest-first as
plain dicts with epoch-second integer 'ts' keys (replay-friendly).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import List, Optional

from modules.market_data_warehouse.core.normalizer import (
    normalize_symbol,
    normalize_timeframe,
)

logger = logging.getLogger("market_data_warehouse")

_MAX_LIMIT = 10000

# When the caller does not pin a source, prefer these in order, then anything.
_PREFERRED_SOURCES = ("binance", "bybit")


def _clamp_limit(limit: int) -> int:
    try:
        return max(1, min(int(limit), _MAX_LIMIT))
    except (TypeError, ValueError):
        return 1000


def _to_dt(value) -> Optional[datetime]:
    """Accept datetime or epoch seconds; warehouse timestamps are UTC."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    try:
        return datetime.fromtimestamp(float(value), tz=timezone.utc)
    except (TypeError, ValueError, OSError):
        return None


async def _best_source(conn, table: str, symbol: str, *,
                       timeframe: Optional[str] = None,
                       metric: Optional[str] = None) -> Optional[str]:
    """Pick the source with the most rows for the key (preferred ones win
    ties), so unpinned reads are stable and dense."""
    try:
        if table == "market_candles":
            rows = await conn.fetch(
                "SELECT source, COUNT(*) AS n FROM market_candles "
                "WHERE symbol=$1 AND timeframe=$2 GROUP BY source",
                symbol, timeframe,
            )
        else:
            rows = await conn.fetch(
                "SELECT source, COUNT(*) AS n FROM market_series "
                "WHERE symbol=$1 AND metric=$2 GROUP BY source",
                symbol, metric,
            )
    except Exception as exc:
        logger.debug("_best_source fail-soft: %s", exc)
        return None
    if not rows:
        return None

    def rank(r):
        src = r["source"]
        pref = (_PREFERRED_SOURCES.index(src)
                if src in _PREFERRED_SOURCES else len(_PREFERRED_SOURCES))
        return (-int(r["n"]), pref, src)

    return sorted(rows, key=rank)[0]["source"]


async def get_candles(conn, symbol: str, timeframe: str = "1h", *,
                      source: Optional[str] = None,
                      start=None, end=None, limit: int = 1000) -> List[dict]:
    """OHLCV candles, oldest-first. start/end accept datetime or epoch secs.
    With no source pinned, the densest stored source is used (never mixed —
    one series, one venue). Returns [] on any error or no data."""
    sym = normalize_symbol(symbol)
    tf = normalize_timeframe(timeframe)
    if sym is None or tf is None:
        return []
    try:
        src = source or await _best_source(conn, "market_candles", sym,
                                           timeframe=tf)
        if src is None:
            return []
        clauses = ["source=$1", "symbol=$2", "timeframe=$3"]
        args: list = [src, sym, tf]
        start_dt, end_dt = _to_dt(start), _to_dt(end)
        if start_dt is not None:
            args.append(start_dt)
            clauses.append(f"ts >= ${len(args)}")
        if end_dt is not None:
            args.append(end_dt)
            clauses.append(f"ts <= ${len(args)}")
        args.append(_clamp_limit(limit))
        rows = await conn.fetch(
            "SELECT source, symbol, timeframe, ts, open, high, low, close, "
            "volume, quote_volume FROM market_candles "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY ts DESC LIMIT ${len(args)}",
            *args,
        )
    except Exception as exc:
        logger.debug("get_candles(%s,%s) fail-soft: %s", sym, tf, exc)
        return []
    out = [{
        "source": r["source"], "symbol": r["symbol"],
        "timeframe": r["timeframe"], "ts": int(r["ts"].timestamp()),
        "open": float(r["open"]), "high": float(r["high"]),
        "low": float(r["low"]), "close": float(r["close"]),
        "volume": float(r["volume"]),
        "quote_volume": (float(r["quote_volume"])
                         if r["quote_volume"] is not None else None),
    } for r in rows]
    out.reverse()  # newest-first query (limit grabs the tail) -> oldest-first
    return out


async def get_series(conn, symbol: str, metric: str, *,
                     source: Optional[str] = None,
                     start=None, end=None, limit: int = 1000) -> List[dict]:
    """Scalar series points (e.g. metric='funding_rate'), oldest-first."""
    sym = normalize_symbol(symbol)
    if sym is None or not metric:
        return []
    try:
        src = source or await _best_source(conn, "market_series", sym,
                                           metric=metric)
        if src is None:
            return []
        clauses = ["source=$1", "symbol=$2", "metric=$3"]
        args: list = [src, sym, metric]
        start_dt, end_dt = _to_dt(start), _to_dt(end)
        if start_dt is not None:
            args.append(start_dt)
            clauses.append(f"ts >= ${len(args)}")
        if end_dt is not None:
            args.append(end_dt)
            clauses.append(f"ts <= ${len(args)}")
        args.append(_clamp_limit(limit))
        rows = await conn.fetch(
            "SELECT source, symbol, metric, ts, value FROM market_series "
            f"WHERE {' AND '.join(clauses)} "
            f"ORDER BY ts DESC LIMIT ${len(args)}",
            *args,
        )
    except Exception as exc:
        logger.debug("get_series(%s,%s) fail-soft: %s", sym, metric, exc)
        return []
    out = [{
        "source": r["source"], "symbol": r["symbol"], "metric": r["metric"],
        "ts": int(r["ts"].timestamp()), "value": float(r["value"]),
    } for r in rows]
    out.reverse()
    return out


async def get_funding(conn, symbol: str, *, source: Optional[str] = None,
                      start=None, end=None, limit: int = 1000) -> List[dict]:
    """Realized perp funding-rate history (convenience wrapper)."""
    return await get_series(conn, symbol, "funding_rate", source=source,
                            start=start, end=end, limit=limit)


async def get_latest_close(conn, symbol: str, timeframe: str = "1m", *,
                           source: Optional[str] = None) -> Optional[float]:
    """Most recent stored close. NOT a live ticker — staleness is bounded by
    the ingest interval; callers needing freshness must check get_coverage."""
    candles = await get_candles(conn, symbol, timeframe,
                                source=source, limit=1)
    return candles[-1]["close"] if candles else None


async def get_coverage(conn, symbol: str, timeframe: str = "1h", *,
                       source: Optional[str] = None) -> dict:
    """What history exists for a key: rows + first/last ts (epoch seconds) per
    source (or just the pinned one). Consumers MUST check this before trusting
    a backtest window — a gap-free warehouse is not guaranteed."""
    sym = normalize_symbol(symbol)
    tf = normalize_timeframe(timeframe)
    empty = {"symbol": sym, "timeframe": tf, "sources": {}}
    if sym is None or tf is None:
        return empty
    try:
        if source:
            rows = await conn.fetch(
                "SELECT source, COUNT(*) AS n, MIN(ts) AS first_ts, "
                "MAX(ts) AS last_ts FROM market_candles "
                "WHERE symbol=$1 AND timeframe=$2 AND source=$3 GROUP BY source",
                sym, tf, source,
            )
        else:
            rows = await conn.fetch(
                "SELECT source, COUNT(*) AS n, MIN(ts) AS first_ts, "
                "MAX(ts) AS last_ts FROM market_candles "
                "WHERE symbol=$1 AND timeframe=$2 GROUP BY source",
                sym, tf,
            )
    except Exception as exc:
        logger.debug("get_coverage(%s,%s) fail-soft: %s", sym, tf, exc)
        return empty
    return {
        "symbol": sym, "timeframe": tf,
        "sources": {
            r["source"]: {
                "rows": int(r["n"]),
                "first_ts": int(r["first_ts"].timestamp()),
                "last_ts": int(r["last_ts"].timestamp()),
            } for r in rows
        },
    }


async def list_symbols(conn) -> List[dict]:
    """Distinct (symbol, timeframe, source) keys currently stored."""
    try:
        rows = await conn.fetch(
            "SELECT symbol, timeframe, source, COUNT(*) AS n "
            "FROM market_candles GROUP BY symbol, timeframe, source "
            "ORDER BY symbol, timeframe, source"
        )
    except Exception as exc:
        logger.debug("list_symbols fail-soft: %s", exc)
        return []
    return [{"symbol": r["symbol"], "timeframe": r["timeframe"],
             "source": r["source"], "rows": int(r["n"])} for r in rows]
