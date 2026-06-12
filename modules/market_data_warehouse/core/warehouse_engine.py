"""MARKET_DATA_WAREHOUSE engine — slow ingest loop. PURE DATA, never trades.

One tick:
  1. Load config (config_type='market_data_warehouse').
  2. For each enabled candle source x symbol x timeframe: read the warehouse's
     own high-water mark (MAX(ts)), fetch only what is new (minus one bar so
     the previously in-progress candle gets corrected), normalize, dedup,
     UPSERT on the unique (source,symbol,timeframe,ts) key.
  3. Same for funding-rate history into market_series (INSERT .. DO NOTHING —
     realized funding is immutable).
  4. Disk discipline: purge candles/series past their retention windows.
     Fine-grained 1m candles get a short window (the coarser timeframes ARE
     the downsampled history and are kept much longer).

Fail-soft everywhere: a dead source / bad symbol / DB hiccup is logged and
skipped; the tick continues. A per-tick request budget + inter-request spacing
keeps us polite on free public endpoints. No paid LLM, no keys, no orders.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional

from modules.market_data_warehouse.core.normalizer import (
    Candle,
    SeriesPoint,
    normalize_symbol,
    normalize_timeframe,
    timeframe_seconds,
)
from modules.market_data_warehouse.core.sources import (
    CANDLE_FETCHERS,
    FUNDING_FETCHERS,
)

logger = logging.getLogger("market_data_warehouse")

_PAUSE_FLAG = Path("logs/.pause_market_data_warehouse")
_KILLSWITCH = Path("logs/.killswitch")

_DEFAULTS = {
    "ingest_interval_seconds": 300,
    "symbols": "BTC/USDT,ETH/USDT,SOL/USDT",
    "candle_timeframes": "1m,1h",
    "candle_sources": "binance,bybit",
    "funding_enabled": True,
    "funding_sources": "binance,bybit",
    "candles_per_request": 200,
    "max_requests_per_tick": 40,
    "request_spacing_seconds": 0.35,
    "retention_days_1m": 30,
    "retention_days_default": 365,
    "retention_days_series": 365,
    "purge_enabled": True,
}


# ───────────────────────────── config ──────────────────────────────────────

async def load_warehouse_config(conn) -> dict:
    """config_settings rows (config_type='market_data_warehouse') -> typed
    dict over safe defaults. Fail-soft: DB error returns pure defaults."""
    out = dict(_DEFAULTS)
    try:
        rows = await conn.fetch(
            "SELECT key, value FROM config_settings "
            "WHERE config_type='market_data_warehouse'"
        )
        for r in rows:
            v = r["value"]
            if isinstance(v, str):
                low = v.lower()
                if low in ("true", "false"):
                    v = low == "true"
                else:
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
            out[r["key"]] = v
    except Exception as exc:
        logger.warning("load_warehouse_config fail-soft (defaults): %s", exc)
    return out


def _csv(value, fallback: str) -> List[str]:
    raw = value if isinstance(value, str) and value.strip() else fallback
    return [p.strip() for p in raw.split(",") if p.strip()]


# ───────────────────────────── persistence ─────────────────────────────────

def _ts_dt(epoch_seconds: int) -> datetime:
    return datetime.fromtimestamp(epoch_seconds, tz=timezone.utc)


async def _candle_high_water(conn, source: str, symbol: str,
                             timeframe: str) -> Optional[datetime]:
    return await conn.fetchval(
        "SELECT MAX(ts) FROM market_candles "
        "WHERE source=$1 AND symbol=$2 AND timeframe=$3",
        source, symbol, timeframe,
    )


async def _series_high_water(conn, source: str, symbol: str,
                             metric: str) -> Optional[datetime]:
    return await conn.fetchval(
        "SELECT MAX(ts) FROM market_series "
        "WHERE source=$1 AND symbol=$2 AND metric=$3",
        source, symbol, metric,
    )


async def _upsert_candles(conn, candles: List[Candle]) -> int:
    """Idempotent upsert. The newest fetch wins so the previously in-progress
    bar is corrected on the next tick."""
    if not candles:
        return 0
    await conn.executemany(
        "INSERT INTO market_candles "
        "(source, symbol, timeframe, ts, open, high, low, close, volume, "
        " quote_volume, ingested_at) "
        "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10, NOW()) "
        "ON CONFLICT (source, symbol, timeframe, ts) DO UPDATE SET "
        "open=EXCLUDED.open, high=EXCLUDED.high, low=EXCLUDED.low, "
        "close=EXCLUDED.close, volume=EXCLUDED.volume, "
        "quote_volume=EXCLUDED.quote_volume, ingested_at=NOW()",
        [(c.source, c.symbol, c.timeframe, _ts_dt(c.ts), c.open, c.high,
          c.low, c.close, c.volume, c.quote_volume) for c in candles],
    )
    return len(candles)


async def _insert_series(conn, points: List[SeriesPoint]) -> int:
    """Realized series history is immutable -> DO NOTHING on conflict."""
    if not points:
        return 0
    await conn.executemany(
        "INSERT INTO market_series (source, symbol, metric, ts, value, ingested_at) "
        "VALUES ($1,$2,$3,$4,$5, NOW()) "
        "ON CONFLICT (source, symbol, metric, ts) DO NOTHING",
        [(p.source, p.symbol, p.metric, _ts_dt(p.ts), p.value) for p in points],
    )
    return len(points)


async def _purge(conn, cfg: dict) -> int:
    """Retention: short window for 1m candles (coarser timeframes ARE the
    long-lived downsample), long window for everything else."""
    purged = 0
    try:
        days_1m = max(1, int(cfg.get("retention_days_1m", 30)))
        days_default = max(1, int(cfg.get("retention_days_default", 365)))
        days_series = max(1, int(cfg.get("retention_days_series", 365)))
        r1 = await conn.execute(
            "DELETE FROM market_candles WHERE timeframe='1m' "
            "AND ts < NOW() - $1::interval", timedelta(days=days_1m),
        )
        r2 = await conn.execute(
            "DELETE FROM market_candles WHERE timeframe <> '1m' "
            "AND ts < NOW() - $1::interval", timedelta(days=days_default),
        )
        r3 = await conn.execute(
            "DELETE FROM market_series WHERE ts < NOW() - $1::interval",
            timedelta(days=days_series),
        )
        for r in (r1, r2, r3):
            try:
                purged += int(str(r).rsplit(" ", 1)[-1])
            except (ValueError, IndexError):
                pass
    except Exception as exc:
        logger.warning("retention purge fail-soft: %s", exc)
    return purged


# ───────────────────────────── tick + loop ─────────────────────────────────

class _Budget:
    """Per-tick request budget + polite spacing between public-API calls."""

    def __init__(self, max_requests: int, spacing_s: float):
        self.left = max(1, int(max_requests))
        self.spacing_s = max(0.0, float(spacing_s))

    async def take(self) -> bool:
        if self.left <= 0:
            return False
        self.left -= 1
        if self.spacing_s:
            await asyncio.sleep(self.spacing_s)
        return True


async def run_tick(pool, cfg: dict) -> dict:
    """One ingest cycle. Returns a summary dict (surfaced on /status)."""
    try:
        import aiohttp
    except Exception as exc:  # pure-data module: no aiohttp -> no-op tick
        logger.warning("aiohttp unavailable, tick skipped: %s", exc)
        return {"candles_upserted": 0, "series_inserted": 0, "errors": 1,
                "skipped": "aiohttp-missing"}

    symbols = [s for s in (normalize_symbol(x) for x in
                           _csv(cfg.get("symbols"), _DEFAULTS["symbols"]))
               if s is not None]
    timeframes = [t for t in (normalize_timeframe(x) for x in
                              _csv(cfg.get("candle_timeframes"),
                                   _DEFAULTS["candle_timeframes"]))
                  if t is not None]
    candle_sources = [s for s in _csv(cfg.get("candle_sources"),
                                      _DEFAULTS["candle_sources"])
                      if s in CANDLE_FETCHERS]
    funding_sources = [s for s in _csv(cfg.get("funding_sources"),
                                       _DEFAULTS["funding_sources"])
                       if s in FUNDING_FETCHERS]
    limit = max(1, int(cfg.get("candles_per_request", 200)))
    budget = _Budget(cfg.get("max_requests_per_tick", 40),
                     cfg.get("request_spacing_seconds", 0.35))

    summary: Dict = {"candles_upserted": 0, "series_inserted": 0,
                     "purged": 0, "errors": 0, "budget_exhausted": False,
                     "symbols": len(symbols), "timeframes": timeframes,
                     "candle_sources": candle_sources,
                     "funding_sources": funding_sources}

    async with aiohttp.ClientSession() as session:
        async with pool.acquire() as conn:
            # 1) candles
            for source in candle_sources:
                fetch = CANDLE_FETCHERS[source]
                for symbol in symbols:
                    for tf in timeframes:
                        if not await budget.take():
                            summary["budget_exhausted"] = True
                            break
                        try:
                            hw = await _candle_high_water(conn, source, symbol, tf)
                            start_ms = None
                            if hw is not None:
                                # re-fetch from one bar back so the previously
                                # open candle gets its final OHLCV
                                secs = timeframe_seconds(tf) or 60
                                start_ms = int((hw.timestamp() - secs) * 1000)
                            candles = await fetch(session, symbol, tf,
                                                  limit=limit, start_ms=start_ms)
                            summary["candles_upserted"] += await _upsert_candles(conn, candles)
                        except Exception as exc:
                            summary["errors"] += 1
                            logger.warning("candles %s %s %s fail-soft: %s",
                                           source, symbol, tf, exc)
                    if summary["budget_exhausted"]:
                        break
                if summary["budget_exhausted"]:
                    break

            # 2) funding-rate series
            if bool(cfg.get("funding_enabled", True)):
                for source in funding_sources:
                    fetch = FUNDING_FETCHERS[source]
                    for symbol in symbols:
                        if not await budget.take():
                            summary["budget_exhausted"] = True
                            break
                        try:
                            hw = await _series_high_water(conn, source, symbol,
                                                          "funding_rate")
                            start_ms = (int(hw.timestamp() * 1000) + 1
                                        if hw is not None else None)
                            points = await fetch(session, symbol,
                                                 limit=200, start_ms=start_ms)
                            summary["series_inserted"] += await _insert_series(conn, points)
                        except Exception as exc:
                            summary["errors"] += 1
                            logger.warning("funding %s %s fail-soft: %s",
                                           source, symbol, exc)
                    if summary["budget_exhausted"]:
                        break

            # 3) retention
            if bool(cfg.get("purge_enabled", True)):
                summary["purged"] = await _purge(conn, cfg)

    return summary


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, run a tick (unless killswitch/paused), sleep."""
    while True:
        cfg = await get_config() if get_config else dict(_DEFAULTS)
        if _KILLSWITCH.exists():
            logger.info("killswitch present — ingest tick skipped")
        elif _PAUSE_FLAG.exists():
            logger.info("module paused (%s) — ingest tick skipped", _PAUSE_FLAG)
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info(
                    "ingest tick: candles=%d series=%d purged=%d errors=%d%s",
                    s.get("candles_upserted", 0), s.get("series_inserted", 0),
                    s.get("purged", 0), s.get("errors", 0),
                    " (budget exhausted)" if s.get("budget_exhausted") else "")
            except Exception as exc:
                logger.error("ingest tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int((cfg or {}).get(
            "ingest_interval_seconds",
            _DEFAULTS["ingest_interval_seconds"])))
