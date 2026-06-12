"""CATALYST_CALENDAR engine — fetch -> normalize -> upsert -> purge.

One tick:
  1. For every ENABLED source (DB config), fetch the raw payload (fail-soft)
     and normalize it via the pure functions in normalizer.py.
  2. Upsert each event into `catalysts` keyed on
     (source, symbol, event_type, event_time) — re-seen events only bump
     last_seen_at, so consumers can judge feed freshness per row.
  3. Purge events older than purge_after_days (a stale calendar is worse than
     none; consumers must also check last_seen_at before trusting a row).

PURE ADVISORY: this engine never trades, never writes pause/killswitch flags,
and holds no keys (FMP key is optional and only widens macro coverage).
Everything is fail-soft: a dead source logs a warning and contributes zero
events; the tick continues.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.catalyst_calendar.core import normalizer
from modules.catalyst_calendar.core.sources import (
    HttpFetcher,
    fetch_binance_announcements,
    fetch_defillama_emissions,
    fetch_fmp_macro,
    resolve_fmp_api_key,
)

logger = logging.getLogger("catalyst_calendar")

_PAUSE_FLAG = Path("logs/.pause_catalyst_calendar")
_KILLSWITCH = Path("logs/.killswitch")

_UPSERT_SQL = (
    "INSERT INTO catalysts "
    "(source, symbol, event_type, event_time, title, magnitude, confidence, "
    " url, raw, first_seen_at, last_seen_at) "
    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, NOW(), NOW()) "
    "ON CONFLICT (source, symbol, event_type, event_time) DO UPDATE SET "
    "title=EXCLUDED.title, magnitude=EXCLUDED.magnitude, "
    "confidence=EXCLUDED.confidence, url=EXCLUDED.url, raw=EXCLUDED.raw, "
    "last_seen_at=NOW()"
)


async def load_calendar_config(pool) -> dict:
    """config_settings rows with config_type='catalyst_calendar' -> typed dict.
    Fail-soft to {} (code defaults apply)."""
    out: dict = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings "
                "WHERE config_type='catalyst_calendar'"
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
        logger.error("load_calendar_config fail-soft: %s", exc)
    return out


def _macro_override_from_config(cfg: dict) -> Optional[List[Dict[str, str]]]:
    """Optional operator override of the built-in macro schedule (JSON list)."""
    raw = cfg.get("macro_static_override")
    if not raw or not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
    except (ValueError, TypeError):
        logger.warning("macro_static_override is not valid JSON — ignored")
        return None
    return parsed if isinstance(parsed, list) and parsed else None


async def _collect_events(cfg: dict, fetcher: HttpFetcher,
                          summary: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fetch + normalize every enabled source. Per-source fail-soft."""
    now = datetime.now(timezone.utc)
    lookahead = int(cfg.get("lookahead_days", 30))
    events: List[Dict[str, Any]] = []
    src = summary["sources"]

    if bool(cfg.get("source_defillama_unlocks_enabled", True)):
        payload = await fetch_defillama_emissions(fetcher)
        parsed = normalizer.parse_defillama_emissions(
            payload, now=now, lookahead_days=lookahead)
        src["defillama_unlocks"] = {
            "fetched": payload is not None, "events": len(parsed),
            "error": fetcher.last_error if payload is None else None,
        }
        events.extend(parsed)

    if bool(cfg.get("source_binance_listings_enabled", True)):
        payload = await fetch_binance_announcements(fetcher)
        parsed = normalizer.parse_binance_announcements(
            payload, now=now,
            recent_window_hours=int(cfg.get("listing_recent_window_hours", 72)))
        src["binance_announcements"] = {
            "fetched": payload is not None, "events": len(parsed),
            "error": fetcher.last_error if payload is None else None,
        }
        events.extend(parsed)

    if bool(cfg.get("source_macro_static_enabled", True)):
        override = _macro_override_from_config(cfg)
        parsed = normalizer.parse_static_macro(
            override, now=now, lookahead_days=lookahead)
        stale = override is None and normalizer.static_macro_is_stale(now)
        if stale:
            logger.warning(
                "static macro schedule EXPIRED (%s) — emitting nothing; "
                "refresh STATIC_MACRO_2026 or set macro_static_override",
                normalizer.STATIC_MACRO_EXPIRES)
        src["static_macro"] = {"fetched": True, "events": len(parsed),
                               "stale": stale, "error": None}
        events.extend(parsed)

    if bool(cfg.get("source_fmp_macro_enabled", False)):
        api_key = resolve_fmp_api_key()
        payload = await fetch_fmp_macro(fetcher, api_key=api_key,
                                        lookahead_days=lookahead)
        parsed = normalizer.parse_fmp_macro(payload, now=now,
                                            lookahead_days=lookahead)
        src["fmp_macro"] = {
            "fetched": payload is not None, "events": len(parsed),
            "error": ("no FMP_API_KEY configured" if not api_key
                      else (fetcher.last_error if payload is None else None)),
        }
        events.extend(parsed)

    return events


async def _persist_events(pool, events: List[Dict[str, Any]]) -> int:
    """Upsert normalized events. Returns rows written (insert or refresh)."""
    written = 0
    async with pool.acquire() as conn:
        for ev in events:
            try:
                await conn.execute(
                    _UPSERT_SQL,
                    ev["source"], ev["symbol"], ev["event_type"],
                    ev["event_time"], ev["title"],
                    (float(ev["magnitude"]) if ev["magnitude"] is not None else None),
                    ev["confidence"], ev["url"],
                    json.dumps(ev.get("raw") or {}, default=str),
                )
                written += 1
            except Exception as exc:
                logger.warning("catalysts upsert fail-soft (%s/%s): %s",
                               ev.get("source"), ev.get("symbol"), exc)
    return written


async def _purge_old(pool, purge_after_days: int) -> int:
    """Drop long-past events so the table stays a forward calendar."""
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM catalysts WHERE event_time < NOW() - "
                "make_interval(days => $1)", max(1, int(purge_after_days)))
        return int(result.split()[-1]) if result else 0
    except Exception as exc:
        logger.warning("purge fail-soft: %s", exc)
        return 0


async def run_tick(pool, cfg: dict, fetcher: HttpFetcher) -> dict:
    """One full refresh cycle. Returns a summary dict (for /status)."""
    summary: Dict[str, Any] = {
        "tick_at": datetime.now(timezone.utc).isoformat(),
        "sources": {}, "events_collected": 0, "rows_upserted": 0, "purged": 0,
    }
    events = await _collect_events(cfg, fetcher, summary)

    # In-memory dedup across sources sharing an identity (defensive).
    unique: Dict[tuple, Dict[str, Any]] = {}
    for ev in events:
        unique[normalizer.dedup_key(ev)] = ev
    summary["events_collected"] = len(unique)

    summary["rows_upserted"] = await _persist_events(pool, list(unique.values()))
    summary["purged"] = await _purge_old(pool, int(cfg.get("purge_after_days", 90)))
    return summary


async def run_loop(pool, *, get_config, fetcher: Optional[HttpFetcher] = None,
                   on_summary=None) -> None:
    """Forever: load config, run a tick (unless killswitch/paused), sleep."""
    while True:
        cfg = await get_config() if get_config else {}
        if fetcher is None or (fetcher.max_requests_per_minute
                               != int(cfg.get("max_requests_per_minute", 10))):
            fetcher = HttpFetcher(
                max_requests_per_minute=int(cfg.get("max_requests_per_minute", 10)))
        if _KILLSWITCH.exists():
            logger.info("killswitch present — calendar tick skipped")
        elif _PAUSE_FLAG.exists():
            logger.info("catalyst_calendar paused — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg, fetcher)
                if on_summary:
                    try:
                        on_summary(s)
                    except Exception:
                        pass
                logger.info("calendar tick: collected=%d upserted=%d purged=%d "
                            "sources=%s", s["events_collected"],
                            s["rows_upserted"], s["purged"],
                            {k: v.get("events") for k, v in s["sources"].items()})
            except Exception as exc:
                logger.error("calendar tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int((cfg or {}).get("refresh_interval_seconds", 3600)))
