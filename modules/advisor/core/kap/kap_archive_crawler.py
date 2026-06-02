"""
KAP Archive Crawler -- historical disclosure backfill helper.

This module provides the crawl_ticker() coroutine used by the standalone
operator script scripts/crawl_kap_history.py.

Design
------
- Per-ticker crawl using PyKap BISTCompany.get_historical_disclosure_list()
  (date-ranged, disclosure_type='FR' and 'ODA').
- Fallback: direct kap.org.tr/tr/api/disclosure/members/byCriteria POST.
- Resumable: tracks last-crawled date per ticker in kap_company_profiles.extra
  (key: 'last_crawl_date'). Re-run skips already-covered windows.
- Rate-limited: configurable delay between tickers (default 3s) and between
  pages (default 2s). MUCH slower than the real-time listener -- this is by
  design for polite historical crawling.
- Backfill only: this module NEVER runs in the advice loop. It is called
  exclusively by scripts/crawl_kap_history.py.

IMPORTANT: aggressive crawling WILL get your IP rate-limited by KAP's CDN.
Default pace (3s/ticker, 2s/page) is conservative. For large backfills
(e.g. 200 tickers, 5 years), run overnight or across multiple days using
the --since flag to limit the window.

Disclosure types fetched:
  'FR'  -- Financial Reports (finansal rapor)
  'ODA' -- Material Disclosures (ozel durum aciklamasi) -- most price-relevant

Legal note: see package __init__ for ToS discussion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import date, datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

from modules.advisor.core.kap.kap_store import upsert_disclosure, upsert_company_profile
from modules.advisor.core.kap.kap_listener import (
    _pykap_available,
    _normalise_pykap_item,
    _normalise_kap_rest_item,
    _parse_kap_date,
    _rate_limit,
    _USER_AGENT,
    _KAP_BASE,
    _MIN_REQUEST_GAP_S,
)

logger = logging.getLogger("advisor.kap.crawler")

# Disclosure classes to backfill (most price-relevant + financial reports)
_CRAWL_TYPES_PYKAP = ("FAR",)      # pykap get_disclosures() types
_CRAWL_CLASSES_REST = ("FR", "ODA") # direct REST byCriteria classes

# Default inter-ticker pause (polite pace for backfill -- much slower than listener)
DEFAULT_TICKER_PAUSE_S = 3.0
DEFAULT_PAGE_PAUSE_S   = 2.0


# ---------------------------------------------------------------------------
# PyKap crawl path
# ---------------------------------------------------------------------------

def _crawl_ticker_pykap_sync(ticker: str, since: date, until: date,
                               rate_state: dict) -> List[dict]:
    """
    Sync: crawl all disclosures for ticker between since..until via PyKap.
    Runs in executor.
    """
    try:
        from pykap import BISTCompany
    except ImportError:
        return []

    bare = ticker.upper().replace(".IS", "")
    results = []

    try:
        comp = BISTCompany(bare)
    except Exception as exc:
        logger.debug("[kap.crawler] BISTCompany(%r) init failed: %s", bare, exc)
        return []

    comp_name = comp.name

    # 1. Date-ranged historical list (FR class -- financial reports)
    for disc_class in ("FR", "ODA"):
        _rate_limit(rate_state)
        try:
            hist = comp.get_historical_disclosure_list(
                fromdate=since,
                todate=until,
                disclosure_type=disc_class,
                subject=("4028328c594bfdca01594c0af9aa0057"
                         if disc_class == "FR"
                         else "4028328d594c04f201594c5155dd0076"),
            )
            for item in (hist or []):
                normalised = _normalise_pykap_item(
                    item, ticker=bare, dtype=disc_class, comp_name=comp_name
                )
                if normalised:
                    results.append(normalised)
        except Exception as exc:
            logger.debug("[kap.crawler] pykap historical %r %r failed: %s",
                         bare, disc_class, exc)
        time.sleep(DEFAULT_PAGE_PAUSE_S)

    # 2. Type-based disclosures (FAR, KYUR, etc.) -- no date filter available
    for dtype in _CRAWL_TYPES_PYKAP:
        _rate_limit(rate_state)
        try:
            raw_list = comp.get_disclosures(dtype)
            for item in (raw_list or []):
                normalised = _normalise_pykap_item(
                    item, ticker=bare, dtype=dtype, comp_name=comp_name
                )
                if normalised:
                    disc_date = normalised["disclosed_at"].date()
                    if since <= disc_date <= until:
                        results.append(normalised)
        except Exception as exc:
            logger.debug("[kap.crawler] pykap get_disclosures(%r, %r) failed: %s",
                         bare, dtype, exc)
        time.sleep(DEFAULT_PAGE_PAUSE_S)

    return results


# ---------------------------------------------------------------------------
# Direct REST crawl path
# ---------------------------------------------------------------------------

def _crawl_ticker_rest_sync(company_id: Optional[str], since: date, until: date,
                              rate_state: dict) -> List[dict]:
    """
    Sync: crawl disclosures via direct KAP REST API for a given company_id.
    If company_id is None, fetches market-wide (empty mkkMemberOidList).
    Runs in executor.
    """
    try:
        import requests as req
    except ImportError:
        logger.warning("[kap.crawler] 'requests' not installed.")
        return []

    url = f"{_KAP_BASE}/tr/api/disclosure/members/byCriteria"
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    results = []

    for disc_class in _CRAWL_CLASSES_REST:
        _rate_limit(rate_state)
        payload = {
            "fromDate":                 since.strftime("%Y-%m-%d"),
            "toDate":                   until.strftime("%Y-%m-%d"),
            "disclosureClass":          disc_class,
            "subjectList":              [],
            "mkkMemberOidList":         [company_id] if company_id else [],
            "inactiveMkkMemberOidList": [],
            "bdkMemberOidList":         [],
            "fromSrc":                  False,
            "disclosureIndexList":      [],
        }

        for attempt in range(3):
            try:
                resp = req.post(url, json=payload, headers=headers, timeout=30)
                if resp.status_code in (429, 503):
                    backoff = 300
                    logger.warning("[kap.crawler] KAP returned %d -- backing off %ds",
                                   resp.status_code, backoff)
                    time.sleep(backoff)
                    continue
                resp.raise_for_status()
                data = resp.json()
                break
            except Exception as exc:
                backoff = 5 * (2 ** attempt)
                logger.debug("[kap.crawler] REST attempt %d failed: %s. Backoff %ds",
                             attempt + 1, exc, backoff)
                time.sleep(min(backoff, 60))
        else:
            continue

        for item in (data if isinstance(data, list) else []):
            normalised = _normalise_kap_rest_item(item)
            if normalised:
                results.append(normalised)
        time.sleep(DEFAULT_PAGE_PAUSE_S)

    return results


# ---------------------------------------------------------------------------
# Main crawl_ticker coroutine (used by scripts/crawl_kap_history.py)
# ---------------------------------------------------------------------------

async def crawl_ticker(ticker: str, since: date, until: date,
                        db_pool, rate_state: dict,
                        ticker_pause_s: float = DEFAULT_TICKER_PAUSE_S,
                        dry_run: bool = False) -> int:
    """
    Crawl and persist all disclosures for ticker in [since, until].

    Parameters
    ----------
    ticker       : BIST ticker (bare, e.g. 'THYAO'). .IS suffix stripped.
    since        : Start date (inclusive).
    until        : End date (inclusive; default today).
    db_pool      : asyncpg pool.
    rate_state   : Shared rate-limiting state dict (mutated in place).
    ticker_pause_s: Seconds to sleep after this ticker is done.
    dry_run      : If True, fetch but do not persist to DB.

    Returns the number of NEW disclosures stored (0 on any error).
    """
    loop = asyncio.get_event_loop()
    bare = ticker.upper().replace(".IS", "")
    stored = 0

    # Check resumability: skip if last_crawl_date >= until
    if db_pool and not dry_run:
        try:
            row = await db_pool.fetchrow(
                "SELECT extra FROM kap_company_profiles WHERE ticker=$1", bare
            )
            if row:
                extra = row["extra"] or {}
                last_crawl = extra.get("last_crawl_date")
                if last_crawl:
                    last_dt = date.fromisoformat(str(last_crawl))
                    if last_dt >= until:
                        logger.info("[kap.crawler] %s already crawled up to %s, skipping.",
                                    bare, last_crawl)
                        return 0
        except Exception:
            pass  # DB not available or table missing -- proceed

    # Fetch
    if _pykap_available():
        logger.info("[kap.crawler] Crawling %s via PyKap (%s..%s)", bare, since, until)
        disclosures = await loop.run_in_executor(
            None, _crawl_ticker_pykap_sync, bare, since, until, rate_state
        )
    else:
        logger.info("[kap.crawler] Crawling %s via REST fallback (%s..%s)", bare, since, until)
        # Without company_id, we get market-wide results -- filter by ticker
        disclosures = await loop.run_in_executor(
            None, _crawl_ticker_rest_sync, None, since, until, rate_state
        )
        disclosures = [d for d in disclosures
                       if d.get("ticker", "").upper() == bare]

    # Deduplicate
    seen: set = set()
    unique = []
    for d in disclosures:
        if d["disclosure_id"] not in seen:
            seen.add(d["disclosure_id"])
            unique.append(d)

    if dry_run:
        logger.info("[kap.crawler] DRY RUN: %s -> %d disclosures (not persisted).",
                    bare, len(unique))
        return len(unique)

    # Persist
    for d in unique:
        if db_pool:
            row_id = await upsert_disclosure(db_pool, d)
            if row_id:
                stored += 1

    # Update company profile + last_crawl_date
    if db_pool and unique:
        first = unique[0]
        await upsert_company_profile(
            db_pool, ticker=bare,
            company_name=first.get("company_name", bare),
        )
    if db_pool:
        try:
            await db_pool.execute("""
                INSERT INTO kap_company_profiles (ticker, company_name, extra)
                VALUES ($1, $2, jsonb_build_object('last_crawl_date', $3::text))
                ON CONFLICT (ticker) DO UPDATE SET
                    extra = kap_company_profiles.extra ||
                            jsonb_build_object('last_crawl_date', $3::text),
                    updated_at = NOW()
            """, bare, bare, until.isoformat())
        except Exception as exc:
            logger.debug("[kap.crawler] last_crawl_date update failed: %s", exc)

    logger.info("[kap.crawler] %s: %d/%d new disclosures stored.",
                bare, stored, len(unique))
    await asyncio.sleep(ticker_pause_s)
    return stored
