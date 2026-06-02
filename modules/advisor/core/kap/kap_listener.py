"""
KAP Listener -- polls for new disclosures on a configurable interval.

ADVICE-ONLY. Stores raw disclosures; does not execute trades.

Data source priority
--------------------
1. PyKap (pykap v0.2.0, MIT, REAL -- verified on PyPI 2026-06-02):
   Uses BISTCompany.get_disclosures(type) per ticker in watchlist_bist.
   Also uses the byCriteria API via get_historical_disclosure_list() for
   date-ranged fetches. PyKap is import-guarded; absence falls through.

2. Direct KAP REST fallback (kap.org.tr/tr/api/):
   Uses the same public JSON endpoints PyKap wraps, called directly.
   Applied when PyKap is not installed or returns errors.
   Rate-limited: min inter-request gap + exponential backoff on errors.

IMPORTANT -- "all-disclosures" API
-----------------------------------
PyKap v0.2.0 does NOT expose a global "list all recent disclosures across all
companies" endpoint in its Python API.  Its BISTCompany class is per-ticker.
The listener therefore iterates over watchlist_bist tickers.
For broader coverage (all BIST disclosures regardless of watchlist), the
direct fallback path uses kap.org.tr/tr/api/disclosure/members/byCriteria
with an empty mkkMemberOidList, which returns market-wide disclosures.

Rate limiting (scraping path)
------------------------------
- Minimum 2 seconds between HTTP requests (KAP CDN is not hardened against
  reasonable polling, but abuse will result in 429 / IP ban).
- Exponential backoff: 5, 10, 20, 40 seconds on consecutive errors.
- Max 3 retries per request.
- User-Agent: "ClaudeDex-AdvisorResearch/1.0 (financial research, non-commercial)"
- On HTTP 429 or 503: back off for 5 minutes before retrying.

Legal note: see package __init__ for ToS discussion.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set

from modules.advisor.core.kap.kap_store import upsert_disclosure, upsert_company_profile

logger = logging.getLogger("advisor.kap.listener")

_USER_AGENT = "ClaudeDex-AdvisorResearch/1.0 (financial research, non-commercial)"
_KAP_BASE   = "https://www.kap.org.tr"

# Disclosure types to poll (subset -- most market-relevant)
_DISCLOSURE_TYPES_PYKAP = ("FAR", "KYUR", "SUR", "KDP", "DEG")


def _pykap_available() -> bool:
    """Return True if pykap is importable (import-guarded)."""
    try:
        import pykap  # noqa: F401
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# PyKap path
# ---------------------------------------------------------------------------

def _fetch_via_pykap_sync(ticker: str, since: datetime,
                           disclosure_types=_DISCLOSURE_TYPES_PYKAP) -> List[dict]:
    """
    Sync: fetch disclosures for a single ticker via PyKap.
    Returns list of normalised disclosure dicts.
    Runs in executor (blocking network I/O).
    """
    try:
        from pykap import BISTCompany
    except ImportError:
        return []

    results = []
    bare = ticker.upper().replace(".IS", "")
    try:
        comp = BISTCompany(bare)
    except Exception as exc:
        logger.debug("[kap.listener] BISTCompany(%r) init failed: %s", bare, exc)
        return []

    for dtype in disclosure_types:
        try:
            raw_list = comp.get_disclosures(dtype)
        except Exception as exc:
            logger.debug("[kap.listener] get_disclosures(%r, %r) failed: %s",
                         bare, dtype, exc)
            continue
        for item in (raw_list or []):
            normalised = _normalise_pykap_item(item, ticker=bare, dtype=dtype,
                                                comp_name=comp.name)
            if normalised and normalised["disclosed_at"] >= since:
                results.append(normalised)

    # Also try date-ranged historical for fresh disclosures
    try:
        hist = comp.get_historical_disclosure_list(
            fromdate=since.date(),
            todate=datetime.now(timezone.utc).date(),
            disclosure_type="FR",
        )
        for item in (hist or []):
            normalised = _normalise_pykap_item(item, ticker=bare, dtype="FR",
                                                comp_name=comp.name)
            if normalised:
                results.append(normalised)
    except Exception as exc:
        logger.debug("[kap.listener] get_historical_disclosure_list(%r) failed: %s",
                     bare, exc)

    return results


def _normalise_pykap_item(item: dict, ticker: str, dtype: str,
                            comp_name: str) -> Optional[dict]:
    """
    Normalise a raw PyKap disclosure dict into our storage schema.
    Returns None if essential fields are missing.
    """
    if not isinstance(item, dict):
        return None
    disc_id = str(item.get("disclosureIndex", ""))
    if not disc_id:
        return None

    publish_date_raw = item.get("publishDate", "")
    disclosed_at = _parse_kap_date(publish_date_raw)
    if disclosed_at is None:
        return None

    return {
        "disclosure_id":   disc_id,
        "ticker":          ticker,
        "tickers":         [ticker],
        "company_name":    item.get("stockCode") or comp_name or ticker,
        "subject":         item.get("title") or item.get("summary", "")[:256],
        "disclosure_type": dtype,
        "summary":         str(item.get("summary", "")),
        "full_text":       "",   # PyKap does not return full text in list calls
        "url":             f"{_KAP_BASE}/tr/Bildirim/{disc_id}",
        "disclosed_at":    disclosed_at,
        "source":          "pykap",
        "raw_payload":     item,
    }


# ---------------------------------------------------------------------------
# Direct KAP REST fallback path
# ---------------------------------------------------------------------------

def _fetch_via_kap_api_sync(since: datetime, member_oid_list: Optional[List[str]] = None,
                              rate_state: Optional[dict] = None) -> List[dict]:
    """
    Sync: fetch disclosures directly from kap.org.tr/tr/api/ REST.
    Polite rate limiting; exponential backoff on errors.
    member_oid_list=None or [] -> market-wide disclosures.
    Returns list of normalised disclosure dicts.
    Runs in executor.
    """
    try:
        import requests as req
    except ImportError:
        logger.warning("[kap.listener] 'requests' not installed; cannot use KAP scrape fallback.")
        return []

    if rate_state is None:
        rate_state = {}

    _rate_limit(rate_state)

    url = f"{_KAP_BASE}/tr/api/disclosure/members/byCriteria"
    payload = {
        "fromDate":             since.strftime("%Y-%m-%d"),
        "toDate":               datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "disclosureClass":      "ODA",
        "subjectList":          [],
        "mkkMemberOidList":     member_oid_list or [],
        "inactiveMkkMemberOidList": [],
        "bdkMemberOidList":     [],
        "fromSrc":              False,
        "disclosureIndexList":  [],
    }
    headers = {
        "User-Agent": _USER_AGENT,
        "Accept":     "application/json",
        "Content-Type": "application/json",
    }

    for attempt in range(3):
        try:
            resp = req.post(url, json=payload, headers=headers, timeout=30)
            if resp.status_code == 429 or resp.status_code == 503:
                backoff = 300  # 5 minute back-off on rate limit
                logger.warning(
                    "[kap.listener] KAP API returned %d -- backing off %ds",
                    resp.status_code, backoff
                )
                time.sleep(backoff)
                rate_state["backoff"] = min(rate_state.get("backoff", 5) * 2, 300)
                continue
            resp.raise_for_status()
            rate_state["backoff"] = 0
            rate_state["last_request_ts"] = time.monotonic()
            data = resp.json()
            break
        except Exception as exc:
            backoff = rate_state.get("backoff", 5) * (2 ** attempt)
            logger.warning(
                "[kap.listener] KAP REST attempt %d failed: %s. Backoff %ds",
                attempt + 1, exc, backoff
            )
            time.sleep(min(backoff, 60))
    else:
        return []

    results = []
    for item in (data if isinstance(data, list) else []):
        normalised = _normalise_kap_rest_item(item)
        if normalised and normalised["disclosed_at"] >= since:
            results.append(normalised)
    return results


def _normalise_kap_rest_item(item: dict) -> Optional[dict]:
    """Normalise a raw KAP byCriteria API item."""
    if not isinstance(item, dict):
        return None
    disc_basic = item.get("disclosureBasic") or item
    disc_id = str(disc_basic.get("disclosureIndex", ""))
    if not disc_id:
        return None

    disclosed_at = _parse_kap_date(disc_basic.get("publishDate", ""))
    if not disclosed_at:
        return None

    ticker = disc_basic.get("stockCode", "").upper() or None
    return {
        "disclosure_id":   disc_id,
        "ticker":          ticker,
        "tickers":         [ticker] if ticker else [],
        "company_name":    disc_basic.get("title", "")[:200] or disc_basic.get("stockCode", ""),
        "subject":         disc_basic.get("title", "")[:256],
        "disclosure_type": disc_basic.get("disclosureClass", ""),
        "summary":         disc_basic.get("summary", ""),
        "full_text":       "",
        "url":             f"{_KAP_BASE}/tr/Bildirim/{disc_id}",
        "disclosed_at":    disclosed_at,
        "source":          "scrape",
        "raw_payload":     disc_basic,
    }


# ---------------------------------------------------------------------------
# Rate limiting helpers
# ---------------------------------------------------------------------------

_MIN_REQUEST_GAP_S = 2.0   # minimum seconds between KAP requests

def _rate_limit(state: dict) -> None:
    """Block until at least _MIN_REQUEST_GAP_S has elapsed since last request."""
    last = state.get("last_request_ts", 0.0)
    elapsed = time.monotonic() - last
    if elapsed < _MIN_REQUEST_GAP_S:
        time.sleep(_MIN_REQUEST_GAP_S - elapsed)
    state["last_request_ts"] = time.monotonic()


# ---------------------------------------------------------------------------
# Date parsing
# ---------------------------------------------------------------------------

def _parse_kap_date(raw: str) -> Optional[datetime]:
    """
    Parse a KAP publishDate string to a tz-aware datetime (UTC).
    KAP returns dates in Istanbul time (UTC+3) without timezone annotation.
    Formats seen: '2026-01-15', '2026-01-15 17:50:00', '15.01.2026 17:50:00'.
    """
    if not raw:
        return None
    raw = str(raw).strip()
    tz_istanbul = timezone(timedelta(hours=3))
    fmts = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d.%m.%Y %H:%M:%S",
        "%d.%m.%Y",
    ]
    for fmt in fmts:
        try:
            naive = datetime.strptime(raw, fmt)
            return naive.replace(tzinfo=tz_istanbul).astimezone(timezone.utc)
        except ValueError:
            continue
    logger.debug("[kap.listener] Could not parse date %r", raw)
    return None


# ---------------------------------------------------------------------------
# KAP Listener class
# ---------------------------------------------------------------------------

class KapListener:
    """
    Async periodic listener for KAP disclosures.

    Usage (from AdvisorApplication or standalone):
        listener = KapListener(config=config, db_pool=pool)
        asyncio.create_task(listener.run())

    The listener will not run if advisor_kap_enabled != 'true'.
    Stops cleanly on asyncio.CancelledError.
    """

    def __init__(self, config: dict, db_pool=None):
        self.config = config
        self.db_pool = db_pool
        self._seen_ids: Set[str] = set()   # in-memory dedup (DB is authoritative)
        self._rate_state: dict = {}        # shared across REST calls
        self._running = False

    @property
    def enabled(self) -> bool:
        return str(self.config.get("advisor_kap_enabled", "false")).lower() == "true"

    @property
    def poll_interval(self) -> int:
        try:
            return int(self.config.get("advisor_kap_poll_interval_s", 60))
        except (ValueError, TypeError):
            return 60

    async def run(self) -> None:
        """
        Main loop. Polls until CancelledError.
        If advisor_kap_enabled=false, logs once and returns.
        """
        if not self.enabled:
            logger.info(
                "[kap.listener] advisor_kap_enabled=false -- listener not started. "
                "Set advisor_kap_enabled=true in advisor_config to enable."
            )
            return

        logger.info("[kap.listener] Starting KAP listener (poll_interval=%ds). "
                    "PyKap available: %s", self.poll_interval, _pykap_available())
        self._running = True
        try:
            while True:
                try:
                    await self._poll_cycle()
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    logger.warning("[kap.listener] Poll cycle error (will retry): %s", exc)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            logger.info("[kap.listener] Listener stopped (cancelled).")
        finally:
            self._running = False

    async def _poll_cycle(self) -> None:
        """One poll cycle: fetch new disclosures + persist."""
        since = datetime.now(timezone.utc) - timedelta(hours=2)
        tickers = self._get_watchlist_tickers()
        new_count = 0

        loop = asyncio.get_event_loop()

        if _pykap_available() and tickers:
            # PyKap path: per-ticker
            for ticker in tickers:
                try:
                    disclosures = await loop.run_in_executor(
                        None, _fetch_via_pykap_sync, ticker, since
                    )
                    for d in disclosures:
                        if d["disclosure_id"] not in self._seen_ids:
                            await self._persist(d)
                            self._seen_ids.add(d["disclosure_id"])
                            new_count += 1
                    # Polite inter-ticker gap
                    await asyncio.sleep(1.0)
                except Exception as exc:
                    logger.warning("[kap.listener] PyKap fetch for %r failed: %s",
                                   ticker, exc)
        else:
            # Direct REST fallback: market-wide
            try:
                disclosures = await loop.run_in_executor(
                    None, _fetch_via_kap_api_sync, since, None, self._rate_state
                )
                for d in disclosures:
                    if d["disclosure_id"] not in self._seen_ids:
                        await self._persist(d)
                        self._seen_ids.add(d["disclosure_id"])
                        new_count += 1
            except Exception as exc:
                logger.warning("[kap.listener] REST fallback poll failed: %s", exc)

        if new_count:
            logger.info("[kap.listener] Poll cycle complete: %d new disclosures stored.",
                        new_count)

        # Prune in-memory seen set (cap at 10k to avoid memory growth)
        if len(self._seen_ids) > 10_000:
            self._seen_ids = set(list(self._seen_ids)[-5_000:])

    async def _persist(self, d: dict) -> None:
        """Persist one disclosure dict to DB."""
        if self.db_pool is None:
            return
        try:
            row_id = await upsert_disclosure(self.db_pool, d)
            if row_id and d.get("ticker"):
                await upsert_company_profile(
                    self.db_pool,
                    ticker=d["ticker"],
                    company_name=d.get("company_name", d["ticker"]),
                )
        except Exception as exc:
            logger.warning("[kap.listener] _persist failed for %s: %s",
                           d.get("disclosure_id"), exc)

    def _get_watchlist_tickers(self) -> List[str]:
        """Parse watchlist_bist config into a list of bare tickers."""
        raw = self.config.get("watchlist_bist", "")
        if not raw:
            return []
        tickers = []
        for t in raw.split(","):
            t = t.strip().upper().replace(".IS", "")
            if t:
                tickers.append(t)
        return tickers
