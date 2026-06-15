"""Free-source HTTP clients for catalyst_calendar. Rate-limited, fail-soft.

Every fetch returns the RAW payload (parsing lives in normalizer.py) or None on
any failure — network errors, non-200, schema surprises upstream — and never
raises. aiohttp is imported lazily so the module stays importable without it.

Cost discipline: all sources here are FREE. The only key-gated source (FMP)
returns None immediately when no key is configured — it degrades to empty, it
never blocks the loop and never spends anything.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("catalyst_calendar")

DEFAULT_DEFILLAMA_EMISSIONS_URL = "https://api.llama.fi/emissions"
# Binance CMS announcement endpoint (scrape-class; catalogId 48 = new listings).
DEFAULT_BINANCE_CMS_URL = (
    "https://www.binance.com/bapi/apex/v1/public/apex/cms/article/list/query"
)
DEFAULT_FMP_CALENDAR_URL = "https://financialmodelingprep.com/api/v3/economic_calendar"


class HttpFetcher:
    """Minimal rate-limited GET-JSON wrapper. Shared across all sources so the
    module's TOTAL outbound request rate is capped, not per-source."""

    def __init__(self, *, timeout_s: float = 15.0,
                 max_requests_per_minute: int = 10):
        self.timeout_s = timeout_s
        self.max_requests_per_minute = max(1, int(max_requests_per_minute))
        self._request_times: List[float] = []
        self.last_error: Optional[str] = None
        self.last_fetch_at: Optional[float] = None
        # URLs that returned a HARD-unavailable status (payment/forbidden/gone):
        # warn ONCE, then skip + log at DEBUG so a dead free source (e.g.
        # DefiLlama emissions now 402) can't spam the log every cycle.
        self._dead_urls: set = set()

    async def _throttle(self) -> None:
        now = time.monotonic()
        self._request_times = [t for t in self._request_times if now - t < 60.0]
        if len(self._request_times) >= self.max_requests_per_minute:
            wait = 60.0 - (now - self._request_times[0]) + 0.05
            await asyncio.sleep(max(wait, 0.05))
        self._request_times.append(time.monotonic())

    async def get_json(self, url: str,
                       params: Optional[Dict[str, Any]] = None) -> Optional[Any]:
        """One GET. Parsed JSON or None. Never raises (except CancelledError)."""
        try:
            import aiohttp  # lazy: keep the module importable without aiohttp
        except ImportError:
            self.last_error = "aiohttp not installed"
            return None
        # Skip URLs already known to be hard-unavailable (don't spend the request
        # or re-warn every cycle). Operator can re-enable by restarting once the
        # source is back, or disable the source via its config flag.
        if url in self._dead_urls:
            self.last_error = f"skipped (previously hard-unavailable): {url}"
            logger.debug("skip dead source %s", url)
            return None
        await self._throttle()
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            headers = {"User-Agent": "Mozilla/5.0 (compatible; calendar-bot)"}
            async with aiohttp.ClientSession(timeout=timeout,
                                             headers=headers) as session:
                async with session.get(url, params=params or {}) as resp:
                    if resp.status != 200:
                        self.last_error = f"HTTP {resp.status} on {url}"
                        # 402/403/404/410 = the source won't recover on retry
                        # (payment/forbidden/gone) — warn ONCE then go quiet.
                        if resp.status in (401, 402, 403, 404, 410):
                            if url not in self._dead_urls:
                                self._dead_urls.add(url)
                                logger.warning(
                                    "fetch %s -> HTTP %s (hard-unavailable; "
                                    "suppressing further attempts this run)",
                                    url, resp.status)
                            else:
                                logger.debug("fetch %s -> HTTP %s", url, resp.status)
                        else:
                            logger.warning("fetch %s -> HTTP %s", url, resp.status)
                        return None
                    payload = await resp.json(content_type=None)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # fail-soft: any network/parse error -> None
            self.last_error = f"{type(exc).__name__}: {exc}"
            logger.warning("fetch failed (%s): %s", url, self.last_error)
            return None
        self.last_error = None
        self.last_fetch_at = time.time()
        return payload


async def fetch_defillama_emissions(fetcher: HttpFetcher,
                                    url: str = DEFAULT_DEFILLAMA_EMISSIONS_URL
                                    ) -> Optional[Any]:
    """Unofficial DefiLlama emissions/unlocks aggregate. Free, no key.
    Schema is undocumented — the parser is defensive; None/[] on any change."""
    return await fetcher.get_json(url)


async def fetch_binance_announcements(fetcher: HttpFetcher,
                                      url: str = DEFAULT_BINANCE_CMS_URL,
                                      page_size: int = 20) -> Optional[Any]:
    """Binance CMS announcements (catalogId 48 = new crypto listings).
    Scrape-class and FRAGILE: Binance can change or geo-block this endpoint at
    any time; failures degrade to None and the calendar simply lacks listings."""
    params = {
        "type": 1,
        "catalogId": 48,
        "pageNo": 1,
        "pageSize": max(1, min(int(page_size), 50)),
    }
    return await fetcher.get_json(url, params)


def resolve_fmp_api_key() -> Optional[str]:
    """FMP key: secrets_manager first, env fallback. None = source disabled."""
    key = None
    try:
        from security.secrets_manager import secrets
        key = secrets.get("FMP_API_KEY")
    except Exception:
        key = None
    if not key:
        key = os.getenv("FMP_API_KEY")
    return key or None


async def fetch_fmp_macro(fetcher: HttpFetcher, *,
                          api_key: Optional[str],
                          lookahead_days: int,
                          url: str = DEFAULT_FMP_CALENDAR_URL) -> Optional[Any]:
    """FMP economic calendar — KEY-GATED. No key -> None immediately (the
    source degrades to empty; the built-in static macro schedule still runs)."""
    if not api_key:
        return None
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    params = {
        "from": now.strftime("%Y-%m-%d"),
        "to": (now + timedelta(days=max(1, int(lookahead_days)))).strftime("%Y-%m-%d"),
        "apikey": api_key,
    }
    return await fetcher.get_json(url, params)
