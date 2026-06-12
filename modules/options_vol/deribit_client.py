"""Read-only Deribit PUBLIC API client (free, no key) for the options_vol module.

Plain aiohttp against https://www.deribit.com/api/v2/public/* with a
client-side rate limiter. Every method is fail-soft: errors are logged,
last_error is set, and None/[] is returned — a cycle never dies on a feed
hiccup. LIVE order placement does NOT live here (see executor.py, which uses
ccxt's deribit private API and is gated off by default).
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("OptionsVolModule.Deribit")

DEFAULT_BASE_URL = "https://www.deribit.com"


class DeribitPublicClient:
    def __init__(self, base_url: str = DEFAULT_BASE_URL,
                 max_requests_per_minute: int = 20,
                 timeout_s: float = 15.0):
        self.base_url = base_url.rstrip("/")
        self._min_interval = 60.0 / max(1, int(max_requests_per_minute))
        self._last_request_ts = 0.0
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self.last_error: Optional[str] = None

    async def _get(self, path: str, params: Dict[str, Any]) -> Optional[Any]:
        """Rate-limited GET; returns parsed 'result' or None (never raises)."""
        wait = self._min_interval - (time.monotonic() - self._last_request_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        self._last_request_ts = time.monotonic()
        url = f"{self.base_url}/api/v2/public/{path}"
        try:
            async with aiohttp.ClientSession(timeout=self._timeout) as session:
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        self.last_error = f"{path}: HTTP {resp.status}"
                        logger.warning("Deribit %s", self.last_error)
                        return None
                    body = await resp.json(content_type=None)
        except Exception as e:
            self.last_error = f"{path}: {type(e).__name__}: {e}"
            logger.warning("Deribit %s", self.last_error)
            return None
        if not isinstance(body, dict) or "result" not in body:
            self.last_error = f"{path}: malformed response"
            logger.warning("Deribit %s", self.last_error)
            return None
        self.last_error = None
        return body["result"]

    async def get_index_price(self, currency: str) -> Optional[float]:
        """USD index, e.g. currency='BTC' -> index_name='btc_usd'."""
        res = await self._get("get_index_price",
                              {"index_name": f"{currency.lower()}_usd"})
        try:
            return float(res["index_price"]) if res else None
        except (KeyError, TypeError, ValueError):
            return None

    async def get_option_book_summaries(self, currency: str) -> List[Dict[str, Any]]:
        """All live option book summaries for a currency.

        Each row carries instrument_name, mark_price (in COIN — inverse
        quoting; USD = mark_price * underlying_price), underlying_price,
        bid_price/ask_price (coin), open_interest, volume.
        """
        res = await self._get("get_book_summary_by_currency",
                              {"currency": currency.upper(), "kind": "option"})
        return res if isinstance(res, list) else []

    async def get_hourly_closes(self, currency: str, hours: int) -> List[float]:
        """Hourly perpetual closes for realized vol (index proxy)."""
        now_ms = int(time.time() * 1000)
        res = await self._get("get_tradingview_chart_data", {
            "instrument_name": f"{currency.upper()}-PERPETUAL",
            "resolution": "60",
            "start_timestamp": now_ms - hours * 3600 * 1000,
            "end_timestamp": now_ms,
        })
        if not isinstance(res, dict) or res.get("status") != "ok":
            return []
        closes = res.get("close") or []
        out: List[float] = []
        for c in closes:
            try:
                out.append(float(c))
            except (TypeError, ValueError):
                continue
        return out
