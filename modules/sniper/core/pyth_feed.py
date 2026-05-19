"""Pyth Hermes price-feed helper for SNIPER price resolution.

Wave-3 SNIPER enhancement. Provides:

    PythFeedClient.get_price(feed_id) -> float | None

Targets the free public Pyth Hermes endpoint
(https://hermes.pyth.network/v2/updates/price/latest?ids[]=<feed_id>).
Includes per-feed TTL cache and a process-wide min-interval throttle
so a flapping monitor loop cannot DoS the public endpoint.

Design choices:
- Singleton-per-process via module-level `pyth_client` so the cache /
  rate-limit state are shared by every caller (monitor loop, snipe
  evaluation, future SL/TP feed redundancy).
- Async; uses aiohttp because the rest of the sniper module already
  imports it. No new dependency.
- Fail-soft: any exception or non-200 returns None, never raises.
  The caller's chain falls through to Jupiter / Birdeye.
- TTL default 3s matches the sniper monitor tick (1s) with headroom
  for ~3 ticks worth of cache reuse. Pyth publish frequency for
  blue-chips is sub-second so 3s is well within freshness budget.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

from modules.sniper.core.pyth_feed_ids import PYTH_HERMES_BASE_URL

logger = logging.getLogger("SniperPythFeed")


class PythFeedClient:
    """Async client for Pyth Hermes latest-price queries.

    Not thread-safe — assumed to be driven from a single asyncio loop
    (the sniper engine loop). The per-feed cache uses (price, datetime).
    """

    # Default per-feed cache TTL. Pyth blue-chip publish frequency is
    # sub-second; we cache 3s to absorb the 1s monitor tick across
    # multiple positions without burning HTTP requests.
    DEFAULT_TTL = timedelta(seconds=3)

    # Process-wide minimum interval between any two HTTP requests to
    # Hermes. Free public endpoint; be polite. 100ms = 10 req/s
    # ceiling regardless of how many positions are monitored.
    MIN_REQUEST_INTERVAL = timedelta(milliseconds=100)

    # Per-request HTTP timeout. Pyth Hermes p95 is well under 500ms;
    # 2s is generous and prevents the caller from stalling the
    # monitor loop on a transient brown-out.
    REQUEST_TIMEOUT_SECONDS = 2

    def __init__(self, ttl: Optional[timedelta] = None):
        self._cache: dict = {}  # feed_id -> (price: float, ts: datetime)
        self._ttl = ttl or self.DEFAULT_TTL
        self._last_request_ts: datetime = datetime.min
        self._lock = asyncio.Lock()  # serializes the rate-limit gate
        # Cumulative counters surfaced to the engine for dashboard.
        self.hits = 0           # cache hit (fast path)
        self.fetches = 0        # successful HTTP fetch
        self.fetch_errors = 0   # HTTP / parse failure
        self.rate_limited = 0   # request gated by MIN_REQUEST_INTERVAL

    async def get_price(self, feed_id: Optional[str]) -> Optional[float]:
        """Return the current USD price for a Pyth feed-id, or None.

        Resolution order:
          1) cache hit within TTL → return cached value
          2) per-process rate-limit gate → sleep if needed
          3) HTTP GET /v2/updates/price/latest?ids[]=<feed_id>
          4) parse `price.price * 10**price.expo` → cache + return

        Any failure path returns None and increments fetch_errors so
        the dashboard can surface Pyth brown-outs. Caller falls through.
        """
        if not feed_id:
            return None

        now = datetime.now()
        cached = self._cache.get(feed_id)
        if cached:
            price, ts = cached
            if now - ts < self._ttl:
                self.hits += 1
                return price

        # Rate-limit gate. Acquire the lock so concurrent calls serialize
        # through the throttle (otherwise two parallel callers both pass
        # the timestamp check and burst).
        async with self._lock:
            now = datetime.now()
            delta = now - self._last_request_ts
            if delta < self.MIN_REQUEST_INTERVAL:
                sleep_s = (self.MIN_REQUEST_INTERVAL - delta).total_seconds()
                self.rate_limited += 1
                await asyncio.sleep(max(0.0, sleep_s))
            self._last_request_ts = datetime.now()

        url = f"{PYTH_HERMES_BASE_URL}/v2/updates/price/latest?ids[]={feed_id}"
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=self.REQUEST_TIMEOUT_SECONDS
                ) as response:
                    if response.status != 200:
                        self.fetch_errors += 1
                        logger.debug(
                            f"Pyth Hermes non-200 for {feed_id[:10]}: "
                            f"status={response.status}"
                        )
                        return None
                    data = await response.json()
                    price = self._parse_price(data)
                    if price is None or price <= 0:
                        self.fetch_errors += 1
                        return None
                    self._cache[feed_id] = (price, datetime.now())
                    self.fetches += 1
                    return price
        except Exception as e:
            self.fetch_errors += 1
            logger.debug(f"Pyth Hermes fetch error for {feed_id[:10]}: {e}")
            return None

    @staticmethod
    def _parse_price(data) -> Optional[float]:
        """Parse the Hermes v2 latest-price response shape into a float.

        Response shape (abbreviated):
          {
            "parsed": [
              {
                "id": "<feed_id_hex_no_0x>",
                "price": {"price": "12345678", "expo": -8, ... },
                ...
              }
            ]
          }

        Final USD = int(price) * 10 ** expo.

        Returns None if the response shape is unexpected or values are
        malformed. Never raises.
        """
        try:
            if not isinstance(data, dict):
                return None
            parsed = data.get('parsed') or []
            if not parsed:
                return None
            first = parsed[0] or {}
            price_obj = first.get('price') or {}
            raw = price_obj.get('price')
            expo = price_obj.get('expo')
            if raw is None or expo is None:
                return None
            return float(int(raw)) * (10 ** int(expo))
        except (ValueError, TypeError, KeyError):
            return None

    def stats(self) -> dict:
        """Snapshot of cumulative counters for dashboard surfacing."""
        return {
            'pyth_cache_hits': self.hits,
            'pyth_fetches': self.fetches,
            'pyth_fetch_errors': self.fetch_errors,
            'pyth_rate_limited': self.rate_limited,
            'pyth_cache_size': len(self._cache),
        }


# Module-level singleton. Engine code imports this directly so cache /
# rate-limit state are shared across the entire process.
pyth_client = PythFeedClient()
