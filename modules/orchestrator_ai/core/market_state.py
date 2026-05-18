"""Market-state collector for the orchestrator scorer.

Provides BTC + ETH 24h percentage changes from CoinGecko's free
endpoint with a 5-minute in-process cache (matching the orchestrator's
default tick interval, so we make at most one external call per tick).

If CoinGecko is unreachable, returns None — the scorer falls back to
a neutral regime signal (0.5).

CoinGecko's /simple/price endpoint with include_24hr_change=true is
the same one used elsewhere in monitoring/enhanced_dashboard.py, just
parameterized for two ids and the 24h delta field.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

logger = logging.getLogger("orchestrator_ai")

_CACHE_TTL_SECONDS = 5 * 60
_REQUEST_TIMEOUT_SECONDS = 6.0

_CG_URL = (
    "https://api.coingecko.com/api/v3/simple/price"
    "?ids=bitcoin,ethereum&vs_currencies=usd&include_24hr_change=true"
)


@dataclass
class MarketState:
    btc_24h_change_pct: Optional[float]
    eth_24h_change_pct: Optional[float]
    fetched_at: float          # unix time when this snapshot was taken
    source: str                # 'coingecko' / 'cache' / 'error'


class MarketStateCache:
    """Tiny in-process cache. One getter coroutine; CoinGecko hits at
    most once per TTL window even if multiple ticks coincide.

    Not thread-safe across event loops. Each subprocess gets its own
    cache instance — which is what we want.
    """

    def __init__(self, ttl_seconds: int = _CACHE_TTL_SECONDS):
        self._ttl = ttl_seconds
        self._state: Optional[MarketState] = None
        self._lock = asyncio.Lock()

    def _is_fresh(self) -> bool:
        return self._state is not None and (
            time.time() - self._state.fetched_at < self._ttl
        )

    async def get(self) -> MarketState:
        """Returns cached state if fresh, otherwise hits CoinGecko.
        On network failure, returns the LAST cached state if any
        (even if stale) — drift beats nothing. If we've never had a
        success, returns a MarketState with None values + source='error'."""
        if self._is_fresh():
            assert self._state is not None
            return MarketState(
                btc_24h_change_pct=self._state.btc_24h_change_pct,
                eth_24h_change_pct=self._state.eth_24h_change_pct,
                fetched_at=self._state.fetched_at,
                source='cache',
            )
        async with self._lock:
            # Re-check under lock (concurrent ticks could both see stale).
            if self._is_fresh():
                assert self._state is not None
                return MarketState(
                    btc_24h_change_pct=self._state.btc_24h_change_pct,
                    eth_24h_change_pct=self._state.eth_24h_change_pct,
                    fetched_at=self._state.fetched_at,
                    source='cache',
                )
            fetched = await _fetch_coingecko()
            if fetched is not None:
                self._state = fetched
                return fetched
            # Network failed. Return stale-but-existing state if available.
            if self._state is not None:
                logger.warning(
                    "CoinGecko fetch failed; returning stale snapshot "
                    "(age=%.0fs)", time.time() - self._state.fetched_at,
                )
                return MarketState(
                    btc_24h_change_pct=self._state.btc_24h_change_pct,
                    eth_24h_change_pct=self._state.eth_24h_change_pct,
                    fetched_at=self._state.fetched_at,
                    source='cache',
                )
            return MarketState(
                btc_24h_change_pct=None,
                eth_24h_change_pct=None,
                fetched_at=time.time(),
                source='error',
            )


async def _fetch_coingecko() -> Optional[MarketState]:
    timeout = aiohttp.ClientTimeout(total=_REQUEST_TIMEOUT_SECONDS)
    try:
        async with aiohttp.ClientSession(timeout=timeout) as sess:
            async with sess.get(_CG_URL) as resp:
                if resp.status != 200:
                    logger.debug(
                        "CoinGecko returned HTTP %d", resp.status
                    )
                    return None
                payload = await resp.json()
        btc = payload.get('bitcoin', {})
        eth = payload.get('ethereum', {})
        return MarketState(
            btc_24h_change_pct=btc.get('usd_24h_change'),
            eth_24h_change_pct=eth.get('usd_24h_change'),
            fetched_at=time.time(),
            source='coingecko',
        )
    except (aiohttp.ClientError, asyncio.TimeoutError) as e:
        logger.debug("CoinGecko fetch error: %s", e)
        return None
    except Exception as e:
        logger.warning("CoinGecko unexpected error: %s", e)
        return None
