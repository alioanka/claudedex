"""Read-only HTTP clients for the intent_solver shadow scaffold.

Free public APIs only, no keys, no signing, no on-chain calls. Every
network/parse error returns []/None and the caller's loop continues.
Sources: CoW orderbook (open auction orders + reference quotes),
UniswapX open Dutch orders, DexScreener USD pricing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("IntentSolverModule.Clients")

DEFAULT_COW_BASE_URL = "https://api.cow.fi"
DEFAULT_UNISWAPX_BASE_URL = "https://api.uniswap.org"
DEFAULT_DEXSCREENER_BASE_URL = "https://api.dexscreener.com"
COW_CHAINS = ("mainnet", "xdai", "arbitrum_one", "base")


class _JsonClient:
    """Rate-limited, fail-soft JSON-over-HTTP helper."""

    def __init__(self, base_url: str, *, timeout_s: float = 10.0,
                 max_requests_per_minute: int = 30):
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.max_requests_per_minute = max(1, int(max_requests_per_minute))
        self._request_times: List[float] = []
        self.last_error: Optional[str] = None

    async def _throttle(self) -> None:
        now = time.monotonic()
        self._request_times = [t for t in self._request_times if now - t < 60.0]
        if len(self._request_times) >= self.max_requests_per_minute:
            wait = 60.0 - (now - self._request_times[0]) + 0.05
            await asyncio.sleep(max(wait, 0.05))
        self._request_times.append(time.monotonic())

    async def request_json(self, method: str, path: str, *,
                           params: Optional[dict] = None,
                           json_body: Optional[dict] = None) -> Optional[Any]:
        try:
            import aiohttp  # lazy: module stays importable without aiohttp
        except ImportError:
            self.last_error = "aiohttp not installed"
            return None
        await self._throttle()
        url = f"{self.base_url}{path}"
        try:
            timeout = aiohttp.ClientTimeout(total=self.timeout_s)
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.request(method, url, params=params,
                                           json=json_body) as resp:
                    if resp.status != 200:
                        self.last_error = f"HTTP {resp.status} on {path}"
                        return None
                    payload = await resp.json(content_type=None)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning("%s %s failed: %s", method, path, self.last_error)
            return None
        self.last_error = None
        return payload


class CowClient(_JsonClient):
    """CoW Protocol orderbook: open orders (current auction) + reference quotes."""

    def __init__(self, base_url: str = DEFAULT_COW_BASE_URL, **kw):
        super().__init__(base_url, **kw)

    async def fetch_open_orders(self, chain: str, limit: int) -> List[dict]:
        """Raw orders from the current batch auction. [] on any failure."""
        if chain not in COW_CHAINS:
            self.last_error = f"unsupported cow chain {chain!r}"
            return []
        payload = await self.request_json("GET", f"/{chain}/api/v1/auction")
        orders = (payload or {}).get("orders")
        if not isinstance(orders, list):
            return []
        return orders[: max(1, int(limit))]

    async def fetch_quote(self, chain: str, *, kind: str, sell_token: str,
                          buy_token: str, amount: int,
                          from_address: str) -> Optional[int]:
        """Reference DEX-obtainable amount for one leg via POST /quote.

        sell kind -> returns obtainable buy amount (net of fee).
        buy kind  -> returns required sell amount INCLUDING feeAmount.
        None on any failure (caller skips the order).
        """
        body: Dict[str, Any] = {
            "sellToken": sell_token, "buyToken": buy_token,
            "from": from_address, "kind": kind,
            "priceQuality": "fast", "signingScheme": "eip712",
            "onchainOrder": False,
        }
        if kind == "sell":
            body["sellAmountBeforeFee"] = str(amount)
        elif kind == "buy":
            body["buyAmountAfterFee"] = str(amount)
        else:
            return None
        payload = await self.request_json("POST", f"/{chain}/api/v1/quote",
                                          json_body=body)
        quote = (payload or {}).get("quote")
        if not isinstance(quote, dict):
            return None
        try:
            if kind == "sell":
                return int(quote["buyAmount"])
            return int(quote["sellAmount"]) + int(quote.get("feeAmount", 0))
        except (KeyError, TypeError, ValueError):
            return None


class UniswapXClient(_JsonClient):
    """UniswapX public order API: open Dutch orders. [] on any failure."""

    def __init__(self, base_url: str = DEFAULT_UNISWAPX_BASE_URL, **kw):
        super().__init__(base_url, **kw)

    async def fetch_open_orders(self, chain_id: int, limit: int) -> List[dict]:
        payload = await self.request_json(
            "GET", "/v2/orders",
            params={"orderStatus": "open", "chainId": int(chain_id),
                    "limit": max(1, min(int(limit), 500))})
        orders = (payload or {}).get("orders")
        return orders if isinstance(orders, list) else []


class DexScreenerPriceClient(_JsonClient):
    """Best-effort USD token pricing (max-liquidity pair). Cached, fail-soft."""

    def __init__(self, base_url: str = DEFAULT_DEXSCREENER_BASE_URL,
                 *, cache_ttl_s: float = 120.0, **kw):
        super().__init__(base_url, **kw)
        self.cache_ttl_s = cache_ttl_s
        self._cache: Dict[str, tuple] = {}  # token -> (fetched_at, price|None)

    async def price_usd(self, token_address: str) -> Optional[float]:
        key = token_address.lower()
        cached = self._cache.get(key)
        if cached and (time.monotonic() - cached[0]) < self.cache_ttl_s:
            return cached[1]
        payload = await self.request_json("GET", f"/latest/dex/tokens/{key}")
        price: Optional[float] = None
        pairs = (payload or {}).get("pairs") or []
        best_liq = -1.0
        for p in pairs:
            try:
                liq = float((p.get("liquidity") or {}).get("usd") or 0.0)
                pu = float(p.get("priceUsd"))
            except (TypeError, ValueError):
                continue
            if pu > 0 and liq > best_liq:
                best_liq, price = liq, pu
        self._cache[key] = (time.monotonic(), price)
        if len(self._cache) > 256:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest, None)
        return price
