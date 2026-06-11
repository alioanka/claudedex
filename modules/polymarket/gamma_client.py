"""Read-only Polymarket Gamma API client. No key, no on-chain risk.

Fetches active markets + YES/NO prices + volume/liquidity from
https://gamma-api.polymarket.com. Cached, rate-limited, fail-soft:
every network/parse error returns [] / None and the caller's loop continues.

Self-test (offline, uses the static fixture):
    python -m modules.polymarket.gamma_client
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger("PolymarketModule.Gamma")

DEFAULT_GAMMA_BASE_URL = "https://gamma-api.polymarket.com"


def _as_float(value: Any) -> Optional[float]:
    """Defensive numeric coercion: Gamma returns numbers, numeric strings, or null."""
    if value is None:
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def _as_list(value: Any) -> Optional[list]:
    """Gamma encodes list fields either as JSON arrays or as JSON-encoded strings."""
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except (ValueError, TypeError):
            return None
        return parsed if isinstance(parsed, list) else None
    return None


def parse_market(raw: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalize one raw Gamma market dict into a flat shape, or None if unusable.

    Only BINARY (two-outcome) markets are kept. Outcome index 0 is treated as the
    YES side (Gamma convention for Yes/No markets; verified by outcome labels when
    present). Prices are clamped to [0, 1].
    """
    if not isinstance(raw, dict):
        return None
    market_id = raw.get("id") or raw.get("conditionId")
    if market_id is None:
        return None

    prices = _as_list(raw.get("outcomePrices"))
    outcomes = _as_list(raw.get("outcomes"))
    token_ids = _as_list(raw.get("clobTokenIds"))
    if not prices or len(prices) != 2:
        return None  # not binary or prices missing — skip, never guess

    yes_idx, no_idx = 0, 1
    if outcomes and len(outcomes) == 2:
        labels = [str(o).strip().lower() for o in outcomes]
        if "no" in labels and "yes" in labels:
            yes_idx = labels.index("yes")
            no_idx = labels.index("no")
        elif labels[0] not in ("yes", "no"):
            # Non Yes/No binary market (e.g. TeamA/TeamB): keep index order,
            # downstream treats index 0 as the YES-equivalent leg.
            pass

    yes_price = _as_float(prices[yes_idx])
    no_price = _as_float(prices[no_idx])
    if yes_price is None or no_price is None:
        return None
    yes_price = min(max(yes_price, 0.0), 1.0)
    no_price = min(max(no_price, 0.0), 1.0)

    yes_token = no_token = None
    if token_ids and len(token_ids) == 2:
        yes_token = str(token_ids[yes_idx])
        no_token = str(token_ids[no_idx])

    return {
        "market_id": str(market_id),
        "question": str(raw.get("question") or "")[:500],
        "category": str(raw.get("category") or ""),
        "yes_price": yes_price,
        "no_price": no_price,
        "yes_token_id": yes_token,
        "no_token_id": no_token,
        "best_bid": _as_float(raw.get("bestBid")),
        "best_ask": _as_float(raw.get("bestAsk")),
        "volume_24h": _as_float(raw.get("volume24hr")) or 0.0,
        "liquidity": _as_float(raw.get("liquidity")) or 0.0,
        "end_date": raw.get("endDate"),
        "active": bool(raw.get("active", False)),
        "closed": bool(raw.get("closed", False)),
    }


class GammaClient:
    """Cached + rate-limited GET wrapper over the Gamma REST API."""

    def __init__(
        self,
        base_url: str = DEFAULT_GAMMA_BASE_URL,
        *,
        timeout_s: float = 10.0,
        cache_ttl_s: float = 30.0,
        max_requests_per_minute: int = 30,
    ):
        self.base_url = (base_url or DEFAULT_GAMMA_BASE_URL).rstrip("/")
        self.timeout_s = timeout_s
        self.cache_ttl_s = cache_ttl_s
        self.max_requests_per_minute = max(1, int(max_requests_per_minute))
        self._request_times: List[float] = []
        self._cache: Dict[str, Any] = {}  # url -> (fetched_at, payload)
        self.last_error: Optional[str] = None
        self.last_fetch_at: Optional[float] = None

    async def _throttle(self) -> None:
        now = time.monotonic()
        self._request_times = [t for t in self._request_times if now - t < 60.0]
        if len(self._request_times) >= self.max_requests_per_minute:
            wait = 60.0 - (now - self._request_times[0]) + 0.05
            await asyncio.sleep(max(wait, 0.05))
        self._request_times.append(time.monotonic())

    async def _get_json(self, path: str, params: Dict[str, Any]) -> Optional[Any]:
        """One GET. Returns parsed JSON or None. Never raises."""
        query = "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        cache_key = f"{path}?{query}"
        cached = self._cache.get(cache_key)
        if cached and (time.monotonic() - cached[0]) < self.cache_ttl_s:
            return cached[1]
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
                async with session.get(url, params=params) as resp:
                    if resp.status != 200:
                        self.last_error = f"HTTP {resp.status} on {path}"
                        logger.warning("Gamma %s -> HTTP %s", path, resp.status)
                        return None
                    payload = await resp.json(content_type=None)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # fail-soft: any network/parse error -> None
            self.last_error = f"{type(e).__name__}: {e}"
            logger.warning("Gamma fetch failed (%s): %s", path, self.last_error)
            return None
        self.last_error = None
        self.last_fetch_at = time.time()
        self._cache[cache_key] = (time.monotonic(), payload)
        if len(self._cache) > 64:
            oldest = min(self._cache, key=lambda k: self._cache[k][0])
            self._cache.pop(oldest, None)
        return payload

    async def fetch_active_markets(
        self,
        *,
        limit: int = 200,
        page_size: int = 100,
        category_filter: str = "",
    ) -> List[Dict[str, Any]]:
        """Fetch + normalize active, open, binary markets. [] on any failure."""
        out: List[Dict[str, Any]] = []
        offset = 0
        wanted = max(1, min(int(limit), 1000))
        categories = {c.strip().lower() for c in category_filter.split(",") if c.strip()}
        while len(out) < wanted:
            batch = await self._get_json(
                "/markets",
                {
                    "active": "true",
                    "closed": "false",
                    "limit": min(page_size, wanted - len(out)),
                    "offset": offset,
                    "order": "volume24hr",
                    "ascending": "false",
                },
            )
            if not isinstance(batch, list) or not batch:
                break
            for raw in batch:
                parsed = parse_market(raw)
                if parsed is None or parsed["closed"] or not parsed["active"]:
                    continue
                if categories and parsed["category"].lower() not in categories:
                    continue
                out.append(parsed)
            if len(batch) < page_size:
                break
            offset += len(batch)
        return out[:wanted]


def _self_test() -> int:
    """Offline parse self-test against the static fixture. Returns exit code."""
    from pathlib import Path

    fixture = Path(__file__).parent / "fixtures" / "gamma_markets_fixture.json"
    raw_markets = json.loads(fixture.read_text())
    parsed = [m for m in (parse_market(r) for r in raw_markets) if m is not None]
    assert len(parsed) == 2, f"expected 2 parseable fixture markets, got {len(parsed)}"
    m0 = parsed[0]
    assert m0["market_id"] == "500001"
    assert abs(m0["yes_price"] - 0.46) < 1e-9 and abs(m0["no_price"] - 0.51) < 1e-9
    assert m0["yes_token_id"] and m0["no_token_id"]
    m1 = parsed[1]
    assert m1["market_id"] == "500002" and abs(m1["volume_24h"] - 54000) < 1e-9
    assert parse_market({"id": "x", "outcomePrices": "broken"}) is None
    assert parse_market("not-a-dict") is None
    print("gamma_client self-test OK (2 parsed, malformed + non-binary skipped)")
    return 0


if __name__ == "__main__":
    raise SystemExit(_self_test())
