"""
fonoloji_client — shared client for the Fonoloji API (https://fonoloji.com/v1).

Fonoloji is the operator's verified Turkish-market data backbone for the ADVICE-
ONLY Financial Advisor: TEFAS fund NAV history, BIST equities (chart / list /
screener / movers), and gold + live indices (USD/TRY, BIST100, ...).

VERIFIED CONTRACT (from the operator's official api-docs, 2026-06):
  - Base URL   : https://fonoloji.com/v1
  - Auth       : header `X-API-Key: <key>` (NO Bearer/scheme) OR `?api_key=<key>`
  - Free tier  : 15,000 req/MONTH (~500/day) + per-minute + per-day caps.
  - Rate-limit headers: x-ratelimit-remaining, x-ratelimit-remaining-monthly,
                 retry-after. Honor 429/503 + retry-after.
  - Dates ISO-8601, money in TRY.

Endpoints used by the advisor:
  GET /funds/{code}/history?period=1w|1m|3m|6m|1y|5y|all
      -> {code, period, points:[{date, price, total_value, investor_count}]}
  GET /funds/{code}
      -> {fund:{code,name,current_price,current_date,return_1y,...}, portfolio:{...}}
  GET /stocks/{ticker}/chart?period=1d|5d|1mo|3mo|6mo|1y|5y
      -> price series (date + price/close; mapped defensively)
  GET /stocks/list                 -> [{name, sector, last price, ...}]
  GET /screener/bist?pe_max=&pb_max=&roe_min=&...&sort_by=&sort_order=&limit=
  GET /market/stock-movers         -> BIST gainers/losers (intraday)
  GET /gold/live                   -> gram/çeyrek/ons
  GET /market/live                 -> BIST100, USD/TRY, EUR/TRY, silver

RATE-LIMIT DISCIPLINE (hard requirement): a process-wide TTL cache (default 6h,
configurable `advisor_fonoloji_cache_ttl_s`) keyed per endpoint+params. Fund NAV
and BIST bars are DAILY data, so one daily-ish refresh per symbol keeps usage
far under 15k/month. The monthly-remaining header is logged (WARNING once) so the
operator can watch quota; when it is low we prefer serving stale cache.

FAIL-SOFT: any error / missing key / unexpected shape => the caller-facing
methods return None so the analyzer falls back to its existing source. The client
NEVER raises into the advice loop.

Self-test (no live network): python -m modules.advisor.core.data.fonoloji_client
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("advisor.data.fonoloji")

# Verified contract defaults (operator-overridable via advisor_config).
_DEFAULT_BASE_URL = "https://fonoloji.com/v1"
_DEFAULT_AUTH_HEADER = "X-API-Key"
_DEFAULT_CACHE_TTL_S = 21600          # 6 hours
_DEFAULT_TIMEOUT_S = 15.0
_MONTHLY_LOW_THRESHOLD = 500          # warn once below this monthly-remaining
_MAX_RETRY_AFTER_S = 30.0             # never block the loop longer than this
_MAX_RETRIES = 2                      # 429/503 retries (bounded)

# Process-wide cache shared across analyzer instances (the advisor reconstructs
# analyzers each cycle, so an instance-level cache would never hit). Keyed by
# (api_key, base_url, path, frozenset(params)).
_CACHE: Dict[Tuple, Tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()
_MONTHLY_WARNED = False
_MONTHLY_WARN_LOCK = threading.Lock()


def resolve_api_key(config: dict) -> str:
    """Resolve the Fonoloji API key from config (Secure Credentials injection)
    then env. Empty string if unset (=> client disabled, fail-soft)."""
    return (
        str((config or {}).get("advisor_fonoloji_api_key", "") or "").strip()
        or os.getenv("ADVISOR_FONOLOJI_API_KEY", "").strip()
    )


def _cache_ttl_s(config: dict) -> float:
    try:
        ttl = float((config or {}).get("advisor_fonoloji_cache_ttl_s", _DEFAULT_CACHE_TTL_S))
        return max(0.0, ttl)
    except (TypeError, ValueError):
        return float(_DEFAULT_CACHE_TTL_S)


class FonolojiClient:
    """
    Thin, fail-soft, cached HTTP client for the Fonoloji v1 API.

    Construct with the advisor config dict; reads the key/base/header/TTL from
    it (with env + verified-contract defaults). `enabled` is True only when a key
    is present — callers should branch on it so that NO key => behavior unchanged.

    All public fetch methods return parsed JSON (dict/list) or None on any
    failure. Transport is injectable for offline self-tests (`transport=`).
    """

    def __init__(self, config: dict, transport=None):
        self.config = config or {}
        self.api_key = resolve_api_key(self.config)
        self.base_url = str(
            self.config.get("advisor_fonoloji_base_url", _DEFAULT_BASE_URL)
            or _DEFAULT_BASE_URL
        ).rstrip("/")
        self.auth_header = str(
            self.config.get("advisor_fonoloji_auth_header", _DEFAULT_AUTH_HEADER)
            or _DEFAULT_AUTH_HEADER
        ).strip()
        self.cache_ttl_s = _cache_ttl_s(self.config)
        # Injectable transport for tests: callable(url, headers, params, timeout)
        # -> _MockResponse-like object with .status_code, .headers, .json().
        self._transport = transport

    @property
    def enabled(self) -> bool:
        return bool(self.api_key)

    # ------------------------------------------------------------------
    # Endpoint wrappers (verified contract)
    # ------------------------------------------------------------------

    def fund_history(self, code: str, period: str = "1y") -> Optional[dict]:
        """GET /funds/{code}/history?period=... -> {code, period, points:[...]}."""
        return self._get(f"/funds/{code.upper()}/history", {"period": period})

    def fund_detail(self, code: str) -> Optional[dict]:
        """GET /funds/{code} -> {fund:{...}, portfolio:{...}}."""
        return self._get(f"/funds/{code.upper()}", {})

    def stock_chart(self, ticker: str, period: str = "1y") -> Optional[dict]:
        """GET /stocks/{ticker}/chart?period=... -> price series."""
        return self._get(f"/stocks/{_bare_ticker(ticker)}/chart", {"period": period})

    def stock_list(self) -> Optional[Any]:
        """GET /stocks/list -> [{name, sector, last price, ...}]."""
        return self._get("/stocks/list", {})

    def screener_bist(self, **filters) -> Optional[Any]:
        """GET /screener/bist?pe_max=&pb_max=&roe_min=&... (all optional)."""
        params = {k: v for k, v in filters.items() if v is not None and v != ""}
        return self._get("/screener/bist", params)

    def stock_movers(self) -> Optional[Any]:
        """GET /market/stock-movers -> BIST gainers/losers (intraday)."""
        return self._get("/market/stock-movers", {})

    def gold_live(self) -> Optional[dict]:
        """GET /gold/live -> gram/çeyrek/ons."""
        return self._get("/gold/live", {})

    def market_live(self) -> Optional[dict]:
        """GET /market/live -> BIST100, USD/TRY, EUR/TRY, silver, ..."""
        return self._get("/market/live", {})

    # ------------------------------------------------------------------
    # AI / analyst endpoints (verified contract). FREE within the 15k/month
    # quota — NO per-token cost. ai-summary + market/digest are a READONLY DB
    # cache that 404s until warmed by a fonoloji.com page visit; 404 => None
    # (handled in _get as a non-error), so callers fail-soft. The
    # recommendations / analyst-consensus JSON shapes are mapped DEFENSIVELY by
    # the callers (bist.py) — confirm exact keys on the first live call.
    # ------------------------------------------------------------------

    def fund_ai_summary(self, code: str) -> Optional[dict]:
        """GET /funds/{code}/ai-summary
        -> {code, summary, cached, model, generated_at}. Turkish 3-5 sentence
        AI fund summary. Used to REPLACE the paid Anthropic rationale for
        Turkish funds (a saving, not a swap). 404 (not yet warmed) => None."""
        return self._get(f"/funds/{code.upper()}/ai-summary", {})

    def market_digest(self) -> Optional[dict]:
        """GET /market/digest -> AI daily market digest (DB cache, readonly).
        404 (not yet warmed) => None."""
        return self._get("/market/digest", {})

    def stock_recommendations(self, ticker: str) -> Optional[Any]:
        """GET /stocks/{ticker}/recommendations -> broker recommendations
        (Turkish: AL=buy / TUT=hold / SAT=sell + target price). Shape mapped
        defensively by the caller; confirm on first live call. 404 => None."""
        return self._get(f"/stocks/{_bare_ticker(ticker)}/recommendations", {})

    def fund_analyst_consensus(self, code: str) -> Optional[Any]:
        """GET /funds/{code}/analyst-consensus -> broker target prices for the
        fund's holdings. 404 => None."""
        return self._get(f"/funds/{code.upper()}/analyst-consensus", {})

    # ------------------------------------------------------------------
    # Core GET with TTL cache + 429/503 retry-after + quota awareness
    # ------------------------------------------------------------------

    def _get(self, path: str, params: dict) -> Optional[Any]:
        if not self.enabled:
            return None

        cache_key = (
            self.api_key, self.base_url, path,
            frozenset((k, str(v)) for k, v in (params or {}).items()),
        )
        now = time.monotonic()

        # Fresh cache hit.
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            ts, payload = cached
            if (now - ts) <= self.cache_ttl_s:
                return payload

        fresh = self._fetch_remote(path, params)
        if fresh is not None:
            with _CACHE_LOCK:
                _CACHE[cache_key] = (now, fresh)
            return fresh

        # Fetch failed — serve STALE cache if we have any (fail-soft + quota-safe).
        if cached is not None:
            logger.debug("[fonoloji] %s fetch failed; serving stale cache.", path)
            return cached[1]
        return None

    def _fetch_remote(self, path: str, params: dict) -> Optional[Any]:
        url = f"{self.base_url}{path if path.startswith('/') else '/' + path}"
        headers = {
            self.auth_header: self.api_key,
            "Accept": "application/json",
            "User-Agent": "claudedex-advisor/1.0",
        }
        # Also pass api_key as a query param per the documented alternative auth
        # (harmless for header-auth; covers gateways that only read the param).
        req_params = dict(params or {})
        req_params.setdefault("api_key", self.api_key)

        attempt = 0
        while attempt <= _MAX_RETRIES:
            attempt += 1
            try:
                resp = self._do_request(url, headers, req_params)
            except Exception as exc:
                logger.debug("[fonoloji] %s transport error: %s", path, exc)
                return None

            status = getattr(resp, "status_code", 0)
            rl_headers = getattr(resp, "headers", {}) or {}
            self._note_quota(rl_headers)

            if status == 200:
                try:
                    return resp.json()
                except Exception as exc:
                    logger.debug("[fonoloji] %s JSON parse failed: %s", path, exc)
                    return None

            if status == 401:
                logger.warning(
                    "[fonoloji] auth rejected (401) for %s — check "
                    "ADVISOR_FONOLOJI_API_KEY / advisor_fonoloji_auth_header.",
                    path,
                )
                return None

            if status == 404:
                logger.debug("[fonoloji] %s not found (404).", path)
                return None

            if status in (429, 503):
                retry_after = _parse_retry_after(rl_headers)
                if attempt > _MAX_RETRIES or retry_after is None:
                    logger.warning(
                        "[fonoloji] %s rate/quota-limited (HTTP %d); giving up "
                        "this cycle (will serve cache).", path, status,
                    )
                    return None
                sleep_s = min(retry_after, _MAX_RETRY_AFTER_S)
                logger.info(
                    "[fonoloji] %s HTTP %d; honoring retry-after=%.1fs (attempt %d).",
                    path, status, sleep_s, attempt,
                )
                time.sleep(sleep_s)
                continue

            if 500 <= status < 600:
                if attempt > _MAX_RETRIES:
                    logger.debug("[fonoloji] %s HTTP %d; giving up.", path, status)
                    return None
                time.sleep(min(5.0, _MAX_RETRY_AFTER_S))
                continue

            logger.debug("[fonoloji] %s unexpected HTTP %d.", path, status)
            return None
        return None

    def _do_request(self, url, headers, params):
        """Perform one HTTP GET. Uses the injected transport if present (tests),
        else `requests`. Returns a response object with status_code/headers/json."""
        if self._transport is not None:
            return self._transport(url, headers, params, _DEFAULT_TIMEOUT_S)
        import requests  # local import: optional dependency, fail-soft on absence
        return requests.get(url, headers=headers, params=params, timeout=_DEFAULT_TIMEOUT_S)

    def _note_quota(self, headers: dict) -> None:
        """Log the monthly-remaining header once when it dips low (operator watch)."""
        global _MONTHLY_WARNED
        try:
            remaining = headers.get("x-ratelimit-remaining-monthly")
            if remaining is None:
                # header keys can be case-insensitive dicts; try a manual scan
                for k, v in (headers or {}).items():
                    if str(k).lower() == "x-ratelimit-remaining-monthly":
                        remaining = v
                        break
            if remaining is None:
                return
            rem = int(float(remaining))
        except (TypeError, ValueError):
            return
        if rem <= _MONTHLY_LOW_THRESHOLD:
            with _MONTHLY_WARN_LOCK:
                if not _MONTHLY_WARNED:
                    _MONTHLY_WARNED = True
                    logger.warning(
                        "[fonoloji] monthly quota LOW: %d requests remaining of "
                        "the 15,000/month free tier — will prefer cached data. "
                        "Reduce watchlist/universe size or raise "
                        "advisor_fonoloji_cache_ttl_s.", rem,
                    )
        else:
            logger.debug("[fonoloji] monthly-remaining=%d", rem)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _bare_ticker(ticker: str) -> str:
    """Strip the yfinance .IS suffix Fonoloji does not use."""
    t = str(ticker or "").strip().upper()
    if t.endswith(".IS"):
        t = t[:-3]
    return t


def _parse_retry_after(headers: dict) -> Optional[float]:
    """Parse the retry-after header (seconds form). None if absent/unparseable."""
    try:
        val = headers.get("retry-after")
        if val is None:
            for k, v in (headers or {}).items():
                if str(k).lower() == "retry-after":
                    val = v
                    break
        if val is None:
            return 5.0  # sane default backoff when the server gives no hint
        return max(0.0, float(val))
    except (TypeError, ValueError):
        return 5.0


def clear_cache() -> None:
    """Test hook: drop the process-wide cache."""
    with _CACHE_LOCK:
        _CACHE.clear()


# ---------------------------------------------------------------------------
# Offline self-test (mock transport — NO live network).
# Run: python -m modules.advisor.core.data.fonoloji_client
# ---------------------------------------------------------------------------

class _MockResponse:
    def __init__(self, status_code, payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}

    def json(self):
        if self._payload is None:
            raise ValueError("no json")
        return self._payload


def _self_test() -> int:
    clear_cache()
    failures = 0
    calls = {"n": 0}

    nav_payload = {
        "code": "TPP",
        "period": "1y",
        "points": [
            {"date": "2026-01-01", "price": 10.0, "total_value": 1e6, "investor_count": 100},
            {"date": "2026-01-02", "price": 10.1, "total_value": 1.1e6, "investor_count": 101},
        ],
    }

    def ok_transport(url, headers, params, timeout):
        calls["n"] += 1
        # Verify verified-contract auth header is set.
        assert headers.get("X-API-Key") == "TESTKEY", headers
        return _MockResponse(200, nav_payload,
                             {"x-ratelimit-remaining-monthly": "9000"})

    cfg = {"advisor_fonoloji_api_key": "TESTKEY"}
    c = FonolojiClient(cfg, transport=ok_transport)

    if not c.enabled:
        print("FAIL: client should be enabled with a key")
        failures += 1
    if c.base_url != _DEFAULT_BASE_URL:
        print(f"FAIL: base_url {c.base_url}")
        failures += 1
    if c.auth_header != "X-API-Key":
        print(f"FAIL: auth_header {c.auth_header}")
        failures += 1

    r1 = c.fund_history("TPP", "1y")
    r2 = c.fund_history("TPP", "1y")  # should hit cache, NOT call transport again
    if r1 != nav_payload or r2 != nav_payload:
        print("FAIL: fund_history payload mismatch")
        failures += 1
    if calls["n"] != 1:
        print(f"FAIL: cache miss — transport called {calls['n']}x (expected 1)")
        failures += 1

    # No key => disabled => None, no transport call.
    c_nokey = FonolojiClient({}, transport=ok_transport)
    if c_nokey.enabled or c_nokey.fund_history("TPP") is not None:
        print("FAIL: no-key client must be disabled and return None")
        failures += 1

    # 401 => None (fail-soft).
    def auth_fail(url, headers, params, timeout):
        return _MockResponse(401, headers={})
    c401 = FonolojiClient(cfg, transport=auth_fail)
    clear_cache()
    if c401.fund_detail("ZZZ") is not None:
        print("FAIL: 401 must return None")
        failures += 1

    # 429 with retry-after=0 then 200 => recovers (bounded retry).
    state = {"hits": 0}

    def rate_then_ok(url, headers, params, timeout):
        state["hits"] += 1
        if state["hits"] == 1:
            return _MockResponse(429, headers={"retry-after": "0"})
        return _MockResponse(200, {"ok": True},
                             {"x-ratelimit-remaining-monthly": "100"})
    c429 = FonolojiClient(cfg, transport=rate_then_ok)
    clear_cache()
    if c429.gold_live() != {"ok": True}:
        print("FAIL: 429-then-200 should recover")
        failures += 1

    # Stale-cache fallback: prime cache, then make transport fail.
    served = {"phase": "ok"}

    def flaky(url, headers, params, timeout):
        if served["phase"] == "ok":
            return _MockResponse(200, {"v": 1},
                                 {"x-ratelimit-remaining-monthly": "9000"})
        raise RuntimeError("network down")
    cflaky = FonolojiClient({"advisor_fonoloji_api_key": "K",
                             "advisor_fonoloji_cache_ttl_s": "0"},
                            transport=flaky)
    clear_cache()
    first = cflaky.market_live()       # populates cache (ttl=0 => always "stale")
    served["phase"] = "down"
    second = cflaky.market_live()      # fetch fails -> serve stale cache
    if first != {"v": 1} or second != {"v": 1}:
        print(f"FAIL: stale-cache fallback {first} {second}")
        failures += 1

    # AI/analyst endpoints: 200 payload returns, correct path, 404 => None.
    seen_paths = {"p": []}

    def ai_transport(url, headers, params, timeout):
        seen_paths["p"].append(url)
        if url.endswith("/ai-summary"):
            return _MockResponse(
                200,
                {"code": "TPP", "summary": "Fon istikrarli getiri sagliyor.",
                 "cached": True, "model": "gpt", "generated_at": "2026-06-04"},
                {"x-ratelimit-remaining-monthly": "9000"},
            )
        if url.endswith("/recommendations"):
            return _MockResponse(
                200,
                {"ticker": "THYAO",
                 "recommendations": [{"broker": "X", "rating": "AL",
                                      "target_price": 350.0}]},
                {"x-ratelimit-remaining-monthly": "9000"},
            )
        if url.endswith("/digest"):
            return _MockResponse(404, headers={})  # not yet warmed
        return _MockResponse(200, {"ok": True}, {})

    c_ai = FonolojiClient(cfg, transport=ai_transport)
    clear_cache()
    s = c_ai.fund_ai_summary("tpp")
    if not (s and s.get("summary")):
        print("FAIL: fund_ai_summary should return payload")
        failures += 1
    if not any(u.endswith("/funds/TPP/ai-summary") for u in seen_paths["p"]):
        print(f"FAIL: ai-summary path wrong: {seen_paths['p']}")
        failures += 1
    rec = c_ai.stock_recommendations("THYAO.IS")
    if not (rec and rec.get("recommendations")):
        print("FAIL: stock_recommendations should return payload")
        failures += 1
    if not any(u.endswith("/stocks/THYAO/recommendations") for u in seen_paths["p"]):
        print(f"FAIL: recommendations path wrong (suffix strip): {seen_paths['p']}")
        failures += 1
    if c_ai.market_digest() is not None:
        print("FAIL: market_digest 404 must return None (fail-soft)")
        failures += 1
    cons = c_ai.fund_analyst_consensus("TPP")
    if cons != {"ok": True}:
        print("FAIL: fund_analyst_consensus should return payload")
        failures += 1

    print("SELF-TEST", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(_self_test())
