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

import json
import logging
import os
import threading
import time
from typing import Any, Dict, Optional, Tuple

logger = logging.getLogger("advisor.data.fonoloji")

# Verified contract defaults (operator-overridable via advisor_config).
_DEFAULT_BASE_URL = "https://fonoloji.com/v1"
_DEFAULT_AUTH_HEADER = "X-API-Key"
_DEFAULT_CACHE_TTL_S = 10800          # 3 hours (Developer plan, migration 083)
_DEFAULT_TIMEOUT_S = 15.0
# Developer plan (2026-06 upgrade): 30,000/month, 3,000/day, 60/min.
_MONTHLY_LOW_THRESHOLD = 1500         # warn once below this monthly-remaining
_DEFAULT_DAILY_BUDGET = 3000          # plan per-day cap (config-overridable)
_DEFAULT_DAILY_WARN_PCT = 0.8         # warn once at 80% of the daily budget
_MAX_RETRY_AFTER_S = 30.0             # never block the loop longer than this
_MAX_RETRIES = 2                      # 429/503 retries (bounded)

# Per-endpoint TTL FLOORS (seconds). Slow-moving data is cached LONGER than the
# base TTL so the daily budget is spent on price/NAV series, not static lists.
# Effective TTL = max(base advisor_fonoloji_cache_ttl_s, floor). Matched by
# substring against the request path.
_TTL_FLOORS: Dict[str, float] = {
    "/stocks/list": 86400.0,          # canonical equity list: daily
    "/recommendations": 86400.0,      # broker ratings: daily
    "/economy/cpi": 86400.0,          # CPI: monthly series, daily refresh plenty
    "/percentile": 86400.0,           # fund category percentile: daily
    "/estimate-accuracy": 86400.0,    # estimate honesty stat: daily
    "/insights/": 21600.0,            # movers / flow / trend: 6h
    "/tools/": 21600.0,               # portfolio-xray / fund-overlap: 6h
}
# Endpoints allowed a SHORTER TTL than the base (intraday by nature).
_TTL_CEILINGS: Dict[str, float] = {
    "/live-estimate": 3600.0,         # intraday NAV estimate: 1h
    "/market/live": 3600.0,           # live indices: 1h
}

# Process-wide cache shared across analyzer instances (the advisor reconstructs
# analyzers each cycle, so an instance-level cache would never hit). Keyed by
# (api_key, base_url, method, path, frozenset(params/body)).
_CACHE: Dict[Tuple, Tuple[float, Any]] = {}
_CACHE_LOCK = threading.Lock()
_MONTHLY_WARNED = False
_MONTHLY_WARN_LOCK = threading.Lock()

# Daily LOCAL request counter (UTC day) — budget guard + usage logging. The
# plan enforces 3,000/day server-side; this guard keeps the advisor from
# burning the whole day's quota on a misconfiguration, logs usage every 250
# calls, and prefers stale cache once the local budget is exhausted.
_DAILY = {"day": "", "count": 0, "warned": False, "exhausted": False}
_DAILY_LOCK = threading.Lock()
# Last rate-limit headers seen (startup diagnostics / dashboard surface).
_LAST_QUOTA: Dict[str, Any] = {}


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
    # Developer-plan endpoints (2026-06 upgrade). All JSON shapes are mapped
    # DEFENSIVELY by the callers — no live Fonoloji access in dev, so confirm
    # exact keys on the first live call. Every method is fail-soft (None).
    # ------------------------------------------------------------------

    def funds_list(self, sort: Optional[str] = None,
                   limit: Optional[int] = None, **filters) -> Optional[Any]:
        """GET /funds?sort=...&limit=... -> fund list with rich metrics
        (sharpe_90, sortino, calmar, max_drawdown_1y, return_*, aum,
        investor_count). Basis of the Top Funds screener."""
        params = {k: v for k, v in dict(filters, sort=sort, limit=limit).items()
                  if v is not None and v != ""}
        return self._get("/funds", params)

    def fund_percentile(self, code: str) -> Optional[Any]:
        """GET /funds/{code}/percentile -> category percentile context."""
        return self._get(f"/funds/{code.upper()}/percentile", {})

    def fund_live_estimate(self, code: str) -> Optional[Any]:
        """GET /funds/{code}/live-estimate -> intraday NAV estimate (1h TTL)."""
        return self._get(f"/funds/{code.upper()}/live-estimate", {})

    def fund_estimate_accuracy(self, code: str) -> Optional[Any]:
        """GET /funds/{code}/estimate-accuracy -> historical accuracy of the
        live estimate (honesty note for the operator)."""
        return self._get(f"/funds/{code.upper()}/estimate-accuracy", {})

    def insights_movers(self) -> Optional[Any]:
        """GET /insights/movers -> fund momentum leaders."""
        return self._get("/insights/movers", {})

    def insights_flow(self) -> Optional[Any]:
        """GET /insights/flow -> money inflow leaders (fund flows)."""
        return self._get("/insights/flow", {})

    def insights_trend(self) -> Optional[Any]:
        """GET /insights/trend -> MA30/200 rising/falling trend signals."""
        return self._get("/insights/trend", {})

    def market_patterns(self) -> Optional[Any]:
        """GET /market/patterns -> TradingView Candle.* pattern matches."""
        return self._get("/market/patterns", {})

    def economy_cpi(self) -> Optional[Any]:
        """GET /economy/cpi -> Turkish CPI series (real-return context)."""
        return self._get("/economy/cpi", {})

    def gold_compare(self) -> Optional[Any]:
        """GET /gold/compare -> fund returns vs gram-gold parity."""
        return self._get("/gold/compare", {})

    def stock_price(self, ticker: str) -> Optional[Any]:
        """GET /stocks/{ticker}/price -> last price for one BIST equity
        (the cheap KAP/accumulator price path)."""
        return self._get(f"/stocks/{_bare_ticker(ticker)}/price", {})

    def summary_today(self) -> Optional[Any]:
        """GET /summary/today -> daily market summary (also the probe target)."""
        return self._get("/summary/today", {})

    def portfolio_xray(self, holdings: list) -> Optional[Any]:
        """POST /tools/portfolio-xray with {holdings:[{code, weight|amount}]}.
        Cached 6h; shape confirmed on first live call."""
        return self._post("/tools/portfolio-xray", {"holdings": holdings})

    def fund_overlap(self, codes: list) -> Optional[Any]:
        """POST /tools/fund-overlap with {codes:[...]} -> holdings overlap."""
        return self._post("/tools/fund-overlap",
                          {"codes": [str(c).upper() for c in (codes or [])]})

    # ------------------------------------------------------------------
    # Core GET with TTL cache + 429/503 retry-after + quota awareness
    # ------------------------------------------------------------------

    def _ttl_for(self, path: str) -> float:
        """Effective TTL for a path: base TTL raised to any matching floor,
        lowered to any matching ceiling (intraday endpoints)."""
        ttl = self.cache_ttl_s
        for frag, floor in _TTL_FLOORS.items():
            if frag in path:
                ttl = max(ttl, floor)
        for frag, ceil in _TTL_CEILINGS.items():
            if frag in path:
                ttl = min(ttl if ttl > 0 else ceil, ceil)
        return ttl

    def _daily_budget(self) -> int:
        try:
            return max(1, int(float(self.config.get(
                "advisor_fonoloji_daily_budget", _DEFAULT_DAILY_BUDGET))))
        except (TypeError, ValueError):
            return _DEFAULT_DAILY_BUDGET

    def _budget_allows(self) -> bool:
        """Count one prospective remote call against the UTC-day local budget.
        Returns False (skip the fetch, serve stale/None) once exhausted."""
        import datetime as _dt
        today = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%d")
        budget = self._daily_budget()
        warn_at = budget * _daily_warn_pct(self.config)
        with _DAILY_LOCK:
            if _DAILY["day"] != today:
                if _DAILY["day"] and _DAILY["count"]:
                    logger.info(
                        "[fonoloji] daily usage %s: %d local request(s) "
                        "(budget %d).", _DAILY["day"], _DAILY["count"], budget)
                _DAILY.update(day=today, count=0, warned=False, exhausted=False)
            if _DAILY["count"] >= budget:
                if not _DAILY["exhausted"]:
                    _DAILY["exhausted"] = True
                    logger.warning(
                        "[fonoloji] local daily budget EXHAUSTED (%d/%d) — "
                        "serving cache only until UTC midnight. Raise "
                        "advisor_fonoloji_daily_budget or cache TTLs if this "
                        "recurs.", _DAILY["count"], budget)
                return False
            _DAILY["count"] += 1
            if _DAILY["count"] >= warn_at and not _DAILY["warned"]:
                _DAILY["warned"] = True
                logger.warning(
                    "[fonoloji] daily usage at %d/%d (>=%.0f%% of budget).",
                    _DAILY["count"], budget,
                    100.0 * _daily_warn_pct(self.config))
            if _DAILY["count"] % 250 == 0:
                logger.info("[fonoloji] daily usage: %d/%d local request(s).",
                            _DAILY["count"], budget)
        return True

    def _get(self, path: str, params: dict) -> Optional[Any]:
        if not self.enabled:
            return None

        cache_key = (
            self.api_key, self.base_url, "GET", path,
            frozenset((k, str(v)) for k, v in (params or {}).items()),
        )
        now = time.monotonic()

        # Fresh cache hit.
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            ts, payload = cached
            if (now - ts) <= self._ttl_for(path):
                return payload

        # Local daily-budget guard: when exhausted, prefer stale cache.
        if not self._budget_allows():
            return cached[1] if cached is not None else None

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

    def _post(self, path: str, body: dict) -> Optional[Any]:
        """Cached POST (the /tools/* endpoints are deterministic for a given
        body, so the TTL cache + daily budget guard apply exactly like _get)."""
        if not self.enabled:
            return None
        try:
            body_key = json.dumps(body or {}, sort_keys=True, default=str)
        except Exception:
            body_key = str(body)
        cache_key = (self.api_key, self.base_url, "POST", path, body_key)
        now = time.monotonic()
        with _CACHE_LOCK:
            cached = _CACHE.get(cache_key)
        if cached is not None:
            ts, payload = cached
            if (now - ts) <= self._ttl_for(path):
                return payload
        if not self._budget_allows():
            return cached[1] if cached is not None else None
        fresh = self._fetch_remote(path, {}, json_body=body or {})
        if fresh is not None:
            with _CACHE_LOCK:
                _CACHE[cache_key] = (now, fresh)
            return fresh
        if cached is not None:
            logger.debug("[fonoloji] %s POST failed; serving stale cache.", path)
            return cached[1]
        return None

    def _fetch_remote(self, path: str, params: dict,
                      json_body: Optional[dict] = None) -> Optional[Any]:
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
                resp = self._do_request(url, headers, req_params,
                                        json_body=json_body)
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

    def _do_request(self, url, headers, params, json_body=None):
        """Perform one HTTP GET (or POST when json_body is given). Uses the
        injected transport if present (tests), else `requests`. Returns a
        response object with status_code/headers/json. The test transport
        signature stays (url, headers, params, timeout); POST bodies are passed
        through params under the reserved '_json' key for mock transports."""
        if self._transport is not None:
            p = dict(params or {})
            if json_body is not None:
                p["_json"] = json_body
            return self._transport(url, headers, p, _DEFAULT_TIMEOUT_S)
        import requests  # local import: optional dependency, fail-soft on absence
        if json_body is not None:
            return requests.post(url, headers=headers, params=params,
                                 json=json_body, timeout=_DEFAULT_TIMEOUT_S)
        return requests.get(url, headers=headers, params=params,
                            timeout=_DEFAULT_TIMEOUT_S)

    def probe(self) -> dict:
        """
        ONE diagnostic call for the startup banner: GET /summary/today (the
        cheapest documented endpoint), bypassing nothing (it warms the cache).
        Returns {"ok": bool, "quota": {...rate-limit headers...}}. Never raises.
        """
        out = {"ok": False, "quota": {}}
        if not self.enabled:
            return out
        try:
            payload = self._get("/summary/today", {})
            out["ok"] = payload is not None
        except Exception as exc:
            logger.debug("[fonoloji] probe failed: %s", exc)
        out["quota"] = dict(_LAST_QUOTA)
        return out

    @staticmethod
    def daily_usage() -> dict:
        """Local daily-counter snapshot (for diagnostics/dashboard)."""
        with _DAILY_LOCK:
            return dict(_DAILY)

    def _note_quota(self, headers: dict) -> None:
        """Log the monthly-remaining header once when it dips low (operator watch)."""
        global _MONTHLY_WARNED
        try:
            for k, v in (headers or {}).items():
                kl = str(k).lower()
                if kl.startswith("x-ratelimit") or kl == "retry-after":
                    _LAST_QUOTA[kl] = v
        except Exception:
            pass
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
                        "the 30,000/month Developer plan — will prefer cached "
                        "data. Reduce watchlist/universe size or raise "
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


def _daily_warn_pct(config: dict) -> float:
    """Fraction of the daily budget at which to warn once (default 0.8)."""
    try:
        pct = float((config or {}).get(
            "advisor_fonoloji_daily_warn_pct", _DEFAULT_DAILY_WARN_PCT))
        return min(1.0, max(0.05, pct))
    except (TypeError, ValueError):
        return _DEFAULT_DAILY_WARN_PCT


def clear_cache() -> None:
    """Test hook: drop the process-wide cache + reset the daily counter."""
    with _CACHE_LOCK:
        _CACHE.clear()
    with _DAILY_LOCK:
        _DAILY.update(day="", count=0, warned=False, exhausted=False)


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

    # --- Developer-plan endpoints: paths + POST tools (mock transport). ---
    clear_cache()
    seen = {"urls": [], "json": []}

    def dev_transport(url, headers, params, timeout):
        seen["urls"].append(url)
        if "_json" in params:
            seen["json"].append(params["_json"])
        return _MockResponse(200, {"ok": True}, {})

    c_dev = FonolojiClient(cfg, transport=dev_transport)
    c_dev.funds_list(sort="sharpe_90", limit=20)
    c_dev.fund_percentile("tpp")
    c_dev.stock_price("THYAO.IS")
    c_dev.portfolio_xray([{"code": "TPP", "weight": 0.5}])
    c_dev.fund_overlap(["tpp", "AKP"])
    want_suffixes = ("/funds", "/funds/TPP/percentile", "/stocks/THYAO/price",
                     "/tools/portfolio-xray", "/tools/fund-overlap")
    for suf in want_suffixes:
        if not any(u.endswith(suf) for u in seen["urls"]):
            print(f"FAIL: endpoint path missing: {suf} in {seen['urls']}")
            failures += 1
    if seen["json"] and seen["json"][-1] != {"codes": ["TPP", "AKP"]}:
        print(f"FAIL: fund_overlap body {seen['json'][-1]}")
        failures += 1
    # POST result is cached: repeat must NOT hit transport again.
    n_before = len(seen["urls"])
    c_dev.fund_overlap(["tpp", "AKP"])
    if len(seen["urls"]) != n_before:
        print("FAIL: POST cache miss on identical body")
        failures += 1
    clear_cache()

    # --- TTL floors/ceilings: static lists cached >= 1 day, live <= 1h. ---
    c_ttl = FonolojiClient(cfg, transport=ok_transport)
    if c_ttl._ttl_for("/stocks/list") < 86400.0:
        print("FAIL: /stocks/list TTL floor not applied")
        failures += 1
    if c_ttl._ttl_for("/funds/TPP/live-estimate") > 3600.0:
        print("FAIL: live-estimate TTL ceiling not applied")
        failures += 1
    if c_ttl._ttl_for("/funds/TPP/history") != c_ttl.cache_ttl_s:
        print("FAIL: default TTL changed for plain endpoints")
        failures += 1

    # --- Daily budget guard: budget=2 -> third distinct fetch is skipped. ---
    clear_cache()
    budget_calls = {"n": 0}

    def counting_transport(url, headers, params, timeout):
        budget_calls["n"] += 1
        return _MockResponse(200, {"u": url}, {})
    c_budget = FonolojiClient(
        {"advisor_fonoloji_api_key": "K", "advisor_fonoloji_daily_budget": "2"},
        transport=counting_transport,
    )
    c_budget.fund_detail("AAA")
    c_budget.fund_detail("BBB")
    r3 = c_budget.fund_detail("CCC")   # over budget -> no fetch, None
    if budget_calls["n"] != 2 or r3 is not None:
        print(f"FAIL: daily budget guard (calls={budget_calls['n']}, r3={r3})")
        failures += 1
    # ... but a CACHED endpoint still serves (stale ok).
    if c_budget.fund_detail("AAA") is None:
        print("FAIL: budget exhaustion must still serve cache")
        failures += 1
    clear_cache()

    print("SELF-TEST", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(_self_test())
