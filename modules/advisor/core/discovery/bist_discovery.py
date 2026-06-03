"""
bist_discovery — Borsa Istanbul gainers / most-active (degraded, fail-soft).

This is the FRAGILE discoverer, by design. Borsa Istanbul has no clean free
movers API. We try, in order, the cheapest free options and FAIL SOFT to an
empty list at every step:

  1. borsapy listing methods, IF the installed version exposes any (the API
     surface varies by version — we probe a few likely names defensively and
     accept whatever shape we get). borsapy is already an advisor dependency
     (used by bist.py via get_history).
  2. A configurable light public scrape (advisor_discovery_bist_scrape_url) —
     OFF by default (empty url). Operators can point this at a free movers JSON
     if they have one; we never ship a hard-coded scrape target because BIST
     endpoints are unstable and region-gated.

If neither yields data, discovery for BIST is simply empty for that cycle —
the watchlist BIST advice is completely unaffected.

Returns ".IS"-suffixed tickers (e.g. "THYAO.IS") for the existing BISTAnalyzer.

CAVEAT: expect this to return [] in most environments. That is acceptable and
intended — BIST discovery is best-effort. Trending != good.
"""

from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Set

from modules.advisor.core.discovery.base import (
    DiscoveryCandidate,
    cfg_float,
    cfg_int,
    screen_dedupe_rank,
)

logger = logging.getLogger("advisor.discovery.bist")

# borsapy listing methods we will probe (version-dependent; all optional).
_PROBE_LIST_METHODS = (
    "get_gainers", "gainers", "get_most_active", "most_active",
    "get_index_constituents", "get_all_symbols", "list_symbols",
)


async def discover(
    config: dict,
    exclude: Set[str],
    top_n: int,
) -> List[DiscoveryCandidate]:
    """Return up to top_n NEW BIST candidates. Fail-soft -> []."""
    try:
        min_abs_chg = cfg_float(config, "advisor_discovery_bist_min_abs_change_pct", 2.0)
        universe_cap = cfg_int(config, "advisor_discovery_bist_universe_cap", 100)

        loop = asyncio.get_event_loop()
        candidates = await loop.run_in_executor(
            None, _fetch_borsapy_movers, universe_cap
        )

        if not candidates:
            scrape_url = str(config.get("advisor_discovery_bist_scrape_url", "")).strip()
            if scrape_url:
                candidates = await _fetch_scrape(scrape_url, universe_cap)

        if not candidates:
            logger.info("[discovery.bist] no candidates (fail-soft / degraded).")
            return []

        ranked = screen_dedupe_rank(
            candidates,
            exclude,
            min_abs_change_pct=min_abs_chg,
            exclude_stablecoins=False,
            top_n=top_n,
        )
        logger.info(
            "[discovery.bist] %d universe -> %d screened candidates.",
            len(candidates), len(ranked),
        )
        return ranked
    except Exception as exc:
        logger.warning("[discovery.bist] FAIL-SOFT: %s", exc)
        return []


def _fetch_borsapy_movers(universe_cap: int) -> List[DiscoveryCandidate]:
    """
    Probe the installed borsapy for any movers/listing method. Defensive: the
    API surface differs across versions, so we try several names and parse
    whatever list-of-dicts / DataFrame we get. Fail-soft -> [].
    """
    try:
        import borsapy  # noqa: F401
    except ImportError:
        logger.debug("[discovery.bist] borsapy not installed.")
        return []

    client = None
    try:
        from borsapy import Client as BorsapyClient
        client = BorsapyClient()
    except Exception:
        client = None

    for owner in (client, _safe_import_borsapy()):
        if owner is None:
            continue
        for name in _PROBE_LIST_METHODS:
            fn = getattr(owner, name, None)
            if not callable(fn):
                continue
            try:
                raw = fn()
            except Exception:
                continue
            cands = _coerce_listing(raw, universe_cap)
            if cands:
                logger.info("[discovery.bist] borsapy.%s -> %d rows.", name, len(cands))
                return cands
    return []


def _safe_import_borsapy():
    try:
        import borsapy
        return borsapy
    except Exception:
        return None


def _coerce_listing(raw, universe_cap: int) -> List[DiscoveryCandidate]:
    """Turn a borsapy listing result (DataFrame or list-of-dict) into candidates."""
    rows: List[dict] = []
    try:
        # pandas DataFrame
        if hasattr(raw, "to_dict") and hasattr(raw, "columns"):
            df = raw
            if hasattr(df, "reset_index"):
                df = df.reset_index()
            rows = df.to_dict("records")
        elif isinstance(raw, (list, tuple)):
            rows = [r for r in raw if isinstance(r, dict)]
        elif isinstance(raw, dict):
            rows = [raw]
    except Exception:
        return []

    out: List[DiscoveryCandidate] = []
    for r in rows:
        sym = (
            r.get("symbol") or r.get("code") or r.get("ticker")
            or r.get("Symbol") or r.get("Code")
        )
        if not sym:
            continue
        pct = _f(r.get("change_pct") or r.get("changePercent") or r.get("change"))
        price = _f(r.get("price") or r.get("last") or r.get("close"))
        vol = _f(r.get("volume") or r.get("amount") or r.get("turnover"))
        out.append(DiscoveryCandidate(
            symbol=_is_suffix(str(sym)),
            market="bist",
            source="borsapy",
            price=price,
            change_pct_24h=pct,
            quote_volume=vol,
        ))
    if universe_cap > 0:
        out = out[:universe_cap]
    return out


async def _fetch_scrape(url: str, universe_cap: int) -> List[DiscoveryCandidate]:
    """
    Optional operator-configured JSON movers source. Expects a JSON array of
    objects with symbol/change/price/volume-ish keys. Fail-soft -> [].
    """
    try:
        import aiohttp
    except ImportError:
        return []
    try:
        timeout = aiohttp.ClientTimeout(total=10.0)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(url) as resp:
                if resp.status != 200:
                    return []
                data = await resp.json(content_type=None)
        if isinstance(data, dict):
            data = data.get("data") or data.get("rows") or []
        return _coerce_listing(data, universe_cap)
    except Exception as exc:
        logger.debug("[discovery.bist] scrape failed (soft): %s", exc)
        return []


def _is_suffix(sym: str) -> str:
    s = sym.strip().upper()
    if not s:
        return s
    if s.endswith(".IS"):
        return s
    # bare BIST tickers -> add .IS for yfinance/borsapy consistency
    return f"{s}.IS"


def _f(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Guarded self-test (no network) — coerce a mock listing + screen.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from modules.advisor.core.discovery.base import build_exclude_set

    raw = [
        {"symbol": "THYAO", "change_pct": 8.0, "price": 300.0, "volume": 1e7},
        {"symbol": "EREGL.IS", "change_pct": -5.0, "price": 40.0, "volume": 5e6},
        {"code": "FLAT", "change": 0.5, "last": 10.0, "volume": 1e6},  # below min change
        {"nope": 1},  # no symbol -> skipped
    ]
    cands = _coerce_listing(raw, 100)
    assert len(cands) == 3, f"coerce count {len(cands)}"
    assert any(c.symbol == "THYAO.IS" for c in cands), "suffix not added"
    assert any(c.symbol == "EREGL.IS" for c in cands), "existing suffix mangled"

    exclude = build_exclude_set(watchlist=["THYAO.IS"])
    out = screen_dedupe_rank(
        cands, exclude, min_abs_change_pct=2.0, exclude_stablecoins=False, top_n=5
    )
    syms = [c.symbol for c in out]
    assert "THYAO.IS" not in syms, "watchlist not excluded"
    assert "FLAT.IS" not in syms, "low-move not screened"
    assert "EREGL.IS" in syms
    print("bist_discovery._selftest OK:", syms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
