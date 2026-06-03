"""
modules.advisor.core.discovery — the "New Gems" discovery layer.

Surfaces promising tickers BEYOND the operator's watchlists, per market, by
pulling a free candidate universe, screening/deduping/ranking it locally (no
LLM, no paid data), and handing the top-N NEW symbols to the SAME analyzer
pipeline so they receive normal advice — flagged as DISCOVERY (origin marker)
rather than watchlist.

Public API
----------
    discover(market, config, watchlist, exclude) -> list[str]
        Returns up to advisor_discovery_max_per_market NEW native symbols for
        `market`, already screened/deduped/ranked and excluding anything in
        `watchlist` / open sims / blocklist (passed in via `exclude`).

Everything is FAIL-SOFT: an unsupported market or any source error yields [].
ADVICE-ONLY. Default OFF — gated by advisor_discovery_enabled in advice_engine.
"""

from __future__ import annotations

import logging
from typing import Iterable, List, Optional

from modules.advisor.core.discovery.base import (
    DiscoveryCandidate,
    build_exclude_set,
    cfg_int,
)

logger = logging.getLogger("advisor.discovery")

__all__ = ["discover", "discover_candidates", "DiscoveryCandidate", "build_exclude_set"]


async def discover_candidates(
    market: str,
    config: dict,
    watchlist: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    open_sims: Optional[Iterable[str]] = None,
    blocklist: Optional[Iterable[str]] = None,
) -> List[DiscoveryCandidate]:
    """
    Run the per-market discoverer and return ranked DiscoveryCandidate objects
    (richer than discover(), used by the engine to attach score/metrics).

    `exclude` is an optional pre-built set; if given it is used directly.
    Otherwise an exclusion set is built from watchlist + open_sims + blocklist.
    """
    market = (market or "").strip().lower()
    top_n = cfg_int(config, "advisor_discovery_max_per_market", 5)
    if top_n <= 0:
        return []

    if exclude is not None:
        exclude_set = set(s.strip().upper() for s in exclude if s)
        # still fold in crypto bases for robust cross-quote dedupe
        exclude_set |= build_exclude_set(watchlist=exclude_set)
    else:
        exclude_set = build_exclude_set(
            watchlist=watchlist, open_sims=open_sims, blocklist=blocklist
        )

    try:
        if market == "crypto":
            from modules.advisor.core.discovery import crypto_discovery
            return await crypto_discovery.discover(config, exclude_set, top_n)
        if market == "us_equities":
            from modules.advisor.core.discovery import us_equities_discovery
            return await us_equities_discovery.discover(config, exclude_set, top_n)
        if market == "bist":
            from modules.advisor.core.discovery import bist_discovery
            return await bist_discovery.discover(config, exclude_set, top_n)
        logger.debug("[discovery] unsupported market '%s' -> [].", market)
        return []
    except Exception as exc:
        logger.warning("[discovery] discover_candidates(%s) FAIL-SOFT: %s", market, exc)
        return []


async def discover(
    market: str,
    config: dict,
    watchlist: Optional[Iterable[str]] = None,
    exclude: Optional[Iterable[str]] = None,
    open_sims: Optional[Iterable[str]] = None,
    blocklist: Optional[Iterable[str]] = None,
) -> List[str]:
    """
    Return up to advisor_discovery_max_per_market NEW native symbols for market.
    Thin wrapper over discover_candidates() returning just the symbol strings.
    """
    cands = await discover_candidates(
        market, config, watchlist=watchlist, exclude=exclude,
        open_sims=open_sims, blocklist=blocklist,
    )
    return [c.symbol for c in cands]
