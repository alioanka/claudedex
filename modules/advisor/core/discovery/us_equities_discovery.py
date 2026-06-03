"""
us_equities_discovery — free US-equity movers (no paid key).

Source: Yahoo Finance public "predefined screener" JSON endpoint
(`/v1/finance/screener/predefined/saved?scrIds=...`). Free, no key. The same
endpoint backs yfinance's screener helpers. Screener ids used:
  - day_gainers          : biggest % up moves today
  - most_actives         : highest share volume today
  - undervalued_growth_stocks : growth names trading cheap

Screening (local, free): min average dollar-volume, price floor (avoid sub-$1
penny junk), exclude watchlist / open sims / blocklist. Rank by abs %move +
relative volume.

Returns yfinance-format tickers (e.g. "NVDA").

FAIL-SOFT: any source error -> []. NEVER raises into the advice cycle.

CAVEAT: Yahoo's screener endpoint is undocumented and rate-limited; it can
return 401/429 or change shape without notice. On any failure this discoverer
yields [] and the normal watchlist advice is unaffected. Trending != good.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Set

from modules.advisor.core.discovery.base import (
    DiscoveryCandidate,
    cfg_float,
    cfg_int,
    screen_dedupe_rank,
)

logger = logging.getLogger("advisor.discovery.us_equities")

_SCREENER_URL = (
    "https://query1.finance.yahoo.com/v1/finance/screener/predefined/saved"
)
_DEFAULT_SCREENS = "day_gainers,most_actives,undervalued_growth_stocks"
_HTTP_TIMEOUT_S = 10.0
# Yahoo blocks default python-requests UA; a browser-ish UA is required.
_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)


async def discover(
    config: dict,
    exclude: Set[str],
    top_n: int,
) -> List[DiscoveryCandidate]:
    """Return up to top_n NEW US-equity candidates as DiscoveryCandidate objects."""
    try:
        min_price = cfg_float(config, "advisor_discovery_us_min_price", 2.0)
        min_dollar_vol = cfg_float(config, "advisor_discovery_us_min_dollar_vol", 5_000_000.0)
        min_abs_chg = cfg_float(config, "advisor_discovery_us_min_abs_change_pct", 2.0)
        count = cfg_int(config, "advisor_discovery_us_screen_count", 50)
        screens = str(
            config.get("advisor_discovery_us_screens", _DEFAULT_SCREENS)
        ).split(",")

        candidates: List[DiscoveryCandidate] = []
        for scr in screens:
            scr = scr.strip()
            if not scr:
                continue
            rows = await _fetch_screen(scr, count)
            candidates.extend(rows)

        if not candidates:
            logger.info("[discovery.us_equities] no candidates (fail-soft).")
            return []

        ranked = screen_dedupe_rank(
            candidates,
            exclude,
            min_quote_volume=min_dollar_vol,
            min_price=min_price,
            min_abs_change_pct=min_abs_chg,
            exclude_stablecoins=False,
            top_n=top_n,
        )
        logger.info(
            "[discovery.us_equities] %d universe -> %d screened candidates.",
            len(candidates), len(ranked),
        )
        return ranked
    except Exception as exc:
        logger.warning("[discovery.us_equities] FAIL-SOFT: %s", exc)
        return []


async def _fetch_screen(scr_id: str, count: int) -> List[DiscoveryCandidate]:
    """Hit one Yahoo predefined screener; return candidates. Fail-soft -> []."""
    try:
        import aiohttp
    except ImportError:
        logger.warning("[discovery.us_equities] aiohttp not installed.")
        return []
    params = {"scrIds": scr_id, "count": str(max(1, min(count, 100)))}
    headers = {"User-Agent": _UA, "Accept": "application/json"}
    try:
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
            async with session.get(_SCREENER_URL, params=params) as resp:
                if resp.status != 200:
                    logger.info(
                        "[discovery.us_equities] screen '%s' HTTP %d (soft).",
                        scr_id, resp.status,
                    )
                    return []
                data = await resp.json()
        return _parse_screen(data, scr_id)
    except Exception as exc:
        logger.debug("[discovery.us_equities] screen '%s' failed (soft): %s", scr_id, exc)
        return []


def _parse_screen(data: dict, scr_id: str) -> List[DiscoveryCandidate]:
    """Extract candidates from a Yahoo screener JSON payload."""
    out: List[DiscoveryCandidate] = []
    try:
        results = (data or {}).get("finance", {}).get("result", [])
        if not results:
            return []
        quotes = results[0].get("quotes", [])
    except Exception:
        return []

    for q in quotes:
        try:
            symbol = q.get("symbol")
            if not symbol:
                continue
            # equities/ETFs only; skip options/futures/crypto symbols
            qtype = (q.get("quoteType") or "").upper()
            if qtype not in ("EQUITY", "ETF", ""):
                continue
            price = _f(q.get("regularMarketPrice"))
            pct = _f(q.get("regularMarketChangePercent"))
            vol = _f(q.get("regularMarketVolume"))
            avg_vol = _f(q.get("averageDailyVolume3Month")) or _f(
                q.get("averageDailyVolume10Day")
            )
            mcap = _f(q.get("marketCap"))
            dollar_vol = (price * vol) if (price and vol) else None
            rel_vol = (vol / avg_vol) if (vol and avg_vol and avg_vol > 0) else None
            out.append(DiscoveryCandidate(
                symbol=str(symbol).upper(),
                market="us_equities",
                source=f"yahoo:{scr_id}",
                price=price,
                change_pct_24h=pct,
                quote_volume=dollar_vol,
                rel_volume=rel_vol,
                market_cap=mcap,
                extra={"name": q.get("shortName") or q.get("longName") or ""},
            ))
        except Exception:
            continue
    return out


def _f(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Guarded self-test (no network) — parse a mock Yahoo payload.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from modules.advisor.core.discovery.base import build_exclude_set

    payload = {
        "finance": {"result": [{"quotes": [
            {"symbol": "AAA", "quoteType": "EQUITY", "regularMarketPrice": 25.0,
             "regularMarketChangePercent": 12.0, "regularMarketVolume": 5_000_000,
             "averageDailyVolume3Month": 1_000_000, "marketCap": 2e9},
            {"symbol": "PENNY", "quoteType": "EQUITY", "regularMarketPrice": 0.4,
             "regularMarketChangePercent": 80.0, "regularMarketVolume": 9_000_000,
             "averageDailyVolume3Month": 1_000_000},
            {"symbol": "NVDA", "quoteType": "EQUITY", "regularMarketPrice": 120.0,
             "regularMarketChangePercent": 3.0, "regularMarketVolume": 50_000_000,
             "averageDailyVolume3Month": 40_000_000},
        ]}]}
    }
    cands = _parse_screen(payload, "day_gainers")
    assert len(cands) == 3
    exclude = build_exclude_set(watchlist=["NVDA"])
    out = screen_dedupe_rank(
        cands, exclude,
        min_quote_volume=5_000_000, min_price=2.0,
        min_abs_change_pct=2.0, exclude_stablecoins=False, top_n=5,
    )
    syms = [c.symbol for c in out]
    assert "NVDA" not in syms, "watchlist not excluded"
    assert "PENNY" not in syms, "sub-$1 penny not screened"
    assert "AAA" in syms
    print("us_equities_discovery._selftest OK:", syms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
