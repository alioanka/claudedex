"""
crypto_discovery — free crypto candidate universe (no API key).

Sources (all free, public, rate-limited, fail-soft):
  1. ccxt fetch_tickers() on the configured advisor_crypto_exchange — 24h
     %-movers + quote-volume. This is the PRIMARY source: it is already a
     dependency, needs no key, and returns ccxt-native pairs that the existing
     CryptoAnalyzer consumes directly.
  2. CoinGecko /search/trending — community-trending coin ids (no key). Used
     only to BIAS/annotate; symbols are mapped onto exchange pairs via ccxt
     markets so we never hand the analyzer a coin the exchange doesn't list.

Screening (local, free): min 24h quote-volume USD, optional market-cap band,
exclude stablecoins/wrapped pegs, exclude watchlist / open sims / blocklist.
Rank: transparent momentum (abs 24h %) + volume + relative-volume score.

Returns ccxt-format pairs (e.g. "PEPE/USDT") for the analyzer pipeline.

FAIL-SOFT: any source error -> []. NEVER raises into the advice cycle.

CAVEAT: this surfaces TRENDING / high-volume names. Trending != good. The
analyzer still produces the actual directional advice; discovery only chooses
WHICH new symbols get analyzed.
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

logger = logging.getLogger("advisor.discovery.crypto")

# Quote currencies we accept for a discovered pair (USD-settled liquidity).
_ACCEPT_QUOTES = ("USDT", "USDC", "USD", "FDUSD", "BUSD")

# CoinGecko trending endpoint (no key). Used as a soft bias only.
_COINGECKO_TRENDING = "https://api.coingecko.com/api/v3/search/trending"
_HTTP_TIMEOUT_S = 8.0


async def discover(
    config: dict,
    exclude: Set[str],
    top_n: int,
) -> List[DiscoveryCandidate]:
    """
    Return up to top_n NEW crypto candidates as DiscoveryCandidate objects.

    Parameters
    ----------
    config  : advisor_config dict.
    exclude : pre-built normalised exclusion set (watchlist + open sims + blocklist).
    top_n   : max candidates to return.
    """
    try:
        exchange_id = config.get("advisor_crypto_exchange", "binance")
        min_qv = cfg_float(config, "advisor_discovery_crypto_min_quote_vol_usd", 1_000_000.0)
        min_mc = cfg_float(config, "advisor_discovery_crypto_min_market_cap_usd", 0.0)
        max_mc = cfg_float(config, "advisor_discovery_crypto_max_market_cap_usd", 0.0)
        min_abs_chg = cfg_float(config, "advisor_discovery_crypto_min_abs_change_pct", 3.0)
        universe_cap = cfg_int(config, "advisor_discovery_crypto_universe_cap", 400)

        loop = asyncio.get_event_loop()
        candidates = await loop.run_in_executor(
            None, _fetch_ccxt_movers, exchange_id, universe_cap
        )
        if not candidates:
            logger.info("[discovery.crypto] ccxt returned no movers (fail-soft).")
            return []

        # Soft trending bias (best-effort, never blocks).
        trending = await _fetch_trending_symbols()
        if trending:
            for c in candidates:
                base = c.symbol.split("/")[0].upper()
                if base in trending:
                    c.extra["coingecko_trending"] = True
                    # small bump via rel_volume proxy if unset
                    if c.rel_volume is None:
                        c.rel_volume = 1.5

        ranked = screen_dedupe_rank(
            candidates,
            exclude,
            min_quote_volume=min_qv,
            min_market_cap=min_mc,
            max_market_cap=(max_mc if max_mc > 0 else None),
            min_abs_change_pct=min_abs_chg,
            exclude_stablecoins=True,
            top_n=top_n,
        )
        logger.info(
            "[discovery.crypto] %d universe -> %d screened candidates (exchange=%s).",
            len(candidates), len(ranked), exchange_id,
        )
        return ranked
    except Exception as exc:
        logger.warning("[discovery.crypto] FAIL-SOFT: %s", exc)
        return []


# ---------------------------------------------------------------------------
# Source fetchers (sync — run in executor)
# ---------------------------------------------------------------------------

def _fetch_ccxt_movers(exchange_id: str, universe_cap: int) -> List[DiscoveryCandidate]:
    """
    Pull fetch_tickers() from the public exchange REST and build candidates.

    No API key required. Only USD-settled spot pairs are kept. Percentage move
    and quote volume come straight from the ticker payload when present.
    """
    try:
        import ccxt
    except ImportError:
        logger.warning("[discovery.crypto] ccxt not installed.")
        return []

    try:
        exchange_cls = getattr(ccxt, exchange_id)
    except AttributeError:
        logger.warning(
            "[discovery.crypto] unknown exchange '%s', using binance.", exchange_id
        )
        exchange_cls = ccxt.binance

    exchange = exchange_cls({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    out: List[DiscoveryCandidate] = []
    try:
        tickers = exchange.fetch_tickers()
    except Exception as exc:
        logger.warning("[discovery.crypto] fetch_tickers failed: %s", exc)
        try:
            exchange.close()
        except Exception:
            pass
        return []

    try:
        for sym, t in tickers.items():
            if not sym or "/" not in sym:
                continue
            base, _, quote = sym.partition("/")
            # strip any ccxt settlement suffix (e.g. ':USDT' on swaps)
            quote = quote.split(":")[0].upper()
            if quote not in _ACCEPT_QUOTES:
                continue

            pct = t.get("percentage")
            last = t.get("last") or t.get("close")
            qv = t.get("quoteVolume")
            bv = t.get("baseVolume")
            if qv is None and bv is not None and last:
                try:
                    qv = float(bv) * float(last)
                except (TypeError, ValueError):
                    qv = None

            out.append(DiscoveryCandidate(
                symbol=f"{base}/{quote}",
                market="crypto",
                source=f"ccxt:{exchange_id}",
                price=_f(last),
                change_pct_24h=_f(pct),
                quote_volume=_f(qv),
            ))
    finally:
        try:
            exchange.close()
        except Exception:
            pass

    # Pre-trim the universe by volume so we never rank tens of thousands of rows.
    out.sort(key=lambda c: (c.quote_volume or 0.0), reverse=True)
    if universe_cap > 0:
        out = out[:universe_cap]
    return out


async def _fetch_trending_symbols() -> Set[str]:
    """
    Fetch CoinGecko /search/trending coin symbols (no key). Best-effort bias.
    Returns an UPPERCASE set of base symbols, or empty on any failure.
    """
    try:
        import aiohttp
    except ImportError:
        return set()
    try:
        timeout = aiohttp.ClientTimeout(total=_HTTP_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.get(_COINGECKO_TRENDING) as resp:
                if resp.status != 200:
                    return set()
                data = await resp.json()
        out: Set[str] = set()
        for coin in (data or {}).get("coins", []):
            item = coin.get("item", {}) if isinstance(coin, dict) else {}
            sym = item.get("symbol")
            if sym:
                out.add(str(sym).upper())
        return out
    except Exception as exc:
        logger.debug("[discovery.crypto] trending fetch failed (soft): %s", exc)
        return set()


def _f(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Guarded self-test (no network) — exercises candidate-building from a mock
# ticker payload via the same screen/rank path.
# ---------------------------------------------------------------------------

def _selftest() -> None:
    from modules.advisor.core.discovery.base import build_exclude_set

    mock = [
        DiscoveryCandidate("PEPE/USDT", "crypto", change_pct_24h=22.0, quote_volume=5e8),
        DiscoveryCandidate("USDC/USDT", "crypto", change_pct_24h=0.0, quote_volume=9e9),
        DiscoveryCandidate("BTC/USDT", "crypto", change_pct_24h=1.0, quote_volume=8e9),
        DiscoveryCandidate("WIF/USDT", "crypto", change_pct_24h=-9.0, quote_volume=1e8),
        DiscoveryCandidate("FLAT/USDT", "crypto", change_pct_24h=0.5, quote_volume=2e8),  # below min_abs_change
    ]
    exclude = build_exclude_set(watchlist=["BTC/USDT"])
    out = screen_dedupe_rank(
        mock, exclude,
        min_quote_volume=1e6, min_abs_change_pct=3.0,
        exclude_stablecoins=True, top_n=5,
    )
    syms = [c.symbol for c in out]
    assert "USDC/USDT" not in syms
    assert "BTC/USDT" not in syms
    assert "FLAT/USDT" not in syms
    assert "PEPE/USDT" in syms and "WIF/USDT" in syms
    print("crypto_discovery._selftest OK:", syms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
