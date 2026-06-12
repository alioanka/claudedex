"""SENTINEL price board — two independent FREE public price sources.

Coinbase Exchange + Kraken public REST (no keys, no cost). Used only for
cross-validation math (depeg, divergence) — never for execution. Every fetch
is fail-soft: a dead source returns {} and the detectors that need both sides
simply do not fire. Parsers are pure and self-tested offline:
`python -m modules.sentinel.core.price_board`.

Deliberately NOT routed through pool_engine: these are exchange REST tickers,
not chain RPC, and sentinel must keep an information path independent of the
infrastructure it is watching.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, Optional

logger = logging.getLogger("sentinel")

# Reference (volatile) assets cross-checked between the two sources.
REFERENCE_ASSETS = ("BTC", "ETH", "SOL")
# Stablecoins checked against the 1.0 peg.
STABLE_ASSETS = ("USDT", "USDC", "DAI")

# Coinbase Exchange products (USDC-USD does not trade on Coinbase — USDC is
# treated as USD there — so USDC is Kraken-only and the depeg detector caps
# its single-source severity at WARN).
COINBASE_PRODUCTS: Dict[str, str] = {
    "BTC": "BTC-USD", "ETH": "ETH-USD", "SOL": "SOL-USD",
    "USDT": "USDT-USD", "DAI": "DAI-USD",
}
KRAKEN_PAIRS: Dict[str, str] = {
    "BTC": "XBTUSD", "ETH": "ETHUSD", "SOL": "SOLUSD",
    "USDT": "USDTZUSD", "USDC": "USDCUSD", "DAI": "DAIUSD",
}

_COINBASE_URL = "https://api.exchange.coinbase.com/products/{product}/ticker"
_KRAKEN_URL = "https://api.kraken.com/0/public/Ticker?pair={pair}"


# ───────────────────────── pure parsers ─────────────────────────

def parse_coinbase_ticker(payload: dict) -> Optional[float]:
    """Coinbase Exchange ticker -> last price, or None."""
    try:
        price = float(payload["price"])
        return price if price > 0 else None
    except (KeyError, TypeError, ValueError):
        return None


def parse_kraken_ticker(payload: dict) -> Optional[float]:
    """Kraken public Ticker -> last-trade price ('c'[0]) of the first result
    pair (Kraken renames pairs in the response, e.g. XBTUSD -> XXBTZUSD), or
    None. A non-empty 'error' list is a non-result."""
    try:
        if payload.get("error"):
            return None
        result = payload["result"]
        first = next(iter(result.values()))
        price = float(first["c"][0])
        return price if price > 0 else None
    except (KeyError, TypeError, ValueError, StopIteration):
        return None


# ───────────────────────── fail-soft fetch ─────────────────────────

async def _get_json(session, url: str, timeout_s: float) -> Optional[dict]:
    try:
        import aiohttp
        async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=timeout_s),
                headers={"User-Agent": "claudedex-sentinel/1.0"}) as resp:
            if resp.status != 200:
                return None
            return await resp.json(content_type=None)
    except Exception as exc:
        logger.debug("price_board GET %s fail-soft: %s", url, exc)
        return None


async def fetch_price_board(timeout_s: float = 8.0) -> Dict[str, Dict[str, float]]:
    """Fetch all symbols from both sources concurrently.

    Returns {"coinbase": {sym: price}, "kraken": {sym: price}} with missing
    quotes simply absent. Returns {} sources on any transport failure."""
    out: Dict[str, Dict[str, float]] = {"coinbase": {}, "kraken": {}}
    try:
        import aiohttp
    except Exception as exc:
        logger.warning("price_board unavailable (aiohttp import failed): %s", exc)
        return out
    try:
        async with aiohttp.ClientSession() as session:
            cb_syms = list(COINBASE_PRODUCTS)
            kr_syms = list(KRAKEN_PAIRS)
            tasks = [
                _get_json(session,
                          _COINBASE_URL.format(product=COINBASE_PRODUCTS[s]),
                          timeout_s)
                for s in cb_syms
            ] + [
                _get_json(session, _KRAKEN_URL.format(pair=KRAKEN_PAIRS[s]),
                          timeout_s)
                for s in kr_syms
            ]
            payloads = await asyncio.gather(*tasks, return_exceptions=True)
            for sym, payload in zip(cb_syms, payloads[:len(cb_syms)]):
                if isinstance(payload, dict):
                    price = parse_coinbase_ticker(payload)
                    if price is not None:
                        out["coinbase"][sym] = price
            for sym, payload in zip(kr_syms, payloads[len(cb_syms):]):
                if isinstance(payload, dict):
                    price = parse_kraken_ticker(payload)
                    if price is not None:
                        out["kraken"][sym] = price
    except Exception as exc:
        logger.warning("price_board fetch fail-soft: %s", exc)
    return out


# ───────────────────────── self-test (offline) ─────────────────────────

def _self_test() -> None:
    assert parse_coinbase_ticker({"price": "50123.45", "volume": "1"}) == 50123.45
    assert parse_coinbase_ticker({"price": "-1"}) is None
    assert parse_coinbase_ticker({"message": "NotFound"}) is None
    assert parse_coinbase_ticker({}) is None
    assert parse_coinbase_ticker({"price": "abc"}) is None

    kraken_ok = {"error": [], "result": {"XXBTZUSD": {
        "a": ["50001.0", "1", "1"], "b": ["49999.0", "1", "1"],
        "c": ["50000.5", "0.01"]}}}
    assert parse_kraken_ticker(kraken_ok) == 50000.5
    assert parse_kraken_ticker({"error": ["EQuery:Unknown asset pair"]}) is None
    assert parse_kraken_ticker({"error": [], "result": {}}) is None
    assert parse_kraken_ticker({}) is None
    assert parse_kraken_ticker({"error": [], "result": {"P": {"c": ["0"]}}}) is None

    # symbol maps cover exactly the declared assets
    for s in REFERENCE_ASSETS:
        assert s in COINBASE_PRODUCTS and s in KRAKEN_PAIRS
    for s in STABLE_ASSETS:
        assert s in KRAKEN_PAIRS  # coinbase optional (no USDC-USD product)

    print("sentinel price_board self-test: OK")


if __name__ == "__main__":
    _self_test()
