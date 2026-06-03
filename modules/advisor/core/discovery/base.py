"""
Discovery base — shared helpers + the DiscoveryCandidate dataclass.

The discovery layer surfaces promising tickers BEYOND the operator's
configured watchlists, per market (crypto / us_equities / bist). Each
market discoverer pulls a CANDIDATE UNIVERSE from free, rate-limited,
fail-soft data sources, screens it locally (no LLM, no paid data), dedupes
against the watchlist / open sims / blocklist, ranks by a transparent
momentum+volume score, and returns the top-N NEW symbols.

ADVICE-ONLY. Nothing here places orders. A source failing must NEVER crash
the advice cycle — every discoverer fails soft to an empty list.

HONESTY: "trending" is NOT the same as "good". This layer only surfaces
candidates for the operator (and the normal analyzer pipeline) to review.
The screen filters obvious junk (illiquid, micro-cap, stablecoins, sub-$1
penny names) but cannot judge fundamentals or detect a pump in progress.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Set

logger = logging.getLogger("advisor.discovery")


# ---------------------------------------------------------------------------
# Candidate model
# ---------------------------------------------------------------------------

@dataclass
class DiscoveryCandidate:
    """
    A single discovered candidate, pre-analyzer.

    symbol        : Native ticker for the market (ccxt pair / yfinance ticker /
                    ".IS" suffix for BIST). This is what gets handed to the
                    existing analyzer.
    market        : Market enum value string ("crypto" | "us_equities" | "bist").
    source        : Which free source produced it (for transparency/debugging).
    price         : Last price in quote currency (USD-ish), if known.
    change_pct_24h: 24h percentage move (e.g. 12.5 for +12.5%), if known.
    quote_volume  : 24h quote-currency volume / dollar volume, if known.
    rel_volume    : Relative volume vs its own average (>1 = unusually active).
    market_cap    : Market cap / fully-diluted-ish value, if known.
    score         : Transparent rank score (filled in by rank()).
    extra         : Free-form metadata (liquidity, name, etc.).
    """

    symbol: str
    market: str
    source: str = ""
    price: Optional[float] = None
    change_pct_24h: Optional[float] = None
    quote_volume: Optional[float] = None
    rel_volume: Optional[float] = None
    market_cap: Optional[float] = None
    score: float = 0.0
    extra: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Common helpers — pure functions, fully unit-testable with no network.
# ---------------------------------------------------------------------------

# Stablecoin / wrapped / pegged bases we never want to "discover" (crypto).
STABLE_BASES: Set[str] = {
    "USDT", "USDC", "BUSD", "DAI", "TUSD", "USDP", "USDD", "FDUSD", "PYUSD",
    "GUSD", "FRAX", "LUSD", "USDE", "EUR", "EURT", "EURS", "USTC", "UST",
    "USD", "BSC-USD", "WBTC", "WETH", "STETH", "WSTETH", "WBETH", "CBETH",
}


def _norm(sym: str) -> str:
    """Normalise a symbol for dedupe comparison (upper, strip whitespace)."""
    if not sym:
        return ""
    return sym.strip().upper()


def _crypto_base(pair: str) -> str:
    """Extract the base asset from a ccxt-style pair, e.g. 'BTC/USDT' -> 'BTC'."""
    if not pair:
        return ""
    p = pair.strip().upper()
    for sep in ("/", "-", ":"):
        if sep in p:
            return p.split(sep)[0]
    return p


def is_stablecoin_pair(pair: str) -> bool:
    """True if the base asset of a crypto pair is a stablecoin / wrapped peg."""
    return _crypto_base(pair) in STABLE_BASES


def build_exclude_set(
    watchlist: Optional[Iterable[str]] = None,
    open_sims: Optional[Iterable[str]] = None,
    blocklist: Optional[Iterable[str]] = None,
) -> Set[str]:
    """
    Build a normalised exclusion set from the watchlist + open sims + blocklist.

    For crypto we also add the bare base asset so 'BTC/USDT' in the watchlist
    excludes a 'BTC/USD' candidate (same underlying), and a blocklisted 'BTC'
    excludes any BTC pair.
    """
    out: Set[str] = set()
    for group in (watchlist, open_sims, blocklist):
        if not group:
            continue
        for s in group:
            n = _norm(s)
            if not n:
                continue
            out.add(n)
            # add bare base for crypto-style pairs so cross-quote dedupe works
            if "/" in n or "-" in n:
                out.add(_crypto_base(n))
    return out


def is_excluded(symbol: str, exclude: Set[str]) -> bool:
    """True if a candidate symbol collides with the exclusion set."""
    n = _norm(symbol)
    if not n:
        return True
    if n in exclude:
        return True
    # crypto: a candidate 'DOGE/USDT' is excluded if bare 'DOGE' is excluded.
    if "/" in n or "-" in n:
        if _crypto_base(n) in exclude:
            return True
    return False


def dedupe(
    candidates: List[DiscoveryCandidate],
    exclude: Set[str],
) -> List[DiscoveryCandidate]:
    """
    Drop candidates that are in the exclusion set or duplicate an earlier
    candidate (by normalised symbol AND by crypto base, first occurrence wins).
    Order is preserved.
    """
    seen: Set[str] = set()
    out: List[DiscoveryCandidate] = []
    for c in candidates:
        n = _norm(c.symbol)
        if not n or is_excluded(c.symbol, exclude):
            continue
        base = _crypto_base(n) if ("/" in n or "-" in n) else n
        if n in seen or base in seen:
            continue
        seen.add(n)
        seen.add(base)
        out.append(c)
    return out


def screen(
    candidates: List[DiscoveryCandidate],
    *,
    min_quote_volume: float = 0.0,
    min_price: float = 0.0,
    min_market_cap: float = 0.0,
    max_market_cap: Optional[float] = None,
    min_abs_change_pct: float = 0.0,
    exclude_stablecoins: bool = True,
) -> List[DiscoveryCandidate]:
    """
    Apply transparent local screening thresholds. A candidate is kept only if
    every populated metric clears its floor/band. Missing metrics are treated
    as "unknown -> do not reject on that axis" (the source may not provide it),
    so screening never silently drops everything when a feed omits a field.
    """
    out: List[DiscoveryCandidate] = []
    for c in candidates:
        if exclude_stablecoins and is_stablecoin_pair(c.symbol):
            continue
        if c.quote_volume is not None and c.quote_volume < min_quote_volume:
            continue
        if c.price is not None and min_price > 0 and c.price < min_price:
            continue
        if c.market_cap is not None:
            if min_market_cap > 0 and c.market_cap < min_market_cap:
                continue
            if max_market_cap is not None and max_market_cap > 0 and c.market_cap > max_market_cap:
                continue
        if (
            min_abs_change_pct > 0
            and c.change_pct_24h is not None
            and abs(c.change_pct_24h) < min_abs_change_pct
        ):
            continue
        out.append(c)
    return out


def _safe_log_volume(vol: Optional[float]) -> float:
    """log10-scaled volume contribution, clamped to [0, 1] over ~$10k..$1B."""
    import math
    if not vol or vol <= 0:
        return 0.0
    # 10k -> 0.0, 1B -> 1.0 (5 decades)
    lo, hi = 4.0, 9.0
    x = (math.log10(vol) - lo) / (hi - lo)
    return max(0.0, min(1.0, x))


def rank(candidates: List[DiscoveryCandidate]) -> List[DiscoveryCandidate]:
    """
    Assign a transparent momentum+volume score and return sorted desc.

    score = 0.55 * momentum + 0.30 * volume + 0.15 * relative_volume
      momentum        : abs(24h %) scaled, capped at 30% -> 1.0
      volume          : log10 dollar volume scaled 10k..1B -> [0,1]
      relative_volume : (rel_vol - 1) clamped to [0, 1] (2x avg -> 1.0)

    Absolute move is used so strong DOWN movers (short candidates) also rank.
    All terms are continuous and free to compute. Ties keep input order.
    """
    for c in candidates:
        mom = 0.0
        if c.change_pct_24h is not None:
            mom = min(abs(c.change_pct_24h) / 30.0, 1.0)
        vol = _safe_log_volume(c.quote_volume)
        relv = 0.0
        if c.rel_volume is not None:
            relv = max(0.0, min(1.0, c.rel_volume - 1.0))
        c.score = round(0.55 * mom + 0.30 * vol + 0.15 * relv, 6)
    return sorted(candidates, key=lambda x: x.score, reverse=True)


def screen_dedupe_rank(
    candidates: List[DiscoveryCandidate],
    exclude: Set[str],
    *,
    min_quote_volume: float = 0.0,
    min_price: float = 0.0,
    min_market_cap: float = 0.0,
    max_market_cap: Optional[float] = None,
    min_abs_change_pct: float = 0.0,
    exclude_stablecoins: bool = True,
    top_n: Optional[int] = None,
) -> List[DiscoveryCandidate]:
    """Full pipeline: screen -> dedupe -> rank -> top_n. Pure, no network."""
    screened = screen(
        candidates,
        min_quote_volume=min_quote_volume,
        min_price=min_price,
        min_market_cap=min_market_cap,
        max_market_cap=max_market_cap,
        min_abs_change_pct=min_abs_change_pct,
        exclude_stablecoins=exclude_stablecoins,
    )
    deduped = dedupe(screened, exclude)
    ranked = rank(deduped)
    if top_n is not None and top_n > 0:
        return ranked[:top_n]
    return ranked


# ---------------------------------------------------------------------------
# Config coercion helpers
# ---------------------------------------------------------------------------

def cfg_float(config: dict, key: str, default: float) -> float:
    try:
        return float(config.get(key, default))
    except (TypeError, ValueError):
        return default


def cfg_int(config: dict, key: str, default: int) -> int:
    try:
        return int(float(config.get(key, default)))
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Guarded self-test (no network) — `python -m modules.advisor.core.discovery.base`
# ---------------------------------------------------------------------------

def _selftest() -> None:
    cands = [
        DiscoveryCandidate("BTC/USDT", "crypto", change_pct_24h=2.0, quote_volume=5e9),
        DiscoveryCandidate("USDC/USDT", "crypto", change_pct_24h=0.1, quote_volume=9e9),
        DiscoveryCandidate("PEPE/USDT", "crypto", change_pct_24h=28.0, quote_volume=3e8, rel_volume=3.0),
        DiscoveryCandidate("DOGE/USDT", "crypto", change_pct_24h=-15.0, quote_volume=2e8, rel_volume=2.0),
        DiscoveryCandidate("TINY/USDT", "crypto", change_pct_24h=40.0, quote_volume=2e3),  # below vol floor
        DiscoveryCandidate("ETH/USD", "crypto", change_pct_24h=5.0, quote_volume=4e9),     # excluded via base
    ]
    exclude = build_exclude_set(watchlist=["BTC/USDT", "ETH/USDT"], blocklist=["SCAM"])

    out = screen_dedupe_rank(
        cands, exclude,
        min_quote_volume=1e5, exclude_stablecoins=True, top_n=5,
    )
    syms = [c.symbol for c in out]
    assert "USDC/USDT" not in syms, "stablecoin not screened out"
    assert "BTC/USDT" not in syms, "watchlist not excluded"
    assert "ETH/USD" not in syms, "cross-quote base not excluded"
    assert "TINY/USDT" not in syms, "low-volume not screened"
    assert "PEPE/USDT" in syms and "DOGE/USDT" in syms, "movers dropped"
    assert syms.index("PEPE/USDT") < syms.index("DOGE/USDT"), "rank order wrong"

    dups = [
        DiscoveryCandidate("AAA/USDT", "crypto", change_pct_24h=10, quote_volume=1e6),
        DiscoveryCandidate("AAA/USDT", "crypto", change_pct_24h=11, quote_volume=2e6),
    ]
    assert len(dedupe(dups, set())) == 1, "dedupe failed"

    unknown = [DiscoveryCandidate("XXX", "us_equities")]
    assert len(screen(unknown, min_quote_volume=1e6, min_price=1.0)) == 1, \
        "unknown-metric candidate wrongly screened"

    print("base._selftest OK:", syms)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
