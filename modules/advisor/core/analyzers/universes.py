"""
universes.py — maintained ticker universes for BIST + FX/metals scanning.

The operator wants the WHOLE BIST (BIST-30/50/100) and a broad FX/metals set
scanned every cycle, not just the 4 watchlist tickers. Analysis (signals +
horizon levels) is FREE local math — only the LLM rationale costs, and that is
already capped by the daily LLM budget + min_confidence gate — so scanning a
large universe is cost-safe.

This module provides:
  - Hardcoded, maintained BIST-50 and BIST-30 constituent lists (bare tickers;
    the BIST analyzer normalises .IS for yfinance / strips it for borsapy).
  - A broad FX major/cross + metals set (yfinance "=X" / "=F" format).
  - `expand_universe(market, config, watchlist)` — merge the configured universe
    (off / bist30 / bist50 / bist100-if-provided) with the watchlist, dedupe,
    and cap. Watchlist tickers are ALWAYS included.

Config keys (advisor_config, migrations 078 + 083):
  advisor_bist_universe  : 'auto' (DEFAULT since 083 — Fonoloji live list when a
                           key is present, else watchlist-only) | '' (= auto) |
                           'watchlist' | 'bist30' | 'bist50' | 'fonoloji' | 'custom'
                           Resolution is delegated to
                           activation.effective_universe_mode (auto-prefer rule;
                           explicit presets always win).
  advisor_fx_universe    : '' | 'watchlist' | 'majors' | 'extended'
                           (default 'watchlist')
  advisor_bist_universe_custom : comma-sep extra BIST tickers (for 'custom' or to
                                 extend any preset)
  advisor_fx_universe_custom   : comma-sep extra FX/metal symbols
  advisor_universe_max   : hard cap on total symbols per market (default 60)

HONESTY: these constituent lists are a point-in-time snapshot (2026-06). BIST
index membership changes ~quarterly; the operator can override/extend via the
*_custom keys without a code change. ADVICE-ONLY.

Self-test: python -m modules.advisor.core.analyzers.universes
"""

from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger("advisor.universes")

# A well-formed BIST equity ticker: letter-led, 3-6 chars, uppercase
# letters/digits only (THYAO, SASA, GARAN, A1CAP). This deliberately rejects the
# non-equity rows Fonoloji's /stocks/list returns — option codes (030E0626P1600,
# digit-led + long), ISINs (0K0060615755), Bloomberg codes (1211 HK EQUITY),
# and fund names (100 TL PORSFOY) — which all start with a digit or contain a
# space/symbol. (Fonoloji support flagged ~70% of chart calls were 404 from these.)
_BIST_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9]{2,5}$")


def _valid_bist_ticker(sym: str) -> bool:
    return bool(_BIST_TICKER_RE.match((sym or "").strip().upper()))


# ---------------------------------------------------------------------------
# Midas-US row rejection (Wave-F5 fix 6). Fonoloji's /stocks/list also carries
# the Midas-tradable US symbols with an exchange suffix glued on (AAPLUS/
# ADBEUS/AAOIUS = <US base>+US; AAPLO = AAPL+O; ABBVN = ABBV+N). Those pass
# _BIST_TICKER_RE, flood the 60-slot universe alphabetically (53/60 garbage
# observed) and fail all three BIST price sources every cycle. Preference
# order: a row-level market/exchange field is authoritative when present;
# otherwise reject suffix-decorated known US bases (and the bare bases
# themselves). BIST_50 constituents are always protected (PGSUS ends in 'US'
# and must survive).
# ---------------------------------------------------------------------------
_ROW_MARKET_KEYS = ("market", "exchange", "borsa", "market_code",
                    "exchange_code", "market_type", "venue")
_BIST_MARKET_FRAGS = ("BIST", "XIST", "BORSA", "IST")
_US_MARKET_FRAGS = ("NASDAQ", "NYSE", "US", "AMEX", "ARCA", "MIDAS")
# Known Midas-tradable US bases (mega-caps + the common Midas menu). Snapshot;
# used ONLY for the fallback suffix rule when the row has no market field.
_US_BASES = frozenset({
    "AAPL", "AAOI", "ABBV", "ABNB", "ADBE", "AMD", "AMZN", "AVGO", "BA",
    "BABA", "BAC", "C", "CCL", "COIN", "CRM", "CSCO", "CVX", "DAL", "DIS",
    "F", "GE", "GM", "GOOG", "GOOGL", "HOOD", "IBM", "INTC", "JNJ", "JPM",
    "KO", "LCID", "MA", "MCD", "META", "MRK", "MSFT", "MU", "NFLX", "NKE",
    "NVDA", "ORCL", "PEP", "PFE", "PLTR", "PYPL", "QCOM", "RIVN", "SBUX",
    "SHOP", "SNAP", "SOFI", "SQ", "T", "TSLA", "TXN", "UBER", "V", "VZ",
    "WMT", "XOM",
})
_US_SUFFIXES = ("US", "O", "N")


def _is_midas_us_row(sym: str, row=None) -> bool:
    """True when a /stocks/list row is a Midas-US symbol, not a BIST equity."""
    s = (sym or "").strip().upper()
    if not s:
        return False
    # Positive rule first: an explicit market/exchange field is authoritative.
    if isinstance(row, dict):
        for key in _ROW_MARKET_KEYS:
            val = row.get(key) or row.get(key.capitalize())
            if val is None or str(val).strip() == "":
                continue
            v = str(val).strip().upper()
            if any(frag in v for frag in _BIST_MARKET_FRAGS):
                return False
            if any(frag in v for frag in _US_MARKET_FRAGS):
                return True
            break  # unknown venue value -> fall through to the suffix rule
    # Known BIST constituents are always protected (PGSUS ends in 'US').
    if s in _BIST50_SET:
        return False
    # Bare US base (AAPL) or suffix-decorated US base (AAPLUS/AAPLO/ABBVN).
    if s in _US_BASES:
        return True
    for suf in _US_SUFFIXES:
        if s.endswith(suf) and s[: -len(suf)] in _US_BASES:
            return True
    return False

# ---------------------------------------------------------------------------
# BIST constituent lists — bare tickers (no .IS). Snapshot 2026-06; maintained.
# BIST-30 is the most-liquid blue-chip subset; BIST-50 adds the next tier.
# ---------------------------------------------------------------------------
BIST_30: List[str] = [
    "AKBNK", "ALARK", "ARCLK", "ASELS", "ASTOR", "BIMAS", "BRSAN", "EKGYO",
    "ENKAI", "EREGL", "FROTO", "GARAN", "GUBRF", "HEKTS", "ISCTR", "KCHOL",
    "KONTR", "KOZAL", "KRDMD", "ODAS", "OYAKC", "PETKM", "PGSUS", "SAHOL",
    "SASA", "SISE", "TCELL", "THYAO", "TOASO", "TUPRS",
]

# BIST-50 = BIST-30 + next 20 liquid names (snapshot 2026-06).
_BIST_50_EXTRA: List[str] = [
    "AEFES", "AGHOL", "AKSEN", "ALFAS", "BERA", "CIMSA", "DOHOL", "ENJSA",
    "GESAN", "HALKB", "ISMEN", "KOZAA", "MGROS", "MIATK", "SMRTG", "TAVHL",
    "TSKB", "TTKOM", "VAKBN", "YKBNK",
]

BIST_50: List[str] = BIST_30 + _BIST_50_EXTRA

# Set form for the Midas-US filter (protects PGSUS & co. from the *US rule).
_BIST50_SET = frozenset(BIST_50)

# ---------------------------------------------------------------------------
# FX + metals universes (yfinance format: "EURUSD=X", metals via futures "GC=F").
# ---------------------------------------------------------------------------
FX_MAJORS: List[str] = [
    "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", "AUDUSD=X",
    "USDCAD=X", "NZDUSD=X",
    "GC=F", "SI=F",          # gold / silver futures
]

# Extended: majors + popular crosses + TRY pairs + platinum/copper.
_FX_EXTENDED_EXTRA: List[str] = [
    "EURGBP=X", "EURJPY=X", "GBPJPY=X", "AUDJPY=X", "EURCHF=X",
    "USDTRY=X", "EURTRY=X",
    "PL=F", "HG=F",          # platinum / copper futures
    "XAUUSD=X", "XAGUSD=X",  # spot gold/silver (yfinance proxies)
]

FX_EXTENDED: List[str] = FX_MAJORS + _FX_EXTENDED_EXTRA


def _split_csv(raw) -> List[str]:
    return [s.strip() for s in str(raw or "").split(",") if s.strip()]


def _dedupe_preserve(items: List[str]) -> List[str]:
    seen = set()
    out: List[str] = []
    for it in items:
        k = it.strip().upper()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(it.strip())
    return out


def _fonoloji_bist_list(config: dict) -> List[str]:
    """
    Pull the LIVE BIST ticker list from Fonoloji (GET /stocks/list) when a key
    is present. Returns bare tickers (no .IS). Fail-soft -> [] (caller falls
    back to the hardcoded BIST-50 snapshot). The shared client TTL-caches this
    (one call per refresh), so it is rate-limit cheap.
    """
    try:
        from modules.advisor.core.data.fonoloji_client import (
            FonolojiClient, resolve_api_key,
        )
    except Exception:
        return []
    if not resolve_api_key(config):
        return []
    try:
        client = FonolojiClient(config)
        payload = client.stock_list()
    except Exception as exc:
        logger.debug("[universes] Fonoloji /stocks/list failed (soft): %s", exc)
        return []

    rows = payload
    if isinstance(payload, dict):
        for k in ("stocks", "data", "result", "results", "rows", "items", "list"):
            v = payload.get(k)
            if isinstance(v, list) and v:
                rows = v
                break
    if not isinstance(rows, list):
        return []

    out: List[tuple] = []  # (symbol, source_row_or_None)
    for r in rows:
        if isinstance(r, str) and r.strip():
            out.append((r.strip().upper().removesuffix(".IS"), None))
            continue
        if not isinstance(r, dict):
            continue
        sym = (r.get("ticker") or r.get("code") or r.get("symbol")
               or r.get("Ticker") or r.get("Code"))
        if sym:
            s = str(sym).strip().upper()
            if s.endswith(".IS"):
                s = s[:-3]
            if s:
                out.append((s, r))
    # Fonoloji's /stocks/list returns the WHOLE TEFAS/BIST universe — funds,
    # warrants, option codes (030E0626P1600), ISINs (0K0060615755), and broken
    # fund names (100 TL PORSFOY) — which 404 the per-stock chart endpoint and
    # flood the BIST analyzer (~70% of calls per Fonoloji's own advice). Keep
    # only well-formed equity tickers: a letter-led short alnum code (THYAO,
    # A1CAP). Everything garbage starts with a digit or has spaces/symbols.
    # Wave-F5 fix 6: the list ALSO carries Midas-tradable US symbols
    # (AAPLO/AAPLUS/ADBEUS/...) that pass the shape check — drop those too
    # (row market/exchange field when present, else US-base suffix rule).
    cleaned = [s for s, r in out
               if _valid_bist_ticker(s) and not _is_midas_us_row(s, r)]
    n_shape = sum(1 for s, _ in out if not _valid_bist_ticker(s))
    n_us = len(out) - n_shape - len(cleaned)
    if len(cleaned) != len(out):
        logger.info("[universes] Fonoloji list: kept %d valid BIST tickers, "
                    "dropped %d non-equity rows (funds/options/ISINs) + %d "
                    "Midas-US rows.", len(cleaned), n_shape, n_us)
    return cleaned


def expand_bist_universe(config: dict, watchlist: List[str]) -> List[str]:
    """
    Build the BIST scan universe: preset + custom + watchlist (always included).
    Returns bare tickers; the BIST analyzer handles .IS normalisation.
    Default ('watchlist'/'') => returns the watchlist unchanged.

    Mode resolution is delegated to activation.effective_universe_mode:
    'auto' (the migration-083 default, also '') resolves to the Fonoloji live
    list when a key is present + auto-prefer on, else watchlist-only; explicit
    presets ('watchlist'/'bist30'/'bist50'/'custom') always win; 'fonoloji'/
    'bist_all'/'all' pull the LIVE BIST list (fail-soft to the hardcoded
    BIST-50 snapshot). The universe cap still applies.
    """
    try:
        from modules.advisor.core.data.activation import effective_universe_mode
        mode = effective_universe_mode(config)
    except Exception:  # fail-soft: pre-083 behaviour
        mode = str(config.get("advisor_bist_universe", "watchlist")
                   or "watchlist").lower()
    custom = _split_csv(config.get("advisor_bist_universe_custom", ""))

    preset: List[str] = []
    if mode in ("bist30", "bist_30", "30"):
        preset = list(BIST_30)
    elif mode in ("bist50", "bist_50", "50"):
        preset = list(BIST_50)
    elif mode == "snapshot":
        # Fonoloji explicitly requested but no key — fail-soft snapshot.
        logger.info("[universes] Fonoloji universe requested without a key; "
                    "using hardcoded BIST-50 snapshot.")
        preset = list(BIST_50)
    elif mode in ("fonoloji", "bist_all", "bistall", "all", "live"):
        live = _fonoloji_bist_list(config)
        if live:
            logger.info("[universes] BIST universe from Fonoloji /stocks/list: "
                        "%d tickers.", len(live))
            preset = live
        else:
            logger.info("[universes] Fonoloji BIST list unavailable; falling "
                        "back to hardcoded BIST-50 snapshot.")
            preset = list(BIST_50)
    elif mode in ("custom",):
        preset = []  # custom-only; rely on the custom list
    # 'watchlist' / unknown -> no preset (watchlist-only behaviour)

    # Strip any .IS the operator added so the universe is consistent (analyzer
    # re-adds .IS for yfinance). Watchlist is appended verbatim + always kept.
    merged = preset + custom + list(watchlist or [])
    out = _dedupe_preserve(merged)
    cap = _max(config)
    if len(out) > cap:
        # Keep watchlist tickers, then fill the remaining slots with a DAILY
        # ROTATION through the rest (Wave-F5 fix 6). The old alphabetical
        # head-fill meant a large live list only ever scanned the A-names;
        # rotating by UTC day-of-year covers the whole universe over time
        # while staying deterministic within a day (cache-friendly).
        wl = [w for w in (watchlist or [])]
        rest = [s for s in out if s not in wl]
        if rest:
            from datetime import datetime, timezone
            slots = max(1, cap - len(_dedupe_preserve(wl)))
            offset = (datetime.now(timezone.utc).timetuple().tm_yday
                      * slots) % len(rest)
            rest = rest[offset:] + rest[:offset]
        out = _dedupe_preserve(wl + rest)[:cap]
    return out


def expand_fx_universe(config: dict, watchlist: List[str]) -> List[str]:
    """Build the FX/metals scan universe: preset + custom + watchlist."""
    mode = str(config.get("advisor_fx_universe", "watchlist") or "watchlist").lower()
    custom = _split_csv(config.get("advisor_fx_universe_custom", ""))

    preset: List[str] = []
    if mode in ("majors", "major"):
        preset = list(FX_MAJORS)
    elif mode in ("extended", "all"):
        preset = list(FX_EXTENDED)
    elif mode in ("custom",):
        preset = []

    merged = preset + custom + list(watchlist or [])
    out = _dedupe_preserve(merged)
    cap = _max(config)
    if len(out) > cap:
        wl = [w for w in (watchlist or [])]
        rest = [s for s in out if s not in wl]
        out = _dedupe_preserve(wl + rest)[:cap]
    return out


def _max(config: dict) -> int:
    try:
        return max(1, int(float(config.get("advisor_universe_max", 60))))
    except (TypeError, ValueError):
        return 60


# ---------------------------------------------------------------------------
# Self-test (no network) — python -m modules.advisor.core.analyzers.universes
# ---------------------------------------------------------------------------
def _selftest() -> None:
    # Default mode = watchlist-only (unchanged behaviour).
    wl = ["THYAO", "GARAN"]
    assert expand_bist_universe({}, wl) == wl, "default must be watchlist-only"
    assert expand_fx_universe({}, ["EURUSD=X"]) == ["EURUSD=X"]

    # bist50 preset includes BIST-30 names + extras, and always keeps watchlist.
    u = expand_bist_universe({"advisor_bist_universe": "bist50"}, ["XYZ"])
    assert "THYAO" in u and "VAKBN" in u and "XYZ" in u
    assert len(u) == len(_dedupe_preserve(u)), "dedupe failed"

    # Custom extends a preset.
    u2 = expand_bist_universe(
        {"advisor_bist_universe": "bist30",
         "advisor_bist_universe_custom": "FOOBAR,THYAO"}, [])
    assert "FOOBAR" in u2 and u2.count("THYAO") == 1

    # Cap is enforced but watchlist survives.
    u3 = expand_bist_universe(
        {"advisor_bist_universe": "bist50", "advisor_universe_max": "5"},
        ["MYWATCH"])
    assert len(u3) == 5 and "MYWATCH" in u3, (len(u3), u3)

    # FX extended.
    fx = expand_fx_universe({"advisor_fx_universe": "extended"}, [])
    assert "EURUSD=X" in fx and "USDTRY=X" in fx and "GC=F" in fx

    # Midas-US row rejection (Wave-F5 fix 6).
    for bad in ("AAPLUS", "AAPLO", "ABBVN", "ADBEUS", "AAOIUS", "AAPL"):
        assert _is_midas_us_row(bad), f"{bad} must be rejected as Midas-US"
    for good in ("THYAO", "PGSUS", "GARAN", "A1CAP"):
        assert not _is_midas_us_row(good), f"{good} must survive the US filter"
    # Row market/exchange field is authoritative in both directions.
    assert _is_midas_us_row("XYZAB", {"market": "NASDAQ"}), "market field US"
    assert not _is_midas_us_row("AAPLUS", {"market": "BIST"}), "market field BIST"

    # Fonoloji mode with NO key => fail-soft to the BIST-50 snapshot.
    flive = expand_bist_universe({"advisor_bist_universe": "fonoloji"}, ["XYZ"])
    assert "THYAO" in flive and "XYZ" in flive, "fonoloji no-key must fall back to BIST-50"

    # AUTO mode (default '' / 'auto') + key => Fonoloji live list (stubbed —
    # no network) feeds the universe; explicit 'watchlist' still opts out.
    global _fonoloji_bist_list
    _orig = _fonoloji_bist_list
    _fonoloji_bist_list = lambda cfg: ["AAA", "BBB"]  # noqa: E731
    try:
        ulive = expand_bist_universe(
            {"advisor_fonoloji_api_key": "K"}, ["XYZ"])
        assert "AAA" in ulive and "XYZ" in ulive, \
            "auto+key must use the Fonoloji live list"
        uwl = expand_bist_universe(
            {"advisor_fonoloji_api_key": "K",
             "advisor_bist_universe": "watchlist"}, ["XYZ"])
        assert uwl == ["XYZ"], "explicit watchlist must opt out of auto-prefer"
    finally:
        _fonoloji_bist_list = _orig

    print("universes._selftest OK: bist30=%d bist50=%d fx_majors=%d fx_ext=%d"
          % (len(BIST_30), len(BIST_50), len(FX_MAJORS), len(FX_EXTENDED)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
