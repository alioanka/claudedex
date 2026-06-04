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

Config keys (advisor_config, migration 078):
  advisor_bist_universe  : '' | 'watchlist' | 'bist30' | 'bist50' | 'custom'
                           (default 'watchlist' — unchanged behaviour)
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
from typing import List

logger = logging.getLogger("advisor.universes")

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


def expand_bist_universe(config: dict, watchlist: List[str]) -> List[str]:
    """
    Build the BIST scan universe: preset + custom + watchlist (always included).
    Returns bare tickers; the BIST analyzer handles .IS normalisation.
    Default ('watchlist'/'') => returns the watchlist unchanged.
    """
    mode = str(config.get("advisor_bist_universe", "watchlist") or "watchlist").lower()
    custom = _split_csv(config.get("advisor_bist_universe_custom", ""))

    preset: List[str] = []
    if mode in ("bist30", "bist_30", "30"):
        preset = list(BIST_30)
    elif mode in ("bist50", "bist_50", "50"):
        preset = list(BIST_50)
    elif mode in ("custom",):
        preset = []  # custom-only; rely on the custom list
    # 'watchlist' / '' / unknown -> no preset (watchlist-only behaviour)

    # Strip any .IS the operator added so the universe is consistent (analyzer
    # re-adds .IS for yfinance). Watchlist is appended verbatim + always kept.
    merged = preset + custom + list(watchlist or [])
    out = _dedupe_preserve(merged)
    cap = _max(config)
    if len(out) > cap:
        # Keep watchlist tickers, then fill from the preset/custom head.
        wl = [w for w in (watchlist or [])]
        rest = [s for s in out if s not in wl]
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

    print("universes._selftest OK: bist30=%d bist50=%d fx_majors=%d fx_ext=%d"
          % (len(BIST_30), len(BIST_50), len(FX_MAJORS), len(FX_EXTENDED)))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _selftest()
