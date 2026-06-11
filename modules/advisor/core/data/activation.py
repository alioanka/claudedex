"""
activation — ONE place that decides which data source each Turkish-stack path
resolves to, so the analyzers, the KAP/accumulator price path and the startup
diagnostics can never drift apart.

ROOT CAUSE this module exists for: the Fonoloji integration shipped across
migrations 078-082 but stayed INACTIVE for a week because activation depended
on flipping several advisor_config rows that were never flipped
(advisor_bist_universe='watchlist', advisor_midas_data_source='tefas_scrape'
etc.). The rules here implement AUTO-PREFER: when the key is present and
`advisor_fonoloji_auto_prefer` (default true) is on, Fonoloji is tried FIRST
for BIST data, the BIST universe, Midas NAV and KAP/accumulator prices —
without the operator touching five settings. Explicit opt-outs still win:
  * advisor_midas_data_source='manual'   (operator-entered NAV)
  * advisor_bist_data_source='matriks'   (paid path, stub)
  * advisor_bist_universe explicit preset ('watchlist'/'bist30'/'bist50'/'custom')
  * advisor_fonoloji_auto_prefer='false' (restores only-when-source-empty)
Every Fonoloji miss still falls through to the previous source chain
(fail-soft) — auto-prefer can only ADD a first attempt, never remove one.

Also owns the yfinance log-noise filter: yfinance logs
"$TICKER: possibly delisted; no price data found" at ERROR, which flooded
advisor_errors.log for non-equity KAP tickers (bonds, warrants, funds). Those
are EXPECTED misses, so the filter demotes them to DEBUG.

Self-test (offline): python -m modules.advisor.core.data.activation
"""

from __future__ import annotations

import logging
from typing import Dict

from modules.advisor.core.data.fonoloji_client import resolve_api_key

logger = logging.getLogger("advisor.data.activation")


def _src(config: dict, key: str) -> str:
    return str((config or {}).get(key, "") or "").strip().lower()


def _flag(config: dict, key: str, default: bool) -> bool:
    val = (config or {}).get(key, default)
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("true", "1", "yes", "on")


def fonoloji_active(config: dict) -> bool:
    """Key present AND the auto-prefer master switch on (default true)."""
    return bool(resolve_api_key(config)) and _flag(
        config, "advisor_fonoloji_auto_prefer", True
    )


def prefer_fonoloji_bist(config: dict) -> bool:
    """Should the BIST analyzer try Fonoloji FIRST for price series?
    Explicit 'fonoloji' or legacy '' always prefer (pre-083 behaviour);
    auto-prefer extends that to any non-'matriks' source value."""
    if not resolve_api_key(config):
        return False
    source = _src(config, "advisor_bist_data_source")
    if source in ("fonoloji", ""):
        return True
    if source == "matriks":
        return False  # explicit paid path wins
    return fonoloji_active(config)


def prefer_fonoloji_midas(config: dict) -> bool:
    """Should the Midas funds analyzer try Fonoloji NAV FIRST?
    'manual' is a hard explicit opt-out (operator-entered NAV)."""
    if not resolve_api_key(config):
        return False
    source = _src(config, "advisor_midas_data_source")
    if source in ("fonoloji", ""):
        return True
    if source == "manual":
        return False
    return fonoloji_active(config)


def prefer_fonoloji_prices(config: dict) -> bool:
    """Should the KAP price lookups / forward-return accumulator try the
    Fonoloji /stocks endpoints FIRST?"""
    return fonoloji_active(config)


def effective_universe_mode(config: dict) -> str:
    """
    Resolve advisor_bist_universe to the EFFECTIVE mode the scan will use:
      'fonoloji'  — live /stocks/list (key present)
      'snapshot'  — hardcoded BIST-50 snapshot (fonoloji requested but no key,
                    or auto requested with auto_prefer off but key present? no:
                    snapshot is the fail-soft target of a fonoloji attempt)
      'watchlist' — watchlist-only (explicit, or auto with no key)
      'bist30' / 'bist50' / 'custom' — explicit presets, unchanged.
    """
    mode = _src(config, "advisor_bist_universe") or "auto"
    if mode in ("auto",):
        return "fonoloji" if fonoloji_active(config) else "watchlist"
    if mode in ("fonoloji", "bist_all", "bistall", "all", "live"):
        return "fonoloji" if resolve_api_key(config) else "snapshot"
    if mode in ("bist30", "bist_30", "30"):
        return "bist30"
    if mode in ("bist50", "bist_50", "50"):
        return "bist50"
    if mode in ("custom",):
        return "custom"
    return "watchlist"


def resolve_sources(config: dict) -> Dict[str, str]:
    """Diagnostics summary: which source each Turkish-stack path resolves to
    RIGHT NOW (import probes included). Used by the startup banner."""
    def _importable(name: str) -> bool:
        try:
            __import__(name)
            return True
        except Exception:
            return False

    bist_source = _src(config, "advisor_bist_data_source")
    if bist_source == "matriks":
        bist = "matriks(stub)"
    elif prefer_fonoloji_bist(config):
        bist = "fonoloji"
    elif _importable("borsapy") and bist_source in ("borsapy", ""):
        bist = "borsapy"
    elif _importable("yfinance"):
        bist = "yfinance(degraded)"
    else:
        bist = "not_configured"

    midas_source = _src(config, "advisor_midas_data_source")
    if midas_source == "manual":
        midas = "manual"
    elif prefer_fonoloji_midas(config):
        midas = "fonoloji"
    elif _importable("tefas") and midas_source in ("tefas_crawler", ""):
        midas = "tefas_crawler"
    elif _importable("tefasfon") and midas_source in ("tefasfon", ""):
        midas = "tefasfon"
    else:
        midas = "tefas_scrape(fragile)"

    if prefer_fonoloji_prices(config):
        kap = "fonoloji"
    elif _importable("borsapy"):
        kap = "borsapy"
    else:
        kap = "yfinance(degraded)"

    return {
        "bist": bist,
        "midas": midas,
        "universe": effective_universe_mode(config),
        "kap_prices": kap,
    }


# ---------------------------------------------------------------------------
# yfinance log-noise filter
# ---------------------------------------------------------------------------

class _YFinanceNoiseFilter(logging.Filter):
    """Demote EXPECTED yfinance misses from ERROR to DEBUG. Non-equity KAP
    tickers (bonds, warrants, funds) legitimately have no Yahoo .IS price —
    that is not an error condition for the advisor."""

    _NOISE = ("possibly delisted", "no price data found",
              "no timezone found", "symbol may be delisted")

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg = record.getMessage().lower()
        except Exception:
            return True
        if any(frag in msg for frag in self._NOISE):
            record.levelno = logging.DEBUG
            record.levelname = "DEBUG"
        return True


def install_yfinance_noise_filter() -> None:
    """Attach the noise filter to the yfinance loggers (idempotent). Mutating
    record.levelno in a logger-level filter runs BEFORE handler level checks,
    so demoted records no longer reach advisor_errors.log (WARNING+)."""
    for name in ("yfinance", "yfinance.utils", "yfinance.data"):
        lg = logging.getLogger(name)
        if not any(isinstance(f, _YFinanceNoiseFilter) for f in lg.filters):
            lg.addFilter(_YFinanceNoiseFilter())
    logger.debug("[activation] yfinance noise filter installed.")


# ---------------------------------------------------------------------------
# Offline self-test — python -m modules.advisor.core.data.activation
# ---------------------------------------------------------------------------

def _self_test() -> int:
    failures = 0
    KEY = {"advisor_fonoloji_api_key": "K"}

    def check(cond, label):
        nonlocal failures
        if not cond:
            print(f"FAIL: {label}")
            failures += 1

    # No key => nothing prefers Fonoloji; universe auto => watchlist.
    check(not prefer_fonoloji_bist({}), "no-key bist must not prefer")
    check(not prefer_fonoloji_midas({}), "no-key midas must not prefer")
    check(effective_universe_mode({}) == "watchlist", "no-key auto universe")

    # Key => auto-prefer everywhere, even over stale explicit free sources.
    check(prefer_fonoloji_bist(KEY), "key bist auto-prefer")
    check(prefer_fonoloji_bist({**KEY, "advisor_bist_data_source": "yfinance"}),
          "key bist auto-prefer over stale yfinance value")
    check(prefer_fonoloji_midas({**KEY, "advisor_midas_data_source": "tefas_scrape"}),
          "key midas auto-prefer over dead tefas_scrape value")
    check(effective_universe_mode(KEY) == "fonoloji", "key auto universe")

    # Explicit opt-outs win.
    check(not prefer_fonoloji_bist({**KEY, "advisor_bist_data_source": "matriks"}),
          "matriks opt-out")
    check(not prefer_fonoloji_midas({**KEY, "advisor_midas_data_source": "manual"}),
          "manual opt-out")
    off = {**KEY, "advisor_fonoloji_auto_prefer": "false"}
    check(not prefer_fonoloji_bist({**off, "advisor_bist_data_source": "yfinance"}),
          "auto_prefer=false restores explicit yfinance")
    check(prefer_fonoloji_bist(off), "auto_prefer=false keeps legacy ''-prefers")
    check(effective_universe_mode({**KEY, "advisor_bist_universe": "watchlist"})
          == "watchlist", "explicit watchlist universe")
    check(effective_universe_mode({**KEY, "advisor_bist_universe": "bist50"})
          == "bist50", "explicit bist50 universe")
    check(effective_universe_mode({"advisor_bist_universe": "fonoloji"})
          == "snapshot", "fonoloji universe without key -> snapshot")

    # resolve_sources returns the four diagnostic keys.
    src = resolve_sources(KEY)
    check(set(src) == {"bist", "midas", "universe", "kap_prices"},
          f"resolve_sources keys {src}")
    check(src["bist"] == "fonoloji" and src["midas"] == "fonoloji",
          f"resolve_sources prefers fonoloji with key: {src}")

    # Noise filter demotes the delisted ERROR to DEBUG, passes others through.
    install_yfinance_noise_filter()
    install_yfinance_noise_filter()  # idempotent
    lg = logging.getLogger("yfinance")
    check(sum(isinstance(f, _YFinanceNoiseFilter) for f in lg.filters) == 1,
          "filter must be installed exactly once")
    rec = logging.LogRecord("yfinance", logging.ERROR, __file__, 1,
                            "XYZ.IS: possibly delisted; no price data found",
                            None, None)
    _YFinanceNoiseFilter().filter(rec)
    check(rec.levelno == logging.DEBUG, "delisted record must demote to DEBUG")
    rec2 = logging.LogRecord("yfinance", logging.ERROR, __file__, 1,
                             "real failure: connection refused", None, None)
    _YFinanceNoiseFilter().filter(rec2)
    check(rec2.levelno == logging.ERROR, "real errors must stay ERROR")

    print("SELF-TEST", "PASS" if failures == 0 else f"FAIL ({failures})")
    return 1 if failures else 0


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    sys.exit(_self_test())
