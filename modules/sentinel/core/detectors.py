"""SENTINEL detectors — pure, deterministic anomaly math. No I/O, no clock.

Every function here takes plain numbers/dicts collected by sentinel_engine and
returns an Anomaly (or None). All thresholds arrive as arguments so the engine
can feed DB-configured values. Self-test: `python -m modules.sentinel.core.detectors`.

Severity grading is uniform: WARN at the warn threshold, CRITICAL at the
critical threshold. Detectors NEVER decide actuation — they only describe what
they saw; FREEZE_ELIGIBLE_DETECTORS below is the single source of which
detector classes the (default-off) autopilot may act on, and it deliberately
contains only the "module is actively losing money" class. Fleet-scoped market
anomalies (depeg, divergence) and liveness anomalies (silent module, full
rejection) stay advisory in v1 — freezing cannot fix a dead feed or a dead
process, and auto-pausing the whole fleet on a price-board glitch is a worse
failure than the one being detected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median
from typing import Dict, List, Optional, Tuple

SEVERITY_INFO = "info"
SEVERITY_WARN = "warn"
SEVERITY_CRITICAL = "critical"

DETECTOR_STABLE_DEPEG = "stable_depeg"
DETECTOR_PRICE_DIVERGENCE = "price_divergence"
DETECTOR_SILENT_MODULE = "silent_module"
DETECTOR_FULL_REJECTION = "full_rejection"
DETECTOR_LOSS_VELOCITY = "loss_velocity"
DETECTOR_CORRELATED_DRAWDOWN = "correlated_drawdown"

# The ONLY detector classes the autopilot may freeze on (CRITICAL severity
# required as well). Single-sourced here so engine + docs cannot drift.
FREEZE_ELIGIBLE_DETECTORS = frozenset({
    DETECTOR_LOSS_VELOCITY,
    DETECTOR_CORRELATED_DRAWDOWN,
})


@dataclass
class Anomaly:
    """One detected anomaly. `subject` is a module short key, an asset symbol,
    or 'fleet'. `module_scoped` is True iff subject is a pause-file short key.
    `details` may carry `freeze_candidates` (list of module short keys) for
    fleet-scoped anomalies whose response is per-module freezing."""
    detector: str
    subject: str
    severity: str
    value: float
    threshold: float
    message: str
    details: dict = field(default_factory=dict)
    module_scoped: bool = False


def _grade(value: float, warn: float, critical: float) -> Optional[str]:
    """Uniform two-step grading on a magnitude (value >= 0)."""
    if critical > 0 and value >= critical:
        return SEVERITY_CRITICAL
    if warn > 0 and value >= warn:
        return SEVERITY_WARN
    return None


# ───────────────────────── market detectors ─────────────────────────

def detect_stable_depeg(symbol: str, prices: List[float],
                        warn_bps: float, critical_bps: float) -> Optional[Anomaly]:
    """Stablecoin depeg: median deviation of `prices` (independent USD quotes
    for one stable) from 1.0, in bps. With a SINGLE source the severity is
    capped at WARN — one feed alone never produces a CRITICAL depeg (the
    feed itself is the likelier failure)."""
    clean = [p for p in prices if isinstance(p, (int, float)) and p > 0]
    if not clean:
        return None
    dev_bps = abs(median(clean) - 1.0) * 10_000.0
    sev = _grade(dev_bps, warn_bps, critical_bps)
    if sev is None:
        return None
    # SINGLE-SOURCE noise filter: with only one feed, a small deviation
    # (e.g. an illiquid Coinbase DAI-USD print 75bps off) is far more likely a
    # feed artifact than a real depeg, and it WARN-spammed every tick. Require a
    # single source to clear the CRITICAL threshold before it fires at all, and
    # still cap it at WARN (one feed alone never escalates to CRITICAL).
    if len(clean) < 2:
        if dev_bps < critical_bps:
            return None
        sev = SEVERITY_WARN
    return Anomaly(
        detector=DETECTOR_STABLE_DEPEG, subject=symbol, severity=sev,
        value=round(dev_bps, 2), threshold=warn_bps,
        message=(f"{symbol} median price {median(clean):.4f} deviates "
                 f"{dev_bps:.0f}bps from peg across {len(clean)} source(s)"),
        details={"prices": clean, "sources": len(clean)},
    )


def detect_price_divergence(symbol: str, price_a: Optional[float],
                            price_b: Optional[float],
                            warn_bps: float, critical_bps: float,
                            source_a: str = "a", source_b: str = "b") -> Optional[Anomaly]:
    """Oracle/venue deviation: |a-b| relative to their mid, in bps, for one
    asset quoted by two independent sources. Needs both sides — missing data
    is a non-result, never an anomaly (fail-soft)."""
    if not price_a or not price_b or price_a <= 0 or price_b <= 0:
        return None
    mid = (price_a + price_b) / 2.0
    div_bps = abs(price_a - price_b) / mid * 10_000.0
    sev = _grade(div_bps, warn_bps, critical_bps)
    if sev is None:
        return None
    return Anomaly(
        detector=DETECTOR_PRICE_DIVERGENCE, subject=symbol, severity=sev,
        value=round(div_bps, 2), threshold=warn_bps,
        message=(f"{symbol} diverges {div_bps:.0f}bps between "
                 f"{source_a}={price_a:.6g} and {source_b}={price_b:.6g}"),
        details={source_a: price_a, source_b: price_b},
    )


# ───────────────────────── liveness detectors ─────────────────────────

def detect_silent_module(module: str, heartbeat_age_s: Optional[float],
                         expected_max_age_s: float,
                         warn_factor: float, critical_factor: float) -> Optional[Anomaly]:
    """Silent module death: heartbeat (runtime_stats updated_at) age measured
    in multiples of the module's expected refresh interval. The engine only
    calls this for modules that were recently alive, so a disabled module
    never alarms. age None (table missing/unreadable) is a non-result."""
    if heartbeat_age_s is None or expected_max_age_s <= 0:
        return None
    factor = heartbeat_age_s / expected_max_age_s
    sev = _grade(factor, warn_factor, critical_factor)
    if sev is None:
        return None
    return Anomaly(
        detector=DETECTOR_SILENT_MODULE, subject=module, severity=sev,
        value=round(factor, 2), threshold=warn_factor,
        message=(f"{module} heartbeat is {heartbeat_age_s:.0f}s old "
                 f"({factor:.1f}x its {expected_max_age_s:.0f}s refresh)"),
        details={"age_seconds": round(heartbeat_age_s, 1),
                 "expected_max_age_seconds": expected_max_age_s},
        module_scoped=True,
    )


def extract_candidate_flow(stats: dict) -> Optional[Tuple[int, int]]:
    """Generic (seen, accepted) extraction from a runtime_stats `stats` JSONB.

    seen     = tokens_analyzed if present, else passed/accepted + sum(*_rejected)
    accepted = passed_safety if present, else None -> non-result

    Counters are cumulative since process start, so this detects the
    "100%-rejection since restart" failure class (the DEX vol/liq and Futures
    volume-gate bugs). Modules whose stats lack these keys are skipped."""
    if not isinstance(stats, dict):
        return None
    accepted = stats.get("passed_safety")
    if not isinstance(accepted, int):
        return None
    rejected = sum(v for k, v in stats.items()
                   if k.endswith("_rejected") and isinstance(v, int))
    seen = stats.get("tokens_analyzed")
    if not isinstance(seen, int):
        seen = accepted + rejected
    return (seen, accepted)


def detect_full_rejection(module: str, seen: int, accepted: int,
                          min_seen: int,
                          critical_seen_factor: float = 4.0) -> Optional[Anomaly]:
    """100%-rejection pattern: the module evaluated >= min_seen candidates and
    accepted ZERO. WARN at min_seen, CRITICAL once the sample is
    critical_seen_factor times larger (chance is no longer plausible)."""
    if min_seen <= 0 or seen < min_seen or accepted > 0:
        return None
    sev = (SEVERITY_CRITICAL if seen >= min_seen * critical_seen_factor
           else SEVERITY_WARN)
    return Anomaly(
        detector=DETECTOR_FULL_REJECTION, subject=module, severity=sev,
        value=float(seen), threshold=float(min_seen),
        message=(f"{module} rejected 100% of {seen} candidates since process "
                 f"start (0 accepted) — a gate is likely mis-set"),
        details={"seen": seen, "accepted": accepted},
        module_scoped=True,
    )


# ───────────────────────── P&L detectors ─────────────────────────

def detect_loss_velocity(module: str, window_pnl_usd: float,
                         window_minutes: float,
                         warn_usd_per_hr: float,
                         critical_usd_per_hr: float) -> Optional[Anomaly]:
    """Abnormal loss velocity: realized window loss converted to USD/hour.
    Profits and zero windows are non-results."""
    if window_minutes <= 0 or window_pnl_usd >= 0:
        return None
    loss_per_hr = -window_pnl_usd / (window_minutes / 60.0)
    sev = _grade(loss_per_hr, warn_usd_per_hr, critical_usd_per_hr)
    if sev is None:
        return None
    return Anomaly(
        detector=DETECTOR_LOSS_VELOCITY, subject=module, severity=sev,
        value=round(loss_per_hr, 2), threshold=warn_usd_per_hr,
        message=(f"{module} losing {loss_per_hr:.2f} USD/hr "
                 f"({window_pnl_usd:.2f} USD over {window_minutes:.0f}m)"),
        details={"window_pnl_usd": round(window_pnl_usd, 4),
                 "window_minutes": window_minutes},
        module_scoped=True,
    )


def detect_correlated_drawdown(module_pnls: Dict[str, float],
                               min_modules: int,
                               module_loss_usd: float,
                               critical_total_usd: float) -> Optional[Anomaly]:
    """Correlated drawdown: >= min_modules each losing >= module_loss_usd in
    the same window (everything is crypto-beta; this is the fleet tail).
    CRITICAL when the combined loss of the losers exceeds critical_total_usd.
    The losing modules are exposed as `freeze_candidates` for the autopilot."""
    if min_modules <= 0 or module_loss_usd <= 0:
        return None
    losers = {m: p for m, p in module_pnls.items()
              if isinstance(p, (int, float)) and p <= -module_loss_usd}
    if len(losers) < min_modules:
        return None
    total_loss = -sum(losers.values())
    sev = (SEVERITY_CRITICAL
           if critical_total_usd > 0 and total_loss >= critical_total_usd
           else SEVERITY_WARN)
    names = sorted(losers)
    return Anomaly(
        detector=DETECTOR_CORRELATED_DRAWDOWN, subject="fleet", severity=sev,
        value=round(total_loss, 2), threshold=float(min_modules),
        message=(f"correlated drawdown: {len(losers)} modules "
                 f"({', '.join(names)}) lost {total_loss:.2f} USD combined "
                 f"in the same window"),
        details={"losers": {m: round(p, 4) for m, p in losers.items()},
                 "freeze_candidates": names},
        module_scoped=False,
    )


# ───────────────────────── self-test ─────────────────────────

def _self_test() -> None:
    # stable depeg: 1.0 never fires; 0.99 = 100bps fires WARN at 50/150
    assert detect_stable_depeg("USDC", [1.0, 1.0001], 50, 150) is None
    a = detect_stable_depeg("USDC", [0.99, 0.991], 50, 150)
    assert a and a.severity == SEVERITY_WARN and 85 < a.value < 105
    a = detect_stable_depeg("USDT", [0.97, 0.972], 50, 150)
    assert a and a.severity == SEVERITY_CRITICAL
    # single source caps CRITICAL at WARN
    a = detect_stable_depeg("DAI", [0.95], 50, 150)
    assert a and a.severity == SEVERITY_WARN and a.details["sources"] == 1
    assert detect_stable_depeg("USDC", [], 50, 150) is None
    assert detect_stable_depeg("USDC", [-1.0, 0.0], 50, 150) is None

    # price divergence: 1% on BTC = 100bps; midpoint denominator
    assert detect_price_divergence("BTC", 50000, 50000, 100, 300) is None
    a = detect_price_divergence("BTC", 50000, 50600, 100, 300, "cb", "kr")
    assert a and a.severity == SEVERITY_WARN and 118 < a.value < 121
    a = detect_price_divergence("SOL", 100.0, 104.0, 100, 300)
    assert a and a.severity == SEVERITY_CRITICAL
    assert detect_price_divergence("ETH", None, 3000, 100, 300) is None
    assert detect_price_divergence("ETH", 3000, 0, 100, 300) is None

    # silent module: 120s refresh; 200s old = 1.67x < warn 3x
    assert detect_silent_module("sniper", 200, 120, 3, 10) is None
    a = detect_silent_module("sniper", 600, 120, 3, 10)
    assert a and a.severity == SEVERITY_WARN and a.module_scoped
    a = detect_silent_module("dex", 1600, 150, 3, 10)
    assert a and a.severity == SEVERITY_CRITICAL
    assert detect_silent_module("ai", None, 1800, 3, 10) is None
    assert detect_silent_module("ai", 100, 0, 3, 10) is None

    # candidate flow extraction (sniper-shaped stats)
    flow = extract_candidate_flow({
        "tokens_analyzed": 80, "passed_safety": 0,
        "high_tax_rejected": 30, "low_liquidity_rejected": 50,
        "last_stats_log": "2026-06-12",
    })
    assert flow == (80, 0)
    flow = extract_candidate_flow({"passed_safety": 5, "low_score_rejected": 20})
    assert flow == (25, 5)
    assert extract_candidate_flow({"other": 1}) is None
    assert extract_candidate_flow("nope") is None

    # full rejection: 0 accepted of 30 (min 25) = WARN; of 100 = CRITICAL (4x)
    assert detect_full_rejection("dex", 10, 0, 25) is None
    assert detect_full_rejection("dex", 30, 1, 25) is None
    a = detect_full_rejection("dex", 30, 0, 25)
    assert a and a.severity == SEVERITY_WARN and a.module_scoped
    a = detect_full_rejection("futures", 100, 0, 25)
    assert a and a.severity == SEVERITY_CRITICAL

    # loss velocity: -10 USD over 60m = 10/hr; warn 15 no, -20 -> 20/hr WARN,
    # -60 -> 60/hr CRITICAL at 50
    assert detect_loss_velocity("solana", -10, 60, 15, 50) is None
    assert detect_loss_velocity("solana", 10, 60, 15, 50) is None
    a = detect_loss_velocity("solana", -20, 60, 15, 50)
    assert a and a.severity == SEVERITY_WARN and abs(a.value - 20) < 1e-9
    a = detect_loss_velocity("copy_trading", -30, 30, 15, 50)
    assert a and a.severity == SEVERITY_CRITICAL and abs(a.value - 60) < 1e-9
    assert a.detector in FREEZE_ELIGIBLE_DETECTORS

    # correlated drawdown: 3 losers >= 5 USD each; CRITICAL on 100 total
    pnls = {"dex": -6.0, "solana": -7.0, "futures": -8.0, "ai": 2.0}
    a = detect_correlated_drawdown(pnls, 3, 5.0, 100.0)
    assert a and a.severity == SEVERITY_WARN and a.subject == "fleet"
    assert a.details["freeze_candidates"] == ["dex", "futures", "solana"]
    assert not a.module_scoped
    a = detect_correlated_drawdown({"dex": -40, "solana": -40, "futures": -40},
                                   3, 5.0, 100.0)
    assert a and a.severity == SEVERITY_CRITICAL and a.value == 120.0
    assert detect_correlated_drawdown({"dex": -6, "solana": -7}, 3, 5, 100) is None
    assert detect_correlated_drawdown(pnls, 0, 5, 100) is None

    # freeze eligibility is exactly the active-loss class
    assert FREEZE_ELIGIBLE_DETECTORS == {"loss_velocity", "correlated_drawdown"}

    print("sentinel detectors self-test: OK")


if __name__ == "__main__":
    _self_test()
