"""Variant-challenge evaluation for param_tuner proposals — pure math.

Pattern adapted from HKUDS/AI-Trader (MIT): "challenge" evaluation runs the
SAME strategy under parameter VARIANTS and scores every variant with the
IDENTICAL live mark-to-market harness over the identical window. Ported
here as the acceptance gate for bandit proposals: instead of trusting the
bandit's in-sample (counterfactual-free but regime-confounded) reward
history, a proposal opens a CHALLENGE — baseline (current value) vs
variant (proposed value) scored side by side against the module's REAL
closed trades that happen AFTER the challenge opens. Out-of-sample by
construction: no replay, no counterfactual reward, no trade that existed
before the challenge started. This directly answers the module's
documented "counterfactual rewards = overfit risk, acceptance must be
out-of-sample" caveat.

HONESTY CONSTRAINT: only ONE config value is actually live at a time, so
a variant can only be scored via an observable proxy on realized trades.
The one proxy that is exactly correct is the ENTRY-GATE proxy: for a
threshold knob whose only effect is WHICH candidate trades get taken, the
variant's counterfactual book is a deterministic SUBSET of the realized
book (variant takes the trade iff its recorded gate feature passes the
variant threshold; a skipped trade scores neutral). Two hard caveats are
enforced in code, not prose:
  * CENSORING: a LOOSER threshold would have taken trades that were never
    taken (and never recorded). Those challenges are unscoreable and
    BYPASS the challenge (old direct-to-pending behavior, logged).
  * EXEMPT KNOBS: knobs whose effect needs unrecorded data (price paths
    for time-based exits, MFE for TP distances, features not persisted
    per trade) have NO honest proxy. They are listed in CHALLENGE_EXEMPT
    with the reason, bypass the challenge, and keep old behavior.

No DB, no async, no randomness in this file — deterministic and
self-tested: ``python -m modules.param_tuner.core.challenge``
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

VERDICT_PASSED = "passed"
VERDICT_FAILED = "challenge_failed"
VERDICT_ABANDONED = "abandoned"

ROUTE_CHALLENGE = "challenge"   # honest proxy exists -> open a challenge
ROUTE_EXEMPT = "exempt"         # no honest proxy -> old behavior, logged
ROUTE_CENSORED = "censored"     # proxy exists but direction unobservable


@dataclass(frozen=True)
class GateScorer:
    """Entry-gate knob: trade taken iff recorded feature passes threshold.

    direction='min_gate' means taken iff feature >= threshold (the only
    direction currently needed). The columns name REAL per-trade fields in
    the owning module's closed-trades table.
    """

    module: str
    table: str
    feature_col: str      # recorded per-trade gate feature at entry
    pnl_pct_col: str      # realized per-trade P&L percent (the MTM truth)
    entry_col: str        # entry timestamp (out-of-sample cut)
    exit_col: str         # exit timestamp (incremental-scoring cursor)
    has_status: bool      # table needs status='closed' filter
    direction: str = "min_gate"


# Knobs with an honest out-of-sample proxy. Keyed by (config_type, key).
KNOB_SCORERS: Dict[Tuple[str, str], GateScorer] = {
    ("ai_config", "confidence_threshold"): GateScorer(
        module="ai", table="ai_trades",
        feature_col="confidence_score", pnl_pct_col="profit_loss_pct",
        entry_col="entry_timestamp", exit_col="exit_timestamp",
        has_status=True,
    ),
}

# Knobs with NO honest proxy given what the trade tables record today.
# Each reason states what recording would unlock a real challenge.
CHALLENGE_EXEMPT: Dict[Tuple[str, str], str] = {
    ("trading", "min_vol_liq_ratio"):
        "vol/liq ratio at entry is not persisted per trade in the dex "
        "trades table; recording it would enable the gate proxy",
    ("sniper_config", "max_hold_minutes"):
        "time-based exit re-mark needs the token price at the candidate "
        "cutoff; no per-trade price path is recorded",
    ("solana_jupiter", "jupiter_auto_exit"):
        "time-based exit re-mark needs the token price at the candidate "
        "cutoff; no per-trade price path is recorded",
    ("futures_risk", "atr_tp_rr_ratio"):
        "whether a nearer TP would have filled needs max-favorable-"
        "excursion (MFE), which futures_trades does not record; scoring "
        "TP-exits only would structurally bias against lower ratios",
}


def classify_route(config_type: str, key: str,
                   baseline_value: float, variant_value: float
                   ) -> Tuple[str, object]:
    """(route, detail): ROUTE_CHALLENGE + scorer, or ROUTE_EXEMPT /
    ROUTE_CENSORED + human-readable reason. Unknown knobs are exempt —
    a knob is only challengeable once someone writes its honest proxy."""
    k = (str(config_type), str(key))
    scorer = KNOB_SCORERS.get(k)
    if scorer is None:
        return ROUTE_EXEMPT, CHALLENGE_EXEMPT.get(
            k, "no scoring function registered for this knob")
    if scorer.direction == "min_gate" and variant_value < baseline_value:
        return ROUTE_CENSORED, (
            "loosening a min-gate is unobservable: trades below the live "
            "threshold were never taken/recorded (censored direction)")
    return ROUTE_CHALLENGE, scorer


# ────────────────────────────── scoring math ────────────────────────────────

def trade_reward(pnl_pct: float, cap_pct: float) -> float:
    """Realized per-trade P&L percent -> reward in [0,1], neutral 0.5.
    Saturates at +/- cap_pct so one moonshot cannot decide a challenge."""
    cap = max(1e-9, float(cap_pct))
    x = max(-1.0, min(1.0, float(pnl_pct) / cap))
    return 0.5 + 0.5 * x

NEUTRAL_REWARD = 0.5   # the reward of NOT taking a trade


def gate_reward(feature: float, pnl_pct: float, threshold: float,
                cap_pct: float, direction: str = "min_gate") -> float:
    """Reward a candidate threshold earns on ONE realized trade: the
    trade's MTM reward if the candidate would have taken it, else exactly
    neutral (skipping a loser gains vs baseline, skipping a winner costs)."""
    taken = feature >= threshold - 1e-12 if direction == "min_gate" \
        else feature <= threshold + 1e-12
    return trade_reward(pnl_pct, cap_pct) if taken else NEUTRAL_REWARD


def accumulate(observations: Iterable[Tuple[float, float]],
               baseline_value: float, variant_value: float,
               cap_pct: float, direction: str = "min_gate"
               ) -> Tuple[int, float, float]:
    """Score a slice of realized (feature, pnl_pct) trades under BOTH
    values with the identical harness. Returns (n, baseline_sum,
    variant_sum) deltas to add to the challenge's running stats."""
    n, bsum, vsum = 0, 0.0, 0.0
    for feature, pnl_pct in observations:
        if feature is None or pnl_pct is None:
            continue
        n += 1
        bsum += gate_reward(float(feature), float(pnl_pct),
                            baseline_value, cap_pct, direction)
        vsum += gate_reward(float(feature), float(pnl_pct),
                            variant_value, cap_pct, direction)
    return n, bsum, vsum


def resolve(n: int, baseline_sum: float, variant_sum: float, *,
            min_samples: int, min_edge_pct: float,
            window_elapsed: bool) -> Optional[Tuple[str, str]]:
    """Verdict once the window has elapsed; None while still running.

    Conservative by design: the variant must BEAT the baseline mean by
    min_edge_pct percent (relative) over at least min_samples out-of-
    sample observations, else the challenge fails — including the
    'window ended with too little tape' case (no evidence != pass)."""
    if not window_elapsed:
        return None
    if n < max(1, int(min_samples)):
        return VERDICT_FAILED, (
            f"insufficient out-of-sample evidence: {n} observation(s) < "
            f"{int(min_samples)} required at window end")
    bm, vm = baseline_sum / n, variant_sum / n
    edge = vm - bm
    required = bm * (float(min_edge_pct) / 100.0)
    if edge > 0 and edge >= required:
        return VERDICT_PASSED, (
            f"variant mean {vm:.4f} beat baseline {bm:.4f} by {edge:.4f} "
            f"(required {required:.4f} = {min_edge_pct:g}% of baseline, "
            f"n={n} out-of-sample trades)")
    return VERDICT_FAILED, (
        f"variant mean {vm:.4f} vs baseline {bm:.4f}: edge {edge:.4f} < "
        f"required {required:.4f} ({min_edge_pct:g}% of baseline, n={n})")


# ────────────────────────────── self-test ───────────────────────────────────

def _selftest() -> None:
    AI = ("ai_config", "confidence_threshold")

    # 1. Routing: honest-proxy knob, tightening direction -> challenge.
    route, detail = classify_route(*AI, 0.35, 0.50)
    assert route == ROUTE_CHALLENGE and isinstance(detail, GateScorer), route

    # 2. Loosening a min-gate is censored -> bypass, never a fake score.
    route, reason = classify_route(*AI, 0.50, 0.35)
    assert route == ROUTE_CENSORED and "censored" in reason, (route, reason)

    # 3. Every documented exempt knob bypasses with its reason.
    for k, expected in CHALLENGE_EXEMPT.items():
        route, reason = classify_route(k[0], k[1], 1.0, 2.0)
        assert route == ROUTE_EXEMPT and reason == expected, (k, route)
    # ... and so does an unknown knob (exempt-by-default).
    route, _ = classify_route("nope", "unknown_knob", 0.0, 1.0)
    assert route == ROUTE_EXEMPT

    # 4. MTM reward mapping: neutral at 0, saturates at the cap, symmetric.
    assert trade_reward(0.0, 20) == 0.5
    assert trade_reward(20.0, 20) == 1.0 and trade_reward(500.0, 20) == 1.0
    assert trade_reward(-20.0, 20) == 0.0
    assert abs(trade_reward(10.0, 20) - 0.75) < 1e-12

    # 5. Gate reward: variant skips a below-threshold trade -> neutral.
    assert gate_reward(0.40, -8.0, 0.50, 20) == NEUTRAL_REWARD
    assert gate_reward(0.60, -8.0, 0.50, 20) == trade_reward(-8.0, 20)

    # 6. Accumulate = challenge OPENS and ACCUMULATES: baseline 0.35 vs
    #    variant 0.50 on a tape where low-confidence trades lose. The
    #    variant skips the losers -> higher mean. Both sides scored on the
    #    identical trades with the identical harness.
    tape = [(0.40, -10.0), (0.42, -6.0), (0.55, +8.0), (0.60, +12.0),
            (0.38, -12.0), (0.70, +4.0)] * 4     # 24 obs >= min_samples 20
    n, bsum, vsum = accumulate(tape, 0.35, 0.50, cap_pct=20)
    assert n == 24 and vsum > bsum, (n, bsum, vsum)
    none_tape = [(None, 5.0), (0.5, None)]
    assert accumulate(none_tape, 0.35, 0.50, 20) == (0, 0.0, 0.0)

    # 7. Resolves PASS: window elapsed, enough samples, edge above 5%.
    v = resolve(n, bsum, vsum, min_samples=20, min_edge_pct=5,
                window_elapsed=True)
    assert v is not None and v[0] == VERDICT_PASSED, v

    # 8. Still running -> no verdict; window end w/o samples -> FAILED.
    assert resolve(n, bsum, vsum, min_samples=20, min_edge_pct=5,
                   window_elapsed=False) is None
    v = resolve(3, 1.5, 1.6, min_samples=20, min_edge_pct=5,
                window_elapsed=True)
    assert v[0] == VERDICT_FAILED and "insufficient" in v[1], v

    # 9. Resolves FAIL: variant edge inside the required margin.
    v = resolve(24, 12.0, 12.1, min_samples=20, min_edge_pct=5,
                window_elapsed=True)   # +0.83% < 5% required
    assert v[0] == VERDICT_FAILED, v
    #    ... and a variant that is outright worse also fails.
    v = resolve(24, 12.0, 10.0, min_samples=20, min_edge_pct=5,
                window_elapsed=True)
    assert v[0] == VERDICT_FAILED, v

    # 10. Determinism: identical inputs -> identical verdict text.
    a = resolve(n, bsum, vsum, min_samples=20, min_edge_pct=5,
                window_elapsed=True)
    b = resolve(n, bsum, vsum, min_samples=20, min_edge_pct=5,
                window_elapsed=True)
    assert a == b

    print("challenge self-test OK")


if __name__ == "__main__":
    _selftest()
