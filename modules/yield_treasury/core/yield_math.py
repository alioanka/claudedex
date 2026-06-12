"""PURE carry math for the yield_treasury advisor. No I/O, no side effects.

Honest framing: parking idle float at 3-8% APR is small carry, not edge.
The math therefore charges every cost that makes small carry negative —
round-trip gas/swap fees, a breakeven-days gate, and a withdrawal-latency
cap so capital is never advised into a venue it cannot leave fast enough
when a trading module needs it.

Self-test: python -m modules.yield_treasury.core.yield_math
"""

from __future__ import annotations

import math
from typing import Optional

HOLD = "HOLD"
DEPLOY_CANDIDATE = "DEPLOY_CANDIDATE"


def deployable_amount(idle: float, undeployed_floor: float,
                      max_deploy_frac: float, venue_cap_frac: float,
                      already_deployed: float = 0.0) -> float:
    """How much of `idle` MAY be parked at one venue (same units as idle).

    - `undeployed_floor` (gas + trading float) is never touchable.
    - at most `max_deploy_frac` of the touchable idle, fleet-wide.
    - at most `venue_cap_frac` of total idle at any single venue.
    - `already_deployed` at this venue eats into the venue cap.
    """
    if idle <= 0:
        return 0.0
    touchable = max(0.0, idle - max(0.0, undeployed_floor))
    fleet_cap = touchable * min(max(max_deploy_frac, 0.0), 1.0)
    venue_cap = idle * min(max(venue_cap_frac, 0.0), 1.0) - max(0.0, already_deployed)
    return max(0.0, min(fleet_cap, venue_cap))


def roundtrip_cost(deployable: float, fixed_cost: float, fee_bps: float) -> float:
    """Total deposit+withdraw cost: fixed (gas, both legs) + proportional (bps)."""
    return max(0.0, fixed_cost) + max(0.0, deployable) * max(0.0, fee_bps) / 10_000.0


def carry_advice(*, deployable: float, apr: float,
                 fixed_cost: float, fee_bps: float,
                 horizon_days: float, max_breakeven_days: float,
                 withdrawal_latency_s: float, max_withdrawal_latency_s: float,
                 min_deployable: float) -> dict:
    """One venue's advisory verdict. Pure; all amounts share one unit.

    Returns: gross_daily, roundtrip_cost, breakeven_days (inf-safe None),
    net_horizon, recommendation (HOLD | DEPLOY_CANDIDATE), reason.
    A DEPLOY_CANDIDATE is ADVICE ONLY — nothing in this module deposits.
    """
    cost = roundtrip_cost(deployable, fixed_cost, fee_bps)
    gross_daily = max(0.0, deployable) * max(0.0, apr) / 365.0
    breakeven: Optional[float] = (cost / gross_daily) if gross_daily > 0 else None
    net_horizon = gross_daily * max(0.0, horizon_days) - cost

    out = {
        "gross_daily": gross_daily,
        "roundtrip_cost": cost,
        "breakeven_days": breakeven,
        "net_horizon": net_horizon,
        "recommendation": HOLD,
        "reason": "",
    }
    if deployable < min_deployable or deployable <= 0:
        out["reason"] = "idle_below_min"
    elif apr <= 0:
        out["reason"] = "no_yield"
    elif withdrawal_latency_s > max_withdrawal_latency_s:
        out["reason"] = "withdrawal_latency_exceeds_cap"
    elif breakeven is None or breakeven > max_breakeven_days:
        out["reason"] = "breakeven_too_slow"
    elif net_horizon <= 0:
        out["reason"] = "negative_net_over_horizon"
    else:
        out["recommendation"] = DEPLOY_CANDIDATE
        out["reason"] = "carry_clears_costs"
    return out


def latency_note(withdrawal_latency_s: float) -> str:
    """The caveat every advice row must carry: parked capital is NOT float."""
    return (f"recall takes ~{int(withdrawal_latency_s)}s; while parked, this "
            f"capital cannot fund trading entries/exits or gas")


# ───────────────────────────── self-test ──────────────────────────────────────

if __name__ == "__main__":
    # deployable: floor first, then fleet frac, then venue cap binds
    assert deployable_amount(1000, 100, 0.5, 0.25) == 250.0          # venue cap binds
    assert deployable_amount(1000, 100, 0.2, 0.25) == 180.0          # fleet frac binds
    assert deployable_amount(1000, 100, 0.5, 0.25, already_deployed=200) == 50.0
    assert deployable_amount(50, 100, 0.5, 0.25) == 0.0              # under floor
    assert deployable_amount(0, 0, 0.5, 0.25) == 0.0
    assert deployable_amount(1000, 0, 2.0, 2.0) == 1000.0            # fracs clamped to 1

    # Arbitrum Aave USDC: $250 at 5%, $0.30 roundtrip gas -> pays inside 10d
    a = carry_advice(deployable=250.0, apr=0.05, fixed_cost=0.30, fee_bps=0.0,
                     horizon_days=30, max_breakeven_days=10,
                     withdrawal_latency_s=120, max_withdrawal_latency_s=86400,
                     min_deployable=50.0)
    assert a["recommendation"] == DEPLOY_CANDIDATE, a
    assert math.isclose(a["gross_daily"], 250 * 0.05 / 365), a
    assert 8.0 < a["breakeven_days"] < 9.0, a
    assert a["net_horizon"] > 0, a

    # Same float on Ethereum mainnet ($16 roundtrip gas) -> honest HOLD
    a = carry_advice(deployable=250.0, apr=0.05, fixed_cost=16.0, fee_bps=0.0,
                     horizon_days=30, max_breakeven_days=10,
                     withdrawal_latency_s=120, max_withdrawal_latency_s=86400,
                     min_deployable=50.0)
    assert a["recommendation"] == HOLD and a["reason"] == "breakeven_too_slow", a

    # jitoSOL: 4.95 SOL at 7%, 10bps swap roundtrip + 0.001 SOL tx -> deploys
    a = carry_advice(deployable=4.95, apr=0.07, fixed_cost=0.001, fee_bps=10.0,
                     horizon_days=30, max_breakeven_days=10,
                     withdrawal_latency_s=120, max_withdrawal_latency_s=86400,
                     min_deployable=0.5)
    assert a["recommendation"] == DEPLOY_CANDIDATE, a
    assert math.isclose(a["roundtrip_cost"], 0.001 + 4.95 * 0.001), a

    # stETH withdrawal queue (3d) over a 1d latency cap -> HOLD on latency,
    # regardless of APR (capital a trading module may need is never queued)
    a = carry_advice(deployable=1.0, apr=0.99, fixed_cost=0.0, fee_bps=0.0,
                     horizon_days=30, max_breakeven_days=10,
                     withdrawal_latency_s=259200, max_withdrawal_latency_s=86400,
                     min_deployable=0.01)
    assert a["reason"] == "withdrawal_latency_exceeds_cap", a

    # zero/negative yield and dust floors
    a = carry_advice(deployable=250.0, apr=0.0, fixed_cost=0.0, fee_bps=0.0,
                     horizon_days=30, max_breakeven_days=10,
                     withdrawal_latency_s=120, max_withdrawal_latency_s=86400,
                     min_deployable=50.0)
    assert a["reason"] == "no_yield" and a["breakeven_days"] is None, a
    a = carry_advice(deployable=10.0, apr=0.05, fixed_cost=0.1, fee_bps=0.0,
                     horizon_days=30, max_breakeven_days=10,
                     withdrawal_latency_s=120, max_withdrawal_latency_s=86400,
                     min_deployable=50.0)
    assert a["reason"] == "idle_below_min", a

    # horizon shorter than breakeven -> negative net even if breakeven gate set loose
    a = carry_advice(deployable=250.0, apr=0.05, fixed_cost=0.30, fee_bps=0.0,
                     horizon_days=5, max_breakeven_days=100,
                     withdrawal_latency_s=120, max_withdrawal_latency_s=86400,
                     min_deployable=50.0)
    assert a["reason"] == "negative_net_over_horizon", a

    assert "cannot fund trading" in latency_note(120)
    print("yield_math self-test OK")
