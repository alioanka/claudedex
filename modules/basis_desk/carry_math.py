"""Delta-neutral carry math for the basis_desk module — PURE, self-tested.

Edge source (one sentence): perp funding is a persistent, directly observable
risk premium paid by the crowded side of the book; pairing the funding-
collecting perp leg with an equal-base-quantity SPOT hedge leg harvests that
premium with ~zero price delta — the piece the futures funding-carry v2
strategy explicitly leaves to a manual operator action.

Structure (by funding sign):
    funding > 0  (longs pay shorts)  -> SHORT perp + LONG spot   (no borrow)
    funding < 0  (shorts pay longs)  -> LONG perp + SHORT spot   (needs spot
                                        borrow — flagged, OFF by default)

Cost model (per round trip, both legs, in bps of notional — extends the
funding-carry v2 model with the spot leg):
    round_trip_cost_bps = 2*perp_taker_fee_bps + 2*spot_taker_fee_bps
                        + perp_slippage_bps + spot_slippage_bps
                        + liquidation_premium_bps
    adverse_basis_cost_bps = max(0, -favorable_basis_bps)
        where favorable_basis_bps = +basis for SHORT-perp, -basis for
        LONG-perp; favorable basis is NEVER credited (conservative — basis
        convergence is not guaranteed on a perp), adverse basis is ALWAYS
        charged in full.
    total_cost_bps      = round_trip_cost_bps + adverse_basis_cost_bps
    breakeven_intervals = total_cost_bps / |funding_bps|
    net_carry_bps(H)    = |funding_bps| * H - total_cost_bps

This file has NO I/O and NO engine imports so it is unit-testable offline:
    python -m modules.basis_desk.carry_math
"""

import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

SHORT_PERP_LONG_SPOT = 'SHORT_PERP_LONG_SPOT'
LONG_PERP_SHORT_SPOT = 'LONG_PERP_SHORT_SPOT'

HOURS_PER_YEAR = 8760.0


@dataclass(frozen=True)
class FeeModel:
    """Round-trip cost assumptions in bps of notional (taker both legs)."""
    perp_taker_fee_bps: float = 6.0    # Bybit linear taker 0.055% ~= 5.5
    spot_taker_fee_bps: float = 10.0   # Bybit spot taker 0.10%
    perp_slippage_bps: float = 5.0
    spot_slippage_bps: float = 5.0
    liquidation_premium_bps: float = 2.0  # margin-buffer / forced-exit reserve

    def round_trip_cost_bps(self) -> float:
        return (2.0 * self.perp_taker_fee_bps
                + 2.0 * self.spot_taker_fee_bps
                + self.perp_slippage_bps
                + self.spot_slippage_bps
                + self.liquidation_premium_bps)


@dataclass(frozen=True)
class CarryPlan:
    """A fully-costed two-leg delta-neutral carry suggestion."""
    symbol: str
    venue: str
    direction: str            # SHORT_PERP_LONG_SPOT | LONG_PERP_SHORT_SPOT
    perp_side: str            # SELL | BUY
    spot_side: str            # BUY | SELL
    funding_bps: float        # signed, per funding interval
    funding_interval_hours: float
    perp_price: float
    spot_price: float
    basis_bps: float          # (perp - spot)/spot * 1e4, signed
    adverse_basis_cost_bps: float
    gross_carry_bps_per_interval: float
    round_trip_cost_bps: float
    total_cost_bps: float
    breakeven_intervals: float
    horizon_intervals: float
    net_carry_bps_at_horizon: float
    apr_gross_pct: float
    needs_borrow: bool
    actionable: bool
    reason: str


def compute_basis_bps(perp_price: float, spot_price: float) -> float:
    """Signed perp-vs-spot basis in bps. Raises on non-positive prices."""
    if perp_price <= 0 or spot_price <= 0:
        raise ValueError('prices must be positive')
    return (perp_price - spot_price) / spot_price * 10000.0


def hedge_legs(notional_usd: float, perp_price: float,
               spot_price: float) -> Tuple[float, float]:
    """Equal BASE-quantity legs (true delta-neutral): perp_qty == spot_qty.

    Quantity is sized off the perp price so perp notional == notional_usd;
    the spot leg uses the SAME base quantity (its USD notional differs by the
    basis, which is exactly what keeps base-asset delta at zero).
    """
    if notional_usd <= 0 or perp_price <= 0 or spot_price <= 0:
        raise ValueError('notional and prices must be positive')
    qty = notional_usd / perp_price
    return qty, qty


def legs_balanced(perp_qty: float, spot_qty: float,
                  epsilon_frac: float = 0.02) -> bool:
    """Reconciliation predicate: |perp_qty - spot_qty| / perp_qty <= epsilon.

    The operator/executor must assert this every tick on a live book;
    a False here means the book is leg-out (directional) and must be
    flattened, not resized.
    """
    if perp_qty <= 0:
        return False
    return abs(perp_qty - spot_qty) / perp_qty <= epsilon_frac


def plan_carry(
    *,
    symbol: str,
    venue: str,
    funding_bps: float,
    perp_price: float,
    spot_price: float,
    fees: FeeModel,
    funding_interval_hours: float = 8.0,
    horizon_intervals: float = 6.0,
    min_net_carry_bps: float = 10.0,
    max_breakeven_intervals: float = 3.0,
    allow_short_spot: bool = False,
) -> Optional[CarryPlan]:
    """Cost out the hedged structure; returns None only on invalid inputs.

    A returned plan always carries `actionable` + `reason`, so rejected
    structures stay observable (the shadow ledger records WHY we passed).
    """
    if perp_price <= 0 or spot_price <= 0 or funding_interval_hours <= 0:
        return None
    if not math.isfinite(funding_bps):
        return None

    basis_bps = compute_basis_bps(perp_price, spot_price)
    abs_funding = abs(funding_bps)

    if funding_bps > 0:
        direction, perp_side, spot_side = SHORT_PERP_LONG_SPOT, 'SELL', 'BUY'
        favorable_basis = basis_bps     # short perp gains if basis falls
        needs_borrow = False
    elif funding_bps < 0:
        direction, perp_side, spot_side = LONG_PERP_SHORT_SPOT, 'BUY', 'SELL'
        favorable_basis = -basis_bps    # long perp gains if basis rises
        needs_borrow = True
    else:
        direction, perp_side, spot_side = SHORT_PERP_LONG_SPOT, 'SELL', 'BUY'
        favorable_basis, needs_borrow = 0.0, False

    adverse_basis_cost_bps = max(0.0, -favorable_basis)
    round_trip = fees.round_trip_cost_bps()
    total_cost_bps = round_trip + adverse_basis_cost_bps
    breakeven = (total_cost_bps / abs_funding) if abs_funding > 0 else math.inf
    net_at_h = abs_funding * max(0.0, horizon_intervals) - total_cost_bps
    apr_gross_pct = (abs_funding / 10000.0
                     * (HOURS_PER_YEAR / funding_interval_hours) * 100.0)

    actionable, reason = True, 'carry_clears_gates'
    if abs_funding <= 0:
        actionable, reason = False, 'no_funding_edge'
    elif needs_borrow and not allow_short_spot:
        actionable, reason = False, 'needs_spot_borrow_disabled'
    elif breakeven > max_breakeven_intervals:
        actionable, reason = (
            False,
            f'breakeven {breakeven:.1f} > {max_breakeven_intervals:.1f} intervals')
    elif net_at_h < min_net_carry_bps:
        actionable, reason = (
            False,
            f'net {net_at_h:.1f}bps@{horizon_intervals:.0f}iv '
            f'< {min_net_carry_bps:.1f}bps')

    return CarryPlan(
        symbol=symbol, venue=venue, direction=direction,
        perp_side=perp_side, spot_side=spot_side,
        funding_bps=funding_bps,
        funding_interval_hours=funding_interval_hours,
        perp_price=perp_price,
        spot_price=spot_price,
        basis_bps=basis_bps,
        adverse_basis_cost_bps=adverse_basis_cost_bps,
        gross_carry_bps_per_interval=abs_funding,
        round_trip_cost_bps=round_trip,
        total_cost_bps=total_cost_bps,
        breakeven_intervals=breakeven,
        horizon_intervals=horizon_intervals,
        net_carry_bps_at_horizon=net_at_h,
        apr_gross_pct=apr_gross_pct,
        needs_borrow=needs_borrow,
        actionable=actionable,
        reason=reason,
    )


class ConfirmTracker:
    """Require N consecutive actionable evaluations per (venue, symbol)
    before a suggestion is recorded — the basis_desk analogue of funding-
    carry v2's stability window (one funding print can mean-revert before
    the first payment is even collected). Direction flips reset the streak.
    Pure in-memory; after restart the streak refills (no evidence -> no
    suggestion, fail-safe)."""

    def __init__(self, confirm_polls: int = 2):
        self.confirm_polls = max(1, int(confirm_polls))
        self._streaks: Dict[str, Tuple[str, int]] = {}

    def update(self, key: str, actionable: bool, direction: str) -> bool:
        """Feed one evaluation; True when the streak reaches confirm_polls."""
        if not actionable:
            self._streaks.pop(key, None)
            return False
        prev_dir, n = self._streaks.get(key, (direction, 0))
        n = n + 1 if prev_dir == direction else 1
        self._streaks[key] = (direction, n)
        return n >= self.confirm_polls


def _self_test() -> None:
    fees = FeeModel()  # 12 + 20 + 5 + 5 + 2 = 44 bps round trip
    assert abs(fees.round_trip_cost_bps() - 44.0) < 1e-9

    # 1. Positive funding -> SHORT perp + LONG spot, no borrow.
    p = plan_carry(symbol='BTCUSDT', venue='bybit', funding_bps=12.0,
                   perp_price=100.0, spot_price=100.0, fees=fees)
    assert p and p.direction == SHORT_PERP_LONG_SPOT
    assert p.perp_side == 'SELL' and p.spot_side == 'BUY'
    assert not p.needs_borrow
    # zero basis: total cost = 44; breakeven = 44/12 ~ 3.67 > 3 -> blocked.
    assert abs(p.total_cost_bps - 44.0) < 1e-9
    assert not p.actionable and 'breakeven' in p.reason

    # 2. Strong funding clears both gates: 25bps -> breakeven 1.76,
    #    net@6 = 150 - 44 = 106 bps.
    p2 = plan_carry(symbol='BTCUSDT', venue='bybit', funding_bps=25.0,
                    perp_price=100.0, spot_price=100.0, fees=fees)
    assert p2 and p2.actionable
    assert abs(p2.breakeven_intervals - 44.0 / 25.0) < 1e-9
    assert abs(p2.net_carry_bps_at_horizon - (150.0 - 44.0)) < 1e-9

    # 3. Favorable basis is NEVER credited (conservative): short perp with
    #    perp ABOVE spot (+20bps basis) -> same cost as zero basis.
    p3 = plan_carry(symbol='ETHUSDT', venue='bybit', funding_bps=25.0,
                    perp_price=100.20, spot_price=100.0, fees=fees)
    assert p3 and p3.adverse_basis_cost_bps == 0.0
    assert abs(p3.total_cost_bps - 44.0) < 1e-9

    # 4. Adverse basis IS charged: short perp with perp BELOW spot
    #    (-30bps basis) -> total cost 74, breakeven 74/25 ~ 2.96, net@6 = 76.
    p4 = plan_carry(symbol='ETHUSDT', venue='bybit', funding_bps=25.0,
                    perp_price=99.70, spot_price=100.0, fees=fees)
    assert p4 and p4.adverse_basis_cost_bps > 29.0
    assert p4.total_cost_bps > 73.0 and p4.actionable

    # 5. Negative funding -> LONG perp + SHORT spot; blocked without borrow.
    p5 = plan_carry(symbol='SOLUSDT', venue='binance', funding_bps=-30.0,
                    perp_price=100.0, spot_price=100.0, fees=fees)
    assert p5 and p5.direction == LONG_PERP_SHORT_SPOT and p5.needs_borrow
    assert not p5.actionable and p5.reason == 'needs_spot_borrow_disabled'
    p5b = plan_carry(symbol='SOLUSDT', venue='binance', funding_bps=-30.0,
                     perp_price=100.0, spot_price=100.0, fees=fees,
                     allow_short_spot=True)
    assert p5b and p5b.actionable
    # for LONG perp, perp BELOW spot is favorable -> not charged
    p5c = plan_carry(symbol='SOLUSDT', venue='binance', funding_bps=-30.0,
                     perp_price=99.70, spot_price=100.0, fees=fees,
                     allow_short_spot=True)
    assert p5c and p5c.adverse_basis_cost_bps == 0.0

    # 6. Zero funding -> never actionable; breakeven infinite.
    p6 = plan_carry(symbol='BTCUSDT', venue='bybit', funding_bps=0.0,
                    perp_price=100.0, spot_price=100.0, fees=fees)
    assert p6 and not p6.actionable and math.isinf(p6.breakeven_intervals)

    # 7. Invalid inputs -> None.
    assert plan_carry(symbol='X', venue='v', funding_bps=10.0, perp_price=0.0,
                      spot_price=1.0, fees=fees) is None
    assert plan_carry(symbol='X', venue='v', funding_bps=float('nan'),
                      perp_price=1.0, spot_price=1.0, fees=fees) is None

    # 8. APR sanity: 10bps (0.10%) every 8h = 1095 intervals/yr ~= 109.5%
    #    simple gross APR; the baseline 1bp print is ~10.95%.
    p8 = plan_carry(symbol='BTCUSDT', venue='bybit', funding_bps=10.0,
                    perp_price=100.0, spot_price=100.0, fees=fees)
    assert p8 and abs(p8.apr_gross_pct - 109.5) < 0.01

    # 9. Hedge legs: equal base quantity; balance predicate.
    pq, sq = hedge_legs(200.0, 50.0, 49.9)
    assert pq == sq == 4.0
    assert legs_balanced(4.0, 4.0)
    assert legs_balanced(4.0, 3.95, epsilon_frac=0.02)
    assert not legs_balanced(4.0, 3.0, epsilon_frac=0.02)
    assert not legs_balanced(0.0, 0.0)

    # 10. ConfirmTracker: streak builds, direction flip resets, non-
    #     actionable resets.
    t = ConfirmTracker(confirm_polls=2)
    assert not t.update('bybit:BTCUSDT', True, SHORT_PERP_LONG_SPOT)
    assert t.update('bybit:BTCUSDT', True, SHORT_PERP_LONG_SPOT)
    assert not t.update('bybit:BTCUSDT', True, LONG_PERP_SHORT_SPOT)  # flip
    assert not t.update('bybit:BTCUSDT', False, LONG_PERP_SHORT_SPOT)
    assert not t.update('bybit:BTCUSDT', True, LONG_PERP_SHORT_SPOT)
    assert t.update('bybit:BTCUSDT', True, LONG_PERP_SHORT_SPOT)

    print('basis_desk carry_math self-test OK')


if __name__ == '__main__':
    _self_test()
