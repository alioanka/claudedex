"""
Funding-Carry v2 planner (FUT-QC-01) — bidirectional, stability-gated.

Edge source (one sentence): persistent perp funding is a risk premium paid
by the crowded side of the book (over-leveraged longs in euphoria, panicked
shorts in capitulation); taking the opposite side harvests that premium
without a price-direction view.

How v2 differs from FUT-RM-25 (the existing carry scan in futures_engine):
  1. BIDIRECTIONAL — significantly NEGATIVE funding (shorts pay longs)
     yields a LONG-perp entry; FUT-RM-25 is SHORT-only.
  2. STABILITY WINDOW — entry requires the funding rate to be persistently
     beyond the threshold across a rolling time window (multiple samples,
     consistent sign, every sample beyond the threshold). FUT-RM-25 fires
     on a single snapshot, which is exposed to one-interval spikes that
     mean-revert before the first funding payment is even collected.
  3. ABS thresholds — a single `min_abs_funding_bps` applies to both signs.

Cost model (per 8h funding interval, Bybit taker both legs):
    gross_carry_bps_per_interval = |funding_bps|
    round_trip_cost_bps          = 2 * taker_fee_bps + slippage_bps
                                 ~ 2 * 6 + 5 = 17 bps (Bybit 0.06% taker)
    breakeven_intervals          = round_trip_cost_bps / |funding_bps|
At the default 10 bps threshold, breakeven is ~1.7 intervals (~14h), which
is why the carry hold cap (carry_max_hold_minutes, default 960 = 2
intervals) and the stability gate (so the rate is unlikely to decay before
breakeven) matter.

Delta-neutrality note for the operator: the perp leg alone is directional.
The textbook trade pairs a SHORT perp with an equal-notional SPOT BUY (or a
LONG perp with spot sell/borrow). Spot is NOT wired in this module, so the
hedge leg is a MANUAL operator action; unhedged, the position relies on the
engine's normal SL/TP/max-hold to bound price risk. Run in DRY_RUN first
and observe `[carry-v2]` log lines to measure entry quality.

This file is PURE LOGIC (no I/O, no engine imports) so it can be unit-
tested offline: `python -m modules.futures_trading.strategies.funding_carry`
runs the embedded self-test.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque, Dict, Optional, Tuple


@dataclass
class CarryDecision:
    """Outcome of an entry evaluation. `side` is 'SHORT'|'LONG'|None."""
    enter: bool
    side: Optional[str]
    reason: str
    mean_bps: float = 0.0
    min_abs_bps: float = 0.0
    samples: int = 0
    span_minutes: float = 0.0


@dataclass
class FundingCarryPlanner:
    """Rolling per-symbol funding-rate window + entry/exit decisions.

    All state is in-memory; after a restart the window must refill before a
    new entry can arm (fail-safe: no evidence -> no trade). Exits do NOT
    depend on the window (they use the live rate), so restart never strands
    a position.
    """

    min_abs_funding_bps: float = 10.0
    stability_window_minutes: int = 240
    min_samples: int = 4
    exit_abs_funding_bps: float = 3.0
    # Minimum spacing between recorded samples. Aligned with the engine's
    # funding-rate cache TTL (300s) so we never record the same cached
    # reading twice and fake "stability" out of one snapshot.
    sample_spacing_seconds: int = 300
    # Fraction of the window that must be covered by samples before the
    # gate can arm — prevents 4 samples in 20 minutes from passing a
    # 240-minute persistence requirement.
    min_span_fraction: float = 0.8

    _windows: Dict[str, Deque[Tuple[datetime, float]]] = field(
        default_factory=dict)

    # ------------------------------------------------------------------
    # Recording
    # ------------------------------------------------------------------

    def record(self, symbol: str, rate_bps: float,
               now: Optional[datetime] = None) -> bool:
        """Append a funding sample; returns True if recorded (spacing ok)."""
        now = now or datetime.utcnow()
        dq = self._windows.setdefault(symbol, deque())
        if dq and (now - dq[-1][0]).total_seconds() < self.sample_spacing_seconds:
            return False
        dq.append((now, float(rate_bps)))
        self._prune(dq, now)
        return True

    def _prune(self, dq: Deque[Tuple[datetime, float]], now: datetime) -> None:
        cutoff = now - timedelta(minutes=self.stability_window_minutes)
        while dq and dq[0][0] < cutoff:
            dq.popleft()

    # ------------------------------------------------------------------
    # Entry decision
    # ------------------------------------------------------------------

    def evaluate_entry(self, symbol: str,
                       now: Optional[datetime] = None) -> CarryDecision:
        """Stability-gated entry decision.

        Requirements (ALL must hold):
          - at least `min_samples` samples inside the window
          - samples span >= min_span_fraction * window (true persistence)
          - every sample has the SAME sign (no flip inside the window)
          - every sample has |rate| >= min_abs_funding_bps (the WEAKEST
            sample clears the bar, not just the average — conservative)
        Positive funding -> longs pay shorts -> enter SHORT.
        Negative funding -> shorts pay longs -> enter LONG.
        """
        now = now or datetime.utcnow()
        dq = self._windows.get(symbol)
        if dq:
            self._prune(dq, now)
        if not dq:
            return CarryDecision(False, None, 'no_samples')
        n = len(dq)
        if n < max(1, self.min_samples):
            return CarryDecision(False, None, f'samples {n}<{self.min_samples}',
                                 samples=n)
        span_min = (dq[-1][0] - dq[0][0]).total_seconds() / 60.0
        need_span = self.min_span_fraction * self.stability_window_minutes
        if span_min < need_span:
            return CarryDecision(
                False, None,
                f'span {span_min:.0f}min < {need_span:.0f}min',
                samples=n, span_minutes=span_min)
        rates = [r for _, r in dq]
        pos = all(r > 0 for r in rates)
        neg = all(r < 0 for r in rates)
        if not (pos or neg):
            return CarryDecision(False, None, 'sign_flip_in_window',
                                 samples=n, span_minutes=span_min)
        min_abs = min(abs(r) for r in rates)
        mean = sum(rates) / n
        if min_abs < self.min_abs_funding_bps:
            return CarryDecision(
                False, None,
                f'weakest |{min_abs:.2f}|bps < {self.min_abs_funding_bps:.2f}bps',
                mean_bps=mean, min_abs_bps=min_abs, samples=n,
                span_minutes=span_min)
        side = 'SHORT' if pos else 'LONG'
        return CarryDecision(
            True, side,
            f'stable {"+" if pos else "-"}funding, weakest |{min_abs:.2f}|bps',
            mean_bps=mean, min_abs_bps=min_abs, samples=n,
            span_minutes=span_min)

    # ------------------------------------------------------------------
    # Exit decision
    # ------------------------------------------------------------------

    def should_exit(self, carry_side: str,
                    rate_bps: Optional[float]) -> Tuple[bool, str]:
        """Exit when the carry edge is gone: the live rate no longer pays
        our side by at least exit_abs_funding_bps. None rate -> hold
        (fail-open on missing data; SL/TP/max-hold still bound the trade)."""
        if rate_bps is None:
            return False, 'rate_unavailable'
        side = (carry_side or '').upper()
        if side == 'SHORT':
            # Short collects when funding is positive.
            if rate_bps < self.exit_abs_funding_bps:
                return True, (
                    f'funding {rate_bps:+.2f}bps < +{self.exit_abs_funding_bps:.2f}bps')
        elif side == 'LONG':
            # Long collects when funding is negative.
            if rate_bps > -self.exit_abs_funding_bps:
                return True, (
                    f'funding {rate_bps:+.2f}bps > -{self.exit_abs_funding_bps:.2f}bps')
        return False, 'edge_intact'

    # ------------------------------------------------------------------
    # Economics helper (for logging / DRY_RUN measurability)
    # ------------------------------------------------------------------

    @staticmethod
    def expected_net_carry_bps(abs_funding_bps: float, intervals_held: float,
                               taker_fee_bps: float = 6.0,
                               slippage_bps: float = 5.0) -> float:
        """Net carry on notional, in bps: gross funding collected minus the
        round-trip taker fees and modeled slippage."""
        gross = abs_funding_bps * max(0.0, intervals_held)
        cost = 2.0 * taker_fee_bps + slippage_bps
        return gross - cost


def _self_test() -> None:
    base = datetime(2026, 1, 1, 0, 0, 0)

    # 1. Spacing: duplicate cached reading inside spacing is rejected.
    p = FundingCarryPlanner(min_abs_funding_bps=10, stability_window_minutes=240,
                            min_samples=4)
    assert p.record('BTC/USDT', 12.0, base)
    assert not p.record('BTC/USDT', 12.0, base + timedelta(seconds=30))
    assert p.record('BTC/USDT', 12.0, base + timedelta(seconds=300))

    # 2. Too few samples / too little span -> no entry.
    d = p.evaluate_entry('BTC/USDT', base + timedelta(minutes=10))
    assert not d.enter, d

    # 3. Persistent positive funding over the window -> SHORT.
    p2 = FundingCarryPlanner(min_abs_funding_bps=10, stability_window_minutes=240,
                             min_samples=4)
    for i in range(5):
        p2.record('ETH/USDT', 11.0 + i * 0.5, base + timedelta(minutes=50 * i))
    d = p2.evaluate_entry('ETH/USDT', base + timedelta(minutes=200))
    assert d.enter and d.side == 'SHORT', d

    # 4. Persistent negative funding -> LONG.
    p3 = FundingCarryPlanner(min_abs_funding_bps=10, stability_window_minutes=240,
                             min_samples=4)
    for i in range(5):
        p3.record('SOL/USDT', -13.0, base + timedelta(minutes=50 * i))
    d = p3.evaluate_entry('SOL/USDT', base + timedelta(minutes=200))
    assert d.enter and d.side == 'LONG', d

    # 5. A single spike sample below threshold blocks entry (weakest-sample rule).
    p4 = FundingCarryPlanner(min_abs_funding_bps=10, stability_window_minutes=240,
                             min_samples=4)
    for i, r in enumerate([15.0, 14.0, 4.0, 16.0, 15.0]):
        p4.record('XRP/USDT', r, base + timedelta(minutes=50 * i))
    d = p4.evaluate_entry('XRP/USDT', base + timedelta(minutes=200))
    assert not d.enter and 'weakest' in d.reason, d

    # 6. Sign flip inside window blocks entry.
    p5 = FundingCarryPlanner(min_abs_funding_bps=10, stability_window_minutes=240,
                             min_samples=4)
    for i, r in enumerate([15.0, 14.0, -12.0, 16.0, 15.0]):
        p5.record('ADA/USDT', r, base + timedelta(minutes=50 * i))
    d = p5.evaluate_entry('ADA/USDT', base + timedelta(minutes=200))
    assert not d.enter and d.reason == 'sign_flip_in_window', d

    # 7. Old samples roll out of the window.
    p6 = FundingCarryPlanner(min_abs_funding_bps=10, stability_window_minutes=240,
                             min_samples=4)
    p6.record('BNB/USDT', 12.0, base)
    p6.record('BNB/USDT', 12.0, base + timedelta(minutes=500))
    d = p6.evaluate_entry('BNB/USDT', base + timedelta(minutes=500))
    assert not d.enter and d.samples == 1, d

    # 8. Exit logic, both sides.
    pl = FundingCarryPlanner(exit_abs_funding_bps=3.0)
    assert pl.should_exit('SHORT', 2.0)[0]          # decayed
    assert not pl.should_exit('SHORT', 8.0)[0]      # intact
    assert pl.should_exit('SHORT', -5.0)[0]         # flipped against us
    assert pl.should_exit('LONG', -2.0)[0]          # decayed
    assert not pl.should_exit('LONG', -8.0)[0]      # intact
    assert pl.should_exit('LONG', 5.0)[0]           # flipped against us
    assert not pl.should_exit('SHORT', None)[0]     # missing data -> hold

    # 9. Economics: 10 bps for 2 intervals beats 17 bps round-trip cost.
    net = FundingCarryPlanner.expected_net_carry_bps(10.0, 2.0)
    assert abs(net - 3.0) < 1e-9, net
    assert FundingCarryPlanner.expected_net_carry_bps(10.0, 1.0) < 0

    print('funding_carry self-test OK')


if __name__ == '__main__':
    _self_test()
