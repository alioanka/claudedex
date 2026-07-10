"""Counterfactual replay engine.

Given trades + recommendations from trade_loader and a chosen
strategy from strategies, compute the per-module counterfactual P&L
that would have resulted if the strategy had been followed.

Pure compute. No DB writes. The output is a structured report the
dashboard can render directly.

Replay model:
  - Walk chronologically through the merged stream of (trade, rec)
    events.
  - Each module has a `live_at` timestamp (None initially = DRY_RUN).
  - When the strategy approves a `to_live` rec, set live_at = rec.ts.
  - When the strategy approves a `to_dry` rec or a circuit-breaker
    rec, clear live_at.
  - For each trade: if it happens after live_at → it's "what would
    have been live"; before → counterfactual is the DRY_RUN value
    (same as actual). The replay only DIFFERS from reality during
    windows where the operator's actual behaviour and the simulated
    strategy disagree.

Live-window cost haircut (Wave-F7 — external audit: "dry-run PnL
understates fees", and worse: the counterfactual NEVER differed from
actual because flip_ts was computed but unused). Trades that fall
inside the simulated "would have been live" window now subtract
`live_haircut_usd_per_trade` (strategy_params, default 0.5) from
their pnl — a flat per-trade estimate of the extra live costs
(slippage, gas variance, adverse selection, failed tx amortisation)
DRY_RUN never pays. Flat-USD because TradeRow deliberately carries no
notional; a bps-of-notional model is the documented follow-up (needs
per-table notional columns in trade_loader). Set the param to 0 to
reproduce the old cost-free counterfactual — the report always
surfaces `n_live_window_trades` so a zero-haircut run is visibly
optimistic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from .strategies import get_strategy
from .trade_loader import RecRow, TradeRow


@dataclass
class ModuleReplayResult:
    module: str
    n_trades: int = 0
    n_recs_seen: int = 0
    n_recs_approved: int = 0
    actual_pnl_usd: float = 0.0           # what actually happened
    counterfactual_pnl_usd: float = 0.0   # what the strategy would have produced
    pnl_delta_usd: float = 0.0            # counterfactual - actual
    max_drawdown_pct: float = 0.0
    sharpe: Optional[float] = None
    equity_curve: List[Tuple[str, float]] = field(default_factory=list)
    # Wave-F7: trades that fell in the simulated live window (haircut
    # applied) + the total haircut taken, so the cost model is auditable.
    n_live_window_trades: int = 0
    live_haircut_usd: float = 0.0


@dataclass
class ReplayReport:
    start_ts: str
    end_ts: str
    strategy: str
    strategy_params: dict
    per_module: List[ModuleReplayResult]
    total_actual_pnl_usd: float = 0.0
    total_counterfactual_pnl_usd: float = 0.0
    total_pnl_delta_usd: float = 0.0


def _sharpe(pnls: List[float]) -> Optional[float]:
    if len(pnls) < 5:
        return None
    mean = sum(pnls) / len(pnls)
    var = sum((x - mean) ** 2 for x in pnls) / (len(pnls) - 1)
    stdev = math.sqrt(var)
    if stdev == 0:
        return None
    return mean / stdev


def _max_drawdown_pct(equity: List[float]) -> float:
    """Equity is a cumulative P&L curve (starts at 0). Max drawdown
    is the worst peak-to-trough percentage decline. Returns positive
    %; 0 if no drawdown observed."""
    if len(equity) < 2:
        return 0.0
    peak = equity[0]
    max_dd = 0.0
    for v in equity:
        if v > peak:
            peak = v
        # Avoid divide-by-zero on a fresh book where peak is 0/near-0.
        denom = abs(peak) if abs(peak) > 1.0 else 1.0
        dd = (peak - v) / denom * 100.0
        if dd > max_dd:
            max_dd = dd
    return max_dd


def run_replay(
    trades_by_module: Dict[str, List[TradeRow]],
    recs: List[RecRow],
    strategy: str,
    strategy_params: Optional[dict] = None,
    start_ts: Optional[datetime] = None,
    end_ts: Optional[datetime] = None,
) -> ReplayReport:
    """Run the simulation. Inputs are from trade_loader; output is
    a ReplayReport. Pure function — same inputs → same outputs."""
    strategy_fn = get_strategy(strategy)
    params = strategy_params or {}
    if start_ts is None:
        all_ts = (
            [t.ts for trades in trades_by_module.values() for t in trades]
            + [r.ts for r in recs]
        )
        start_ts = min(all_ts) if all_ts else datetime.utcnow()
    if end_ts is None:
        all_ts = (
            [t.ts for trades in trades_by_module.values() for t in trades]
            + [r.ts for r in recs]
        )
        end_ts = max(all_ts) if all_ts else datetime.utcnow()

    # live_at: when the simulated strategy flipped this module to LIVE.
    # None = stayed in DRY for the whole window so far.
    live_at: Dict[str, Optional[datetime]] = {m: None for m in trades_by_module}

    # Walk recs in chronological order, updating live_at as we go.
    # We do NOT touch trades here; that's the second pass.
    n_seen: Dict[str, int] = {m: 0 for m in trades_by_module}
    n_approved: Dict[str, int] = {m: 0 for m in trades_by_module}
    for rec in recs:
        if rec.module not in trades_by_module:
            continue
        n_seen[rec.module] += 1
        if not strategy_fn(rec, params):
            continue
        n_approved[rec.module] += 1
        if rec.recommended == "to_live":
            if live_at[rec.module] is None:
                live_at[rec.module] = rec.ts
        elif rec.recommended == "to_dry":
            live_at[rec.module] = None
        # 'enable' / 'disable' / 'hold' have no effect on live state.

    # Build per-module result by walking trades in time order.
    per_module: List[ModuleReplayResult] = []
    total_actual = 0.0
    total_counter = 0.0
    for module, trades in trades_by_module.items():
        result = ModuleReplayResult(module=module)
        result.n_trades = len(trades)
        result.n_recs_seen = n_seen.get(module, 0)
        result.n_recs_approved = n_approved.get(module, 0)
        cum_actual = 0.0
        cum_counter = 0.0
        cum_counter_curve: List[float] = []
        per_trade_pnls: List[float] = []
        flip_ts = live_at.get(module)
        # Wave-F7 live-window cost haircut (see module docstring). Flat
        # USD per trade; default 0.5 keeps the counterfactual honestly
        # pessimistic vs cost-free DRY_RUN fills. 0 restores old behavior.
        try:
            haircut = max(0.0, float(params.get(
                "live_haircut_usd_per_trade", 0.5)))
        except (TypeError, ValueError):
            haircut = 0.5
        for t in trades:
            cum_actual += t.pnl_usd
            # Counterfactual: same pnl when DRY (no operator change);
            # in the simulated live window, subtract the per-trade live
            # cost estimate DRY_RUN fills never pay.
            counter_pnl = t.pnl_usd
            if flip_ts is not None and t.ts >= flip_ts and haircut > 0:
                counter_pnl -= haircut
                result.n_live_window_trades += 1
                result.live_haircut_usd += haircut
            cum_counter += counter_pnl
            cum_counter_curve.append(cum_counter)
            per_trade_pnls.append(counter_pnl)
        result.actual_pnl_usd = cum_actual
        result.counterfactual_pnl_usd = cum_counter
        result.pnl_delta_usd = cum_counter - cum_actual
        result.max_drawdown_pct = _max_drawdown_pct(cum_counter_curve)
        result.sharpe = _sharpe(per_trade_pnls)
        # Equity curve: sample 1 point per trade, cap at 200 for
        # response size sanity. Sparse-sample if more — use ceil
        # division on the step so length stays ≤ 200 even on prime
        # sample counts (300/200=1 in floor math would leave the
        # series un-sampled; ceil bumps to step=2 → 150 points).
        N_MAX = 200
        if len(cum_counter_curve) <= N_MAX:
            sampled = list(zip([t.ts for t in trades], cum_counter_curve))
        else:
            step = -(-len(cum_counter_curve) // N_MAX)  # ceil division
            sampled = [
                (trades[i].ts, cum_counter_curve[i])
                for i in range(0, len(cum_counter_curve), step)
            ]
        result.equity_curve = [(ts.isoformat(), v) for ts, v in sampled]
        per_module.append(result)
        total_actual += cum_actual
        total_counter += cum_counter

    return ReplayReport(
        start_ts=start_ts.isoformat() if start_ts else "",
        end_ts=end_ts.isoformat() if end_ts else "",
        strategy=strategy,
        strategy_params=params,
        per_module=per_module,
        total_actual_pnl_usd=total_actual,
        total_counterfactual_pnl_usd=total_counter,
        total_pnl_delta_usd=total_counter - total_actual,
    )
