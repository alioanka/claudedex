"""Replay strategies: given an orchestrator recommendation, decide
whether the strategy would have approved it.

Each strategy is a pure function taking a RecRow and returning a
bool (True = approve, False = reject). The replay engine calls one
of these for each rec in chronological order.

Adding a new strategy: implement a function with the signature
`(rec: RecRow, params: dict) -> bool`, then register it in
STRATEGY_FUNCS below. The dashboard's strategy-picker reads from
STRATEGY_FUNCS so a new strategy lights up automatically.
"""

from __future__ import annotations

from typing import Callable, Dict

from .trade_loader import RecRow


def approve_all(rec: RecRow, params: dict) -> bool:
    """The most aggressive: approve every non-'hold' recommendation.
    Used as the upper-bound counterfactual ("what's the most P&L the
    orchestrator could have made me if I'd trusted it blindly?")."""
    return rec.recommended != "hold"


def approve_on_confidence(rec: RecRow, params: dict) -> bool:
    """Approve only when confidence >= threshold. Closer to how an
    experienced operator behaves — they wait for the orchestrator to
    be confident before pulling the trigger."""
    threshold = float(params.get("confidence_threshold", 0.8))
    if rec.recommended == "hold":
        return False
    return rec.confidence >= threshold


def never_approve(rec: RecRow, params: dict) -> bool:
    """Baseline: the operator never approves anything. The replay's
    counterfactual P&L equals the DRY_RUN P&L (no live conversions).
    Useful as a control to confirm the simulator is well-calibrated."""
    return False


def operator_replay(rec: RecRow, params: dict) -> bool:
    """Replay exactly what the operator actually did. Approved recs
    fire; rejected/pending recs don't. This is the "what really
    happened" counterfactual — useful as ground truth in tests."""
    return rec.approved is True


# Registry. Dashboard exposes these keys directly via the strategy
# dropdown on /backtest-replay.
STRATEGY_FUNCS: Dict[str, Callable[[RecRow, dict], bool]] = {
    "approve_all": approve_all,
    "approve_on_confidence": approve_on_confidence,
    "never_approve": never_approve,
    "operator_replay": operator_replay,
}


def get_strategy(name: str) -> Callable[[RecRow, dict], bool]:
    """Returns the function for `name` or raises ValueError. Used by
    the replay engine so the strategy-selection failure surfaces at
    the API boundary, not deep in the engine."""
    if name not in STRATEGY_FUNCS:
        raise ValueError(
            f"unknown replay strategy {name!r}; valid: {list(STRATEGY_FUNCS)}"
        )
    return STRATEGY_FUNCS[name]
