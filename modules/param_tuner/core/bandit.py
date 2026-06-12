"""Pure UCB1 bandit math for the param tuner.

No DB calls, no async, no LLM, no randomness — UCB1 was chosen over
Thompson sampling precisely because it is DETERMINISTIC: the same state
always yields the same selection, so every proposal is reproducible by
the operator with a calculator and offline self-tests need no seeds.

Model: each registered tunable (one DB config_settings row with
operator-set [min, max] bounds) becomes one bandit over a discretized
grid of candidate values ("arms"). The engine feeds one reward per
observation window — the owning module's rolling performance score in
[0, 1] — attributed to the arm matching the value that was ACTUALLY
configured during that window. There is no counterfactual magic: arms
the operator never ran accumulate no evidence, and proposals to such
arms are honestly labeled kind='explore' (UCB optimism), never
kind='exploit'.

Self-test: ``python -m modules.param_tuner.core.bandit``
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

KIND_EXPLOIT = "exploit"   # best arm has real evidence and beats current
KIND_EXPLORE = "explore"   # best arm is under-sampled; UCB optimism only


@dataclass
class ArmState:
    """One discretized candidate value of a tunable."""

    value: float
    pulls: int = 0
    reward_sum: float = 0.0
    reward_sq_sum: float = 0.0   # kept for variance diagnostics

    @property
    def mean(self) -> Optional[float]:
        return None if self.pulls <= 0 else self.reward_sum / self.pulls


@dataclass
class BanditState:
    """Full bandit state for one tunable. JSON-serializable round-trip."""

    key_id: str                  # '<config_type>/<key>' (display only)
    value_type: str = "float"    # 'float' | 'int'
    arms: List[ArmState] = field(default_factory=list)
    total_pulls: int = 0


@dataclass
class Proposal:
    """A proposed value change. Pure output — persisting it is the
    engine's job; nothing here writes anything anywhere."""

    kind: str                    # KIND_EXPLOIT | KIND_EXPLORE
    current_value: float
    proposed_value: float
    current_mean: Optional[float]
    proposed_mean: Optional[float]
    proposed_pulls: int
    ucb_score: float
    reason: str


# ---------------------------------------------------------------- grid

def make_grid(lo: float, hi: float, steps: int, value_type: str = "float") -> List[float]:
    """Inclusive linear grid of `steps` values over [lo, hi]. Ints are
    rounded then deduped (preserving order). Deterministic."""
    if hi < lo:
        lo, hi = hi, lo
    steps = max(2, int(steps))
    raw = [lo + (hi - lo) * i / (steps - 1) for i in range(steps)]
    if value_type == "int":
        seen: List[float] = []
        for v in (float(round(x)) for x in raw):
            if v not in seen:
                seen.append(v)
        return seen
    return [round(x, 10) for x in raw]


def clamp(value: float, lo: float, hi: float) -> float:
    """Hard bound — every value that leaves this module passes through
    here. The operator's [min, max] is inviolable."""
    if hi < lo:
        lo, hi = hi, lo
    return min(hi, max(lo, value))


def nearest_arm_index(state: BanditState, value: float) -> int:
    """Index of the arm closest to `value` (first wins on exact ties)."""
    if not state.arms:
        raise ValueError(f"bandit {state.key_id} has no arms")
    best_i, best_d = 0, abs(state.arms[0].value - value)
    for i, arm in enumerate(state.arms[1:], start=1):
        d = abs(arm.value - value)
        if d < best_d:
            best_i, best_d = i, d
    return best_i


def new_state(key_id: str, lo: float, hi: float, steps: int,
              value_type: str = "float") -> BanditState:
    grid = make_grid(lo, hi, steps, value_type)
    return BanditState(
        key_id=key_id,
        value_type=value_type,
        arms=[ArmState(value=v) for v in grid],
    )


# ---------------------------------------------------------------- core math

def update(state: BanditState, observed_value: float, reward: float) -> int:
    """Record one reward observation for the arm nearest the value that
    was actually configured. Reward is clamped to [0, 1] so a buggy
    upstream score can never blow up the means. Returns the arm index."""
    r = min(1.0, max(0.0, float(reward)))
    i = nearest_arm_index(state, observed_value)
    arm = state.arms[i]
    arm.pulls += 1
    arm.reward_sum += r
    arm.reward_sq_sum += r * r
    state.total_pulls += 1
    return i


def ucb_score(arm: ArmState, total_pulls: int, c: float) -> float:
    """UCB1: mean + c * sqrt(2 ln N / n). Unpulled arms get +inf so each
    arm is tried (in grid order) before any exploitation."""
    if arm.pulls <= 0:
        return math.inf
    bonus = c * math.sqrt(2.0 * math.log(max(2, total_pulls)) / arm.pulls)
    return (arm.reward_sum / arm.pulls) + bonus


def select_arm(state: BanditState, c: float) -> int:
    """Deterministic argmax of UCB scores; first index wins ties (so the
    same state always selects the same arm)."""
    best_i, best_s = 0, -math.inf
    for i, arm in enumerate(state.arms):
        s = ucb_score(arm, state.total_pulls, c)
        if s > best_s:
            best_i, best_s = i, s
    return best_i


def propose(state: BanditState, current_value: float, *,
            c: float, min_pulls_exploit: int,
            improvement_margin: float,
            lo: float, hi: float) -> Optional[Proposal]:
    """Turn bandit state into at most one proposal.

    Rules, in order:
      1. UCB-select the best arm. If it IS the current arm -> no proposal.
      2. If the best arm has >= min_pulls_exploit observations AND its
         mean reward beats the current arm's mean by improvement_margin
         (absolute, rewards live in [0,1]) -> kind='exploit'.
      3. Else if the best arm is under-sampled -> kind='explore' (the
         honest label: this is UCB optimism, not evidence). Explore
         proposals are NEVER auto-applied by the engine.
      4. Anything else -> no proposal.
    The proposed value is clamped to the operator's [lo, hi] bounds.
    """
    if not state.arms:
        return None
    cur_i = nearest_arm_index(state, current_value)
    best_i = select_arm(state, c)
    if best_i == cur_i:
        return None

    cur = state.arms[cur_i]
    best = state.arms[best_i]
    proposed_value = clamp(best.value, lo, hi)
    if state.value_type == "int":
        proposed_value = float(round(clamp(proposed_value, lo, hi)))
    if proposed_value == cur.value:
        return None
    score = ucb_score(best, state.total_pulls, c)

    if best.pulls >= max(1, min_pulls_exploit) and best.mean is not None:
        cur_mean = cur.mean if cur.mean is not None else 0.0
        gain = best.mean - cur_mean
        if gain >= improvement_margin:
            return Proposal(
                kind=KIND_EXPLOIT,
                current_value=cur.value, proposed_value=proposed_value,
                current_mean=cur.mean, proposed_mean=best.mean,
                proposed_pulls=best.pulls, ucb_score=score,
                reason=(
                    f"EXPLOIT: value {proposed_value:g} averaged reward "
                    f"{best.mean:.3f} over {best.pulls} windows vs "
                    f"{cur_mean:.3f} at current {cur.value:g} "
                    f"(gain {gain:.3f} >= margin {improvement_margin:.3f})."
                ),
            )
        return None  # evidence exists but the edge is inside the margin

    return Proposal(
        kind=KIND_EXPLORE,
        current_value=cur.value, proposed_value=proposed_value,
        current_mean=cur.mean, proposed_mean=best.mean,
        proposed_pulls=best.pulls, ucb_score=score,
        reason=(
            f"EXPLORE: value {proposed_value:g} has only {best.pulls} "
            f"observation(s) (< {min_pulls_exploit}); UCB optimism ranks it "
            f"first. This is NOT evidence of improvement — run it in "
            f"DRY_RUN before trusting it."
        ),
    )


# ---------------------------------------------------------------- (de)serialize

def to_dict(state: BanditState) -> dict:
    return {
        "key_id": state.key_id,
        "value_type": state.value_type,
        "total_pulls": state.total_pulls,
        "arms": [
            {"value": a.value, "pulls": a.pulls,
             "reward_sum": a.reward_sum, "reward_sq_sum": a.reward_sq_sum}
            for a in state.arms
        ],
    }


def from_dict(d: dict) -> BanditState:
    return BanditState(
        key_id=str(d.get("key_id", "")),
        value_type=str(d.get("value_type", "float")),
        total_pulls=int(d.get("total_pulls", 0)),
        arms=[
            ArmState(
                value=float(a["value"]), pulls=int(a.get("pulls", 0)),
                reward_sum=float(a.get("reward_sum", 0.0)),
                reward_sq_sum=float(a.get("reward_sq_sum", 0.0)),
            )
            for a in d.get("arms", [])
        ],
    )


def reconcile_grid(state: BanditState, lo: float, hi: float, steps: int,
                   value_type: str) -> BanditState:
    """If the operator changed bounds/steps in the registry, rebuild the
    grid and carry evidence over by nearest-arm mapping; out-of-bounds
    evidence is dropped (its value can no longer be configured)."""
    fresh = new_state(state.key_id, lo, hi, steps, value_type)
    for old in state.arms:
        if old.pulls <= 0:
            continue
        if old.value < min(lo, hi) - 1e-12 or old.value > max(lo, hi) + 1e-12:
            continue
        i = nearest_arm_index(fresh, old.value)
        arm = fresh.arms[i]
        arm.pulls += old.pulls
        arm.reward_sum += old.reward_sum
        arm.reward_sq_sum += old.reward_sq_sum
        fresh.total_pulls += old.pulls
    return fresh


# ---------------------------------------------------------------- self-test

def _selftest() -> None:
    # 1. Grid construction: inclusive, deterministic, int dedupe.
    g = make_grid(0.0, 1.0, 5)
    assert g == [0.0, 0.25, 0.5, 0.75, 1.0], g
    gi = make_grid(1, 3, 5, "int")
    assert gi == [1.0, 2.0, 3.0], gi    # rounding collapses 1.5/2.5
    assert make_grid(1.0, 0.0, 3) == [0.0, 0.5, 1.0]  # swapped bounds OK

    # 2. Clamp is inviolable.
    assert clamp(99.0, 0.0, 1.0) == 1.0 and clamp(-1.0, 0.0, 1.0) == 0.0

    # 3. Fresh state proposes EXPLORE (unpulled arm, never exploit).
    st = new_state("t/x", 0.0, 1.0, 5)
    p = propose(st, 0.5, c=0.5, min_pulls_exploit=3,
                improvement_margin=0.05, lo=0.0, hi=1.0)
    assert p is not None and p.kind == KIND_EXPLORE, p

    # 4. Reward updates land on the nearest arm and clamp to [0,1].
    i = update(st, 0.49, 0.6)
    assert st.arms[i].value == 0.5 and st.arms[i].pulls == 1
    update(st, 0.5, 5.0)   # clamped to 1.0
    assert st.arms[i].reward_sum == 1.6, st.arms[i]

    # 5. Exploit fires only with evidence + margin; clamped to bounds.
    st = new_state("t/y", 0.0, 1.0, 3)   # arms 0.0 / 0.5 / 1.0
    for _ in range(5):
        update(st, 0.5, 0.30)    # current arm: mediocre
        update(st, 1.0, 0.80)    # alternative: strong
        update(st, 0.0, 0.10)    # alternative: bad
    p = propose(st, 0.5, c=0.1, min_pulls_exploit=3,
                improvement_margin=0.05, lo=0.0, hi=1.0)
    assert p is not None and p.kind == KIND_EXPLOIT, p
    assert p.proposed_value == 1.0 and p.proposed_mean is not None
    assert abs(p.proposed_mean - 0.80) < 1e-9, p

    # 6. Determinism: identical state -> identical proposal, every time.
    p2 = propose(st, 0.5, c=0.1, min_pulls_exploit=3,
                 improvement_margin=0.05, lo=0.0, hi=1.0)
    assert p2 == p, (p, p2)

    # 7. Inside-margin edge -> NO proposal (anti-churn).
    st2 = new_state("t/z", 0.0, 1.0, 3)
    for _ in range(5):
        update(st2, 0.5, 0.50)
        update(st2, 1.0, 0.52)   # +0.02 < margin 0.05
        update(st2, 0.0, 0.10)
    p = propose(st2, 0.5, c=0.0, min_pulls_exploit=3,
                improvement_margin=0.05, lo=0.0, hi=1.0)
    assert p is None, p

    # 8. Current arm best -> no proposal.
    st3 = new_state("t/w", 0.0, 1.0, 3)
    for _ in range(5):
        update(st3, 0.5, 0.90)
        update(st3, 0.0, 0.10)
        update(st3, 1.0, 0.10)
    assert propose(st3, 0.5, c=0.1, min_pulls_exploit=3,
                   improvement_margin=0.05, lo=0.0, hi=1.0) is None

    # 9. Serialization round-trip is lossless.
    rt = from_dict(to_dict(st))
    assert to_dict(rt) == to_dict(st)

    # 10. Grid reconcile carries evidence; out-of-bounds evidence drops.
    nar = reconcile_grid(st, 0.0, 0.5, 3, "float")   # arms 0.0/0.25/0.5
    assert nar.arms[-1].value == 0.5 and nar.arms[-1].pulls == 5
    assert nar.total_pulls == 10, nar.total_pulls    # the 1.0 arm dropped

    # 11. Int tunable proposals are integral (all arms sampled so the
    #     unpulled-arm UCB priority does not preempt the exploit pick).
    sti = new_state("t/i", 10, 60, 6, "int")
    for _ in range(4):
        for v in (10, 20, 30, 40, 50):
            update(sti, v, 0.2)
        update(sti, 60, 0.9)
    p = propose(sti, 10, c=0.0, min_pulls_exploit=3,
                improvement_margin=0.05, lo=10, hi=60)
    assert p is not None and p.proposed_value == 60.0 and \
        p.proposed_value == round(p.proposed_value), p

    print("bandit self-test OK")


if __name__ == "__main__":
    _selftest()
