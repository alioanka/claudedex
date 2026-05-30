"""Pure-function allocator: input per-module Sharpe + enabled flag,
output normalized allocation percentages.

Modified fractional-Kelly:
  raw_f_i = clamp(sharpe_i / MAX_SHARPE, 0, 1)   # disabled modules get 0
  alloc_i = max(MODULE_FLOOR_PCT, min(MODULE_CEILING_PCT,
                                       raw_f_i * (100 - RESERVE_PCT)))
  # Then normalize so sum(alloc_i) + RESERVE_PCT = 100

No DB calls. Testable with synthetic Sharpe inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


# Allocator constants. Operator can override at construction.
DEFAULT_MAX_SHARPE = 2.0            # Sharpe at which a module gets full weight
DEFAULT_MODULE_FLOOR_PCT = 5.0      # Minimum % per ENABLED module
DEFAULT_MODULE_CEILING_PCT = 40.0   # No single-module concentration risk
DEFAULT_RESERVE_PCT = 10.0          # Always-uninvested cushion


@dataclass
class AllocationInput:
    module: str
    enabled: bool
    sharpe: Optional[float]    # None when insufficient data — treated as 0


@dataclass
class AllocationProposal:
    module: str
    pct_of_book: float
    usd_amount: float
    reason: str
    raw_kelly_f: float
    components: dict = field(default_factory=dict)


@dataclass
class AllocationReport:
    proposals: List[AllocationProposal]
    reserve_pct: float
    total_book_usd: float
    notes: List[str] = field(default_factory=list)


def allocate(
    inputs: List[AllocationInput],
    total_book_usd: float,
    *,
    max_sharpe: float = DEFAULT_MAX_SHARPE,
    module_floor_pct: float = DEFAULT_MODULE_FLOOR_PCT,
    module_ceiling_pct: float = DEFAULT_MODULE_CEILING_PCT,
    reserve_pct: float = DEFAULT_RESERVE_PCT,
) -> AllocationReport:
    """Produce one AllocationProposal per ENABLED module. Disabled
    modules get an explicit 0% proposal so the dashboard can show
    them in the table.

    Math:
      1. raw_f_i = clamp(sharpe_i / max_sharpe, 0, 1)  for enabled modules
      2. Pre-normalize: each enabled module gets
         raw_f_i * (1 - reserve_pct/100) * 100  (so they share the
         non-reserve pie proportionally to Kelly fraction).
      3. Apply floor / ceiling.
      4. Re-normalize so the sum of enabled allocations equals
         100 - reserve_pct. If all modules had floor-only after step 3,
         this is the operative path.
    """
    enabled = [x for x in inputs if x.enabled]
    disabled = [x for x in inputs if not x.enabled]
    notes: List[str] = []

    if not enabled:
        notes.append("no enabled modules — reserve = 100%")
        return AllocationReport(
            proposals=[
                AllocationProposal(
                    module=x.module, pct_of_book=0.0, usd_amount=0.0,
                    reason="module disabled", raw_kelly_f=0.0,
                ) for x in inputs
            ],
            reserve_pct=100.0,
            total_book_usd=total_book_usd,
            notes=notes,
        )

    # Step 1: Kelly fraction per enabled module.
    raw_fs: Dict[str, float] = {}
    for x in enabled:
        sharpe_val = x.sharpe if x.sharpe is not None else 0.0
        raw = max(0.0, min(1.0, sharpe_val / max_sharpe))
        raw_fs[x.module] = raw

    sum_raw = sum(raw_fs.values())
    risk_budget = 100.0 - reserve_pct  # what's available for module allocations

    if sum_raw <= 0:
        # No module has positive Sharpe — give every enabled module
        # the floor and let the rest go to reserve.
        notes.append("all enabled modules have non-positive Sharpe; equal floor allocation")
        n = len(enabled)
        per = min(module_ceiling_pct, max(module_floor_pct, risk_budget / n))
        alloc_pcts = {x.module: per for x in enabled}
    else:
        # Step 2: proportional pre-normalize.
        pre = {m: (raw_fs[m] / sum_raw) * risk_budget for m in raw_fs}
        # Step 3: apply floor/ceiling.
        clipped = {m: max(module_floor_pct, min(module_ceiling_pct, v))
                   for m, v in pre.items()}
        # Step 4: re-normalize so the clipped sum equals risk_budget.
        clipped_sum = sum(clipped.values())
        if clipped_sum > 0:
            factor = risk_budget / clipped_sum
            alloc_pcts = {m: v * factor for m, v in clipped.items()}
        else:
            alloc_pcts = clipped
        # Re-clip after normalization — floor wins if normalization
        # pushed someone below it again (rare but possible).
        alloc_pcts = {m: max(module_floor_pct, min(module_ceiling_pct, v))
                      for m, v in alloc_pcts.items()}
        # One last sanity: if the post-clip sum exceeds the budget,
        # scale down proportionally so the books balance.
        total = sum(alloc_pcts.values())
        if total > risk_budget:
            factor = risk_budget / total
            alloc_pcts = {m: v * factor for m, v in alloc_pcts.items()}
            notes.append(
                f"post-clip sum {total:.2f}% > budget {risk_budget:.2f}%; "
                f"scaled by {factor:.3f}"
            )

    proposals: List[AllocationProposal] = []
    sharpe_lookup = {x.module: x.sharpe for x in inputs}
    for x in inputs:
        if x.enabled:
            pct = round(alloc_pcts.get(x.module, 0.0), 2)
            usd = round(pct / 100.0 * total_book_usd, 2)
            reason = (
                f"Sharpe={x.sharpe:.2f}, raw_kelly_f="
                f"{raw_fs.get(x.module, 0):.2f}, allocation={pct:.2f}%"
                if x.sharpe is not None else
                f"Sharpe=N/A, floor allocation={pct:.2f}%"
            )
            proposals.append(AllocationProposal(
                module=x.module, pct_of_book=pct, usd_amount=usd,
                reason=reason, raw_kelly_f=round(raw_fs.get(x.module, 0), 4),
                components={
                    "sharpe": sharpe_lookup.get(x.module),
                    "raw_kelly_f": round(raw_fs.get(x.module, 0), 4),
                    "module_floor_pct": module_floor_pct,
                    "module_ceiling_pct": module_ceiling_pct,
                },
            ))
        else:
            proposals.append(AllocationProposal(
                module=x.module, pct_of_book=0.0, usd_amount=0.0,
                reason="module disabled", raw_kelly_f=0.0,
            ))

    # Compute the actual reserve given the rounding above. If module
    # allocations exceed 100 - reserve_pct due to floor + few modules,
    # surface that as a note (reserve becomes 100 - sum).
    enabled_sum = sum(p.pct_of_book for p in proposals if p.module in alloc_pcts)
    actual_reserve = max(0.0, 100.0 - enabled_sum)
    if abs(actual_reserve - reserve_pct) > 0.5:
        notes.append(
            f"reserve adjusted: requested {reserve_pct:.1f}% → actual "
            f"{actual_reserve:.1f}% (due to floor on {len(enabled)} modules)"
        )

    return AllocationReport(
        proposals=proposals,
        reserve_pct=round(actual_reserve, 2),
        total_book_usd=total_book_usd,
        notes=notes,
    )
