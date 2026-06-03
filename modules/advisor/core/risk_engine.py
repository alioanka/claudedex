"""
AdvisorRiskEngine — advice-only risk gating.

This engine checks whether an advice result is "worth publishing" based on
operator-configured thresholds. It does NOT execute trades — it gates
whether an AdviceResult is stored and broadcast to the operator.

Gates (all configurable via advisor_config DB keys):
  - min_confidence      : float, default 0.35
  - max_sim_positions   : int,   default 20 — cap on open sim positions PER
                          MARKET/STRATEGY (issue #13). The caller (AdviceEngine)
                          passes the open-sim count for the CURRENT market only,
                          so this cap applies independently to crypto, BIST, US,
                          etc. (e.g. 10 each) rather than as one global cap.
  - blocked_symbols     : comma-sep list, default "" (blocklist)
  - horizon_filter      : "short,mid,long" or subset (default all)
"""

from __future__ import annotations

import logging
from typing import Optional

from modules.advisor.core.models import AdviceResult, DataSourceStatus, Direction

logger = logging.getLogger("advisor.risk_engine")


class AdvisorRiskEngine:
    """
    Risk gate for the advice pipeline.

    Methods
    -------
    should_publish(result) -> (bool, str)
        True + "" if advice passes all gates.
        False + reason string if rejected.
    """

    def __init__(self, config: dict):
        self.config = config

    @property
    def min_confidence(self) -> float:
        return float(self.config.get("min_confidence", 0.35))

    @property
    def max_sim_positions(self) -> int:
        return int(self.config.get("max_sim_positions", 20))

    @property
    def sim_cap_per_channel(self) -> int:
        """
        Per-CHANNEL open-sim cap (migration 077). Reads
        advisor_sim_cap_per_channel (default 15) and falls back to the legacy
        max_sim_positions key if the new key is absent (so a pre-077 config still
        enforces a sensible cap). Each channel (crypto, us_equities, bist, fx,
        midas_funds, gems, kap) is capped independently.
        """
        raw = self.config.get("advisor_sim_cap_per_channel")
        if raw is None or str(raw).strip() == "":
            return self.max_sim_positions
        try:
            return int(float(raw))
        except (TypeError, ValueError):
            return self.max_sim_positions

    @property
    def blocked_symbols(self) -> set:
        raw = self.config.get("blocked_symbols", "")
        return {s.strip().upper() for s in raw.split(",") if s.strip()}

    def should_publish(
        self,
        result: AdviceResult,
        open_sim_count: int = 0,
        min_confidence_override=None,
        channel: str = "",
    ) -> tuple[bool, str]:
        """
        Gate an AdviceResult through risk checks.

        Parameters
        ----------
        result                  : The AdviceResult candidate.
        open_sim_count          : Current count of open sim positions for THIS
                                  advice's CHANNEL (migration 077). The caller
                                  passes the channel-specific count so the cap is
                                  enforced per channel (gems/kap have their own).
        channel                 : The channel this advice's sim would open in
                                  (crypto|...|gems|kap). Used only for the reject
                                  message; defaults to the market value.
        min_confidence_override : When not None, use this confidence floor instead
                                  of self.min_confidence. Discovery ("New Gems")
                                  passes a lower floor (default 0.0) so trending
                                  candidates are SHOWN even at low confidence —
                                  the whole point is to surface them for review.

        Returns
        -------
        (True, "")            — passes all gates
        (False, reason_str)   — rejected; reason explains which gate fired
        """
        # 1. Data source must be available or degraded
        if result.data_source_status not in (
            DataSourceStatus.AVAILABLE,
            DataSourceStatus.DEGRADED,
        ):
            return False, f"data_source_status={result.data_source_status.value}"

        # 2. Confidence floor (overridable for discovery)
        floor = self.min_confidence if min_confidence_override is None else min_confidence_override
        if result.confidence < floor:
            return False, (
                f"confidence={result.confidence:.3f} < "
                f"min_confidence={floor:.3f}"
            )

        # 3. Symbol blocklist
        if result.symbol.upper() in self.blocked_symbols:
            return False, f"symbol={result.symbol} is in blocked_symbols"

        # 4. Sim position cap (only relevant when sim is enabled for this advice).
        # open_sim_count is the PER-CHANNEL open count (migration 077), so the cap
        # is enforced independently per channel — gems and kap have their own 15
        # slots and never consume crypto/us/fx slots.
        if result.sim_enabled and open_sim_count >= self.sim_cap_per_channel:
            ch = channel or result.market.value
            return False, (
                f"per-channel sim cap reached for {ch}: "
                f"{open_sim_count}/{self.sim_cap_per_channel}"
            )

        # 5. NEUTRAL direction is informational only — still publish
        # (operators may want to know "no clear signal")

        return True, ""
