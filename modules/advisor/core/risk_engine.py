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
    def blocked_symbols(self) -> set:
        raw = self.config.get("blocked_symbols", "")
        return {s.strip().upper() for s in raw.split(",") if s.strip()}

    def should_publish(
        self,
        result: AdviceResult,
        open_sim_count: int = 0,
    ) -> tuple[bool, str]:
        """
        Gate an AdviceResult through risk checks.

        Parameters
        ----------
        result          : The AdviceResult candidate.
        open_sim_count  : Current count of open sim positions (for cap check).

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

        # 2. Confidence floor
        if result.confidence < self.min_confidence:
            return False, (
                f"confidence={result.confidence:.3f} < "
                f"min_confidence={self.min_confidence:.3f}"
            )

        # 3. Symbol blocklist
        if result.symbol.upper() in self.blocked_symbols:
            return False, f"symbol={result.symbol} is in blocked_symbols"

        # 4. Sim position cap (only relevant when sim is enabled for this advice).
        # open_sim_count is the PER-MARKET open count (issue #13), so the cap is
        # enforced independently per market/strategy.
        if result.sim_enabled and open_sim_count >= self.max_sim_positions:
            return False, (
                f"per-market sim cap reached for {result.market.value}: "
                f"{open_sim_count}/{self.max_sim_positions}"
            )

        # 5. NEUTRAL direction is informational only — still publish
        # (operators may want to know "no clear signal")

        return True, ""
