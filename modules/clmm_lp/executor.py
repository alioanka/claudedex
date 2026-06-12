"""CLMM position recorder + GATED live path. Shadow/advisory by default.

Every proposal is recorded as a SIMULATED clmm_shadow_positions row
(is_simulated=true). A live mint/rebalance/burn would require EVERY gate
below to pass, in this exact order:

  1. shadow_mode (DB config)            must be False
  2. live_execution_enabled (DB config) must be True   (fail-safe default False)
  3. should_skip_live(module_dry_run, module='clmm_lp')
       -> False (NOT dry-run, NO logs/.killswitch, NO logs/.pause_clmm_lp)
  4. core.risk_manager.RiskManager.validate_trade(pool_address, size_usd) -> ok

Even with all gates green, v1 has NO live transaction path: it records the
row with skip_reason='live_path_not_implemented'. LP-ing below fair value
is a known way to lose money; live minting ships only after the shadow
track proves the net-of-IL model out-of-sample.
"""
import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("ClmmLpModule.Executor")


class ClmmExecutor:
    def __init__(self, db_pool, config: Dict[str, Any], risk_manager=None):
        self.db_pool = db_pool
        self.config = config or {}
        self.risk_manager = risk_manager
        self.simulated_records = 0
        self.live_orders = 0  # stays 0 in v1 by design

    async def _resolve_skip_reason(self, module_dry_run: bool,
                                   pool_address: str, size_usd: float) -> Optional[str]:
        """None iff a live action would be permitted. Never raises."""
        if self.config.get('shadow_mode', True):
            return 'shadow_mode'
        if not self.config.get('live_execution_enabled', False):
            return 'live_execution_disabled'
        try:
            from core.dry_run import should_skip_live
            if should_skip_live(module_dry_run, module='clmm_lp'):
                return 'dry_run_or_killswitch_or_pause'
        except Exception as e:
            logger.error("should_skip_live unavailable (%s) — refusing live", e)
            return 'dry_run_gate_unavailable'
        if self.risk_manager is None:
            return 'risk_manager_missing'
        try:
            ok, reason = await self.risk_manager.validate_trade(pool_address, size_usd)
            if not ok:
                return f"risk_manager:{reason}"[:64]
        except Exception as e:
            logger.error("RiskManager.validate_trade error: %s — refusing live", e)
            return 'risk_manager_error'
        return None

    async def open_position(self, *, proposal: Dict[str, Any],
                            module_dry_run: bool) -> Dict[str, Any]:
        """Record a proposed position; simulated unless every gate passes."""
        size_usd = min(float(proposal['size_usd']),
                       float(self.config.get('max_position_size_usd', 100.0)))
        skip_reason = await self._resolve_skip_reason(
            module_dry_run, proposal['pool_address'], size_usd)
        if skip_reason is None:
            # All gates green — v1 still refuses to mint (see module docstring).
            skip_reason = 'live_path_not_implemented'
            logger.warning("LIVE gates passed for %s but v1 has no mint path — "
                           "recording simulated", proposal['pool_address'])
        row_id = await self._insert(proposal, size_usd, skip_reason)
        return {'status': 'simulated', 'is_simulated': True,
                'skip_reason': skip_reason, 'position_id': row_id}

    async def _insert(self, p: Dict[str, Any], size_usd: float,
                      skip_reason: Optional[str]) -> Optional[int]:
        if not self.db_pool:
            return None
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    INSERT INTO clmm_shadow_positions
                        (chain, pool_address, pool_label, fee_rate_bps,
                         entry_price, price_lower, price_upper, size_usd,
                         amount0, amount1, liquidity,
                         expected_fee_apr, expected_il_apr, expected_rebalance_apr,
                         expected_net_apr, annual_vol_used, vol_source,
                         status, is_simulated, skip_reason,
                         current_price, fees_usd, il_usd, net_usd, details)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                            $16,$17,'open',TRUE,$18,$5,0,0,0,$19)
                    RETURNING id
                    """,
                    p['chain'], p['pool_address'], p.get('label', ''),
                    int(p['fee_rate_bps']), p['entry_price'],
                    p['price_lower'], p['price_upper'], size_usd,
                    p['amount0'], p['amount1'], p['liquidity'],
                    p['expected_fee_apr'], p['expected_il_apr'],
                    p['expected_rebalance_apr'], p['expected_net_apr'],
                    p['annual_vol_used'], p.get('vol_source', 'default'),
                    skip_reason, json.dumps(p.get('details') or {}, default=str),
                )
            self.simulated_records += 1
            return row['id'] if row else None
        except Exception as e:
            logger.error("clmm_shadow_positions insert failed: %s", e)
            return None
