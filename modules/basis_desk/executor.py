"""basis_desk suggestion recorder + GATED (and intentionally unwired) live path.

Default behaviour is ALWAYS an advisory/simulated record into
basis_carry_suggestions (is_simulated=true). The module ADVISES the complete
hedged structure (perp leg + spot hedge leg); it does NOT place orders.

LIVE GATE CHAIN — evaluated in this exact order so the recorded skip_reason
is always the FIRST failed gate (mirrors modules/polymarket/executor.py):

  1. shadow_mode (DB config)            must be False  (default True)
  2. live_execution_enabled (DB config) must be True   (default False)
  3. should_skip_live(module_dry_run, module='basis_desk')
       -> False  (i.e. NOT dry-run, NO logs/.killswitch, NO logs/.pause_basis_desk)
  4. core.risk_manager.RiskManager.validate_trade(symbol, notional) -> (True, _)
  5. live order placement — NOT IMPLEMENTED BY DESIGN. Even with gates 1-4
     green the row records status='live_blocked',
     skip_reason='live_path_not_implemented'. Wiring real two-leg execution
     (perp via the futures module, spot via Bybit V5 spot) is a separate,
     explicitly-reviewed change; this module stays an advisor.

Any gate failure records the suggestion with that skip_reason and returns.
Fail-soft everywhere: a DB or gate error never crashes the engine loop.
"""

import json
import logging
import math
from typing import Any, Dict, Optional

from modules.basis_desk.carry_math import CarryPlan, hedge_legs

logger = logging.getLogger("BasisDeskModule.Executor")


class BasisDeskExecutor:
    def __init__(self, db_pool, config: Dict[str, Any], risk_manager=None):
        self.db_pool = db_pool
        self.config = config or {}
        self.risk_manager = risk_manager
        self.suggestions_recorded = 0
        self.live_orders = 0  # stays 0 by design (live path not implemented)

    async def _resolve_skip_reason(self, module_dry_run: bool,
                                   symbol: str, notional_usd: float) -> str:
        """Returns the FIRST failed live gate; never returns None because the
        final gate (live order wiring) is intentionally not implemented."""
        if self.config.get('shadow_mode', True):
            return 'shadow_mode'
        if not self.config.get('live_execution_enabled', False):
            return 'live_execution_disabled'
        try:
            from core.dry_run import should_skip_live
            if should_skip_live(module_dry_run, module='basis_desk'):
                return 'dry_run_or_killswitch_or_pause'
        except Exception as e:
            logger.error('should_skip_live unavailable (%s) — refusing live', e)
            return 'dry_run_gate_unavailable'
        if self.risk_manager is None:
            return 'risk_manager_missing'
        try:
            ok, reason = await self.risk_manager.validate_trade(symbol,
                                                                notional_usd)
            if not ok:
                return f'risk_manager:{reason}'[:64]
        except Exception as e:
            logger.error('RiskManager.validate_trade error: %s — refusing live', e)
            return 'risk_manager_error'
        return 'live_path_not_implemented'

    async def advise(self, plan: CarryPlan, *, module_dry_run: bool,
                     extra_details: Optional[Dict[str, Any]] = None
                     ) -> Dict[str, Any]:
        """Record the complete hedged structure as an advisory row."""
        notional = min(
            float(self.config.get('max_notional_usd', 200.0)),
            float(self.config.get('suggest_notional_usd', 200.0)),
        )
        try:
            perp_qty, spot_qty = hedge_legs(notional, plan.perp_price,
                                            plan.spot_price)
        except Exception:
            perp_qty = spot_qty = None

        skip_reason = await self._resolve_skip_reason(
            module_dry_run, plan.symbol, notional)
        status = ('live_blocked' if skip_reason == 'live_path_not_implemented'
                  else 'advised')

        details = {
            'reason': plan.reason,
            'hedge_epsilon_frac': float(
                self.config.get('hedge_epsilon_frac', 0.02)),
            'operator_note': (
                'Place BOTH legs or NEITHER. Reconcile every tick: '
                '|perp_qty - spot_qty| / perp_qty <= hedge_epsilon_frac, '
                'else flatten (leg-out book is directional).'),
            **(extra_details or {}),
        }
        row = dict(
            venue=plan.venue, symbol=plan.symbol, direction=plan.direction,
            perp_side=plan.perp_side, spot_side=plan.spot_side,
            funding_bps=plan.funding_bps,
            funding_interval_hours=plan.funding_interval_hours,
            perp_price=plan.perp_price, spot_price=plan.spot_price,
            basis_bps=plan.basis_bps,
            perp_qty=perp_qty, spot_qty=spot_qty, notional_usd=notional,
            gross_carry_bps_per_interval=plan.gross_carry_bps_per_interval,
            round_trip_cost_bps=plan.round_trip_cost_bps,
            total_cost_bps=plan.total_cost_bps,
            breakeven_intervals=(plan.breakeven_intervals
                                 if math.isfinite(plan.breakeven_intervals)
                                 else None),
            horizon_intervals=plan.horizon_intervals,
            net_carry_bps_at_horizon=plan.net_carry_bps_at_horizon,
            apr_gross_pct=plan.apr_gross_pct,
            needs_borrow=plan.needs_borrow,
            status=status, skip_reason=skip_reason,
            is_simulated=True, details=details,
        )
        await self._record(row)
        return {'status': status, 'skip_reason': skip_reason,
                'is_simulated': True}

    async def _record(self, row: Dict[str, Any]) -> None:
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO basis_carry_suggestions
                        (venue, symbol, direction, perp_side, spot_side,
                         funding_bps, funding_interval_hours, perp_price,
                         spot_price, basis_bps,
                         perp_qty, spot_qty, notional_usd,
                         gross_carry_bps_per_interval, round_trip_cost_bps,
                         total_cost_bps, breakeven_intervals,
                         horizon_intervals, net_carry_bps_at_horizon,
                         apr_gross_pct, needs_borrow, status, skip_reason,
                         is_simulated, details)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,
                            $15,$16,$17,$18,$19,$20,$21,$22,$23,$24,$25)
                    """,
                    row['venue'], row['symbol'], row['direction'],
                    row['perp_side'], row['spot_side'], row['funding_bps'],
                    row['funding_interval_hours'], row['perp_price'],
                    row['spot_price'], row['basis_bps'],
                    row['perp_qty'], row['spot_qty'], row['notional_usd'],
                    row['gross_carry_bps_per_interval'],
                    row['round_trip_cost_bps'], row['total_cost_bps'],
                    row['breakeven_intervals'], row['horizon_intervals'],
                    row['net_carry_bps_at_horizon'], row['apr_gross_pct'],
                    bool(row['needs_borrow']), row['status'],
                    row['skip_reason'], bool(row['is_simulated']),
                    json.dumps(row.get('details') or {}, default=str),
                )
            self.suggestions_recorded += 1
        except Exception as e:
            logger.error('basis_carry_suggestions insert failed: %s', e)
