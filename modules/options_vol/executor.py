"""options_vol suggestion recorder + GATED live Deribit order path.

Default behaviour is ALWAYS a simulated record (options_vol_suggestions,
is_simulated=true). A live order is attempted ONLY when every gate in the
LIVE GATE CHAIN passes, in this exact order (mirrors modules/polymarket):

  1. shadow_mode (DB config)            must be False
  2. live_execution_enabled (DB config) must be True   (fail-safe default False)
  3. should_skip_live(module_dry_run, module='options_vol')
       -> False  (i.e. NOT dry-run, NO logs/.killswitch, NO logs/.pause_options_vol)
  4. core.risk_manager.RiskManager.validate_trade(instrument, premium_usd) -> (True, _)
  5. Monthly LIVE premium budget (monthly_premium_budget_usd, DB-counted from
     options_vol_suggestions live rows in the current calendar month) not exceeded
  6. ccxt deribit importable + DERIBIT_API_KEY/DERIBIT_API_SECRET resolvable
       (security/secrets_manager first, env fallback) — else simulated, never raises

ADDITIONALLY: only side='BUY' (long put / long call) can EVER go live here.
SELL legs (the collar's short call) are ALWAYS recorded simulated with
skip_reason='sell_leg_record_only' — premium selling is the classic bot-killer
and its live path is deliberately NOT implemented; the premium_selling_enabled
flag exists only so a future, separately-reviewed change has a named gate.
"""
from __future__ import annotations

import json
import logging
import os
import uuid
from typing import Any, Dict, Optional

logger = logging.getLogger("OptionsVolModule.Executor")


class OptionsVolExecutor:
    def __init__(self, db_pool, config: Dict[str, Any], risk_manager=None):
        self.db_pool = db_pool
        self.config = config or {}
        self.risk_manager = risk_manager
        self._exchange = None  # lazy ccxt deribit (private)
        self.live_orders = 0
        self.simulated_records = 0

    # ------------------------------------------------------------------
    # Gate chain
    # ------------------------------------------------------------------
    async def _month_live_premium_usd(self) -> float:
        """LIVE premium spent this calendar month (BUY legs only). Fail-CLOSED:
        a DB error returns +inf so the budget gate refuses live."""
        if not self.db_pool:
            return float("inf")
        try:
            async with self.db_pool.acquire() as conn:
                row = await conn.fetchrow(
                    """
                    SELECT COALESCE(SUM(GREATEST(premium_usd, 0)), 0)::float AS spent
                    FROM options_vol_suggestions
                    WHERE is_simulated = FALSE
                      AND status = 'live_submitted'
                      AND created_at >= date_trunc('month', NOW())
                    """
                )
            return float(row["spent"] or 0.0) if row else 0.0
        except Exception as e:
            logger.error("premium-budget query failed (%s) — refusing live", e)
            return float("inf")

    async def _resolve_skip_reason(self, *, module_dry_run: bool, side: str,
                                   instrument: str, premium_usd: float
                                   ) -> Optional[str]:
        """Returns None iff live execution is permitted. Never raises."""
        if str(side).upper() != "BUY":
            return "sell_leg_record_only"
        if self.config.get("shadow_mode", True):
            return "shadow_mode"
        if not self.config.get("live_execution_enabled", False):
            return "live_execution_disabled"
        try:
            from core.dry_run import should_skip_live
            if should_skip_live(module_dry_run, module="options_vol"):
                return "dry_run_or_killswitch_or_pause"
        except Exception as e:
            logger.error("should_skip_live unavailable (%s) — refusing live", e)
            return "dry_run_gate_unavailable"
        if self.risk_manager is None:
            return "risk_manager_missing"
        try:
            ok, reason = await self.risk_manager.validate_trade(instrument, premium_usd)
            if not ok:
                return f"risk_manager:{reason}"[:64]
        except Exception as e:
            logger.error("RiskManager.validate_trade error: %s — refusing live", e)
            return "risk_manager_error"
        budget = float(self.config.get("monthly_premium_budget_usd", 50.0))
        spent = await self._month_live_premium_usd()
        if spent + max(premium_usd, 0.0) > budget:
            return f"premium_budget_exhausted({spent:.2f}/{budget:.2f})"[:64]
        return None

    def _get_exchange(self):
        """Lazy ccxt deribit private client. Returns client or None (fail-soft)."""
        if self._exchange is not None:
            return self._exchange
        try:
            import ccxt.async_support as ccxt_async
        except ImportError:
            logger.warning("ccxt not installed — live path unavailable")
            return None
        api_key = api_secret = None
        try:
            from security.secrets_manager import secrets
            api_key = secrets.get("DERIBIT_API_KEY")
            api_secret = secrets.get("DERIBIT_API_SECRET")
        except Exception:
            pass
        if not api_key:
            api_key = os.getenv("DERIBIT_API_KEY")
        if not api_secret:
            api_secret = os.getenv("DERIBIT_API_SECRET")
        if not api_key or not api_secret:
            logger.warning("DERIBIT_API_KEY/SECRET not configured — live path unavailable")
            return None
        try:
            self._exchange = ccxt_async.deribit({
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            })
            return self._exchange
        except Exception as e:
            logger.error("ccxt deribit init failed: %s", e)
            return None

    async def close(self) -> None:
        if self._exchange is not None:
            try:
                await self._exchange.close()
            except Exception:
                pass
            self._exchange = None

    # ------------------------------------------------------------------
    # Execute (simulated by default)
    # ------------------------------------------------------------------
    async def execute_leg(self, *, structure_id: str, suggestion_type: str,
                          currency: str, leg: Dict[str, Any],
                          signal: Dict[str, Any], module_dry_run: bool
                          ) -> Dict[str, Any]:
        """Record one hedge leg; place a live order only if every gate passes.

        leg: {side, instrument_name, option_type, strike, expiry, contracts,
              iv, delta, premium_usd}.  signal: advisory context (index, rv,
              ivrv, fleet delta, hedged delta).
        """
        side = str(leg["side"]).upper()
        instrument = str(leg.get("instrument_name") or "")
        premium_usd = abs(float(leg.get("premium_usd") or 0.0))

        skip_reason = await self._resolve_skip_reason(
            module_dry_run=module_dry_run, side=side,
            instrument=instrument, premium_usd=premium_usd)
        if skip_reason is None and (not instrument or float(leg.get("contracts") or 0) <= 0):
            skip_reason = "missing_instrument_or_size"

        status, order_id, is_simulated = "simulated", None, True
        if skip_reason is None:
            exchange = self._get_exchange()
            if exchange is None:
                skip_reason = "exchange_client_unavailable"
            else:
                try:
                    amount = float(leg["contracts"])
                    # Limit BUY at current mark in coin terms (Deribit options
                    # are quoted in the base coin). usd_mark/index = coin price.
                    index_price = float(signal.get("index_price") or 0.0)
                    coin_price = (premium_usd / amount / index_price
                                  if index_price > 0 and amount > 0 else None)
                    order = await exchange.create_order(
                        instrument, "limit", "buy", amount, coin_price)
                    order_id = (order or {}).get("id")
                    status, is_simulated = "live_submitted", False
                    self.live_orders += 1
                    logger.warning(
                        "LIVE Deribit order submitted: BUY %s x%.4f ~$%.2f id=%s",
                        instrument, amount, premium_usd, order_id)
                except Exception as e:
                    logger.error("Live Deribit order failed: %s", e)
                    status, skip_reason = "live_failed", "deribit_order_error"

        await self._record(
            structure_id=structure_id, suggestion_type=suggestion_type,
            currency=currency, leg=leg, signal=signal, status=status,
            order_id=order_id, skip_reason=skip_reason,
            is_simulated=is_simulated, premium_usd_signed=float(leg.get("premium_usd") or 0.0))
        return {"status": status, "is_simulated": is_simulated,
                "skip_reason": skip_reason, "order_id": order_id}

    async def _record(self, *, structure_id: str, suggestion_type: str,
                      currency: str, leg: Dict[str, Any], signal: Dict[str, Any],
                      status: str, order_id: Optional[str],
                      skip_reason: Optional[str], is_simulated: bool,
                      premium_usd_signed: float) -> None:
        """Persist to options_vol_suggestions. Fail-soft."""
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO options_vol_suggestions
                        (structure_id, suggestion_type, currency, instrument_name,
                         side, option_type, strike, expiry, contracts, index_price,
                         mark_iv, rv, ivrv_ratio, delta, premium_usd,
                         fleet_net_delta_usd, hedged_delta_usd, status, order_id,
                         skip_reason, is_simulated, details)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,
                            $16,$17,$18,$19,$20,$21,$22)
                    """,
                    structure_id, suggestion_type, currency,
                    leg.get("instrument_name"), str(leg["side"]).upper(),
                    leg.get("option_type"), leg.get("strike"), leg.get("expiry"),
                    leg.get("contracts"), signal.get("index_price"),
                    leg.get("iv"), signal.get("rv"), signal.get("ivrv_ratio"),
                    leg.get("delta"), premium_usd_signed,
                    signal.get("fleet_net_delta_usd"),
                    signal.get("hedged_delta_usd"), status, order_id,
                    skip_reason, bool(is_simulated),
                    json.dumps({k: v for k, v in signal.items()
                                if k not in ("legs",)}, default=str),
                )
            if is_simulated:
                self.simulated_records += 1
        except Exception as e:
            logger.error("options_vol_suggestions insert failed: %s", e)


def new_structure_id() -> str:
    return uuid.uuid4().hex[:16]
