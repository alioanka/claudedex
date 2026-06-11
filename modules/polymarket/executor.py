"""Polymarket trade recorder + GATED live CLOB execution path.

Default behaviour is ALWAYS a simulated record (polymarket_trades,
is_simulated=true). A live order is attempted ONLY when every gate in
LIVE GATE CHAIN passes, in this exact order:

  1. shadow_mode (DB config)            must be False
  2. live_execution_enabled (DB config) must be True   (fail-safe default False)
  3. should_skip_live(module_dry_run, module='polymarket')
       -> False  (i.e. NOT dry-run, NO logs/.killswitch, NO logs/.pause_polymarket)
  4. core.risk_manager.RiskManager.validate_trade(token_id, size_usd) -> (True, _)
  5. py-clob-client importable + POLYMARKET_PRIVATE_KEY resolvable
       (security/secrets_manager first, env fallback) — else simulated, never raises

Any gate failure records a simulated row with skip_reason and returns.
"""

import json
import logging
import os
from typing import Any, Dict, Optional

logger = logging.getLogger("PolymarketModule.Executor")


class PolymarketExecutor:
    def __init__(self, db_pool, config: Dict[str, Any], risk_manager=None):
        self.db_pool = db_pool
        self.config = config or {}
        self.risk_manager = risk_manager
        self._clob_client = None
        self.live_orders = 0
        self.simulated_records = 0

    # ------------------------------------------------------------------
    # Gate chain
    # ------------------------------------------------------------------
    async def _resolve_skip_reason(self, module_dry_run: bool,
                                   token_id: str, size_usd: float) -> Optional[str]:
        """Returns None iff live execution is permitted. Never raises."""
        if self.config.get("shadow_mode", True):
            return "shadow_mode"
        if not self.config.get("live_execution_enabled", False):
            return "live_execution_disabled"
        try:
            from core.dry_run import should_skip_live
            if should_skip_live(module_dry_run, module="polymarket"):
                return "dry_run_or_killswitch_or_pause"
        except Exception as e:
            logger.error("should_skip_live unavailable (%s) — refusing live", e)
            return "dry_run_gate_unavailable"
        if self.risk_manager is None:
            return "risk_manager_missing"
        try:
            ok, reason = await self.risk_manager.validate_trade(token_id, size_usd)
            if not ok:
                return f"risk_manager:{reason}"[:64]
        except Exception as e:
            logger.error("RiskManager.validate_trade error: %s — refusing live", e)
            return "risk_manager_error"
        return None

    def _get_clob_client(self):
        """Lazy py-clob-client init. Returns client or None (fail-soft)."""
        if self._clob_client is not None:
            return self._clob_client
        try:
            from py_clob_client.client import ClobClient
        except ImportError:
            logger.warning("py-clob-client not installed — live path unavailable")
            return None
        private_key = None
        try:
            from security.secrets_manager import secrets
            private_key = secrets.get("POLYMARKET_PRIVATE_KEY")
        except Exception:
            private_key = None
        if not private_key:
            private_key = os.getenv("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not configured — live path unavailable")
            return None
        try:
            host = self.config.get("clob_base_url", "https://clob.polymarket.com")
            chain_id = int(self.config.get("chain_id", 137))
            client = ClobClient(host, key=private_key, chain_id=chain_id)
            client.set_api_creds(client.create_or_derive_api_creds())
            self._clob_client = client
            return client
        except Exception as e:
            logger.error("CLOB client init failed: %s", e)
            return None

    # ------------------------------------------------------------------
    # Execute (simulated by default)
    # ------------------------------------------------------------------
    async def execute(
        self,
        *,
        strategy: str,
        market_id: str,
        question: str,
        side: str,            # BUY / SELL
        outcome: str,         # YES / NO / BOTH
        token_id: Optional[str],
        price: Optional[float],
        size_usd: float,
        expected_edge_bps: Optional[float],
        module_dry_run: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record the trade; broadcast live only if the full gate chain passes."""
        max_size = float(self.config.get("max_position_size_usd", 50.0))
        size_usd = min(float(size_usd), max_size)

        skip_reason = await self._resolve_skip_reason(
            module_dry_run, token_id or market_id, size_usd
        )
        if skip_reason is None and (not token_id or price is None):
            skip_reason = "missing_token_or_price"

        status, order_id, is_simulated = "simulated", None, True
        if skip_reason is None:
            client = self._get_clob_client()
            if client is None:
                skip_reason = "clob_client_unavailable"
            else:
                try:
                    from py_clob_client.clob_types import OrderArgs, OrderType
                    from py_clob_client.order_builder.constants import BUY, SELL
                    clob_side = BUY if side.upper() == "BUY" else SELL
                    size_shares = round(size_usd / max(price, 0.01), 2)
                    order = client.create_order(OrderArgs(
                        price=round(price, 3),
                        size=size_shares,
                        side=clob_side,
                        token_id=token_id,
                    ))
                    resp = client.post_order(order, OrderType.GTC)
                    order_id = (resp or {}).get("orderID") or (resp or {}).get("orderId")
                    status, is_simulated = "live_submitted", False
                    self.live_orders += 1
                    logger.warning(
                        "LIVE Polymarket order submitted: %s %s %s @ %.3f $%.2f id=%s",
                        side, outcome, market_id, price, size_usd, order_id,
                    )
                except Exception as e:
                    logger.error("Live CLOB order failed: %s", e)
                    status, skip_reason = "live_failed", "clob_order_error"

        await self._record_trade(
            strategy=strategy, market_id=market_id, question=question, side=side,
            outcome=outcome, token_id=token_id, price=price, size_usd=size_usd,
            expected_edge_bps=expected_edge_bps, status=status, order_id=order_id,
            skip_reason=skip_reason, is_simulated=is_simulated, details=details,
        )
        return {
            "status": status,
            "is_simulated": is_simulated,
            "skip_reason": skip_reason,
            "order_id": order_id,
        }

    async def _record_trade(self, **row) -> None:
        """Persist to polymarket_trades. Fail-soft: a DB error never crashes the loop."""
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO polymarket_trades
                        (market_id, market_question, strategy, side, outcome, token_id,
                         price, size_usd, expected_edge_bps, status, order_id,
                         skip_reason, is_simulated, details)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    """,
                    str(row["market_id"]), row.get("question", ""), row["strategy"],
                    row["side"], row.get("outcome"), row.get("token_id"),
                    row.get("price"), row.get("size_usd"),
                    row.get("expected_edge_bps"), row["status"],
                    row.get("order_id"), row.get("skip_reason"),
                    bool(row.get("is_simulated", True)),
                    json.dumps(row.get("details") or {}, default=str),
                )
            if row.get("is_simulated", True):
                self.simulated_records += 1
        except Exception as e:
            logger.error("polymarket_trades insert failed: %s", e)
