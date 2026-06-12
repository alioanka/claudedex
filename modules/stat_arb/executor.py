"""Stat-arb paired-trade recorder + GATED live execution path.

Default behaviour is ALWAYS a simulated record (stat_arb_trades,
is_simulated=true). A live two-leg order is attempted ONLY when every gate in
the LIVE GATE CHAIN passes, in this exact order (mirrors modules/polymarket):

  1. shadow_mode (DB config)            must be False
  2. live_execution_enabled (DB config) must be True   (fail-safe default False)
  3. should_skip_live(module_dry_run, module='stat_arb')
       -> False  (i.e. NOT dry-run, NO logs/.killswitch, NO logs/.pause_stat_arb)
  4. core.risk_manager.RiskManager.validate_trade(pair_key, total_notional) -> (True, _)
  5. ccxt importable + BYBIT_API_KEY/BYBIT_API_SECRET resolvable
       (security/secrets_manager first, env fallback) — else simulated, never raises

Any gate failure records a simulated row with skip_reason and returns.
Live legs are MARKET orders on Bybit linear perps; if the second leg fails the
first leg is immediately flattened (reduce-only) — never run one-legged.
Closes are reduce-only. There is NO add-to-position API on purpose: a pair is
opened once at fixed notional and closed once (no averaging down, repo rule).
"""

import json
import logging
import os
from typing import Any, Dict, Optional

from modules.stat_arb.core.data import compact_symbol

logger = logging.getLogger("StatArbModule.Executor")


class StatArbExecutor:
    def __init__(self, db_pool, config: Dict[str, Any], risk_manager=None):
        self.db_pool = db_pool
        self.config = config or {}
        self.risk_manager = risk_manager
        self._exchange = None
        self.live_orders = 0
        self.simulated_records = 0

    # ------------------------------------------------------------------
    # Gate chain (None == live permitted)
    # ------------------------------------------------------------------
    async def _resolve_skip_reason(self, module_dry_run: bool, pair_key: str,
                                   total_notional_usd: float) -> Optional[str]:
        if self.config.get("shadow_mode", True):
            return "shadow_mode"
        if not self.config.get("live_execution_enabled", False):
            return "live_execution_disabled"
        try:
            from core.dry_run import should_skip_live
            if should_skip_live(module_dry_run, module="stat_arb"):
                return "dry_run_or_killswitch_or_pause"
        except Exception as e:
            logger.error("should_skip_live unavailable (%s) — refusing live", e)
            return "dry_run_gate_unavailable"
        if self.risk_manager is None:
            return "risk_manager_missing"
        try:
            ok, reason = await self.risk_manager.validate_trade(
                pair_key, total_notional_usd)
            if not ok:
                return f"risk_manager:{reason}"[:64]
        except Exception as e:
            logger.error("RiskManager.validate_trade error: %s — refusing live", e)
            return "risk_manager_error"
        return None

    def _get_exchange(self):
        """Lazy ccxt Bybit init. Returns exchange or None (fail-soft)."""
        if self._exchange is not None:
            return self._exchange
        try:
            import ccxt.async_support as ccxt
        except ImportError:
            logger.warning("ccxt not installed — live path unavailable")
            return None
        api_key = api_secret = None
        try:
            from security.secrets_manager import secrets
            api_key = secrets.get("BYBIT_API_KEY")
            api_secret = secrets.get("BYBIT_API_SECRET")
        except Exception:
            pass
        api_key = api_key or os.getenv("BYBIT_API_KEY")
        api_secret = api_secret or os.getenv("BYBIT_API_SECRET")
        if not api_key or not api_secret:
            logger.warning("BYBIT_API_KEY/SECRET not configured — live path unavailable")
            return None
        try:
            self._exchange = ccxt.bybit({
                "apiKey": api_key, "secret": api_secret,
                "options": {"defaultType": "swap"},
                "enableRateLimit": True,
            })
            return self._exchange
        except Exception as e:
            logger.error("ccxt bybit init failed: %s", e)
            return None

    async def _live_leg(self, exchange, symbol: str, side: str,
                        notional_usd: float, price: float,
                        reduce_only: bool) -> Optional[str]:
        """One MARKET leg; returns order id or None on failure. Never raises."""
        try:
            qty = round(notional_usd / max(price, 1e-9), 6)
            params = {"category": "linear"}
            if reduce_only:
                params["reduceOnly"] = True
            order = await exchange.create_order(
                compact_symbol(symbol), "market", side, qty, None, params)
            return str((order or {}).get("id") or "")
        except Exception as e:
            logger.error("live leg failed (%s %s $%.2f): %s",
                         side, symbol, notional_usd, e)
            return None

    async def _live_open(self, *, long_symbol: str, short_symbol: str,
                         long_price: float, short_price: float,
                         notional_per_leg_usd: float) -> Optional[dict]:
        """Both entry legs or nothing. Returns order ids or None."""
        exchange = self._get_exchange()
        if exchange is None:
            return None
        long_id = await self._live_leg(exchange, long_symbol, "buy",
                                       notional_per_leg_usd, long_price, False)
        if long_id is None:
            return None
        short_id = await self._live_leg(exchange, short_symbol, "sell",
                                        notional_per_leg_usd, short_price, False)
        if short_id is None:
            # NEVER run one-legged: flatten the long leg immediately.
            logger.error("second leg failed — flattening %s", long_symbol)
            await self._live_leg(exchange, long_symbol, "sell",
                                 notional_per_leg_usd, long_price, True)
            return None
        return {"long_order_id": long_id, "short_order_id": short_id}

    async def _live_close(self, *, long_symbol: str, short_symbol: str,
                          long_price: float, short_price: float,
                          notional_per_leg_usd: float) -> Optional[dict]:
        """Reduce-only closes for both legs; best-effort each (close > hold)."""
        exchange = self._get_exchange()
        if exchange is None:
            return None
        sell_id = await self._live_leg(exchange, long_symbol, "sell",
                                       notional_per_leg_usd, long_price, True)
        buy_id = await self._live_leg(exchange, short_symbol, "buy",
                                      notional_per_leg_usd, short_price, True)
        if sell_id is None and buy_id is None:
            return None
        return {"long_close_id": sell_id, "short_close_id": buy_id}

    # ------------------------------------------------------------------
    # Open / close (simulated by default)
    # ------------------------------------------------------------------
    async def open_pair(self, *, pair_key: str, long_symbol: str,
                        short_symbol: str, beta: float, z_at_entry: float,
                        long_price: float, short_price: float,
                        notional_per_leg_usd: float, module_dry_run: bool,
                        details: Optional[dict] = None) -> Dict[str, Any]:
        skip_reason = await self._resolve_skip_reason(
            module_dry_run, pair_key, 2.0 * notional_per_leg_usd)

        status, is_simulated, order_ids = "open", True, None
        if skip_reason is None:
            order_ids = await self._live_open(
                long_symbol=long_symbol, short_symbol=short_symbol,
                long_price=long_price, short_price=short_price,
                notional_per_leg_usd=notional_per_leg_usd)
            if order_ids is None:
                skip_reason = "live_open_failed"
            else:
                is_simulated = False
                self.live_orders += 2
                logger.warning("LIVE pair opened: long %s / short %s $%.2f/leg",
                               long_symbol, short_symbol, notional_per_leg_usd)

        det = dict(details or {})
        if order_ids:
            det.update(order_ids)
        trade_id = None
        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    trade_id = await conn.fetchval(
                        """
                        INSERT INTO stat_arb_trades
                            (pair_key, long_symbol, short_symbol, beta,
                             notional_per_leg_usd, z_at_entry,
                             entry_long_price, entry_short_price, status,
                             skip_reason, is_simulated, details)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,'open',$9,$10,$11)
                        RETURNING id
                        """,
                        pair_key, long_symbol, short_symbol, beta,
                        notional_per_leg_usd, z_at_entry, long_price,
                        short_price, skip_reason, is_simulated,
                        json.dumps(det, default=str),
                    )
                if is_simulated:
                    self.simulated_records += 1
            except Exception as e:
                logger.error("stat_arb_trades insert failed: %s", e)
        return {"trade_id": trade_id, "status": status,
                "is_simulated": is_simulated, "skip_reason": skip_reason}

    async def close_pair(self, *, trade_id: int, pair_key: str,
                         long_symbol: str, short_symbol: str,
                         long_price: float, short_price: float,
                         notional_per_leg_usd: float, z_at_exit: float,
                         exit_reason: str, pnl_usd: Optional[float],
                         was_simulated: bool, module_dry_run: bool) -> Dict[str, Any]:
        """Close an open pair row. Live closes are attempted iff the OPEN was
        live; exits are never blocked by entry gates (close > hold) but still
        refuse on dry_run/killswitch/pause via should_skip_live."""
        close_ids, is_simulated = None, True
        if not was_simulated:
            blocked = True
            try:
                from core.dry_run import should_skip_live
                blocked = should_skip_live(module_dry_run, module="stat_arb")
            except Exception:
                blocked = True
            if not blocked:
                close_ids = await self._live_close(
                    long_symbol=long_symbol, short_symbol=short_symbol,
                    long_price=long_price, short_price=short_price,
                    notional_per_leg_usd=notional_per_leg_usd)
            if close_ids is not None:
                is_simulated = False
                self.live_orders += 2
            else:
                # LIVE position we could not close: record honestly, keep open.
                logger.error("LIVE close blocked/failed for %s — keeping open", pair_key)
                return {"status": "open", "is_simulated": False,
                        "skip_reason": "live_close_failed"}

        if self.db_pool:
            try:
                async with self.db_pool.acquire() as conn:
                    await conn.execute(
                        """
                        UPDATE stat_arb_trades
                        SET status='closed', exit_reason=$2, z_at_exit=$3,
                            exit_long_price=$4, exit_short_price=$5,
                            pnl_usd=$6, closed_at=NOW(),
                            details = COALESCE(details, '{}'::jsonb) || $7::jsonb
                        WHERE id = $1
                        """,
                        trade_id, exit_reason, z_at_exit, long_price,
                        short_price, pnl_usd,
                        json.dumps(close_ids or {}, default=str),
                    )
            except Exception as e:
                logger.error("stat_arb_trades close update failed: %s", e)
        return {"status": "closed", "is_simulated": is_simulated,
                "skip_reason": None}
