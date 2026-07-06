"""Polymarket trade recorder + GATED live CLOB execution path.

Default behaviour is ALWAYS a simulated record (polymarket_trades,
is_simulated=true). A live order is attempted ONLY when every gate in
LIVE GATE CHAIN passes, in this exact order:

  1. shadow_mode (DB config)            must be False
  2. live_execution_enabled (DB config) must be True   (fail-safe default False)
  3. should_skip_live(module_dry_run, module='polymarket')
       -> False  (i.e. NOT dry-run, NO logs/.killswitch, NO logs/.pause_polymarket)
  4. Polymarket risk gate (built-in): per-market exposure cap, total module
       exposure cap, max open markets. Replaces the old
       core.risk_manager.validate_trade call — that gate runs EVM DEX
       honeypot/liquidity analysis on a token ADDRESS; a CLOB token id is a
       ~77-digit numeric string, so it either errored (permanently blocking
       live) or produced garbage (wave-F5 BUG-3).
  5. py-clob-client importable + POLYMARKET_PRIVATE_KEY resolvable
       (security/secrets_manager first, env fallback) — else simulated with
       skip_reason='clob_client_unavailable', never raises. Optional API creds
       (POLYMARKET_API_KEY/SECRET/PASSPHRASE) are used when all three resolve;
       otherwise creds are derived from the key.

Any gate failure records a simulated row with skip_reason and returns.

Live order lifecycle: place (FOK preferred) -> poll status until
filled/killed/timeout -> cancel on timeout -> record honest final status.
Arb is TWO-LEG: leg-2 failure is a CRITICAL event that records the unhedged
exposure honestly and immediately attempts a leg-1 unwind.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, Optional

logger = logging.getLogger("PolymarketModule.Executor")

# CLOB order states (lowercased) that mean the order is done and filled vs dead.
_FILLED_STATES = {"matched", "filled", "complete", "confirmed", "mined"}
_DEAD_STATES = {"cancelled", "canceled", "expired", "rejected", "killed", "unmatched"}


class PolymarketExecutor:
    def __init__(self, db_pool, config: Dict[str, Any]):
        self.db_pool = db_pool
        self.config = config or {}
        self._clob_client = None
        self.live_orders = 0
        self.simulated_records = 0
        # LIVE exposure only (USD notional per market). Rebuilt approximately
        # from the DB ledger by reconcile_open_orders() on startup.
        self.live_exposure: Dict[str, float] = {}
        self.open_orders_at_start: Optional[int] = None

    # ------------------------------------------------------------------
    # Gate chain
    # ------------------------------------------------------------------
    def _validate_risk(self, market_id: str, size_usd: float) -> Optional[str]:
        """Polymarket-appropriate risk gate. Returns None iff the trade fits
        inside every cap; otherwise a skip_reason string. Never raises."""
        try:
            per_market_cap = float(self.config.get("max_market_exposure_usd", 100.0))
            total_cap = float(self.config.get("max_total_exposure_usd", 500.0))
            max_markets = int(self.config.get("max_open_markets", 10))
            current = self.live_exposure.get(market_id, 0.0)
            if current + size_usd > per_market_cap:
                return "risk:market_exposure_cap"
            if sum(self.live_exposure.values()) + size_usd > total_cap:
                return "risk:total_exposure_cap"
            if market_id not in self.live_exposure and len(self.live_exposure) >= max_markets:
                return "risk:max_open_markets"
        except Exception as e:  # malformed config must fail CLOSED, not open
            logger.error("risk gate error: %s — refusing live", e)
            return "risk:gate_error"
        return None

    def _add_exposure(self, market_id: str, size_usd: float) -> None:
        self.live_exposure[market_id] = self.live_exposure.get(market_id, 0.0) + size_usd

    def _reduce_exposure(self, market_id: str, size_usd: float) -> None:
        left = self.live_exposure.get(market_id, 0.0) - size_usd
        if left <= 1e-9:
            self.live_exposure.pop(market_id, None)
        else:
            self.live_exposure[market_id] = left

    async def _resolve_skip_reason(self, module_dry_run: bool,
                                   market_id: str, size_usd: float) -> Optional[str]:
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
        return self._validate_risk(market_id, size_usd)

    # ------------------------------------------------------------------
    # CLOB client (fail-closed: any missing piece -> None, never raises)
    # ------------------------------------------------------------------
    @staticmethod
    def _secret(name: str) -> Optional[str]:
        """secrets_manager first, env fallback. Returns None when unset."""
        try:
            from security.secrets_manager import secrets
            val = secrets.get(name)
            if val:
                return val
        except Exception:
            pass
        return os.getenv(name) or None

    def _get_clob_client(self):
        """Lazy py-clob-client init. Returns client or None (fail-closed)."""
        if self._clob_client is not None:
            return self._clob_client
        try:
            from py_clob_client.client import ClobClient
            from py_clob_client.clob_types import ApiCreds
        except ImportError:
            logger.warning("py-clob-client not installed — live path unavailable "
                           "(orders will record skip_reason=clob_client_unavailable)")
            return None
        private_key = self._secret("POLYMARKET_PRIVATE_KEY")
        if not private_key:
            logger.warning("POLYMARKET_PRIVATE_KEY not configured — live path unavailable")
            return None
        try:
            host = self.config.get("clob_base_url", "https://clob.polymarket.com")
            chain_id = int(self.config.get("chain_id", 137))
            kwargs: Dict[str, Any] = {}
            # Proxy-wallet (Polymarket UI) accounts need signature_type 1/2 +
            # the funder (proxy) address; EOA-L1 accounts leave both unset.
            sig_type = str(self.config.get("signature_type", "") or "").strip()
            funder = str(self.config.get("funder_address", "") or "").strip()
            if sig_type and sig_type != "0":
                kwargs["signature_type"] = int(sig_type)
                if funder:
                    kwargs["funder"] = funder
            client = ClobClient(host, key=private_key, chain_id=chain_id, **kwargs)
            api_key = self._secret("POLYMARKET_API_KEY")
            api_secret = self._secret("POLYMARKET_API_SECRET")
            api_pass = self._secret("POLYMARKET_API_PASSPHRASE")
            if api_key and api_secret and api_pass:
                client.set_api_creds(ApiCreds(
                    api_key=api_key, api_secret=api_secret, api_passphrase=api_pass))
            else:
                client.set_api_creds(client.create_or_derive_api_creds())
            self._clob_client = client
            return client
        except Exception as e:
            logger.error("CLOB client init failed: %s — live path unavailable", e)
            return None

    # ------------------------------------------------------------------
    # Order lifecycle: place -> poll -> (cancel on timeout) -> honest status
    # ------------------------------------------------------------------
    async def _place_and_confirm(self, client, *, token_id: str, price: float,
                                 size_shares: float, side: str) -> Dict[str, Any]:
        """Place ONE CLOB order and poll until filled/dead/timeout.

        Returns {'filled': bool, 'order_id', 'status', 'error'}. Never raises.
        FOK is used when the installed py-clob-client exposes it (an arb leg
        must fill entirely at its price or not at all); GTC + cancel-on-timeout
        otherwise.
        """
        result: Dict[str, Any] = {"filled": False, "order_id": None,
                                  "status": "not_placed", "error": None}
        try:
            from py_clob_client.clob_types import OrderArgs, OrderType
            from py_clob_client.order_builder.constants import BUY, SELL
            order = client.create_order(OrderArgs(
                price=round(float(price), 3),
                size=round(float(size_shares), 2),
                side=BUY if side.upper() == "BUY" else SELL,
                token_id=token_id,
            ))
            order_type = getattr(OrderType, "FOK", None) or OrderType.GTC
            resp = client.post_order(order, order_type) or {}
            order_id = resp.get("orderID") or resp.get("orderId")
            result["order_id"] = order_id
            status = str(resp.get("status") or "").lower()
            result["status"] = status or "submitted"
            if status in _FILLED_STATES:
                result["filled"] = True
                return result
            if status in _DEAD_STATES:
                return result
            if not order_id:
                result["error"] = f"no order id in post_order response: {resp}"
                return result
            timeout_s = float(self.config.get("order_fill_timeout_s", 30))
            deadline = time.monotonic() + max(timeout_s, 2.0)
            while time.monotonic() < deadline:
                await asyncio.sleep(2.0)
                try:
                    info = client.get_order(order_id) or {}
                except Exception as e:
                    result["error"] = f"status poll failed: {e}"
                    continue
                status = str(info.get("status") or "").lower()
                if status:
                    result["status"] = status
                if status in _FILLED_STATES:
                    result["filled"] = True
                    return result
                if status in _DEAD_STATES:
                    return result
            # Timeout: never leave an unknown resting order on the book.
            try:
                client.cancel(order_id)
                result["status"] = "cancelled_timeout"
            except Exception as e:
                result["error"] = f"cancel after timeout failed: {e}"
                result["status"] = "unknown_resting"
            return result
        except Exception as e:
            result["error"] = str(e)
            result["status"] = "error"
            return result

    # ------------------------------------------------------------------
    # Two-leg atomic-ish arb (YES + NO bought together redeem to $1/share)
    # ------------------------------------------------------------------
    async def execute_arb(
        self,
        *,
        market_id: str,
        question: str,
        yes_token_id: Optional[str],
        no_token_id: Optional[str],
        yes_price: Optional[float],
        no_price: Optional[float],
        size_usd: float,
        expected_edge_bps: Optional[float],
        module_dry_run: bool,
        details: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Record (shadow) or execute (live, fully gated) a YES+NO<1 arb.

        Shadow record is ONE honest row: outcome=BOTH, price=pair_cost
        (yes+no, the true cost per $1 redemption — the old single-leg record
        at price=yes_price made the NO-leg cost unrecoverable, wave-F5 BUG-2).
        Live: N = size_usd / pair_cost shares of EACH leg; leg-1 (YES) then
        leg-2 (NO). Leg-2 failure -> CRITICAL log + honest unhedged-exposure
        record + immediate leg-1 unwind attempt.
        """
        max_size = float(self.config.get("max_position_size_usd", 50.0))
        size_usd = min(float(size_usd), max_size)
        pair_cost = (yes_price or 0.0) + (no_price or 0.0)
        details = dict(details or {})
        details["pair_cost"] = round(pair_cost, 6)

        skip_reason = await self._resolve_skip_reason(module_dry_run, market_id, size_usd)
        if skip_reason is None and (
                not yes_token_id or not no_token_id
                or not yes_price or not no_price or pair_cost <= 0.0):
            skip_reason = "missing_token_or_price"

        async def _record(status: str, *, outcome: str = "BOTH",
                          token_id: Optional[str] = None, price: Optional[float] = None,
                          notional: Optional[float] = None, order_id: Optional[str] = None,
                          reason: Optional[str] = None, simulated: bool = True,
                          extra: Optional[Dict[str, Any]] = None) -> None:
            await self._record_trade(
                strategy="risk_free_arb", market_id=market_id, question=question,
                side="BUY", outcome=outcome, token_id=token_id or yes_token_id,
                price=price if price is not None else round(pair_cost, 6),
                size_usd=notional if notional is not None else size_usd,
                expected_edge_bps=expected_edge_bps, status=status,
                order_id=order_id, skip_reason=reason, is_simulated=simulated,
                details={**details, **(extra or {})},
            )

        if skip_reason is not None:
            await _record("simulated", reason=skip_reason,
                          extra={"yes_price": yes_price, "no_price": no_price,
                                 "price_is": "pair_cost_both_legs"})
            return {"status": "simulated", "is_simulated": True,
                    "skip_reason": skip_reason, "order_id": None}

        client = self._get_clob_client()
        if client is None:
            await _record("simulated", reason="clob_client_unavailable",
                          extra={"yes_price": yes_price, "no_price": no_price})
            return {"status": "simulated", "is_simulated": True,
                    "skip_reason": "clob_client_unavailable", "order_id": None}

        # Equal share counts on both legs: N shares of YES + N of NO redeem N dollars.
        shares = size_usd / pair_cost
        leg1_cost = shares * yes_price
        leg2_cost = shares * no_price

        leg1 = await self._place_and_confirm(
            client, token_id=yes_token_id, price=yes_price, size_shares=shares, side="BUY")
        if not leg1["filled"]:
            reason = f"leg1_{leg1['status']}"[:64]
            logger.error("Arb leg-1 (YES) did not fill on %s: status=%s err=%s",
                         market_id, leg1["status"], leg1["error"])
            await _record("live_failed", reason=reason, order_id=leg1["order_id"],
                          extra={"leg1": leg1})
            return {"status": "live_failed", "is_simulated": True,
                    "skip_reason": reason, "order_id": leg1["order_id"]}

        self._add_exposure(market_id, leg1_cost)
        self.live_orders += 1

        leg2 = await self._place_and_confirm(
            client, token_id=no_token_id, price=no_price, size_shares=shares, side="BUY")
        if leg2["filled"]:
            self._add_exposure(market_id, leg2_cost)
            self.live_orders += 1
            order_id = f"{leg1['order_id']}|{leg2['order_id']}"
            logger.warning(
                "LIVE arb pair filled on %s: %.2f shares YES@%.3f + NO@%.3f "
                "cost=$%.2f edge=%sbps ids=%s",
                market_id, shares, yes_price, no_price,
                leg1_cost + leg2_cost, expected_edge_bps, order_id)
            await _record("live_filled", simulated=False, order_id=order_id,
                          notional=round(leg1_cost + leg2_cost, 2),
                          extra={"shares": round(shares, 2), "leg1": leg1, "leg2": leg2})
            return {"status": "live_filled", "is_simulated": False,
                    "skip_reason": None, "order_id": order_id}

        # ── Leg-2 failed: we hold an UNHEDGED YES position. Be loud + honest,
        # then try to get flat immediately. ──
        logger.critical(
            "ARB LEG-2 (NO) FAILED on %s — UNHEDGED YES exposure $%.2f "
            "(%.2f shares @ %.3f). leg2_status=%s err=%s. Attempting leg-1 unwind.",
            market_id, leg1_cost, shares, yes_price, leg2["status"], leg2["error"])
        await _record("live_leg2_failed", simulated=False, outcome="YES",
                      token_id=yes_token_id, price=yes_price,
                      notional=round(leg1_cost, 2), order_id=leg1["order_id"],
                      reason=f"leg2_{leg2['status']}"[:64],
                      extra={"unhedged_usd": round(leg1_cost, 2),
                             "leg1": leg1, "leg2": leg2})

        unwind_price = details.get("best_bid") or yes_price
        unwind = await self._place_and_confirm(
            client, token_id=yes_token_id, price=unwind_price,
            size_shares=shares, side="SELL")
        if unwind["filled"]:
            self._reduce_exposure(market_id, leg1_cost)
            logger.warning("Leg-1 unwind FILLED on %s: sold %.2f YES @ %.3f (id=%s)",
                           market_id, shares, unwind_price, unwind["order_id"])
            await _record("live_unwound", simulated=False, outcome="YES",
                          token_id=yes_token_id, price=unwind_price,
                          notional=round(shares * unwind_price, 2),
                          order_id=unwind["order_id"],
                          extra={"unwind_of": leg1["order_id"], "unwind": unwind})
        else:
            logger.critical(
                "LEG-1 UNWIND FAILED on %s — still holding %.2f YES shares "
                "($%.2f). OPERATOR ACTION REQUIRED (sell manually or hold to "
                "resolution). unwind_status=%s err=%s",
                market_id, shares, leg1_cost, unwind["status"], unwind["error"])
            await _record("live_unwind_failed", simulated=False, outcome="YES",
                          token_id=yes_token_id, price=unwind_price,
                          notional=round(leg1_cost, 2), order_id=unwind["order_id"],
                          reason=f"unwind_{unwind['status']}"[:64],
                          extra={"unhedged_usd": round(leg1_cost, 2), "unwind": unwind})
        return {"status": "live_leg2_failed", "is_simulated": False,
                "skip_reason": f"leg2_{leg2['status']}"[:64],
                "order_id": leg1["order_id"]}

    # ------------------------------------------------------------------
    # Startup reconciliation (LIVE only — shadow mode has nothing resting)
    # ------------------------------------------------------------------
    async def reconcile_open_orders(self, module_dry_run: bool) -> None:
        """On startup with live gates open: list resting CLOB orders (a crash
        between place and cancel must never leave unknown exposure) and rebuild
        the in-memory exposure map from the live rows in polymarket_trades.
        Fail-soft; a reconcile error never blocks the shadow loop."""
        if self.config.get("shadow_mode", True) \
                or not self.config.get("live_execution_enabled", False):
            return
        client = self._get_clob_client()
        if client is None:
            logger.warning("reconcile: CLOB client unavailable — cannot verify "
                           "no orders are resting on the book")
            return
        try:
            try:
                from py_clob_client.clob_types import OpenOrderParams
                orders = client.get_orders(OpenOrderParams()) or []
            except (ImportError, TypeError):
                orders = client.get_orders() or []
            self.open_orders_at_start = len(orders)
            if orders:
                logger.warning("reconcile: %d OPEN CLOB order(s) found at startup:", len(orders))
                for o in orders:
                    logger.warning("reconcile: open order id=%s market=%s side=%s "
                                   "price=%s size=%s",
                                   o.get("id") or o.get("orderID"), o.get("market"),
                                   o.get("side"), o.get("price"),
                                   o.get("original_size") or o.get("size"))
            else:
                logger.info("reconcile: no open CLOB orders at startup")
        except Exception as e:
            logger.error("reconcile: get_orders failed: %s", e)
        # Approximate exposure rebuild from the ledger (BUY adds, SELL reduces).
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT market_id,
                           SUM(CASE WHEN side = 'SELL' THEN -size_usd
                                    ELSE size_usd END) AS net_usd
                    FROM polymarket_trades
                    WHERE is_simulated = FALSE
                      AND status IN ('live_filled', 'live_leg2_failed', 'live_unwound',
                                     'live_submitted', 'live_unwind_failed')
                      AND created_at > NOW() - INTERVAL '30 days'
                    GROUP BY market_id
                    """
                )
            for row in rows:
                net = float(row["net_usd"] or 0.0)
                if net > 0.0:
                    self.live_exposure[str(row["market_id"])] = net
            if self.live_exposure:
                logger.warning("reconcile: rebuilt live exposure map: %s",
                               {k: round(v, 2) for k, v in self.live_exposure.items()})
        except Exception as e:
            logger.error("reconcile: exposure rebuild failed: %s", e)

    # ------------------------------------------------------------------
    # Generic single-order execute (simulated by default)
    # ------------------------------------------------------------------
    async def execute(
        self,
        *,
        strategy: str,
        market_id: str,
        question: str,
        side: str,            # BUY / SELL
        outcome: str,         # YES / NO
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

        skip_reason = await self._resolve_skip_reason(module_dry_run, market_id, size_usd)
        if skip_reason is None and (not token_id or not price):
            skip_reason = "missing_token_or_price"

        status, order_id, is_simulated = "simulated", None, True
        if skip_reason is None:
            client = self._get_clob_client()
            if client is None:
                skip_reason = "clob_client_unavailable"
            else:
                shares = size_usd / max(price, 0.01)
                leg = await self._place_and_confirm(
                    client, token_id=token_id, price=price,
                    size_shares=shares, side=side)
                order_id = leg["order_id"]
                if leg["filled"]:
                    status, is_simulated = "live_filled", False
                    self.live_orders += 1
                    if side.upper() == "BUY":
                        self._add_exposure(market_id, size_usd)
                    else:
                        self._reduce_exposure(market_id, size_usd)
                    logger.warning(
                        "LIVE Polymarket order filled: %s %s %s @ %.3f $%.2f id=%s",
                        side, outcome, market_id, price, size_usd, order_id)
                else:
                    status = "live_failed"
                    skip_reason = f"order_{leg['status']}"[:64]
                    logger.error("Live CLOB order not filled: %s (err=%s)",
                                 leg["status"], leg["error"])

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
