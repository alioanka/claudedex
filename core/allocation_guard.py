"""Central allocation guard — wave-15.

Single source of truth for "can module X commit another $N right now?"

Architecture
------------
Multiple modules (DEX, Arbitrage, AI, Futures, Solana, Sniper, Copy)
share the SAME on-chain wallets:
  EVM wallet  : dex, arbitrage, ai, futures
  Solana wallet: solana, sniper, copy_trading

Without a guard, two modules can spend the same lamports and the
portfolio allocator's USD budgets are advisory only.

This guard is the enforcement layer. It:
  1. Reads each module's current open-position USD exposure from the DB
     (reusing the exposure_aggregator pattern).
  2. Compares (existing_committed + requested) against:
       a. The module's own budget (from orchestrator or static config).
       b. The wallet-group global cap (sum across all co-wallet modules).
       c. The global total cap (all modules combined).
  3. Returns (allowed: bool, reason: str). In DRY_RUN it always logs
     the veto decision but respects the same logic — "DRY_RUN
     observable" mode.

Integration
-----------
The guard is wired into RiskManager.validate_trade as a FINAL advisory
layer (after the existing circuit-breaker and token-risk checks). It
is fail-SOFT: any DB error allows the trade through and logs a warning.
Only ENTRIES are gated; exits are never blocked (exits reduce exposure).

Budget authority
----------------
  1. Orchestrator budget_usd (most recent non-superseded rec, within TTL)
     — wins when use_orchestrator_budget=true (default).
  2. Static config default  (allocation_guard_config.budget_usd_<module>)
     — fallback when orchestrator has no recent budget.
  3. 0 in config_settings means "unlimited" for that module.

Design note: the orchestrator can only TIGHTEN or SET budgets within
operator-defined global caps. It cannot invent capital beyond the global
caps seeded in migration 046.

Usage
-----
    guard = AllocationGuard(db_pool, dry_run=False)
    allowed, reason = await guard.check("copy_trading", requested_usd=50.0)
    if not allowed:
        log.warning("allocation veto: %s", reason)
        return  # skip entry

Thread safety: stateless per-call (all state is in DB). Safe to create
one instance per module process.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger("allocation_guard")

# ---------------------------------------------------------------------------
# Module → wallet group mapping (mirrors migration 046 defaults).
# Overridable at runtime via DB config keys evm_wallet_modules /
# solana_wallet_modules. Hard-coded defaults used when DB is unavailable.
# ---------------------------------------------------------------------------
_DEFAULT_EVM_MODULES: List[str] = ["dex", "arbitrage", "ai", "futures"]
_DEFAULT_SOLANA_MODULES: List[str] = ["solana", "sniper", "copy_trading"]

# ---------------------------------------------------------------------------
# Per-module open-position tables.
# Each entry: (module_label, table, token_col, usd_col, extra_where).
# Generalised from modules/copy_trading/exposure_aggregator.py.
# We SUM all open positions — NOT per-token — to get total committed capital.
# ---------------------------------------------------------------------------
_MODULE_COMMITTED_TABLES = (
    # DEX: usd_value column; open positions only
    ("dex",          "trades",             "token_address", "usd_value",  "status = 'open'"),
    # Sniper
    ("sniper",       "sniper_trades",      "token_address", "entry_usd",  "status = 'open'"),
    # AI
    ("ai",           "ai_trades",          "token_address", "entry_usd",  "status = 'open'"),
    # Copy trading
    ("copy_trading", "copytrading_trades", "token_address", "entry_usd",  "status = 'open'"),
    # Futures: futures_positions is the live open-position table.
    # notional_value is the USD-equivalent position size (size * entry_price).
    ("futures",      "futures_positions",  "symbol",        "notional_value", None),
    # Solana: uses entry_usd added in migration 038; fall back to entry_price*amount
    # when entry_usd is NULL (handled in _sum_solana_committed).
    # Arbitrage: open legs tracked in arbitrage_trades with status='open'.
    ("arbitrage",    "arbitrage_trades",   "token_address", "entry_usd",  "status = 'open'"),
)


async def _sum_table_committed(conn, *, table: str, usd_col: str,
                                extra_where: Optional[str]) -> float:
    """Sum all open-position USD for one module table. Fail-soft."""
    try:
        where_clause = f"WHERE {extra_where}" if extra_where else ""
        sql = (
            f"SELECT COALESCE(SUM({usd_col}), 0)::float AS total "
            f"FROM {table} {where_clause}"
        )
        row = await conn.fetchrow(sql)
        return float(row["total"] or 0.0) if row else 0.0
    except Exception as exc:
        logger.debug("_sum_table_committed(%s) fail-soft: %s", table, exc)
        return 0.0


async def _sum_solana_committed(conn) -> float:
    """Sum open Solana positions in USD.

    Uses entry_usd when available (migration 038). Falls back to
    entry_price * amount (USD-per-token * token-count) for rows where
    entry_usd is NULL.
    """
    try:
        row = await conn.fetchrow(
            """
            SELECT COALESCE(
                SUM(
                    CASE
                        WHEN entry_usd IS NOT NULL THEN entry_usd
                        WHEN entry_price IS NOT NULL AND amount IS NOT NULL
                             THEN entry_price * amount
                        ELSE 0
                    END
                ), 0
            )::float AS total
            FROM solana_positions
            """
        )
        return float(row["total"] or 0.0) if row else 0.0
    except Exception as exc:
        logger.debug("_sum_solana_committed fail-soft: %s", exc)
        return 0.0


async def get_all_committed(conn) -> Dict[str, float]:
    """Return {module: committed_usd} for every trading module.

    Fail-soft per module — a missing/empty table returns 0.0 for that
    module rather than breaking the whole dict. Never raises.
    """
    result: Dict[str, float] = {}
    for label, table, _tok_col, usd_col, extra_where in _MODULE_COMMITTED_TABLES:
        result[label] = await _sum_table_committed(
            conn, table=table, usd_col=usd_col, extra_where=extra_where
        )
    # Solana module has its own dedicated table
    result["solana"] = await _sum_solana_committed(conn)
    return result


async def _get_static_budget(conn, module: str, default_usd: float = 0.0) -> float:
    """Read budget_usd_<module> from allocation_guard_config. Returns
    default_usd if the key is absent or DB fails."""
    try:
        key = f"budget_usd_{module}"
        row = await conn.fetchrow(
            "SELECT value FROM config_settings "
            "WHERE config_type = 'allocation_guard_config' AND key = $1",
            key,
        )
        if row and row["value"]:
            return float(row["value"])
    except Exception as exc:
        logger.debug("_get_static_budget(%s) fail-soft: %s", module, exc)
    return default_usd


async def _get_orchestrator_budget(
    conn, module: str, ttl_minutes: int
) -> Optional[float]:
    """Return the orchestrator's latest recommended budget_usd for this
    module if it exists and is within TTL. Returns None otherwise."""
    try:
        cutoff = datetime.utcnow() - timedelta(minutes=ttl_minutes)
        row = await conn.fetchrow(
            "SELECT budget_usd FROM orchestrator_recommendations "
            "WHERE module = $1 "
            "  AND budget_usd IS NOT NULL "
            "  AND approved IS NULL "
            "  AND superseded_at IS NULL "
            "  AND created_at >= $2 "
            "ORDER BY created_at DESC LIMIT 1",
            module,
            cutoff,
        )
        if row and row["budget_usd"] is not None:
            return float(row["budget_usd"])
    except Exception as exc:
        logger.debug("_get_orchestrator_budget(%s) fail-soft: %s", module, exc)
    return None


async def _get_guard_config(conn) -> Dict:
    """Load all allocation_guard_config rows into a flat dict. Fail-soft."""
    cfg: Dict = {}
    try:
        rows = await conn.fetch(
            "SELECT key, value, value_type FROM config_settings "
            "WHERE config_type = 'allocation_guard_config'"
        )
        for row in rows:
            raw = row["value"]
            vtype = row["value_type"]
            try:
                if vtype == "float":
                    cfg[row["key"]] = float(raw)
                elif vtype == "int":
                    cfg[row["key"]] = int(raw)
                elif vtype == "bool":
                    cfg[row["key"]] = raw.strip().lower() in ("true", "1", "yes")
                elif vtype == "json":
                    cfg[row["key"]] = json.loads(raw)
                else:
                    cfg[row["key"]] = raw
            except Exception:
                cfg[row["key"]] = raw
    except Exception as exc:
        logger.debug("_get_guard_config fail-soft: %s", exc)
    return cfg


class AllocationGuard:
    """Central allocation guard.

    One instance per module subprocess (pass the shared db_pool).
    The guard is stateless between calls — all state lives in the DB.

    Parameters
    ----------
    db_pool : asyncpg pool
    dry_run : bool
        When True the guard logs every veto decision but always returns
        allowed=True so DRY_RUN simulations are not blocked. The veto
        decision is logged with prefix [DRY_RUN] so the operator can
        observe would-be blocks before enabling enforcement.
    module : str
        The module name this guard instance belongs to (used for logging).
    """

    def __init__(self, db_pool, *, dry_run: bool = True, module: str = "unknown"):
        self._pool = db_pool
        self._dry_run = dry_run
        self._module = module

    async def check(
        self, module: str, requested_usd: float
    ) -> Tuple[bool, str]:
        """Check whether this module may commit `requested_usd` more capital.

        Checks (in order):
          1. Guard enabled? (if not, allow with a note)
          2. Module budget: existing_committed + requested <= budget
          3. Wallet-group cap: group_total + requested <= group_cap
          4. Global total cap: all_total + requested <= global_cap

        Returns
        -------
        (allowed: bool, reason: str)
            allowed=True means the entry is within all limits.
            allowed=False means at least one limit would be breached.
            In DRY_RUN mode allowed is always True but the veto reason
            is logged.

        Fail-soft: any DB error returns (True, "guard skipped: <error>")
        so the existing per-module caps remain the safety net.
        """
        if requested_usd <= 0:
            return True, "zero or negative amount, guard skipped"

        try:
            async with self._pool.acquire() as conn:
                return await self._check_inner(conn, module, requested_usd)
        except Exception as exc:
            msg = f"allocation_guard fail-soft ({module}): {exc}"
            logger.warning(msg)
            return True, msg

    async def _check_inner(
        self, conn, module: str, requested_usd: float
    ) -> Tuple[bool, str]:
        cfg = await _get_guard_config(conn)

        guard_enabled: bool = cfg.get("enabled", True)
        if not guard_enabled:
            return True, "allocation guard disabled"

        # --- 1. Resolve module budget -----------------------------------------
        budget_usd: float = 0.0  # 0 = unlimited
        use_orch = cfg.get("use_orchestrator_budget", True)
        ttl_min = int(cfg.get("orchestrator_budget_ttl_minutes", 120))

        if use_orch:
            orch_budget = await _get_orchestrator_budget(conn, module, ttl_min)
            if orch_budget is not None:
                budget_usd = orch_budget
                budget_source = "orchestrator"
            else:
                budget_usd = await _get_static_budget(conn, module, default_usd=0.0)
                budget_source = "static_config"
        else:
            budget_usd = await _get_static_budget(conn, module, default_usd=0.0)
            budget_source = "static_config"

        # --- 2. Read current committed capital ----------------------------------
        all_committed = await get_all_committed(conn)
        module_committed = all_committed.get(module, 0.0)

        # --- 3. Per-module budget check -----------------------------------------
        if budget_usd > 0:
            projected_module = module_committed + requested_usd
            if projected_module > budget_usd:
                reason = (
                    f"module budget exceeded: {module} committed "
                    f"${module_committed:.2f} + requested ${requested_usd:.2f} "
                    f"= ${projected_module:.2f} > budget ${budget_usd:.2f} "
                    f"(source={budget_source})"
                )
                return self._veto(reason, module)

        # --- 4. Wallet-group cap check ------------------------------------------
        evm_modules: List[str] = cfg.get("evm_wallet_modules", _DEFAULT_EVM_MODULES)
        sol_modules: List[str] = cfg.get("solana_wallet_modules", _DEFAULT_SOLANA_MODULES)

        if module in evm_modules:
            group_cap = float(cfg.get("global_evm_wallet_cap_usd", 0.0))
            group_total = sum(all_committed.get(m, 0.0) for m in evm_modules)
            group_name = "evm_wallet"
        elif module in sol_modules:
            group_cap = float(cfg.get("global_solana_wallet_cap_usd", 0.0))
            group_total = sum(all_committed.get(m, 0.0) for m in sol_modules)
            group_name = "solana_wallet"
        else:
            group_cap = 0.0
            group_total = 0.0
            group_name = "no_group"

        if group_cap > 0:
            projected_group = group_total + requested_usd
            if projected_group > group_cap:
                reason = (
                    f"wallet group cap exceeded: {group_name} total "
                    f"${group_total:.2f} + requested ${requested_usd:.2f} "
                    f"= ${projected_group:.2f} > cap ${group_cap:.2f} "
                    f"(module={module}, group={evm_modules if group_name=='evm_wallet' else sol_modules})"
                )
                return self._veto(reason, module)

        # --- 5. Global total cap -----------------------------------------------
        global_cap = float(cfg.get("global_total_cap_usd", 0.0))
        if global_cap > 0:
            total_committed = sum(all_committed.values())
            projected_total = total_committed + requested_usd
            if projected_total > global_cap:
                reason = (
                    f"global total cap exceeded: all-modules committed "
                    f"${total_committed:.2f} + requested ${requested_usd:.2f} "
                    f"= ${projected_total:.2f} > global cap ${global_cap:.2f}"
                )
                return self._veto(reason, module)

        logger.debug(
            "allocation_guard ALLOW: module=%s requested=$%.2f "
            "module_committed=$%.2f budget=$%.2f (source=%s)",
            module, requested_usd, module_committed, budget_usd, budget_source,
        )
        return True, "ok"

    def _veto(self, reason: str, module: str) -> Tuple[bool, str]:
        """Log and return the veto decision. In DRY_RUN always allow."""
        if self._dry_run:
            logger.warning("[DRY_RUN] allocation veto (would block): %s", reason)
            return True, f"[DRY_RUN] veto logged (not enforced): {reason}"
        logger.warning("allocation veto: %s", reason)
        return False, reason

    async def get_committed_summary(self) -> Dict:
        """Return a diagnostic summary: per-module committed capital +
        budget + headroom. For dashboard/logging use. Never raises."""
        try:
            async with self._pool.acquire() as conn:
                cfg = await _get_guard_config(conn)
                all_committed = await get_all_committed(conn)
                use_orch = cfg.get("use_orchestrator_budget", True)
                ttl_min = int(cfg.get("orchestrator_budget_ttl_minutes", 120))
                summary: Dict = {
                    "enabled": cfg.get("enabled", True),
                    "dry_run": self._dry_run,
                    "global_evm_cap": cfg.get("global_evm_wallet_cap_usd", 0.0),
                    "global_solana_cap": cfg.get("global_solana_wallet_cap_usd", 0.0),
                    "global_total_cap": cfg.get("global_total_cap_usd", 0.0),
                    "total_committed": sum(all_committed.values()),
                    "modules": {},
                }
                for module, committed in all_committed.items():
                    budget = 0.0
                    source = "static_config"
                    if use_orch:
                        ob = await _get_orchestrator_budget(conn, module, ttl_min)
                        if ob is not None:
                            budget = ob
                            source = "orchestrator"
                    if budget == 0.0:
                        budget = await _get_static_budget(conn, module)
                    headroom = (budget - committed) if budget > 0 else float("inf")
                    summary["modules"][module] = {
                        "committed_usd": round(committed, 2),
                        "budget_usd": round(budget, 2) if budget > 0 else "unlimited",
                        "headroom_usd": round(headroom, 2) if budget > 0 else "unlimited",
                        "budget_source": source,
                    }
                return summary
        except Exception as exc:
            logger.warning("get_committed_summary fail-soft: %s", exc)
            return {"error": str(exc)}


__all__ = [
    "AllocationGuard",
    "get_all_committed",
]
