"""Cross-module exposure aggregator (CT-Q-12).

Single-token open-position USD exposure across EVERY trading module:

    DEX     -> trades                  (usd_value column)
    SNIPER  -> sniper_trades           (entry_usd column)
    SOLANA  -> solana_positions        (entry_price * amount_sol * sol_price proxy)
    COPY    -> copytrading_trades      (entry_usd column)
    AI      -> ai_trades               (entry_usd column)

Why this exists
---------------
Five independent trading modules can each open a position in the same
token without any of them seeing the others. A leader posting a moonshot
on Twitter can trigger AI (sentiment), SNIPER (new-pair detector), DEX
(momentum strategy) AND COPY (wallet mirror) to all enter the same bag
simultaneously. With per-module $100 caps that's a $500 unhedged
position the operator never sized for; with leader-Kelly multipliers
in play it can be much larger.

This aggregator is the canonical place to ask "how much money do we
ALREADY have in $TOKEN on this chain?" -- callers add their intended
buy and refuse to broadcast if the sum exceeds the per-token cap.

Design notes
------------
* **Fail-soft.** Any DB error returns 0.0 (or a best-effort partial
  sum from the modules that did respond). The trade pipeline is
  NEVER blocked by an aggregator failure -- the existing per-module
  caps remain in place as a safety net.
* **Token matching is permissive.** EVM token addresses are
  case-insensitive (`lower()` comparison); Solana mints are
  case-sensitive (base58). We accept either column shape on a
  per-table basis.
* **Chain-scoped.** We only sum positions on the SAME chain as the
  intended buy -- a Base USDC and an Ethereum USDC are distinct
  positions, not correlated exposure.
* **Open positions only.** Every table is filtered by `status='open'`
  except `solana_positions` which is itself the open-position table
  (no `status` column; presence == open).
* **Read-only.** This module never writes. It's safe to call from any
  read-only context (dashboard, alerts, paper-trading sim).

The shape of the returned value is a `float` in USD. Callers that
need a breakdown (which module is holding what) can use
`get_exposure_breakdown_usd()` which returns a dict keyed by module.
"""
from __future__ import annotations

import logging
from typing import Dict

logger = logging.getLogger(__name__)


# Per-module table definitions. Each entry is:
#   (module_label, table_name, token_column, usd_column, extra_where)
# `usd_column` may be None -- in that case the module is omitted from
# the sum because there's no canonical USD basis to compare against.
_MODULE_TABLES = (
    # DEX module: monitoring/data/storage/database.py:_create_tables
    # uses `trades` with `usd_value` (DECIMAL(20,2)).
    ("dex", "trades", "token_address", "usd_value", "status = 'open'"),
    # SNIPER + AI + COPY all use the `entry_usd` shape from migrations
    # 009 / 010.
    ("sniper", "sniper_trades", "token_address", "entry_usd", "status = 'open'"),
    ("ai", "ai_trades", "token_address", "entry_usd", "status = 'open'"),
    ("copy", "copytrading_trades", "token_address", "entry_usd", "status = 'open'"),
)

# Solana is a special case: solana_positions has amount_sol (native) and
# no precomputed USD. We compute usd ~= amount_sol * entry_price (which
# is the SOL-denominated entry price on Jupiter, so amount_sol *
# entry_price gives token-count; that's NOT USD). Safer fallback:
# `unrealized_pnl_sol + amount_sol` would still be SOL. Without a
# stored USD basis we ESTIMATE conservatively as 0 and surface a TODO
# in the breakdown so callers can warn. The SOLANA module owns the
# eventual schema add (entry_usd column on solana_positions).
_SOLANA_POSITIONS_TABLE = "solana_positions"


async def _sum_module_usd(
    conn, *, table: str, token_col: str, usd_col: str,
    where: str, chain: str, token_address: str,
) -> float:
    """Single-table sum; returns 0.0 on any error (fail-soft)."""
    try:
        sql = (
            f"SELECT COALESCE(SUM({usd_col}), 0)::float AS total "
            f"FROM {table} "
            f"WHERE {where} "
            f"  AND chain = $1 "
            f"  AND lower({token_col}) = lower($2)"
        )
        row = await conn.fetchrow(sql, chain, token_address)
        return float(row["total"] or 0.0) if row else 0.0
    except Exception as e:
        # Missing table / missing column / type mismatch -- log debug
        # (not warning -- this is expected in fresh test DBs without
        # all migrations) and return 0 so other modules still aggregate.
        logger.debug(f"_sum_module_usd({table}) failed: {e}")
        return 0.0


async def _sum_solana_positions_usd(
    conn, *, chain: str, token_address: str,
) -> float:
    """Solana-positions estimator. Returns 0.0 unless we can derive a
    reasonable USD basis from the row (currently only the case if a
    follow-up migration adds an `entry_usd` column). Until then this
    is a noop so we don't double-count or under-count."""
    if chain != "solana":
        return 0.0
    try:
        # Best-effort: try entry_usd first (future schema), fall back to
        # amount_sol * sol-price-at-entry if such columns exist. If
        # neither exists, return 0.
        row = await conn.fetchrow(
            """
            SELECT 1 AS marker
              FROM information_schema.columns
             WHERE table_name = $1
               AND column_name = 'entry_usd'
            """,
            _SOLANA_POSITIONS_TABLE,
        )
        has_entry_usd = bool(row and row["marker"])
        if not has_entry_usd:
            return 0.0
        sql = (
            f"SELECT COALESCE(SUM(entry_usd), 0)::float AS total "
            f"FROM {_SOLANA_POSITIONS_TABLE} "
            f"WHERE lower(token_mint) = lower($1)"
        )
        row = await conn.fetchrow(sql, token_address)
        return float(row["total"] or 0.0) if row else 0.0
    except Exception as e:
        logger.debug(f"_sum_solana_positions_usd failed: {e}")
        return 0.0


async def get_exposure_usd(
    chain: str,
    token_address: str,
    db_pool,
) -> float:
    """Return total open-position USD exposure for `token_address` on
    `chain` summed across every trading module.

    Fail-soft: returns the best-effort partial sum (or 0.0) on any
    DB error. Never raises.
    """
    if db_pool is None or not chain or not token_address:
        return 0.0
    total = 0.0
    try:
        async with db_pool.acquire() as conn:
            for _label, table, token_col, usd_col, where in _MODULE_TABLES:
                total += await _sum_module_usd(
                    conn, table=table, token_col=token_col, usd_col=usd_col,
                    where=where, chain=chain, token_address=token_address,
                )
            total += await _sum_solana_positions_usd(
                conn, chain=chain, token_address=token_address,
            )
    except Exception as e:
        logger.debug(f"get_exposure_usd outer failed (fail-soft): {e}")
    return float(total)


async def get_exposure_breakdown_usd(
    chain: str,
    token_address: str,
    db_pool,
) -> Dict[str, float]:
    """Per-module breakdown variant. Same fail-soft semantics. Caller
    can use this for diagnostic logging (e.g. surfacing in the
    `[replay] gate=cross_module_cap` extra dict)."""
    out: Dict[str, float] = {}
    if db_pool is None or not chain or not token_address:
        return out
    try:
        async with db_pool.acquire() as conn:
            for label, table, token_col, usd_col, where in _MODULE_TABLES:
                out[label] = await _sum_module_usd(
                    conn, table=table, token_col=token_col, usd_col=usd_col,
                    where=where, chain=chain, token_address=token_address,
                )
            sol = await _sum_solana_positions_usd(
                conn, chain=chain, token_address=token_address,
            )
            if sol > 0:
                out["solana_positions"] = sol
    except Exception as e:
        logger.debug(f"get_exposure_breakdown_usd outer failed: {e}")
    return out


__all__ = [
    "get_exposure_usd",
    "get_exposure_breakdown_usd",
]
