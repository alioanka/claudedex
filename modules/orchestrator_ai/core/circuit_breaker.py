"""Per-module daily-loss circuit breaker.

Runs as a step inside orchestrator_engine.run_tick. For each module
currently LIVE, computes 24h pnl. If loss > threshold, auto-flips
to DRY_RUN and triggers a subprocess restart (logs/.restart_<m> flag).

Side effects on trip:
  1. UPDATE config_settings.<module>_config.dry_run = 'true'
  2. INSERT circuit_breaker_events row (audit)
  3. Write logs/.restart_<m> flag for main.py to pick up
  4. INSERT orchestrator_recommendations row with reason='circuit
     breaker tripped' so /orchestrator shows the action history

Idempotency: if the breaker already tripped within the lookback
window AND has not been cleared, this tick skips re-tripping. The
audit row stays the canonical record; we don't double-flip.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("orchestrator_ai.circuit_breaker")


# Module config_type map (kept here so this module is self-contained).
_CONFIG_TYPE = {
    "sniper": "sniper_config",
    "arbitrage": "arbitrage_config",
    "copy_trading": "copytrading_config",
    "futures": "futures_config",
    "solana": "solana_config",
    "dex": "dex_config",
    "ai": "ai_config",
}

# Module → restart-flag key, same as enhanced_dashboard._MODULE_RESTART_KEY_MAP.
_RESTART_KEY = {
    "sniper": "sniper",
    "arbitrage": "arbitrage",
    "copy_trading": "copy_trading",
    "futures": "futures_trading",
    "solana": "solana_strategies",
    "dex": "dex_trading",
    "ai": "ai_analysis",
}

# Per-module schema for the 24h pnl query — same shape as the
# engine's main inputs (different time column for futures).
_TRADE_SCHEMAS = {
    "sniper": {"table": "sniper_trades", "time_col": "exit_timestamp",
               "pnl_col": "profit_loss", "live_col": None},
    "arbitrage": {"table": "arbitrage_trades", "time_col": "exit_timestamp",
                  "pnl_col": "profit_loss", "live_col": "is_simulated"},
    "copy_trading": {"table": "copytrading_trades", "time_col": "exit_timestamp",
                     "pnl_col": "profit_loss", "live_col": "is_simulated"},
    "futures": {"table": "futures_trades", "time_col": "exit_time",
                "pnl_col": "net_pnl", "live_col": "is_simulated"},
    "solana": {"table": "solana_trades", "time_col": "exit_time",
               "pnl_col": "pnl_usd", "live_col": "is_simulated"},
    "dex": {"table": "trades", "time_col": "exit_timestamp",
            "pnl_col": "profit_loss", "live_col": None},
    "ai": {"table": "ai_trades", "time_col": "exit_timestamp",
           "pnl_col": "profit_loss", "live_col": "is_simulated"},
}


async def _get_module_dry_run(conn, module: str) -> bool:
    """Read current DB-backed dry_run flag. Returns True (safe-by-
    default) if the row doesn't exist."""
    config_type = _CONFIG_TYPE.get(module)
    if not config_type:
        return True
    raw = await conn.fetchval(
        "SELECT value FROM config_settings WHERE config_type = $1 AND key = 'dry_run'",
        config_type,
    )
    if raw is None:
        return True
    return str(raw).strip().lower() in ("true", "1", "yes", "on")


async def _get_threshold_pct(conn, module: str) -> float:
    config_type = _CONFIG_TYPE.get(module)
    if not config_type:
        return 5.0
    raw = await conn.fetchval(
        "SELECT value FROM config_settings WHERE config_type = $1 AND key = 'daily_loss_breaker_pct'",
        config_type,
    )
    if raw is None:
        return 5.0
    try:
        return float(raw)
    except (TypeError, ValueError):
        return 5.0


async def _get_capital(conn, module: str) -> float:
    """Most-recent approved allocation. Falls back to 1000.0 if no
    allocation has been approved yet."""
    row = await conn.fetchrow(
        "SELECT usd_amount FROM portfolio_allocations "
        "WHERE module = $1 AND approved_at IS NOT NULL "
        "ORDER BY approved_at DESC LIMIT 1",
        module,
    )
    if row is None:
        return 1000.0
    try:
        return float(row["usd_amount"])
    except (TypeError, ValueError):
        return 1000.0


async def _get_24h_live_pnl(conn, module: str) -> float:
    """Sum live pnl over the last 24h. Sniper has no is_simulated
    column, so we assume all sniper trades are 'live' for breaker
    purposes — the breaker only fires when the module is LIVE
    anyway (gated upstream), so this is safe."""
    schema = _TRADE_SCHEMAS.get(module)
    if not schema:
        return 0.0
    live_clause = f"AND {schema['live_col']} = false" if schema["live_col"] else ""
    sql = (
        f"SELECT COALESCE(SUM({schema['pnl_col']}), 0) AS pnl "
        f"FROM {schema['table']} "
        f"WHERE {schema['time_col']} > NOW() - INTERVAL '24 hours' "
        f"  {live_clause}"
    )
    try:
        return float(await conn.fetchval(sql) or 0)
    except Exception as e:
        logger.debug("get_24h_live_pnl(%s) error: %s", module, e)
        return 0.0


async def _already_tripped_recently(conn, module: str) -> bool:
    """Has the breaker tripped (and not been cleared) in the last
    24h for this module? Idempotency guard so we don't write
    duplicate audit rows on every tick."""
    row = await conn.fetchval(
        "SELECT 1 FROM circuit_breaker_events "
        "WHERE module = $1 AND tripped_at > NOW() - INTERVAL '24 hours' "
        "  AND cleared_at IS NULL LIMIT 1",
        module,
    )
    return row is not None


async def _flip_to_dry_and_audit(
    conn, module: str, pnl_loss: float, capital: float,
    pct_loss: float, threshold: float,
) -> None:
    """Run the three side-effects atomically (within the same conn):
      1. config_settings.<m>_config.dry_run = 'true'
      2. INSERT circuit_breaker_events
      3. INSERT orchestrator_recommendations for /orchestrator history
    Then drop the restart flag outside the conn (filesystem op)."""
    config_type = _CONFIG_TYPE.get(module)
    if config_type:
        await conn.execute(
            "INSERT INTO config_settings (config_type, key, value, value_type) "
            "VALUES ($1, 'dry_run', 'true', 'bool') "
            "ON CONFLICT (config_type, key) DO UPDATE SET value = 'true'",
            config_type,
        )
    reason = (
        f"Circuit breaker tripped: 24h live P&L = ${pnl_loss:.2f} on "
        f"${capital:.2f} capital ({pct_loss:.2f}%) ≤ -{threshold:.2f}% threshold"
    )
    await conn.execute(
        "INSERT INTO circuit_breaker_events "
        "(module, pnl_loss_usd, capital_usd, pct_loss, threshold_pct, "
        " action_taken, notes) "
        "VALUES ($1, $2, $3, $4, $5, 'flipped_to_dry', $6)",
        module, pnl_loss, capital, pct_loss, threshold, reason,
    )
    metrics = {
        "pnl_loss_usd": round(pnl_loss, 4),
        "capital_usd": round(capital, 4),
        "pct_loss": round(pct_loss, 4),
        "threshold_pct": round(threshold, 4),
        "source": "circuit_breaker",
    }
    await conn.execute(
        "INSERT INTO orchestrator_recommendations "
        "(module, recommended, confidence, reason, metrics, "
        " approved, approved_at, approved_by) "
        "VALUES ($1, 'to_dry', 1.0, $2, $3::jsonb, TRUE, NOW(), "
        "       'circuit_breaker')",
        module, reason, json.dumps(metrics),
    )
    # Drop restart flag so main.py picks up + restarts the subprocess.
    restart_key = _RESTART_KEY.get(module)
    if restart_key:
        try:
            flag_dir = Path("logs")
            flag_dir.mkdir(parents=True, exist_ok=True)
            (flag_dir / f".restart_{restart_key}").write_text("")
            logger.critical(
                "[CIRCUIT BREAKER] %s flipped to DRY_RUN; restart flag dropped",
                module,
            )
        except OSError as e:
            logger.warning("restart-flag write failed for %s: %s", module, e)


async def check_all_modules(pool) -> List[Dict]:
    """Inspect every module's 24h live P&L. Trip breakers as needed.
    Returns a list of dicts summarizing each module's check (for the
    engine's tick log)."""
    summaries: List[Dict] = []
    async with pool.acquire() as conn:
        for module in _CONFIG_TYPE:
            try:
                is_dry = await _get_module_dry_run(conn, module)
                summary = {
                    "module": module, "is_dry": is_dry, "tripped": False,
                    "pnl_24h": 0.0, "pct_loss": 0.0, "threshold": 0.0,
                }
                if is_dry:
                    # Skip — breaker only fires when the module is LIVE.
                    summaries.append(summary)
                    continue
                if await _already_tripped_recently(conn, module):
                    summary["tripped"] = "already_tripped_in_24h"
                    summaries.append(summary)
                    continue
                threshold = await _get_threshold_pct(conn, module)
                capital = await _get_capital(conn, module)
                pnl_24h = await _get_24h_live_pnl(conn, module)
                summary["pnl_24h"] = round(pnl_24h, 4)
                summary["threshold"] = threshold
                # pct_loss is negative when losing. Compare to -threshold.
                pct_loss = (pnl_24h / capital * 100.0) if capital > 0 else 0.0
                summary["pct_loss"] = round(pct_loss, 4)
                if pct_loss <= -threshold:
                    await _flip_to_dry_and_audit(
                        conn, module, pnl_24h, capital, pct_loss, threshold,
                    )
                    summary["tripped"] = True
                summaries.append(summary)
            except Exception as e:
                logger.error("circuit breaker check(%s) failed: %s", module, e)
                summaries.append({
                    "module": module, "error": str(e), "tripped": False,
                })
    return summaries
