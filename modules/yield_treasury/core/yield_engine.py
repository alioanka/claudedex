"""YIELD_TREASURY engine — observe APRs, size idle float, write ADVICE rows.

One tick:
  1. Read idle capital from the latest treasury_snapshots rows (fresh only).
  2. Read venue APRs (Aave v3 USDC via pool_engine eth_call; LST free APIs).
  3. Run the pure carry math per allowlisted venue.
  4. Persist one yield_treasury_advice row per venue (HOLD rows included —
     a "carry does not pay here" record is the product).

ADVISORY ONLY by default: this build NEVER deposits/withdraws. The gated
live path exists as a gate chain that, even fully open, terminates in
live_deposit_path_not_built (Phase-2 build, per NEW_MODULE_IDEAS #7).
Fail-soft everywhere; never writes logs/.killswitch or pause flags.
Self-test (pure parts): python -m modules.yield_treasury.core.yield_engine
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from modules.yield_treasury.core import yield_math as ym
from modules.yield_treasury.core import yield_sources as ys

logger = logging.getLogger("yield_treasury")

_KILLSWITCH = Path("logs") / ".killswitch"
_PAUSE_FLAG = Path("logs") / ".pause_yield_treasury"

_IDLE_MAX_AGE_MINUTES = 120  # snapshots older than this are stale, not idle data


# ───────────────────────────── config ──────────────────────────────────────

async def load_yield_config(pool) -> dict:
    """config_type='yield_treasury' rows -> typed dict. Fail-soft to {}."""
    out: dict = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings "
                "WHERE config_type='yield_treasury'"
            )
        for r in rows:
            v = r["value"]
            if isinstance(v, str):
                low = v.lower()
                if low in ("true", "false"):
                    v = low == "true"
                elif v.strip().startswith(("[", "{")):
                    try:
                        v = json.loads(v)
                    except (ValueError, TypeError):
                        pass
                else:
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
            out[r["key"]] = v
    except Exception as exc:
        logger.error("load_yield_config fail-soft: %s", exc)
    return out


def _f(cfg: dict, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


# ───────────────────────────── venue assembly (pure) ─────────────────────────

def build_venues(cfg: dict, idle: dict) -> List[dict]:
    """Allowlisted venues x current idle balances -> evaluation specs.

    idle = {'stable_usd': {chain: usd}, 'native': {chain: units}} from
    treasury_snapshots. Pure; no I/O.
    """
    venues: List[dict] = []
    stable = idle.get("stable_usd", {}) or {}
    native = idle.get("native", {}) or {}

    for chain in ys.AAVE_V3_POOLS:
        venues.append({
            "venue": "aave_v3", "chain": chain, "asset": "USDC", "unit": "USD",
            "idle": float(stable.get(chain, 0.0) or 0.0),
            "floor": _f(cfg, "undeployed_floor_usd", 100.0),
            "fixed_cost": _f(cfg, f"roundtrip_cost_usd_{chain}", 16.0),
            "fee_bps": 0.0,
            "latency_s": _f(cfg, "withdrawal_latency_s_aave", 120.0),
            "min_deployable": _f(cfg, "min_idle_usd", 250.0),
        })

    venues.append({
        "venue": "jito_jitosol", "chain": "solana", "asset": "SOL", "unit": "SOL",
        "idle": float(native.get("solana", 0.0) or 0.0),
        "floor": _f(cfg, "undeployed_floor_sol", 0.2),
        "fixed_cost": _f(cfg, "sol_tx_cost_sol", 0.001),
        "fee_bps": _f(cfg, "lst_roundtrip_bps", 10.0),
        "latency_s": _f(cfg, "withdrawal_latency_s_jitosol", 120.0),
        "min_deployable": _f(cfg, "min_idle_sol", 1.0),
    })

    venues.append({
        "venue": "lido_steth", "chain": "ethereum", "asset": "ETH", "unit": "ETH",
        "idle": float(native.get("ethereum", 0.0) or 0.0),
        "floor": _f(cfg, "undeployed_floor_eth", 0.02),
        "fixed_cost": _f(cfg, "eth_lst_roundtrip_cost_eth", 0.004),
        "fee_bps": _f(cfg, "lst_roundtrip_bps", 10.0),
        # stETH exit is the withdrawal queue (days) — swap-out shortcut is a
        # Phase-2 question; advised conservatively.
        "latency_s": _f(cfg, "withdrawal_latency_s_steth", 259200.0),
        "min_deployable": _f(cfg, "min_idle_eth", 0.05),
    })
    return venues


def evaluate_venue(v: dict, apr: Optional[float], cfg: dict) -> dict:
    """Pure: one venue spec + observed APR -> advice record fields."""
    deployable = ym.deployable_amount(
        v["idle"], v["floor"], _f(cfg, "max_deploy_frac", 0.5),
        _f(cfg, "venue_cap_frac", 0.25))
    if apr is None:
        advice = {"gross_daily": 0.0, "roundtrip_cost": 0.0,
                  "breakeven_days": None, "net_horizon": 0.0,
                  "recommendation": ym.HOLD, "reason": "apr_unavailable"}
    else:
        advice = ym.carry_advice(
            deployable=deployable, apr=apr,
            fixed_cost=v["fixed_cost"], fee_bps=v["fee_bps"],
            horizon_days=_f(cfg, "horizon_days", 30.0),
            max_breakeven_days=_f(cfg, "max_breakeven_days", 10.0),
            withdrawal_latency_s=v["latency_s"],
            max_withdrawal_latency_s=_f(cfg, "max_withdrawal_latency_s", 86400.0),
            min_deployable=v["min_deployable"])
    return {**v, "apr": apr, "deployable": deployable, **advice,
            "latency_note": ym.latency_note(v["latency_s"])}


# ───────────────────────────── gated live path (NOT default) ─────────────────

async def resolve_live_skip_reason(cfg: dict, module_dry_run: bool,
                                   risk_manager, venue_key: str,
                                   amount: float) -> str:
    """Polymarket-shaped gate chain. Returns a non-empty skip reason in THIS
    build always: even with every gate open the deposit path is not built
    (Phase 2, only after treasury Phase 2 is proven). Never raises."""
    if cfg.get("shadow_mode", True):
        return "shadow_mode"
    if not cfg.get("live_execution_enabled", False):
        return "live_execution_disabled"
    try:
        from core.dry_run import should_skip_live
        if should_skip_live(module_dry_run, module="yield_treasury"):
            return "dry_run_or_killswitch_or_pause"
    except Exception as exc:
        logger.error("should_skip_live unavailable (%s) — refusing live", exc)
        return "dry_run_gate_unavailable"
    if risk_manager is None:
        return "risk_manager_missing"
    try:
        ok, reason = await risk_manager.validate_trade(venue_key, amount)
        if not ok:
            return f"risk_manager:{reason}"[:64]
    except Exception as exc:
        logger.error("RiskManager.validate_trade error: %s — refusing live", exc)
        return "risk_manager_error"
    return "live_deposit_path_not_built"


# ───────────────────────────── idle capital read ──────────────────────────────

async def get_idle_balances(conn) -> Optional[dict]:
    """Latest fresh treasury_snapshots row per wallet, aggregated per chain.
    Stables in the wallet are idle by definition; native idle is gated by the
    per-venue undeployed floor in the math. None if no fresh data (fail-soft)."""
    try:
        rows = await conn.fetch(
            "SELECT DISTINCT ON (chain, address) chain, native_balance, stable_usd "
            "FROM treasury_snapshots "
            "WHERE created_at > NOW() - make_interval(mins => $1) "
            "ORDER BY chain, address, created_at DESC",
            _IDLE_MAX_AGE_MINUTES)
    except Exception as exc:
        logger.warning("treasury_snapshots unavailable (is the treasury module "
                       "running? mig 121 applied?) — no idle data: %s", exc)
        return None
    if not rows:
        return None
    idle = {"stable_usd": {}, "native": {}}
    for r in rows:
        c = r["chain"]
        idle["stable_usd"][c] = idle["stable_usd"].get(c, 0.0) + float(r["stable_usd"] or 0.0)
        idle["native"][c] = idle["native"].get(c, 0.0) + float(r["native_balance"] or 0.0)
    return idle


# ───────────────────────────── APR observation ────────────────────────────────

async def observe_aprs(session, cfg: dict) -> Dict[str, Optional[float]]:
    """{'aave_v3:<chain>': apr, 'jito_jitosol:solana': apy, 'lido_steth:ethereum': apr}.
    Aave reads go through pool_engine with success/failure/429 reports."""
    out: Dict[str, Optional[float]] = {}

    rpc = None
    try:
        from config.pool_engine import PoolEngine
        rpc = await PoolEngine.get_instance()
    except Exception as exc:
        logger.warning("pool_engine unavailable — Aave APRs skipped: %s", exc)

    for chain in ys.AAVE_V3_POOLS:
        key = f"aave_v3:{chain}"
        out[key] = None
        if rpc is None:
            continue
        ptype = f"{chain.upper()}_RPC"
        try:
            rpc_url = await rpc.get_endpoint(ptype)
        except Exception as exc:
            logger.debug("get_endpoint(%s) fail-soft: %s", ptype, exc)
            rpc_url = None
        if not rpc_url:
            continue
        started = datetime.utcnow()
        try:
            out[key] = await ys.aave_v3_usdc_supply_apr(session, rpc_url, chain)
            ms = int((datetime.utcnow() - started).total_seconds() * 1000)
            await rpc.report_success(ptype, rpc_url, latency_ms=ms)
        except ys.RateLimited:
            await rpc.report_rate_limit(ptype, rpc_url)
            logger.warning("RPC 429 reading Aave APR on %s — skipped", chain)
        except Exception as exc:
            await rpc.report_failure(ptype, rpc_url, error_type="network_error",
                                     error_message=str(exc)[:200])
            logger.warning("Aave APR read failed on %s: %s", chain, exc)

    out["jito_jitosol:solana"] = await ys.fetch_json_rate(
        session, str(cfg.get("jitosol_apy_url", ys.JITOSOL_APY_URL_DEFAULT)))
    out["lido_steth:ethereum"] = await ys.fetch_json_rate(
        session, str(cfg.get("lido_apr_url", ys.LIDO_APR_URL_DEFAULT)))
    return out


# ───────────────────────────── persistence ────────────────────────────────────

async def _persist_advice(conn, rec: dict, skip_reason: Optional[str]) -> None:
    try:
        await conn.execute(
            "INSERT INTO yield_treasury_advice "
            "(venue, chain, asset, unit, apr, idle_amount, deployable_amount, "
            " gross_daily, roundtrip_cost, breakeven_days, horizon_days, "
            " net_horizon, withdrawal_latency_s, latency_note, recommendation, "
            " reason, shadow, details, created_at) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14,$15,$16,"
            " TRUE,$17, NOW())",
            rec["venue"], rec["chain"], rec["asset"], rec["unit"], rec["apr"],
            float(rec["idle"]), float(rec["deployable"]),
            float(rec["gross_daily"]), float(rec["roundtrip_cost"]),
            rec["breakeven_days"], float(rec.get("horizon_days_used", 0.0)),
            float(rec["net_horizon"]), int(rec["latency_s"]),
            rec["latency_note"], rec["recommendation"], rec["reason"],
            json.dumps({"live_skip_reason": skip_reason} if skip_reason else {}),
        )
    except Exception as exc:
        logger.error("yield_treasury_advice insert failed for %s/%s: %s",
                     rec["venue"], rec["chain"], exc)


# ───────────────────────────── tick + loop ────────────────────────────────────

async def run_tick(pool, cfg: dict) -> dict:
    """One advise cycle. Returns a summary dict for /status."""
    import aiohttp

    summary: dict = {"venues": 0, "advised": 0, "deploy_candidates": 0,
                     "detail": {}}
    async with pool.acquire() as conn:
        idle = await get_idle_balances(conn)
        if idle is None:
            summary["detail"]["note"] = "no_fresh_treasury_snapshots"
            logger.info("no fresh idle data (treasury module snapshots required) "
                        "— idle tick")
            return summary

        async with aiohttp.ClientSession() as session:
            aprs = await observe_aprs(session, cfg)

        horizon = _f(cfg, "horizon_days", 30.0)
        module_dry_run = (os.getenv("YIELD_TREASURY_DRY_RUN", "true")
                          .lower() != "false")
        for v in build_venues(cfg, idle):
            summary["venues"] += 1
            rec = evaluate_venue(v, aprs.get(f"{v['venue']}:{v['chain']}"), cfg)
            rec["horizon_days_used"] = horizon

            skip_reason = None
            if rec["recommendation"] == ym.DEPLOY_CANDIDATE:
                summary["deploy_candidates"] += 1
                # Gated live path — NOT the default; terminates in
                # live_deposit_path_not_built even when every gate is open.
                skip_reason = await resolve_live_skip_reason(
                    cfg, module_dry_run, None, f"{rec['venue']}:{rec['chain']}",
                    rec["deployable"])
                logger.warning(
                    "YIELD ADVICE [%s/%s] park %.4f %s at %.2f%% -> "
                    "+%.4f %s/day, breakeven %.1fd (ADVISORY; live gate: %s). "
                    "Caveat: %s",
                    rec["venue"], rec["chain"], rec["deployable"], rec["unit"],
                    (rec["apr"] or 0) * 100, rec["gross_daily"], rec["unit"],
                    rec["breakeven_days"] or -1, skip_reason, rec["latency_note"])
            await _persist_advice(conn, rec, skip_reason)
            summary["advised"] += 1
            summary["detail"][f"{rec['venue']}:{rec['chain']}"] = {
                "apr": rec["apr"], "idle": round(rec["idle"], 4),
                "deployable": round(rec["deployable"], 4),
                "rec": rec["recommendation"], "reason": rec["reason"],
            }
    return summary


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, advise, sleep poll_interval_seconds."""
    while True:
        cfg = await get_config() if get_config else {}
        if _KILLSWITCH.exists():
            logger.info("killswitch present — yield_treasury tick skipped "
                        "(advisory module; it never writes the killswitch)")
        elif _PAUSE_FLAG.exists():
            logger.info("yield_treasury paused (logs/.pause_yield_treasury) "
                        "— tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info("yield tick: %d/%d venues advised, %d deploy "
                            "candidates", s["advised"], s["venues"],
                            s["deploy_candidates"])
            except Exception as exc:
                logger.error("yield tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int(_f(cfg or {}, "poll_interval_seconds", 900)))


# ───────────────────────────── self-test (pure parts) ─────────────────────────

if __name__ == "__main__":
    cfg = {"max_deploy_frac": 0.5, "venue_cap_frac": 0.25,
           "undeployed_floor_usd": 100.0, "min_idle_usd": 50.0,
           "roundtrip_cost_usd_arbitrum": 0.30,
           "horizon_days": 30, "max_breakeven_days": 10,
           "max_withdrawal_latency_s": 86400}
    idle = {"stable_usd": {"arbitrum": 1000.0, "ethereum": 1000.0},
            "native": {"solana": 20.0, "ethereum": 0.5}}

    venues = build_venues(cfg, idle)
    keys = {(v["venue"], v["chain"]) for v in venues}
    assert ("aave_v3", "arbitrum") in keys and ("aave_v3", "base") in keys
    assert ("jito_jitosol", "solana") in keys and ("lido_steth", "ethereum") in keys
    assert len(venues) == 5, keys  # 3 aave chains + 2 LSTs; allowlist is closed

    by = {(v["venue"], v["chain"]): v for v in venues}
    # Aave arbitrum: cheap gas -> deploys at 5%
    r = evaluate_venue(by[("aave_v3", "arbitrum")], 0.05, cfg)
    assert r["deployable"] == 250.0 and r["recommendation"] == ym.DEPLOY_CANDIDATE, r
    # Aave ethereum: default $16 roundtrip -> honest HOLD on the same float
    r = evaluate_venue(by[("aave_v3", "ethereum")], 0.05, cfg)
    assert r["recommendation"] == ym.HOLD and r["reason"] == "breakeven_too_slow", r
    # stETH: 3-day withdrawal queue > 1-day latency cap -> HOLD, with the caveat
    r = evaluate_venue(by[("lido_steth", "ethereum")], 0.03, cfg)
    assert r["reason"] == "withdrawal_latency_exceeds_cap", r
    assert "cannot fund trading" in r["latency_note"]
    # APR feed down -> HOLD apr_unavailable, never a deploy on stale yield
    r = evaluate_venue(by[("jito_jitosol", "solana")], None, cfg)
    assert r["reason"] == "apr_unavailable" and r["recommendation"] == ym.HOLD, r

    # gate chain order: shadow first, then live flag, then dry-run gate
    async def _gates():
        assert await resolve_live_skip_reason({}, True, None, "v", 1.0) == "shadow_mode"
        assert await resolve_live_skip_reason(
            {"shadow_mode": False}, True, None, "v", 1.0) == "live_execution_disabled"
        r = await resolve_live_skip_reason(
            {"shadow_mode": False, "live_execution_enabled": True}, True, None,
            "v", 1.0)
        assert r == "dry_run_or_killswitch_or_pause", r  # module_dry_run=True
        r = await resolve_live_skip_reason(
            {"shadow_mode": False, "live_execution_enabled": True}, False, None,
            "v", 1.0)
        assert r in ("risk_manager_missing",
                     "dry_run_or_killswitch_or_pause"), r  # killswitch may exist
    asyncio.run(_gates())
    print("yield_engine self-test OK")
