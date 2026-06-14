"""TREASURY engine — observe → reconcile → alert → persist. Phase 1 OBSERVE-ONLY.

One tick, per observed wallet (public addresses only, never keys):
  1. Resolve an RPC URL via the shared pool_engine (report success/failure/429).
  2. Read native balance + configured key-token balances (read-only RPC).
  3. Reconcile against the trade ledgers (open LIVE positions per chain).
  4. Evaluate alert conditions (gas_low / hot_wallet_high / reconcile_drift /
     ledger_unbacked) and log them to the module error log.
  5. Persist one treasury_snapshots row per wallet.

NEVER signs, NEVER transfers, NEVER writes logs/.killswitch or pause flags.
Fail-soft everywhere: an RPC/ledger error skips that chain/table, tick continues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from modules.treasury.core import balance_reader as br

logger = logging.getLogger("treasury")

_KILLSWITCH = Path("logs") / ".killswitch"
_PAUSE_FLAG = Path("logs") / ".pause_treasury"

_NATIVE_SYMBOL = {
    "ethereum": "ETH", "arbitrum": "ETH", "base": "ETH", "bsc": "BNB",
    "polygon": "POL", "avalanche": "AVAX", "fantom": "FTM",
    "cronos": "CRO", "pulsechain": "PLS", "monad": "MON", "solana": "SOL",
}

# Open-position ledgers per chain. usd_col=None -> exposure tracked in native
# units only (solana_positions stores value_sol). live_filter keeps DRY rows
# out of the reconcile (simulated fills never touch the wallet).
_LEDGER_QUERIES = [
    {"module": "dex", "table": "positions", "chain_col": "chain",
     "usd_col": "usd_value", "live_filter": ""},
    {"module": "sniper", "table": "sniper_positions", "chain_col": "chain",
     "usd_col": "entry_usd", "live_filter": ""},
    {"module": "arbitrage", "table": "arbitrage_positions", "chain_col": "chain",
     "usd_col": "entry_usd", "live_filter": ""},
    {"module": "copy_trading", "table": "copytrading_positions", "chain_col": "chain",
     "usd_col": "entry_usd", "live_filter": ""},
    {"module": "ai", "table": "ai_positions", "chain_col": "chain",
     "usd_col": "entry_usd", "live_filter": ""},
    # solana_positions: no status/chain columns — every row is an open
    # solana-chain position; is_simulated marks DRY rows.
    {"module": "solana", "table": "solana_positions", "chain_col": None,
     "usd_col": None, "native_col": "value_sol",
     "live_filter": "NOT is_simulated", "fixed_chain": "solana"},
]


# ───────────────────────────── config ──────────────────────────────────────

async def load_treasury_config(pool) -> dict:
    """config_type='treasury' rows -> typed dict. Fail-soft to {}."""
    out: dict = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type='treasury'"
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
        logger.error("load_treasury_config fail-soft: %s", exc)
    return out


def _cfg_float(cfg: dict, key: str, default: float) -> float:
    try:
        return float(cfg.get(key, default))
    except (TypeError, ValueError):
        return default


def gas_floor_for(cfg: dict, chain: str) -> float:
    """Per-chain native low-water mark, falling back to gas_floor_default."""
    return _cfg_float(cfg, f"gas_floor_{chain}",
                      _cfg_float(cfg, "gas_floor_default", 0.005))


# ───────────────────────────── wallet discovery ─────────────────────────────

def _derive_evm_address() -> Optional[str]:
    """Public EVM address DERIVED from the secrets-managed PRIVATE_KEY. Reads the
    key only to compute its public address — never signs, never persists it.
    Fail-soft -> None."""
    try:
        from security.secrets_manager import secrets
        pk = (secrets.get('PRIVATE_KEY') or '').strip()
        if not pk or pk.lower().startswith('your'):
            return None
        from eth_account import Account
        return Account.from_key(pk).address
    except Exception as exc:
        logger.debug("EVM address derivation failed (fail-soft): %s", exc)
        return None


def _solana_pubkey_from_secret(raw: str) -> Optional[str]:
    """Derive a Solana pubkey string from a secret in base58 or JSON-array form."""
    raw = (raw or '').strip()
    if not raw or raw.lower().startswith('your'):
        return None
    from solders.keypair import Keypair
    if raw.startswith('['):
        kp = Keypair.from_bytes(bytes(json.loads(raw)))
    else:
        import base58
        kp = Keypair.from_bytes(base58.b58decode(raw))
    return str(kp.pubkey())


def _derive_solana_address(secret_key_name: str) -> Optional[str]:
    """Public Solana address DERIVED from a secrets-managed private key. Fail-soft."""
    try:
        from security.secrets_manager import secrets
        return _solana_pubkey_from_secret(secrets.get(secret_key_name) or '')
    except Exception as exc:
        logger.debug("Solana address derivation (%s) failed (fail-soft): %s",
                     secret_key_name, exc)
        return None


def discover_wallets(cfg: dict) -> List[dict]:
    """Wallet PUBLIC addresses are DERIVED from the secrets-managed private keys
    (PRIVATE_KEY / SOLANA_PRIVATE_KEY / SOLANA_MODULE_PRIVATE_KEY) so the observed
    address can never drift from the actual signer. The key is read ONLY to
    compute its public address — this module never signs, never transfers, never
    persists the key. A stored env address is used ONLY as a fallback when the PK
    is unavailable (fresh install / key not yet loaded). + cfg extra_wallets."""
    wallets: List[dict] = []
    chains = cfg.get("evm_chains", "ethereum,arbitrum,base")
    if isinstance(chains, str):
        chains = [c.strip().lower() for c in chains.split(",") if c.strip()]

    # EVM: derive from PRIVATE_KEY; env WALLET_ADDRESS is fallback only.
    evm_addr = _derive_evm_address()
    if not evm_addr:
        fb = (os.getenv("WALLET_ADDRESS") or "").strip()
        if fb.startswith("0x") and len(fb) == 42 and not fb.lower().startswith("your"):
            evm_addr = fb
    if evm_addr:
        for chain in chains:
            wallets.append({"group": "evm", "chain": chain, "address": evm_addr})

    # Solana: derive from each private key; env address is fallback only.
    for group, secret_name, env_var in (
        ("dex_solana", "SOLANA_PRIVATE_KEY", "SOLANA_WALLET"),
        ("solana_module", "SOLANA_MODULE_PRIVATE_KEY", "SOLANA_MODULE_WALLET"),
    ):
        addr = _derive_solana_address(secret_name)
        if not addr:
            fb = (os.getenv(env_var) or "").strip()
            if fb and not fb.lower().startswith("your") and 32 <= len(fb) <= 44:
                addr = fb
        if addr:
            wallets.append({"group": group, "chain": "solana", "address": addr})

    extra = cfg.get("extra_wallets", [])
    if isinstance(extra, list):
        for w in extra:
            if isinstance(w, dict) and w.get("chain") and w.get("address"):
                wallets.append({"group": w.get("label", "extra"),
                                "chain": str(w["chain"]).lower(),
                                "address": str(w["address"])})
    return wallets


# ───────────────────────────── ledger reconcile ─────────────────────────────

async def collect_ledger(conn, chain: str) -> dict:
    """Open LIVE positions implied by the trade ledgers for one chain.
    Per-table fail-soft (a missing table/column skips that table)."""
    out = {"open_positions": 0, "exposure_usd": 0.0,
           "exposure_native": 0.0, "tables": {}}
    for q in _LEDGER_QUERIES:
        if q["chain_col"] is None and q.get("fixed_chain") != chain:
            continue
        try:
            filters = []
            args: list = []
            if q["chain_col"] is not None:
                filters.append("status='open'")
                filters.append(f"{q['chain_col']}=$1")
                args.append(chain)
            if q["live_filter"]:
                filters.append(q["live_filter"])
            where = " AND ".join(filters) or "TRUE"
            usd_expr = f"COALESCE(SUM({q['usd_col']}),0)" if q.get("usd_col") else "0"
            nat_expr = f"COALESCE(SUM({q['native_col']}),0)" if q.get("native_col") else "0"
            row = await conn.fetchrow(
                f"SELECT COUNT(*) AS n, {usd_expr} AS usd, {nat_expr} AS nat "
                f"FROM {q['table']} WHERE {where}", *args
            )
            n = int(row["n"] or 0)
            if n > 0:
                out["open_positions"] += n
                out["exposure_usd"] += float(row["usd"] or 0)
                out["exposure_native"] += float(row["nat"] or 0)
                out["tables"][q["module"]] = {
                    "open": n, "usd": round(float(row["usd"] or 0), 2)}
        except Exception as exc:
            logger.debug("ledger %s fail-soft: %s", q["table"], exc)
    return out


# ───────────────────────────── balance reads ────────────────────────────────

def _provider_type(chain: str) -> str:
    return f"{chain.upper()}_RPC"


async def _read_wallet(session, rpc, rpc_url: str, wallet: dict,
                       cfg: dict) -> Optional[dict]:
    """Native + key-token balances for one wallet. None if native read failed."""
    chain, addr = wallet["chain"], wallet["address"]
    ptype = _provider_type(chain)
    started = datetime.utcnow()
    try:
        if chain == "solana":
            native = await br.solana_native_balance(session, rpc_url, addr)
        else:
            native = await br.evm_native_balance(session, rpc_url, addr)
        if rpc:
            ms = int((datetime.utcnow() - started).total_seconds() * 1000)
            await rpc.report_success(ptype, rpc_url, latency_ms=ms)
    except br.RateLimited as exc:
        if rpc:
            await rpc.report_rate_limit(ptype, rpc_url)
        logger.warning("RPC 429 on %s (%s): %s — chain skipped", chain, ptype, exc)
        return None
    except Exception as exc:
        if rpc:
            await rpc.report_failure(ptype, rpc_url,
                                     error_type="network_error",
                                     error_message=str(exc)[:200])
        logger.warning("balance read failed on %s (%s): %s — chain skipped",
                       chain, addr[:10], exc)
        return None
    if native is None:
        return None

    tokens: Dict[str, float] = {}
    stable_usd = 0.0
    token_map = cfg.get("solana_tokens" if chain == "solana" else "evm_tokens", {})
    token_list = token_map.get(chain, []) if isinstance(token_map, dict) else []
    for t in token_list:
        try:
            if chain == "solana":
                bal = await br.solana_token_balance(session, rpc_url, addr, t["mint"])
            else:
                bal = await br.evm_erc20_balance(
                    session, rpc_url, t["address"], addr, int(t.get("decimals", 18)))
            if bal is not None:
                tokens[t["symbol"]] = bal
                if t.get("stable"):
                    stable_usd += bal
        except Exception as exc:
            logger.debug("token %s read fail-soft on %s: %s",
                         t.get("symbol"), chain, exc)
    return {"native": native, "tokens": tokens, "stable_usd": stable_usd}


# ───────────────────────────── alert rules ──────────────────────────────────

def evaluate_alerts(wallet: dict, balances: dict, ledger: dict,
                    prev_stable_usd: Optional[float], cfg: dict) -> List[dict]:
    """Pure rule evaluation -> [{type, severity, message}]. No side effects."""
    alerts: List[dict] = []
    chain, addr = wallet["chain"], wallet["address"]
    native = balances["native"]
    floor = gas_floor_for(cfg, chain)
    sym = _NATIVE_SYMBOL.get(chain, "NATIVE")
    open_live = ledger["open_positions"]

    if native < floor:
        sev = "error" if open_live > 0 else "warning"
        msg = (f"GAS_LOW {chain} {addr}: {native:.6f} {sym} < floor {floor:.6f}"
               + (f" with {open_live} open LIVE positions — exits at risk of "
                  f"gas starvation" if open_live > 0 else ""))
        alerts.append({"type": "gas_low", "severity": sev, "message": msg})

    ceiling = _cfg_float(cfg, "hot_wallet_ceiling_usd", 1000.0)
    if balances["stable_usd"] > ceiling:
        alerts.append({
            "type": "hot_wallet_high", "severity": "warning",
            "message": (f"HOT_WALLET_HIGH {chain} {addr}: "
                        f"{balances['stable_usd']:.2f} USD in stables > ceiling "
                        f"{ceiling:.2f} — unswept profit sitting as hot-wallet risk"),
        })

    drift = _cfg_float(cfg, "reconcile_drift_usd", 250.0)
    if prev_stable_usd is not None and abs(balances["stable_usd"] - prev_stable_usd) > drift:
        alerts.append({
            "type": "reconcile_drift", "severity": "warning",
            "message": (f"RECONCILE_DRIFT {chain} {addr}: stable balance moved "
                        f"{balances['stable_usd'] - prev_stable_usd:+.2f} USD since "
                        f"last snapshot (threshold {drift:.2f}) — reconcile vs ledgers"),
        })

    dust = _cfg_float(cfg, "native_dust", 0.0005)
    if open_live > 0 and native <= dust and balances["stable_usd"] <= 1.0:
        alerts.append({
            "type": "ledger_unbacked", "severity": "error",
            "message": (f"LEDGER_UNBACKED {chain} {addr}: ledgers imply "
                        f"{open_live} open LIVE positions "
                        f"({ledger['exposure_usd']:.2f} USD) but wallet is "
                        f"empty — ledger/on-chain mismatch"),
        })
    return alerts


# ───────────────────────────── persistence ──────────────────────────────────

async def _prev_stable_usd(conn, chain: str, address: str) -> Optional[float]:
    try:
        v = await conn.fetchval(
            "SELECT stable_usd FROM treasury_snapshots "
            "WHERE chain=$1 AND address=$2 ORDER BY created_at DESC LIMIT 1",
            chain, address)
        return float(v) if v is not None else None
    except Exception:
        return None


async def _persist_snapshot(conn, wallet: dict, balances: dict,
                            ledger: dict, alerts: List[dict]) -> None:
    try:
        await conn.execute(
            "INSERT INTO treasury_snapshots "
            "(wallet_group, chain, address, native_balance, native_symbol, "
            " tokens, stable_usd, ledger_open_positions, ledger_exposure_usd, "
            " ledger_detail, alerts, created_at) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11, NOW())",
            wallet["group"], wallet["chain"], wallet["address"],
            float(balances["native"]),
            _NATIVE_SYMBOL.get(wallet["chain"], "NATIVE"),
            json.dumps(balances["tokens"], default=str),
            float(balances["stable_usd"]),
            int(ledger["open_positions"]), float(ledger["exposure_usd"]),
            json.dumps(ledger["tables"], default=str),
            json.dumps(alerts, default=str),
        )
    except Exception as exc:
        logger.error("treasury_snapshots insert failed for %s/%s: %s",
                     wallet["chain"], wallet["address"][:10], exc)


# ───────────────────────────── tick + loop ──────────────────────────────────

async def run_tick(pool, cfg: dict) -> dict:
    """One observe cycle. Returns a summary dict for /status."""
    import aiohttp

    rpc = None
    try:
        from config.pool_engine import PoolEngine
        rpc = await PoolEngine.get_instance()
        if not rpc.initialized:
            await rpc.initialize(pool)
    except Exception as exc:
        logger.warning("pool_engine unavailable (tick degraded): %s", exc)
        rpc = None

    wallets = discover_wallets(cfg)
    summary: dict = {"wallets": len(wallets), "observed": 0,
                     "alerts": 0, "detail": {}}
    if not wallets:
        logger.warning("no wallet addresses discovered (WALLET_ADDRESS / "
                       "SOLANA_WALLET / SOLANA_MODULE_WALLET unset?) — idle tick")
        return summary

    ledger_cache: Dict[str, dict] = {}
    async with aiohttp.ClientSession() as session:
        async with pool.acquire() as conn:
            for wallet in wallets:
                chain = wallet["chain"]
                rpc_url = None
                if rpc:
                    try:
                        rpc_url = await rpc.get_endpoint(_provider_type(chain))
                    except Exception as exc:
                        logger.debug("get_endpoint(%s) fail-soft: %s", chain, exc)
                if not rpc_url:
                    logger.warning("no RPC endpoint for %s — chain skipped", chain)
                    continue

                balances = await _read_wallet(session, rpc, rpc_url, wallet, cfg)
                if balances is None:
                    continue

                if chain not in ledger_cache:
                    ledger_cache[chain] = await collect_ledger(conn, chain)
                ledger = ledger_cache[chain]

                prev = await _prev_stable_usd(conn, chain, wallet["address"])
                alerts = evaluate_alerts(wallet, balances, ledger, prev, cfg)
                for a in alerts:
                    log = logger.error if a["severity"] == "error" else logger.warning
                    log("TREASURY ALERT [%s] %s", a["type"], a["message"])

                await _persist_snapshot(conn, wallet, balances, ledger, alerts)
                summary["observed"] += 1
                summary["alerts"] += len(alerts)
                summary["detail"][f"{wallet['group']}:{chain}"] = {
                    "native": round(balances["native"], 6),
                    "stable_usd": round(balances["stable_usd"], 2),
                    "ledger_open": ledger["open_positions"],
                    "alerts": [a["type"] for a in alerts],
                }
    return summary


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, observe, sleep poll_interval_seconds."""
    while True:
        cfg = await get_config() if get_config else {}
        if _KILLSWITCH.exists():
            logger.info("killswitch present — treasury tick skipped (observe-only "
                        "module; it never writes the killswitch)")
        elif _PAUSE_FLAG.exists():
            logger.info("treasury paused (logs/.pause_treasury) — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info("treasury tick: observed=%d/%d wallets, alerts=%d",
                            s["observed"], s["wallets"], s["alerts"])
            except Exception as exc:
                logger.error("treasury tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int(_cfg_float(cfg or {}, "poll_interval_seconds", 300)))


# ───────────────────────────── self-test (pure parts) ───────────────────────

if __name__ == "__main__":
    cfg = {"gas_floor_ethereum": 0.01, "gas_floor_default": 0.005,
           "hot_wallet_ceiling_usd": 1000.0, "reconcile_drift_usd": 250.0}
    w = {"group": "evm", "chain": "ethereum", "address": "0x" + "a" * 40}

    # gas low + open LIVE positions -> error severity
    a = evaluate_alerts(w, {"native": 0.001, "tokens": {}, "stable_usd": 0.0},
                        {"open_positions": 2, "exposure_usd": 50.0, "tables": {}},
                        None, cfg)
    assert [x["type"] for x in a] == ["gas_low"] and a[0]["severity"] == "error", a

    # healthy wallet -> no alerts
    a = evaluate_alerts(w, {"native": 0.5, "tokens": {}, "stable_usd": 100.0},
                        {"open_positions": 0, "exposure_usd": 0.0, "tables": {}},
                        100.0, cfg)
    assert a == [], a

    # unswept profit + drift
    a = evaluate_alerts(w, {"native": 0.5, "tokens": {}, "stable_usd": 1500.0},
                        {"open_positions": 0, "exposure_usd": 0.0, "tables": {}},
                        900.0, cfg)
    assert {x["type"] for x in a} == {"hot_wallet_high", "reconcile_drift"}, a

    # ledger says open LIVE but wallet empty -> unbacked (+ gas_low)
    a = evaluate_alerts(w, {"native": 0.0, "tokens": {}, "stable_usd": 0.0},
                        {"open_positions": 1, "exposure_usd": 80.0, "tables": {}},
                        None, cfg)
    assert {x["type"] for x in a} == {"gas_low", "ledger_unbacked"}, a

    # per-chain floor fallback
    assert gas_floor_for(cfg, "ethereum") == 0.01
    assert gas_floor_for(cfg, "base") == 0.005

    ws = discover_wallets({"extra_wallets": [
        {"chain": "Base", "address": "0x" + "b" * 40, "label": "cold"}]})
    assert any(x["group"] == "cold" and x["chain"] == "base" for x in ws), ws
    print("treasury_engine self-test OK")
