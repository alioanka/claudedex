"""SMART_MONEY engine — ingest -> mark -> score -> signal. ADVISORY ONLY.

One tick:
  1. INGEST: scan watched DEX pairs per chain for large swaps, attribute the
     buyer wallet, persist to smart_money_wallet_events (idempotent on
     (chain, tx_hash, log_index)).
  2. MARK: for past events whose forward horizons have FULLY elapsed, record
     the realized forward return (late-marked from the current price — never
     early, so no look-ahead by construction).
  3. SCORE: pure cluster_scorer.score_wallet over fully-elapsed past events
     -> upsert smart_money_wallet_scores.
  4. SIGNAL: detect accumulation clusters among RECENT events; where enough
     historically-profitable wallets participate, write an ADVISORY row to
     smart_money_signals (cooldown-deduped). NO trade is ever placed here.
  5. PURGE: events older than retention are deleted (disk discipline).

Fail-soft everywhere; no paid API/LLM; RPC only via the shared PoolEngine.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

from modules.smart_money.core import chain_scanner
from modules.smart_money.core.cluster_scorer import (
    WalletEvent, build_signals, detect_clusters, score_wallet,
)

logger = logging.getLogger("smart_money")

_KILLSWITCH = Path("logs/.killswitch")
_PAUSE = Path("logs/.pause_smart_money")

# chain -> last scanned block (in-memory; restart rescans a bounded window and
# the events unique key dedupes).
_CURSORS: Dict[str, int] = {}


def _csv(cfg: dict, key: str, default: str) -> List[str]:
    return [s.strip() for s in str(cfg.get(key, default)).split(",") if s.strip()]


def _horizons(cfg: dict) -> List[int]:
    out = []
    for s in _csv(cfg, "forward_horizons_minutes", "60,360,1440"):
        try:
            out.append(int(s))
        except ValueError:
            pass
    return sorted(out) or [60, 360, 1440]


def _jsonb(v) -> dict:
    if isinstance(v, dict):
        return v
    try:
        return json.loads(v) if v else {}
    except Exception:
        return {}


async def load_smart_money_config(pool) -> dict:
    """config_type='smart_money' rows -> typed dict. Fail-soft to {}."""
    out: dict = {}
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings "
                "WHERE config_type='smart_money'")
        for r in rows:
            v = r["value"]
            if isinstance(v, str):
                low = v.lower()
                if low in ("true", "false"):
                    v = low == "true"
                else:
                    try:
                        v = float(v) if "." in v else int(v)
                    except ValueError:
                        pass
            out[r["key"]] = v
    except Exception as exc:
        logger.error("load_smart_money_config fail-soft: %s", exc)
    return out


# ───────────────────────────── tick steps ───────────────────────────────────

async def _ingest(conn, session, rpc_pool, cfg: dict) -> int:
    inserted = 0
    for chain in _csv(cfg, "chains", "ethereum,base,arbitrum"):
        if chain == "solana":
            continue        # v1 coverage limit: EVM only (see CLAUDE.md)
        try:
            pairs = await chain_scanner.discover_pairs(session, chain, cfg)
            events, cursor = await chain_scanner.scan_chain(
                session, rpc_pool, chain, pairs, _CURSORS.get(chain), cfg)
            if cursor is not None:
                _CURSORS[chain] = cursor
            # Diagnostic: surface WHERE a zero comes from — 0 pairs means the
            # DexScreener liquidity/volume filter is too tight (or the search
            # returned nothing); pairs>0 but 0 events usually means the RPC
            # rejects/limits eth_getLogs (free Ankr caps log ranges) — point
            # smart_money at a getLogs-capable RPC (Alchemy/dRPC) in /settings/rpc-api.
            logger.info("ingest[%s]: pairs=%d swaps=%d (>= $%s)",
                        chain, len(pairs), len(events),
                        cfg.get("min_event_usd", 2000))
            for e in events:
                res = await conn.execute(
                    "INSERT INTO smart_money_wallet_events "
                    "(chain, wallet, token, token_symbol, side, amount_usd, "
                    " price_usd_at_event, tx_hash, log_index, block_time) "
                    "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, to_timestamp($10)) "
                    "ON CONFLICT (chain, tx_hash, log_index) DO NOTHING",
                    e["chain"], e["wallet"], e["token"], e["token_symbol"],
                    e["side"], float(e["amount_usd"]), float(e["price_usd"]),
                    e["tx_hash"], int(e["log_index"]), float(e["est_ts"]))
                if res.endswith("1"):
                    inserted += 1
        except Exception as exc:
            logger.warning("ingest(%s) fail-soft: %s", chain, exc)
    return inserted


async def _mark(conn, session, cfg: dict) -> int:
    """Record forward returns for horizons that have FULLY elapsed. The mark
    uses the price observed NOW (>= event_time + horizon) — late, never early."""
    horizons = _horizons(cfg)
    min_h = min(horizons)
    rows = await conn.fetch(
        "SELECT id, chain, token, price_usd_at_event, block_time, fwd_returns "
        "FROM smart_money_wallet_events "
        f"WHERE NOT fully_marked AND block_time <= NOW() - INTERVAL '{min_h} minutes' "
        "ORDER BY block_time ASC LIMIT 500")
    if not rows:
        return 0
    max_tokens = int(cfg.get("max_mark_tokens_per_tick", 60))
    by_chain: Dict[str, List[str]] = {}
    for r in rows:
        toks = by_chain.setdefault(r["chain"], [])
        if r["token"] not in toks and len(toks) < max_tokens:
            toks.append(r["token"])
    prices: Dict[tuple, float] = {}
    for chain, tokens in by_chain.items():
        got = await chain_scanner.fetch_token_prices(session, chain, tokens)
        for t, p in got.items():
            prices[(chain, t)] = p
    now = time.time()
    marked = 0
    for r in rows:
        price_now = prices.get((r["chain"], r["token"]))
        p0 = float(r["price_usd_at_event"] or 0)
        if not price_now or p0 <= 0:
            continue
        ev_ts = r["block_time"].timestamp()
        fwd = _jsonb(r["fwd_returns"])
        changed = False
        for h in horizons:
            if str(h) in fwd:
                continue
            if ev_ts + h * 60.0 <= now:         # horizon fully elapsed only
                fwd[str(h)] = round((price_now - p0) / p0 * 100.0, 4)
                changed = True
        if not changed:
            continue
        fully = all(str(h) in fwd for h in horizons)
        await conn.execute(
            "UPDATE smart_money_wallet_events "
            "SET fwd_returns=$1::jsonb, fully_marked=$2 WHERE id=$3",
            json.dumps(fwd), fully, r["id"])
        marked += 1
    return marked


def _row_to_event(r) -> WalletEvent:
    fwd = {int(k): float(v) for k, v in _jsonb(r["fwd_returns"]).items()}
    return WalletEvent(
        wallet=r["wallet"], chain=r["chain"], token=r["token"],
        side=r["side"], amount_usd=float(r["amount_usd"] or 0),
        price_usd=float(r["price_usd_at_event"] or 0),
        ts=r["block_time"].timestamp(), fwd_returns_pct=fwd)


async def _score(conn, cfg: dict) -> int:
    window_h = int(cfg.get("scoring_window_hours", 336))
    horizons = _horizons(cfg)
    rows = await conn.fetch(
        "SELECT chain, wallet, token, side, amount_usd, price_usd_at_event, "
        "block_time, fwd_returns FROM smart_money_wallet_events "
        f"WHERE side='buy' AND block_time > NOW() - INTERVAL '{window_h} hours'")
    by_wallet: Dict[tuple, List[WalletEvent]] = {}
    for r in rows:
        by_wallet.setdefault((r["chain"], r["wallet"]), []).append(_row_to_event(r))
    now = time.time()
    scored = 0
    for (chain, wallet), events in by_wallet.items():
        s = score_wallet(
            events, now_ts=now, horizons_minutes=horizons,
            half_life_days=float(cfg.get("score_half_life_days", 14)),
            min_events=int(cfg.get("min_events_per_wallet", 3)),
            n_target=int(cfg.get("n_target_events", 12)))
        if s is None:
            continue
        await conn.execute(
            "INSERT INTO smart_money_wallet_scores "
            "(chain, wallet, events_scored, total_buy_usd, avg_fwd_return_pct, "
            " hit_rate, confidence, score, details, scored_at) "
            "VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9, NOW()) "
            "ON CONFLICT (chain, wallet) DO UPDATE SET "
            "events_scored=EXCLUDED.events_scored, "
            "total_buy_usd=EXCLUDED.total_buy_usd, "
            "avg_fwd_return_pct=EXCLUDED.avg_fwd_return_pct, "
            "hit_rate=EXCLUDED.hit_rate, confidence=EXCLUDED.confidence, "
            "score=EXCLUDED.score, details=EXCLUDED.details, scored_at=NOW()",
            chain, wallet, s.events_scored, s.total_buy_usd,
            s.avg_fwd_return_pct, s.hit_rate, s.confidence, s.score,
            json.dumps({"horizons_minutes": horizons, "window_hours": window_h}))
        scored += 1
    return scored


async def _signal(conn, cfg: dict) -> int:
    window_m = int(cfg.get("cluster_window_minutes", 45))
    rows = await conn.fetch(
        "SELECT chain, wallet, token, token_symbol, side, amount_usd, "
        "price_usd_at_event, block_time, fwd_returns "
        "FROM smart_money_wallet_events "
        f"WHERE side='buy' AND block_time > NOW() - INTERVAL '{window_m} minutes'")
    if not rows:
        return 0
    symbols = {(r["chain"], r["token"]): r["token_symbol"] for r in rows}
    events = [_row_to_event(r) for r in rows]
    clusters = detect_clusters(
        events, now_ts=time.time(), window_minutes=window_m,
        min_wallets=int(cfg.get("cluster_min_wallets", 3)))
    if not clusters:
        return 0
    score_rows = await conn.fetch(
        "SELECT chain, wallet, events_scored, total_buy_usd, "
        "avg_fwd_return_pct, hit_rate, confidence, score "
        "FROM smart_money_wallet_scores")
    from modules.smart_money.core.cluster_scorer import WalletScore
    scores = {(r["chain"], r["wallet"]): WalletScore(
        wallet=r["wallet"], chain=r["chain"],
        events_scored=int(r["events_scored"] or 0),
        total_buy_usd=float(r["total_buy_usd"] or 0),
        avg_fwd_return_pct=float(r["avg_fwd_return_pct"] or 0),
        hit_rate=float(r["hit_rate"] or 0),
        confidence=float(r["confidence"] or 0),
        score=float(r["score"] or 0)) for r in score_rows}
    candidates = build_signals(
        clusters, scores,
        min_wallet_score=float(cfg.get("min_wallet_score", 0.55)),
        min_smart_wallets=int(cfg.get("min_smart_wallets", 2)),
        min_forward_return_pct=float(cfg.get("min_forward_return_pct", 3.0)),
        crowding_soft_cap=int(cfg.get("crowding_soft_cap_wallets", 12)))
    cooldown_m = int(cfg.get("signal_cooldown_minutes", 240))
    emitted = 0
    for s in candidates:
        dup = await conn.fetchval(
            "SELECT 1 FROM smart_money_signals WHERE chain=$1 AND token=$2 "
            f"AND created_at > NOW() - INTERVAL '{cooldown_m} minutes' LIMIT 1",
            s.chain, s.token)
        if dup:
            continue
        await conn.execute(
            "INSERT INTO smart_money_signals "
            "(chain, token, token_symbol, signal_type, strength, "
            " cluster_wallets, smart_wallets, avg_wallet_score, "
            " avg_fwd_return_pct, crowding_factor, total_buy_usd, advisory, "
            " details, created_at) "
            "VALUES ($1,$2,$3,'accumulation',$4,$5,$6,$7,$8,$9,$10,TRUE,$11, NOW())",
            s.chain, s.token, symbols.get((s.chain, s.token), "?"),
            s.strength, s.cluster_wallets, s.smart_wallets, s.avg_wallet_score,
            s.avg_fwd_return_pct, s.crowding_factor, s.total_buy_usd,
            json.dumps({"smart_wallets": s.smart_wallet_list[:20]}))
        emitted += 1
        logger.warning(
            "SMART-MONEY SIGNAL (advisory): %s %s strength=%.2f "
            "smart=%d/%d crowding=%.2f hist_fwd=%.1f%%",
            s.chain, symbols.get((s.chain, s.token), s.token), s.strength,
            s.smart_wallets, s.cluster_wallets, s.crowding_factor,
            s.avg_fwd_return_pct)
    return emitted


async def run_tick(pool, cfg: dict) -> dict:
    summary = {"events_ingested": 0, "events_marked": 0,
               "wallets_scored": 0, "signals_emitted": 0}
    try:
        import aiohttp
    except Exception as exc:
        logger.error("aiohttp unavailable, tick skipped: %s", exc)
        return summary
    rpc_pool = None
    try:
        from config.pool_engine import PoolEngine
        rpc_pool = await PoolEngine.get_instance()
    except Exception as exc:
        logger.warning("PoolEngine unavailable (ingest skipped): %s", exc)
    async with aiohttp.ClientSession() as session:
        async with pool.acquire() as conn:
            if rpc_pool is not None:
                summary["events_ingested"] = await _ingest(conn, session,
                                                           rpc_pool, cfg)
            summary["events_marked"] = await _mark(conn, session, cfg)
            summary["wallets_scored"] = await _score(conn, cfg)
            summary["signals_emitted"] = await _signal(conn, cfg)
            try:
                retention_d = int(cfg.get("event_retention_days", 45))
                await conn.execute(
                    "DELETE FROM smart_money_wallet_events "
                    f"WHERE block_time < NOW() - INTERVAL '{retention_d} days'")
            except Exception as exc:
                logger.debug("purge fail-soft: %s", exc)
    return summary


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, run a tick, sleep poll_interval_seconds."""
    while True:
        cfg = await get_config() if get_config else {}
        if _KILLSWITCH.exists():
            logger.info("killswitch present — smart_money tick skipped")
        elif _PAUSE.exists():
            logger.info("smart_money paused — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info("tick: ingested=%d marked=%d scored=%d signals=%d",
                            s["events_ingested"], s["events_marked"],
                            s["wallets_scored"], s["signals_emitted"])
            except Exception as exc:
                logger.error("smart_money tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int((cfg or {}).get("poll_interval_seconds", 300)))
