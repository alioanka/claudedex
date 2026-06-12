"""REGIME_ALLOCATOR engine — fetch -> classify -> propose -> persist.

One tick:
  1. Fetch BTCUSDT + ETHUSDT closes from a FREE public kline endpoint
     (Binance spot REST, Bybit v5 fallback — no API key, no LLM, 2 requests
     per tick). Fail-soft: if both sources fail the tick is skipped.
  2. regime_classifier.classify() -> transparent regime label + confidence
     (pure math, self-tested offline).
  3. Persist one row to regime_snapshots (full audit history).
  4. regime_classifier.propose_weights() -> per-module capital-weight
     proposals conditioned on the regime; supersede prior pending rows and
     insert the new batch into regime_allocation_proposals for OPERATOR
     approval. Optionally (mirror_to_portfolio_allocations=true, default
     false) also mirror the rows into portfolio_allocations with
     proposed_by='regime' so the existing allocator dashboard panel shows
     them side-by-side.

ADVISORY ONLY. This engine never trades, never flips a live flag, never
writes logs/.killswitch or any pause flag. Everything is fail-soft: any
error is logged and the loop continues.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from modules.regime_allocator.core.regime_classifier import (
    ALLOC_MODULES,
    DEFAULT_TILTS,
    RegimeParams,
    RegimeResult,
    classify,
    propose_weights,
)

logger = logging.getLogger("regime_allocator")

_KILLSWITCH = Path("logs") / ".killswitch"
_PAUSE_FLAG = Path("logs") / ".pause_regime_allocator"

# Module -> .env enabled flag (same convention as portfolio_allocator).
# Kept local so this module never imports another module's internals.
_ENABLED_ENV = {
    "sniper": "SNIPER_MODULE_ENABLED",
    "arbitrage": "ARBITRAGE_MODULE_ENABLED",
    "copy_trading": "COPY_TRADING_MODULE_ENABLED",
    "futures": "FUTURES_MODULE_ENABLED",
    "solana": "SOLANA_MODULE_ENABLED",
    "dex": "DEX_MODULE_ENABLED",
    "ai": "AI_MODULE_ENABLED",
}

_BINANCE_KLINES = "https://api.binance.com/api/v3/klines"
_BYBIT_KLINES = "https://api.bybit.com/v5/market/kline"
# kline_interval config value -> (binance interval, bybit interval)
_INTERVAL_MAP = {
    "1h": ("1h", "60"),
    "4h": ("4h", "240"),
    "1d": ("1d", "D"),
}


def _env_flag(key: str) -> bool:
    return os.getenv(key, "false").strip().lower() in ("true", "1", "yes", "on")


def enabled_modules_from_env() -> Dict[str, bool]:
    return {m: _env_flag(env) for m, env in _ENABLED_ENV.items()}


# ───────────────────────────── price fetch (free, fail-soft) ───────────────

async def _fetch_binance_closes(session, symbol: str, interval: str,
                                limit: int) -> Optional[List[float]]:
    params = {"symbol": symbol, "interval": interval, "limit": str(limit)}
    async with session.get(_BINANCE_KLINES, params=params, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"binance HTTP {resp.status}")
        data = await resp.json()
    # Binance returns chronological arrays; close = index 4.
    closes = [float(k[4]) for k in data if isinstance(k, (list, tuple)) and len(k) > 4]
    return closes or None


async def _fetch_bybit_closes(session, symbol: str, interval: str,
                              limit: int) -> Optional[List[float]]:
    params = {"category": "linear", "symbol": symbol,
              "interval": interval, "limit": str(limit)}
    async with session.get(_BYBIT_KLINES, params=params, timeout=15) as resp:
        if resp.status != 200:
            raise RuntimeError(f"bybit HTTP {resp.status}")
        data = await resp.json()
    rows = (((data or {}).get("result") or {}).get("list")) or []
    # Bybit returns newest-first; close = index 4. Reverse to chronological.
    closes = [float(r[4]) for r in reversed(rows)
              if isinstance(r, (list, tuple)) and len(r) > 4]
    return closes or None


async def fetch_closes(interval_key: str, limit: int) -> Optional[Dict[str, List[float]]]:
    """{'BTC': closes, 'ETH': closes} chronological, or None if every free
    source failed (the tick is then skipped — fail-soft, never raises)."""
    try:
        import aiohttp
    except Exception as exc:
        logger.warning("aiohttp unavailable — regime tick skipped: %s", exc)
        return None
    binance_iv, bybit_iv = _INTERVAL_MAP.get(interval_key, _INTERVAL_MAP["4h"])
    out: Dict[str, List[float]] = {}
    try:
        async with aiohttp.ClientSession() as session:
            for asset, symbol in (("BTC", "BTCUSDT"), ("ETH", "ETHUSDT")):
                closes = None
                for name, fn, iv in (
                    ("binance", _fetch_binance_closes, binance_iv),
                    ("bybit", _fetch_bybit_closes, bybit_iv),
                ):
                    try:
                        closes = await fn(session, symbol, iv, limit)
                        if closes:
                            out[f"{asset}_source"] = name  # type: ignore[assignment]
                            break
                    except Exception as exc:
                        logger.warning("%s klines via %s failed (fail-soft): %s",
                                       symbol, name, exc)
                if not closes:
                    logger.warning("no kline source for %s — regime tick skipped", symbol)
                    return None
                out[asset] = closes
    except Exception as exc:
        logger.warning("fetch_closes fail-soft: %s", exc)
        return None
    return out


# ───────────────────────────── config ──────────────────────────────────────

async def load_regime_config(conn) -> dict:
    """config_type='regime_allocator' rows -> typed dict. Fail-soft to {}."""
    out: dict = {}
    try:
        rows = await conn.fetch(
            "SELECT key, value FROM config_settings WHERE config_type='regime_allocator'"
        )
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
        logger.error("load_regime_config fail-soft: %s", exc)
    return out


def _params_from_config(cfg: dict) -> RegimeParams:
    p = RegimeParams()
    for f in ("vol_expand_ratio", "vol_compress_ratio", "er_trend",
              "er_range", "btc_weight"):
        if f in cfg:
            try:
                setattr(p, f, float(cfg[f]))
            except (TypeError, ValueError):
                pass
    for f in ("short_vol_bars", "long_vol_bars", "trend_bars"):
        if f in cfg:
            try:
                setattr(p, f, int(cfg[f]))
            except (TypeError, ValueError):
                pass
    return p


def _tilts_from_config(cfg: dict) -> Dict[str, Dict[str, float]]:
    """Optional operator override of the tilt matrix via the 'regime_tilts'
    JSON knob. Partial overrides merge over the defaults; junk is ignored."""
    raw = cfg.get("regime_tilts")
    if not raw or not isinstance(raw, str):
        return DEFAULT_TILTS
    try:
        parsed = json.loads(raw)
        tilts = {k: dict(v) for k, v in DEFAULT_TILTS.items()}
        for regime, per_module in parsed.items():
            if regime in tilts and isinstance(per_module, dict):
                for m, t in per_module.items():
                    if m in ALLOC_MODULES:
                        tilts[regime][m] = float(t)
        return tilts
    except Exception as exc:
        logger.warning("regime_tilts override unparseable (using defaults): %s", exc)
        return DEFAULT_TILTS


def _base_weights_from_config(cfg: dict) -> Optional[Dict[str, float]]:
    raw = cfg.get("base_weights")
    if not raw or not isinstance(raw, str):
        return None
    try:
        parsed = json.loads(raw)
        return {m: float(parsed[m]) for m in parsed if m in ALLOC_MODULES} or None
    except Exception as exc:
        logger.warning("base_weights override unparseable (using equal): %s", exc)
        return None


# ───────────────────────────── persistence ─────────────────────────────────

async def _persist_snapshot(conn, result: RegimeResult, source: str) -> Optional[int]:
    try:
        return await conn.fetchval(
            "INSERT INTO regime_snapshots "
            "(regime, confidence, reason, components, price_source, created_at) "
            "VALUES ($1,$2,$3,$4,$5, NOW()) RETURNING id",
            result.regime, float(result.confidence), result.reason,
            json.dumps(result.components, default=str), source,
        )
    except Exception as exc:
        logger.error("regime_snapshots insert failed: %s", exc)
        return None


async def _persist_proposals(conn, snapshot_id: Optional[int],
                             result: RegimeResult, proposals) -> int:
    """Supersede prior pending proposals, insert the new batch. Returns rows
    inserted. (Same supersede invariant the portfolio allocator learned the
    hard way: without it every tick stacks duplicate pending rows.)"""
    try:
        await conn.execute(
            "UPDATE regime_allocation_proposals SET superseded_at = NOW() "
            "WHERE approved_at IS NULL AND superseded_at IS NULL"
        )
    except Exception as exc:
        logger.warning("supersede prior pending regime proposals failed: %s", exc)
    n = 0
    for p in proposals:
        try:
            await conn.execute(
                "INSERT INTO regime_allocation_proposals "
                "(snapshot_id, regime, confidence, module, weight_pct, reason, "
                " components, created_at) "
                "VALUES ($1,$2,$3,$4,$5,$6,$7, NOW())",
                snapshot_id, result.regime, float(result.confidence),
                p.module, float(p.weight_pct), p.reason,
                json.dumps(p.components, default=str),
            )
            n += 1
        except Exception as exc:
            logger.warning("regime proposal insert(%s) failed: %s", p.module, exc)
    return n


async def _mirror_to_portfolio_allocations(conn, proposals,
                                           total_book_usd: float) -> int:
    """OPTIONAL mirror (knob default false): also write the regime weights as
    pending portfolio_allocations rows (proposed_by='regime') so the existing
    allocator dashboard shows both authorities side-by-side. Supersedes only
    its OWN prior pending rows; never touches allocator/operator rows."""
    try:
        await conn.execute(
            "UPDATE portfolio_allocations SET effective_until = NOW() "
            "WHERE proposed_by = 'regime' AND approved_at IS NULL "
            "  AND effective_until IS NULL"
        )
    except Exception as exc:
        logger.warning("supersede prior regime rows in portfolio_allocations failed: %s", exc)
    n = 0
    for p in proposals:
        try:
            await conn.execute(
                "INSERT INTO portfolio_allocations "
                "(module, pct_of_book, usd_amount, proposed_by, reason, metrics) "
                "VALUES ($1,$2,$3,'regime',$4,$5::jsonb)",
                p.module, float(p.weight_pct),
                round(p.weight_pct / 100.0 * total_book_usd, 2),
                p.reason, json.dumps(p.components, default=str),
            )
            n += 1
        except Exception as exc:
            logger.warning("portfolio_allocations mirror(%s) failed: %s", p.module, exc)
    return n


# ───────────────────────────── tick + loop ─────────────────────────────────

async def run_tick(pool, cfg: dict) -> dict:
    """One full regime cycle. Returns a small summary dict (for /status)."""
    summary: dict = {"regime": None, "confidence": None, "proposals_written": 0,
                     "mirrored": 0, "price_source": None, "skipped": None}

    interval_key = str(cfg.get("kline_interval", "4h"))
    limit = int(cfg.get("kline_limit", 180))
    closes = await fetch_closes(interval_key, limit)
    if not closes:
        summary["skipped"] = "no_price_data"
        return summary

    params = _params_from_config(cfg)
    result = classify(closes["BTC"], closes["ETH"], params)
    source = f"{closes.get('BTC_source', '?')}/{closes.get('ETH_source', '?')}:{interval_key}"
    summary.update(regime=result.regime,
                   confidence=round(result.confidence, 3),
                   price_source=source)
    logger.info("regime: %s", result.reason)

    proposals = propose_weights(
        result.regime, result.confidence, enabled_modules_from_env(),
        base_weights=_base_weights_from_config(cfg),
        tilts=_tilts_from_config(cfg),
        reserve_pct=float(cfg.get("reserve_pct", 10.0)),
        chop_extra_reserve_pct=float(cfg.get("chop_extra_reserve_pct", 10.0)),
        min_confidence=float(cfg.get("min_confidence", 0.25)),
    )

    async with pool.acquire() as conn:
        snapshot_id = await _persist_snapshot(conn, result, source)
        summary["proposals_written"] = await _persist_proposals(
            conn, snapshot_id, result, proposals)
        if bool(cfg.get("mirror_to_portfolio_allocations", False)):
            total_book = float(os.getenv("PORTFOLIO_TOTAL_BOOK_USD", "1000.0"))
            summary["mirrored"] = await _mirror_to_portfolio_allocations(
                conn, proposals, total_book)
    return summary


async def run_loop(pool, *, get_config=None) -> None:
    """Forever: load config, run a tick, sleep tick_interval_seconds.
    Skips the tick (but keeps looping) while the killswitch or this module's
    pause flag is present. Never writes either flag itself."""
    while True:
        cfg = await get_config() if get_config else {}
        if _KILLSWITCH.exists():
            logger.info("killswitch present — regime tick skipped")
        elif _PAUSE_FLAG.exists():
            logger.info("regime_allocator paused — tick skipped")
        else:
            try:
                s = await run_tick(pool, cfg)
                logger.info("regime tick: regime=%s conf=%s proposals=%d mirrored=%d skipped=%s",
                            s["regime"], s["confidence"], s["proposals_written"],
                            s["mirrored"], s["skipped"])
            except Exception as exc:
                logger.error("regime tick failed (fail-soft): %s", exc)
        await asyncio.sleep(int((cfg or {}).get("tick_interval_seconds", 3600)))
