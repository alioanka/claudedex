"""Stat-arb engine: scan pairs -> record spread state -> manage shadow positions.

Cycle shape (every poll_interval_s):
  1. skip if paused (logs/.pause_stat_arb) — killswitch is polled by the entry
     point and also enforced inside the executor's gate chain;
  2. reload config_type='stat_arb' rows;
  3. fetch CLOSED-bar closes for every universe symbol (free data layer);
  4. evaluate every candidate pair (pure pair_math) + record spread state
     (throttled);
  5. manage OPEN pairs first (exits before entries — capital honesty):
     hard z-stop / mean-revert exit / time stop / persistent retest failure;
  6. then consider entries: tradable pair, |z| >= entry_z, below the
     max-concurrent cap, pair not on post-stop cooldown, one position per
     pair, fixed notional. NO averaging down, NO widening stops — there is
     simply no code path that adds to or resizes an open pair.

A pair that exits on the HARD z-stop goes on cooldown for
pair_cooldown_hours: the stop says the relationship broke, so we refuse to
re-enter until a fresh window has re-qualified it.
"""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import time
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from modules.stat_arb.core import pair_math
from modules.stat_arb.core.data import PriceFeed

logger = logging.getLogger("StatArbModule.Engine")

DEFAULT_UNIVERSE = ("BTC/USDT,ETH/USDT,SOL/USDT,BNB/USDT,XRP/USDT,"
                    "DOGE/USDT,ADA/USDT,AVAX/USDT,LINK/USDT,DOT/USDT")


def _csv(value, default: str = "") -> List[str]:
    raw = str(value if value is not None else default)
    return [s.strip().upper() for s in raw.split(",") if s.strip()]


def candidate_pairs(config: dict) -> List[Tuple[str, str]]:
    """Explicit `pairs` config ('Y/USDT|X/USDT,...') or all universe combos.
    Canonical ordering (alphabetical y,x) so a pair has ONE key forever."""
    explicit = str(config.get("pairs", "") or "")
    out: List[Tuple[str, str]] = []
    if explicit.strip():
        for chunk in explicit.split(","):
            legs = [s.strip().upper() for s in chunk.split("|") if s.strip()]
            if len(legs) == 2 and legs[0] != legs[1]:
                out.append((min(legs), max(legs)))
    else:
        symbols = sorted(set(_csv(config.get("universe"), DEFAULT_UNIVERSE)))
        out = list(itertools.combinations(symbols, 2))
    seen, uniq = set(), []
    for p in out:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    cap = int(config.get("max_candidate_pairs", 45))
    return uniq[:max(1, cap)]


class StatArbEngine:
    def __init__(self, db_pool, config: dict, executor, module_dry_run: bool,
                 load_config=None):
        self.db_pool = db_pool
        self.config = config or {}
        self.executor = executor
        self.module_dry_run = module_dry_run
        self._load_config = load_config
        self.running = False
        self.open_pairs: Dict[str, dict] = {}     # pair_key -> position dict
        self.cooldown_until: Dict[str, float] = {}  # pair_key -> epoch seconds
        self._last_state_recorded: Dict[str, float] = {}
        self.stats = {"cycles": 0, "pairs_scanned": 0, "tradable_pairs": 0,
                      "open_pairs": 0, "entries": 0, "exits": 0,
                      "hard_stops": 0, "last_cycle_at": None,
                      "last_error": None}

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------
    async def reconcile_open_positions(self) -> None:
        """Restart reconcile: reload status='open' rows so a subprocess restart
        never orphans (or double-opens) a shadow/live pair."""
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT id, pair_key, long_symbol, short_symbol, beta, "
                    "notional_per_leg_usd, z_at_entry, entry_long_price, "
                    "entry_short_price, is_simulated, details, opened_at "
                    "FROM stat_arb_trades WHERE status = 'open'")
        except Exception as e:
            logger.error("reconcile query failed (starting flat): %s", e)
            return
        for r in rows:
            details = r["details"]
            if isinstance(details, str):
                try:
                    details = json.loads(details)
                except Exception:
                    details = {}
            details = details or {}
            self.open_pairs[r["pair_key"]] = {
                "trade_id": r["id"], "pair_key": r["pair_key"],
                "long_symbol": r["long_symbol"],
                "short_symbol": r["short_symbol"],
                "beta": float(r["beta"]),
                "notional_per_leg_usd": float(r["notional_per_leg_usd"]),
                "z_at_entry": float(r["z_at_entry"]),
                "entry_long_price": float(r["entry_long_price"]),
                "entry_short_price": float(r["entry_short_price"]),
                "is_simulated": bool(r["is_simulated"]),
                "position_side": details.get("position_side", "short_spread"),
                "opened_at": r["opened_at"], "retest_fails": 0,
            }
        if self.open_pairs:
            logger.info("Reconciled %d open pair(s) from DB: %s",
                        len(self.open_pairs), sorted(self.open_pairs))

    async def _record_spread_state(self, pair_key: str, y: str, x: str,
                                   ev: dict) -> None:
        interval = float(self.config.get("spread_state_record_interval_s", 900))
        now = time.monotonic()
        last = self._last_state_recorded.get(pair_key)
        if last is not None and (now - last) < interval:
            return
        self._last_state_recorded[pair_key] = now
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO stat_arb_spread_state
                        (pair_key, y_symbol, x_symbol, alpha, beta, spread,
                         spread_mean, spread_std, zscore, correlation,
                         ar1_tstat, half_life_bars, tradable, reason)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)
                    """,
                    pair_key, y, x, ev.get("alpha"), ev.get("beta"),
                    ev.get("spread"), ev.get("spread_mean"),
                    ev.get("spread_std"), ev.get("zscore"),
                    ev.get("correlation"), ev.get("ar1_tstat"),
                    ev.get("half_life_bars"), bool(ev.get("tradable")),
                    str(ev.get("reason", ""))[:64],
                )
        except Exception as e:
            logger.error("stat_arb_spread_state insert failed: %s", e)

    # ------------------------------------------------------------------
    # Position management
    # ------------------------------------------------------------------
    async def _close(self, pos: dict, closes: Dict[str, List[float]],
                     z: Optional[float], exit_reason: str) -> None:
        long_px = (closes.get(pos["long_symbol"]) or [None])[-1]
        short_px = (closes.get(pos["short_symbol"]) or [None])[-1]
        if not long_px or not short_px:
            logger.warning("No mark price for %s — exit deferred a cycle",
                           pos["pair_key"])
            return
        pnl = pair_math.simulated_pair_pnl_usd(
            notional_per_leg_usd=pos["notional_per_leg_usd"],
            long_entry=pos["entry_long_price"], long_exit=long_px,
            short_entry=pos["entry_short_price"], short_exit=short_px,
            taker_fee_bps=float(self.config.get("taker_fee_bps", 5.5)),
            slippage_bps=float(self.config.get("slippage_bps", 3.0)))
        result = await self.executor.close_pair(
            trade_id=pos["trade_id"], pair_key=pos["pair_key"],
            long_symbol=pos["long_symbol"], short_symbol=pos["short_symbol"],
            long_price=long_px, short_price=short_px,
            notional_per_leg_usd=pos["notional_per_leg_usd"],
            z_at_exit=z if z is not None else 0.0, exit_reason=exit_reason,
            pnl_usd=pnl, was_simulated=pos["is_simulated"],
            module_dry_run=self.module_dry_run)
        if result["status"] != "closed":
            return  # live close failed: row stays open, retry next cycle
        self.open_pairs.pop(pos["pair_key"], None)
        self.stats["exits"] += 1
        if exit_reason == "hard_z_stop":
            self.stats["hard_stops"] += 1
            hours = float(self.config.get("pair_cooldown_hours", 48))
            self.cooldown_until[pos["pair_key"]] = time.time() + hours * 3600
        logger.info("[exit:%s] %s z=%s pnl=$%s sim=%s", exit_reason,
                    pos["pair_key"], None if z is None else round(z, 2),
                    None if pnl is None else round(pnl, 2),
                    pos["is_simulated"])

    async def _manage_open(self, pair_key: str, ev: dict,
                           closes: Dict[str, List[float]]) -> None:
        pos = self.open_pairs.get(pair_key)
        if pos is None:
            return
        z = ev.get("zscore")

        # Persistent retest failure (data gone / fit broken): bounded patience.
        if z is None:
            pos["retest_fails"] += 1
            if pos["retest_fails"] >= int(self.config.get("retest_fail_exits", 6)):
                await self._close(pos, closes, None, "pair_retest_fail")
            return
        pos["retest_fails"] = 0

        # Time stop: thesis is hours-days reversion; stale trades get cut.
        opened_at = pos.get("opened_at")
        max_hold_s = float(self.config.get("max_hold_hours", 96)) * 3600
        if opened_at is not None:
            age = (datetime.now(timezone.utc) - opened_at).total_seconds()
            if age >= max_hold_s:
                await self._close(pos, closes, z, "time_stop")
                return

        action = pair_math.decide(
            z, True, pos["position_side"],
            entry_z=float(self.config.get("entry_z", 2.0)),
            exit_z=float(self.config.get("exit_z", 0.5)),
            stop_z=float(self.config.get("stop_z", 4.0)))
        if action == "EXIT_HARD_STOP":
            await self._close(pos, closes, z, "hard_z_stop")
        elif action == "EXIT_MEAN_REVERT":
            await self._close(pos, closes, z, "mean_revert")

    async def _maybe_enter(self, pair_key: str, y: str, x: str, ev: dict,
                           closes: Dict[str, List[float]]) -> None:
        if pair_key in self.open_pairs or not ev.get("tradable"):
            return
        if len(self.open_pairs) >= int(self.config.get("max_concurrent_pairs", 3)):
            return
        if time.time() < self.cooldown_until.get(pair_key, 0.0):
            return
        action = pair_math.decide(
            ev.get("zscore"), False, None,
            entry_z=float(self.config.get("entry_z", 2.0)),
            exit_z=float(self.config.get("exit_z", 0.5)),
            stop_z=float(self.config.get("stop_z", 4.0)))
        if action not in ("ENTER_SHORT_SPREAD", "ENTER_LONG_SPREAD"):
            return
        # spread = ln(y) - beta*ln(x): rich (z>0) -> short y / long x.
        if action == "ENTER_SHORT_SPREAD":
            side, long_symbol, short_symbol = "short_spread", x, y
        else:
            side, long_symbol, short_symbol = "long_spread", y, x
        long_px = (closes.get(long_symbol) or [None])[-1]
        short_px = (closes.get(short_symbol) or [None])[-1]
        if not long_px or not short_px:
            return
        notional = float(self.config.get("notional_per_leg_usd", 100.0))
        result = await self.executor.open_pair(
            pair_key=pair_key, long_symbol=long_symbol,
            short_symbol=short_symbol, beta=float(ev["beta"]),
            z_at_entry=float(ev["zscore"]), long_price=long_px,
            short_price=short_px, notional_per_leg_usd=notional,
            module_dry_run=self.module_dry_run,
            details={"position_side": side, "y_symbol": y, "x_symbol": x,
                     "alpha": ev.get("alpha"),
                     "half_life_bars": ev.get("half_life_bars"),
                     "ar1_tstat": ev.get("ar1_tstat"),
                     "correlation": ev.get("correlation")})
        if result.get("trade_id") is None and self.db_pool:
            return  # insert failed: do not track an unpersisted position
        self.open_pairs[pair_key] = {
            "trade_id": result.get("trade_id"), "pair_key": pair_key,
            "long_symbol": long_symbol, "short_symbol": short_symbol,
            "beta": float(ev["beta"]), "notional_per_leg_usd": notional,
            "z_at_entry": float(ev["zscore"]), "entry_long_price": long_px,
            "entry_short_price": short_px,
            "is_simulated": bool(result.get("is_simulated", True)),
            "position_side": side, "opened_at": datetime.now(timezone.utc),
            "retest_fails": 0,
        }
        self.stats["entries"] += 1
        logger.info("[enter:%s] %s z=%.2f long=%s short=%s $%.0f/leg sim=%s%s",
                    side, pair_key, ev["zscore"], long_symbol, short_symbol,
                    notional, result.get("is_simulated", True),
                    f" skip={result['skip_reason']}" if result.get("skip_reason") else "")

    # ------------------------------------------------------------------
    # Cycle / loop
    # ------------------------------------------------------------------
    async def _cycle(self) -> None:
        from core.dry_run import is_module_paused
        if is_module_paused("stat_arb"):
            logger.info("Module paused (logs/.pause_stat_arb) — idling")
            return
        if self._load_config is not None:
            self.config.update(await self._load_config(self.db_pool))
            self.executor.config = self.config

        pairs = candidate_pairs(self.config)
        symbols = sorted({s for p in pairs for s in p}
                         | {s for pos in self.open_pairs.values()
                            for s in (pos["long_symbol"], pos["short_symbol"])})
        timeframe = str(self.config.get("timeframe", "1h"))
        lookback = int(self.config.get("lookback_bars", 240))

        feed = PriceFeed(self.db_pool, request_spacing_s=float(
            self.config.get("request_spacing_s", 0.25)))
        closes: Dict[str, List[float]] = {}
        try:
            for sym in symbols:
                series = await feed.get_closes(sym, timeframe, lookback)
                if series:
                    closes[sym] = series
                else:
                    logger.warning("No data for %s (%s %d bars) — skipped",
                                   sym, timeframe, lookback)
        finally:
            await feed.close()

        evaluations: Dict[str, dict] = {}
        tradable = 0
        for y, x in pairs:
            pair_key = f"{y}~{x}"
            if y not in closes or x not in closes:
                continue
            ev = pair_math.evaluate_pair(
                closes[y], closes[x],
                min_correlation=float(self.config.get("min_correlation", 0.6)),
                adf_tstat_max=float(self.config.get("adf_tstat_max", -2.9)),
                min_half_life_bars=float(self.config.get("min_half_life_bars", 4)),
                max_half_life_bars=float(self.config.get("max_half_life_bars", 120)))
            evaluations[pair_key] = (y, x, ev)
            tradable += 1 if ev["tradable"] else 0
            await self._record_spread_state(pair_key, y, x, ev)
        self.stats["pairs_scanned"] = len(evaluations)
        self.stats["tradable_pairs"] = tradable

        # Exits FIRST (free the concurrency budget before new entries).
        for pair_key in list(self.open_pairs):
            entry = evaluations.get(pair_key)
            ev = entry[2] if entry else {"zscore": None}
            await self._manage_open(pair_key, ev, closes)

        for pair_key, (y, x, ev) in evaluations.items():
            await self._maybe_enter(pair_key, y, x, ev, closes)
        self.stats["open_pairs"] = len(self.open_pairs)

    async def run(self) -> None:
        self.running = True
        logger.info("Stat-arb engine loop starting (shadow_mode=%s, "
                    "live_execution_enabled=%s)",
                    self.config.get("shadow_mode", True),
                    self.config.get("live_execution_enabled", False))
        await self.reconcile_open_positions()
        while self.running:
            try:
                await self._cycle()
                self.stats["cycles"] += 1
                self.stats["last_cycle_at"] = datetime.now(timezone.utc).isoformat()
                self.stats["last_error"] = None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # fail-soft: never die on a cycle error
                self.stats["last_error"] = f"{type(e).__name__}: {e}"
                logger.error("Cycle error (continuing): %s", self.stats["last_error"])
            await asyncio.sleep(float(self.config.get("poll_interval_s", 300)))

    async def stop(self) -> None:
        self.running = False
