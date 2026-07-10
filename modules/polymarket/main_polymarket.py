#!/usr/bin/env python3
"""
Polymarket Module — prediction-market entry point (shadow-first).

Data source: Gamma REST API (read-only, free, no key, no on-chain risk).
Strategies: risk-free YES+NO arbitrage detector + event-momentum advice,
both shadow-recorded to polymarket_signals / polymarket_trades
(is_simulated=true). LIVE CLOB execution is OFF by default and gated:
shadow_mode=false AND live_execution_enabled=true AND not should_skip_live
AND the built-in Polymarket risk gate (per-market/total exposure caps,
max open markets) — see modules/polymarket/executor.py.

Launched by main.py when POLYMARKET_MODULE_ENABLED=true (default false).
Health server: http://0.0.0.0:8089 (env override: POLYMARKET_HEALTH_PORT).
Config: DB-backed config_type='polymarket_config' (migration 101).
Kill switches: logs/.killswitch (global), logs/.pause_polymarket (per-module).
"""
import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from aiohttp import web  # noqa: E402

# Setup logging
log_dir = Path("logs/polymarket")
log_dir.mkdir(parents=True, exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("PolymarketModule")
logger.setLevel(logging.INFO)
main_handler = RotatingFileHandler(log_dir / 'polymarket.log', maxBytes=10*1024*1024, backupCount=5)
main_handler.setFormatter(log_formatter)
logger.addHandler(main_handler)
error_handler = RotatingFileHandler(log_dir / 'polymarket_errors.log', maxBytes=5*1024*1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)
# Sub-loggers are children of "PolymarketModule" — records propagate to the
# parent's handlers automatically. Attaching the same handlers again wrote
# every sub-logger line TWICE (wave-F5 BUG-6); level-only setup is enough.
for sub in ("PolymarketModule.Gamma", "PolymarketModule.Executor"):
    logging.getLogger(sub).setLevel(logging.INFO)

from modules.polymarket.gamma_client import GammaClient, DEFAULT_GAMMA_BASE_URL  # noqa: E402
from modules.polymarket.strategies import detect_risk_free_arb, score_momentum  # noqa: E402
from modules.polymarket.executor import PolymarketExecutor  # noqa: E402

# Forward-outcome horizons (label -> seconds). Marked LATE like smart_money:
# a horizon is written only after it has FULLY elapsed, from the price
# observed at mark time — never an early estimate.
OUTCOME_HORIZONS = (("1h", 3600), ("6h", 21600), ("24h", 86400))


async def load_config(db_pool) -> dict:
    """Load config_type='polymarket_config' rows (mirrors arbitrage loader)."""
    settings = {}
    if not db_pool:
        return settings
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type = 'polymarket_config'"
            )
        for row in rows:
            val = row['value']
            if isinstance(val, str):
                low = val.lower()
                if low in ('true', 'false'):
                    val = low == 'true'
                else:
                    try:
                        val = float(val) if '.' in val else int(val)
                    except ValueError:
                        pass
            settings[row['key']] = val
        logger.info(f"Loaded {len(settings)} polymarket settings from database")
    except Exception as e:
        logger.error(f"Failed to load polymarket settings from DB: {e}")
    return settings


class PolymarketEngine:
    """Poll Gamma -> run shadow strategies -> record signals/simulated trades."""

    def __init__(self, db_pool, config: dict, module_dry_run: bool):
        self.db_pool = db_pool
        self.config = config
        self.module_dry_run = module_dry_run
        self.gamma = GammaClient(
            str(config.get('gamma_base_url', DEFAULT_GAMMA_BASE_URL)),
            max_requests_per_minute=int(config.get('gamma_max_requests_per_minute', 30)),
        )
        self.executor = PolymarketExecutor(db_pool, config)
        self.running = False
        self._prev_yes_prices: dict = {}
        self._prev_seen_at = None
        self._last_recorded: dict = {}  # (signal_type, market_id) -> monotonic ts
        # market_id -> (direction, monotonic ts) of the last RECORDED momentum
        # signal; a direction flip inside the cooldown window is suppressed
        # (wave-F5: in-play books flip YES/NO within minutes — that is noise).
        self._last_momentum_dir: dict = {}
        self._last_snapshot_prune = 0.0
        self.stats = {
            'cycles': 0, 'markets_seen': 0, 'arb_signals': 0,
            'momentum_signals': 0, 'snapshots_last_cycle': 0,
            'outcomes_marked': 0, 'last_cycle_at': None, 'last_error': None,
        }

    def _throttled(self, signal_type: str, market_id: str) -> bool:
        """True if this (type, market) was recorded inside the throttle window."""
        interval = float(self.config.get('shadow_record_interval_s', 300))
        key = (signal_type, market_id)
        now = time.monotonic()
        last = self._last_recorded.get(key)
        if last is not None and (now - last) < interval:
            return True
        self._last_recorded[key] = now
        if len(self._last_recorded) > 5000:
            self._last_recorded.clear()
        return False

    def _flip_suppressed(self, market_id: str, direction: str) -> bool:
        """True if this momentum signal flips direction inside the cooldown.

        Cooldown counts from the last RECORDED signal, so a market that keeps
        flip-flopping stays suppressed until it holds one direction long enough.
        """
        cooldown_s = float(self.config.get('momentum_flip_cooldown_minutes', 30)) * 60.0
        now = time.monotonic()
        last = self._last_momentum_dir.get(market_id)
        if last is not None and last[0] != direction and (now - last[1]) < cooldown_s:
            return True
        self._last_momentum_dir[market_id] = (direction, now)
        if len(self._last_momentum_dir) > 5000:
            self._last_momentum_dir.clear()
        return False

    async def _save_signal(self, sig: dict) -> None:
        if not self.db_pool:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO polymarket_signals
                        (signal_type, market_id, market_question, direction,
                         yes_price, no_price, edge_bps, score, details, is_simulated)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,TRUE)
                    """,
                    sig['signal_type'], str(sig['market_id']), sig.get('question', ''),
                    sig.get('direction'), sig.get('yes_price'), sig.get('no_price'),
                    sig.get('edge_bps'), sig.get('score'),
                    json.dumps(sig.get('details') or {}, default=str),
                )
        except Exception as e:
            logger.error(f"polymarket_signals insert failed: {e}")

    async def _cycle(self) -> None:
        from core.dry_run import is_module_paused
        if is_module_paused('polymarket'):
            logger.info("Module paused (logs/.pause_polymarket) — idling")
            return

        self.config.update(await load_config(self.db_pool))
        self.executor.config = self.config

        markets = await self.gamma.fetch_active_markets(
            limit=int(self.config.get('max_markets_per_poll', 200)),
            category_filter=str(self.config.get('category_filter', '') or ''),
        )
        if not markets:
            logger.warning(f"No markets fetched (last_error={self.gamma.last_error})")
            return
        self.stats['markets_seen'] = len(markets)

        # (a) Risk-free arb: YES_price + NO_price < 1 - buffer.
        arbs = detect_risk_free_arb(
            markets,
            min_edge_bps=float(self.config.get('min_arb_edge_bps', 100)),
            fee_gas_buffer_bps=float(self.config.get('fee_gas_buffer_bps', 100)),
            min_liquidity_usd=float(self.config.get('arb_min_liquidity_usd', 1000)),
            max_edge_bps=float(self.config.get('arb_max_edge_bps', 500)),
        )
        for sig in arbs:
            if self._throttled('risk_free_arb', sig['market_id']):
                continue
            self.stats['arb_signals'] += 1
            await self._save_signal(sig)
            # Shadow trade record (executor simulates unless ALL live gates pass).
            # Two-leg path: YES + NO together; leg-2 failure unwinds leg-1.
            result = await self.executor.execute_arb(
                market_id=sig['market_id'], question=sig['question'],
                yes_token_id=sig['details'].get('yes_token_id'),
                no_token_id=sig['details'].get('no_token_id'),
                yes_price=sig['yes_price'], no_price=sig['no_price'],
                size_usd=float(self.config.get('max_position_size_usd', 50)),
                expected_edge_bps=sig['edge_bps'], module_dry_run=self.module_dry_run,
                details=sig['details'],
            )
            logger.info(
                f"[arb] {sig['market_id']} edge={sig['edge_bps']}bps "
                f"pair_cost={sig['details']['pair_cost']} -> {result['status']}"
                f"{' (' + str(result['skip_reason']) + ')' if result['skip_reason'] else ''}"
            )

        # (b) Event-momentum ADVICE signals (never traded automatically).
        exclude_raw = str(self.config.get('momentum_exclude_categories', '') or '')
        moms = score_momentum(
            markets, self._prev_yes_prices, self._prev_seen_at,
            min_liquidity_usd=float(self.config.get('momentum_min_liquidity_usd', 10000)),
            min_volume_24h_usd=float(self.config.get('momentum_min_volume_24h_usd', 5000)),
            min_move_frac=float(self.config.get('momentum_min_move_frac', 0.05)),
            min_score=float(self.config.get('momentum_min_score', 0.3)),
            exclude_categories={c.strip() for c in exclude_raw.split(',') if c.strip()},
            new_market_max_age_hours=float(self.config.get('new_market_max_age_hours', 24)),
        )
        for sig in moms:
            if (sig['signal_type'] == 'momentum'
                    and self._flip_suppressed(sig['market_id'], sig['direction'])):
                continue
            if self._throttled(sig['signal_type'], sig['market_id']):
                continue
            self.stats['momentum_signals'] += 1
            await self._save_signal(sig)
            logger.info(
                f"[{sig['signal_type']}] {sig['market_id']} dir={sig['direction']} "
                f"score={sig['score']} yes={sig['yes_price']}"
            )

        # (c) Per-cycle price snapshots (dashboard charts + LATE outcome marks).
        await self._save_snapshots(markets)
        # (d) Forward-outcome marking for past signals (edge proof).
        marked, pending = await self._mark_outcomes(
            {m['market_id']: m for m in markets})
        # Wave-F6 heartbeat: the edge-proof pipeline used to log only on
        # error, making the outcome scorecard unverifiable from logs
        # (04_advisory_sweep.md). One INFO line per cycle, logging only.
        logger.info(
            f"polymarket outcomes: {marked} marked, "
            f"{self.stats['snapshots_last_cycle']} snapshots written, "
            f"{pending} signals pending-horizon "
            f"(cumulative marked={self.stats['outcomes_marked']})"
        )

        self._prev_yes_prices = {m['market_id']: m['yes_price'] for m in markets}
        self._prev_seen_at = time.time()

    async def _save_snapshots(self, markets: list) -> None:
        """Write one polymarket_price_snapshots row per top-N watched market
        each cycle; prune old rows hourly. Fail-soft."""
        self.stats['snapshots_last_cycle'] = 0  # honest even when nothing writes
        if not self.db_pool:
            return
        top_n = int(self.config.get('snapshot_top_n_markets', 50))
        rows = sorted(markets, key=lambda m: m.get('volume_24h') or 0.0,
                      reverse=True)[:max(top_n, 0)]
        if not rows:
            return
        try:
            async with self.db_pool.acquire() as conn:
                await conn.executemany(
                    """
                    INSERT INTO polymarket_price_snapshots
                        (market_id, question, category, yes_price, no_price,
                         best_bid, best_ask, volume_24h, liquidity)
                    VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9)
                    """,
                    [(m['market_id'], m.get('question', ''), m.get('category', ''),
                      m.get('yes_price'), m.get('no_price'),
                      m.get('best_bid'), m.get('best_ask'),
                      m.get('volume_24h'), m.get('liquidity')) for m in rows],
                )
                self.stats['snapshots_last_cycle'] = len(rows)
                # Retention: prune at most once an hour (disk discipline).
                if time.monotonic() - self._last_snapshot_prune > 3600.0:
                    self._last_snapshot_prune = time.monotonic()
                    days = int(self.config.get('snapshot_retention_days', 14))
                    await conn.execute(
                        "DELETE FROM polymarket_price_snapshots "
                        "WHERE ts < NOW() - make_interval(days => $1)", max(days, 1))
        except Exception as e:
            logger.error(f"polymarket_price_snapshots insert failed: {e}")

    async def _mark_outcomes(self, markets_by_id: dict) -> tuple:
        """LATE forward-outcome marks for polymarket_signals (edge proof).
        Returns (marked_this_cycle, pending_horizon_count) for the heartbeat.

        Mirrors smart_money: each horizon (1h/6h/24h) is marked only after it
        has FULLY elapsed, using the yes_price observed NOW — from the current
        cycle if the market is still watched, else the latest snapshot.
        fwd_return_* is probability points x100, SIGNED by direction (positive
        = the signal pointed the right way). Signals older than 48h with no
        price source left are closed out honestly with NULL returns.
        """
        if not self.db_pool:
            return 0, 0
        try:
            async with self.db_pool.acquire() as conn:
                rows = await conn.fetch(
                    """
                    SELECT s.id, s.market_id, s.signal_type, s.direction,
                           s.yes_price, s.created_at,
                           (o.signal_id IS NOT NULL) AS has_row,
                           o.yes_price_1h, o.yes_price_6h, o.yes_price_24h
                    FROM polymarket_signals s
                    LEFT JOIN polymarket_signal_outcomes o ON o.signal_id = s.id
                    WHERE s.created_at <= NOW() - INTERVAL '1 hour'
                      AND s.created_at >= NOW() - INTERVAL '7 days'
                      AND (o.signal_id IS NULL OR NOT o.fully_marked)
                    ORDER BY s.created_at ASC
                    LIMIT 200
                    """
                )
                if not rows:
                    return 0, 0
                missing = list({r['market_id'] for r in rows
                                if r['market_id'] not in markets_by_id})
                snap_prices: dict = {}
                if missing:
                    snaps = await conn.fetch(
                        """
                        SELECT DISTINCT ON (market_id) market_id, yes_price
                        FROM polymarket_price_snapshots
                        WHERE market_id = ANY($1::text[])
                        ORDER BY market_id, ts DESC
                        """, missing)
                    snap_prices = {s['market_id']: s['yes_price'] for s in snaps}
                now = time.time()
                marked = 0
                pending = 0
                for r in rows:
                    market = markets_by_id.get(r['market_id'])
                    price_now = market['yes_price'] if market \
                        else snap_prices.get(r['market_id'])
                    sig_ts = r['created_at'].timestamp()
                    abandoned = price_now is None and (now - sig_ts) > 48 * 3600
                    if (price_now is None and not abandoned) or r['yes_price'] is None:
                        pending += 1
                        continue
                    sign = -1.0 if r['direction'] == 'NO' else 1.0
                    prices, returns = {}, {}
                    for label, secs in OUTCOME_HORIZONS:
                        existing = r[f'yes_price_{label}'] if r['has_row'] else None
                        if existing is not None:
                            val = float(existing)
                        elif price_now is not None and sig_ts + secs <= now:
                            val = float(price_now)  # horizon fully elapsed: LATE mark
                        else:
                            val = None
                        prices[label] = val
                        returns[label] = (
                            round((val - float(r['yes_price'])) * 100.0 * sign, 4)
                            if val is not None else None)
                    fully = abandoned or all(v is not None for v in prices.values())
                    if not fully and all(
                            prices[label] == (r[f'yes_price_{label}'] if r['has_row'] else None)
                            for label, _ in OUTCOME_HORIZONS):
                        pending += 1
                        continue  # nothing new to write this tick
                    await conn.execute(
                        """
                        INSERT INTO polymarket_signal_outcomes
                            (signal_id, market_id, signal_type, direction,
                             yes_price_at_signal, yes_price_1h, yes_price_6h,
                             yes_price_24h, fwd_return_1h, fwd_return_6h,
                             fwd_return_24h, fully_marked, updated_at)
                        VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,NOW())
                        ON CONFLICT (signal_id) DO UPDATE SET
                            yes_price_1h  = EXCLUDED.yes_price_1h,
                            yes_price_6h  = EXCLUDED.yes_price_6h,
                            yes_price_24h = EXCLUDED.yes_price_24h,
                            fwd_return_1h  = EXCLUDED.fwd_return_1h,
                            fwd_return_6h  = EXCLUDED.fwd_return_6h,
                            fwd_return_24h = EXCLUDED.fwd_return_24h,
                            fully_marked   = EXCLUDED.fully_marked,
                            updated_at     = NOW()
                        """,
                        r['id'], r['market_id'], r['signal_type'], r['direction'],
                        float(r['yes_price']), prices['1h'], prices['6h'],
                        prices['24h'], returns['1h'], returns['6h'],
                        returns['24h'], fully,
                    )
                    marked += 1
                    if not fully:
                        pending += 1  # partially marked; later horizons still due
                if marked:
                    self.stats['outcomes_marked'] += marked
                return marked, pending
        except Exception as e:
            logger.error(f"outcome marking failed (continuing): {e}")
            return 0, 0

    async def run(self) -> None:
        self.running = True
        logger.info("Polymarket engine loop starting "
                    f"(shadow_mode={self.config.get('shadow_mode', True)}, "
                    f"live_execution_enabled={self.config.get('live_execution_enabled', False)})")
        # Startup reconciliation: no-op in shadow; with live gates open it lists
        # resting CLOB orders + rebuilds the exposure map from the trade ledger.
        try:
            await self.executor.reconcile_open_orders(self.module_dry_run)
        except Exception as e:
            logger.error(f"Startup reconcile failed (continuing): {e}")
        while self.running:
            try:
                await self._cycle()
                self.stats['cycles'] += 1
                self.stats['last_cycle_at'] = datetime.now(timezone.utc).isoformat()
                self.stats['last_error'] = None
            except asyncio.CancelledError:
                raise
            except Exception as e:  # fail-soft: the subprocess never dies on a cycle error
                self.stats['last_error'] = f"{type(e).__name__}: {e}"
                logger.error(f"Cycle error (continuing): {self.stats['last_error']}")
            await asyncio.sleep(float(self.config.get('poll_interval_s', 60)))

    async def stop(self) -> None:
        self.running = False


class PolymarketHealthServer:
    """GET /health (liveness) + GET /status (stats). Port: POLYMARKET_HEALTH_PORT (8089)."""

    def __init__(self, engine: PolymarketEngine, host: str = "0.0.0.0", port: int = 8089):
        self.engine = engine
        self.host = host
        self.port = port
        self._runner = None

    async def start(self) -> None:
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_get("/status", self._health)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.info(f"Health server running at http://{self.host}:{self.port}")

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()

    async def _health(self, request: web.Request) -> web.Response:
        body = {
            "status": "running" if self.engine.running else "stopped",
            "module": "polymarket",
            "shadow_mode": bool(self.engine.config.get('shadow_mode', True)),
            "live_execution_enabled": bool(self.engine.config.get('live_execution_enabled', False)),
            "dry_run": self.engine.module_dry_run,
            "gamma_last_error": self.engine.gamma.last_error,
            "live_orders": self.engine.executor.live_orders,
            "simulated_records": self.engine.executor.simulated_records,
            "live_exposure_usd": round(sum(self.engine.executor.live_exposure.values()), 2),
            "live_open_markets": len(self.engine.executor.live_exposure),
            "open_orders_at_start": self.engine.executor.open_orders_at_start,
            **self.engine.stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return web.Response(text=json.dumps(body), content_type="application/json")


async def main():
    logger.info("=== Polymarket Module Starting (shadow-first) ===")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Per-module DRY_RUN (POLYMARKET_DRY_RUN beats DRY_RUN; safe default True).
    module_dry_run = True
    try:
        from core.dry_run import resolve_module_dry_run, start_killswitch_poller
        module_dry_run = resolve_module_dry_run('polymarket', default=True)
        logger.info(f"   DRY_RUN (resolved per-module): {module_dry_run}")
        start_killswitch_poller()
    except Exception as e:
        logger.warning(f"   Could not resolve DRY_RUN / start killswitch poller: {e}")

    # DB connection
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except ImportError:
        db_url = os.getenv('DATABASE_URL')
    if not db_url:
        logger.error("No database credentials (Docker secrets or DATABASE_URL). Exiting.")
        return
    try:
        import asyncpg
        db_pool = await asyncpg.create_pool(db_url, min_size=1, max_size=3)
        logger.info("   Database connected")
    except Exception as e:
        logger.error(f"   Database connection failed: {e}. Exiting.")
        return

    # Secrets manager (only needed if the operator later enables LIVE).
    try:
        from security.secrets_manager import secrets
        secrets.initialize(db_pool)
    except Exception as e:
        logger.warning(f"   Secrets manager init failed (live path degraded): {e}")

    config = await load_config(db_pool)
    # DB dry_run row (if present) takes precedence over env resolution.
    if 'dry_run' in config:
        try:
            from core.dry_run import resolve_module_dry_run
            module_dry_run = resolve_module_dry_run(
                'polymarket', db_row_value=str(config['dry_run']), default=True)
            logger.info(f"   DRY_RUN (DB override): {module_dry_run}")
        except Exception:
            pass

    # Live risk gate is built into the executor (per-market/total exposure
    # caps + max open markets). The old core.risk_manager.validate_trade call
    # ran EVM honeypot analysis on CLOB token ids — semantically wrong for
    # prediction markets (wave-F5 BUG-3) — and was replaced, NOT removed:
    # every live order still passes shadow_mode -> live_execution_enabled ->
    # should_skip_live -> the Polymarket risk gate, in that order.
    engine = PolymarketEngine(db_pool, config, module_dry_run)
    health_port = int(os.getenv('POLYMARKET_HEALTH_PORT', '8089'))
    health = PolymarketHealthServer(engine, port=health_port)
    try:
        await health.start()
    except Exception as e:
        logger.warning(f"   Health server failed to start (continuing): {e}")

    try:
        await engine.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Shutdown requested")
    finally:
        await engine.stop()
        await health.stop()
        await db_pool.close()
        logger.info("Polymarket module stopped")


if __name__ == "__main__":
    asyncio.run(main())
