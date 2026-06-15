#!/usr/bin/env python3
"""
Polymarket Module — prediction-market entry point (shadow-first).

Data source: Gamma REST API (read-only, free, no key, no on-chain risk).
Strategies: risk-free YES+NO arbitrage detector + event-momentum advice,
both shadow-recorded to polymarket_signals / polymarket_trades
(is_simulated=true). LIVE CLOB execution is OFF by default and gated:
shadow_mode=false AND live_execution_enabled=true AND not should_skip_live
AND RiskManager.validate_trade — see modules/polymarket/executor.py.

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
for sub in ("PolymarketModule.Gamma", "PolymarketModule.Executor"):
    sub_logger = logging.getLogger(sub)
    sub_logger.setLevel(logging.INFO)
    sub_logger.addHandler(main_handler)
    sub_logger.addHandler(error_handler)
    sub_logger.addHandler(console)

from modules.polymarket.gamma_client import GammaClient, DEFAULT_GAMMA_BASE_URL  # noqa: E402
from modules.polymarket.strategies import detect_risk_free_arb, score_momentum  # noqa: E402
from modules.polymarket.executor import PolymarketExecutor  # noqa: E402


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

    def __init__(self, db_pool, config: dict, risk_manager, module_dry_run: bool):
        self.db_pool = db_pool
        self.config = config
        self.module_dry_run = module_dry_run
        self.gamma = GammaClient(
            str(config.get('gamma_base_url', DEFAULT_GAMMA_BASE_URL)),
            max_requests_per_minute=int(config.get('gamma_max_requests_per_minute', 30)),
        )
        self.executor = PolymarketExecutor(db_pool, config, risk_manager)
        self.running = False
        self._prev_yes_prices: dict = {}
        self._prev_seen_at = None
        self._last_recorded: dict = {}  # (signal_type, market_id) -> monotonic ts
        self.stats = {
            'cycles': 0, 'markets_seen': 0, 'arb_signals': 0,
            'momentum_signals': 0, 'last_cycle_at': None, 'last_error': None,
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
        )
        for sig in arbs:
            if self._throttled('risk_free_arb', sig['market_id']):
                continue
            self.stats['arb_signals'] += 1
            await self._save_signal(sig)
            # Shadow trade record (executor simulates unless ALL live gates pass).
            result = await self.executor.execute(
                strategy='risk_free_arb', market_id=sig['market_id'],
                question=sig['question'], side='BUY', outcome='BOTH',
                token_id=sig['details'].get('yes_token_id'),
                price=sig['yes_price'],
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
        moms = score_momentum(
            markets, self._prev_yes_prices, self._prev_seen_at,
            min_liquidity_usd=float(self.config.get('momentum_min_liquidity_usd', 10000)),
            min_volume_24h_usd=float(self.config.get('momentum_min_volume_24h_usd', 5000)),
            min_move_frac=float(self.config.get('momentum_min_move_frac', 0.05)),
            min_score=float(self.config.get('momentum_min_score', 0.3)),
        )
        for sig in moms:
            if self._throttled(sig['signal_type'], sig['market_id']):
                continue
            self.stats['momentum_signals'] += 1
            await self._save_signal(sig)
            logger.info(
                f"[{sig['signal_type']}] {sig['market_id']} dir={sig['direction']} "
                f"score={sig['score']} yes={sig['yes_price']}"
            )

        self._prev_yes_prices = {m['market_id']: m['yes_price'] for m in markets}
        self._prev_seen_at = time.time()

    async def run(self) -> None:
        self.running = True
        logger.info("Polymarket engine loop starting "
                    f"(shadow_mode={self.config.get('shadow_mode', True)}, "
                    f"live_execution_enabled={self.config.get('live_execution_enabled', False)})")
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

    risk_manager = None
    try:
        from core.risk_manager import RiskManager
        risk_manager = RiskManager(config={}, portfolio_manager=None, config_manager=None)
        logger.info("   core.risk_manager wired (validate_trade gates any live order)")
    except Exception as e:
        logger.warning(f"   RiskManager unavailable ({e}) — live path will refuse to fire")

    engine = PolymarketEngine(db_pool, config, risk_manager, module_dry_run)
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
