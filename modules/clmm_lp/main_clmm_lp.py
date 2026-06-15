#!/usr/bin/env python3
"""
CLMM LP Module — concentrated-liquidity market-making entry point
(SHADOW/ADVISORY-FIRST; default DISABLED).

Polls free public pool data (DexScreener; on-chain fee verification via
config.pool_engine, read-only) for a small configured candidate set
(Uniswap v3 / Orca) and shadow-records proposed range positions with a
transparent net-of-IL expected APR (is_simulated=true). LIVE mint/rebalance/
burn is gated (shadow_mode + live_execution_enabled + should_skip_live +
RiskManager) and additionally NOT IMPLEMENTED in v1 — see executor.py.

Launched by main.py when CLMM_LP_MODULE_ENABLED=true (default false).
Health server: http://0.0.0.0:8100 (env override: CLMM_LP_HEALTH_PORT).
Config: DB-backed config_type='clmm_lp' (migration 128).
Kill switches: logs/.killswitch (global), logs/.pause_clmm_lp (per-module).
"""
import asyncio
import json
import logging
import os
import sys
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from aiohttp import web  # noqa: E402

# Setup logging
log_dir = Path("logs/clmm_lp")
log_dir.mkdir(parents=True, exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("ClmmLpModule")
logger.setLevel(logging.INFO)
main_handler = RotatingFileHandler(log_dir / 'clmm_lp.log',
                                   maxBytes=10 * 1024 * 1024, backupCount=5)
main_handler.setFormatter(log_formatter)
logger.addHandler(main_handler)
error_handler = RotatingFileHandler(log_dir / 'clmm_lp_errors.log',
                                    maxBytes=5 * 1024 * 1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)
for sub in ("ClmmLpModule.Engine", "ClmmLpModule.Executor", "ClmmLpModule.PoolData"):
    sub_logger = logging.getLogger(sub)
    sub_logger.setLevel(logging.INFO)
    sub_logger.addHandler(main_handler)
    sub_logger.addHandler(error_handler)
    sub_logger.addHandler(console)

from modules.clmm_lp.engine import ClmmLpEngine, load_config  # noqa: E402


class ClmmLpHealthServer:
    """GET /health + /status. Port: CLMM_LP_HEALTH_PORT (default 8100)."""

    def __init__(self, engine: ClmmLpEngine, host: str = "0.0.0.0", port: int = 8100):
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
            "module": "clmm_lp",
            "shadow_mode": bool(self.engine.config.get('shadow_mode', True)),
            "live_execution_enabled": bool(self.engine.config.get('live_execution_enabled', False)),
            "dry_run": self.engine.module_dry_run,
            "pool_data_last_error": self.engine.pool_data.last_error,
            "live_orders": self.engine.executor.live_orders,
            "simulated_records": self.engine.executor.simulated_records,
            **self.engine.stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return web.Response(text=json.dumps(body), content_type="application/json")


async def main():
    logger.info("=== CLMM LP Module Starting (shadow/advisory-first) ===")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Per-module DRY_RUN (CLMM_LP_DRY_RUN beats DRY_RUN; safe default True).
    module_dry_run = True
    try:
        from core.dry_run import resolve_module_dry_run, start_killswitch_poller
        module_dry_run = resolve_module_dry_run('clmm_lp', default=True)
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

    # Connect the shared PoolEngine ONCE with this subprocess's DB pool so
    # get_endpoint() reads RPC URLs from the DB rpc_api_pool table (set via the
    # dashboard /settings/rpc-api page). Without this the engine would fall back
    # to .env RPC URLs. READ-ONLY use (endpoint selection + health reports).
    try:
        from config.pool_engine import PoolEngine
        rpc = await PoolEngine.get_instance()
        if not rpc.initialized:
            await rpc.initialize(db_pool)
        logger.info("   PoolEngine connected (DB-sourced RPC) for clmm_lp")
    except Exception as exc:
        logger.warning("PoolEngine init failed (fail-soft to .env RPC): %s", exc)

    config = await load_config(db_pool)
    logger.info(f"   Loaded {len(config)} clmm_lp settings from database")
    # DB dry_run row (if present) takes precedence over env resolution.
    if 'dry_run' in config:
        try:
            from core.dry_run import resolve_module_dry_run
            module_dry_run = resolve_module_dry_run(
                'clmm_lp', db_row_value=str(config['dry_run']), default=True)
            logger.info(f"   DRY_RUN (DB override): {module_dry_run}")
        except Exception:
            pass

    risk_manager = None
    try:
        from core.risk_manager import RiskManager
        risk_manager = RiskManager(config={}, portfolio_manager=None, config_manager=None)
        logger.info("   core.risk_manager wired (gates any future live action)")
    except Exception as e:
        logger.warning(f"   RiskManager unavailable ({e}) — live path will refuse to fire")

    engine = ClmmLpEngine(db_pool, config, risk_manager, module_dry_run)
    health_port = int(os.getenv('CLMM_LP_HEALTH_PORT', '8100'))
    health = ClmmLpHealthServer(engine, port=health_port)
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
        logger.info("CLMM LP module stopped")


if __name__ == "__main__":
    asyncio.run(main())
