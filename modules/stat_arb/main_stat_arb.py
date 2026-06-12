#!/usr/bin/env python3
"""
Stat-Arb Module — market-neutral pairs mean-reversion entry point (shadow-first).

Data source: FREE — market_data_warehouse reader if populated, else Bybit/
Binance public klines (no keys). Strategy: cointegration-gated pairs spread
z-score reversion on liquid USDT perps; every paired suggestion is recorded to
stat_arb_trades (is_simulated=true) with spread diagnostics in
stat_arb_spread_state. LIVE execution is OFF by default and gated:
shadow_mode=false AND live_execution_enabled=true AND not should_skip_live
AND RiskManager.validate_trade — see modules/stat_arb/executor.py.

Launched by main.py when STAT_ARB_MODULE_ENABLED=true (default false).
Health server: http://0.0.0.0:8104 (env override: STAT_ARB_HEALTH_PORT).
Config: DB-backed config_type='stat_arb' (migration 132).
Kill switches: logs/.killswitch (global), logs/.pause_stat_arb (per-module).
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
log_dir = Path("logs/stat_arb")
log_dir.mkdir(parents=True, exist_ok=True)
log_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("StatArbModule")
logger.setLevel(logging.INFO)
main_handler = RotatingFileHandler(log_dir / 'stat_arb.log',
                                   maxBytes=10 * 1024 * 1024, backupCount=5)
main_handler.setFormatter(log_formatter)
logger.addHandler(main_handler)
error_handler = RotatingFileHandler(log_dir / 'stat_arb_errors.log',
                                    maxBytes=5 * 1024 * 1024, backupCount=3)
error_handler.setFormatter(log_formatter)
error_handler.setLevel(logging.ERROR)
logger.addHandler(error_handler)
console = logging.StreamHandler()
console.setFormatter(log_formatter)
logger.addHandler(console)
for sub in ("StatArbModule.Engine", "StatArbModule.Executor", "StatArbModule.Data"):
    sub_logger = logging.getLogger(sub)
    sub_logger.setLevel(logging.INFO)
    sub_logger.addHandler(main_handler)
    sub_logger.addHandler(error_handler)
    sub_logger.addHandler(console)

from modules.stat_arb.core.engine import StatArbEngine  # noqa: E402
from modules.stat_arb.executor import StatArbExecutor  # noqa: E402


async def load_config(db_pool) -> dict:
    """Load config_type='stat_arb' rows (mirrors the polymarket loader)."""
    settings = {}
    if not db_pool:
        return settings
    try:
        async with db_pool.acquire() as conn:
            rows = await conn.fetch(
                "SELECT key, value FROM config_settings WHERE config_type = 'stat_arb'"
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
    except Exception as e:
        logger.error(f"Failed to load stat_arb settings from DB: {e}")
    return settings


class StatArbHealthServer:
    """GET /health (liveness) + GET /status (stats). Port: STAT_ARB_HEALTH_PORT (8104)."""

    def __init__(self, engine: StatArbEngine, host: str = "0.0.0.0", port: int = 8104):
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
            "module": "stat_arb",
            "shadow_mode": bool(self.engine.config.get('shadow_mode', True)),
            "live_execution_enabled": bool(
                self.engine.config.get('live_execution_enabled', False)),
            "dry_run": self.engine.module_dry_run,
            "open_pair_keys": sorted(self.engine.open_pairs),
            "live_orders": self.engine.executor.live_orders,
            "simulated_records": self.engine.executor.simulated_records,
            **self.engine.stats,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        return web.Response(text=json.dumps(body), content_type="application/json")


async def main():
    logger.info("=== Stat-Arb Module Starting (shadow-first, market-neutral pairs) ===")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Per-module DRY_RUN (STAT_ARB_DRY_RUN beats DRY_RUN; safe default True).
    module_dry_run = True
    try:
        from core.dry_run import resolve_module_dry_run, start_killswitch_poller
        module_dry_run = resolve_module_dry_run('stat_arb', default=True)
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
    logger.info(f"   Loaded {len(config)} stat_arb settings from database")
    # DB dry_run row (if present) takes precedence over env resolution.
    if 'dry_run' in config:
        try:
            from core.dry_run import resolve_module_dry_run
            module_dry_run = resolve_module_dry_run(
                'stat_arb', db_row_value=str(config['dry_run']), default=True)
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

    executor = StatArbExecutor(db_pool, config, risk_manager)
    engine = StatArbEngine(db_pool, config, executor, module_dry_run,
                           load_config=load_config)
    health_port = int(os.getenv('STAT_ARB_HEALTH_PORT', '8104'))
    health = StatArbHealthServer(engine, port=health_port)
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
        logger.info("Stat-arb module stopped")


if __name__ == "__main__":
    asyncio.run(main())
