#!/usr/bin/env python3
"""SMART_MONEY module — entry point. ADVISORY ONLY, never trades.

Detects wallets that repeatedly front profitable on-chain moves (scored by
realized FORWARD return, no look-ahead), clusters their fresh accumulation,
and writes advisory rows to smart_money_signals. No broadcast path exists in
this module; the operator (or a future copy/DEX consumer) decides what to do.

Launched by main.py when SMART_MONEY_MODULE_ENABLED=true (default false).
Health server: port SMART_MONEY_HEALTH_PORT (default 8105).
"""
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

log_dir = Path("logs/smart_money")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SmartMoneyModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'smart_money.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

err_h = RotatingFileHandler(log_dir / 'smart_money_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.WARNING)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("smart_money")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"events_ingested": 0, "events_marked": 0,
                       "wallets_scored": 0, "signals_emitted": 0}


async def _start_health_server(port: int) -> None:
    """Minimal liveness/status server (fail-soft if aiohttp/port unavailable)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "smart_money"})

    async def status(_req):
        return web.json_response({"status": "ok", **_LAST_SUMMARY})

    app = web.Application()
    app.add_routes([web.get('/health', health), web.get('/status', status)])
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        logger.info("   Health server on :%d", port)
    except Exception as exc:
        logger.warning("health server bind failed on :%d (fail-soft): %s", port, exc)


async def main() -> None:
    logger.info("Smart Money Module Starting (ADVISORY ONLY)...")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Never trades — but honor the killswitch poll so the dashboard can stop it.
    from core.dry_run import start_killswitch_poller
    try:
        start_killswitch_poller()
    except Exception:
        pass

    import asyncpg
    db_host = os.getenv('DB_HOST', 'postgres')
    db_port = int(os.getenv('DB_PORT', '5432'))
    db_name = os.getenv('DB_NAME', 'tradingbot')
    db_user = (
        Path('/run/secrets/db_user').read_text().strip()
        if Path('/run/secrets/db_user').exists()
        else os.getenv('DB_USER', 'tradingbot')
    )
    db_pass = (
        Path('/run/secrets/db_password').read_text().strip()
        if Path('/run/secrets/db_password').exists()
        else os.getenv('DB_PASSWORD', '')
    )
    pool = await asyncpg.create_pool(
        host=db_host, port=db_port, database=db_name,
        user=db_user, password=db_pass, min_size=1, max_size=2,
    )
    logger.info(f"   DB pool connected to {db_host}:{db_port}/{db_name}")

    # Connect the shared PoolEngine ONCE (idempotent; READ-ONLY use: endpoint
    # selection + health reports for eth_getLogs / eth_call reads).
    try:
        from config.pool_engine import PoolEngine
        rpc = await PoolEngine.get_instance()
        if not rpc.initialized:
            await rpc.initialize(pool)
        logger.info("   PoolEngine connected for smart_money RPC reads")
    except Exception as exc:
        logger.warning("PoolEngine init failed (tick will retry, fail-soft): %s", exc)

    health_port = int(os.getenv('SMART_MONEY_HEALTH_PORT', '8105'))
    await _start_health_server(health_port)

    from modules.smart_money.core.flow_engine import run_loop, load_smart_money_config

    async def get_config():
        cfg = await load_smart_money_config(pool)
        # env fallback so the subprocess runs before migration 133 is applied
        cfg.setdefault('poll_interval_seconds',
                       int(os.getenv('SMART_MONEY_POLL_INTERVAL', '300')))
        return cfg

    # Wrap run_tick to capture the latest summary for /status (fail-soft).
    import modules.smart_money.core.flow_engine as _eng
    _orig_run_tick = _eng.run_tick

    async def _wrapped(pool_, cfg_):
        global _LAST_SUMMARY
        s = await _orig_run_tick(pool_, cfg_)
        _LAST_SUMMARY = s
        return s

    _eng.run_tick = _wrapped

    try:
        await run_loop(pool, get_config=get_config)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
