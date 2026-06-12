#!/usr/bin/env python3
"""YIELD_TREASURY module — entry point. ADVISORY / OBSERVE-FIRST.

Tracks idle-capital carry options (Aave v3 USDC supply, jitoSOL, stETH) via
free public RPC/APIs, sizes what the bot's idle float could earn net of
gas/fees, and writes ADVICE rows (yield_treasury_advice). It NEVER deposits,
NEVER signs, NEVER touches the killswitch in this build.

Launched by main.py when YIELD_TREASURY_MODULE_ENABLED=true (default false).
Health server: port YIELD_TREASURY_HEALTH_PORT (default 8098).
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

log_dir = Path("logs/yield_treasury")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("YieldTreasuryModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'yield_treasury.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

# WARNING level so YIELD ADVICE lines land where the operator watches.
err_h = RotatingFileHandler(log_dir / 'yield_treasury_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.WARNING)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("yield_treasury")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"venues": 0, "advised": 0, "deploy_candidates": 0,
                       "detail": {}}


async def _start_health_server(port: int) -> None:
    """Minimal liveness/status server (fail-soft if aiohttp/port unavailable)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "yield_treasury"})

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
    logger.info("Yield Treasury Module Starting (ADVISORY — observe-first)...")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Advisory-only — but honor the killswitch poll so the dashboard can stop it.
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

    # Connect the shared PoolEngine ONCE (idempotent; READ-ONLY use:
    # endpoint selection + health reports for the Aave eth_call reads).
    try:
        from config.pool_engine import PoolEngine
        rpc = await PoolEngine.get_instance()
        if not rpc.initialized:
            await rpc.initialize(pool)
        logger.info("   PoolEngine connected for yield_treasury RPC reads")
    except Exception as exc:
        logger.warning("PoolEngine init failed (tick will retry, fail-soft): %s", exc)

    health_port = int(os.getenv('YIELD_TREASURY_HEALTH_PORT', '8098'))
    await _start_health_server(health_port)

    from modules.yield_treasury.core.yield_engine import run_loop, load_yield_config

    async def get_config():
        cfg = await load_yield_config(pool)
        # env fallback so the subprocess runs before migration 126 is applied
        cfg.setdefault('poll_interval_seconds',
                       int(os.getenv('YIELD_TREASURY_POLL_INTERVAL', '900')))
        return cfg

    # Wrap run_tick to capture the latest summary for /status (fail-soft).
    import modules.yield_treasury.core.yield_engine as _eng
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
