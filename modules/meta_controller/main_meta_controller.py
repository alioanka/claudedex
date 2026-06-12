#!/usr/bin/env python3
"""META_CONTROLLER module — entry point.

Reads every trading module's rolling DRY_RUN + LIVE performance, writes a
transparent ACTIVATE / KEEP / PAUSE decision per module to meta_decisions, and
(only when meta_autopilot_enabled) actuates pause/resume flags with a dwell
guard. ADVISORY BY DEFAULT. Never trades, never touches the killswitch, never
edits a risk gate.

Launched by main.py when META_CONTROLLER_MODULE_ENABLED=true (default false).
Health server: port META_CONTROLLER_HEALTH_PORT (default 8090).
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

log_dir = Path("logs/meta_controller")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MetaControllerModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'meta_controller.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

err_h = RotatingFileHandler(log_dir / 'meta_controller_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.ERROR)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("meta_controller")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"scored": 0, "actuated": 0, "autopilot": False, "decisions": {}}


async def _start_health_server(port: int) -> None:
    """Minimal liveness/status server (fail-soft if aiohttp/port unavailable)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "meta_controller"})

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
    logger.info("🧠 Meta Controller Module Starting...")
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

    health_port = int(os.getenv('META_CONTROLLER_HEALTH_PORT', '8090'))
    await _start_health_server(health_port)

    from modules.meta_controller.core.meta_engine import run_loop, load_meta_config, run_tick

    async def get_config():
        cfg = await load_meta_config(pool)
        # env fallbacks so the subprocess starts before the config_type exists
        cfg.setdefault('tick_interval_seconds',
                       int(os.getenv('META_TICK_INTERVAL', '900')))
        cfg.setdefault('lookback_hours',
                       int(os.getenv('META_LOOKBACK_HOURS', '24')))
        return cfg

    # Patch run_tick to capture the latest summary for /status (wrap, fail-soft).
    import modules.meta_controller.core.meta_engine as _eng
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
