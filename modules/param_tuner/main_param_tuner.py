#!/usr/bin/env python3
"""PARAM_TUNER module — entry point.

Bandit-based, SHADOW-ONLY self-tuning of registered module knobs. Reads
each owning module's rolling closed-trade performance, feeds it to a
deterministic UCB1 bandit per tunable, and writes value-change PROPOSALS
to param_proposals for operator approval. By default it never writes a
live config row; auto-apply is gated behind DB auto_apply_enabled=false
and, when enabled, applies only kind='exploit' proposals within the
operator's [min, max] bounds to config_settings — logged + reversible.
It never trades, never touches the killswitch, never edits risk knobs.

Launched by main.py when PARAM_TUNER_MODULE_ENABLED=true (default false).
Health server: port PARAM_TUNER_HEALTH_PORT (default 8101).
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

log_dir = Path("logs/param_tuner")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ParamTunerModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'param_tuner.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

err_h = RotatingFileHandler(log_dir / 'param_tuner_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.ERROR)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("param_tuner")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"tunables": 0, "rewarded": 0, "proposed": 0,
                       "applied": 0, "auto_apply": False}


async def _start_health_server(port: int) -> None:
    """Minimal liveness/status server (fail-soft if aiohttp/port unavailable)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "param_tuner"})

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
    logger.info("🎛️ Param Tuner Module Starting (SHADOW-ONLY by default)...")
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

    health_port = int(os.getenv('PARAM_TUNER_HEALTH_PORT', '8101'))
    await _start_health_server(health_port)

    import modules.param_tuner.core.tuner_engine as _eng
    from modules.param_tuner.core.tuner_engine import load_tuner_config, run_loop

    async def get_config():
        async with pool.acquire() as conn:
            cfg = await load_tuner_config(conn)
        # env fallbacks so the subprocess starts before the config_type exists
        cfg.setdefault('tick_interval_seconds',
                       int(os.getenv('PARAM_TUNER_TICK_INTERVAL', '3600')))
        cfg.setdefault('lookback_hours',
                       int(os.getenv('PARAM_TUNER_LOOKBACK_HOURS', '24')))
        return cfg

    # Wrap run_tick to capture the latest summary for /status (fail-soft).
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
