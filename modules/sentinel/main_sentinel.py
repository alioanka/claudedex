#!/usr/bin/env python3
"""SENTINEL module — entry point.

Cross-module anomaly detection + auto-freeze advisor. Watches minutes-scale
distributions no per-module gate sees (stablecoin depeg, cross-source price
divergence, silent module death, 100%-rejection patterns, abnormal loss
velocity, correlated drawdown) and records graded anomaly events to
sentinel_anomalies. ADVISORY BY DEFAULT. Only when sentinel_autopilot_enabled
(DB, default false) may it freeze an offending module by writing
logs/.pause_<module> (dwell-guarded; clears only its own freezes). It never
trades, never touches logs/.killswitch, never edits a risk gate.

Launched by main.py when SENTINEL_MODULE_ENABLED=true (default false).
Health server: port SENTINEL_HEALTH_PORT (default 8094).
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

log_dir = Path("logs/sentinel")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("SentinelModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'sentinel.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

err_h = RotatingFileHandler(log_dir / 'sentinel_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.ERROR)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("sentinel")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"anomalies": 0, "critical": 0, "actions": 0,
                       "autopilot": False, "detected": []}


async def _start_health_server(port: int) -> None:
    """Minimal liveness/status server (fail-soft if aiohttp/port unavailable)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "sentinel"})

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
    logger.info("🛰️ Sentinel Module Starting...")
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

    health_port = int(os.getenv('SENTINEL_HEALTH_PORT', '8094'))
    await _start_health_server(health_port)

    from modules.sentinel.core.sentinel_engine import (
        DEFAULT_CONFIG, load_sentinel_config, run_loop,
    )
    import modules.sentinel.core.sentinel_engine as _eng

    async def get_config():
        cfg = await load_sentinel_config(pool)
        # env fallbacks so the subprocess starts before the config_type exists
        if 'tick_interval_seconds' not in cfg or cfg['tick_interval_seconds'] == \
                DEFAULT_CONFIG['tick_interval_seconds']:
            cfg['tick_interval_seconds'] = int(
                os.getenv('SENTINEL_TICK_INTERVAL',
                          str(DEFAULT_CONFIG['tick_interval_seconds'])))
        return cfg

    # Capture the latest tick summary for /status (wrap, fail-soft).
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
