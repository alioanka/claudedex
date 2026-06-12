#!/usr/bin/env python3
"""CATALYST_CALENDAR module — entry point.

Advisory forward-catalyst feed: aggregates token unlock cliffs (DefiLlama,
best-effort), exchange listing/delisting announcements (Binance CMS,
scrape-class), and scheduled macro events (static FOMC schedule + key-gated
FMP) into the `catalysts` table for other modules and the operator to read.
PURE ADVISORY — never trades, never writes flags, holds no required keys.

Launched by main.py when CATALYST_CALENDAR_MODULE_ENABLED=true (default false).
Health server: port CATALYST_CALENDAR_HEALTH_PORT (default 8096).
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

log_dir = Path("logs/catalyst_calendar")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("CatalystCalendarModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'catalyst_calendar.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

err_h = RotatingFileHandler(log_dir / 'catalyst_calendar_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.ERROR)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("catalyst_calendar")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"tick_at": None, "events_collected": 0,
                       "rows_upserted": 0, "sources": {}}


async def _start_health_server(port: int) -> None:
    """Minimal liveness/status server (fail-soft if aiohttp/port unavailable)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "catalyst_calendar"})

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
    logger.info("Catalyst Calendar Module Starting...")
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

    health_port = int(os.getenv('CATALYST_CALENDAR_HEALTH_PORT', '8096'))
    await _start_health_server(health_port)

    from modules.catalyst_calendar.core.calendar_engine import (
        load_calendar_config, run_loop)

    async def get_config():
        cfg = await load_calendar_config(pool)
        # env fallbacks so the subprocess starts before the config_type exists
        cfg.setdefault('refresh_interval_seconds',
                       int(os.getenv('CATALYST_REFRESH_INTERVAL', '3600')))
        cfg.setdefault('lookahead_days',
                       int(os.getenv('CATALYST_LOOKAHEAD_DAYS', '30')))
        return cfg

    def on_summary(s: dict) -> None:
        global _LAST_SUMMARY
        _LAST_SUMMARY = s

    try:
        await run_loop(pool, get_config=get_config, on_summary=on_summary)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
