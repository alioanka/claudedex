#!/usr/bin/env python3
"""EXECUTION_GATEWAY module — optional health/diagnostics subprocess.

The value lives in the importable library (modules/execution_gateway/gateway.py)
that other modules call in-process; this subprocess only serves liveness +
route/send diagnostics. It NEVER trades and NEVER broadcasts.

Launched by main.py when EXECUTION_GATEWAY_MODULE_ENABLED=true (default false).
Health server: port EXECUTION_GATEWAY_HEALTH_PORT (default 8099).
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

log_dir = Path("logs/execution_gateway")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("ExecutionGatewayModule")
logger.setLevel(logging.INFO)
for h in (
    RotatingFileHandler(log_dir / 'execution_gateway.log',
                        maxBytes=5 * 1024 * 1024, backupCount=3),
    logging.StreamHandler(),
):
    h.setFormatter(_fmt)
    logger.addHandler(h)

_LAST_SUMMARY: dict = {"config_keys": 0, "sends_last_hour": {}, "db": False}


async def _start_health_server(port: int) -> None:
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "execution_gateway"})

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


async def _refresh_summary(pool) -> None:
    global _LAST_SUMMARY
    summary = {"config_keys": 0, "sends_last_hour": {}, "db": pool is not None}
    if pool is not None:
        try:
            summary["config_keys"] = await pool.fetchval(
                "SELECT COUNT(*) FROM config_settings "
                "WHERE config_type='execution_gateway'") or 0
            rows = await pool.fetch(
                "SELECT route, status, COUNT(*) AS n FROM execution_gateway_sends "
                "WHERE created_at > NOW() - INTERVAL '1 hour' "
                "GROUP BY route, status")
            summary["sends_last_hour"] = {
                f"{r['route'] or 'none'}_{r['status']}": r['n'] for r in rows}
        except Exception as exc:
            logger.warning("summary refresh fail-soft: %s", exc)
    _LAST_SUMMARY = summary


async def main() -> None:
    logger.info("Execution Gateway diagnostics starting (library lives in "
                "modules/execution_gateway/gateway.py; this process never sends)")

    from core.dry_run import start_killswitch_poller
    try:
        start_killswitch_poller()
    except Exception:
        pass

    pool = None
    try:
        import asyncpg  # optional — diagnostics degrade to liveness-only
        db_user = (Path('/run/secrets/db_user').read_text().strip()
                   if Path('/run/secrets/db_user').exists()
                   else os.getenv('DB_USER', 'tradingbot'))
        db_pass = (Path('/run/secrets/db_password').read_text().strip()
                   if Path('/run/secrets/db_password').exists()
                   else os.getenv('DB_PASSWORD', ''))
        pool = await asyncpg.create_pool(
            host=os.getenv('DB_HOST', 'postgres'),
            port=int(os.getenv('DB_PORT', '5432')),
            database=os.getenv('DB_NAME', 'tradingbot'),
            user=db_user, password=db_pass, min_size=1, max_size=2,
        )
        logger.info("   DB pool connected")
    except Exception as exc:
        logger.warning("asyncpg unavailable — liveness-only mode: %s", exc)

    health_port = int(os.getenv('EXECUTION_GATEWAY_HEALTH_PORT', '8099'))
    await _start_health_server(health_port)

    tick = int(os.getenv('EXECUTION_GATEWAY_TICK_SECONDS', '60'))
    try:
        while True:
            await _refresh_summary(pool)
            await asyncio.sleep(tick)
    finally:
        if pool is not None:
            await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
