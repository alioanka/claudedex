#!/usr/bin/env python3
"""MARKET_DATA_WAREHOUSE module — entry point.

Unified historical market-data store. Periodically ingests normalized OHLCV
candles + funding-rate series for the symbols the bot trades from FREE public
sources (Binance/Bybit public REST — no keys) into deduplicated Postgres
tables (market_candles / market_series, migration 123), with retention-based
disk discipline. PURE DATA: never trades, never signs, holds no secrets.
Consumers read via modules.market_data_warehouse.reader.

Launched by main.py when MARKET_DATA_WAREHOUSE_MODULE_ENABLED=true (default
false). Health server: port MARKET_DATA_WAREHOUSE_HEALTH_PORT (default 8095)
with /health, /status, and a read-only /query endpoint.
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

log_dir = Path("logs/market_data_warehouse")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("MarketDataWarehouseModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'market_data_warehouse.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

err_h = RotatingFileHandler(log_dir / 'market_data_warehouse_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.ERROR)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("market_data_warehouse")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"candles_upserted": 0, "series_inserted": 0,
                       "purged": 0, "errors": 0}
_POOL = None  # set in main(); used by the read-only /query endpoint


async def _start_health_server(port: int) -> None:
    """Liveness/status + read-only /query (fail-soft if aiohttp/port missing)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok",
                                  "module": "market_data_warehouse"})

    async def status(_req):
        return web.json_response({"status": "ok", **_LAST_SUMMARY})

    async def query(req):
        """Read-only peek for operators/consumers, e.g.
        /query?symbol=BTC/USDT&timeframe=1h&limit=100
        /query?symbol=BTC/USDT&metric=funding_rate&limit=50
        /query?symbol=BTC/USDT&coverage=1   (history extent per source)"""
        if _POOL is None:
            return web.json_response({"error": "db not ready"}, status=503)
        from modules.market_data_warehouse import reader
        symbol = req.query.get("symbol", "")
        timeframe = req.query.get("timeframe", "1h")
        metric = req.query.get("metric")
        source = req.query.get("source") or None
        try:
            limit = min(int(req.query.get("limit", "100")), 500)
        except ValueError:
            limit = 100
        try:
            if not symbol:
                return web.json_response({"keys": await reader.list_symbols(_POOL)})
            if req.query.get("coverage"):
                return web.json_response(
                    await reader.get_coverage(_POOL, symbol, timeframe,
                                              source=source))
            if metric:
                rows = await reader.get_series(_POOL, symbol, metric,
                                               source=source, limit=limit)
            else:
                rows = await reader.get_candles(_POOL, symbol, timeframe,
                                                source=source, limit=limit)
            return web.json_response({"count": len(rows), "rows": rows})
        except Exception as exc:
            logger.warning("/query fail-soft: %s", exc)
            return web.json_response({"error": "query failed"}, status=500)

    app = web.Application()
    app.add_routes([web.get('/health', health), web.get('/status', status),
                    web.get('/query', query)])
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, '0.0.0.0', port)
        await site.start()
        logger.info("   Health server on :%d", port)
    except Exception as exc:
        logger.warning("health server bind failed on :%d (fail-soft): %s", port, exc)


async def main() -> None:
    global _POOL
    logger.info("Market Data Warehouse Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Pure data — but honor the killswitch poll so the dashboard can stop it.
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
    _POOL = pool
    logger.info(f"   DB pool connected to {db_host}:{db_port}/{db_name}")

    health_port = int(os.getenv('MARKET_DATA_WAREHOUSE_HEALTH_PORT', '8095'))
    await _start_health_server(health_port)

    from modules.market_data_warehouse.core.warehouse_engine import (
        load_warehouse_config, run_loop,
    )
    import modules.market_data_warehouse.core.warehouse_engine as _eng

    async def get_config():
        cfg = await load_warehouse_config(pool)
        # env fallback so the subprocess runs before the config_type is seeded
        cfg.setdefault('ingest_interval_seconds',
                       int(os.getenv('WAREHOUSE_INGEST_INTERVAL', '300')))
        return cfg

    # Wrap run_tick to surface the latest summary on /status (fail-soft).
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
