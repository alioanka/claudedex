#!/usr/bin/env python3
"""TREASURY module — entry point. Phase 1 OBSERVE-ONLY.

Polls native + key token balances of the bot's wallets (public addresses
only), reconciles against the trade ledgers, writes treasury_snapshots, and
logs gas-starvation / unswept-profit / drift alerts. NEVER signs, NEVER
transfers, NEVER touches the killswitch.

Launched by main.py when TREASURY_MODULE_ENABLED=true (default false).
Health server: port TREASURY_HEALTH_PORT (default 8093).
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

log_dir = Path("logs/treasury")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TreasuryModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'treasury.log',
                             maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

# WARNING level so treasury ALERTS (gas_low / hot_wallet_high / drift) land in
# the error log the operator watches, not only ERROR-class failures.
err_h = RotatingFileHandler(log_dir / 'treasury_errors.log',
                            maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.WARNING)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("treasury")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


_LAST_SUMMARY: dict = {"wallets": 0, "observed": 0, "alerts": 0, "detail": {}}


async def _start_health_server(port: int) -> None:
    """Minimal liveness/status server (fail-soft if aiohttp/port unavailable)."""
    try:
        from aiohttp import web
    except Exception as exc:
        logger.warning("health server unavailable (aiohttp import failed): %s", exc)
        return

    async def health(_req):
        return web.json_response({"status": "ok", "module": "treasury"})

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
    logger.info("Treasury Module Starting (Phase 1 OBSERVE-ONLY)...")
    logger.info(f"   Working dir: {Path.cwd()}")

    # Observe-only — but honor the killswitch poll so the dashboard can stop it.
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

    # Secrets manager MUST be initialized with the DB pool BEFORE the first
    # tick: discover_wallets() derives the observed wallet addresses from the
    # secrets-managed private keys (PRIVATE_KEY / SOLANA_PRIVATE_KEY /
    # SOLANA_MODULE_PRIVATE_KEY). Without this call every derivation returned
    # None and — with the env address fallbacks unset — the module idled for
    # 20 days with zero snapshots. Same pattern as options_vol/copy_trading.
    # Fail-soft: on failure the tick still runs on env-address fallbacks.
    try:
        from security.secrets_manager import secrets
        secrets.initialize(pool)
        logger.info("   Secrets manager initialized (wallet addresses derive "
                    "from the secrets-managed private keys)")
    except Exception as exc:
        logger.warning("Secrets manager init failed — wallet discovery falls "
                       "back to env addresses (WALLET_ADDRESS/SOLANA_WALLET/"
                       "SOLANA_MODULE_WALLET): %s", exc)

    # Connect the shared PoolEngine ONCE (idempotent; fail-soft to .env pool)
    # so get_endpoint() inside the tick hits the live pool, not per-call
    # fallbacks. READ-ONLY use: endpoint selection + health reports only.
    try:
        from config.pool_engine import PoolEngine
        rpc = await PoolEngine.get_instance()
        if not rpc.initialized:
            await rpc.initialize(pool)
        logger.info("   PoolEngine connected for treasury RPC reads")
    except Exception as exc:
        logger.warning("PoolEngine init failed (tick will retry, fail-soft): %s", exc)

    health_port = int(os.getenv('TREASURY_HEALTH_PORT', '8093'))
    await _start_health_server(health_port)

    from modules.treasury.core.treasury_engine import run_loop, load_treasury_config

    async def get_config():
        cfg = await load_treasury_config(pool)
        # env fallback so the subprocess runs before migration 121 is applied
        cfg.setdefault('poll_interval_seconds',
                       int(os.getenv('TREASURY_POLL_INTERVAL', '300')))
        return cfg

    # Wrap run_tick to capture the latest summary for /status (fail-soft).
    import modules.treasury.core.treasury_engine as _eng
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
