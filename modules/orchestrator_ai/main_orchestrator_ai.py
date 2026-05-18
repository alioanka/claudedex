#!/usr/bin/env python3
"""Orchestrator AI module — entry point.

Reads per-module DRY_RUN performance + market state every
`tick_interval_seconds`, emits non-'hold' recommendations to the
`orchestrator_recommendations` table. Operator approves via dashboard.

No live trading. No automatic actions on its own.
"""
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Project root on path so `from core.dry_run import ...` works the
# same way every other module's main does.
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

load_dotenv()

log_dir = Path("logs/orchestrator_ai")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("OrchestratorAIModule")
logger.setLevel(logging.INFO)

main_h = RotatingFileHandler(log_dir / 'orchestrator_ai.log',
                              maxBytes=5 * 1024 * 1024, backupCount=3)
main_h.setFormatter(_fmt)
logger.addHandler(main_h)

err_h = RotatingFileHandler(log_dir / 'orchestrator_ai_errors.log',
                             maxBytes=2 * 1024 * 1024, backupCount=3)
err_h.setFormatter(_fmt)
err_h.setLevel(logging.ERROR)
logger.addHandler(err_h)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

# Propagate the orchestrator engine's own logger to the same handlers.
engine_logger = logging.getLogger("orchestrator_ai")
engine_logger.setLevel(logging.INFO)
for h in logger.handlers:
    engine_logger.addHandler(h)


async def main() -> None:
    logger.info("🤖 Orchestrator AI Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")
    # This module never trades — DRY_RUN is irrelevant. But honor the
    # kill-switch poll so the dashboard can stop it cleanly.
    from core.dry_run import start_killswitch_poller
    try:
        start_killswitch_poller()
    except Exception:
        pass

    # DB pool reuse: connect via the same path every other module uses.
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
        user=db_user, password=db_pass,
        min_size=1, max_size=2,
    )
    logger.info(f"   DB pool connected to {db_host}:{db_port}/{db_name}")

    # Config — read from env with sane defaults; DB-config wiring is a
    # follow-up commit so this subprocess can start before the
    # orchestrator config_type exists in DB.
    tick_interval = int(os.getenv('ORCHESTRATOR_TICK_INTERVAL', '300'))
    lookback_hours = int(os.getenv('ORCHESTRATOR_LOOKBACK_HOURS', '24'))
    ttl_minutes = int(os.getenv('ORCHESTRATOR_REC_TTL_MINUTES', '60'))

    from modules.orchestrator_ai.core.orchestrator_engine import run_loop
    try:
        await run_loop(
            pool,
            tick_interval_seconds=tick_interval,
            lookback_hours=lookback_hours,
            recommendation_ttl_minutes=ttl_minutes,
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
