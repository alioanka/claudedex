#!/usr/bin/env python3
"""Portfolio Allocator subprocess entry."""
import asyncio
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
load_dotenv()

log_dir = Path("logs/portfolio_allocator")
log_dir.mkdir(parents=True, exist_ok=True)

_fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PortfolioAllocatorModule")
logger.setLevel(logging.INFO)

h = RotatingFileHandler(log_dir / 'portfolio_allocator.log',
                        maxBytes=5 * 1024 * 1024, backupCount=3)
h.setFormatter(_fmt)
logger.addHandler(h)

eh = RotatingFileHandler(log_dir / 'portfolio_allocator_errors.log',
                          maxBytes=2 * 1024 * 1024, backupCount=3)
eh.setFormatter(_fmt)
eh.setLevel(logging.ERROR)
logger.addHandler(eh)

console = logging.StreamHandler()
console.setFormatter(_fmt)
logger.addHandler(console)

engine_logger = logging.getLogger("portfolio_allocator")
engine_logger.setLevel(logging.INFO)
for handler in logger.handlers:
    engine_logger.addHandler(handler)


async def main() -> None:
    logger.info("📊 Portfolio Allocator Module Starting...")
    logger.info(f"   Working dir: {Path.cwd()}")
    logger.info(f"   Log dir: {log_dir.absolute()}")

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

    tick_interval = int(os.getenv('PORTFOLIO_ALLOCATOR_TICK_INTERVAL', '3600'))
    lookback_hours = int(os.getenv('PORTFOLIO_ALLOCATOR_LOOKBACK_HOURS', '168'))
    total_book_usd = float(os.getenv('PORTFOLIO_TOTAL_BOOK_USD', '1000.0'))

    from modules.portfolio_allocator.core.rebalance_engine import run_loop
    try:
        await run_loop(
            pool,
            tick_interval_seconds=tick_interval,
            lookback_hours=lookback_hours,
            total_book_usd=total_book_usd,
        )
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
