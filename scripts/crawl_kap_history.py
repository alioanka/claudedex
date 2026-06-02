#!/usr/bin/env python3
"""
scripts/crawl_kap_history.py -- KAP historical disclosure backfill

Standalone operator script. Run ONCE (or periodically) to bootstrap the
kap_disclosures table with historical data before the real-time listener
accumulates enough for the classifier to learn from.

Usage
-----
  python scripts/crawl_kap_history.py --tickers THYAO,GARAN,EREGL --since 2023-01-01
  python scripts/crawl_kap_history.py --tickers ALL --since 2024-01-01
  python scripts/crawl_kap_history.py --tickers ASELS --since 2022-01-01 --dry-run
  python scripts/crawl_kap_history.py --resume   # re-run; skips already-crawled tickers

Flags
-----
  --tickers   Comma-separated BIST tickers (bare, no .IS suffix).
              Use ALL to crawl every ticker in watchlist_bist config.
  --since     Start date (YYYY-MM-DD). Default: 365 days ago.
  --until     End date (YYYY-MM-DD). Default: today.
  --pause     Inter-ticker pause in seconds. Default: 3.
  --dry-run   Fetch but do NOT write to DB. Shows count only.
  --resume    Skip tickers whose last_crawl_date >= until in DB.

IMPORTANT: Rate limiting
  Default pace: 3s between tickers, 2s between API pages.
  For 200 tickers x 5 years: estimated run time ~30 minutes.
  KAP CDN will rate-limit aggressively if you reduce these below 1s.
  If you get 429 responses, the script backs off 5 minutes automatically.
  For large backfills, run overnight or split across multiple --since/--until
  windows spread across multiple days.

Environment
-----------
  Reads DB connection from environment (DATABASE_URL or individual PG* vars),
  same as the rest of ClaudeDex.

Output
------
  Writes disclosures to kap_disclosures table (migration 062).
  Requires: ADVISOR_MODULE_ENABLED is irrelevant (this is an admin script).
  Requires: migration 062 has been applied.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import date, datetime, timedelta
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("crawl_kap_history")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Backfill KAP disclosures for BIST tickers.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--tickers", default="",
                   help="Comma-separated tickers (e.g. THYAO,GARAN) or ALL for watchlist.")
    p.add_argument("--since", default="",
                   help="Start date YYYY-MM-DD. Default: 365 days ago.")
    p.add_argument("--until", default="",
                   help="End date YYYY-MM-DD. Default: today.")
    p.add_argument("--pause", type=float, default=3.0,
                   help="Inter-ticker pause in seconds (default: 3).")
    p.add_argument("--dry-run", action="store_true",
                   help="Fetch but do not persist to DB.")
    p.add_argument("--resume", action="store_true",
                   help="Skip tickers already crawled up to --until date.")
    return p.parse_args()


async def get_db_pool():
    """Connect to DB using the same env vars as the rest of ClaudeDex."""
    import asyncpg
    from dotenv import load_dotenv
    load_dotenv()

    dsn = os.getenv("DATABASE_URL")
    if dsn:
        return await asyncpg.create_pool(dsn, min_size=1, max_size=3)

    return await asyncpg.create_pool(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=int(os.getenv("POSTGRES_PORT", "5432")),
        database=os.getenv("POSTGRES_DB", "claudedex"),
        user=os.getenv("POSTGRES_USER", "postgres"),
        password=os.getenv("POSTGRES_PASSWORD", ""),
        min_size=1, max_size=3,
    )


async def load_advisor_config(pool) -> dict:
    """Load advisor_config from DB (same pattern as AdvisorConfigManager)."""
    try:
        rows = await pool.fetch(
            "SELECT key, value FROM config_settings WHERE config_type='advisor_config'"
        )
        return {r["key"]: r["value"] for r in rows}
    except Exception as exc:
        logger.warning("Could not load advisor_config: %s", exc)
        return {}


async def main() -> None:
    args = parse_args()

    since_date = (
        date.fromisoformat(args.since) if args.since
        else (datetime.utcnow() - timedelta(days=365)).date()
    )
    until_date = (
        date.fromisoformat(args.until) if args.until
        else datetime.utcnow().date()
    )

    if since_date > until_date:
        logger.error("--since (%s) is after --until (%s).", since_date, until_date)
        sys.exit(1)

    # DB connection
    try:
        pool = await get_db_pool()
        logger.info("DB connected.")
    except Exception as exc:
        logger.error("Cannot connect to DB: %s", exc)
        sys.exit(1)

    # Resolve tickers
    config = await load_advisor_config(pool)
    ticker_arg = args.tickers.strip()

    if ticker_arg.upper() == "ALL" or not ticker_arg:
        raw = config.get("watchlist_bist", "")
        tickers = [t.strip().upper().replace(".IS", "")
                   for t in raw.split(",") if t.strip()]
        if not tickers:
            logger.error(
                "No tickers provided and watchlist_bist is empty in advisor_config. "
                "Either set watchlist_bist in the DB or pass --tickers THYAO,GARAN,..."
            )
            await pool.close()
            sys.exit(1)
    else:
        tickers = [t.strip().upper().replace(".IS", "")
                   for t in ticker_arg.split(",") if t.strip()]

    logger.info("KAP backfill: %d tickers, %s..%s, dry_run=%s",
                len(tickers), since_date, until_date, args.dry_run)

    from modules.advisor.core.kap.kap_archive_crawler import crawl_ticker
    from modules.advisor.core.kap.kap_listener import _pykap_available

    logger.info("PyKap available: %s", _pykap_available())
    if not _pykap_available():
        logger.warning(
            "PyKap not installed. Using direct KAP REST fallback. "
            "Install with: pip install pykap"
        )

    rate_state: dict = {}
    total_stored = 0
    failed_tickers = []

    for i, ticker in enumerate(tickers, 1):
        logger.info("[%d/%d] Crawling %s...", i, len(tickers), ticker)
        try:
            n = await crawl_ticker(
                ticker=ticker,
                since=since_date,
                until=until_date,
                db_pool=pool if not args.dry_run else None,
                rate_state=rate_state,
                ticker_pause_s=args.pause,
                dry_run=args.dry_run,
            )
            total_stored += n
        except KeyboardInterrupt:
            logger.info("Interrupted by operator. Stored %d disclosures so far.", total_stored)
            break
        except Exception as exc:
            logger.warning("Ticker %s failed: %s. Continuing.", ticker, exc)
            failed_tickers.append(ticker)

    logger.info("=" * 60)
    logger.info("KAP backfill complete.")
    logger.info("  Tickers processed : %d", len(tickers) - len(failed_tickers))
    logger.info("  Disclosures stored: %d", total_stored)
    if failed_tickers:
        logger.warning("  Failed tickers    : %s", ", ".join(failed_tickers))
    if args.dry_run:
        logger.info("  (DRY RUN -- nothing written to DB)")
    logger.info("=" * 60)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
