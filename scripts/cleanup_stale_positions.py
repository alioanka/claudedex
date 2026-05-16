"""Operator utility: close stale-open positions in DRY_RUN data.

After the May 2026 hardening session, a few thousand DRY_RUN positions
may have been orphaned with `status='open'` even though the engine
already retired them in-memory (synthetic-close exited the loop before
the close-row write completed, or a crash dropped them mid-update).

This script identifies and bulk-marks them as closed in the sniper_trades
table so the dashboard's active-positions count matches the engine's
in-memory state. Idempotent: rerunning a second time is a no-op.

Usage:
    python scripts/cleanup_stale_positions.py            # dry-run (prints count)
    python scripts/cleanup_stale_positions.py --apply    # actually executes UPDATE

Connects via DATABASE_URL or Docker-secret-style postgres on
the trading-postgres container; falls back to interactive prompt.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path


async def main(apply: bool, stale_minutes: int) -> int:
    try:
        import asyncpg
    except ImportError:
        print("ERROR: asyncpg not installed. Run inside the trading-bot container.")
        return 1

    db_url = os.getenv('DATABASE_URL')
    if not db_url:
        # Try the standard Docker-secret path used inside the bot container
        try:
            user = Path('/run/secrets/db_user').read_text().strip()
            pw = Path('/run/secrets/db_password').read_text().strip()
            db_url = f"postgresql://{user}:{pw}@postgres:5432/tradingbot"
        except Exception:
            print("ERROR: no DATABASE_URL env and no Docker secrets readable.")
            return 1

    conn = await asyncpg.connect(db_url)
    try:
        # Count first so we know what we're about to touch
        count = await conn.fetchval(
            f"""
            SELECT COUNT(*) FROM sniper_trades
            WHERE status = 'open'
              AND entry_timestamp < NOW() - INTERVAL '{stale_minutes} minutes'
            """
        )
        print(f"Stale-open sniper_trades (>{stale_minutes} min old): {count}")

        if not apply:
            print("\nDry-run: no rows touched. Add --apply to execute.")
            return 0

        if count == 0:
            print("Nothing to do.")
            return 0

        result = await conn.execute(
            f"""
            UPDATE sniper_trades
            SET status = 'closed',
                exit_timestamp = NOW(),
                exit_price = entry_price,
                exit_reason = 'stale_cleanup',
                profit_loss = COALESCE(profit_loss, 0)
            WHERE status = 'open'
              AND entry_timestamp < NOW() - INTERVAL '{stale_minutes} minutes'
            """
        )
        print(f"UPDATE result: {result}")
        return 0
    finally:
        await conn.close()


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--apply', action='store_true',
                    help='Actually execute the UPDATE (without this, dry-run)')
    ap.add_argument('--stale-minutes', type=int, default=60,
                    help='Rows with entry_timestamp older than this and status=open are considered stale (default 60)')
    args = ap.parse_args()
    sys.exit(asyncio.run(main(args.apply, args.stale_minutes)))
