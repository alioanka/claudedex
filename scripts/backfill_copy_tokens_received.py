#!/usr/bin/env python3
"""
Backfill metadata.tokens_received for OPEN copytrading_trades positions
that were INSERTed before commit 14ce4a0. Without this field the
dashboard cannot compute live PnL (we don't know how many tokens were
actually bought — only the SOL amount spent).

Usage (inside the trading-bot container):
    docker exec trading-bot python scripts/backfill_copy_tokens_received.py
    docker exec trading-bot python scripts/backfill_copy_tokens_received.py --dry-run
    docker exec trading-bot python scripts/backfill_copy_tokens_received.py --trade-id <id>

Resolution strategy per row:
  1. Skip if metadata.tokens_received is already populated.
  2. Try Jupiter Price v3 for the current USD price of token_address.
     Compute tokens_received_estimate = entry_usd / current_price.
     This is an APPROXIMATION — assumes price hasn't moved much since
     entry. For positions held for hours this is fine within ~20%; for
     long-held meme positions it can be wildly off. The dashboard will
     show 'approximate' as a hint.
  3. If Jupiter returns no price, log and leave the field absent.

This is safe to run repeatedly — only INSERTs the field on rows where
it's missing.
"""
import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Make project root importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import asyncpg
    import aiohttp
except ImportError as e:
    print(f"Missing dependency: {e}. Run inside the trading-bot container.", file=sys.stderr)
    sys.exit(1)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("backfill_tokens_received")

JUPITER_PRICE_V3 = "https://api.jup.ag/price/v3"
HTTP_TIMEOUT_S = 8


async def fetch_token_price_usd(session: aiohttp.ClientSession, mint: str) -> float | None:
    """Return USD price for `mint` via Jupiter Price v3, or None on
    any failure. Handles both v3 and legacy v2 response shapes."""
    url = f"{JUPITER_PRICE_V3}?ids={mint}"
    try:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=HTTP_TIMEOUT_S)) as resp:
            if resp.status != 200:
                logger.warning(f"  Jupiter {mint[:10]} returned HTTP {resp.status}")
                return None
            data = await resp.json()
    except Exception as e:
        logger.warning(f"  Jupiter fetch failed for {mint[:10]}: {e}")
        return None
    payload = data.get("data") if isinstance(data, dict) and "data" in data else data
    if not isinstance(payload, dict):
        return None
    row = payload.get(mint)
    if not isinstance(row, dict):
        return None
    raw = row.get("usdPrice") or row.get("price") or row.get("usd") or 0
    try:
        price = float(raw)
        return price if price > 0 else None
    except (TypeError, ValueError):
        return None


async def get_db_url() -> str:
    """Resolve DATABASE_URL the same way the rest of the bot does:
    Docker secrets first, env var second."""
    try:
        from security.docker_secrets import get_database_url
        url = get_database_url()
        if url:
            return url
    except ImportError:
        pass
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError(
            "Could not resolve DATABASE_URL. Set the env var or run inside the trading-bot container."
        )
    return url


async def main(args):
    db_url = await get_db_url()
    pool = await asyncpg.create_pool(db_url, min_size=1, max_size=2)

    try:
        # Pull every OPEN copytrading position that's missing tokens_received.
        async with pool.acquire() as conn:
            where = (
                "status = 'open' "
                "AND chain = 'solana' "
                "AND (metadata IS NULL "
                "     OR NOT (metadata::jsonb ? 'tokens_received'))"
            )
            params = ()
            if args.trade_id:
                where += " AND trade_id = $1"
                params = (args.trade_id,)
            rows = await conn.fetch(
                f"SELECT trade_id, token_address, entry_usd, metadata "
                f"FROM copytrading_trades WHERE {where} "
                f"ORDER BY entry_timestamp ASC",
                *params,
            )

        if not rows:
            logger.info("No backfill candidates — every OPEN Solana position already has metadata.tokens_received.")
            return 0

        logger.info(f"Found {len(rows)} OPEN Solana position(s) needing backfill.")
        if args.dry_run:
            logger.info("--dry-run set: not writing to DB.")

        async with aiohttp.ClientSession() as session:
            for r in rows:
                trade_id = r["trade_id"]
                mint = r["token_address"]
                entry_usd = float(r["entry_usd"] or 0)
                if not mint or entry_usd <= 0:
                    logger.warning(f"  skip {trade_id}: missing mint or entry_usd")
                    continue

                price = await fetch_token_price_usd(session, mint)
                if not price:
                    logger.warning(f"  skip {trade_id}: Jupiter returned no price for {mint[:10]}")
                    continue

                tokens_received = entry_usd / price
                logger.info(
                    f"  {trade_id} {mint[:10]}: "
                    f"${entry_usd:.4f} ÷ ${price:.6f}/tok = {tokens_received:,.6f} tokens"
                )

                if args.dry_run:
                    continue

                # Merge into existing metadata JSONB without clobbering keys.
                meta = r["metadata"]
                if isinstance(meta, str):
                    try:
                        meta = json.loads(meta)
                    except Exception:
                        meta = {}
                if not isinstance(meta, dict):
                    meta = {}
                meta["tokens_received"] = tokens_received
                meta["tokens_received_source"] = "backfill_jupiter_v3"
                meta["tokens_received_approx"] = True  # warn dashboard this is post-hoc

                async with pool.acquire() as conn:
                    await conn.execute(
                        "UPDATE copytrading_trades "
                        "SET metadata = $1::jsonb "
                        "WHERE trade_id = $2",
                        json.dumps(meta),
                        trade_id,
                    )
        logger.info("Backfill complete.")
        return 0
    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be written without modifying the DB.")
    parser.add_argument("--trade-id", type=str, default=None,
                        help="Process only this specific trade_id.")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)))
