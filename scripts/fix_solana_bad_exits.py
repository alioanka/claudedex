#!/usr/bin/env python3
"""
Fix corrupt Solana exit prices / blown-up PnL rows.

THE BUG THIS REPAIRS
--------------------
The Solana Trades page can show a corrupt CLOSED row whose exit price is a
wildly wrong denomination (a SOL/USD or wrong-token price-unit mixup leaking
into position.current_price and being written verbatim as exit_price). The
canonical example is PYTH (a ~$0.03 token): entry $0.0352, exit $153.52
(~4360x, which is roughly the SOL price in USD), P&L +745.52 SOL / +2000.00%
where +2000% is the pnl clamp CEILING firing. Such a single bad row poisons
win rate and total-PnL analytics.

The persist-side guard in solana_engine._save_trade_to_db now PREVENTS new
poisoned rows (it pins an implausible exit_price and recomputes PnL). This
script repairs rows that were ALREADY written before that guard landed.

WHAT IT DOES
------------
Scans solana_trades for rows where EITHER:
  * exit_price / entry_price ratio is implausible (> --max-ratio or its
    inverse), OR
  * |pnl_pct| has hit the +2000% / -100% clamp boundary (a classic
    "blown-up move tripped the clamp" fingerprint).

For each match it (in --apply mode):
  * pins exit_price to the nearest plausible ratio bound,
  * recomputes pnl_pct from the pinned exit (long-only on Solana),
  * recomputes pnl_sol / pnl_usd from the SOL notional (amount_sol),
  * tags metadata {"excluded": true, "exclude_reason": "implausible_exit_price",
    ...originals...} so dashboards/stats can DROP the row,
WITHOUT touching any other column. With --exclude-only it ONLY tags metadata
and leaves the numeric columns intact (use this if you prefer to filter rather
than rewrite).

DRY-RUN BY DEFAULT. Nothing is written unless --apply is passed.

USAGE
-----
    # Preview (default, no writes):
    docker exec trading-bot python scripts/fix_solana_bad_exits.py

    # Only the PYTH row(s):
    docker exec trading-bot python scripts/fix_solana_bad_exits.py --symbol PYTH

    # Apply the corrections:
    docker exec trading-bot python scripts/fix_solana_bad_exits.py --apply

    # Just tag rows excluded (don't rewrite numbers):
    docker exec trading-bot python scripts/fix_solana_bad_exits.py --apply --exclude-only

    # Tighten/loosen the implausibility threshold (default 50x):
    docker exec trading-bot python scripts/fix_solana_bad_exits.py --max-ratio 25 --apply

This script is read-only against trading logic; it touches ONLY the
solana_trades analytics table.
"""

import os
import sys
import json
import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

try:
    import asyncpg
    DEPS_AVAILABLE = True
except ImportError as e:  # pragma: no cover
    logger.error(f"Missing dependencies: {e}")
    DEPS_AVAILABLE = False

# These mirror the persist-side guard / clamp in solana_engine._save_trade_to_db
PNL_PCT_CLAMP_HIGH = 2000.0
PNL_PCT_CLAMP_LOW = -100.0
CLAMP_EPS = 0.01  # tolerance for "sitting on the clamp boundary"


async def connect_database():
    """Connect to PostgreSQL using the same idiom as clear_trade_records.py."""
    db_host = os.getenv('DB_HOST', 'postgres')
    db_port = int(os.getenv('DB_PORT', 5432))
    db_name = os.getenv('DB_NAME', 'tradingbot')

    user_secret = Path('/run/secrets/db_user')
    db_user = user_secret.read_text().strip() if user_secret.exists() else os.getenv('DB_USER')
    if not db_user:
        raise ValueError(
            "DB_USER not set. Run inside the trading-bot container (Docker secrets) "
            "or export DB_USER explicitly."
        )

    pw_secret = Path('/run/secrets/db_password')
    db_password = pw_secret.read_text().strip() if pw_secret.exists() else os.getenv('DB_PASSWORD')
    if not db_password:
        raise ValueError(
            "DB_PASSWORD not set. Run inside the trading-bot container (Docker secrets) "
            "or export DB_PASSWORD explicitly."
        )

    pool = await asyncpg.create_pool(
        host=db_host, port=db_port, database=db_name,
        user=db_user, password=db_password, min_size=1, max_size=3,
    )
    logger.info(f"Connected to database at {db_host}:{db_port}/{db_name}")
    return pool


def _is_clamp_boundary(pnl_pct) -> bool:
    if pnl_pct is None:
        return False
    p = float(pnl_pct)
    return (abs(p - PNL_PCT_CLAMP_HIGH) <= CLAMP_EPS
            or abs(p - PNL_PCT_CLAMP_LOW) <= CLAMP_EPS)


def _ratio(entry, exit_) -> float:
    try:
        e = float(entry or 0)
        x = float(exit_ or 0)
        if e > 0 and x > 0:
            return x / e
    except Exception:
        pass
    return 0.0


def _already_excluded(metadata) -> bool:
    if not metadata:
        return False
    try:
        md = metadata if isinstance(metadata, dict) else json.loads(metadata)
        return bool(md.get('excluded'))
    except Exception:
        return False


async def find_bad_rows(pool, max_ratio: float, symbol: str = None):
    """Return rows that look corrupt by ratio or by clamp-boundary pnl_pct."""
    where = []
    params = []
    if symbol:
        params.append(symbol)
        where.append(f"token_symbol = ${len(params)}")
    where_sql = (" WHERE " + " AND ".join(where)) if where else ""

    sql = f"""
        SELECT id, trade_id, token_symbol, token_mint, strategy,
               entry_price, exit_price, amount_sol, pnl_sol, pnl_usd,
               pnl_pct, sol_price_usd, is_simulated, exit_time, metadata
        FROM solana_trades{where_sql}
        ORDER BY exit_time DESC
    """
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql, *params)

    bad = []
    for r in rows:
        if _already_excluded(r['metadata']):
            continue
        ratio = _ratio(r['entry_price'], r['exit_price'])
        ratio_bad = ratio > 0 and (ratio > max_ratio or ratio < (1.0 / max_ratio))
        clamp_bad = _is_clamp_boundary(r['pnl_pct'])
        if ratio_bad or clamp_bad:
            reason = []
            if ratio_bad:
                reason.append(f"ratio={ratio:.1f}x")
            if clamp_bad:
                reason.append(f"pnl_pct@clamp={float(r['pnl_pct']):.2f}%")
            bad.append((r, ratio, "; ".join(reason)))
    return bad


def _corrected_values(row, max_ratio: float):
    """Compute pinned exit + consistent pnl for a bad row.

    Returns (new_exit, new_pnl_pct, new_pnl_sol, new_pnl_usd) or None if the
    row has no usable entry_price to anchor against (can't repair -> caller
    should exclude-only)."""
    entry = float(row['entry_price'] or 0)
    if entry <= 0:
        return None
    exit_ = float(row['exit_price'] or 0)
    ratio = _ratio(entry, exit_)

    if ratio > max_ratio:
        new_exit = entry * max_ratio
    elif 0 < ratio < (1.0 / max_ratio):
        new_exit = entry / max_ratio
    else:
        # No bad ratio (matched only on clamp). Re-derive exit from the clamped
        # pct so the row is at least self-consistent at the clamp bound.
        pnl_pct = float(row['pnl_pct'] or 0)
        pnl_pct = max(PNL_PCT_CLAMP_LOW, min(PNL_PCT_CLAMP_HIGH, pnl_pct))
        new_exit = entry * (1.0 + pnl_pct / 100.0)

    new_pnl_pct = ((new_exit - entry) / entry) * 100.0
    new_pnl_pct = max(PNL_PCT_CLAMP_LOW, min(PNL_PCT_CLAMP_HIGH, new_pnl_pct))
    notional = float(row['amount_sol'] or 0)
    new_pnl_sol = notional * (new_pnl_pct / 100.0)
    sol_px = float(row['sol_price_usd'] or 0)
    new_pnl_usd = new_pnl_sol * sol_px
    return new_exit, new_pnl_pct, new_pnl_sol, new_pnl_usd


def _build_metadata(row, ratio, max_ratio, recomputed: bool):
    existing = {}
    if row['metadata']:
        try:
            existing = row['metadata'] if isinstance(row['metadata'], dict) else json.loads(row['metadata'])
        except Exception:
            existing = {}
    existing.update({
        'excluded': True,
        'exclude_reason': 'implausible_exit_price',
        'original_exit_price': float(row['exit_price'] or 0),
        'original_pnl_sol': float(row['pnl_sol'] or 0),
        'original_pnl_pct': float(row['pnl_pct'] or 0),
        'exit_entry_ratio': ratio,
        'guard_max_ratio': max_ratio,
        'recomputed': recomputed,
        'fixed_by': 'scripts/fix_solana_bad_exits.py',
        'fixed_at': datetime.now(timezone.utc).isoformat(),
    })
    return json.dumps(existing)


async def main():
    parser = argparse.ArgumentParser(
        description='Find and repair corrupt Solana exit-price / PnL rows.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--apply', action='store_true',
                        help='Write changes. Without this flag the script is a DRY RUN.')
    parser.add_argument('--exclude-only', action='store_true',
                        help='Only tag metadata excluded; do NOT rewrite numeric columns.')
    parser.add_argument('--max-ratio', type=float, default=50.0,
                        help='exit/entry ratio considered implausible (default 50).')
    parser.add_argument('--symbol', type=str, default=None,
                        help='Restrict to one token symbol (e.g. PYTH).')
    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        logger.error("Required dependencies not available")
        sys.exit(1)

    max_ratio = max(2.0, float(args.max_ratio))
    mode = "APPLY" if args.apply else "DRY RUN"
    logger.info("=" * 70)
    logger.info(f"Solana bad-exit fixer — mode: {mode} | max_ratio: {max_ratio:.0f}x | "
                f"exclude_only: {args.exclude_only}")
    logger.info("=" * 70)

    pool = await connect_database()
    try:
        bad = await find_bad_rows(pool, max_ratio, args.symbol)
        if not bad:
            logger.info("✅ No corrupt rows found. Nothing to do.")
            return

        logger.info(f"Found {len(bad)} suspect row(s):\n")
        changed = 0
        for row, ratio, why in bad:
            corrected = None if args.exclude_only else _corrected_values(row, max_ratio)
            logger.info(
                f"  • {row['token_symbol']} [{row['strategy']}] trade_id={row['trade_id']} "
                f"sim={row['is_simulated']}"
            )
            logger.info(
                f"      reason: {why}"
            )
            logger.info(
                f"      entry=${float(row['entry_price'] or 0):.8f}  "
                f"exit=${float(row['exit_price'] or 0):.6f}  "
                f"pnl={float(row['pnl_sol'] or 0):.4f} SOL ({float(row['pnl_pct'] or 0):+.2f}%)"
            )
            if corrected:
                ne, npct, nsol, nusd = corrected
                logger.info(
                    f"      ->  exit=${ne:.8f}  pnl={nsol:.4f} SOL ({npct:+.2f}%, ${nusd:.2f})  + tag excluded"
                )
            else:
                if args.exclude_only:
                    logger.info("      ->  tag excluded ONLY (numeric columns untouched)")
                else:
                    logger.info("      ->  cannot recompute (entry_price<=0); will tag excluded ONLY")

            if args.apply:
                metadata_json = _build_metadata(
                    row, ratio, max_ratio, recomputed=bool(corrected))
                async with pool.acquire() as conn:
                    if corrected:
                        ne, npct, nsol, nusd = corrected
                        await conn.execute(
                            """
                            UPDATE solana_trades
                               SET exit_price = $1, pnl_pct = $2, pnl_sol = $3,
                                   pnl_usd = $4, metadata = $5::jsonb
                             WHERE id = $6
                            """,
                            ne, npct, nsol, nusd, metadata_json, row['id'],
                        )
                    else:
                        await conn.execute(
                            "UPDATE solana_trades SET metadata = $1::jsonb WHERE id = $2",
                            metadata_json, row['id'],
                        )
                changed += 1

        logger.info("")
        logger.info("=" * 70)
        if args.apply:
            logger.info(f"✅ Updated {changed} row(s).")
        else:
            logger.info(f"DRY RUN: would update {len(bad)} row(s). Re-run with --apply to write.")
        logger.info("=" * 70)
    finally:
        await pool.close()


if __name__ == '__main__':
    asyncio.run(main())
