#!/usr/bin/env python3
"""
W6 commit 3/5 — operator diagnostic for the "ARB engine silent" symptom.

Symptom this answers:
  - /api/arbitrage/diagnostics shows stale: true
  - last_trades months old
  - near_miss_counters returns 0 rows

Usage:
    docker exec trading-bot python scripts/arb_engine_health.py
    docker exec trading-bot python scripts/arb_engine_health.py --tail 50

Checks (read-only): arbitrage_runtime_stats freshness, last arbitrage_trades
row, kill/pause/restart flag files, tail of arbitrage_errors.log. Suggests
the next operator action — restart via touch logs/.restart_arbitrage (W6
flag-file pattern picked up by main.py within 5s) if the engine is dead.
"""
import argparse
import asyncio
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

try:
    import asyncpg
except ImportError as e:
    print(f"Missing dependency: {e}. Run inside the trading-bot container.", file=sys.stderr)
    sys.exit(1)

STALE_AGE_S = 600  # matches /api/arbitrage/diagnostics' 10-min stale gate
RESTART_FLAG = Path("logs/.restart_arbitrage")
KILLSWITCH_FLAG = Path("logs/.killswitch")
PAUSE_FLAG = Path("logs/.pause_arbitrage")
ERROR_LOG = Path("logs/arbitrage/arbitrage_errors.log")


async def _get_db_url() -> str:
    try:
        from security.docker_secrets import get_database_url
        url = get_database_url()
        if url:
            return url
    except ImportError:
        pass
    url = os.getenv("DATABASE_URL")
    if not url:
        raise RuntimeError("Could not resolve DATABASE_URL. Set env var or run in container.")
    return url


async def _check_runtime_stats(conn) -> tuple[bool, list[str]]:
    lines: list[str] = []
    try:
        rows = await conn.fetch(
            "SELECT chain, updated_at, "
            "EXTRACT(EPOCH FROM (NOW() - updated_at)) AS age_s "
            "FROM arbitrage_runtime_stats ORDER BY updated_at DESC"
        )
    except Exception as e:
        lines.append(f"  arbitrage_runtime_stats query failed: {e}")
        return False, lines
    if not rows:
        lines.append(
            "  arbitrage_runtime_stats has 0 rows — engine never persisted. "
            "Subprocess dead or crashed before its first 5-min snapshot."
        )
        return False, lines
    any_fresh = False
    for r in rows:
        age = float(r["age_s"] or 0)
        fresh = age < STALE_AGE_S
        any_fresh = any_fresh or fresh
        lines.append(
            f"  chain={r['chain']:<10} updated_at={r['updated_at']} "
            f"age={age:.0f}s {'FRESH' if fresh else 'STALE'}"
        )
    return any_fresh, lines


async def _check_last_trade(conn) -> list[str]:
    try:
        row = await conn.fetchrow(
            "SELECT chain, entry_timestamp, "
            "EXTRACT(EPOCH FROM (NOW() - entry_timestamp)) AS age_s "
            "FROM arbitrage_trades ORDER BY entry_timestamp DESC LIMIT 1"
        )
    except Exception as e:
        return [f"  arbitrage_trades query failed: {e}"]
    if not row:
        return ["  arbitrage_trades empty — engine has never fired a trade."]
    age = float(row["age_s"] or 0)
    return [
        f"  last fired chain={row['chain']} at {row['entry_timestamp']} "
        f"({age/86400:.1f} days ago)"
    ]


def _check_flags() -> list[str]:
    out: list[str] = []
    for flag, label in (
        (KILLSWITCH_FLAG, "global kill switch (engine refuses live trades)"),
        (PAUSE_FLAG, "module paused via dashboard"),
        (RESTART_FLAG, "orchestrator will restart on next 5s poll"),
    ):
        out.append(f"  {flag} {'EXISTS — ' + label if flag.exists() else 'absent'}")
    return out


def _tail_error_log(n_lines: int) -> list[str]:
    if not ERROR_LOG.exists():
        return [f"  {ERROR_LOG} does not exist (no error log yet, or wrong cwd)"]
    try:
        with open(ERROR_LOG, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-n_lines:]
    except Exception as e:
        return [f"  reading {ERROR_LOG} failed: {e}"]
    if not lines:
        return [f"  {ERROR_LOG} is empty"]
    return [f"  --- last {len(lines)} lines of {ERROR_LOG} ---"] + [
        "  " + ln.rstrip() for ln in lines
    ]


async def main(args):
    pool = await asyncpg.create_pool(await _get_db_url(), min_size=1, max_size=1)
    try:
        async with pool.acquire() as conn:
            print("[1] arbitrage_runtime_stats (per-chain liveness):")
            any_fresh, rs_lines = await _check_runtime_stats(conn)
            for ln in rs_lines:
                print(ln)
            print("\n[2] arbitrage_trades most-recent row:")
            for ln in await _check_last_trade(conn):
                print(ln)
        print("\n[3] kill-switch / pause / restart flags:")
        for ln in _check_flags():
            print(ln)
        print(f"\n[4] tail of {ERROR_LOG}:")
        for ln in _tail_error_log(args.tail):
            print(ln)
        print("\nVERDICT:")
        if any_fresh:
            print(
                "  Engine APPEARS ALIVE — at least one chain persisted runtime "
                f"stats within {STALE_AGE_S}s. If no trades are firing, check "
                "near_miss_counters in /api/arbitrage/diagnostics for gate reasons."
            )
        else:
            print(
                "  Engine appears DEAD or stuck (no runtime_stats row younger "
                f"than {STALE_AGE_S}s). Next operator actions:"
            )
            print("    1) grep -i arbitrage logs/orchestrator.log | tail")
            print("    2) confirm ARBITRAGE_MODULE_ENABLED=true in .env")
            print("    3) request a restart via the W6 flag-file pattern:")
            print("       touch logs/.restart_arbitrage")
            print("       (main.py polls every 5s and calls module.restart())")
    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tail", type=int, default=20,
                        help="Lines of arbitrage_errors.log to print (default 20)")
    args = parser.parse_args()
    sys.exit(asyncio.run(main(args)) or 0)
