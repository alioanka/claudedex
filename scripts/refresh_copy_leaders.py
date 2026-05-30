#!/usr/bin/env python3
"""
Refresh the copy_leader_scores table by running the wallet-discovery
sweep across configured public sources, then scoring every discovered
candidate against our copytrading_trades history.

Usage:
    python scripts/refresh_copy_leaders.py [--chain solana] [--mock]

Operator notes:
  * Safe to run from cron — the discovery layer is rate-limited and
    fully bounded per source. A run with no third-party keys
    configured falls through to the on-chain fallback and finishes
    in < 1 second.
  * Pass --mock to test the upsert + DB plumbing without hitting any
    network endpoint.
  * Honors the standard DATABASE_URL / Docker-secret path used by
    every other module entrypoint.

This script is the operator-facing entrypoint for the wallet
discovery rebuild (Wave-2 quant rebuild step 2/3). The dashboard
button at /copytrading/leaders -> "Refresh Discovery" calls the
same code path; both write to copy_leader_scores (migration 023).
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for module imports.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("refresh_copy_leaders")


async def _open_db_pool():
    """Open an asyncpg pool using the same resolver every module
    entrypoint uses (Docker secrets first, env fallback).

    Returns None on any failure (missing URL, asyncpg not installed,
    connect error). The discovery layer is DB-optional in mock mode
    so the script still runs end-to-end for plumbing checks.
    """
    try:
        from security.docker_secrets import get_database_url
        db_url = get_database_url()
    except Exception:
        db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.warning("No DATABASE_URL configured; running without DB")
        return None
    try:
        import asyncpg
    except ImportError:
        logger.warning("asyncpg not installed; running without DB")
        return None
    try:
        return await asyncpg.create_pool(db_url)
    except Exception as e:
        logger.warning(f"asyncpg.create_pool failed ({e}); running without DB")
        return None


async def _resolve_keys():
    """Pull Helius / Birdeye keys from secrets manager / env."""
    helius_key = birdeye_key = None
    try:
        from security.secrets_manager import secrets
        helius_key = secrets.get("HELIUS_API_KEY", log_access=False)
        birdeye_key = secrets.get("BIRDEYE_API_KEY", log_access=False)
    except Exception:
        pass
    if not helius_key:
        helius_key = os.getenv("HELIUS_API_KEY")
    if not birdeye_key:
        birdeye_key = os.getenv("BIRDEYE_API_KEY")
    return helius_key, birdeye_key


async def main(chains, mock: bool):
    from modules.copy_trading.wallet_discovery import (
        DiscoveryConfig, discover_and_score,
    )

    db_pool = await _open_db_pool()
    if db_pool is None and not mock:
        # Without DB we have no on-chain fallback, but mock mode still works.
        logger.warning(
            "No DB pool; running without persistence (results will print only)"
        )

    helius_key, birdeye_key = await _resolve_keys()
    if not mock and not helius_key and not birdeye_key:
        logger.warning(
            "Neither HELIUS_API_KEY nor BIRDEYE_API_KEY configured — sweep "
            "will only use the on-chain fallback."
        )

    cfg = DiscoveryConfig(
        chains=tuple(chains),
        helius_api_key=helius_key,
        birdeye_api_key=birdeye_key,
        mock=mock,
    )
    logger.info(
        f"Starting discovery sweep: chains={list(chains)} mock={mock} "
        f"helius={'set' if helius_key else 'none'} "
        f"birdeye={'set' if birdeye_key else 'none'}"
    )

    scored = await discover_and_score(db_pool, cfg)
    if not scored:
        logger.warning("Discovery sweep produced 0 candidates")
        return 0

    logger.info(f"Scored {len(scored)} leaders. Top 10 by composite score:")
    for i, m in enumerate(scored[:10], 1):
        logger.info(
            f"  #{i:2d} {m.chain:8s} {m.wallet_address[:14]}... "
            f"score={m.score} kelly={m.kelly_fraction} "
            f"pnl30d={m.realized_pnl_usd_30d:.2f} trades={m.trade_count_30d}"
        )
    if db_pool is not None:
        await db_pool.close()
    return 0


def _parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    p.add_argument(
        "--chain", action="append", dest="chains", default=None,
        help="Chain to discover (repeatable). Default: solana ethereum base.",
    )
    p.add_argument(
        "--mock", action="store_true",
        help="Use deterministic mock candidates; do not hit the network.",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    chains = args.chains or ["solana", "ethereum", "base"]
    rc = asyncio.run(main(chains, args.mock))
    sys.exit(int(rc or 0))
