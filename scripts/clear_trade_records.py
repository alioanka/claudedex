#!/usr/bin/env python3
"""
Clear Trade Records Script for ClaudeDex

This script clears paper trade (DRY_RUN) data from the database
before switching to live trading.

WHAT THIS SCRIPT DOES:
1. Lists all trade tables and record counts
2. Optionally clears trades for specific modules
3. Resets performance metrics
4. Optionally resets position counters

USAGE:
    # Inside Docker container:
    docker exec -it trading-bot python scripts/clear_trade_records.py

    # Options:
    python scripts/clear_trade_records.py --list              # List all tables and counts
    python scripts/clear_trade_records.py --module futures    # Clear only futures trades
    python scripts/clear_trade_records.py --module solana     # Clear only solana trades
    python scripts/clear_trade_records.py --module dex        # Clear only DEX trades
    python scripts/clear_trade_records.py --all               # Clear ALL trade records
    python scripts/clear_trade_records.py --dry-run --all     # Preview what would be deleted

Author: ClaudeDex
Date: December 2024
"""

import os
import sys
import argparse
import asyncio
import logging
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

try:
    import asyncpg
    DEPS_AVAILABLE = True
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    DEPS_AVAILABLE = False


# Trade tables configuration per module
TRADE_TABLES = {
    'dex': {
        'trades': 'trades',
        'positions': 'positions',
        'performance': 'performance_metrics',
        'market_data': 'market_data',
    },
    'futures': {
        'trades': 'futures_trades',
        'positions': 'futures_positions',
    },
    'solana': {
        'trades': 'solana_trades',
        'positions': 'solana_positions',
    },
    'arbitrage': {
        'trades': 'arbitrage_trades',
        'positions': 'arbitrage_positions',
        'opportunities': 'arbitrage_opportunities',
    },
    'copytrading': {
        'trades': 'copytrading_trades',
        'positions': 'copytrading_positions',
    },
    'sniper': {
        'trades': 'sniper_trades',
        'positions': 'sniper_positions',
    },
    'ai': {
        'trades': 'ai_trades',
        'positions': 'ai_positions',
        'signals': 'ai_signals',
    },
}

# Additional tables to optionally clear
AUXILIARY_TABLES = [
    'alerts',
    'credential_access_log',
]


class TradeCleaner:
    """Handles clearing trade records from database"""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.db_pool = None

    async def connect_database(self):
        """Connect to PostgreSQL database"""
        db_host = os.getenv('DB_HOST', 'postgres')
        db_port = int(os.getenv('DB_PORT', 5432))
        db_name = os.getenv('DB_NAME', 'tradingbot')  # Match docker-compose POSTGRES_DB

        # Try Docker secret first for db_user, then environment
        db_user = None
        user_secret_file = Path('/run/secrets/db_user')
        if user_secret_file.exists():
            db_user = user_secret_file.read_text().strip()
        else:
            db_user = os.getenv('DB_USER')
            if not db_user:
                raise ValueError(
                    "DB_USER not set. Either:\n"
                    "  1. Run inside trading-bot container (has Docker secrets), or\n"
                    "  2. Set DB_USER environment variable explicitly"
                )

        # Try Docker secret first for db_password, then environment
        db_password = None
        password_secret_file = Path('/run/secrets/db_password')
        if password_secret_file.exists():
            db_password = password_secret_file.read_text().strip()
        else:
            db_password = os.getenv('DB_PASSWORD')
            if not db_password:
                raise ValueError(
                    "DB_PASSWORD not set. Either:\n"
                    "  1. Run inside trading-bot container (has Docker secrets), or\n"
                    "  2. Set DB_PASSWORD environment variable explicitly"
                )

        self.db_pool = await asyncpg.create_pool(
            host=db_host,
            port=db_port,
            database=db_name,
            user=db_user,
            password=db_password,
            min_size=1,
            max_size=5
        )
        logger.info(f"Connected to database at {db_host}:{db_port}/{db_name}")

    async def get_table_count(self, table_name: str) -> int:
        """Get record count for a table"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if table exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = $1
                    )
                """, table_name)

                if not exists:
                    return -1  # Table doesn't exist

                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                return count
        except Exception as e:
            logger.debug(f"Error getting count for {table_name}: {e}")
            return -1

    async def list_all_tables(self) -> dict:
        """List all trade tables and their record counts"""
        results = {}

        for module, tables in TRADE_TABLES.items():
            results[module] = {}
            for table_type, table_name in tables.items():
                count = await self.get_table_count(table_name)
                results[module][table_name] = count

        return results

    async def clear_table(self, table_name: str) -> int:
        """Clear all records from a table"""
        try:
            async with self.db_pool.acquire() as conn:
                # Check if table exists
                exists = await conn.fetchval("""
                    SELECT EXISTS (
                        SELECT FROM information_schema.tables
                        WHERE table_name = $1
                    )
                """, table_name)

                if not exists:
                    return -1

                if self.dry_run:
                    count = await conn.fetchval(f"SELECT COUNT(*) FROM {table_name}")
                    return count

                # Use TRUNCATE for faster deletion (resets auto-increment)
                await conn.execute(f"TRUNCATE TABLE {table_name} CASCADE")

                # If it's a hypertable, we might need different handling
                # But TRUNCATE CASCADE should work for most cases

                return 0
        except Exception as e:
            logger.error(f"Error clearing {table_name}: {e}")
            return -1

    async def clear_module(self, module: str) -> dict:
        """Clear all trade records for a specific module"""
        if module not in TRADE_TABLES:
            return {'error': f"Unknown module: {module}"}

        results = {
            'module': module,
            'tables_cleared': [],
            'tables_failed': [],
            'dry_run': self.dry_run
        }

        tables = TRADE_TABLES[module]
        for table_type, table_name in tables.items():
            count_before = await self.get_table_count(table_name)

            if count_before < 0:
                logger.warning(f"  Table {table_name} does not exist")
                continue

            if self.dry_run:
                logger.info(f"  [DRY RUN] Would clear {table_name}: {count_before} records")
                results['tables_cleared'].append({
                    'table': table_name,
                    'records': count_before
                })
            else:
                result = await self.clear_table(table_name)
                if result >= 0:
                    logger.info(f"  ✅ Cleared {table_name}: {count_before} records deleted")
                    results['tables_cleared'].append({
                        'table': table_name,
                        'records': count_before
                    })
                else:
                    logger.error(f"  ❌ Failed to clear {table_name}")
                    results['tables_failed'].append(table_name)

        return results

    async def clear_all(self, include_auxiliary: bool = False) -> dict:
        """Clear all trade records from all modules"""
        results = {
            'modules_cleared': [],
            'total_records': 0,
            'dry_run': self.dry_run
        }

        for module in TRADE_TABLES.keys():
            logger.info(f"\nClearing {module.upper()} module...")
            module_result = await self.clear_module(module)
            results['modules_cleared'].append(module_result)

            for table in module_result.get('tables_cleared', []):
                results['total_records'] += table.get('records', 0)

        if include_auxiliary:
            logger.info(f"\nClearing auxiliary tables...")
            for table_name in AUXILIARY_TABLES:
                count = await self.get_table_count(table_name)
                if count < 0:
                    continue

                if self.dry_run:
                    logger.info(f"  [DRY RUN] Would clear {table_name}: {count} records")
                else:
                    await self.clear_table(table_name)
                    logger.info(f"  ✅ Cleared {table_name}: {count} records")

                results['total_records'] += count

        return results

    async def reset_sequences(self):
        """Reset auto-increment sequences for all tables"""
        async with self.db_pool.acquire() as conn:
            # Get all sequences
            sequences = await conn.fetch("""
                SELECT sequence_name
                FROM information_schema.sequences
                WHERE sequence_schema = 'public'
            """)

            for seq in sequences:
                seq_name = seq['sequence_name']
                if self.dry_run:
                    logger.info(f"  [DRY RUN] Would reset sequence: {seq_name}")
                else:
                    await conn.execute(f"ALTER SEQUENCE {seq_name} RESTART WITH 1")
                    logger.debug(f"  Reset sequence: {seq_name}")


async def main():
    parser = argparse.ArgumentParser(
        description='Clear paper trade records from database',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/clear_trade_records.py --list              # Show all tables and counts
  python scripts/clear_trade_records.py --module futures    # Clear futures module only
  python scripts/clear_trade_records.py --all               # Clear ALL modules
  python scripts/clear_trade_records.py --dry-run --all     # Preview deletion

MODULES:
  dex, futures, solana, arbitrage, copytrading, sniper, ai

IMPORTANT:
  - Stop all modules before clearing data
  - This action is IRREVERSIBLE
  - Create a database backup before clearing production data
        """
    )

    parser.add_argument('--list', action='store_true',
                        help='List all tables and record counts')
    parser.add_argument('--module', type=str,
                        help='Clear specific module (dex, futures, solana, arbitrage, copytrading, sniper, ai)')
    parser.add_argument('--all', action='store_true',
                        help='Clear ALL trade records from all modules')
    parser.add_argument('--include-auxiliary', action='store_true',
                        help='Also clear auxiliary tables (alerts, logs)')
    parser.add_argument('--reset-sequences', action='store_true',
                        help='Reset auto-increment sequences after clearing')
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without making them')

    args = parser.parse_args()

    if not DEPS_AVAILABLE:
        logger.error("Required dependencies not available")
        sys.exit(1)

    cleaner = TradeCleaner(dry_run=args.dry_run)
    await cleaner.connect_database()

    try:
        if args.list:
            logger.info("\n" + "=" * 60)
            logger.info("TRADE TABLES SUMMARY")
            logger.info("=" * 60)

            tables = await cleaner.list_all_tables()
            total = 0

            for module, module_tables in tables.items():
                logger.info(f"\n📦 {module.upper()}")
                for table_name, count in module_tables.items():
                    if count < 0:
                        status = "❌ (not found)"
                    elif count == 0:
                        status = "✅ (empty)"
                    else:
                        status = f"📊 {count} records"
                        total += count
                    logger.info(f"   {table_name}: {status}")

            logger.info(f"\n{'=' * 60}")
            logger.info(f"TOTAL RECORDS: {total}")
            logger.info("=" * 60)

        elif args.module:
            if args.module not in TRADE_TABLES:
                logger.error(f"Unknown module: {args.module}")
                logger.info(f"Available modules: {', '.join(TRADE_TABLES.keys())}")
                sys.exit(1)

            if not args.dry_run:
                print(f"\n⚠️  This will DELETE all {args.module.upper()} trade records!")
                confirm = input(f"Type '{args.module}' to confirm: ")
                if confirm != args.module:
                    print("Cancelled")
                    sys.exit(0)

            logger.info(f"\nClearing {args.module.upper()} module...")
            result = await cleaner.clear_module(args.module)

            if args.reset_sequences and not args.dry_run:
                await cleaner.reset_sequences()

            total = sum(t.get('records', 0) for t in result.get('tables_cleared', []))
            logger.info(f"\n{'=' * 60}")
            logger.info(f"CLEARED: {total} records from {args.module.upper()}")
            logger.info("=" * 60)

        elif args.all:
            if not args.dry_run:
                print("\n" + "=" * 60)
                print("⚠️  WARNING: CLEAR ALL TRADE RECORDS")
                print("=" * 60)
                print("\nThis will DELETE:")
                print("  - All trade history")
                print("  - All position data")
                print("  - All performance metrics")
                print("\nThis action is IRREVERSIBLE!")
                print("\n")
                confirm = input("Type 'DELETE ALL TRADES' to confirm: ")
                if confirm != 'DELETE ALL TRADES':
                    print("Cancelled")
                    sys.exit(0)

            result = await cleaner.clear_all(include_auxiliary=args.include_auxiliary)

            if args.reset_sequences and not args.dry_run:
                logger.info("\nResetting sequences...")
                await cleaner.reset_sequences()

            logger.info(f"\n{'=' * 60}")
            logger.info(f"TOTAL CLEARED: {result['total_records']} records")
            logger.info("=" * 60)

            if args.dry_run:
                logger.info("\n[DRY RUN] No changes were made")

        else:
            parser.print_help()

    finally:
        if cleaner.db_pool:
            await cleaner.db_pool.close()


if __name__ == '__main__':
    asyncio.run(main())
