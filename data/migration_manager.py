"""
Database Migration System
Automatically runs migrations on database connection
"""
import logging
import asyncpg
from pathlib import Path
from typing import List, Optional
import hashlib

logger = logging.getLogger(__name__)


class MigrationManager:
    """
    Manages database schema migrations
    Tracks applied migrations and runs new ones automatically
    """

    def __init__(self, db_pool: asyncpg.Pool, migrations_dir: str = "migrations"):
        self.pool = db_pool
        self.migrations_dir = Path(migrations_dir)

    async def initialize(self):
        """Initialize migration tracking table"""
        async with self.pool.acquire() as conn:
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id SERIAL PRIMARY KEY,
                    migration_name VARCHAR(255) UNIQUE NOT NULL,
                    checksum VARCHAR(64) NOT NULL,
                    applied_at TIMESTAMPTZ DEFAULT NOW(),
                    success BOOLEAN DEFAULT TRUE
                );

                CREATE INDEX IF NOT EXISTS idx_migrations_name ON schema_migrations(migration_name);
            """)

        logger.info("Migration tracking table initialized")

    async def get_applied_migrations(self) -> List[str]:
        """Get list of already applied migrations"""
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT migration_name FROM schema_migrations
                WHERE success = TRUE
                ORDER BY id
            """)

            return [row['migration_name'] for row in rows]

    async def get_pending_migrations(self) -> List[Path]:
        """Get list of migrations that need to be applied"""
        if not self.migrations_dir.exists():
            logger.warning(f"Migrations directory not found: {self.migrations_dir}")
            return []

        # Get all SQL files
        all_migrations = sorted(self.migrations_dir.glob("*.sql"))

        # Get already applied migrations
        applied = await self.get_applied_migrations()

        # Filter to pending only
        pending = []
        for migration_file in all_migrations:
            if migration_file.name not in applied:
                pending.append(migration_file)

        return pending

    def _calculate_checksum(self, content: str) -> str:
        """Calculate SHA256 checksum of migration content"""
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    async def apply_migration(self, migration_file: Path) -> bool:
        """Apply a single migration"""
        try:
            logger.info(f"📝 Applying migration: {migration_file.name}")

            # Read migration content
            with open(migration_file, 'r') as f:
                content = f.read()

            checksum = self._calculate_checksum(content)

            # Execute migration in a transaction
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Execute the migration SQL
                    await conn.execute(content)

                    # Record migration as applied
                    await conn.execute("""
                        INSERT INTO schema_migrations (migration_name, checksum, success)
                        VALUES ($1, $2, TRUE)
                        ON CONFLICT (migration_name) DO UPDATE
                        SET checksum = $2, applied_at = NOW()
                    """, migration_file.name, checksum)

            logger.info(f"✅ Migration applied successfully: {migration_file.name}")
            return True

        except Exception as e:
            logger.error(f"❌ Failed to apply migration {migration_file.name}: {e}", exc_info=True)

            # Record failed migration
            try:
                async with self.pool.acquire() as conn:
                    await conn.execute("""
                        INSERT INTO schema_migrations (migration_name, checksum, success)
                        VALUES ($1, $2, FALSE)
                        ON CONFLICT (migration_name) DO NOTHING
                    """, migration_file.name, self._calculate_checksum(content))
            except:
                pass

            return False

    async def run_migrations(self, fail_fast: bool = True) -> int:
        """
        Run all pending migrations

        Args:
            fail_fast: Stop on first failure if True, continue if False

        Returns:
            Number of migrations applied
        """
        try:
            # Initialize tracking table
            await self.initialize()

            # Get pending migrations
            pending = await self.get_pending_migrations()

            if not pending:
                logger.info("✅ No pending migrations")
                return 0

            logger.info(f"📋 Found {len(pending)} pending migration(s)")

            applied_count = 0

            for migration_file in pending:
                success = await self.apply_migration(migration_file)

                if success:
                    applied_count += 1
                elif fail_fast:
                    logger.error("Migration failed - stopping (fail_fast=True)")
                    break

            if applied_count == len(pending):
                logger.info(f"✅ All {applied_count} migration(s) applied successfully")
            else:
                logger.warning(f"⚠️  Applied {applied_count}/{len(pending)} migrations")

            return applied_count

        except Exception as e:
            logger.error(f"❌ Error running migrations: {e}", exc_info=True)
            return 0

    async def verify_migrations(self) -> bool:
        """Verify all applied migrations still match their checksums"""
        try:
            async with self.pool.acquire() as conn:
                applied = await conn.fetch("""
                    SELECT migration_name, checksum FROM schema_migrations
                    WHERE success = TRUE
                    ORDER BY id
                """)

            mismatches = []

            for row in applied:
                migration_file = self.migrations_dir / row['migration_name']

                if not migration_file.exists():
                    logger.warning(f"⚠️  Migration file missing: {row['migration_name']}")
                    continue

                with open(migration_file, 'r') as f:
                    content = f.read()

                current_checksum = self._calculate_checksum(content)

                if current_checksum != row['checksum']:
                    mismatches.append(row['migration_name'])
                    logger.warning(
                        f"⚠️  Migration checksum mismatch: {row['migration_name']}\n"
                        f"   Applied: {row['checksum']}\n"
                        f"   Current: {current_checksum}"
                    )

            if mismatches:
                logger.warning(f"❌ {len(mismatches)} migration(s) have changed since being applied!")
                return False

            logger.info("✅ All applied migrations verified successfully")
            return True

        except Exception as e:
            logger.error(f"Error verifying migrations: {e}")
            return False


async def run_migrations_cli(database_url: str):
    """CLI entry point for running migrations"""
    import asyncpg

    try:
        # Connect to database
        pool = await asyncpg.create_pool(database_url, min_size=1, max_size=2)

        # Create migration manager
        mgr = MigrationManager(pool)

        # Run migrations
        applied = await mgr.run_migrations(fail_fast=True)

        # Close pool
        await pool.close()

        if applied > 0:
            print(f"\n✅ Successfully applied {applied} migration(s)")
            return 0
        elif applied == 0:
            print("\n✅ Database schema is up to date")
            return 0
        else:
            print("\n❌ Failed to apply migrations")
            return 1

    except Exception as e:
        print(f"\n❌ Migration error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    import asyncio
    import os
    import sys

    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://bot_user:bot_password@localhost:5432/tradingbot'
    )

    print(f"🔄 Running migrations on: {database_url}\n")
    exit_code = asyncio.run(run_migrations_cli(database_url))
    sys.exit(exit_code)
