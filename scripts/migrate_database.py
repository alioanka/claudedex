"""
Enhanced Database Migration Script with Docker Support

Features:
- Supports both fresh DB creation and existing DB migration
- Looks in correct migration directory (data/storage/migrations)
- Idempotent - safe to run multiple times
- Transaction support for rollback on errors
- Proper error handling and logging
"""
import asyncio
import asyncpg
import os
import sys
from pathlib import Path
from datetime import datetime


async def migrate_database():
    """Run database migrations"""

    print("=" * 70)
    print("DATABASE MIGRATION SYSTEM")
    print("=" * 70)

    # Database connection parameters from Docker secrets or environment
    # Import here to avoid circular imports
    try:
        from security.docker_secrets import get_db_credentials
        db_creds = get_db_credentials()
        db_host = db_creds['host']
        db_port = int(db_creds['port'])
        db_name = db_creds['name']
        db_user = db_creds['user']
        db_password = db_creds['password']
    except ImportError:
        # Fallback if docker_secrets not available
        db_host = os.getenv("DB_HOST", "localhost")
        db_port = int(os.getenv("DB_PORT", "5432"))
        db_name = os.getenv("DB_NAME", "tradingbot")
        db_user = os.getenv("DB_USER", "bot_user")
        db_password = os.getenv("DB_PASSWORD", "")

    print(f"\n📊 Connecting to: {db_user}@{db_host}:{db_port}/{db_name}")

    # Retry logic for Docker startup (database might not be ready immediately)
    max_retries = 30
    retry_delay = 2

    conn = None
    for attempt in range(1, max_retries + 1):
        try:
            conn = await asyncpg.connect(
                host=db_host,
                port=db_port,
                database=db_name,
                user=db_user,
                password=db_password,
                timeout=10
            )
            print(f"✅ Connected to database (attempt {attempt}/{max_retries})")
            break
        except Exception as e:
            if attempt < max_retries:
                print(f"⏳ Waiting for database... (attempt {attempt}/{max_retries})")
                await asyncio.sleep(retry_delay)
            else:
                print(f"❌ Failed to connect after {max_retries} attempts: {e}")
                sys.exit(1)

    if not conn:
        print("❌ No database connection established")
        sys.exit(1)

    try:
        # Create migrations tracking table if not exists
        print("\n📋 Setting up migrations table...")
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                version VARCHAR(100) UNIQUE NOT NULL,
                description TEXT,
                applied_at TIMESTAMP DEFAULT NOW(),
                checksum VARCHAR(64)
            )
        """)

        # Add checksum column if it doesn't exist (for existing databases)
        try:
            await conn.execute("""
                ALTER TABLE migrations ADD COLUMN IF NOT EXISTS checksum VARCHAR(64)
            """)
        except Exception as e:
            # Column might already exist, ignore
            pass

        print("✅ Migrations table ready")

        # Get applied migrations
        applied = await conn.fetch("SELECT version FROM migrations ORDER BY version")
        applied_versions = {m['version'] for m in applied}

        print(f"\n📊 Currently applied migrations: {len(applied_versions)}")
        for version in sorted(applied_versions):
            print(f"   ✓ {version}")

        # Find migration files in correct directory
        # Support both absolute and relative paths
        migrations_dir = Path(__file__).parent.parent / 'data' / 'storage' / 'migrations'

        # Fallback to relative path if absolute doesn't exist
        if not migrations_dir.exists():
            migrations_dir = Path('data/storage/migrations')

        # Create directory if it doesn't exist (for fresh installations)
        migrations_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n🔍 Scanning migrations directory: {migrations_dir.absolute()}")

        # Find all SQL migration files
        migration_files = sorted(migrations_dir.glob('*.sql'))

        if not migration_files:
            print("\n⚠️  No migration files found")
            print(f"   Expected location: {migrations_dir.absolute()}")
            print("   This is normal for a fresh installation using init.sql")
            return

        print(f"📁 Found {len(migration_files)} migration file(s)")

        # Apply pending migrations
        pending_count = 0
        failed_count = 0

        for migration_file in migration_files:
            version = migration_file.stem  # Filename without .sql

            if version in applied_versions:
                print(f"\n✓ {version:40s} [ALREADY APPLIED]")
                continue

            print(f"\n→ Applying {version}...")
            print(f"  File: {migration_file.name}")

            try:
                # Read migration SQL
                with open(migration_file, 'r', encoding='utf-8') as f:
                    sql = f.read()

                # Calculate checksum for verification
                import hashlib
                checksum = hashlib.sha256(sql.encode()).hexdigest()[:16]

                # Execute in transaction for atomicity
                async with conn.transaction():
                    # Execute the migration
                    await conn.execute(sql)

                    # Record migration in tracking table
                    await conn.execute("""
                        INSERT INTO migrations (version, description, checksum)
                        VALUES ($1, $2, $3)
                    """, version, f"Migration from {migration_file.name}", checksum)

                print(f"  ✅ Applied successfully (checksum: {checksum})")
                pending_count += 1

            except Exception as e:
                print(f"  ❌ Failed to apply migration: {str(e)}")
                print(f"  ⚠️  Error details: {type(e).__name__}")
                failed_count += 1

                # For critical errors, stop the migration process
                if "syntax error" in str(e).lower() or "does not exist" in str(e).lower():
                    print("\n❌ Critical migration error detected. Stopping migration process.")
                    raise

        # Summary
        print("\n" + "=" * 70)
        print("MIGRATION SUMMARY")
        print("=" * 70)
        print(f"Total migrations found:    {len(migration_files)}")
        print(f"Already applied:           {len(applied_versions)}")
        print(f"Applied this run:          {pending_count}")
        print(f"Failed:                    {failed_count}")
        print(f"Database status:           {'✅ UP TO DATE' if pending_count == 0 and failed_count == 0 else '⚠️  NEEDS ATTENTION' if failed_count > 0 else '✅ UPDATED'}")
        print("=" * 70)

        if failed_count > 0:
            print("\n⚠️  Some migrations failed. Please review errors above.")
            sys.exit(1)
        elif pending_count > 0:
            print(f"\n✅ Successfully applied {pending_count} new migration(s)")
        else:
            print("\n✅ Database schema is up to date")

    except Exception as e:
        print(f"\n❌ Migration process failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    finally:
        if conn:
            await conn.close()
            print("\n🔌 Database connection closed")


if __name__ == "__main__":
    print("\n🚀 Starting migration process...\n")
    try:
        asyncio.run(migrate_database())
        print("\n✅ Migration process completed successfully\n")
    except KeyboardInterrupt:
        print("\n⚠️  Migration interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Migration failed: {e}")
        sys.exit(1)
