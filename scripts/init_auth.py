#!/usr/bin/env python3
"""
Initialize Authentication System
Creates auth tables and default admin user
"""
import asyncio
import asyncpg
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


async def init_auth(database_url: str):
    """Initialize authentication system"""
    print("🔐 Initializing Authentication System...")

    try:
        # Connect to database
        conn = await asyncpg.connect(database_url)
        print("✅ Connected to database")

        # Read migration SQL
        migration_file = Path(__file__).parent.parent / "migrations" / "001_add_auth_tables.sql"

        with open(migration_file, 'r') as f:
            migration_sql = f.read()

        print("📝 Applying auth migration...")

        # Execute migration
        await conn.execute(migration_sql)

        print("✅ Auth tables created successfully")

        # Check if admin user exists
        admin_exists = await conn.fetchval("""
            SELECT EXISTS(SELECT 1 FROM users WHERE username = 'admin')
        """)

        if admin_exists:
            print("ℹ️  Admin user already exists")
        else:
            print("✅ Default admin user created")
            print()
            print("=" * 60)
            print("⚠️  IMPORTANT: Default Credentials")
            print("=" * 60)
            print("Username: admin")
            print("Password: admin123")
            print()
            print("🔴 SECURITY WARNING:")
            print("Please change the default password immediately after first login!")
            print("=" * 60)

        await conn.close()
        print("\n✅ Authentication system initialized successfully!")
        print("🚀 You can now start the dashboard and login")

    except FileNotFoundError:
        print("❌ Migration file not found!")
        print(f"Expected: {migration_file}")
        sys.exit(1)

    except Exception as e:
        print(f"❌ Error initializing auth system: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    # Get database URL from environment or use default
    database_url = os.getenv(
        'DATABASE_URL',
        'postgresql://bot_user:bot_password@localhost:5432/tradingbot'
    )

    print(f"Database URL: {database_url}")
    print()

    asyncio.run(init_auth(database_url))
