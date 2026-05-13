#!/usr/bin/env python3
"""
Initialize Authentication System
Creates auth tables and seeds-or-rotates the admin user with a
randomly-generated password printed ONCE to stdout.
"""
import asyncio
import asyncpg
import bcrypt
import os
import secrets
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# bcrypt hash of the historical leaked default "admin123" that shipped in
# scripts/init.sql and migrations/001_add_auth_tables.sql before MB-29b.
KNOWN_BAD_HASH = b"$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/Lfw99hhm1qJYT8sFm"


def _gen_password() -> str:
    """Return a ~32-char URL-safe random password."""
    return secrets.token_urlsafe(24)


def _print_once_block(title: str, password: str) -> None:
    # Use bare print(), NOT logger - logs may ship to Loki/CloudWatch.
    print()
    print("=" * 72)
    print(f"⚠️  {title}")
    print("=" * 72)
    print("Username: admin")
    print(f"Password: {password}")
    print()
    print("🔴 RECORD THIS NOW — it will NOT be shown again and is NOT logged.")
    print("   Rotate it via the dashboard after first login.")
    print("=" * 72)
    print()


async def _ensure_admin(conn) -> None:
    """Seed admin on fresh install; rotate if still on the leaked default."""
    row = await conn.fetchrow(
        "SELECT id, password_hash FROM users WHERE username='admin'"
    )
    if row is None:
        pw = _gen_password()
        h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()
        await conn.execute(
            "INSERT INTO users (username, password_hash, role, email, "
            "is_active, require_2fa) "
            "VALUES ('admin', $1, 'admin', NULL, TRUE, FALSE)",
            h,
        )
        _print_once_block("Default admin user CREATED", pw)
        return

    stored_raw = row['password_hash']
    stored = stored_raw.encode() if isinstance(stored_raw, str) else stored_raw
    is_leaked_default = stored == KNOWN_BAD_HASH
    if not is_leaked_default:
        try:
            is_leaked_default = bcrypt.checkpw(b"admin123", stored)
        except ValueError:
            is_leaked_default = False

    if is_leaked_default:
        pw = _gen_password()
        h = bcrypt.hashpw(pw.encode(), bcrypt.gensalt(rounds=12)).decode()
        await conn.execute(
            "UPDATE users SET password_hash=$1, updated_at=NOW() WHERE id=$2",
            h, row['id'],
        )
        _print_once_block(
            "Default admin password ROTATED (was leaked default)", pw,
        )
        return

    print("ℹ️  Admin user already exists with non-default password "
          "(no rotation needed)")


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

        await _ensure_admin(conn)

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
