#!/usr/bin/env python3
"""Unlock admin account and reset password.

By default generates a random URL-safe password and prints it ONCE
to stdout. Pass --password to supply your own value.
"""
import argparse
import asyncio
import asyncpg
import bcrypt
import os
import secrets
from pathlib import Path

async def unlock_admin(new_password: str, generated: bool):
    """Unlock admin account and reset password to the supplied value."""

    # Database connection using Docker secrets or environment
    try:
        from security.docker_secrets import get_db_credentials
        db_creds = get_db_credentials()
        print(f"Connecting using Docker secrets/environment...")
        conn = await asyncpg.connect(
            host=db_creds['host'],
            port=int(db_creds['port']),
            database=db_creds['name'],
            user=db_creds['user'],
            password=db_creds['password']
        )
    except ImportError:
        # Fallback to DATABASE_URL or individual vars
        database_url = os.getenv("DATABASE_URL")
        if database_url:
            print(f"Connecting using DATABASE_URL...")
            conn = await asyncpg.connect(database_url)
        else:
            print(f"Connecting using individual DB env vars...")
            conn = await asyncpg.connect(
                host=os.getenv("DB_HOST", "postgres"),
                port=int(os.getenv("DB_PORT", 5432)),
                database=os.getenv("DB_NAME", "tradingbot"),
                user=os.getenv("DB_USER", "bot_user"),
                password=os.getenv("DB_PASSWORD", "")
            )

    try:
        password_hash = bcrypt.hashpw(
            new_password.encode('utf-8'),
            bcrypt.gensalt(rounds=12),
        ).decode('utf-8')

        # Update admin user - reset failed attempts and update password
        result = await conn.execute("""
            UPDATE users
            SET failed_login_attempts = 0,
                password_hash = $1,
                is_active = TRUE,
                updated_at = NOW()
            WHERE username = 'admin'
        """, password_hash)

        print(f"✅ Admin account unlocked!")
        print(f"✅ Failed login attempts reset to 0")
        if generated:
            # Bare print(), NOT logger - must not land in log shippers.
            print()
            print("=" * 72)
            print(f"New admin password: {new_password}")
            print("RECORD THIS NOW. It is not logged and will not be shown again.")
            print("=" * 72)
        else:
            print("✅ Password updated to operator-supplied value")
        print()

        # Verify the user exists
        user = await conn.fetchrow("SELECT * FROM users WHERE username = 'admin'")
        if user:
            print(f"Admin User Details:")
            print(f"  ID: {user['id']}")
            print(f"  Username: {user['username']}")
            print(f"  Role: {user['role']}")
            print(f"  Active: {user['is_active']}")
            print(f"  Failed Attempts: {user['failed_login_attempts']}")
            print(f"  Last Login: {user['last_login']}")
        else:
            print("⚠️  Admin user not found!")

    finally:
        await conn.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Unlock admin account and reset password.",
    )
    parser.add_argument(
        "--password",
        help=(
            "New password. If omitted, a random URL-safe password is "
            "generated and printed once to stdout."
        ),
    )
    args = parser.parse_args()
    generated = args.password is None
    pw = args.password or secrets.token_urlsafe(24)
    asyncio.run(unlock_admin(pw, generated))
