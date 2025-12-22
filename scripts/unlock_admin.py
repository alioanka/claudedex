#!/usr/bin/env python3
"""Unlock admin account and reset password"""
import asyncio
import asyncpg
import bcrypt
import os
from pathlib import Path

async def unlock_admin():
    """Unlock admin account and reset password to admin123"""

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
        # Generate new password hash for "admin123"
        password = "admin123"
        password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(rounds=12)).decode('utf-8')

        print(f"Generated new password hash for: {password}")
        print(f"Hash: {password_hash}")
        print()

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
        print(f"✅ Password reset to: {password}")
        print(f"✅ Failed login attempts reset to 0")
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
    asyncio.run(unlock_admin())
