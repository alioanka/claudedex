#!/usr/bin/env python3
"""
Diagnostic script to check authentication setup
Run this inside the Docker container to verify auth system
"""
import asyncio
import asyncpg
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment
load_dotenv()


async def check_auth_setup():
    """Check authentication setup"""
    print("=" * 80)
    print("🔍 AUTHENTICATION SYSTEM DIAGNOSTIC")
    print("=" * 80)

    # Check if bcrypt is available
    print("\n1. Checking bcrypt installation...")
    try:
        import bcrypt
        print("✅ bcrypt is installed")

        # Test bcrypt with the migration hash
        test_password = 'admin123'
        migration_hash = '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/Lfw99hhm1qJYT8sFm'

        match = bcrypt.checkpw(test_password.encode('utf-8'), migration_hash.encode('utf-8'))
        print(f"   Testing migration hash with '{test_password}': {'✅ VALID' if match else '❌ INVALID'}")

        if not match:
            print("   ⚠️  Migration hash does not match 'admin123'")
            correct_hash = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt(12))
            print(f"   Correct hash for 'admin123': {correct_hash.decode('utf-8')}")

    except ImportError as e:
        print(f"❌ bcrypt is NOT installed: {e}")
        print("   Run: pip install bcrypt")
        return False

    # Check database connection
    print("\n2. Checking database connection...")
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'trading_bot')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')

    try:
        conn = await asyncpg.connect(
            host=db_host,
            port=int(db_port),
            database=db_name,
            user=db_user,
            password=db_password
        )
        print(f"✅ Connected to database: {db_host}:{db_port}/{db_name}")

        # Check if migrations table exists
        print("\n3. Checking schema_migrations table...")
        migrations_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'schema_migrations')"
        )

        if migrations_exists:
            print("✅ schema_migrations table exists")

            # List applied migrations
            migrations = await conn.fetch("SELECT * FROM schema_migrations ORDER BY applied_at DESC")
            print(f"   Applied migrations: {len(migrations)}")
            for mig in migrations:
                print(f"   - {mig['migration_file']} (applied: {mig['applied_at']})")
        else:
            print("❌ schema_migrations table does NOT exist")
            print("   Migrations may not have run!")

        # Check if users table exists
        print("\n4. Checking users table...")
        users_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'users')"
        )

        if users_exists:
            print("✅ users table exists")

            # Count users
            user_count = await conn.fetchval("SELECT COUNT(*) FROM users")
            print(f"   Total users: {user_count}")

            # Check admin user
            admin = await conn.fetchrow("SELECT * FROM users WHERE username = 'admin'")

            if admin:
                print("✅ Admin user exists")
                print(f"   Username: {admin['username']}")
                print(f"   Role: {admin['role']}")
                print(f"   Is Active: {admin['is_active']}")
                print(f"   Password Hash: {admin['password_hash'][:50]}...")
                print(f"   Created: {admin['created_at']}")

                # Test password
                print("\n5. Testing admin password...")
                stored_hash = admin['password_hash']

                for test_pwd in ['admin123', 'Admin123', 'admin', 'password']:
                    try:
                        match = bcrypt.checkpw(test_pwd.encode('utf-8'), stored_hash.encode('utf-8'))
                        if match:
                            print(f"   ✅ Password '{test_pwd}' matches!")
                        else:
                            print(f"   ❌ Password '{test_pwd}' does not match")
                    except Exception as e:
                        print(f"   ❌ Error testing password '{test_pwd}': {e}")

            else:
                print("❌ Admin user does NOT exist!")
                print("   Creating admin user...")

                # Create admin user
                new_hash = bcrypt.hashpw(b'admin123', bcrypt.gensalt(12)).decode('utf-8')

                await conn.execute("""
                    INSERT INTO users (username, password_hash, role, is_active)
                    VALUES ($1, $2, $3, $4)
                """, 'admin', new_hash, 'admin', True)

                print("✅ Admin user created!")
                print("   Username: admin")
                print("   Password: admin123")

        else:
            print("❌ users table does NOT exist!")
            print("   Run migrations: python scripts/run_migrations.py")

        # Check sessions table
        print("\n6. Checking sessions table...")
        sessions_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'sessions')"
        )

        if sessions_exists:
            print("✅ sessions table exists")
            session_count = await conn.fetchval("SELECT COUNT(*) FROM sessions WHERE is_active = true")
            print(f"   Active sessions: {session_count}")
        else:
            print("❌ sessions table does NOT exist!")

        # Check audit_logs table
        print("\n7. Checking audit_logs table...")
        audit_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'audit_logs')"
        )

        if audit_exists:
            print("✅ audit_logs table exists")
            log_count = await conn.fetchval("SELECT COUNT(*) FROM audit_logs")
            print(f"   Total audit logs: {log_count}")

            # Show recent login attempts
            recent_logins = await conn.fetch("""
                SELECT username, action, success, error_message, timestamp
                FROM audit_logs
                WHERE action = 'login'
                ORDER BY timestamp DESC
                LIMIT 5
            """)

            if recent_logins:
                print("   Recent login attempts:")
                for log in recent_logins:
                    status = "✅" if log['success'] else "❌"
                    print(f"   {status} {log['username']} - {log['timestamp']}")
                    if log['error_message']:
                        print(f"      Error: {log['error_message']}")
        else:
            print("❌ audit_logs table does NOT exist!")

        await conn.close()

        print("\n" + "=" * 80)
        print("✅ Diagnostic complete!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return False


if __name__ == "__main__":
    asyncio.run(check_auth_setup())
