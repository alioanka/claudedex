#!/usr/bin/env python3
"""
Emergency script to create admin user if auth initialization is failing
Run this inside the Docker container
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


async def create_admin_user():
    """Create admin user and auth tables if they don't exist"""
    print("=" * 80)
    print("🔧 EMERGENCY ADMIN USER CREATION")
    print("=" * 80)

    # Check if bcrypt is available
    try:
        import bcrypt
        print("✅ bcrypt is available")
    except ImportError:
        print("❌ bcrypt is NOT available - installing...")
        os.system("pip install bcrypt")
        import bcrypt
        print("✅ bcrypt installed")

    # Database connection
    db_host = os.getenv('DB_HOST', 'localhost')
    db_port = os.getenv('DB_PORT', '5432')
    db_name = os.getenv('DB_NAME', 'trading_bot')
    db_user = os.getenv('DB_USER', 'postgres')
    db_password = os.getenv('DB_PASSWORD', 'postgres')

    print(f"\n📡 Connecting to database: {db_host}:{db_port}/{db_name}")

    try:
        conn = await asyncpg.connect(
            host=db_host,
            port=int(db_port),
            database=db_name,
            user=db_user,
            password=db_password
        )
        print("✅ Connected to database")

        # Check if users table exists
        users_exists = await conn.fetchval(
            "SELECT EXISTS(SELECT 1 FROM information_schema.tables WHERE table_name = 'users')"
        )

        if not users_exists:
            print("\n❌ users table does NOT exist - creating auth tables...")

            # Read and execute migration
            migration_file = Path(__file__).parent.parent / 'migrations' / '001_add_auth_tables.sql'

            if migration_file.exists():
                print(f"📄 Reading migration: {migration_file}")
                with open(migration_file, 'r') as f:
                    migration_sql = f.read()

                print("🔄 Executing migration...")
                await conn.execute(migration_sql)
                print("✅ Auth tables created")
            else:
                print(f"❌ Migration file not found: {migration_file}")
                print("   Creating tables manually...")

                # Create tables manually
                await conn.execute("""
                    CREATE TABLE IF NOT EXISTS users (
                        id SERIAL PRIMARY KEY,
                        username VARCHAR(50) UNIQUE NOT NULL,
                        password_hash VARCHAR(255) NOT NULL,
                        role VARCHAR(20) NOT NULL CHECK (role IN ('admin', 'operator', 'viewer')),
                        email VARCHAR(255),
                        is_active BOOLEAN DEFAULT TRUE,
                        require_2fa BOOLEAN DEFAULT FALSE,
                        totp_secret VARCHAR(32),
                        failed_login_attempts INTEGER DEFAULT 0,
                        last_login TIMESTAMPTZ,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        updated_at TIMESTAMPTZ DEFAULT NOW()
                    );

                    CREATE INDEX IF NOT EXISTS idx_users_username ON users(username);
                    CREATE INDEX IF NOT EXISTS idx_users_role ON users(role);
                    CREATE INDEX IF NOT EXISTS idx_users_is_active ON users(is_active);

                    CREATE TABLE IF NOT EXISTS sessions (
                        session_id VARCHAR(64) PRIMARY KEY,
                        user_id INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
                        ip_address VARCHAR(45) NOT NULL,
                        user_agent TEXT,
                        created_at TIMESTAMPTZ DEFAULT NOW(),
                        expires_at TIMESTAMPTZ NOT NULL,
                        last_activity TIMESTAMPTZ DEFAULT NOW(),
                        is_active BOOLEAN DEFAULT TRUE
                    );

                    CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);
                    CREATE INDEX IF NOT EXISTS idx_sessions_expires_at ON sessions(expires_at);
                    CREATE INDEX IF NOT EXISTS idx_sessions_is_active ON sessions(is_active);

                    CREATE TABLE IF NOT EXISTS audit_logs (
                        id SERIAL PRIMARY KEY,
                        user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
                        username VARCHAR(50) NOT NULL,
                        action VARCHAR(50) NOT NULL,
                        resource_type VARCHAR(50) NOT NULL,
                        resource_id VARCHAR(255),
                        old_value TEXT,
                        new_value TEXT,
                        ip_address VARCHAR(45) NOT NULL,
                        user_agent TEXT,
                        timestamp TIMESTAMPTZ DEFAULT NOW(),
                        success BOOLEAN DEFAULT TRUE,
                        error_message TEXT
                    );

                    CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
                    CREATE INDEX IF NOT EXISTS idx_audit_logs_resource ON audit_logs(resource_type, resource_id);
                """)
                print("✅ Auth tables created manually")

        else:
            print("✅ users table exists")

        # Check if admin user exists
        admin = await conn.fetchrow("SELECT * FROM users WHERE username = 'admin'")

        if admin:
            print("\n✅ Admin user already exists")
            print(f"   Username: {admin['username']}")
            print(f"   Role: {admin['role']}")
            print(f"   Is Active: {admin['is_active']}")

            # Test password
            print("\n🔐 Testing password...")
            test_password = 'admin123'
            stored_hash = admin['password_hash']

            try:
                match = bcrypt.checkpw(test_password.encode('utf-8'), stored_hash.encode('utf-8'))
                if match:
                    print(f"   ✅ Password '{test_password}' is CORRECT")
                else:
                    print(f"   ❌ Password '{test_password}' is INCORRECT")
                    print("   Updating password...")

                    new_hash = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt(12)).decode('utf-8')
                    await conn.execute(
                        "UPDATE users SET password_hash = $1, updated_at = NOW() WHERE username = 'admin'",
                        new_hash
                    )
                    print("   ✅ Password updated to 'admin123'")
            except Exception as e:
                print(f"   ❌ Error testing password: {e}")
                print("   Resetting password...")

                new_hash = bcrypt.hashpw(test_password.encode('utf-8'), bcrypt.gensalt(12)).decode('utf-8')
                await conn.execute(
                    "UPDATE users SET password_hash = $1, updated_at = NOW() WHERE username = 'admin'",
                    new_hash
                )
                print("   ✅ Password reset to 'admin123'")

        else:
            print("\n❌ Admin user does NOT exist - creating...")

            # Create admin user
            password = 'admin123'
            password_hash = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt(12)).decode('utf-8')

            await conn.execute("""
                INSERT INTO users (username, password_hash, role, is_active, require_2fa)
                VALUES ($1, $2, $3, $4, $5)
            """, 'admin', password_hash, 'admin', True, False)

            print("✅ Admin user created!")
            print(f"   Username: admin")
            print(f"   Password: {password}")
            print(f"   Role: admin")

        await conn.close()

        print("\n" + "=" * 80)
        print("✅ ADMIN USER READY")
        print("=" * 80)
        print("   Login URL: http://your-server:8080/login")
        print("   Username: admin")
        print("   Password: admin123")
        print("   ⚠️  CHANGE PASSWORD IMMEDIATELY AFTER LOGIN!")
        print("=" * 80)

        return True

    except Exception as e:
        print(f"\n❌ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(create_admin_user())
    sys.exit(0 if success else 1)
