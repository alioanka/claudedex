"""Database migration script"""
import asyncio
import asyncpg
import os
from pathlib import Path
from datetime import datetime

async def migrate_database():
    """Run database migrations"""
    
    print("DATABASE MIGRATION")
    print("="*60)
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        # Create migrations table if not exists
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS migrations (
                id SERIAL PRIMARY KEY,
                version VARCHAR(50) UNIQUE NOT NULL,
                description TEXT,
                applied_at TIMESTAMP DEFAULT NOW()
            )
        """)
        
        # Get applied migrations
        applied = await conn.fetch("SELECT version FROM migrations ORDER BY version")
        applied_versions = {m['version'] for m in applied}
        
        # Find migration files
        migrations_dir = Path('migrations')
        migrations_dir.mkdir(parents=True, exist_ok=True)
        
        migration_files = sorted(migrations_dir.glob('*.sql'))
        
        if not migration_files:
            print("\nNo migration files found")
            return
        
        print(f"\nFound {len(migration_files)} migration files")
        print(f"Applied migrations: {len(applied_versions)}")
        
        # Apply pending migrations
        pending_count = 0
        
        for migration_file in migration_files:
            version = migration_file.stem
            
            if version in applied_versions:
                print(f"\n✓ {version} - already applied")
                continue
            
            print(f"\n→ Applying {version}...")
            
            try:
                # Read migration SQL
                with open(migration_file, 'r') as f:
                    sql = f.read()
                
                # Execute in transaction
                async with conn.transaction():
                    await conn.execute(sql)
                    
                    # Record migration
                    await conn.execute("""
                        INSERT INTO migrations (version, description)
                        VALUES ($1, $2)
                    """, version, f"Migration from {migration_file.name}")
                
                print(f"  ✓ Applied successfully")
                pending_count += 1
                
            except Exception as e:
                print(f"  ✗ Failed: {str(e)}")
                raise
        
        if pending_count == 0:
            print("\n✓ Database is up to date")
        else:
            print(f"\n✓ Applied {pending_count} migrations")
        
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(migrate_database())