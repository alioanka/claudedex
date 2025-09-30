"""Database optimization script"""
import asyncio
import asyncpg
import os
from datetime import datetime, timedelta

async def optimize_db():
    """Optimize database performance"""
    
    print("DATABASE OPTIMIZATION")
    print("="*60)
    
    conn = await asyncpg.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123")
    )
    
    try:
        # Analyze table sizes
        print("\nAnalyzing table sizes...")
        tables = await conn.fetch("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                pg_total_relation_size(schemaname||'.'||tablename) AS bytes
            FROM pg_tables
            WHERE schemaname = 'public'
            ORDER BY pg_total_relation_size(schemaname||'.'||tablename) DESC
        """)
        
        for table in tables:
            print(f"   {table['tablename']}: {table['size']}")
        
        # Vacuum analyze all tables
        print("\nRunning VACUUM ANALYZE...")
        for table in tables:
            table_name = table['tablename']
            print(f"   Processing {table_name}...")
            await conn.execute(f"VACUUM ANALYZE {table_name}")
        print("   VACUUM complete")
        
        # Reindex tables
        print("\nReindexing tables...")
        for table in tables:
            table_name = table['tablename']
            try:
                await conn.execute(f"REINDEX TABLE {table_name}")
                print(f"   Reindexed {table_name}")
            except Exception as e:
                print(f"   Warning on {table_name}: {str(e)[:50]}")
        
        # Check for missing indexes
        print("\nChecking for missing indexes...")
        missing_indexes = await conn.fetch("""
            SELECT 
                schemaname,
                tablename,
                attname,
                n_distinct,
                correlation
            FROM pg_stats
            WHERE schemaname = 'public'
            AND n_distinct > 100
            AND correlation < 0.5
            AND NOT EXISTS (
                SELECT 1 FROM pg_index i
                JOIN pg_attribute a ON a.attrelid = i.indrelid
                WHERE a.attname = pg_stats.attname
            )
        """)
        
        if missing_indexes:
            print("   Suggested indexes:")
            for idx in missing_indexes[:5]:
                print(f"   - CREATE INDEX ON {idx['tablename']}({idx['attname']})")
        else:
            print("   No missing indexes found")
        
        # Cleanup old data
        print("\nCleaning up old data...")
        
        # Delete old audit logs (>90 days)
        deleted = await conn.fetchval("""
            DELETE FROM audit_logs
            WHERE timestamp < NOW() - INTERVAL '90 days'
            RETURNING COUNT(*)
        """)
        print(f"   Deleted {deleted or 0} old audit logs")
        
        # Delete old market data (>30 days)
        deleted = await conn.fetchval("""
            DELETE FROM market_data
            WHERE timestamp < NOW() - INTERVAL '30 days'
            RETURNING COUNT(*)
        """)
        print(f"   Deleted {deleted or 0} old market data records")
        
        # Update statistics
        print("\nUpdating statistics...")
        await conn.execute("ANALYZE")
        print("   Statistics updated")
        
        # Check for bloat
        print("\nChecking for table bloat...")
        bloat = await conn.fetch("""
            SELECT 
                schemaname,
                tablename,
                pg_size_pretty(pg_total_relation_size(schemaname||'.'||tablename)) AS size,
                ROUND((pg_relation_size(schemaname||'.'||tablename) - 
                       pg_relation_size(schemaname||'.'||tablename, 'main')) * 100.0 / 
                       NULLIF(pg_relation_size(schemaname||'.'||tablename), 0), 2) AS bloat_pct
            FROM pg_tables
            WHERE schemaname = 'public'
            AND pg_total_relation_size(schemaname||'.'||tablename) > 1000000
        """)
        
        for table in bloat:
            if table['bloat_pct'] and table['bloat_pct'] > 20:
                print(f"   {table['tablename']}: {table['bloat_pct']}% bloat")
        
        # Connection pool stats
        print("\nConnection pool stats...")
        pool_stats = await conn.fetchrow("""
            SELECT 
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE state = 'active') as active,
                COUNT(*) FILTER (WHERE state = 'idle') as idle
            FROM pg_stat_activity
            WHERE datname = $1
        """, os.getenv("DB_NAME", "tradingbot"))
        
        print(f"   Total connections: {pool_stats['total']}")
        print(f"   Active: {pool_stats['active']}")
        print(f"   Idle: {pool_stats['idle']}")
        
        # Performance recommendations
        print("\nPerformance Recommendations:")
        
        # Check for sequential scans
        seqscans = await conn.fetch("""
            SELECT 
                schemaname,
                tablename,
                seq_scan,
                idx_scan,
                ROUND(100.0 * seq_scan / NULLIF(seq_scan + idx_scan, 0), 2) AS seq_pct
            FROM pg_stat_user_tables
            WHERE seq_scan > 0
            AND seq_scan + idx_scan > 100
            ORDER BY seq_scan DESC
            LIMIT 5
        """)
        
        for table in seqscans:
            if table['seq_pct'] and table['seq_pct'] > 50:
                print(f"   Consider adding index to {table['tablename']} (seq scans: {table['seq_pct']}%)")
        
        print("\nOptimization complete!")
        
    except Exception as e:
        print(f"\nError during optimization: {str(e)}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(optimize_db())