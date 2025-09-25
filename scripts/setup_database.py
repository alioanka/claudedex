# scripts/setup_database.py
"""
Database setup and migration script
"""
import asyncio
import asyncpg
from pathlib import Path
import sys
import os

sys.path.append(str(Path(__file__).parent.parent))

async def setup_database():
    """Setup database with tables and extensions"""
    
    # Database configuration
    config = {
        "host": os.getenv("DB_HOST", "localhost"),
        "port": int(os.getenv("DB_PORT", 5432)),
        "database": os.getenv("DB_NAME", "tradingbot"),
        "user": os.getenv("DB_USER", "trading"),
        "password": os.getenv("DB_PASSWORD", "trading123")
    }
    
    # Connect to database
    conn = await asyncpg.connect(**config)
    
    try:
        print("📦 Setting up database...")
        
        # Enable extensions
        await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS pgcrypto;")
        await conn.execute("CREATE EXTENSION IF NOT EXISTS uuid-ossp;")
        
        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                token VARCHAR(42) NOT NULL,
                chain VARCHAR(20) NOT NULL,
                type VARCHAR(10) NOT NULL,
                amount DECIMAL(30, 18) NOT NULL,
                price DECIMAL(30, 18) NOT NULL,
                total DECIMAL(30, 18) NOT NULL,
                gas_price BIGINT,
                gas_used BIGINT,
                tx_hash VARCHAR(66),
                status VARCHAR(20) NOT NULL,
                error_message TEXT,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                token VARCHAR(42) NOT NULL,
                chain VARCHAR(20) NOT NULL,
                entry_price DECIMAL(30, 18) NOT NULL,
                current_price DECIMAL(30, 18),
                quantity DECIMAL(30, 18) NOT NULL,
                stop_loss DECIMAL(30, 18),
                take_profit DECIMAL(30, 18),
                pnl DECIMAL(30, 18),
                pnl_percentage DECIMAL(10, 2),
                status VARCHAR(20) NOT NULL,
                entry_time TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                exit_time TIMESTAMP WITH TIME ZONE,
                exit_reason VARCHAR(50),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                token VARCHAR(42) NOT NULL,
                chain VARCHAR(20) NOT NULL,
                price DECIMAL(30, 18) NOT NULL,
                volume DECIMAL(30, 18),
                liquidity DECIMAL(30, 18),
                market_cap DECIMAL(30, 18),
                holders INTEGER,
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS token_analysis (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                token VARCHAR(42) NOT NULL,
                chain VARCHAR(20) NOT NULL,
                analysis_type VARCHAR(50) NOT NULL,
                score DECIMAL(5, 2),
                confidence DECIMAL(5, 2),
                details JSONB,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS whale_activities (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                wallet VARCHAR(42) NOT NULL,
                token VARCHAR(42) NOT NULL,
                chain VARCHAR(20) NOT NULL,
                type VARCHAR(20) NOT NULL,
                amount DECIMAL(30, 18) NOT NULL,
                value_usd DECIMAL(30, 2),
                timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
                created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS audit_logs (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                event_type VARCHAR(50) NOT NULL,
                severity VARCHAR(20) NOT NULL,
                source VARCHAR(100) NOT NULL,
                action VARCHAR(100) NOT NULL,
                resource VARCHAR(100),
                user_id VARCHAR(50),
                details JSONB,
                checksum VARCHAR(64) NOT NULL,
                timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW()
            );
        """)
        
        # Create TimescaleDB hypertables
        await conn.execute("""
            SELECT create_hypertable('market_data', 'timestamp', 
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE);
        """)
        
        await conn.execute("""
            SELECT create_hypertable('whale_activities', 'timestamp',
                chunk_time_interval => INTERVAL '1 day',
                if_not_exists => TRUE);
        """)
        
        # Create indexes
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token, created_at DESC);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_positions_token ON positions(token, status);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_market_data_token ON market_data(token, timestamp DESC);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_token_analysis_token ON token_analysis(token, analysis_type);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_whale_activities_wallet ON whale_activities(wallet);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_audit_logs_event ON audit_logs(event_type, timestamp DESC);")
        
        # Create continuous aggregates for TimescaleDB
        await conn.execute("""
            CREATE MATERIALIZED VIEW IF NOT EXISTS hourly_market_data
            WITH (timescaledb.continuous) AS
            SELECT 
                token,
                chain,
                time_bucket('1 hour', timestamp) AS hour,
                AVG(price) as avg_price,
                MAX(price) as high_price,
                MIN(price) as low_price,
                SUM(volume) as total_volume
            FROM market_data
            GROUP BY token, chain, hour
            WITH NO DATA;
        """)
        
        await conn.execute("""
            SELECT add_continuous_aggregate_policy('hourly_market_data',
                start_offset => INTERVAL '3 days',
                end_offset => INTERVAL '1 hour',
                schedule_interval => INTERVAL '1 hour',
                if_not_exists => TRUE);
        """)
        
        print("✅ Database setup complete!")
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    asyncio.run(setup_database())
