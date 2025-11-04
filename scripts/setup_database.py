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
        
        await conn.execute("""
            CREATE SCHEMA IF NOT EXISTS trading;
        """)

        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trading.position_manager_state (
                id INT PRIMARY KEY,
                consecutive_losses INT NOT NULL DEFAULT 0,
                consecutive_losses_blocked_at TIMESTAMPTZ,
                consecutive_losses_block_count INT NOT NULL DEFAULT 0,
                last_reset_at TIMESTAMPTZ,
                updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
            );
        """)

        # Create tables
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS trades (
                id SERIAL PRIMARY KEY,
                trade_id UUID UNIQUE NOT NULL,
                token_address VARCHAR(128) NOT NULL,
                chain VARCHAR(30) NOT NULL,
                side VARCHAR(10) NOT NULL,
                entry_price NUMERIC(40, 18),
                exit_price NUMERIC(40, 18),
                amount NUMERIC(40, 18),
                usd_value NUMERIC(40, 18),
                gas_fee NUMERIC(40, 18),
                slippage NUMERIC(10, 4),
                profit_loss NUMERIC(40, 18),
                profit_loss_percentage NUMERIC(10, 4),
                strategy VARCHAR(100),
                risk_score NUMERIC(10, 4),
                ml_confidence NUMERIC(10, 4),
                entry_timestamp TIMESTAMPTZ NOT NULL,
                exit_timestamp TIMESTAMPTZ,
                status VARCHAR(20) NOT NULL,
                metadata JSONB,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS positions (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                token VARCHAR(128) NOT NULL,
                chain VARCHAR(30) NOT NULL,
                entry_price NUMERIC(40, 18) NOT NULL,
                current_price NUMERIC(40, 18),
                amount NUMERIC(40, 18) NOT NULL,
                stop_loss NUMERIC(40, 18),
                take_profit NUMERIC(40, 18),
                pnl NUMERIC(40, 18),
                pnl_percentage NUMERIC(10, 4),
                status VARCHAR(20) NOT NULL,
                entry_time TIMESTAMPTZ DEFAULT NOW(),
                exit_time TIMESTAMPTZ,
                exit_reason VARCHAR(50),
                created_at TIMESTAMPTZ DEFAULT NOW(),
                updated_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS market_data (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                token VARCHAR(128) NOT NULL,
                chain VARCHAR(30) NOT NULL,
                price NUMERIC(40, 18) NOT NULL,
                volume NUMERIC(40, 18),
                liquidity NUMERIC(40, 18),
                market_cap NUMERIC(40, 18),
                holders INTEGER,
                timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS token_analysis (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                token VARCHAR(128) NOT NULL,
                chain VARCHAR(30) NOT NULL,
                analysis_type VARCHAR(50) NOT NULL,
                score NUMERIC(10, 4),
                confidence NUMERIC(10, 4),
                details JSONB,
                timestamp TIMESTAMPTZ DEFAULT NOW(),
                created_at TIMESTAMPTZ DEFAULT NOW()
            );
        """)
        
        await conn.execute("""
            CREATE TABLE IF NOT EXISTS whale_activities (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                wallet VARCHAR(128) NOT NULL,
                token VARCHAR(128) NOT NULL,
                chain VARCHAR(30) NOT NULL,
                type VARCHAR(20) NOT NULL,
                amount NUMERIC(40, 18) NOT NULL,
                value_usd NUMERIC(40, 18),
                timestamp TIMESTAMPTZ NOT NULL,
                created_at TIMESTAMPTZ DEFAULT NOW()
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
                timestamp TIMESTAMPTZ DEFAULT NOW()
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
        await conn.execute("CREATE INDEX IF NOT EXISTS trades_trade_id_idx ON trades(trade_id);")
        await conn.execute("CREATE INDEX IF NOT EXISTS idx_trades_token ON trades(token_address, created_at DESC);")
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
