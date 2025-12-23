# data/storage/database.py

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta
from decimal import Decimal
from contextlib import asynccontextmanager

import asyncpg
from asyncpg.pool import Pool
import orjson

logger = logging.getLogger(__name__)


def _get_db_password() -> str:
    """Get database password from Docker secrets or environment."""
    # Try Docker secrets first
    try:
        from security.docker_secrets import get_secret
        password = get_secret('db_password', env_var='DB_PASSWORD')
        if password:
            return password
    except ImportError:
        pass

    # Fall back to environment
    return os.getenv('DB_PASSWORD', '')


class DatabaseManager:
    """
    PostgreSQL database manager with TimescaleDB extensions for time-series data.
    Handles all database operations for the trading bot.
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.pool: Optional[Pool] = None
        self.is_connected = False

    async def connect(self) -> None:
        """Establish connection pool to PostgreSQL database."""
        try:
            # Check if DATABASE_URL exists (single connection string)
            database_url = self.config.get('DATABASE_URL') or os.getenv('DATABASE_URL')

            if database_url:
                # Parse DATABASE_URL: postgresql://user:pass@host:port/dbname
                import urllib.parse
                parsed = urllib.parse.urlparse(database_url)

                host = parsed.hostname or 'postgres'
                port = parsed.port or 5432
                user = parsed.username or 'bot_user'
                password = parsed.password or _get_db_password()
                database = parsed.path.lstrip('/') or 'tradingbot'

                logger.info(f"Connecting to database at {host}:{port}/{database}")

                self.pool = await asyncpg.create_pool(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    min_size=self.config.get('DB_POOL_MIN', 10),
                    max_size=self.config.get('DB_POOL_MAX', 20),
                    max_queries=self.config.get('DB_MAX_QUERIES', 50000),
                    max_inactive_connection_lifetime=self.config.get('DB_CONN_LIFETIME', 300),
                    command_timeout=self.config.get('DB_COMMAND_TIMEOUT', 60),
                )
            else:
                # Fall back to individual config keys with Docker secrets support
                host = self.config.get('DB_HOST') or os.getenv('DB_HOST', 'postgres')
                port = self.config.get('DB_PORT') or int(os.getenv('DB_PORT', 5432))
                user = self.config.get('DB_USER') or os.getenv('DB_USER', 'bot_user')
                password = self.config.get('DB_PASSWORD') or _get_db_password()
                database = self.config.get('DB_NAME') or os.getenv('DB_NAME', 'tradingbot')

                logger.info(f"Connecting to database at {host}:{port}")

                self.pool = await asyncpg.create_pool(
                    host=host,
                    port=port,
                    user=user,
                    password=password,
                    database=database,
                    min_size=self.config.get('DB_POOL_MIN', 10),
                    max_size=self.config.get('DB_POOL_MAX', 20),
                    max_queries=self.config.get('DB_MAX_QUERIES', 50000),
                    max_inactive_connection_lifetime=self.config.get('DB_CONN_LIFETIME', 300),
                    command_timeout=self.config.get('DB_COMMAND_TIMEOUT', 60),
                )

            # Rest of the method stays the same...
            await self._initialize_timescaledb()

            # Run database migrations
            await self._run_migrations()

            await self._create_tables()

            self.is_connected = True
            logger.info("✅ Successfully connected to PostgreSQL database")
            
        except Exception as e:
            logger.error(f"Failed to connect to database: {e}")
            raise
    
    async def disconnect(self) -> None:
        """Close database connection pool."""
        if self.pool:
            await self.pool.close()
            self.is_connected = False
            logger.info("Disconnected from database")
    
    @asynccontextmanager
    async def acquire(self):
        """Acquire a connection from the pool."""
        async with self.pool.acquire() as connection:
            yield connection

    @asynccontextmanager
    async def transaction(self):
        """
        CRITICAL FIX (P1): Transaction context manager for ACID guarantees.

        Wraps multi-step database operations in a transaction to ensure:
        - Atomicity: All operations succeed or all fail
        - Consistency: Database remains in valid state
        - Isolation: Concurrent transactions don't interfere
        - Durability: Committed changes are permanent

        Usage:
            async with db.transaction() as conn:
                await conn.execute("INSERT INTO trades ...")
                await conn.execute("UPDATE portfolio ...")
                # Auto-commit on success, auto-rollback on exception
        """
        async with self.pool.acquire() as connection:
            async with connection.transaction():
                yield connection

    async def fetch_all(self, query: str, *args):
        """
        Execute a SELECT query and return all results.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            List of Row objects from asyncpg
        """
        if not self.pool or not self.is_connected:
            return []

        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetch_one(self, query: str, *args):
        """
        Execute a SELECT query and return a single result.

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Single Row object from asyncpg or None
        """
        if not self.pool or not self.is_connected:
            return None

        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def execute(self, query: str, *args):
        """
        Execute a query (INSERT, UPDATE, DELETE).

        Args:
            query: SQL query string
            *args: Query parameters

        Returns:
            Query result status
        """
        if not self.pool or not self.is_connected:
            return None

        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def _initialize_timescaledb(self) -> None:
        """Initialize TimescaleDB extensions and hypertables."""
        async with self.acquire() as conn:
            # Create TimescaleDB extension
            await conn.execute("CREATE EXTENSION IF NOT EXISTS timescaledb CASCADE;")

            # Create pg_stat_statements for query performance monitoring
            await conn.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements;")

            logger.info("TimescaleDB extensions initialized")

    async def _run_migrations(self) -> None:
        """Run database migrations automatically"""
        try:
            from data.migration_manager import MigrationManager

            logger.info("🔄 Checking for database migrations...")

            migration_mgr = MigrationManager(self.pool)
            applied_count = await migration_mgr.run_migrations(fail_fast=False)

            if applied_count > 0:
                logger.info(f"✅ Applied {applied_count} migration(s)")
            else:
                logger.info("✅ Database schema is up to date")

        except ImportError:
            logger.warning("⚠️  Migration manager not available - skipping migrations")
        except Exception as e:
            logger.error(f"❌ Migration error: {e}", exc_info=True)
            logger.warning("⚠️  Continuing without migrations - manual intervention may be required")

    async def _create_tables(self) -> None:
        """Create all required database tables."""
        async with self.acquire() as conn:
            # Trades table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id SERIAL PRIMARY KEY,
                    trade_id TEXT UNIQUE NOT NULL,
                    token_address TEXT NOT NULL,
                    chain TEXT NOT NULL,
                    side TEXT NOT NULL CHECK (side IN ('buy', 'sell')),
                    entry_price DECIMAL(30, 18) NOT NULL,
                    exit_price DECIMAL(30, 18),
                    amount DECIMAL(30, 18) NOT NULL,
                    usd_value DECIMAL(20, 2) NOT NULL,
                    gas_fee DECIMAL(20, 8),
                    slippage DECIMAL(5, 4),
                    profit_loss DECIMAL(20, 8),
                    profit_loss_percentage DECIMAL(10, 4),
                    strategy TEXT NOT NULL,
                    risk_score DECIMAL(5, 4),
                    ml_confidence DECIMAL(5, 4),
                    entry_timestamp TIMESTAMPTZ NOT NULL,
                    exit_timestamp TIMESTAMPTZ,
                    status TEXT NOT NULL DEFAULT 'open',
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    updated_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_trades_token_address ON trades(token_address);
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_trades_entry_timestamp ON trades(entry_timestamp DESC);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy);
                CREATE INDEX IF NOT EXISTS idx_trades_metadata ON trades USING gin(metadata);
            """)
            
            # Positions table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    id SERIAL PRIMARY KEY,
                    position_id TEXT UNIQUE NOT NULL,
                    token_address TEXT NOT NULL,
                    chain TEXT NOT NULL,
                    entry_price DECIMAL(30, 18) NOT NULL,
                    current_price DECIMAL(30, 18),
                    stop_loss DECIMAL(30, 18),
                    take_profit JSONB,
                    amount DECIMAL(30, 18) NOT NULL,
                    usd_value DECIMAL(20, 2) NOT NULL,
                    unrealized_pnl DECIMAL(20, 8),
                    unrealized_pnl_percentage DECIMAL(10, 4),
                    risk_score DECIMAL(5, 4),
                    correlation_score DECIMAL(5, 4),
                    opened_at TIMESTAMPTZ NOT NULL,
                    last_updated TIMESTAMPTZ DEFAULT NOW(),
                    status TEXT NOT NULL DEFAULT 'open',
                    metadata JSONB
                );
                
                CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
                CREATE INDEX IF NOT EXISTS idx_positions_token ON positions(token_address);
                CREATE INDEX IF NOT EXISTS idx_positions_opened_at ON positions(opened_at DESC);
            """)
            
            # Market data table (time-series)
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS market_data (
                    time TIMESTAMPTZ NOT NULL,
                    token_address TEXT NOT NULL,
                    chain TEXT NOT NULL,
                    price DECIMAL(30, 18) NOT NULL,
                    volume_24h DECIMAL(30, 18),
                    volume_5m DECIMAL(30, 18),
                    liquidity_usd DECIMAL(20, 2),
                    market_cap DECIMAL(20, 2),
                    holders INTEGER,
                    buy_count_5m INTEGER,
                    sell_count_5m INTEGER,
                    unique_buyers_5m INTEGER,
                    unique_sellers_5m INTEGER,
                    price_change_5m DECIMAL(10, 4),
                    price_change_1h DECIMAL(10, 4),
                    price_change_24h DECIMAL(10, 4),
                    metadata JSONB,
                    PRIMARY KEY (time, token_address, chain)
                );
                
                CREATE INDEX IF NOT EXISTS idx_market_data_token ON market_data(token_address);
                CREATE INDEX IF NOT EXISTS idx_market_data_time ON market_data(time DESC);
            """)
            
            # Convert market_data to hypertable
            try:
                await conn.execute("""
                    SELECT create_hypertable('market_data', 'time', 
                        chunk_time_interval => INTERVAL '1 day',
                        if_not_exists => TRUE
                    );
                """)
            except Exception as e:
                if "already a hypertable" not in str(e):
                    logger.warning(f"Could not create hypertable: {e}")
            
            # Alerts table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id SERIAL PRIMARY KEY,
                    alert_id TEXT UNIQUE NOT NULL,
                    alert_type TEXT NOT NULL,
                    priority TEXT NOT NULL,
                    title TEXT NOT NULL,
                    message TEXT,
                    data JSONB,
                    channels JSONB,
                    sent_at TIMESTAMPTZ DEFAULT NOW(),
                    acknowledged BOOLEAN DEFAULT FALSE,
                    acknowledged_at TIMESTAMPTZ
                );
                
                CREATE INDEX IF NOT EXISTS idx_alerts_type ON alerts(alert_type);
                CREATE INDEX IF NOT EXISTS idx_alerts_priority ON alerts(priority);
                CREATE INDEX IF NOT EXISTS idx_alerts_sent_at ON alerts(sent_at DESC);
            """)
            
            # Performance metrics table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS performance_metrics (
                    id SERIAL PRIMARY KEY,
                    period TEXT NOT NULL,
                    start_date TIMESTAMPTZ NOT NULL,
                    end_date TIMESTAMPTZ NOT NULL,
                    total_trades INTEGER,
                    winning_trades INTEGER,
                    losing_trades INTEGER,
                    total_pnl DECIMAL(20, 8),
                    total_pnl_percentage DECIMAL(10, 4),
                    sharpe_ratio DECIMAL(10, 4),
                    sortino_ratio DECIMAL(10, 4),
                    calmar_ratio DECIMAL(10, 4),
                    max_drawdown DECIMAL(10, 4),
                    win_rate DECIMAL(5, 4),
                    avg_win DECIMAL(20, 8),
                    avg_loss DECIMAL(20, 8),
                    best_trade DECIMAL(20, 8),
                    worst_trade DECIMAL(20, 8),
                    var_95 DECIMAL(20, 8),
                    cvar_95 DECIMAL(20, 8),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW()
                );
                
                CREATE INDEX IF NOT EXISTS idx_performance_period ON performance_metrics(period);
                CREATE INDEX IF NOT EXISTS idx_performance_dates ON performance_metrics(start_date, end_date);
            """)
            
            # Token analysis table
            await conn.execute("""
                CREATE TABLE IF NOT EXISTS token_analysis (
                    id SERIAL PRIMARY KEY,
                    token_address TEXT NOT NULL,
                    chain TEXT NOT NULL,
                    analysis_timestamp TIMESTAMPTZ NOT NULL,
                    risk_score DECIMAL(5, 4),
                    honeypot_risk DECIMAL(5, 4),
                    rug_probability DECIMAL(5, 4),
                    pump_probability DECIMAL(5, 4),
                    liquidity_score DECIMAL(5, 4),
                    holder_score DECIMAL(5, 4),
                    contract_score DECIMAL(5, 4),
                    developer_score DECIMAL(5, 4),
                    social_score DECIMAL(5, 4),
                    technical_score DECIMAL(5, 4),
                    volume_score DECIMAL(5, 4),
                    metadata JSONB,
                    created_at TIMESTAMPTZ DEFAULT NOW(),
                    UNIQUE(token_address, chain, analysis_timestamp)
                );
                
                CREATE INDEX IF NOT EXISTS idx_token_analysis_address ON token_analysis(token_address);
                CREATE INDEX IF NOT EXISTS idx_token_analysis_timestamp ON token_analysis(analysis_timestamp DESC);
            """)
            
            logger.info("Database tables created successfully")
    
    async def save_trade(self, trade: Dict[str, Any]) -> str:
        """Save a trade record to the database."""
        
        # ✅ Helper function to prevent database numeric overflow
        def safe_float(value, default=0.0, max_val=1e11):
            """Ensure float values don't overflow database precision (max 10^12)"""
            try:
                f = float(value) if value is not None else default
                return min(max(-max_val, f), max_val)
            except (ValueError, TypeError):
                return default
        
        async with self.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO trades (
                    trade_id, token_address, chain, side, entry_price, exit_price,
                    amount, usd_value, gas_fee, slippage, profit_loss, profit_loss_percentage,
                    strategy, risk_score, ml_confidence, entry_timestamp, exit_timestamp,
                    status, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19)
                RETURNING id, trade_id
            """,
                trade['trade_id'], 
                trade['token_address'], 
                trade['chain'],
                trade['side'], 
                safe_float(trade['entry_price']),
                safe_float(trade.get('exit_price')),
                safe_float(trade['amount']),
                safe_float(trade['usd_value']),
                safe_float(trade.get('gas_fee')),
                safe_float(trade.get('slippage')),
                safe_float(trade.get('profit_loss')),
                safe_float(trade.get('profit_loss_percentage')),
                trade['strategy'],
                safe_float(trade.get('risk_score')),
                safe_float(trade.get('ml_confidence')),
                trade['entry_timestamp'], 
                trade.get('exit_timestamp'),
                trade.get('status', 'open'),
                orjson.dumps(trade.get('metadata', {})).decode()
            )
            
            logger.info(f"Saved trade: {result['trade_id']}")
            return result['trade_id']
    
    async def update_trade(self, trade_id: Union[str, int], updates: Dict[str, Any]) -> bool:
        """Update an existing trade record."""
        async with self.acquire() as conn:
            # Build update query dynamically
            set_clauses = []
            values = []
            for i, (key, value) in enumerate(updates.items(), 1):
                if key == 'metadata':
                    set_clauses.append(f"{key} = ${i}::jsonb")
                    values.append(orjson.dumps(value).decode())
                else:
                    set_clauses.append(f"{key} = ${i}")
                    values.append(value)
            
            # ✅ Convert trade_id to appropriate type based on what we're matching
            # If trade_id is an integer, we're matching against 'id' column
            # If it's a string, we're matching against 'trade_id' column
            if isinstance(trade_id, int):
                # Match by integer id column
                values.append(trade_id)
                query = f"""
                    UPDATE trades 
                    SET {', '.join(set_clauses)}, updated_at = NOW()
                    WHERE id = ${len(values)}
                """
            else:
                # Match by text trade_id column
                values.append(str(trade_id))
                query = f"""
                    UPDATE trades 
                    SET {', '.join(set_clauses)}, updated_at = NOW()
                    WHERE trade_id = ${len(values)}
                """
            
            result = await conn.execute(query, *values)
            return result != "UPDATE 0"
    
    async def save_position(self, position: Dict[str, Any]) -> str:
        """Save a position record to the database."""
        async with self.acquire() as conn:
            result = await conn.fetchrow("""
                INSERT INTO positions (
                    position_id, token_address, chain, entry_price, current_price,
                    stop_loss, take_profit, amount, usd_value, unrealized_pnl,
                    unrealized_pnl_percentage, risk_score, correlation_score,
                    opened_at, status, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16)
                RETURNING id, position_id
            """,
                position['position_id'], position['token_address'], position['chain'],
                position['entry_price'], position.get('current_price'),
                position.get('stop_loss'),
                orjson.dumps(position.get('take_profit', [])).decode(),
                position['amount'], position['usd_value'],
                position.get('unrealized_pnl'), position.get('unrealized_pnl_percentage'),
                position.get('risk_score'), position.get('correlation_score'),
                position['opened_at'], position.get('status', 'open'),
                orjson.dumps(position.get('metadata', {})).decode()
            )
            
            logger.info(f"Saved position: {result['position_id']}")
            return result['position_id']
    
    async def update_position(self, position_id: str, updates: Dict[str, Any]) -> bool:
        """Update an existing position."""
        async with self.acquire() as conn:
            # Build update query
            set_clauses = []
            values = []
            for i, (key, value) in enumerate(updates.items(), 1):
                if key in ['metadata', 'take_profit']:
                    set_clauses.append(f"{key} = ${i}::jsonb")
                    values.append(orjson.dumps(value).decode())
                else:
                    set_clauses.append(f"{key} = ${i}")
                    values.append(value)
            
            values.append(position_id)
            query = f"""
                UPDATE positions 
                SET {', '.join(set_clauses)}, last_updated = NOW()
                WHERE position_id = ${len(values)}
            """
            
            result = await conn.execute(query, *values)
            return result != "UPDATE 0"
    
    async def save_market_data(self, data: Dict[str, Any]) -> None:
        """Save market data point to time-series table."""
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO market_data (
                    time, token_address, chain, price, volume_24h, volume_5m,
                    liquidity_usd, market_cap, holders, buy_count_5m, sell_count_5m,
                    unique_buyers_5m, unique_sellers_5m, price_change_5m,
                    price_change_1h, price_change_24h, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17)
                ON CONFLICT (time, token_address, chain) DO UPDATE
                SET price = EXCLUDED.price,
                    volume_24h = EXCLUDED.volume_24h,
                    volume_5m = EXCLUDED.volume_5m,
                    liquidity_usd = EXCLUDED.liquidity_usd,
                    market_cap = EXCLUDED.market_cap,
                    holders = EXCLUDED.holders,
                    buy_count_5m = EXCLUDED.buy_count_5m,
                    sell_count_5m = EXCLUDED.sell_count_5m,
                    unique_buyers_5m = EXCLUDED.unique_buyers_5m,
                    unique_sellers_5m = EXCLUDED.unique_sellers_5m,
                    price_change_5m = EXCLUDED.price_change_5m,
                    price_change_1h = EXCLUDED.price_change_1h,
                    price_change_24h = EXCLUDED.price_change_24h,
                    metadata = EXCLUDED.metadata
            """,
                data['time'], data['token_address'], data['chain'],
                data['price'], data.get('volume_24h'), data.get('volume_5m'),
                data.get('liquidity_usd'), data.get('market_cap'),
                data.get('holders'), data.get('buy_count_5m'),
                data.get('sell_count_5m'), data.get('unique_buyers_5m'),
                data.get('unique_sellers_5m'), data.get('price_change_5m'),
                data.get('price_change_1h'), data.get('price_change_24h'),
                orjson.dumps(data.get('metadata', {})).decode()
            )
    
    async def save_market_data_batch(self, data_points: List[Dict[str, Any]]) -> None:
        """Save multiple market data points efficiently."""
        async with self.acquire() as conn:
            # Prepare data for batch insert
            records = [
                (
                    d['time'], d['token_address'], d['chain'], d['price'],
                    d.get('volume_24h'), d.get('volume_5m'), d.get('liquidity_usd'),
                    d.get('market_cap'), d.get('holders'), d.get('buy_count_5m'),
                    d.get('sell_count_5m'), d.get('unique_buyers_5m'),
                    d.get('unique_sellers_5m'), d.get('price_change_5m'),
                    d.get('price_change_1h'), d.get('price_change_24h'),
                    orjson.dumps(d.get('metadata', {})).decode()
                )
                for d in data_points
            ]
            
            await conn.copy_records_to_table(
                'market_data',
                records=records,
                columns=[
                    'time', 'token_address', 'chain', 'price', 'volume_24h',
                    'volume_5m', 'liquidity_usd', 'market_cap', 'holders',
                    'buy_count_5m', 'sell_count_5m', 'unique_buyers_5m',
                    'unique_sellers_5m', 'price_change_5m', 'price_change_1h',
                    'price_change_24h', 'metadata'
                ]
            )
            
            logger.debug(f"Saved {len(data_points)} market data points")
    
    # Fix get_historical_data signature (line ~433)
    # REPLACE the existing method with this simplified version:

    async def get_historical_data(
        self,
        token: str,
        timeframe: str = '1h'
    ) -> List[Dict[str, Any]]:
        """
        Get historical market data for a token.
        
        Args:
            token: Token address
            timeframe: Time interval (1m, 5m, 15m, 30m, 1h, 4h, 1d)
            
        Returns:
            List of OHLCV data points
        """
        async with self.acquire() as conn:
            # Determine time bucket based on timeframe
            time_buckets = {
                '1m': '1 minute',
                '5m': '5 minutes',
                '15m': '15 minutes',
                '30m': '30 minutes',
                '1h': '1 hour',
                '4h': '4 hours',
                '1d': '1 day'
            }
            bucket = time_buckets.get(timeframe, '1 hour')
            
            # Default to ethereum chain and 500 limit for API compatibility
            chain = 'ethereum'
            limit = 500
            
            rows = await conn.fetch("""
                SELECT 
                    time_bucket($1::interval, time) AS bucket,
                    first(price, time) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price, time) AS close,
                    avg(volume_5m) AS volume,
                    avg(liquidity_usd) AS liquidity,
                    avg(holders) AS holders,
                    last(price_change_5m, time) AS price_change_5m,
                    last(price_change_1h, time) AS price_change_1h,
                    last(price_change_24h, time) AS price_change_24h
                FROM market_data
                WHERE token_address = $2 AND chain = $3
                    AND time > NOW() - INTERVAL '30 days'
                GROUP BY bucket
                ORDER BY bucket DESC
                LIMIT $4
            """, bucket, token, chain, limit)
            
            return [dict(row) for row in rows]

    # For backward compatibility, keep the original method with a different name:
    async def get_historical_data_extended(
        self,
        token_address: str,
        timeframe: str = '1h',
        limit: int = 500,
        chain: str = 'ethereum'
    ) -> List[Dict[str, Any]]:
        """
        Get historical market data with extended parameters (internal use).
        
        Args:
            token_address: Token contract address
            timeframe: Time interval
            limit: Maximum number of data points
            chain: Blockchain network
            
        Returns:
            List of OHLCV data points
        """
        # Original implementation remains the same
        async with self.acquire() as conn:
            time_buckets = {
                '1m': '1 minute',
                '5m': '5 minutes',
                '15m': '15 minutes',
                '30m': '30 minutes',
                '1h': '1 hour',
                '4h': '4 hours',
                '1d': '1 day'
            }
            bucket = time_buckets.get(timeframe, '1 hour')
            
            rows = await conn.fetch("""
                SELECT 
                    time_bucket($1::interval, time) AS bucket,
                    first(price, time) AS open,
                    max(price) AS high,
                    min(price) AS low,
                    last(price, time) AS close,
                    avg(volume_5m) AS volume,
                    avg(liquidity_usd) AS liquidity,
                    avg(holders) AS holders,
                    last(price_change_5m, time) AS price_change_5m,
                    last(price_change_1h, time) AS price_change_1h,
                    last(price_change_24h, time) AS price_change_24h
                FROM market_data
                WHERE token_address = $2 AND chain = $3
                    AND time > NOW() - INTERVAL '30 days'
                GROUP BY bucket
                ORDER BY bucket DESC
                LIMIT $4
            """, bucket, token_address, chain, limit)
            
            return [dict(row) for row in rows]
    
    async def get_active_positions(self) -> List[Dict[str, Any]]:
        """Get all active trading positions."""
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM positions
                WHERE status = 'open'
                ORDER BY opened_at DESC
            """)
            
            return [dict(row) for row in rows]
    
    async def get_recent_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent trades."""
        if not self.pool or not self.is_connected:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT *
                FROM trades
                ORDER BY entry_timestamp DESC
                LIMIT $1
            """, limit)
            
            return [dict(row) for row in rows]

    async def get_closed_trades(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get closed trades (positions that have been exited)."""
        if not self.pool or not self.is_connected:
            return []
        
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT *
                FROM trades
                WHERE status = 'closed' 
                AND side = 'buy'
                AND exit_timestamp IS NOT NULL
                ORDER BY exit_timestamp DESC
                LIMIT $1
            """, limit)
            
            return [dict(row) for row in rows]
    
    async def save_token_analysis(self, analysis: Dict[str, Any]) -> None:
        """Save token analysis results."""
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO token_analysis (
                    token_address, chain, analysis_timestamp, risk_score,
                    honeypot_risk, rug_probability, pump_probability,
                    liquidity_score, holder_score, contract_score,
                    developer_score, social_score, technical_score,
                    volume_score, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15)
                ON CONFLICT (token_address, chain, analysis_timestamp) DO UPDATE
                SET risk_score = EXCLUDED.risk_score,
                    honeypot_risk = EXCLUDED.honeypot_risk,
                    rug_probability = EXCLUDED.rug_probability,
                    pump_probability = EXCLUDED.pump_probability,
                    liquidity_score = EXCLUDED.liquidity_score,
                    holder_score = EXCLUDED.holder_score,
                    contract_score = EXCLUDED.contract_score,
                    developer_score = EXCLUDED.developer_score,
                    social_score = EXCLUDED.social_score,
                    technical_score = EXCLUDED.technical_score,
                    volume_score = EXCLUDED.volume_score,
                    metadata = EXCLUDED.metadata
            """,
                analysis['token_address'], analysis['chain'],
                analysis['analysis_timestamp'], analysis.get('risk_score'),
                analysis.get('honeypot_risk'), analysis.get('rug_probability'),
                analysis.get('pump_probability'), analysis.get('liquidity_score'),
                analysis.get('holder_score'), analysis.get('contract_score'),
                analysis.get('developer_score'), analysis.get('social_score'),
                analysis.get('technical_score'), analysis.get('volume_score'),
                orjson.dumps(analysis.get('metadata', {})).decode()
            )
    
    async def get_token_analysis(
        self,
        token_address: str,
        chain: str = 'ethereum',
        hours_back: int = 24
    ) -> List[Dict[str, Any]]:
        """Get recent token analysis results."""
        async with self.acquire() as conn:
            rows = await conn.fetch("""
                SELECT * FROM token_analysis
                WHERE token_address = $1 AND chain = $2
                    AND analysis_timestamp > NOW() - INTERVAL '%s hours'
                ORDER BY analysis_timestamp DESC
            """ % hours_back, token_address, chain)
            
            return [dict(row) for row in rows]
    
    async def save_performance_metrics(self, metrics: Dict[str, Any]) -> None:
        """Save performance metrics snapshot."""
        async with self.acquire() as conn:
            await conn.execute("""
                INSERT INTO performance_metrics (
                    period, start_date, end_date, total_trades,
                    winning_trades, losing_trades, total_pnl,
                    total_pnl_percentage, sharpe_ratio, sortino_ratio,
                    calmar_ratio, max_drawdown, win_rate, avg_win,
                    avg_loss, best_trade, worst_trade, var_95,
                    cvar_95, metadata
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10,
                         $11, $12, $13, $14, $15, $16, $17, $18, $19, $20)
            """,
                metrics['period'], metrics['start_date'], metrics['end_date'],
                metrics['total_trades'], metrics['winning_trades'],
                metrics['losing_trades'], metrics['total_pnl'],
                metrics['total_pnl_percentage'], metrics.get('sharpe_ratio'),
                metrics.get('sortino_ratio'), metrics.get('calmar_ratio'),
                metrics.get('max_drawdown'), metrics['win_rate'],
                metrics.get('avg_win'), metrics.get('avg_loss'),
                metrics.get('best_trade'), metrics.get('worst_trade'),
                metrics.get('var_95'), metrics.get('cvar_95'),
                orjson.dumps(metrics.get('metadata', {})).decode()
            )
    
    async def cleanup_old_data(self, days: int = 90) -> None:
        """Clean up old data to manage storage."""
        async with self.acquire() as conn:
            # Clean old market data (keep aggregated data)
            await conn.execute("""
                DELETE FROM market_data
                WHERE time < NOW() - INTERVAL '%s days'
                AND time NOT IN (
                    SELECT time_bucket('1 hour', time) 
                    FROM market_data 
                    WHERE time < NOW() - INTERVAL '%s days'
                )
            """ % (days, days))
            
            # Archive old closed trades
            await conn.execute("""
                DELETE FROM trades
                WHERE status = 'closed'
                AND exit_timestamp < NOW() - INTERVAL '%s days'
            """ % (days * 2))
            
            logger.info(f"Cleaned up data older than {days} days")
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics."""
        async with self.acquire() as conn:
            stats = {}
            
            # Table sizes
            tables = ['trades', 'positions', 'market_data', 'alerts', 'token_analysis']
            for table in tables:
                result = await conn.fetchrow(f"SELECT COUNT(*) as count FROM {table}")
                stats[f"{table}_count"] = result['count']
            
            # Active positions
            result = await conn.fetchrow(
                "SELECT COUNT(*) as count FROM positions WHERE status = 'open'"
            )
            stats['active_positions'] = result['count']
            
            # Today's trades
            result = await conn.fetchrow("""
                SELECT COUNT(*) as count FROM trades 
                WHERE entry_timestamp > NOW() - INTERVAL '24 hours'
            """)
            stats['trades_24h'] = result['count']
            
            # Database size
            result = await conn.fetchrow("""
                SELECT pg_database_size(current_database()) as size
            """)
            stats['db_size_mb'] = result['size'] / (1024 * 1024)
            
            return stats

    async def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary from database"""
        try:
            # ✅ FIX: Use self.pool instead of self.db
            if not self.pool or not self.is_connected:
                return {'error': 'Database not connected'}
            
            # ✅ FIX: Use self.pool.acquire() instead of self.db.acquire()
            async with self.pool.acquire() as conn:
                # Get all closed trades
                trades = await conn.fetch("""
                    SELECT 
                        COUNT(*) as total_trades,
                        COUNT(*) FILTER (WHERE profit_loss > 0) as winning_trades,
                        COUNT(*) FILTER (WHERE profit_loss <= 0) as losing_trades,
                        SUM(profit_loss) as total_pnl,
                        AVG(profit_loss) as avg_pnl,
                        AVG(profit_loss) FILTER (WHERE profit_loss > 0) as avg_win,
                        AVG(profit_loss) FILTER (WHERE profit_loss <= 0) as avg_loss,
                        MAX(profit_loss) as best_trade,
                        MIN(profit_loss) as worst_trade
                    FROM trades
                    WHERE status = 'closed' AND side = 'buy'
                """)
                
                row = trades[0] if trades else None
                if not row:
                    return {
                        'total_trades': 0,
                        'winning_trades': 0,
                        'losing_trades': 0,
                        'total_pnl': 0,
                        'win_rate': 0,
                        'message': 'No closed trades yet'
                    }
                
                total = row['total_trades'] or 0
                wins = row['winning_trades'] or 0
                losses = row['losing_trades'] or 0
                
                return {
                    'total_trades': total,
                    'winning_trades': wins,
                    'losing_trades': losses,
                    'win_rate': (wins / total * 100) if total > 0 else 0,
                    'total_pnl': float(row['total_pnl'] or 0),
                    'avg_pnl': float(row['avg_pnl'] or 0),
                    'avg_win': float(row['avg_win'] or 0),
                    'avg_loss': float(row['avg_loss'] or 0),
                    'best_trade': float(row['best_trade'] or 0),
                    'worst_trade': float(row['worst_trade'] or 0),
                    'active_positions': len(self.active_positions) if hasattr(self, 'active_positions') else 0,
                    'positions_on_cooldown': len(self.recently_closed) if hasattr(self, 'recently_closed') else 0
                }
        except Exception as e:
            logger.error(f"Error getting performance summary: {e}")
            return {'error': str(e)}

    async def find_trade_by_token(self, token_address: str) -> Optional[int]:
        """
        Find most recent trade ID by token address
        
        Args:
            token_address: Token contract address
            
        Returns:
            Trade ID if found, None otherwise
        """
        try:
            query = """
                SELECT id FROM trades 
                WHERE token_address = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            result = await self.pool.fetchval(query, token_address)
            
            if result:
                logger.debug(f"Found trade {result} for token {token_address}")
            else:
                logger.warning(f"No trade found for token {token_address}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error finding trade by token {token_address}: {e}")
            return None

    async def find_position_by_token(self, token_address: str) -> Optional[int]:
        """
        Find most recent position ID by token address
        
        Args:
            token_address: Token contract address
            
        Returns:
            Position ID if found, None otherwise
        """
        try:
            query = """
                SELECT id FROM positions 
                WHERE token_address = $1 
                ORDER BY created_at DESC 
                LIMIT 1
            """
            result = await self.pool.fetchval(query, token_address)
            
            if result:
                logger.debug(f"Found position {result} for token {token_address}")
            else:
                logger.warning(f"No position found for token {token_address}")
                
            return result
            
        except Exception as e:
            logger.error(f"Error finding position by token {token_address}: {e}")
            return None