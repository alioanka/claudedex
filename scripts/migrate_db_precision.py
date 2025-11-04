"""
Database Migration Script
Updates the precision of numeric columns in the database to prevent overflow errors.
"""
import asyncio
import asyncpg
import os

async def migrate_database():
    """Applies database schema migrations."""

    # Database configuration from environment variables
    db_url = os.getenv("DATABASE_URL", "postgresql://bot_user:bot_password@postgres:5432/tradingbot")

    conn = await asyncpg.connect(db_url)

    try:
        print("🚀 Starting database migration...")

        # Alter 'trades' table
        print("Updating 'trades' table...")
        await conn.execute("""
            ALTER TABLE trades
            ALTER COLUMN entry_price TYPE NUMERIC(40, 18) USING entry_price::numeric(40,18),
            ALTER COLUMN exit_price TYPE NUMERIC(40, 18) USING exit_price::numeric(40,18),
            ALTER COLUMN amount TYPE NUMERIC(40, 18) USING amount::numeric(40,18),
            ALTER COLUMN usd_value TYPE NUMERIC(40, 18) USING usd_value::numeric(40,18),
            ALTER COLUMN gas_fee TYPE NUMERIC(40, 18) USING gas_fee::numeric(40,18),
            ALTER COLUMN profit_loss TYPE NUMERIC(40, 18) USING profit_loss::numeric(40,18);
        """)

        # Alter 'positions' table
        print("Updating 'positions' table...")
        await conn.execute("""
            ALTER TABLE positions
            ALTER COLUMN entry_price TYPE NUMERIC(40, 18) USING entry_price::numeric(40,18),
            ALTER COLUMN current_price TYPE NUMERIC(40, 18) USING current_price::numeric(40,18),
            ALTER COLUMN amount TYPE NUMERIC(40, 18) USING amount::numeric(40,18),
            ALTER COLUMN stop_loss TYPE NUMERIC(40, 18) USING stop_loss::numeric(40,18),
            ALTER COLUMN take_profit TYPE NUMERIC(40, 18) USING take_profit::numeric(40,18),
            ALTER COLUMN pnl TYPE NUMERIC(40, 18) USING pnl::numeric(40,18);
        """)

        # Alter 'market_data' table
        print("Updating 'market_data' table...")
        await conn.execute("""
            ALTER TABLE market_data
            ALTER COLUMN price TYPE NUMERIC(40, 18) USING price::numeric(40,18),
            ALTER COLUMN volume TYPE NUMERIC(40, 18) USING volume::numeric(40,18),
            ALTER COLUMN liquidity TYPE NUMERIC(40, 18) USING liquidity::numeric(40,18),
            ALTER COLUMN market_cap TYPE NUMERIC(40, 18) USING market_cap::numeric(40,18);
        """)

        # Alter 'whale_activities' table
        print("Updating 'whale_activities' table...")
        await conn.execute("""
            ALTER TABLE whale_activities
            ALTER COLUMN amount TYPE NUMERIC(40, 18) USING amount::numeric(40,18),
            ALTER COLUMN value_usd TYPE NUMERIC(40, 18) USING value_usd::numeric(40,18);
        """)

        print("✅ Database migration completed successfully!")

    except Exception as e:
        print(f"❌ Database migration failed: {e}")
        raise
    finally:
        await conn.close()

if __name__ == "__main__":
    print("This script will alter your database tables to update numeric precision.")
    print("It is recommended to back up your database before proceeding.")
    response = input("Do you want to continue? (y/n): ")
    if response.lower() == 'y':
        asyncio.run(migrate_database())
    else:
        print("Migration cancelled.")
