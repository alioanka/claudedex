"""
Database Migration Script
Updates the precision of numeric columns in the database to prevent overflow errors.
This script is designed to be resilient to schema differences and will only
alter columns that exist in the target database.
"""
import asyncio
import asyncpg
import os
from typing import List, Optional

async def build_alter_query(conn: asyncpg.Connection, table_name: str, columns: List[str]) -> Optional[str]:
    """
    Builds a resilient ALTER TABLE query that only includes columns that exist.
    """
    alter_parts = []
    for column in columns:
        column_exists = await conn.fetchval("""
            SELECT EXISTS (
                SELECT 1
                FROM information_schema.columns
                WHERE table_name = $1 AND column_name = $2
            );
        """, table_name, column)

        if column_exists:
            alter_parts.append(f"ALTER COLUMN {column} TYPE NUMERIC(40, 18) USING {column}::numeric(40,18)")
        else:
            print(f"ℹ️  '{column}' column not found in '{table_name}' table, skipping alteration.")

    if not alter_parts:
        return None

    return f"ALTER TABLE {table_name} " + ",\n".join(alter_parts) + ";"

async def migrate_database():
    """Applies database schema migrations."""

    db_url = os.getenv("DATABASE_URL", "postgresql://bot_user:bot_password@postgres:5432/tradingbot")

    conn = await asyncpg.connect(db_url)

    try:
        print("🚀 Starting database migration...")

        # Define tables and columns to migrate
        tables_to_migrate = {
            "trades": ["entry_price", "exit_price", "amount", "usd_value", "gas_fee", "profit_loss"],
            "positions": ["entry_price", "current_price", "amount", "stop_loss", "take_profit", "pnl"],
            "market_data": ["price", "volume", "liquidity", "market_cap"],
            "whale_activities": ["amount", "value_usd"]
        }

        for table, columns in tables_to_migrate.items():
            print(f"\nChecking '{table}' table...")
            alter_query = await build_alter_query(conn, table, columns)
            if alter_query:
                print(f"Updating '{table}' table...")
                await conn.execute(alter_query)
            else:
                print(f"No columns to update in '{table}' table.")

        print("\n✅ Database migration completed successfully!")

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
