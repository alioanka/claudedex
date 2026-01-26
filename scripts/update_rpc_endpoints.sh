#!/bin/bash
# Script to update RPC endpoints in the database with free public endpoints
# This fixes the expired Ankr/Alchemy API key issues

set -e

echo "=== Updating RPC Endpoints in Database ==="
echo ""

# Get the script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SQL_FILE="$SCRIPT_DIR/update_rpc_endpoints.sql"

# Check if SQL file exists
if [ ! -f "$SQL_FILE" ]; then
    echo "Error: SQL file not found at $SQL_FILE"
    exit 1
fi

# Detect Docker container name for Postgres
POSTGRES_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E 'postgres|trading.*postgres' | head -1)

if [ -z "$POSTGRES_CONTAINER" ]; then
    echo "Error: No running Postgres container found"
    echo "Make sure the database container is running"
    exit 1
fi

echo "Found Postgres container: $POSTGRES_CONTAINER"
echo ""

# Get database credentials from Docker secrets or environment
# Try to read from container's secrets first
DB_USER=$(docker exec "$POSTGRES_CONTAINER" cat /run/secrets/db_user 2>/dev/null || echo "")
if [ -z "$DB_USER" ]; then
    # Try environment variable from container
    DB_USER=$(docker exec "$POSTGRES_CONTAINER" printenv POSTGRES_USER 2>/dev/null || echo "")
fi
if [ -z "$DB_USER" ]; then
    # Default fallback
    DB_USER="${POSTGRES_USER:-bot_user}"
fi

# Get database name from environment or default
DB_NAME=$(docker exec "$POSTGRES_CONTAINER" printenv POSTGRES_DB 2>/dev/null || echo "tradingbot")

echo "Using database: $DB_NAME"
echo "Using user: $DB_USER"
echo ""
echo "Executing SQL script to update RPC endpoints..."
echo ""

# Execute the SQL script
docker exec -i "$POSTGRES_CONTAINER" psql -U "$DB_USER" -d "$DB_NAME" < "$SQL_FILE"

echo ""
echo "=== RPC Endpoints Updated Successfully ==="
echo ""
echo "Now restart the bot to use the new endpoints:"
echo "  docker-compose down"
echo "  docker-compose up -d"
