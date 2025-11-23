#!/bin/bash
# Docker Entrypoint Script
# Runs database migrations before starting the application

set -e  # Exit on error

echo "========================================"
echo "  TRADING BOT STARTUP"
echo "========================================"

# Wait for database to be ready (handled by migrate_database.py retry logic)
echo ""
echo "🔄 Running database migrations..."
echo ""

# Run migrations
python scripts/migrate_database.py

# Check migration exit code
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Migrations completed successfully"
    echo ""
else
    echo ""
    echo "❌ Migration failed! Check logs above."
    echo ""
    exit 1
fi

# Start the main application
echo "========================================"
echo "  STARTING TRADING BOT"
echo "========================================"
echo ""

exec python "$@"
