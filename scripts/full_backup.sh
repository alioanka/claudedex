#!/bin/bash
# Full system backup script

set -e

echo "🔒 FULL SYSTEM BACKUP"
echo "======================================"

# Configuration
BACKUP_DIR="backups/$(date +%Y%m%d_%H%M%S)"
DB_NAME="${DB_NAME:-tradingbot}"
DB_USER="${DB_USER:-trading}"
DB_HOST="${DB_HOST:-localhost}"
DB_PORT="${DB_PORT:-5432}"

# Create backup directory
mkdir -p "$BACKUP_DIR"

echo ""
echo "📁 Backup location: $BACKUP_DIR"

# Backup database
echo ""
echo "💾 Backing up database..."
PGPASSWORD=$DB_PASSWORD pg_dump \
    -h $DB_HOST \
    -p $DB_PORT \
    -U $DB_USER \
    -d $DB_NAME \
    -F c \
    -f "$BACKUP_DIR/database.dump"

if [ $? -eq 0 ]; then
    echo "   ✅ Database backup complete"
    DB_SIZE=$(du -h "$BACKUP_DIR/database.dump" | cut -f1)
    echo "   Size: $DB_SIZE"
else
    echo "   ❌ Database backup failed"
    exit 1
fi

# Backup configuration files
echo ""
echo "⚙️  Backing up configuration..."
mkdir -p "$BACKUP_DIR/config"
cp -r config/*.yaml "$BACKUP_DIR/config/" 2>/dev/null || true
cp -r config/*.json "$BACKUP_DIR/config/" 2>/dev/null || true
cp .env "$BACKUP_DIR/.env.backup" 2>/dev/null || true
echo "   ✅ Configuration backup complete"

# Backup ML models
echo ""
echo "🤖 Backing up ML models..."
mkdir -p "$BACKUP_DIR/models"
cp -r ml/models/*.pkl "$BACKUP_DIR/models/" 2>/dev/null || true
cp -r ml/models/*.h5 "$BACKUP_DIR/models/" 2>/dev/null || true
echo "   ✅ ML models backup complete"

# Backup logs (last 7 days)
echo ""
echo "📋 Backing up recent logs..."
mkdir -p "$BACKUP_DIR/logs"
find logs -name "*.log" -mtime -7 -exec cp {} "$BACKUP_DIR/logs/" \; 2>/dev/null || true
echo "   ✅ Logs backup complete"

# Backup blacklists
echo ""
echo "🚫 Backing up blacklists..."
mkdir -p "$BACKUP_DIR/blacklists"
cp -r data/blacklists/*.json "$BACKUP_DIR/blacklists/" 2>/dev/null || true
echo "   ✅ Blacklists backup complete"

# Create backup manifest
echo ""
echo "📄 Creating backup manifest..."
cat > "$BACKUP_DIR/MANIFEST.txt" << EOF
ClaudeDex Trading Bot - Full Backup
====================================
Backup Date: $(date)
Backup Location: $BACKUP_DIR

Contents:
- Database dump (PostgreSQL)
- Configuration files
- ML models
- Recent logs (7 days)
- Blacklists

Database Info:
- Name: $DB_NAME
- Size: $DB_SIZE

System Info:
- Hostname: $(hostname)
- OS: $(uname -s)
- Python: $(python --version 2>&1)

EOF
echo "   ✅ Manifest created"

# Compress backup
echo ""
echo "🗜️  Compressing backup..."
tar -czf "$BACKUP_DIR.tar.gz" -C backups "$(basename $BACKUP_DIR)"

if [ $? -eq 0 ]; then
    COMPRESSED_SIZE=$(du -h "$BACKUP_DIR.tar.gz" | cut -f1)
    echo "   ✅ Backup compressed"
    echo "   Size: $COMPRESSED_SIZE"
    
    # Remove uncompressed backup
    rm -rf "$BACKUP_DIR"
else
    echo "   ⚠️  Compression failed, keeping uncompressed backup"
fi

# Cleanup old backups (keep last 30 days)
echo ""
echo "🧹 Cleaning up old backups..."
find backups -name "*.tar.gz" -mtime +30 -delete 2>/dev/null || true
REMAINING=$(find backups -name "*.tar.gz" | wc -l)
echo "   ✅ Cleanup complete ($REMAINING backups remaining)"

echo ""
echo "======================================"
echo "✅ BACKUP COMPLETE!"
echo "Backup file: $BACKUP_DIR.tar.gz"
echo ""