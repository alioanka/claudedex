#!/bin/bash
#
# Setup Docker Secrets for ClaudeDex Trading Bot
# This script creates the secrets directory and files needed by Docker
#

set -e

echo "========================================"
echo "  Docker Secrets Setup"
echo "========================================"
echo ""

SECRETS_DIR="./secrets"

# Create secrets directory
if [ ! -d "$SECRETS_DIR" ]; then
    echo "📁 Creating secrets directory..."
    mkdir -p "$SECRETS_DIR"
    chmod 700 "$SECRETS_DIR"
    echo "✅ Created $SECRETS_DIR with permissions 700"
else
    echo "✅ Secrets directory already exists"
fi

# Function to create a secret file
create_secret() {
    local name=$1
    local prompt=$2
    local file="$SECRETS_DIR/$name"

    if [ -f "$file" ]; then
        echo "⏭️  $name already exists, skipping..."
        return
    fi

    echo ""
    echo "🔐 $prompt"
    read -s -p "Enter value: " value
    echo ""

    if [ -z "$value" ]; then
        echo "⚠️  Empty value, skipping $name"
        return
    fi

    echo -n "$value" > "$file"
    chmod 600 "$file"
    echo "✅ Created $name"
}

# Create secret files
create_secret "db_user" "Database username (e.g., bot_user)"
create_secret "db_password" "Database password (use a strong password!)"
create_secret "redis_password" "Redis password (use a strong password!)"

# Set final permissions
echo ""
echo "🔒 Setting secure permissions..."
chmod 700 "$SECRETS_DIR"
chmod 600 "$SECRETS_DIR"/* 2>/dev/null || true
echo "✅ Permissions set"

# Check .gitignore
echo ""
if grep -q "secrets/" .gitignore 2>/dev/null; then
    echo "✅ secrets/ is in .gitignore"
else
    echo "⚠️  Adding secrets/ to .gitignore..."
    echo -e "\n# Docker secrets\nsecrets/" >> .gitignore
    echo "✅ Added to .gitignore"
fi

# Show summary
echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Secret files created in $SECRETS_DIR/:"
ls -la "$SECRETS_DIR/" 2>/dev/null || echo "  (directory is empty)"
echo ""
echo "⚠️  IMPORTANT: These secrets are needed to START the containers."
echo "   They are different from application secrets (API keys, wallet keys)"
echo "   which are stored encrypted in the database."
echo ""
echo "Next steps:"
echo "1. docker-compose down"
echo "2. docker-compose up -d --build"
echo ""
