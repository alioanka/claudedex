#!/bin/bash
#
# Setup Encryption Key Security
# This script ensures your encryption key is properly secured
#

set -e

echo "========================================"
echo "  Encryption Key Security Setup"
echo "========================================"
echo ""

KEY_FILE=".encryption_key"
ENV_FILE=".env"

# Check if .encryption_key already exists
if [ -f "$KEY_FILE" ]; then
    echo "✅ $KEY_FILE already exists"
    KEY=$(cat "$KEY_FILE")
    echo "   Key ends with: ...${KEY: -8}"

    # Validate it's a proper Fernet key (44 chars, ends with =)
    if [ ${#KEY} -eq 44 ] && [[ "$KEY" == *= ]]; then
        echo "✅ Key format is valid (44 chars, ends with =)"
    else
        echo "⚠️  Key format may be invalid. Fernet keys should be 44 chars ending with ="
        echo "   Current length: ${#KEY}"
    fi
else
    echo "❌ $KEY_FILE does not exist"
    echo ""

    # Try to extract from .env
    if grep -q "^ENCRYPTION_KEY=" "$ENV_FILE" 2>/dev/null; then
        echo "📥 Found ENCRYPTION_KEY in .env, extracting..."
        grep "^ENCRYPTION_KEY=" "$ENV_FILE" | sed 's/^ENCRYPTION_KEY=//' > "$KEY_FILE"
        echo "✅ Created $KEY_FILE from .env"
    else
        echo "🔑 Generating new Fernet encryption key..."
        python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > "$KEY_FILE"
        echo "✅ Generated new encryption key"
        echo ""
        echo "⚠️  WARNING: If you have existing encrypted data, it will NOT be readable!"
        echo "   You need to re-import credentials from .env after this."
    fi
fi

# Set correct permissions
echo ""
echo "🔒 Setting file permissions to 600 (owner read/write only)..."
chmod 600 "$KEY_FILE"
echo "✅ Permissions set"

# Show current permissions
echo ""
echo "📋 Current file status:"
ls -la "$KEY_FILE"

# Check if in .gitignore
echo ""
if grep -q "\.encryption_key" .gitignore 2>/dev/null; then
    echo "✅ .encryption_key is in .gitignore"
else
    echo "⚠️  Adding .encryption_key to .gitignore..."
    echo ".encryption_key" >> .gitignore
    echo "✅ Added to .gitignore"
fi

# Check if tracked by git
echo ""
if git ls-files --error-unmatch "$KEY_FILE" 2>/dev/null; then
    echo "❌ WARNING: $KEY_FILE is tracked by git!"
    echo "   Run: git rm --cached $KEY_FILE"
else
    echo "✅ .encryption_key is NOT tracked by git"
fi

# Check .env for ENCRYPTION_KEY
echo ""
if grep -q "^ENCRYPTION_KEY=" "$ENV_FILE" 2>/dev/null; then
    echo "⚠️  ENCRYPTION_KEY still exists in .env"
    echo "   For security, remove it with: sed -i '/^ENCRYPTION_KEY=/d' .env"
else
    echo "✅ ENCRYPTION_KEY not in .env (good!)"
fi

echo ""
echo "========================================"
echo "  Setup Complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo "1. Rebuild: docker-compose down && docker-compose up -d --build"
echo "2. Re-import: docker exec -it trading-bot python scripts/force_reimport_credentials.py"
echo ""
