#!/bin/sh
# Redis Entrypoint Script with Password Decryption
# Decrypts REDIS_PASSWORD using ENCRYPTION_KEY (NO prefix required)

set -e

echo "🔐 Redis Entrypoint: Processing password..."
echo "   REDIS_PASSWORD length: ${#REDIS_PASSWORD}"
echo "   ENCRYPTION_KEY length: ${#ENCRYPTION_KEY}"

# Check if REDIS_PASSWORD and ENCRYPTION_KEY are set
if [ -n "$REDIS_PASSWORD" ] && [ -n "$ENCRYPTION_KEY" ]; then
    echo "   Attempting decryption..."
    # Try to decrypt the password (works with or without 'encrypted:' prefix)
    DECRYPTED_PASSWORD=$(python3 2>&1 <<'EOF'
import os
import sys
from cryptography.fernet import Fernet
import base64

try:
    encryption_key = os.getenv('ENCRYPTION_KEY')
    redis_password = os.getenv('REDIS_PASSWORD')

    # Remove 'encrypted:' prefix if present (for compatibility)
    if redis_password.startswith('encrypted:'):
        redis_password = redis_password[10:]

    # Try to decrypt using Fernet
    fernet = Fernet(encryption_key.encode())
    encrypted_bytes = base64.b64decode(redis_password.encode())
    decrypted = fernet.decrypt(encrypted_bytes).decode()

    print(decrypted, end='')
    sys.exit(0)

except Exception as e:
    # Decryption failed - use password as-is (plaintext)
    print(redis_password, end='')
    sys.exit(1)
EOF
)

    EXIT_CODE=$?
    if [ $EXIT_CODE -eq 0 ]; then
        echo "✅ Password decrypted successfully"
        REDIS_PASS="$DECRYPTED_PASSWORD"
    else
        echo "⚠️  Decryption failed with exit code: $EXIT_CODE"
        echo "   Python output: $DECRYPTED_PASSWORD"
        echo "   Using password as plaintext"
        REDIS_PASS="$REDIS_PASSWORD"
    fi
else
    echo "⚠️  REDIS_PASSWORD or ENCRYPTION_KEY not set"
    echo "   Using default password"
    REDIS_PASS="${REDIS_PASSWORD:-change_me_in_production}"
fi

# Start Redis with the password
echo "🚀 Starting Redis server..."
exec redis-server --appendonly yes --requirepass "$REDIS_PASS"
