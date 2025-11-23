#!/bin/sh
# Redis Entrypoint Script with Password Decryption
# This script decrypts the REDIS_PASSWORD using ENCRYPTION_KEY before starting Redis

set -e

echo "🔐 Redis Entrypoint: Decrypting password..."

# Check if REDIS_PASSWORD is encrypted (starts with 'encrypted:' or is base64)
if [ -n "$REDIS_PASSWORD" ] && [ -n "$ENCRYPTION_KEY" ]; then
    # Use Python to decrypt the password
    DECRYPTED_PASSWORD=$(python3 <<EOF
import os
import sys
from cryptography.fernet import Fernet
import base64

try:
    encryption_key = os.getenv('ENCRYPTION_KEY')
    redis_password = os.getenv('REDIS_PASSWORD')

    # Remove 'encrypted:' prefix if present
    if redis_password.startswith('encrypted:'):
        redis_password = redis_password[10:]

    # Decrypt using Fernet
    fernet = Fernet(encryption_key.encode())
    encrypted_bytes = base64.b64decode(redis_password.encode())
    decrypted = fernet.decrypt(encrypted_bytes).decode()

    print(decrypted, end='')

except Exception as e:
    # If decryption fails, use the password as-is (might be plaintext)
    print(redis_password, end='')
    sys.stderr.write(f"Warning: Password decryption failed: {e}\n")
    sys.stderr.write("Using password as-is (might be plaintext)\n")
EOF
)

    if [ $? -eq 0 ] && [ -n "$DECRYPTED_PASSWORD" ]; then
        echo "✅ Password decrypted successfully"
        export REDIS_PASSWORD_PLAIN="$DECRYPTED_PASSWORD"
    else
        echo "⚠️  Decryption failed, using password as-is"
        export REDIS_PASSWORD_PLAIN="$REDIS_PASSWORD"
    fi
else
    echo "⚠️  No encryption key found, using password as-is"
    export REDIS_PASSWORD_PLAIN="${REDIS_PASSWORD:-change_me_in_production}"
fi

# Start Redis with decrypted password
echo "🚀 Starting Redis server..."
exec redis-server --appendonly yes --requirepass "$REDIS_PASSWORD_PLAIN"
