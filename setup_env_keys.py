#!/usr/bin/env python3
"""
Setup script to generate ENCRYPTION_KEY and encrypt PRIVATE_KEY for .env file
"""

import os
import sys
from pathlib import Path
from cryptography.fernet import Fernet
import secrets

def generate_encryption_key():
    """Generate a new encryption key"""
    return Fernet.generate_key().decode()

def generate_jwt_secret():
    """Generate a JWT secret key"""
    return secrets.token_urlsafe(32)

def encrypt_private_key(private_key: str, encryption_key: str) -> str:
    """Encrypt a private key using the provided encryption key"""
    from cryptography.fernet import Fernet
    
    # Create Fernet instance with the encryption key
    fernet = Fernet(encryption_key.encode())
    
    # Encrypt the private key
    encrypted_key = fernet.encrypt(private_key.encode())
    
    return encrypted_key.decode()

def main():
    print("🔐 ClaudeDex Environment Key Setup")
    print("=" * 40)
    
    # Generate encryption key
    encryption_key = generate_encryption_key()
    print(f"✅ Generated ENCRYPTION_KEY: {encryption_key}")
    
    # Generate JWT secret
    jwt_secret = generate_jwt_secret()
    print(f"✅ Generated JWT_SECRET: {jwt_secret}")
    
    # Get private key from user
    print("\n📝 Please enter your wallet private key (64 hex characters, no 0x prefix):")
    private_key = input("Private Key: ").strip()
    
    if not private_key or len(private_key) != 64:
        print("❌ Invalid private key format. Must be 64 hexadecimal characters.")
        return
    
    # Encrypt private key
    try:
        encrypted_private_key = encrypt_private_key(private_key, encryption_key)
        print(f"✅ Encrypted PRIVATE_KEY: {encrypted_private_key}")
    except Exception as e:
        print(f"❌ Error encrypting private key: {e}")
        return
    
    # Generate .env content
    env_content = f"""# Environment
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://trading:trading123@localhost:5432/tradingbot

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys (replace with your actual keys)
DEXSCREENER_API_KEY=your_dexscreener_api_key_here
ETHERSCAN_API_KEY=your_etherscan_api_key_here
COINGECKO_API_KEY=your_coingecko_api_key_here

# Security
JWT_SECRET_KEY={jwt_secret}
ENCRYPTION_KEY={encryption_key}
PRIVATE_KEY={encrypted_private_key}

# Blockchain RPC
ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your_alchemy_key
BSC_RPC_URL=https://bsc-dataseed.binance.org/

# Monitoring
TELEGRAM_BOT_TOKEN=your_telegram_bot_token_here
TELEGRAM_CHAT_ID=your_telegram_chat_id_here
DISCORD_WEBHOOK_URL=your_discord_webhook_url_here

# Trading
MAX_POSITION_SIZE=1000
MIN_LIQUIDITY=50000
"""
    
    # Write .env file
    with open('.env', 'w') as f:
        f.write(env_content)
    
    print(f"\n✅ Created .env file with encrypted keys")
    print(f"🔒 Your private key has been encrypted and stored securely")
    print(f"⚠️  Remember to add .env to your .gitignore file!")
    
    # Create .gitignore if it doesn't exist
    gitignore_content = """# Environment variables
.env
.env.local
.env.production

# Encryption keys
.encryption_key
*.key

# Database
*.db
*.sqlite3

# Logs
logs/
*.log

# Python
__pycache__/
*.pyc
*.pyo
*.pyd
.Python
env/
venv/
.venv/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
"""
    
    if not os.path.exists('.gitignore'):
        with open('.gitignore', 'w') as f:
            f.write(gitignore_content)
        print(f"✅ Created .gitignore file")

if __name__ == "__main__":
    main()
