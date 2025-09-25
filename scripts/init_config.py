# scripts/init_config.py
"""
Initialize configuration files
"""
import json
import os
from pathlib import Path
import secrets
from cryptography.fernet import Fernet

def init_config():
    """Initialize configuration files"""
    
    config_dir = Path("config")
    config_dir.mkdir(exist_ok=True)
    
    # Trading configuration
    trading_config = {
        "max_position_size": 1000,
        "max_slippage": 0.05,
        "min_liquidity": 50000,
        "max_daily_trades": 20,
        "strategies": {
            "momentum": {
                "enabled": True,
                "min_volume": 10000,
                "min_price_change": 0.1
            },
            "scalping": {
                "enabled": True,
                "profit_target": 0.02,
                "stop_loss": 0.01
            }
        },
        "risk_management": {
            "max_position_pct": 0.1,
            "max_daily_loss_pct": 0.2,
            "stop_loss_pct": 0.05,
            "take_profit_pct": 0.15,
            "var_confidence": 0.95
        }
    }
    
    # Security configuration
    security_config = {
        "require_2fa": True,
        "session_timeout": 3600,
        "max_login_attempts": 5,
        "api_rate_limits": {
            "default": {"requests": 100, "window": 60},
            "authenticated": {"requests": 1000, "window": 60}
        },
        "wallet_security": {
            "require_multisig": True,
            "min_signatures": 2,
            "key_rotation_days": 30
        },
        "encryption": {
            "algorithm": "AES-256-GCM",
            "key_derivation": "PBKDF2"
        }
    }
    
    # Database configuration
    database_config = {
        "host": "localhost",
        "port": 5432,
        "database": "tradingbot",
        "user": "trading",
        "pool_size": 20,
        "max_overflow": 10,
        "pool_timeout": 30,
        "echo": False
    }
    
    # API configuration
    api_config = {
        "dexscreener": {
            "base_url": "https://api.dexscreener.com",
            "rate_limit": 300
        },
        "coingecko": {
            "base_url": "https://api.coingecko.com/api/v3",
            "rate_limit": 50
        },
        "etherscan": {
            "base_url": "https://api.etherscan.io/api",
            "rate_limit": 5
        }
    }
    
    # ML configuration
    ml_config = {
        "models": {
            "rug_classifier": {
                "retrain_interval": 86400,
                "min_confidence": 0.7,
                "feature_count": 50
            },
            "pump_predictor": {
                "sequence_length": 24,
                "prediction_horizon": 6,
                "min_confidence": 0.75
            }
        },
        "ensemble": {
            "voting": "weighted",
            "weights": {
                "rug_classifier": 0.3,
                "pump_predictor": 0.4,
                "volume_validator": 0.3
            }
        }
    }
    
    # Save configurations
    configs = {
        "trading": trading_config,
        "security": security_config,
        "database": database_config,
        "api": api_config,
        "ml": ml_config
    }
    
    for name, config in configs.items():
        path = config_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"✅ Created {path}")
    
    # Generate encryption key
    encryption_key = Fernet.generate_key()
    key_file = config_dir / ".encryption_key"
    with open(key_file, "wb") as f:
        f.write(encryption_key)
    os.chmod(key_file, 0o600)  # Restrict access
    print(f"🔐 Generated encryption key")
    
    # Create .env.example
    env_example = """
# Environment
ENVIRONMENT=development

# Database
DATABASE_URL=postgresql://trading:trading123@localhost:5432/tradingbot

# Redis
REDIS_URL=redis://localhost:6379/0

# API Keys (replace with your keys)
DEXSCREENER_API_KEY=your_key_here
ETHERSCAN_API_KEY=your_key_here
COINGECKO_API_KEY=your_key_here

# Security
JWT_SECRET_KEY=your_secret_key_here
ENCRYPTION_KEY=your_encryption_key_here

# Blockchain RPC
ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your_key
BSC_RPC_URL=https://bsc-dataseed.binance.org/

# Monitoring
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here
DISCORD_WEBHOOK_URL=your_webhook_here

# Trading
MAX_POSITION_SIZE=1000
MIN_LIQUIDITY=50000
"""
    
    with open(".env.example", "w") as f:
        f.write(env_example.strip())
    print("✅ Created .env.example")
    
    print("\n🎉 Configuration initialization complete!")
    print("📝 Please edit the configuration files in the 'config' directory")
    print("🔑 Copy .env.example to .env and add your API keys")

if __name__ == "__main__":
    init_config()
