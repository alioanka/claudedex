# 🚀 ClaudeDex Trading Bot - Complete Deployment & User Guide

## Table of Contents
1. [Prerequisites & System Requirements](#prerequisites--system-requirements)
2. [Environment Setup (.env Configuration)](#environment-setup-env-configuration)
3. [Security Setup (Wallet & Encryption)](#security-setup-wallet--encryption)
4. [Database Installation](#database-installation)
5. [Paper Trading Mode Setup](#paper-trading-mode-setup)
6. [Production Deployment](#production-deployment)
7. [Monitoring & Dashboard](#monitoring--dashboard)
8. [Troubleshooting Guide](#troubleshooting-guide)
9. [Maintenance & Updates](#maintenance--updates)

---

## Prerequisites & System Requirements

### Minimum System Requirements
```yaml
CPU: 4 cores (8 cores recommended)
RAM: 8GB minimum (16GB recommended for ML models)
Storage: 100GB SSD (500GB recommended)
Network: 50Mbps stable connection
OS: Ubuntu 20.04+ / Windows 10+ / macOS 11+
```

### Required Software
```bash
# Core Requirements
Python: 3.9-3.11 (3.11 recommended)
PostgreSQL: 14+ with TimescaleDB extension
Redis: 7.0+
Git: 2.30+
Node.js: 18+ (for dashboard)

# Optional (for Docker deployment)
Docker: 20.10+
Docker Compose: 2.0+
```

### Required API Keys
```yaml
Critical (Must Have):
- Ethereum RPC endpoint (Infura/Alchemy/QuickNode)
- DexScreener API key (free tier available)
- Telegram Bot Token (for alerts)

Important (Highly Recommended):
- GoPlus Security API key (for safety checks)
- TokenSniffer API key (additional security)
- Etherscan API key (for contract verification)

Optional (Enhanced Features):
- Twitter API (sentiment analysis)
- Discord Webhook (additional alerts)
- 1inch API key (better routing)
```

---

## Environment Setup (.env Configuration)

### Step 1: Clone Repository
```bash
# Clone the repository
git clone https://github.com/your-repo/claudedex-bot.git
cd claudedex-bot

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Create .env File
```bash
# Copy example environment file
cp .env.example .env

# Open in editor
nano .env  # or use your preferred editor
```

### Step 3: Complete .env Configuration
```env
# ============================================
# NETWORK CONFIGURATION
# ============================================

# Ethereum Mainnet RPC (Get from Infura/Alchemy)
ETH_RPC_URL=https://mainnet.infura.io/v3/YOUR_PROJECT_ID
ETH_CHAIN_ID=1

# Backup RPC endpoints (for failover)
ETH_RPC_BACKUP_1=https://eth-mainnet.g.alchemy.com/v2/YOUR_API_KEY
ETH_RPC_BACKUP_2=https://rpc.ankr.com/eth

# BSC Configuration (if trading on BSC)
BSC_RPC_URL=https://bsc-dataseed1.binance.org
BSC_CHAIN_ID=56

# Polygon Configuration (if trading on Polygon)
POLYGON_RPC_URL=https://polygon-rpc.com
POLYGON_CHAIN_ID=137

# ============================================
# DATABASE CONFIGURATION
# ============================================

# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=claudedex_trading
DB_USER=claudedex
DB_PASSWORD=CHANGE_THIS_SECURE_PASSWORD_123!

# Redis Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379
REDIS_PASSWORD=CHANGE_THIS_REDIS_PASSWORD_456!
REDIS_DB=0

# ============================================
# API KEYS
# ============================================

# DexScreener API (Required - Get from dexscreener.com/api)
DEXSCREENER_API_KEY=your_dexscreener_api_key_here

# Security APIs (Highly Recommended)
GOPLUS_API_KEY=your_goplus_api_key_here
TOKENSNIFFER_API_KEY=your_tokensniffer_api_key_here
HONEYPOT_IS_API_KEY=your_honeypot_api_key_here

# Blockchain Explorer APIs
ETHERSCAN_API_KEY=your_etherscan_api_key_here
BSCSCAN_API_KEY=your_bscscan_api_key_here
POLYGONSCAN_API_KEY=your_polygonscan_api_key_here

# DEX Aggregator APIs (Optional but recommended)
ONEINCH_API_KEY=your_1inch_api_key_here
PARASWAP_API_KEY=your_paraswap_api_key_here

# Social APIs (Optional)
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_BEARER_TOKEN=your_twitter_bearer_token

# ============================================
# WALLET CONFIGURATION (CRITICAL SECURITY)
# ============================================

# NEVER store plain private keys! Always encrypted!
# See Security Setup section for encryption instructions
WALLET_PRIVATE_KEY_ENCRYPTED=your_encrypted_private_key_here
WALLET_ENCRYPTION_KEY=your_32_character_encryption_key_here

# Optional: Hardware wallet config
USE_HARDWARE_WALLET=false
HARDWARE_WALLET_TYPE=ledger  # ledger or trezor

# ============================================
# NOTIFICATION SERVICES
# ============================================

# Telegram Alerts (Required for notifications)
# How to get: Create bot via @BotFather on Telegram
TELEGRAM_BOT_TOKEN=1234567890:ABCdefGHIjklMNOpqrsTUVwxyz
TELEGRAM_CHAT_ID=your_chat_id_here

# Discord Alerts (Optional)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/xxx/yyy

# Email Alerts (Optional)
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
EMAIL_FROM=your_bot@gmail.com
EMAIL_TO=your_email@gmail.com
EMAIL_PASSWORD=your_app_specific_password

# ============================================
# TRADING CONFIGURATION
# ============================================

# Risk Management (CRITICAL - Start Conservative!)
MAX_POSITION_SIZE_PERCENT=0.02  # Max 2% of portfolio per trade
DEFAULT_STOP_LOSS_PERCENT=0.05  # 5% stop loss
DEFAULT_TAKE_PROFIT_PERCENT=0.15  # 15% take profit
MAX_DAILY_LOSS_PERCENT=0.10  # Stop trading after 10% daily loss
MAX_CONCURRENT_POSITIONS=3  # Maximum open positions

# Position Sizing
MIN_POSITION_SIZE_USD=100  # Minimum trade size
MAX_POSITION_SIZE_USD=5000  # Maximum trade size
USE_KELLY_CRITERION=false  # Advanced position sizing

# Token Filtering
MIN_LIQUIDITY_USD=50000  # Minimum liquidity required
MIN_HOLDERS=100  # Minimum number of holders
MIN_VOLUME_24H_USD=10000  # Minimum 24h volume
MAX_PRICE_IMPACT_PERCENT=0.02  # Max acceptable slippage

# Timing Configuration
ANALYSIS_COOLDOWN_SECONDS=60  # Wait between analysis
ORDER_TIMEOUT_SECONDS=30  # Order execution timeout
POSITION_CHECK_INTERVAL=300  # Check positions every 5 min

# ============================================
# SECURITY SETTINGS
# ============================================

# Authentication
JWT_SECRET=CHANGE_THIS_TO_RANDOM_32_CHAR_STRING_789xyz
SESSION_TIMEOUT_MINUTES=60

# API Security
API_RATE_LIMIT_PER_MINUTE=60
ENABLE_API_KEY_AUTH=true
API_KEY=GENERATE_SECURE_API_KEY_HERE

# Wallet Security
REQUIRE_TRANSACTION_CONFIRMATION=true
MAX_GAS_PRICE_GWEI=100
ENABLE_FLASHBOTS=true  # MEV protection
FLASHBOTS_RPC=https://rpc.flashbots.net

# ============================================
# MONITORING & DASHBOARD
# ============================================

# Dashboard Configuration
DASHBOARD_ENABLED=true
DASHBOARD_PORT=8080
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PASSWORD=CHANGE_THIS_DASHBOARD_PASSWORD

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
ENABLE_HEALTH_CHECK=true
HEALTH_CHECK_PORT=8081

# Logging
LOG_LEVEL=INFO  # DEBUG, INFO, WARNING, ERROR, CRITICAL
LOG_TO_FILE=true
LOG_FILE_PATH=./logs/trading_bot.log
LOG_ROTATION_SIZE_MB=100
LOG_RETENTION_DAYS=30

# ============================================
# ML MODEL CONFIGURATION
# ============================================

# Model Settings
ENABLE_ML_PREDICTIONS=true
ML_CONFIDENCE_THRESHOLD=0.75
ENSEMBLE_MODEL_PATH=./models/ensemble/
PUMP_PREDICTOR_PATH=./models/pump/
RUG_CLASSIFIER_PATH=./models/rug/

# Model Retraining
ENABLE_ONLINE_LEARNING=true
RETRAIN_INTERVAL_HOURS=24
MIN_SAMPLES_FOR_RETRAIN=100

# ============================================
# TRADING MODE
# ============================================

# CRITICAL: Start with paper trading!
TRADING_MODE=paper  # paper, testnet, or mainnet

# Paper Trading Settings
PAPER_TRADING_BALANCE=10000  # Simulated USD balance
PAPER_TRADING_SAVE_RESULTS=true
PAPER_TRADING_RESULTS_PATH=./results/paper_trades.json

# Testnet Configuration (for real blockchain testing)
USE_TESTNET=false
TESTNET_RPC_URL=https://goerli.infura.io/v3/YOUR_PROJECT_ID
TESTNET_CHAIN_ID=5

# ============================================
# ADVANCED SETTINGS
# ============================================

# Strategy Selection
DEFAULT_STRATEGY=momentum  # momentum, scalping, ai_strategy
ENABLE_MULTI_STRATEGY=true

# DEX Configuration
PREFERRED_DEX=uniswap_v3  # uniswap_v2, uniswap_v3, sushiswap, pancakeswap
ENABLE_DEX_AGGREGATION=true

# Mempool Monitoring
ENABLE_MEMPOOL_MONITORING=true
MEMPOOL_WS_URL=wss://mainnet.infura.io/ws/v3/YOUR_PROJECT_ID

# Database Optimization
DB_CONNECTION_POOL_SIZE=20
DB_CONNECTION_TIMEOUT=30
ENABLE_DB_SSL=false

# Cache Configuration
CACHE_TTL_SECONDS=300
CACHE_MAX_SIZE_MB=1000

# ============================================
# DEBUG & DEVELOPMENT
# ============================================

DEBUG_MODE=false
ENABLE_PROFILING=false
MOCK_TRADES=false  # Use mock data for testing
SAVE_DEBUG_DATA=false
```

---

## Security Setup (Wallet & Encryption)

### Step 1: Generate Encryption Key
```python
# Run this Python script to generate a secure encryption key
import secrets
import string

# Generate 32-character encryption key
alphabet = string.ascii_letters + string.digits
encryption_key = ''.join(secrets.choice(alphabet) for i in range(32))
print(f"Your encryption key: {encryption_key}")
print("SAVE THIS KEY SECURELY! You'll need it to decrypt your wallet.")
```

### Step 2: Encrypt Your Private Key
```python
# Save as encrypt_wallet.py and run it
from cryptography.fernet import Fernet
import base64
import hashlib

def encrypt_private_key(private_key, encryption_key):
    # Generate Fernet key from encryption key
    key = base64.urlsafe_b64encode(
        hashlib.sha256(encryption_key.encode()).digest()
    )
    fernet = Fernet(key)
    
    # Encrypt the private key
    encrypted = fernet.encrypt(private_key.encode())
    return encrypted.decode()

# Input your details (NEVER share these!)
private_key = input("Enter your wallet private key (will be hidden): ")
encryption_key = input("Enter your 32-char encryption key: ")

encrypted = encrypt_private_key(private_key, encryption_key)
print(f"\nEncrypted private key (add to .env):\n{encrypted}")
```

### Step 3: Create Dedicated Trading Wallet
```bash
# IMPORTANT: Create a NEW wallet specifically for trading
# Never use your main wallet!

1. Install MetaMask or similar wallet
2. Create new wallet
3. Save seed phrase SECURELY
4. Export private key
5. Send ONLY trading funds (start small!)
6. Encrypt private key using script above
```

### Step 4: Secure Your System
```bash
# Set proper file permissions
chmod 600 .env  # Only owner can read/write
chmod 700 ClaudeDex/  # Only owner can access directory

# Create secure backup
cp .env .env.backup
# Encrypt backup
gpg -c .env.backup

# Use environment variables in production
export WALLET_ENCRYPTION_KEY="your_key_here"
# Never commit .env to git!
```

---

## Database Installation

### Step 1: Install PostgreSQL with TimescaleDB
```bash
# Ubuntu/Debian
sudo apt update
sudo apt install postgresql-14 postgresql-contrib-14

# Add TimescaleDB repository
sudo sh -c "echo 'deb https://packagecloud.io/timescale/timescaledb/ubuntu/ $(lsb_release -c -s) main' > /etc/apt/sources.list.d/timescaledb.list"
wget --quiet -O - https://packagecloud.io/timescale/timescaledb/gpgkey | sudo apt-key add -
sudo apt update
sudo apt install timescaledb-2-postgresql-14

# macOS
brew install postgresql@14
brew install timescaledb

# Windows
# Download installer from https://www.postgresql.org/download/windows/
# Download TimescaleDB from https://www.timescale.com/downloads
```

### Step 2: Configure PostgreSQL
```bash
# Start PostgreSQL
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Configure TimescaleDB
sudo timescaledb-tune --quiet --yes

# Restart PostgreSQL
sudo systemctl restart postgresql
```

### Step 3: Create Database
```bash
# Login as postgres user
sudo -u postgres psql

# Create database and user
CREATE USER claudedex WITH PASSWORD 'your_secure_password';
CREATE DATABASE claudedex_trading OWNER claudedex;
GRANT ALL PRIVILEGES ON DATABASE claudedex_trading TO claudedex;

# Enable TimescaleDB
\c claudedex_trading
CREATE EXTENSION IF NOT EXISTS timescaledb;

# Exit
\q
```

### Step 4: Initialize Database Schema
```bash
# Run database setup script
python scripts/setup_database.py

# Verify tables created
psql -U claudedex -d claudedex_trading -c "\dt"
```

### Step 5: Install Redis
```bash
# Ubuntu/Debian
sudo apt install redis-server
sudo systemctl start redis-server
sudo systemctl enable redis-server

# Set password
redis-cli
CONFIG SET requirepass "your_redis_password"
exit

# macOS
brew install redis
brew services start redis

# Windows
# Download from https://github.com/microsoftarchive/redis/releases
```

---

## Paper Trading Mode Setup

### Step 1: Configure Paper Trading
```bash
# In your .env file, set:
TRADING_MODE=paper
PAPER_TRADING_BALANCE=10000
PAPER_TRADING_SAVE_RESULTS=true
```

### Step 2: Start Paper Trading
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Start bot in paper mode
python main.py --mode paper

# You should see:
# [INFO] Starting ClaudeDex Trading Bot v1.0.0
# [INFO] Mode: PAPER TRADING (Simulation)
# [INFO] Initial Balance: $10,000.00
# [INFO] Connecting to data sources...
# [INFO] Loading ML models...
# [INFO] Bot ready! Monitoring markets...
```

### Step 3: Monitor Paper Trading
```bash
# Watch real-time logs
tail -f logs/trading_bot.log

# Check performance
python scripts/check_performance.py

# View trades
python scripts/view_trades.py --mode paper
```

### Step 4: Access Dashboard
```bash
# Dashboard will start automatically
# Open browser to: http://localhost:8080
# Login with dashboard password from .env

# Dashboard sections:
# - Portfolio Overview
# - Active Positions
# - Trade History
# - Performance Metrics
# - Risk Analytics
# - ML Model Performance
```

---

## Production Deployment

### Step 1: Pre-Production Checklist
```yaml
Security:
✓ All API keys set and valid
✓ Wallet encrypted properly
✓ .env file permissions set (600)
✓ Firewall configured
✓ SSL certificates installed

Testing:
✓ Paper trading profitable for 2+ weeks
✓ All unit tests passing
✓ Integration tests complete
✓ Stress testing done

Configuration:
✓ Risk parameters conservative
✓ Stop losses configured
✓ Position limits set
✓ Daily loss limits active

Monitoring:
✓ Telegram alerts working
✓ Dashboard accessible
✓ Logging configured
✓ Health checks active

Backup:
✓ Database backups scheduled
✓ Configuration backed up
✓ Recovery plan documented
```

### Step 2: Deploy with Docker
```bash
# Build production image
docker build -t claudedex:production .

# Create docker-compose.yml
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  postgres:
    image: timescale/timescaledb:latest-pg14
    environment:
      POSTGRES_DB: claudedex_trading
      POSTGRES_USER: claudedex
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: always

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass ${REDIS_PASSWORD}
    ports:
      - "6379:6379"
    restart: always

  trading-bot:
    image: claudedex:production
    depends_on:
      - postgres
      - redis
    env_file:
      - .env
    volumes:
      - ./logs:/app/logs
      - ./models:/app/models
    restart: always

  dashboard:
    image: claudedex:production
    command: python -m claudedex.dashboard
    depends_on:
      - trading-bot
    ports:
      - "8080:8080"
    env_file:
      - .env
    restart: always

volumes:
  postgres_data:
EOF

# Start services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f trading-bot
```

### Step 3: Deploy with Systemd (Alternative)
```bash
# Create systemd service file
sudo nano /etc/systemd/system/claudedex.service

# Add content:
[Unit]
Description=ClaudeDex Trading Bot
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=claudedex
WorkingDirectory=/opt/claudedex
Environment="PATH=/opt/claudedex/venv/bin"
ExecStart=/opt/claudedex/venv/bin/python /opt/claudedex/main.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target

# Enable and start service
sudo systemctl daemon-reload
sudo systemctl enable claudedex
sudo systemctl start claudedex

# Check status
sudo systemctl status claudedex
```

### Step 4: Production Configuration
```bash
# Switch to mainnet mode
# In .env:
TRADING_MODE=mainnet

# Set conservative limits for production
MAX_POSITION_SIZE_PERCENT=0.01  # Start with 1%
DEFAULT_STOP_LOSS_PERCENT=0.03  # Tight stop loss
MAX_CONCURRENT_POSITIONS=2  # Limit positions

# Enable all safety features
REQUIRE_TRANSACTION_CONFIRMATION=true
ENABLE_FLASHBOTS=true
ENABLE_MEMPOOL_MONITORING=true
```

---

## Monitoring & Dashboard

### Step 1: Access Web Dashboard
```bash
# Default URL: http://localhost:8080
# Login with DASHBOARD_PASSWORD from .env

# Dashboard Features:
- Real-time portfolio value
- Active positions with P&L
- Trade history and analytics
- Risk metrics and alerts
- ML model performance
- System health status
```

### Step 2: Set Up Telegram Alerts
```bash
# 1. Create Telegram bot:
# - Message @BotFather on Telegram
# - Send: /newbot
# - Choose name and username
# - Copy the token

# 2. Get your chat ID:
# - Message your bot
# - Visit: https://api.telegram.org/bot<TOKEN>/getUpdates
# - Find your chat_id

# 3. Add to .env:
TELEGRAM_BOT_TOKEN=your_token_here
TELEGRAM_CHAT_ID=your_chat_id_here

# 4. Test alerts:
python scripts/test_alerts.py
```

### Step 3: Monitor Logs
```bash
# Real-time log monitoring
tail -f logs/trading_bot.log

# Filter for errors
grep ERROR logs/trading_bot.log

# Filter for trades
grep "TRADE" logs/trading_bot.log

# Parse JSON logs
cat logs/trading_bot.log | jq '.level == "ERROR"'
```

### Step 4: Performance Monitoring
```bash
# Check daily performance
python scripts/daily_report.py

# Generate performance report
python scripts/generate_report.py --days 30

# Export trades to CSV
python scripts/export_trades.py --format csv --output trades.csv

# Analyze strategy performance
python scripts/analyze_strategy.py --strategy momentum
```

### Step 5: Database Monitoring
```sql
-- Connect to database
psql -U claudedex -d claudedex_trading

-- Check recent trades
SELECT * FROM trades 
ORDER BY created_at DESC 
LIMIT 10;

-- Check active positions
SELECT * FROM positions 
WHERE status = 'OPEN';

-- Daily profit summary
SELECT 
    DATE(created_at) as date,
    COUNT(*) as trades,
    SUM(pnl_usd) as total_pnl
FROM trades
GROUP BY DATE(created_at)
ORDER BY date DESC;

-- Token performance
SELECT 
    token_symbol,
    COUNT(*) as trades,
    AVG(pnl_percent) as avg_return,
    SUM(pnl_usd) as total_pnl
FROM trades
GROUP BY token_symbol
ORDER BY total_pnl DESC;
```

---

## Troubleshooting Guide

### Common Issues & Solutions

#### 1. Bot Won't Start
```bash
# Check Python version
python --version  # Should be 3.9-3.11

# Check dependencies
pip list | grep -E "web3|pandas|numpy"

# Reinstall requirements
pip install --force-reinstall -r requirements.txt

# Check database connection
psql -U claudedex -d claudedex_trading -c "SELECT 1;"

# Check Redis connection
redis-cli ping  # Should return PONG
```

#### 2. API Connection Errors
```bash
# Test API endpoints
python scripts/test_apis.py

# Common fixes:
- Verify API keys in .env
- Check rate limits
- Ensure IP whitelisting (if required)
- Try backup RPC endpoints
```

#### 3. Transaction Failures
```bash
# Check gas settings
# Increase in .env:
MAX_GAS_PRICE_GWEI=150

# Check nonce issues
python scripts/reset_nonce.py

# Verify wallet balance
python scripts/check_balance.py
```

#### 4. ML Model Errors
```bash
# Retrain models
python scripts/retrain_models.py

# Reset model cache
rm -rf models/cache/*

# Verify model files
ls -la models/
```

#### 5. Database Issues
```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Check connections
SELECT count(*) FROM pg_stat_activity;

# Vacuum database
VACUUM ANALYZE;

# Reset sequences
python scripts/reset_db_sequences.py
```

#### 6. Memory Issues
```bash
# Check memory usage
free -h

# Clear Redis cache
redis-cli FLUSHDB

# Reduce position limits
# In .env:
MAX_CONCURRENT_POSITIONS=1
DB_CONNECTION_POOL_SIZE=10
```

---

## Maintenance & Updates

### Daily Maintenance
```bash
# Morning checklist:
1. Check overnight trades
   python scripts/overnight_summary.py

2. Verify system health
   python scripts/health_check.py

3. Review error logs
   grep ERROR logs/trading_bot.log | tail -20

4. Check wallet balance
   python scripts/check_balance.py
```

### Weekly Maintenance
```bash
# 1. Backup database
pg_dump -U claudedex claudedex_trading > backup_$(date +%Y%m%d).sql

# 2. Analyze performance
python scripts/weekly_report.py

# 3. Update models
python scripts/update_models.py

# 4. Clean old logs
find logs/ -name "*.log" -mtime +30 -delete

# 5. Update blacklists
python scripts/update_blacklists.py
```

### Monthly Maintenance
```bash
# 1. Full system backup
./scripts/full_backup.sh

# 2. Update dependencies
pip list --outdated
pip install --upgrade -r requirements.txt

# 3. Security audit
python scripts/security_audit.py

# 4. Performance optimization
python scripts/optimize_db.py

# 5. Review and adjust strategies
python scripts/strategy_analysis.py --month
```

### Updating the Bot
```bash
# 1. Backup current version
cp -r ClaudeDex/ ClaudeDex_backup_$(date +%Y%m%d)/

# 2. Pull latest changes
git pull origin main

# 3. Update dependencies
pip install --upgrade -r requirements.txt

# 4. Run migrations
python scripts/migrate_database.py

# 5. Restart bot
sudo systemctl restart claudedex
# or
docker-compose restart trading-bot

# 6. Verify functionality
python scripts/post_update_check.py
```

---

## Safety Guidelines

### Start Small
```yaml
Week 1-2: Paper trading only
Week 3-4: Testnet with fake tokens
Week 5-6: Mainnet with $100 max
Week 7-8: Gradually increase to $500
Week 9+: Scale based on performance
```

### Risk Management Rules
```yaml
Never Risk:
- More than 5% per trade
- More than 10% daily
- More than 20% of portfolio

Always Use:
- Stop losses on every trade
- Position size limits
- Daily loss limits
- Correlation limits
```

### Emergency Procedures
```bash
# EMERGENCY STOP
python scripts/emergency_stop.py

# Close all positions
python scripts/close_all_positions.py --confirm

# Withdraw funds
python scripts/withdraw_funds.py --to YOUR_SAFE_WALLET

# Disable trading
# In .env:
TRADING_MODE=stopped
```

---

## Support & Resources

### Getting Help
```yaml
Documentation: https://claudedex-docs.com
Discord: https://discord.gg/claudedex
Telegram: https://t.me/claudedex_support
Email: support@claudedex.io
```

### Important Notes
```
⚠️ DISCLAIMERS:
- Trading cryptocurrency is highly risky
- Past performance doesn't guarantee future results
- Never invest more than you can afford to lose
- This bot is for educational purposes
- Always do your own research (DYOR)
- The developers are not financial advisors
- Use at your own risk
```

---

## Quick Start Commands

```bash
# Setup
git clone <repository>
cd claudedex-bot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your settings

# Database setup
python scripts/setup_database.py

# Paper trading
python main.py --mode paper

# Production
python main.py --mode mainnet

# Dashboard
# Visit http://localhost:8080

# Emergency stop
python scripts/emergency_stop.py
```

Remember: **Start with paper trading!** Test thoroughly before using real funds.