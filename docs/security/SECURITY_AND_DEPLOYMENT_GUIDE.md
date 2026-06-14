# ClaudeDex Security & Deployment Guide

## Table of Contents
1. [Security Architecture Overview](#security-architecture-overview)
2. [For Existing Installations (Migration Guide)](#for-existing-installations)
3. [Fresh Installation Guide](#fresh-installation-guide)
4. [Credential Management](#credential-management)
5. [Changing Passwords](#changing-passwords)
6. [Files and Scripts Reference](#files-and-scripts-reference)
7. [Troubleshooting](#troubleshooting)

---

## Security Architecture Overview

ClaudeDex uses a **two-tier security model**:

### Tier 1: Infrastructure Credentials (Docker Secrets)
These are needed to **start the containers**:
- Database username/password
- Redis password

**Stored in:** `./secrets/` directory (NOT in git)

### Tier 2: Application Credentials (Encrypted in Database)
These are API keys, wallet keys, and other secrets:
- Private keys (EVM, Solana)
- Exchange API keys (Binance, Bybit)
- Third-party API keys
- Notification credentials

**Stored in:** PostgreSQL `secure_credentials` table, encrypted with Fernet

### Encryption Key
The master encryption key (`.encryption_key`) is stored separately from both:
- NOT in the database
- NOT in docker-compose.yml
- NOT in .env
- Mounted read-only into the container

```
┌─────────────────────────────────────────────────────────────────────┐
│                         HOST MACHINE                                │
│                                                                     │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────────┐   │
│  │ ./secrets/      │  │ .encryption_key │  │ .env (optional)  │   │
│  │ - db_user       │  │ (Fernet key)    │  │ (non-sensitive   │   │
│  │ - db_password   │  │                 │  │  config only)    │   │
│  │ - redis_password│  │                 │  │                  │   │
│  └────────┬────────┘  └────────┬────────┘  └────────┬─────────┘   │
│           │ (Docker secrets)   │ (read-only mount)  │             │
│  ┌────────▼────────────────────▼────────────────────▼─────────┐   │
│  │                    DOCKER CONTAINER                         │   │
│  │  /run/secrets/db_user      /app/.encryption_key             │   │
│  │  /run/secrets/db_password                                   │   │
│  │  /run/secrets/redis_password                                │   │
│  │                            │                                 │   │
│  │                            ▼                                 │   │
│  │              ┌─────────────────────────────┐                │   │
│  │              │    PostgreSQL Database      │                │   │
│  │              │  ┌───────────────────────┐  │                │   │
│  │              │  │ secure_credentials    │  │                │   │
│  │              │  │ (encrypted values)    │  │                │   │
│  │              │  └───────────────────────┘  │                │   │
│  │              └─────────────────────────────┘                │   │
│  └─────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
```

---

## For Existing Installations

If you have an existing running bot with data, follow these steps to migrate to the secure architecture.

### Step 1: Stop the Bot
```bash
cd /path/to/claudedex
docker-compose down
```

### Step 2: Pull Latest Code
```bash
git pull origin claude/secure-credentials-removal-VzTfx
```

### Step 3: Create Secrets Directory
```bash
# Create secrets directory with secure permissions
mkdir -p ./secrets
chmod 700 ./secrets
```

### Step 4: Create Infrastructure Secret Files

**IMPORTANT:** Use NEW, STRONG passwords - your old passwords may be exposed in git history!

```bash
# Generate strong passwords
# Option A: Use openssl
DB_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)
REDIS_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)

# Option B: Use Python
# DB_PASS=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")
# REDIS_PASS=$(python3 -c "import secrets; print(secrets.token_urlsafe(24))")

# Create secret files
echo "bot_user" > ./secrets/db_user
echo "$DB_PASS" > ./secrets/db_password
echo "$REDIS_PASS" > ./secrets/redis_password

# Secure permissions
chmod 600 ./secrets/*

# Display for your records (save these!)
echo "=============================================="
echo "NEW CREDENTIALS (SAVE THESE SECURELY!):"
echo "=============================================="
echo "DB User: bot_user"
echo "DB Password: $DB_PASS"
echo "Redis Password: $REDIS_PASS"
echo "=============================================="
```

### Step 5: Setup Encryption Key
```bash
# If you have an existing encryption key in .env, extract it:
grep "^ENCRYPTION_KEY=" .env | sed 's/^ENCRYPTION_KEY=//' > .encryption_key

# Verify it ends with = (required for Fernet)
cat .encryption_key

# Set permissions
chmod 600 .encryption_key
```

### Step 6: Update Database Password

Since the database already exists with the old password, you need to update it:

```bash
# Start only postgres with old config temporarily
docker-compose up -d postgres

# Wait for it to be ready
sleep 5

# Connect and change password
docker exec -it trading-postgres psql -U bot_user -d tradingbot -c \
    "ALTER USER bot_user WITH PASSWORD '$(cat ./secrets/db_password)';"

# Stop postgres
docker-compose down
```

### Step 7: Update Redis Password

Redis stores its password in the data volume. You need to either:

**Option A: Flush Redis data (if acceptable)**
```bash
# Remove Redis volume (loses cached data)
docker volume rm claudedex_redis-data

# Or use docker-compose
docker-compose down -v  # WARNING: Also removes postgres data!
```

**Option B: Keep Redis data**
```bash
# Start redis with old password, update, then restart with new
# This is complex - Option A is recommended
```

### Step 8: Clean Up .env File

Remove sensitive data from .env (keep only non-sensitive config):

```bash
# Backup first
cp .env .env.backup.$(date +%Y%m%d)

# Remove sensitive entries
sed -i '/^ENCRYPTION_KEY=/d' .env
sed -i '/^DB_PASSWORD=/d' .env
sed -i '/^DB_USER=/d' .env
sed -i '/^REDIS_PASSWORD=/d' .env
sed -i '/^DATABASE_URL=/d' .env
sed -i '/^REDIS_URL=/d' .env

# The following can stay in .env (non-sensitive config):
# DEX_MODULE_ENABLED, FUTURES_MODULE_ENABLED, SOLANA_MODULE_ENABLED
# LOG_LEVEL, etc.
```

### Step 9: Rebuild and Start
```bash
docker-compose up -d --build
```

### Step 10: Re-import Application Credentials
```bash
# Wait for containers to be healthy
sleep 10

# Force re-import all credentials with correct encryption key
docker exec -it trading-bot python scripts/force_reimport_credentials.py
```

### Step 11: Verify Everything Works
```bash
# Check logs
docker logs trading-bot 2>&1 | grep -E "encryption|credentials|decrypted"

# Expected output:
# Using encryption key from file: .encryption_key
# ✅ Successfully decrypted private key
# ✅ Credentials Management routes initialized

# Access dashboard
curl -I http://localhost:8080/login
```

---

## Fresh Installation Guide

For a brand new VPS with no existing data.

### Prerequisites
- Ubuntu 20.04+ or Debian 11+
- Docker and Docker Compose installed
- Git installed
- At least 4GB RAM, 20GB disk

### Step 1: Install Docker (if not installed)
```bash
# Update system
sudo apt update && sudo apt upgrade -y

# Install Docker
curl -fsSL https://get.docker.com -o get-docker.sh
sudo sh get-docker.sh

# Add your user to docker group
sudo usermod -aG docker $USER

# Install Docker Compose
sudo apt install docker-compose-plugin -y

# Logout and login again, or run:
newgrp docker
```

### Step 2: Clone Repository
```bash
cd /opt  # or your preferred directory
git clone https://github.com/yourusername/claudedex.git
cd claudedex
```

### Step 3: Run Setup Scripts
```bash
# Make scripts executable
chmod +x scripts/*.sh

# Setup infrastructure secrets
./scripts/setup_secrets.sh

# Setup encryption key
./scripts/setup_encryption_key.sh
```

### Step 4: Create .env for Non-Sensitive Config
```bash
cat > .env << 'EOF'
# Module Enable/Disable
DEX_MODULE_ENABLED=true
FUTURES_MODULE_ENABLED=false
SOLANA_MODULE_ENABLED=false

# Logging
LOG_LEVEL=INFO

# Optional: RPC URLs (non-sensitive)
# ETH_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/your-key
# BSC_RPC_URL=https://bsc-dataseed.binance.org
EOF
```

### Step 5: Start Containers
```bash
# Start all services
docker-compose up -d --build

# Wait for database to initialize
sleep 15

# Check status
docker-compose ps
```

### Step 6: Initialize Database
The database schema is automatically created via `scripts/init.sql`.

### Step 7: Create Admin User
```bash
docker exec -it trading-bot python scripts/create_admin_user.py
```

### Step 8: Import Your Credentials

**Option A: Via Dashboard**
1. Open http://your-server:8080/login
2. Login with admin/admin123
3. **Change your password immediately!**
4. Go to Secure Credentials page
5. Add each credential manually

**Option B: Via .env Import**
1. Create a temporary .env with your credentials
2. Run import script:
```bash
docker exec -it trading-bot python scripts/force_reimport_credentials.py
```
3. Delete the temporary .env

### Step 9: Verify Installation
```bash
# Check all containers are running
docker-compose ps

# Check bot logs
docker logs -f trading-bot

# Test dashboard
curl http://localhost:8080/login
```

---

## Credential Management

### Adding New Credentials via Dashboard

1. Navigate to `/credentials`
2. Click "Add New Credential"
3. Fill in:
   - **Key Name**: e.g., `MY_NEW_API_KEY` (uppercase, underscores)
   - **Value**: The actual secret (will be encrypted)
   - **Category**: Select appropriate category
   - **Encrypt**: Keep checked for sensitive data
4. Click "Add Credential"

### Updating Existing Credentials

1. Navigate to `/credentials`
2. Find the credential and click ✏️ Edit
3. Enter the new value
4. Click "Save"

The new value is automatically encrypted with your master key.

### Credential Categories

| Category | Examples |
|----------|----------|
| Security | ENCRYPTION_KEY, JWT_SECRET, SESSION_SECRET |
| Wallet | PRIVATE_KEY, SOLANA_PRIVATE_KEY, WALLET_ADDRESS |
| Exchange | BINANCE_API_KEY, BYBIT_API_SECRET |
| Database | DB_PASSWORD, DATABASE_URL, REDIS_URL |
| API | OPENAI_API_KEY, HELIUS_API_KEY, ETHERSCAN_API_KEY |
| Notification | TELEGRAM_BOT_TOKEN, DISCORD_WEBHOOK_URL |

---

## Changing Passwords

### Change Database Password

```bash
# 1. Generate new password
NEW_DB_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)
echo "New DB Password: $NEW_DB_PASS"

# 2. Update in running database
docker exec -it trading-postgres psql -U bot_user -d tradingbot -c \
    "ALTER USER bot_user WITH PASSWORD '$NEW_DB_PASS';"

# 3. Update secret file
echo "$NEW_DB_PASS" > ./secrets/db_password
chmod 600 ./secrets/db_password

# 4. Restart bot to pick up new password
docker-compose restart trading-bot
```

### Change Redis Password

```bash
# 1. Generate new password
NEW_REDIS_PASS=$(openssl rand -base64 24 | tr -d '/+=' | head -c 24)
echo "New Redis Password: $NEW_REDIS_PASS"

# 2. Update secret file
echo "$NEW_REDIS_PASS" > ./secrets/redis_password
chmod 600 ./secrets/redis_password

# 3. Restart all services (Redis needs restart)
docker-compose down
docker-compose up -d
```

### Rotate Encryption Key

⚠️ **WARNING**: This will require re-encrypting ALL credentials!

```bash
# 1. Backup current key
cp .encryption_key .encryption_key.backup.$(date +%Y%m%d)

# 2. Generate new key
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > .encryption_key.new

# 3. Decrypt all credentials with old key, re-encrypt with new
# (This requires a custom migration script - contact maintainer)

# 4. Replace key
mv .encryption_key.new .encryption_key
chmod 600 .encryption_key

# 5. Restart
docker-compose restart trading-bot
```

### Change Dashboard Admin Password

1. Login to dashboard
2. Go to Settings → Security
3. Click "Change Password"
4. Enter current and new password

Or via command line:
```bash
docker exec -it trading-bot python -c "
from auth.auth_service import AuthService
import asyncio

async def change_password():
    # Connect to DB and update password
    pass

asyncio.run(change_password())
"
```

---

## Files and Scripts Reference

### Files That MUST NOT Be in Git

| File/Directory | Purpose | In .gitignore? |
|----------------|---------|----------------|
| `.env` | Environment variables (may contain secrets) | ✅ Yes |
| `.encryption_key` | Master encryption key | ✅ Yes |
| `secrets/` | Docker secrets directory | ✅ Yes |
| `*.secret` | Any secret files | ✅ Yes |
| `logs/` | Log files | ✅ Yes |
| `data/` | Data files | ✅ Yes |

### Scripts Used by Bot (DO NOT DELETE)

| Script | Purpose | Used By |
|--------|---------|---------|
| `scripts/docker-entrypoint.sh` | Container startup | Dockerfile |
| `scripts/init.sql` | Database schema | Docker Compose |
| `scripts/force_reimport_credentials.py` | Credential migration | Manual |
| `scripts/setup_secrets.sh` | Setup Docker secrets | Manual |
| `scripts/setup_encryption_key.sh` | Setup encryption key | Manual |
| `scripts/create_admin_user.py` | Create dashboard admin | Manual |

### Scripts for Operations (Useful but Optional)

| Script | Purpose |
|--------|---------|
| `scripts/health_check.py` | System health check |
| `scripts/emergency_stop.py` | Stop all trading |
| `scripts/daily_report.py` | Generate daily report |
| `scripts/weekly_report.py` | Generate weekly report |
| `scripts/export_trades.py` | Export trade history |
| `scripts/optimize_db.py` | Database optimization |
| `scripts/security_audit.py` | Security audit |

### Scripts NOT Used by Bot (Can be moved to backup)

| Script | Purpose | Notes |
|--------|---------|-------|
| `scripts/dev_autofix_imports.py` | Development tool | Dev only |
| `scripts/fix_illegal_relatives.py` | Development tool | Dev only |
| `scripts/verify_claudedex_plus.py` | Verification script | One-time use |
| `scripts/verify_claudedex_plus2.py` | Verification script | One-time use |
| `scripts/verify_claudedex_plus3.py` | Verification script | One-time use |
| `scripts/train_models.py` | Empty file | Placeholder |
| `scripts/run_tests.py` | Test runner | Dev only |
| `scripts/test_alerts.py` | Test alerts | Dev only |
| `scripts/test_apis.py` | Test APIs | Dev only |
| `scripts/test_solana_setup.py` | Test Solana | Dev only |

### Root Files NOT Used by Bot (Can be moved to backup)

| File | Purpose | Notes |
|------|---------|-------|
| `generate_file_tree.py` | Dev tool | Not needed in prod |
| `scaffold.py` | Dev tool | Not needed in prod |
| `create_tree.py` | Dev tool | Not needed in prod |
| `verify_references.py` | Dev tool | Not needed in prod |
| `xref_symbol_db.py` | Dev tool | Not needed in prod |
| `test_abi_loading.py` | Test script | Dev only |
| `test_dexscreener_solana.py` | Test script | Dev only |
| `test_patches.py` | Test script | Dev only |

### Markdown Files That Can Be Archived

These are development/review documents not needed for production:

**Root folder MD files (can archive):**
- `AUDIT_SUMMARY.md` - Past audit results
- `ACCURATE_DASHBOARD_FIXES.md` - Past fix documentation
- `CRITICAL_FIXES_GUIDE.md` - Past fix documentation
- `IMPLEMENTATION_SUMMARY.md` - Past implementation notes
- `INTEGRATION_MASTER_SUMMARY.md` - Past integration notes
- `PHASE_*_REVIEW_REPORT.md` - Phase review reports (1-4)
- `NEXT_SESSION_*.md` - Session continuation notes
- `REVIEW_22_POINTS.md` - Review notes
- `P1_FIXES_*.md` - Past fix documentation
- `VOLATILITY_SPREAD_FIX.md` - Past fix documentation
- `STRATEGY_SELECTION_FIX.md` - Past fix documentation
- `INTEGRATION_INSTRUCTIONS.md` - Past integration docs
- `PHASE_1_TESTING_GUIDE.md` - Testing documentation
- `DEPLOY_AUTH_FIX.md` - Past deployment fix
- `file_tree*.md` - File structure snapshots
- `verifier_report.md` - Past verification
- `PRODUCTION_READINESS_AUDIT.md` - Past audit
- `FINAL_REPORT.md` - Past report
- `FIXES_VERIFICATION_COMPLETE.md` - Past verification
- `ref_report*.md` - Reference reports
- `xref_report*.md` - Cross-reference reports
- `symbol_db*.md` - Symbol database reports

**Keep these MD files (important for users):**
- `README.md` - Main readme
- `SECURITY_AND_DEPLOYMENT_GUIDE.md` - This guide
- `SECURE_CREDENTIALS_GUIDE.md` - Credentials guide
- `SECURITY_ARCHITECTURE.md` - Architecture overview
- `AUTH_README.md` - Authentication guide
- `AUTH_SETUP_GUIDE.md` - Auth setup
- `CONFIG_SYSTEM.md` - Configuration system
- `DOCKER_BUILD_GUIDE.md` - Docker build instructions

**docs/ folder - Keep all (user documentation):**
- `docs/architecture.md`
- `docs/deployment_guide.md`
- `docs/deployment_checklist.md`
- `docs/Fresh_Deployment_Guide.md`
- `docs/User_Guide.md`
- `docs/Quick_Real_Trade_Switch.md`
- `docs/api_documentation.md`
- `docs/solana_integration_guide.md`
- `docs/solana_roadmap.md`

### Quick Archive Command

To move unused files to a backup folder before publishing:

```bash
# Create backup directories
mkdir -p backup/scripts backup/root_files backup/md_reports

# Move unused scripts
mv scripts/dev_autofix_imports.py backup/scripts/
mv scripts/fix_illegal_relatives.py backup/scripts/
mv scripts/verify_claudedex_plus*.py backup/scripts/
mv scripts/train_models.py backup/scripts/
mv scripts/run_tests.py backup/scripts/
mv scripts/test_*.py backup/scripts/

# Move unused root files
mv generate_file_tree.py backup/root_files/
mv scaffold.py backup/root_files/
mv create_tree.py backup/root_files/
mv verify_references.py backup/root_files/
mv xref_symbol_db.py backup/root_files/
mv test_*.py backup/root_files/

# Move old report MD files
mv *_SUMMARY.md backup/md_reports/
mv *_REPORT.md backup/md_reports/
mv *_FIX*.md backup/md_reports/
mv *_report*.md backup/md_reports/
mv PHASE_*.md backup/md_reports/
mv NEXT_SESSION_*.md backup/md_reports/
mv file_tree*.md backup/md_reports/
mv symbol_db*.md backup/md_reports/

# Add backup folder to gitignore
echo "backup/" >> .gitignore
```

---

## Troubleshooting

### Bot Can't Connect to Database

```bash
# Check postgres is running
docker-compose ps postgres

# Check logs
docker logs trading-postgres

# Verify password is correct
cat ./secrets/db_password
docker exec -it trading-postgres psql -U bot_user -d tradingbot -c "SELECT 1;"
```

### Encryption Key Errors

```bash
# Verify key exists and is valid
cat .encryption_key
# Should be 44 characters ending with =

# Check key is readable by container
docker exec trading-bot cat /app/.encryption_key

# Regenerate if needed (WARNING: will lose encrypted data!)
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" > .encryption_key
```

### Credentials Not Showing in Dashboard

```bash
# Check credentials routes are initialized
docker logs trading-bot 2>&1 | grep "Credentials"

# Check database has data
docker exec trading-postgres psql -U bot_user -d tradingbot -c \
    "SELECT key_name, category, has_value FROM secure_credentials LIMIT 10;"

# Force re-import
docker exec -it trading-bot python scripts/force_reimport_credentials.py
```

### Redis Connection Failed

```bash
# Check Redis is running
docker-compose ps redis

# Verify password
cat ./secrets/redis_password

# Test connection
docker exec trading-redis redis-cli -a "$(cat ./secrets/redis_password)" ping
```

---

## Security Checklist

Before going to production, verify:

- [ ] `.encryption_key` exists and is NOT in git
- [ ] `secrets/` directory exists with proper permissions (700)
- [ ] Secret files have proper permissions (600)
- [ ] No hardcoded passwords in docker-compose.yml
- [ ] No sensitive data in .env (or .env is in .gitignore)
- [ ] Dashboard admin password has been changed
- [ ] All API keys are stored in database (not .env)
- [ ] Private keys are encrypted in database
- [ ] Git history doesn't contain secrets (or repo is private)

```bash
# Quick security check
./scripts/security_audit.py
```

---

## Support

If you encounter issues:
1. Check the logs: `docker logs trading-bot`
2. Review this guide
3. Open an issue on GitHub

**Never share your:**
- `.encryption_key`
- `secrets/*` files
- Private keys
- API keys
