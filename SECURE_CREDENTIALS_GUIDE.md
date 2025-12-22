# ClaudeDex Secure Credentials Implementation Guide

## Table of Contents
1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Files Modified/Created](#files-modifiedcreated)
4. [Step-by-Step Implementation](#step-by-step-implementation)
5. [Docker Configuration](#docker-configuration)
6. [Dashboard Usage](#dashboard-usage)
7. [Cleaning GitHub Repository](#cleaning-github-repository)
8. [Database Password Change](#database-password-change)
9. [Fresh Installation](#fresh-installation)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This guide explains how to migrate from storing sensitive credentials in `.env` to a secure database-backed system where:

1. **Encryption key is stored separately** from encrypted data
2. **Credentials are encrypted at rest** in the database
3. **Fallback to .env** until migration is complete
4. **Dashboard UI** for managing credentials
5. **Docker-compatible** setup

### Security Benefits
- Encryption key not stored alongside encrypted data
- Credentials encrypted in database
- Audit trail for credential access
- Rotation tracking
- Dashboard for management

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     APPLICATION LAYER                            │
├─────────────────────────────────────────────────────────────────┤
│  SecureSecretsManager                                            │
│  ├── Priority 1: Docker Secrets (/run/secrets/)                 │
│  ├── Priority 2: Database (encrypted)                           │
│  └── Priority 3: Environment (.env fallback)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     STORAGE LAYER                                │
├──────────────────────┬──────────────────────────────────────────┤
│  Docker Secret       │  Database (secure_credentials)           │
│  /run/secrets/       │  ├── encrypted_value (Fernet)            │
│  encryption_key      │  ├── value_hash (SHA256)                 │
│                      │  ├── category                             │
│                      │  └── audit metadata                       │
└──────────────────────┴──────────────────────────────────────────┘
```

---

## Files Modified/Created

### New Files Created

| File | Purpose |
|------|---------|
| `migrations/012_add_secure_credentials_table.sql` | Database schema for credentials |
| `security/secrets_manager.py` | SecureSecretsManager class |
| `monitoring/credentials_routes.py` | Dashboard API routes |
| `dashboard/templates/settings_credentials.html` | Credentials management page |
| `scripts/migrate_credentials_to_db.py` | Migration script |
| `scripts/setup_secure_credentials.py` | Credential setup tool |
| `SECURITY_ARCHITECTURE.md` | Security documentation |
| `SECURE_CREDENTIALS_GUIDE.md` | This guide |

### Files Modified

| File | Changes |
|------|---------|
| `monitoring/enhanced_dashboard.py` | Added credentials routes setup |
| `.gitignore` | Added encryption key patterns |

### Files That Use Credentials (75+ files)

The following modules need to use `SecureSecretsManager`:

| Category | Files | Key Credentials |
|----------|-------|-----------------|
| **Core Config** | `config/config_manager.py`, `config/settings.py` | ENCRYPTION_KEY, DATABASE_URL |
| **Main Modules** | `main.py`, `main_dex.py` | PRIVATE_KEY, API keys |
| **Trading** | `trading/executors/base_executor.py` | PRIVATE_KEY, WALLET_ADDRESS |
| **Solana** | `modules/solana_trading/config/solana_config.py` | SOLANA_PRIVATE_KEY, JUPITER_API_KEY |
| **Futures** | `modules/ai_analysis/core/sentiment_engine.py` | BINANCE_API_KEY/SECRET |
| **Dashboard** | `monitoring/enhanced_dashboard.py` | Various wallet addresses |
| **Scripts** | 20+ scripts in `scripts/` | DB_PASSWORD, various keys |

---

## Step-by-Step Implementation

### Phase 1: Preparation (Keep Using .env)

During this phase, everything continues to work with .env. No changes to running system.

#### Step 1.1: Run Database Migration

```bash
# Inside Docker container
docker exec -it trading-bot bash

# Run migration to create secure_credentials table
psql -h postgres -U bot_user -d tradingbot -f migrations/012_add_secure_credentials_table.sql

# Or if using migration manager:
python data/migration_manager.py
```

#### Step 1.2: Verify Table Created

```bash
docker exec -it trading-postgres psql -U bot_user -d tradingbot -c "\d secure_credentials"
```

### Phase 2: Migrate Credentials to Database

#### Step 2.1: Preview Migration

```bash
# Preview what will be migrated (no changes made)
docker exec -it trading-bot python scripts/migrate_credentials_to_db.py --dry-run
```

#### Step 2.2: Execute Migration

```bash
# Actually migrate credentials
docker exec -it trading-bot python scripts/migrate_credentials_to_db.py
```

#### Step 2.3: Verify Migration

```bash
# Validate credentials in database
docker exec -it trading-bot python scripts/migrate_credentials_to_db.py --validate
```

### Phase 3: Access Dashboard

#### Step 3.1: Access Credentials Page

1. Open dashboard in browser: `http://your-server:8080`
2. Navigate to: `http://your-server:8080/credentials`
3. You should see all credentials organized by category

#### Step 3.2: Verify Credentials

- Check that all credentials show as "Configured"
- Required credentials should be green
- Missing required credentials will be red

### Phase 4: Secure the Encryption Key

**This is the critical security step!**

#### Step 4.1: Create Secure Key Storage

```bash
# On your VPS (outside Docker)
sudo mkdir -p /secure
sudo chmod 700 /secure

# Copy encryption key to secure location
echo "bvSd97GRf4nlMqu5ISd7S62VkvtR4NElwDwQ9V-DINU=" | sudo tee /secure/encryption.key > /dev/null
sudo chmod 600 /secure/encryption.key
```

#### Step 4.2: Update Docker Compose

Add the following to your `docker-compose.yml`:

```yaml
services:
  dex:
    volumes:
      - /secure/encryption.key:/run/secrets/encryption_key:ro
    environment:
      - ENCRYPTION_KEY_FILE=/run/secrets/encryption_key
```

Or use Docker secrets:

```yaml
secrets:
  encryption_key:
    file: /secure/encryption.key

services:
  dex:
    secrets:
      - encryption_key
```

#### Step 4.3: Restart Services

```bash
docker-compose down
docker-compose up -d
```

### Phase 5: Clean Up .env File

**Only do this after confirming everything works!**

#### Step 5.1: Backup Current .env

```bash
cp .env .env.backup.$(date +%Y%m%d)
```

#### Step 5.2: Remove Sensitive Values

Edit `.env` and remove/replace sensitive values:

```bash
# BEFORE:
ENCRYPTION_KEY=bvSd97GRf4nlMqu5ISd7S62VkvtR4NElwDwQ9V-DINU=
PRIVATE_KEY=gAAAAABo_94x...

# AFTER:
# ENCRYPTION_KEY is now in /secure/encryption.key
# PRIVATE_KEY is now in database
```

Keep only non-sensitive configuration in `.env`:

```bash
# MODULE FLAGS (keep these)
DEX_MODULE_ENABLED=true
FUTURES_MODULE_ENABLED=true
MODE=production
DRY_RUN=true

# DATABASE CONNECTION (keep URL, password is in DB now)
DATABASE_URL=postgresql://bot_user@postgres:5432/tradingbot
```

---

## Docker Configuration

### Complete docker-compose.yml Example

```yaml
version: '3.8'

secrets:
  encryption_key:
    file: /secure/encryption.key

services:
  dex:
    build: .
    container_name: trading-bot
    restart: unless-stopped
    depends_on:
      - postgres
      - redis
    secrets:
      - encryption_key
    environment:
      - ENCRYPTION_KEY_FILE=/run/secrets/encryption_key
      - DATABASE_URL=postgresql://bot_user:${DB_PASSWORD}@postgres:5432/tradingbot
    volumes:
      - ./logs:/app/logs
      - ./data:/app/data
    ports:
      - "8080:8080"

  postgres:
    image: postgres:15
    container_name: trading-postgres
    restart: unless-stopped
    environment:
      POSTGRES_USER: bot_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
      POSTGRES_DB: tradingbot
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7
    container_name: trading-redis
    restart: unless-stopped
    command: redis-server --requirepass ${REDIS_PASSWORD}

volumes:
  postgres_data:
```

### Running Migration Inside Docker

```bash
# Enter the container
docker exec -it trading-bot bash

# Run migration
python scripts/migrate_credentials_to_db.py

# Validate
python scripts/migrate_credentials_to_db.py --validate

# Exit container
exit
```

---

## Dashboard Usage

### Accessing the Credentials Page

1. **URL**: `http://your-server:8080/credentials`
2. **Authentication**: Use your dashboard credentials

### Dashboard Features

#### View Credentials
- See all credentials organized by category
- Status indicators (Configured/Not Set)
- Required credentials marked with red badge

#### Add New Credential
1. Click "Add New Credential"
2. Enter key name (uppercase with underscores)
3. Enter value
4. Select category
5. Click Save

#### Edit Credential
1. Click the edit icon on any credential card
2. Enter new value (leave blank to keep current)
3. Update metadata
4. Click Save

#### Import from .env
1. Click "Import from .env" button
2. Review credentials to import
3. Click "Import Selected"

### Security Status Indicators

| Status | Meaning |
|--------|---------|
| **Bootstrap Mode** | Using .env fallback - not secure |
| **Secure Mode** | Using database with encryption |
| **No Encryption** | Encryption key not available |

---

## Cleaning GitHub Repository

### Step 1: Remove .env.final from History

```bash
# Install BFG Repo-Cleaner
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Create backup
cd ..
cp -r claudedex claudedex-backup-$(date +%Y%m%d)
cd claudedex

# Remove .env.final from all history
java -jar ../bfg-1.14.0.jar --delete-files .env.final

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (WARNING: This rewrites history!)
git push origin --force --all
```

### Step 2: Verify .gitignore

Ensure `.gitignore` contains:

```gitignore
# Environment Files - CRITICAL SECURITY
.env
.env.local
.env.production
.env.development
.env.test
.env.final
.env.backup
.env.*
!.env.example

# Encryption Keys - NEVER COMMIT
.encryption_key
*.encryption_key
encryption.key
master.key
.master_key
*.fernet
secrets.key
```

### Step 3: Scan for Leaked Secrets

```bash
# Search for any remaining secrets
git log --all -p | grep -i "api_key\|secret\|password\|private_key" | head -50
```

---

## Database Password Change

### Step 1: Update PostgreSQL Password

```bash
# Connect to PostgreSQL
docker exec -it trading-postgres psql -U postgres

# Change password
ALTER USER bot_user WITH PASSWORD 'your_new_secure_password';

# Exit
\q
```

### Step 2: Update Docker Environment

Update your `.env` or docker-compose with new password:

```bash
DB_PASSWORD=your_new_secure_password
```

### Step 3: Update Credential in Database

After restart, update the stored password:

```bash
docker exec -it trading-bot python -c "
import asyncio
from security.secrets_manager import secrets
asyncio.run(secrets.set('DB_PASSWORD', 'your_new_secure_password', 'database'))
"
```

### Step 4: Restart Services

```bash
docker-compose down
docker-compose up -d
```

---

## Fresh Installation

### Step 1: Clone Repository

```bash
git clone https://github.com/alioanka/claudedex.git
cd claudedex
```

### Step 2: Create Secure Directory

```bash
sudo mkdir -p /secure
sudo chmod 700 /secure
```

### Step 3: Generate New Encryption Key

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())" | sudo tee /secure/encryption.key > /dev/null
sudo chmod 600 /secure/encryption.key
```

### Step 4: Create Minimal .env

```bash
cat > .env << 'EOF'
# ClaudeDex Configuration - Minimal Setup
MODE=production
DRY_RUN=true

# Module Flags
DEX_MODULE_ENABLED=true
FUTURES_MODULE_ENABLED=false
SOLANA_MODULE_ENABLED=false

# Database (password will be in secure storage)
DATABASE_URL=postgresql://bot_user@postgres:5432/tradingbot
DB_PASSWORD=initial_password_change_me

# Redis
REDIS_PASSWORD=initial_redis_password_change_me
EOF
```

### Step 5: Start Docker

```bash
docker-compose up -d
```

### Step 6: Run Migrations

```bash
# Wait for database to be ready
sleep 10

# Run all migrations
docker exec -it trading-bot python data/migration_manager.py
```

### Step 7: Add Credentials via Dashboard

1. Open `http://your-server:8080/credentials`
2. Add each required credential:
   - PRIVATE_KEY (EVM wallet)
   - WALLET_ADDRESS
   - BINANCE_API_KEY/SECRET (if using futures)
   - TELEGRAM_BOT_TOKEN (for notifications)
   - etc.

### Step 8: Secure the Installation

1. Change database password
2. Change Redis password
3. Generate new encryption key
4. Update Docker secrets

---

## Troubleshooting

### Issue: "Encryption key not found"

**Solution:**
```bash
# Check if key file exists
ls -la /secure/encryption.key

# Check permissions
sudo chmod 600 /secure/encryption.key

# Verify mount in Docker
docker exec -it trading-bot cat /run/secrets/encryption_key
```

### Issue: "Database connection failed"

**Solution:**
```bash
# Check database is running
docker ps | grep postgres

# Check connection
docker exec -it trading-postgres psql -U bot_user -d tradingbot -c "SELECT 1"
```

### Issue: "Credential not found" but it's in .env

**Solution:**
```bash
# Check if migration was run
docker exec -it trading-bot python scripts/migrate_credentials_to_db.py --validate

# Re-run migration if needed
docker exec -it trading-bot python scripts/migrate_credentials_to_db.py
```

### Issue: Dashboard shows "Bootstrap Mode"

**Meaning:** System is using .env fallback, not database.

**Solution:**
1. Ensure database migration ran successfully
2. Check database connection
3. Verify credentials in database:
```bash
docker exec -it trading-postgres psql -U bot_user -d tradingbot -c \
  "SELECT key_name, encrypted_value != 'PLACEHOLDER' as has_value FROM secure_credentials"
```

### Issue: Can't decrypt credentials

**Cause:** Encryption key changed since credentials were stored.

**Solution:**
```bash
# Re-encrypt with new key
docker exec -it trading-bot python scripts/migrate_credentials_to_db.py
```

---

## Summary

### What You Need to Do

1. ✅ Run database migration
2. ✅ Migrate credentials from .env to database
3. ✅ Access dashboard at `/credentials`
4. ✅ Move encryption key to `/secure/encryption.key`
5. ✅ Update docker-compose.yml
6. ✅ Remove sensitive values from .env
7. ✅ Clean GitHub history (optional but recommended)
8. ✅ Generate NEW credentials (wallets, API keys)

### Security Reminders

- **NEVER** store encryption key in the same location as encrypted data
- **NEVER** commit .env files to git
- **ALWAYS** use Docker secrets or external key storage in production
- **ROTATE** credentials regularly
- **MONITOR** credential access through dashboard

### Support

For issues, create a GitHub issue with:
- Error messages
- Steps to reproduce
- Docker logs (`docker logs trading-bot`)
