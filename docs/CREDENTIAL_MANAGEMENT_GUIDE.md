# ClaudeDex Credential Management Guide

This guide covers three critical operations for managing secure credentials:

1. **Encryption Key Rotation** - How to safely rotate the master encryption key
2. **Clearing Trade Records** - How to clear paper trade data before live trading
3. **Wallet Replacement** - How to update wallet credentials from the dashboard

---

## Table of Contents

- [1. Understanding the Security Architecture](#1-understanding-the-security-architecture)
- [2. Encryption Key Rotation Guide](#2-encryption-key-rotation-guide)
- [3. Clearing Trade Records Guide](#3-clearing-trade-records-guide)
- [4. Wallet Replacement Guide](#4-wallet-replacement-guide)
- [5. Troubleshooting](#5-troubleshooting)

---

## 1. Understanding the Security Architecture

### How Credentials Are Stored

```
┌─────────────────────────────────────────────────────────────────┐
│                    .encryption_key file                         │
│              (44-byte Fernet key, base64 encoded)               │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                  secure_credentials table                        │
│                                                                  │
│  key_name          │ encrypted_value                             │
│  ─────────────────────────────────────────────────────────────  │
│  PRIVATE_KEY       │ gAAAAABxxxxxxxxx...  (Fernet encrypted)    │
│  BINANCE_API_KEY   │ gAAAAABxxxxxxxxx...  (Fernet encrypted)    │
│  TELEGRAM_BOT_TOKEN│ gAAAAABxxxxxxxxx...  (Fernet encrypted)    │
└─────────────────────────────────────────────────────────────────┘
```

### Key Points

- **Encryption Key**: Stored in `.encryption_key` file (NOT in database)
- **Credentials**: Stored in `secure_credentials` table, encrypted with Fernet
- **Fernet Format**: Encrypted values start with `gAAAAAB`
- **Never in .env**: API keys, private keys, and tokens are NOT stored in .env

---

## 2. Encryption Key Rotation Guide

### When to Rotate

- After a suspected security breach
- When team members with access leave
- Periodically (recommended: every 90 days)
- When updating from plaintext to encrypted storage

### Prerequisites

1. **Stop all trading modules** - Critical to avoid data corruption
2. **Backup your database** - Use pg_dump or database backup
3. **Backup current encryption key** - The script does this automatically

### Step-by-Step Rotation

#### Step 1: Connect to Docker Container

```bash
docker exec -it trading-bot bash
```

#### Step 2: Verify Current Encryption (Optional)

```bash
python scripts/rotate_encryption_key.py --verify
```

This checks that all credentials can be decrypted with the current key.

#### Step 3: Preview the Rotation (Dry Run)

```bash
python scripts/rotate_encryption_key.py --dry-run
```

This shows what will be rotated without making changes.

#### Step 4: Perform the Rotation

```bash
python scripts/rotate_encryption_key.py
```

The script will:
1. Ask for confirmation (type `ROTATE`)
2. Backup current key to `backups/encryption_keys/`
3. Decrypt all credentials with old key
4. Generate new Fernet key
5. Re-encrypt all credentials with new key
6. Update the database
7. Save new key to `.encryption_key`

#### Step 5: Restart All Modules

```bash
docker-compose restart
```

Or restart individual modules:
```bash
docker-compose restart dex futures solana
```

### Rollback Procedure

If rotation fails, restore from backup:

```bash
# Find your backup
ls -la backups/encryption_keys/

# Restore the old key
cp backups/encryption_keys/encryption_key_YYYYMMDD_HHMMSS.bak .encryption_key
chmod 600 .encryption_key

# Restart modules
docker-compose restart
```

### Manual Key Rotation (Advanced)

If you need to manually rotate:

```python
from cryptography.fernet import Fernet

# Generate new key
new_key = Fernet.generate_key()
print(f"New key: {new_key.decode()}")

# Save to file
with open('.encryption_key', 'wb') as f:
    f.write(new_key)
```

Then re-encrypt credentials via the Settings UI or database directly.

---

## 3. Clearing Trade Records Guide

### When to Clear

- Before switching from DRY_RUN to live trading
- After testing strategies
- To reset performance metrics
- Before starting fresh with new strategies

### Prerequisites

1. **Stop all trading modules** - Prevents new records during clearing
2. **Export data if needed** - Use pg_dump for backup

### Step-by-Step Clearing

#### Step 1: Connect to Docker Container

```bash
docker exec -it trading-bot bash
```

#### Step 2: List Current Records

```bash
python scripts/clear_trade_records.py --list
```

This shows all tables and record counts:
```
📦 DEX
   trades: 📊 150 records
   positions: ✅ (empty)
   performance_metrics: 📊 50 records

📦 FUTURES
   futures_trades: 📊 75 records
   futures_positions: 📊 2 records
...
TOTAL RECORDS: 277
```

#### Step 3: Preview Deletion (Dry Run)

```bash
# Preview clearing specific module
python scripts/clear_trade_records.py --dry-run --module futures

# Preview clearing all modules
python scripts/clear_trade_records.py --dry-run --all
```

#### Step 4: Clear Records

**Clear specific module:**
```bash
python scripts/clear_trade_records.py --module futures
```
Type `futures` to confirm.

**Clear ALL modules:**
```bash
python scripts/clear_trade_records.py --all
```
Type `DELETE ALL TRADES` to confirm.

**Clear with sequence reset:**
```bash
python scripts/clear_trade_records.py --all --reset-sequences
```

### Available Modules

| Module | Tables |
|--------|--------|
| `dex` | trades, positions, performance_metrics, market_data |
| `futures` | futures_trades, futures_positions |
| `solana` | solana_trades, solana_positions |
| `arbitrage` | arbitrage_trades, arbitrage_positions, arbitrage_opportunities |
| `copytrading` | copytrading_trades, copytrading_positions |
| `sniper` | sniper_trades, sniper_positions |
| `ai` | ai_trades, ai_positions, ai_signals |

### Clearing from Dashboard (Future Feature)

A dashboard button can be added to `/settings/credentials` page to clear trade records.
The endpoint would call the same logic as the script.

---

## 4. Wallet Replacement Guide

### When to Replace

- Setting up new wallet
- Rotating wallet for security
- Switching between wallets
- Recovering from compromised wallet

### Method 1: Via Dashboard (Recommended)

#### Step 1: Access Credentials Page

Navigate to: `http://your-server:8080/settings/credentials`

#### Step 2: Find Wallet Credentials

Look under the **"Wallet"** category for:
- `PRIVATE_KEY` (EVM wallet)
- `WALLET_ADDRESS` (EVM address)
- `SOLANA_PRIVATE_KEY` (Solana wallet)
- `SOLANA_WALLET` (Solana address)

#### Step 3: Update Private Key

1. Click the **Edit** (pencil icon) button next to `PRIVATE_KEY`
2. Enter your new private key in the secure input field
   - For EVM: Enter the 64-character hex string (without 0x prefix)
   - For Solana: Enter the base58 encoded private key
3. Click **Save**

The system will automatically:
- Encrypt the new key with Fernet
- Store it in `secure_credentials` table
- Update the `updated_at` timestamp

#### Step 4: Update Wallet Address

1. Click **Edit** next to `WALLET_ADDRESS`
2. Enter the corresponding public address
3. Click **Save**

#### Step 5: Verify

1. Check the status indicator shows ✅ (configured)
2. Restart the module to pick up the new wallet
3. Check logs for successful wallet initialization

### Method 2: Via Script

```bash
docker exec -it trading-bot bash

# Use the credentials migration script with a single key
python -c "
import asyncio
from cryptography.fernet import Fernet

# Load encryption key
with open('.encryption_key', 'rb') as f:
    key = f.read()
fernet = Fernet(key)

# Your new private key
new_private_key = 'YOUR_NEW_PRIVATE_KEY_HERE'

# Encrypt it
encrypted = fernet.encrypt(new_private_key.encode()).decode()
print(f'Encrypted value: {encrypted}')
"
```

Then update the database:

```sql
-- Connect to database (use your actual db_user from secrets/db_user)
docker exec -it trading-postgres psql -U <your_db_user> -d tradingbot

-- Update the credential
UPDATE secure_credentials
SET encrypted_value = 'gAAAAABxxxxx...',  -- The encrypted value from above
    updated_at = NOW()
WHERE key_name = 'PRIVATE_KEY';

-- Verify
SELECT key_name, LEFT(encrypted_value, 20) as encrypted_preview, updated_at
FROM secure_credentials
WHERE key_name = 'PRIVATE_KEY';
```

### Method 3: Via Python Script (Full Automation)

```python
#!/usr/bin/env python3
"""Update wallet credential"""
import asyncio
import os
from pathlib import Path

async def update_wallet(key_name: str, new_value: str):
    # Load encryption key
    key_file = Path('.encryption_key')
    encryption_key = key_file.read_bytes()

    from cryptography.fernet import Fernet
    fernet = Fernet(encryption_key)

    # Encrypt new value
    encrypted = fernet.encrypt(new_value.encode()).decode()

    # Update database - requires DB_USER and DB_PASSWORD env vars
    import asyncpg
    pool = await asyncpg.create_pool(
        host=os.getenv('DB_HOST', 'postgres'),
        database=os.getenv('DB_NAME', 'tradingbot'),
        user=os.getenv('DB_USER'),  # Required - set explicitly
        password=os.getenv('DB_PASSWORD')  # Required - set explicitly
    )

    async with pool.acquire() as conn:
        await conn.execute("""
            UPDATE secure_credentials
            SET encrypted_value = $1, updated_at = NOW()
            WHERE key_name = $2
        """, encrypted, key_name)

    await pool.close()
    print(f"✅ Updated {key_name}")

# Run
asyncio.run(update_wallet('PRIVATE_KEY', 'your_new_private_key'))
```

### Wallet Types and Key Names

| Wallet Type | Private Key Field | Address Field |
|-------------|-------------------|---------------|
| EVM (Ethereum, BSC, etc.) | `PRIVATE_KEY` | `WALLET_ADDRESS` |
| Solana (DEX Module) | `SOLANA_PRIVATE_KEY` | `SOLANA_WALLET` |
| Solana (Solana Module) | `SOLANA_MODULE_PRIVATE_KEY` | `SOLANA_MODULE_WALLET` |
| Flashbots Signing | `FLASHBOTS_SIGNING_KEY` | - |

### Security Best Practices

1. **Never share your private key** - Even encrypted values
2. **Use hardware wallets** when possible for signing
3. **Generate new wallets** instead of reusing compromised ones
4. **Verify wallet address** matches the private key before trading
5. **Test with small amount** after wallet replacement

---

## 5. Troubleshooting

### "Failed to decrypt credential - key mismatch"

**Cause**: Encryption key doesn't match the one used to encrypt credentials.

**Solution**:
1. Restore the correct `.encryption_key` from backup
2. Or re-import credentials with current key:
   ```bash
   python scripts/import_env_secrets.py
   ```

### "Encryption key not loaded"

**Cause**: `.encryption_key` file missing or unreadable.

**Solution**:
1. Check file exists: `ls -la .encryption_key`
2. Check permissions: `chmod 600 .encryption_key`
3. If lost, create new key and re-import credentials

### "PRIVATE_KEY not found"

**Cause**: Credential not in database or sync issue.

**Solution**:
1. Check credentials page in dashboard
2. Add via UI or import from .env:
   ```bash
   python scripts/migrate_credentials_to_db.py
   ```

### "Non-hexadecimal digit found" (wallet error)

**Cause**: Private key still encrypted when used.

**Solution**:
1. Verify encryption key is loaded: Check logs for "Loaded encryption key"
2. Verify decryption works: `python scripts/rotate_encryption_key.py --verify`
3. Re-encrypt with correct key if needed

### Cannot connect to database

**Cause**: Database credentials issue.

**Solution**:
1. Check Docker secrets: `cat /run/secrets/db_password`
2. Check environment: `echo $DB_PASSWORD`
3. Test connection: `pg_isready -h postgres -p 5432`

---

## Quick Reference Commands

```bash
# Verify encryption
python scripts/rotate_encryption_key.py --verify

# Rotate encryption key
python scripts/rotate_encryption_key.py

# List trade records
python scripts/clear_trade_records.py --list

# Clear futures trades
python scripts/clear_trade_records.py --module futures

# Clear ALL trades
python scripts/clear_trade_records.py --all

# Backup encryption key
cp .encryption_key backups/encryption_key_$(date +%Y%m%d).bak

# Check credential in database (use your actual db_user)
docker exec -it trading-postgres psql -U <your_db_user> -d tradingbot -c \
  "SELECT key_name, category, is_encrypted, updated_at FROM secure_credentials WHERE key_name = 'PRIVATE_KEY';"
```

---

*Last Updated: December 2024*
