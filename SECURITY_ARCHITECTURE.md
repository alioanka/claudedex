# ClaudeDex Security Architecture - Post-Incident Redesign

## Executive Summary

This document outlines a comprehensive security redesign following a credential compromise incident. The goal is to implement defense-in-depth security that protects wallet private keys, API credentials, and other sensitive data even if one layer is breached.

---

## Part 1: Vulnerability Analysis - What Went Wrong

### Critical Vulnerabilities Found

#### 1. `.env.final` Committed to Git Repository
**Severity: CRITICAL**
- A file named `.env.final` was tracked in git and pushed to GitHub
- This file contained partial real credentials including:
  - Partial Telegram bot token
  - Partial email addresses
  - Partial RPC URLs with API keys
  - Partial wallet information
- Even partial information can be used for phishing, social engineering, or brute-force attacks

#### 2. Encryption Key Collocated with Encrypted Data
**Severity: CRITICAL**
- The `ENCRYPTION_KEY` is stored in the same `.env` file as the encrypted private keys
- This completely defeats the purpose of encryption
- If an attacker gains access to `.env`, they have both the encrypted data AND the key to decrypt it
- Current pattern:
  ```
  ENCRYPTION_KEY=bvSd97GRf4nlMqu5ISd7S62VkvtR4NElwDwQ9V-DINU=  # THE KEY
  PRIVATE_KEY=gAAAAABo_94xUqGQf... # ENCRYPTED WITH THE KEY ABOVE
  ```

#### 3. API Keys Stored in Plain Text in Database
**Severity: HIGH**
- The `rpc_api_pool.api_key` column stores API keys without encryption
- Database compromise exposes all API keys
- Dashboard can display these keys in the UI

#### 4. Single Point of Failure - All Secrets in One File
**Severity: HIGH**
- All credentials in a single `.env` file
- Database credentials, API keys, wallet keys all together
- Compromise of one = compromise of all

#### 5. Dashboard Exposes Sensitive Data
**Severity: MEDIUM**
- Settings pages display API keys and credentials
- No role-based access control for sensitive fields
- API endpoints return raw credentials

---

## Part 2: Secure Architecture Design

### Principle: Defense in Depth

No single layer should be sufficient to access sensitive credentials. We implement multiple security layers:

```
Layer 1: Access Control (VPS/Server Security)
    |
Layer 2: Encrypted Storage (File-level)
    |
Layer 3: Separated Key Management (Encryption key NOT with data)
    |
Layer 4: Per-Credential Encryption (Each secret encrypted separately)
    |
Layer 5: Hardware/External Security (HSM/Vault - Production)
```

### Architecture Overview

```
                           +------------------+
                           |   HashiCorp      |
                           |   Vault / AWS    |  <-- Production: External Secrets Manager
                           |   Secrets Mgr    |
                           +--------+---------+
                                    |
                                    v
+------------------+      +------------------+      +------------------+
|   Environment    |      | Secrets Manager  |      |   Application    |
|   Variables      | ---> |   (Local)        | ---> |   Runtime        |
|   (Bootstrap)    |      |                  |      |   Memory Only    |
+------------------+      +------------------+      +------------------+
        |                         |                         |
        v                         v                         v
+------------------+      +------------------+      +------------------+
| Master Key Only  |      | Encrypted        |      | Decrypted for    |
| (1 secret)       |      | Credential Store |      | Use, Never Saved |
+------------------+      +------------------+      +------------------+
```

### Credential Categories

| Category | Examples | Storage Method |
|----------|----------|----------------|
| **CRITICAL** | Wallet Private Keys | Hardware wallet / External HSM / Encrypted with SEPARATE key |
| **HIGH** | Exchange API Keys, Encryption Keys | Vault / Encrypted file NOT in .env |
| **MEDIUM** | Database Passwords, RPC URLs | Environment variables (not in git) |
| **LOW** | Feature Flags, Public Config | .env or Database |

---

## Part 3: Implementation Guide

### Step 1: Immediate Actions (Do Now)

#### 1.1 Revoke ALL Compromised Credentials
```bash
# These are compromised and MUST be regenerated:
- All wallet private keys (create NEW wallets)
- All exchange API keys (Binance, Bybit)
- All RPC API keys (Alchemy, Infura, QuickNode, Ankr, Helius)
- Encryption key (generate new)
- JWT/Session secrets (generate new)
- Database password (change)
- Redis password (change)
- Email password (change)
- Telegram bot token (regenerate)
- All API keys (OpenAI, Anthropic, etc.)
```

#### 1.2 Remove Sensitive Data from Git History
```bash
# Install BFG Repo-Cleaner
wget https://repo1.maven.org/maven2/com/madgag/bfg/1.14.0/bfg-1.14.0.jar

# Create a backup
cp -r . ../claudedex-backup

# Remove .env.final from all history
java -jar bfg-1.14.0.jar --delete-files .env.final

# Clean up
git reflog expire --expire=now --all
git gc --prune=now --aggressive

# Force push (requires --force, be careful!)
git push origin --force --all
```

### Step 2: New Credential Storage Architecture

#### 2.1 Separate Encryption Key Storage

Create a new file structure:
```
/secure/                    # Directory with 700 permissions
  encryption.key           # Master encryption key (600 permissions)

/home/user/claudedex/
  .env                     # Contains ONLY bootstrap config
  credentials.encrypted    # Encrypted credentials file
```

**New `.env` structure (minimal):**
```bash
# ONLY bootstrap configuration - NO SECRETS
MODE=production
DRY_RUN=true

# Module flags
DEX_MODULE_ENABLED=true
FUTURES_MODULE_ENABLED=true

# Database connection (use Docker secrets in production)
DATABASE_URL=postgresql://bot_user@postgres:5432/tradingbot

# Point to external key storage
ENCRYPTION_KEY_FILE=/secure/encryption.key
CREDENTIALS_FILE=/home/user/claudedex/credentials.encrypted
```

#### 2.2 Create Encrypted Credentials Store

Create a Python script to manage encrypted credentials:

```python
# scripts/secure_credentials.py
import os
import json
from cryptography.fernet import Fernet
from pathlib import Path

class SecureCredentialStore:
    def __init__(self):
        # Key is stored SEPARATELY from encrypted data
        key_file = os.getenv('ENCRYPTION_KEY_FILE', '/secure/encryption.key')
        creds_file = os.getenv('CREDENTIALS_FILE', 'credentials.encrypted')

        if not os.path.exists(key_file):
            raise RuntimeError(f"Encryption key not found: {key_file}")

        with open(key_file, 'rb') as f:
            self.key = f.read().strip()

        self.fernet = Fernet(self.key)
        self.creds_file = creds_file
        self._cache = None

    def get(self, key: str) -> str:
        """Get a credential by key"""
        if self._cache is None:
            self._load()
        return self._cache.get(key)

    def _load(self):
        """Load and decrypt credentials"""
        if not os.path.exists(self.creds_file):
            self._cache = {}
            return

        with open(self.creds_file, 'rb') as f:
            encrypted = f.read()

        decrypted = self.fernet.decrypt(encrypted)
        self._cache = json.loads(decrypted)

    def set(self, key: str, value: str):
        """Set a credential"""
        if self._cache is None:
            self._load()
        self._cache[key] = value
        self._save()

    def _save(self):
        """Encrypt and save credentials"""
        data = json.dumps(self._cache).encode()
        encrypted = self.fernet.encrypt(data)
        with open(self.creds_file, 'wb') as f:
            f.write(encrypted)

# Initialize store
credentials = SecureCredentialStore()
```

#### 2.3 Credential Categories and Storage

**Category 1: CRITICAL (Wallet Private Keys)**
- NEVER store on internet-connected machine for large amounts
- Use hardware wallet (Ledger/Trezor) for main funds
- Use hot wallet with MINIMAL funds for trading
- Encrypt with key stored on SEPARATE secure system

```bash
# Structure for wallet keys
/secure/wallets/
  evm_trading.encrypted      # Encrypted with master key
  solana_trading.encrypted   # Encrypted with master key

# Master key NOT on same system as encrypted files in production
# For VPS: Store master key in cloud secrets manager (AWS/GCP)
```

**Category 2: HIGH (Exchange APIs)**
```python
# Exchange APIs should be:
# 1. IP-whitelisted on exchange
# 2. Have minimal permissions (no withdrawal)
# 3. Rotated regularly (monthly)

# Store in encrypted credentials file
credentials.set('BINANCE_API_KEY', 'your_key')
credentials.set('BINANCE_API_SECRET', 'your_secret')
```

**Category 3: MEDIUM (Database/Redis)**
```bash
# Use Docker secrets in production
docker secret create db_password <password_file>

# Or use environment-specific files
# /secure/db.env - NOT in git, permissions 600
DATABASE_PASSWORD=secure_password
```

### Step 3: Code Changes Required

#### 3.1 Update config_manager.py

```python
# config/config_manager.py - Updated

from scripts.secure_credentials import credentials

class ConfigManager:
    def __init__(self):
        self._secure_store = credentials

    def get_api_key(self, name: str) -> str:
        """Get API key from secure store"""
        return self._secure_store.get(name)

    def get_private_key(self, wallet_type: str) -> str:
        """Get wallet private key - decrypts on demand"""
        key = self._secure_store.get(f'{wallet_type}_PRIVATE_KEY')
        if not key:
            raise ValueError(f"Private key not found for {wallet_type}")
        return key
```

#### 3.2 Update Database Schema

```sql
-- Migration: Encrypt API keys in database
ALTER TABLE rpc_api_pool
    ADD COLUMN api_key_encrypted TEXT,
    ADD COLUMN encryption_version INTEGER DEFAULT 1;

-- Remove plain text column after migration
-- ALTER TABLE rpc_api_pool DROP COLUMN api_key;
```

#### 3.3 Update Dashboard to Hide Sensitive Data

```python
# monitoring/enhanced_dashboard.py

def mask_sensitive(value: str) -> str:
    """Mask sensitive data for display"""
    if not value or len(value) < 8:
        return '****'
    return value[:4] + '****' + value[-4:]

# In API endpoints:
@app.route('/api/settings/credentials')
def get_credentials():
    # NEVER return raw credentials
    return {
        'binance_api_key': mask_sensitive(creds.get('BINANCE_API_KEY')),
        'binance_api_secret': '********',  # NEVER expose secrets
        # ...
    }
```

### Step 4: Production Security Recommendations

#### 4.1 Use HashiCorp Vault (Recommended for Production)

```bash
# Install Vault
wget https://releases.hashicorp.com/vault/1.15.0/vault_1.15.0_linux_amd64.zip
unzip vault_1.15.0_linux_amd64.zip
sudo mv vault /usr/local/bin/

# Initialize
vault server -dev  # For testing
vault kv put secret/claudedex/wallets evm_pk=xxx solana_pk=xxx
```

```python
# Integration with Vault
import hvac

class VaultCredentials:
    def __init__(self):
        self.client = hvac.Client(
            url=os.getenv('VAULT_ADDR'),
            token=os.getenv('VAULT_TOKEN')
        )

    def get(self, path: str) -> dict:
        return self.client.secrets.kv.read_secret_version(path)['data']['data']
```

#### 4.2 Use Hardware Wallet for Main Funds

- Keep main funds in Ledger/Trezor
- Only transfer small amounts to hot wallet for trading
- Set up alerts for large transactions

#### 4.3 IP Whitelisting

- Whitelist your VPS IP on all exchanges
- Use VPN with static IP if working remotely
- Enable 2FA on all exchange accounts

#### 4.4 Monitoring & Alerts

```python
# Add security monitoring
async def security_monitor():
    """Monitor for suspicious activity"""
    # Alert on:
    # - Failed login attempts
    # - API key usage from unknown IPs
    # - Large transactions
    # - Unusual trading patterns
```

---

## Part 4: Migration Checklist

### Before Going Live

- [ ] Generate new encryption key (store in /secure/)
- [ ] Create new wallets (NEVER reuse compromised ones)
- [ ] Generate new exchange API keys (with IP whitelist)
- [ ] Regenerate all RPC API keys
- [ ] Change database password
- [ ] Change Redis password
- [ ] Regenerate JWT/session secrets
- [ ] Regenerate Telegram bot
- [ ] Change email password (use app-specific password)
- [ ] Remove .env.final from git history
- [ ] Audit all files in git for sensitive data
- [ ] Update .gitignore
- [ ] Implement new credential storage
- [ ] Update all code to use new credential system
- [ ] Test in dry-run mode first
- [ ] Set up monitoring and alerts

### File Permissions

```bash
# /secure/ directory
chmod 700 /secure/
chmod 600 /secure/encryption.key
chmod 600 /secure/wallets/*

# Application directory
chmod 600 credentials.encrypted
chmod 600 .env
```

---

## Part 5: Quick Reference

### Generating New Secure Keys

```python
# Generate new encryption key
from cryptography.fernet import Fernet
key = Fernet.generate_key()
print(key.decode())

# Generate secure random strings
import secrets
print(secrets.token_urlsafe(32))  # For JWT, session secrets
print(secrets.token_hex(16))      # For API keys
```

### Environment Variable Best Practices

```bash
# DON'T:
PRIVATE_KEY=0x123abc...  # Raw private key
ENCRYPTION_KEY=xxx       # Key with encrypted data

# DO:
ENCRYPTION_KEY_FILE=/secure/encryption.key  # Point to external file
VAULT_ADDR=https://vault.example.com        # Use secrets manager
```

---

## Conclusion

The compromise likely occurred because:
1. `.env.final` with partial credentials was pushed to GitHub
2. The encryption key was stored alongside encrypted data
3. API keys were stored in plain text in the database

To prevent future incidents:
1. **Never store encryption keys with encrypted data**
2. **Never commit any .env files to git**
3. **Use external secrets management in production**
4. **Implement monitoring and alerts**
5. **Keep minimal funds in hot wallets**

Remember: **Assume the old credentials are fully compromised. Create NEW wallets and API keys for everything.**
