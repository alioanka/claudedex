# ClaudeDex Trading Bot - Architecture Summary

## Overview

Multi-module crypto trading bot supporting DEX trading (EVM + Solana), Futures (Binance/Bybit), AI analysis, arbitrage, copy trading, and sniping. Built with Python/FastAPI, PostgreSQL (TimescaleDB), and Redis.

---

## Configuration Architecture

```
┌─────────────────┐                    ┌───────────────────────┐
│   .env file     │                    │   secure_credentials  │
│ (module flags,  │                    │      (PostgreSQL)     │
│  non-sensitive) │                    │   Fernet Encrypted    │
└────────┬────────┘                    └───────────┬───────────┘
         │                                         │
         │  Module Startup                         │  API Keys, Private Keys,
         │  Flags Only                             │  Telegram Tokens, etc.
         │                                         │
         ▼                                         ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Secrets Manager                              │
│              security/secrets_manager.py                         │
│                                                                  │
│  Priority: Docker Secrets → Database → Environment → Default    │
│  Auto-decrypts values starting with 'gAAAAAB'                   │
└────────────────────────────┬────────────────────────────────────┘
                             │
         ┌───────────────────┼───────────────────┐
         ▼                   ▼                   ▼
┌─────────────────┐  ┌───────────────┐  ┌───────────────────┐
│  Config Manager │  │ Trading Engine │  │  Alert Systems    │
│  (loads from DB)│  │ (uses secrets) │  │  (Telegram, etc.) │
└─────────────────┘  └───────────────┘  └───────────────────┘
         ▲
         │
┌─────────────────┐
│   Settings UI   │
│  (saves to DB)  │
└─────────────────┘
```

**Key Principle:** `.env` contains ONLY module enable flags and non-sensitive configuration. ALL credentials (API keys, private keys, tokens) are stored encrypted in PostgreSQL.

---

## Module Architecture

```
main.py (Orchestrator)
├── DEX Module        → main_dex.py (EVM chains via Uniswap/1inch)
├── Futures Module    → modules/futures_trading/main_futures.py (Binance/Bybit)
├── Solana Module     → modules/solana_trading/main_solana.py (Jupiter/Drift)
├── AI Analysis       → modules/ai_analysis/main_ai.py (OpenAI/Anthropic)
├── Arbitrage         → modules/arbitrage/main_arbitrage.py (Cross-DEX + Triangular)
├── Copy Trading      → modules/copy_trading/main_copy.py (Wallet tracking)
└── Sniper            → modules/sniper/main_sniper.py (Token launch sniping)
```

Each module runs as a **separate subprocess** managed by the orchestrator (`main.py`).

---

## Current .env Structure

The `.env` file now contains **ONLY**:

```bash
# ============================================================================
# MODULE ENABLE/DISABLE FLAGS
# ============================================================================
# These control which module processes are started
DEX_MODULE_ENABLED=true
FUTURES_MODULE_ENABLED=true
SOLANA_MODULE_ENABLED=true
SNIPER_MODULE_ENABLED=true
AI_MODULE_ENABLED=true
ARBITRAGE_MODULE_ENABLED=true
COPY_TRADING_MODULE_ENABLED=true

# Operating mode (affects all modules)
MODE=production
DRY_RUN=true  # CRITICAL: Set to false ONLY when ready for live trading!

# External Service URLs (non-sensitive)
FLASHBOTS_RPC=https://relay.flashbots.net
JUPITER_API_URL=https://lite-api.jup.ag
JITO_TIP_ACCOUNT=96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5
JITO_BLOCK_ENGINE_URL=https://mainnet.block-engine.jito.wtf
JUPYTER_TOKEN=...
```

**What is NOT in .env (stored in database instead):**
- Private keys (PRIVATE_KEY, SOLANA_PRIVATE_KEY)
- API keys (BINANCE_API_KEY, OPENAI_API_KEY, etc.)
- API secrets (BINANCE_API_SECRET, etc.)
- Notification tokens (TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, DISCORD_WEBHOOK_URL)
- All exchange credentials

---

## Credentials & Security

### Two-Tier Security Model

**Tier 1: Docker Secrets** (`/run/secrets/`)
- Database password (`db_password`)
- Redis password (`redis_password`)
- Infrastructure-level credentials

**Tier 2: Encrypted Database** (`secure_credentials` table)
- Private keys (Fernet encrypted)
- API keys (Fernet encrypted)
- Exchange secrets
- Notification tokens

### Secrets Manager

**File:** `security/secrets_manager.py`

**Class:** `SecureSecretsManager`

**Priority Order:**
1. Docker secrets (`/run/secrets/<key>`)
2. Memory cache (if not encrypted)
3. Database (`secure_credentials` table with Fernet decryption)
4. Environment variables (with Fernet decryption if needed)
5. Default value

**Key Methods:**

| Method | Use Case |
|--------|----------|
| `secrets.get(key)` | Sync access (skips DB in async context) |
| `await secrets.get_async(key)` | **Preferred** - async DB access with decryption |
| `secrets.has(key)` | Check if credential exists |
| `await secrets.set(key, value)` | Store encrypted credential |

**Usage Patterns:**

```python
from security.secrets_manager import secrets

# In async context (preferred)
api_key = await secrets.get_async('BINANCE_API_KEY', log_access=False)

# In sync context (will skip DB lookup in async event loop)
api_key = secrets.get('BINANCE_API_KEY')
```

**Pre-loading Pattern for Sync Constructors:**

When a class has a sync `__init__` but needs credentials from the database:

```python
async def initialize(self):
    # Pre-load credentials asynchronously
    from security.secrets_manager import secrets

    bot_token = await secrets.get_async('TELEGRAM_BOT_TOKEN', log_access=False)
    chat_id = await secrets.get_async('TELEGRAM_CHAT_ID', log_access=False)

    # Pass to sync constructor
    self.alerts = TelegramAlerts(bot_token=bot_token, chat_id=chat_id)
```

**Final Safety Checks:**

The secrets manager includes automatic decryption safety:
- If a returned value starts with `gAAAAAB` (Fernet prefix), it will attempt decryption
- Prevents accidentally returning encrypted values to callers
- Logs warnings if decryption fails

### Encryption

**File:** `.encryption_key` (project root)
- 32-byte Fernet key (base64 encoded)
- Used to encrypt/decrypt all credentials in database
- **NEVER commit to git** (in `.gitignore`)

**Credential Categories in Database:**

| Category | Examples |
|----------|----------|
| `wallet` | PRIVATE_KEY, SOLANA_PRIVATE_KEY, WALLET_ADDRESS |
| `exchange` | BINANCE_API_KEY, BYBIT_API_SECRET |
| `api` | OPENAI_API_KEY, ANTHROPIC_API_KEY |
| `notification` | TELEGRAM_BOT_TOKEN, DISCORD_WEBHOOK_URL |
| `rpc` | ALCHEMY_API_KEY, HELIUS_API_KEY |

---

## Settings Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌────────────────────┐
│  Dashboard UI   │ ───► │   PostgreSQL    │ ───► │  Config Manager    │
│  /settings/*    │      │ config_settings │      │  *_config_manager  │
└─────────────────┘      └─────────────────┘      └────────────────────┘
                                                           │
                                                           ▼
                                                  ┌────────────────────┐
                                                  │  Trading Engine    │
                                                  └────────────────────┘
```

**Settings Pages:**
- `/settings` - Global module control, DRY_RUN mode
- `/settings/credentials` - Secure credential management
- `/settings/rpc-api` - RPC/API Pool configuration
- `/futures/settings` - Futures trading parameters
- `/solana/settings` - Solana module settings
- `/dex/settings` - DEX trading settings

**Config Managers (per module):**

| Module | Config Manager File | Key Methods |
|--------|---------------------|-------------|
| DEX | `config/config_manager.py` | `get_config()`, `update_config()` |
| Futures | `modules/futures_trading/config/futures_config_manager.py` | `get_api_credentials()`, `initialize()` |
| Solana | `modules/solana_trading/config/solana_config_manager.py` | `get_trading_config()` |
| AI Analysis | Settings in `sentiment_engine.py` | Direct config loading |

**Database Tables for Settings:**

| Table | Purpose |
|-------|---------|
| `config_settings` | All module trading parameters |
| `secure_credentials` | Encrypted API keys and secrets |
| `rpc_api_pool` | RPC endpoint configuration and health |

---

## Dashboard (Web UI)

**Location:** `monitoring/` (FastAPI + Jinja2 templates)

### Main Pages

| Route | Template | Purpose |
|-------|----------|---------|
| `/` | `index.html` | Main dashboard overview |
| `/modules` | `modules.html` | Module control panel (start/stop/restart) |
| `/settings` | `settings.html` | Global settings |

### Module-Specific Dashboards

| Module | Dashboard | Settings | Trades | Positions | Performance |
|--------|-----------|----------|--------|-----------|-------------|
| DEX | `/dex` | `/dex/settings` | `/trades` | `/positions` | `/performance` |
| Futures | `/futures` | `/futures/settings` | `/futures/trades` | `/futures/positions` | `/futures/performance` |
| Solana | `/solana` | `/solana/settings` | `/solana/trades` | `/solana/positions` | `/solana/performance` |
| AI Analysis | `/ai` | `/ai/settings` | - | - | `/ai/performance` |
| Arbitrage | `/arbitrage` | `/arbitrage/settings` | `/arbitrage/trades` | `/arbitrage/positions` | `/arbitrage/performance` |
| Copy Trading | `/copytrading` | `/copytrading/settings` | `/copytrading/trades` | `/copytrading/positions` | `/copytrading/performance` |
| Sniper | `/sniper` | `/sniper/settings` | `/sniper/trades` | `/sniper/positions` | `/sniper/performance` |

### Special Pages

| Route | Template | Purpose |
|-------|----------|---------|
| `/settings/rpc-api` | `settings_rpc_api.html` | RPC/API Pool configuration with health monitoring |
| `/settings/credentials` | `settings_credentials.html` | Secure credential management (encrypted storage) |
| `/settings/global` | `global_settings.html` | System-wide settings |
| `/logs` | `logs.html` | Live log viewer |
| `/analytics` | `analytics.html` | Advanced analytics |
| `/reports` | `reports.html` | Performance reports |
| `/backtest` | `backtest.html` | Strategy backtesting |
| `/simulator` | `simulator.html` | Trade simulator |

### RPC/API Configuration Page (`/settings/rpc-api`)

**Backend:** `config/pool_engine.py`, `monitoring/rpc_pool_routes.py`

Features:
- Add/edit/delete RPC endpoints per chain
- Health status indicators (healthy/rate-limited/unhealthy)
- Latency monitoring and auto-rotation
- Priority ordering for endpoints
- Rate limit detection and cooldown display
- Test endpoints individually

### Secure Credentials Page (`/settings/credentials`)

**Backend:** `monitoring/credentials_routes.py`

Features:
- Category-based organization (Wallet, Exchange, API, Notification)
- Visual status indicators (configured/not-configured/required)
- Secure input fields (password-masked)
- Auto-encryption on save using Fernet
- Credential validation
- Audit logging of access

---

## RPC Pool Engine

**File:** `config/pool_engine.py`

Provides RPC rotation, health checking, and fallback for all chains.

**Key Methods:**

| Method | Purpose |
|--------|---------|
| `get_endpoint()` | Get best available endpoint for provider type |
| `report_success()` | Report successful request (improves health score) |
| `report_failure()` | Report failed request (decreases health score) |
| `report_rate_limit()` | Mark endpoint as rate-limited with cooldown |
| `run_health_checks()` | Periodic health check of all endpoints |

**Supported Chains:** Ethereum, BSC, Polygon, Arbitrum, Base, Solana, Monad, PulseChain, Fantom, Cronos, Avalanche

---

## Database Schema

**Engine:** PostgreSQL with TimescaleDB extensions

### Core Tables

| Table | Purpose |
|-------|---------|
| `trades` | All trade records (hypertable) |
| `positions` | Active positions |
| `market_data` | Price data (hypertable) |
| `alerts` | System alerts |
| `performance_metrics` | P&L tracking |
| `config_settings` | All module settings |
| `secure_credentials` | Encrypted API keys/secrets |
| `rpc_api_pool` | RPC endpoint configuration |

### Module-Specific Tables

| Module | Tables |
|--------|--------|
| Futures | `futures_trades`, `futures_positions` |
| Solana | `solana_trades`, `solana_positions` |
| Arbitrage | `arbitrage_opportunities`, `arbitrage_executions` |
| Copy Trading | `copy_trading_targets`, `copy_trading_executions` |
| Sniper | `sniper_targets`, `sniper_executions` |
| AI Analysis | `ai_signals`, `ai_positions` |
| Auth | `users`, `sessions`, `audit_log` |

---

## Key Files by Module

### Orchestrator
- `main.py` - Module process manager, pre-flight validation

### DEX Trading
| File | Purpose |
|------|---------|
| `main_dex.py` | Entry point, pre-loads credentials async |
| `core/engine.py` | Main trading engine |
| `trading/executors/base_executor.py` | EVM trade execution |
| `trading/chains/solana/jupiter_executor.py` | Solana Jupiter swaps |
| `config/config_manager.py` | DEX configuration |

### Futures Trading
| File | Purpose |
|------|---------|
| `modules/futures_trading/main_futures.py` | Entry point |
| `modules/futures_trading/core/futures_engine.py` | Trading engine, uses `get_async()` for credentials |
| `modules/futures_trading/core/futures_alerts.py` | Telegram alerts |
| `modules/futures_trading/config/futures_config_manager.py` | Config, calls `_reload_sensitive_credentials()` |

### Solana Trading
| File | Purpose |
|------|---------|
| `modules/solana_trading/main_solana.py` | Entry point |
| `modules/solana_trading/core/solana_engine.py` | Jupiter + trading logic |
| `modules/solana_trading/core/solana_alerts.py` | Telegram alerts |
| `modules/solana_trading/config/solana_config_manager.py` | Config |

### AI Analysis
| File | Purpose |
|------|---------|
| `modules/ai_analysis/main_ai.py` | Entry point |
| `modules/ai_analysis/core/sentiment_engine.py` | AI trading logic |

### Arbitrage
| File | Purpose |
|------|---------|
| `modules/arbitrage/main_arbitrage.py` | Entry point |
| `modules/arbitrage/arbitrage_engine.py` | Cross-DEX arb |
| `modules/arbitrage/triangular_engine.py` | Triangular arb |

### Copy Trading
| File | Purpose |
|------|---------|
| `modules/copy_trading/main_copy.py` | Entry point |
| `modules/copy_trading/copy_engine.py` | Wallet tracking + execution |

### Sniper
| File | Purpose |
|------|---------|
| `modules/sniper/main_sniper.py` | Entry point |
| `modules/sniper/core/sniper_engine.py` | Token launch detection |
| `modules/sniper/core/trade_executor.py` | Fast execution |

### Security
| File | Purpose |
|------|---------|
| `security/secrets_manager.py` | **Central credential management** |
| `security/encryption.py` | Fernet utilities, key loading |
| `security/wallet_security.py` | Wallet encryption/decryption |
| `security/audit_logger.py` | Access logging |

---

## Alerting System

**File:** `monitoring/alerts.py`

**Credential Loading:**
All alert modules load Telegram credentials using `secrets.get_async()` in their async `initialize()` method.

**Channels:**
- Telegram (credentials from `secure_credentials` table)
- Discord (webhook from `secure_credentials` table)
- Email (SMTP)

**Module-Specific Alerts:**

| Module | Alert File | Key Class |
|--------|------------|-----------|
| Futures | `modules/futures_trading/core/futures_alerts.py` | `FuturesTelegramAlerts` |
| Solana | `modules/solana_trading/core/solana_alerts.py` | `SolanaTelegramAlerts` |
| DEX | `monitoring/alerts.py` | `AlertManager` |

**Alert Types:**
- Trading: Position opened/closed, SL/TP hit
- Risk: Drawdown, margin call, correlation warning
- System: Errors, API limits, low balance
- Security: Honeypot detected, rug pull warning

---

## Logging

**Structure:**
```
logs/
├── orchestrator.log     # Main orchestrator
├── pool_engine/         # RPC Pool health logs
│   ├── pool_engine.log
│   ├── health_checks.log
│   └── rate_limits.log
├── dex/
│   └── dex.log
├── futures/
│   └── futures.log
├── solana/
│   └── solana.log
├── ai/
│   └── ai.log
├── arbitrage/
│   └── arbitrage.log
├── copy_trading/
│   └── copy_trading.log
└── sniper/
    └── sniper.log
```

---

## Quick Troubleshooting

### "BINANCE API keys required" or similar
1. Ensure credentials are stored in `secure_credentials` table
2. Check that `FuturesConfigManager.initialize()` is called (loads credentials async)
3. Verify `.encryption_key` exists and matches DB encryption

### "Telegram alerts disabled - missing bot token"
1. Store `TELEGRAM_BOT_TOKEN` and `TELEGRAM_CHAT_ID` in `secure_credentials` table
2. Alert initialization should happen in async `initialize()` method, not `__init__`

### "Non-hexadecimal digit found" (private key error)
1. Private key is returning encrypted - decryption failed
2. Check `.encryption_key` file exists and is readable
3. Verify `is_encrypted` flag in database matches actual state

### "can't compare offset-naive and offset-aware datetimes"
1. Database returning timezone-aware datetimes
2. Fixed with `_normalize_datetime()` in pool_engine.py

### Redis "Authentication required"
1. `REDIS_PASSWORD` needs to be loaded from secrets manager
2. Use `await secrets.get_async('REDIS_PASSWORD')` in async context

### Module Won't Start
1. Check `logs/<module>/<module>.log`
2. Verify credentials in `secure_credentials` table
3. Ensure `.encryption_key` file exists and matches DB encryption

---

## Adding New Credentials

1. **Via Dashboard (recommended):**
   - Go to `/settings/credentials`
   - Click "Add Credential"
   - Enter key name, value, category
   - Value is auto-encrypted on save

2. **Via SQL:**
```sql
-- First encrypt the value using Python Fernet
INSERT INTO secure_credentials (key, encrypted_value, is_encrypted, category, subcategory)
VALUES ('NEW_API_KEY', '<fernet-encrypted-value>', true, 'api', 'custom');
```

3. **Access in code:**
```python
from security.secrets_manager import secrets

# In async context (preferred)
api_key = await secrets.get_async('NEW_API_KEY')

# In sync context (limited - skips DB in async event loop)
api_key = secrets.get('NEW_API_KEY')
```

---

## Security Best Practices

1. **Never store credentials in .env** - Use `secure_credentials` table
2. **Always use `get_async()` in async context** - Sync `get()` skips database
3. **Pre-load credentials before sync constructors** - Load async, pass to `__init__`
4. **Keep `.encryption_key` secure** - Never commit to git, backup separately
5. **Rotate credentials periodically** - Update via Settings UI or SQL
6. **Monitor audit logs** - Access to sensitive credentials is logged

---

*Last Updated: December 2024*
