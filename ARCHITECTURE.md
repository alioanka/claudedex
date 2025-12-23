# ClaudeDex Trading Bot - Architecture Summary

## Overview

Multi-module crypto trading bot supporting DEX trading (EVM + Solana), Futures (Binance/Bybit), AI analysis, arbitrage, copy trading, and sniping. Built with Python/FastAPI, PostgreSQL (TimescaleDB), and Redis.

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

## Settings Architecture

```
┌─────────────────┐      ┌─────────────────┐      ┌────────────────────┐
│  Dashboard UI   │ ──►  │   PostgreSQL    │ ──►  │  Config Manager    │
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
- `/futures/settings` - Futures trading parameters
- `/solana/settings` - Solana module settings
- `/dex/settings` - DEX trading settings

**Config Managers (per module):**
| Module | Config Manager File |
|--------|---------------------|
| DEX | `config/config_manager.py` |
| Futures | `modules/futures_trading/config/futures_config_manager.py` |
| Solana | `modules/solana_trading/config/solana_config_manager.py` |
| AI Analysis | Settings loaded in `sentiment_engine.py` |

---

## Dashboard (Web UI)

**Location:** `dashboard/` (FastAPI + Jinja2 templates)

### Main Pages

| Route | Template | Purpose |
|-------|----------|---------|
| `/` | `index.html` | Main dashboard overview |
| `/modules` | `modules.html` | Module control panel (start/stop/restart) |
| `/settings` | `settings.html` | Global settings |

### Module-Specific Dashboards

Each module has its own dedicated dashboard with tabs for monitoring and configuration:

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

Manages the RPC Pool Engine with features:
- Add/edit/delete RPC endpoints per chain
- Health status indicators (healthy/rate-limited/unhealthy)
- Latency monitoring and auto-rotation
- Priority ordering for endpoints
- Rate limit detection and cooldown display

### Secure Credentials Page (`/settings/credentials`)

Manages encrypted credentials with features:
- Category-based organization (Wallet, Exchange, API, Notification, etc.)
- Visual status indicators (configured/not-configured/required)
- Secure input fields (password-masked)
- Auto-encryption on save
- Credential validation

### Dashboard Features

- **Real-time updates** via WebSocket
- **Dark/Light mode** toggle
- **Responsive design** for mobile
- **Module status indicators** (running/stopped/error)
- **Quick actions** (emergency stop, force refresh)
- **Pro Controls** for advanced operations

---

## Credentials & Security

### Two-Tier Security Model

**Tier 1: Docker Secrets** (`/run/secrets/`)
- Database password (`db_password`)
- Redis password (`redis_password`)
- Infrastructure credentials

**Tier 2: Encrypted Database** (`secure_credentials` table)
- Private keys (encrypted with Fernet)
- API keys (encrypted)
- Exchange secrets

### Secrets Manager

**File:** `security/secrets_manager.py`

**Priority Order:**
1. Docker secrets (`/run/secrets/<key>`)
2. Memory cache
3. Database (Fernet encrypted)
4. Environment variables (fallback)

**Usage:**
```python
from security.secrets_manager import secrets

# Sync
api_key = secrets.get('BINANCE_API_KEY')

# Async (with DB access)
api_key = await secrets.get_async('BINANCE_API_KEY')
```

### Encryption

**File:** `.encryption_key` (root directory)
- 32-byte Fernet key
- Used to encrypt/decrypt all credentials in database
- **NEVER commit to git**

**Decryption Pattern (in modules):**
```python
async def _get_decrypted_key(self, key_name: str) -> Optional[str]:
    # 1. Try secrets manager
    value = await secrets.get_async(key_name)

    # 2. Check if still encrypted (starts with gAAAAAB)
    if value and value.startswith('gAAAAAB'):
        # Decrypt with .encryption_key file
        f = Fernet(encryption_key)
        return f.decrypt(value.encode()).decode()

    return value
```

---

## RPC Pool Engine

**File:** `config/rpc_provider.py`

Provides RPC rotation, health checking, and fallback for all chains.

**Usage:**
```python
from config.rpc_provider import RPCProvider

# Async (preferred)
rpc_url = await RPCProvider.get_rpc('ETHEREUM_RPC')

# Sync (for initialization)
rpc_url = RPCProvider.get_rpc_sync('SOLANA_RPC')

# Report rate limits for rotation
await RPCProvider.report_rate_limit('ETHEREUM_RPC', rpc_url)
```

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
| RPC Pool | `rpc_api_pool`, `rpc_health_metrics` |

---

## Key Files by Module

### Orchestrator
- `main.py` - Module process manager, pre-flight validation

### DEX Trading
- `main_dex.py` - Entry point
- `trading/executors/base_executor.py` - Base trade executor
- `trading/chains/solana/jupiter_executor.py` - Jupiter swaps
- `config/config_manager.py` - DEX config

### Futures Trading
- `modules/futures_trading/main_futures.py` - Entry point
- `modules/futures_trading/core/futures_engine.py` - Trading engine
- `modules/futures_trading/exchanges/binance_futures.py` - Binance API
- `modules/futures_trading/config/futures_config_manager.py` - Config

### Solana Trading
- `modules/solana_trading/main_solana.py` - Entry point
- `modules/solana_trading/core/solana_engine.py` - Jupiter + trading logic
- `modules/solana_trading/config/solana_config_manager.py` - Config

### AI Analysis
- `modules/ai_analysis/main_ai.py` - Entry point
- `modules/ai_analysis/core/sentiment_engine.py` - AI trading logic

### Arbitrage
- `modules/arbitrage/main_arbitrage.py` - Entry point
- `modules/arbitrage/arbitrage_engine.py` - Cross-DEX arb
- `modules/arbitrage/solana_engine.py` - Solana arb (Jupiter)
- `modules/arbitrage/triangular_engine.py` - Triangular arb

### Copy Trading
- `modules/copy_trading/main_copy.py` - Entry point
- `modules/copy_trading/copy_engine.py` - Wallet tracking + execution

### Sniper
- `modules/sniper/main_sniper.py` - Entry point
- `modules/sniper/core/sniper_engine.py` - Token launch detection
- `modules/sniper/core/trade_executor.py` - Fast execution

### Security
- `security/secrets_manager.py` - Credential management
- `security/docker_secrets.py` - Docker secrets access
- `security/encryption.py` - Fernet utilities
- `security/audit_logger.py` - Access logging

---

## Logging

**Structure:**
```
logs/
├── orchestrator.log     # Main orchestrator
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

## Alerting System

**File:** `monitoring/alerts.py`

**Channels:**
- Telegram (`TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID`)
- Discord (`DISCORD_WEBHOOK_URL`)
- Email (SMTP)

**Alert Types:**
- Trading: Position opened/closed, SL/TP hit
- Risk: Drawdown, margin call, correlation warning
- System: Errors, API limits, low balance
- Security: Honeypot detected, rug pull warning

---

## Current .env Structure

**Only sensitive data in .env:**
```bash
# Security
ENCRYPTION_KEY=<fernet-key>  # Or use .encryption_key file
JWT_SECRET=<random>
SESSION_SECRET=<random>

# Database (Docker secrets preferred)
DATABASE_URL=postgresql://...
# Or use Docker secrets: /run/secrets/db_password

# Module Flags (process startup only)
DEX_MODULE_ENABLED=true
FUTURES_MODULE_ENABLED=false
SOLANA_MODULE_ENABLED=true

# Mode
DRY_RUN=true  # CRITICAL: false = live trading

# RPC URLs
ETHEREUM_RPC_URLS=https://...
SOLANA_RPC_URL=https://...

# Private Keys (encrypted, store in DB preferred)
PRIVATE_KEY=gAAAAAB...  # Fernet encrypted
SOLANA_MODULE_PRIVATE_KEY=gAAAAAB...

# API Keys (store in DB preferred)
BINANCE_API_KEY=...
OPENAI_API_KEY=...
```

**All trading parameters** (position size, leverage, pairs, strategies, risk settings) are stored in database via Settings UI.

---

## Quick Troubleshooting

### Module Won't Start
1. Check `logs/<module>/<module>.log`
2. Verify credentials in `secure_credentials` table
3. Ensure `.encryption_key` file exists and matches DB encryption

### "Invalid character" in Key
- Key returned from DB is still encrypted
- Check if Fernet decryption is working
- Verify `.encryption_key` matches what was used to encrypt

### "API key not found"
- Check secrets manager can access DB
- Verify key exists in `secure_credentials` table
- Ensure `is_encrypted` flag matches actual encryption state

### Jupiter "No forward quote"
- Rate limiting (check for 429 in logs)
- Invalid token pair / no liquidity
- API timeout (network issues)

---

## Adding New Credentials

1. **Store in DB (recommended):**
```sql
INSERT INTO secure_credentials (key, encrypted_value, is_encrypted, category)
VALUES ('NEW_API_KEY', '<fernet-encrypted-value>', true, 'api');
```

2. **Access in code:**
```python
from security.secrets_manager import secrets
api_key = await secrets.get_async('NEW_API_KEY')
```

3. **For module-specific loading, add to the module's `_get_decrypted_key()` method**

---

*Last Updated: December 2024*
