# ClaudeDex Trading Bot - Architecture & Context Guide

**Version**: 1.0.0
**Last Updated**: January 2026
**Purpose**: Context document for Claude AI sessions to understand the complete bot architecture

---

## 1. Project Overview

**ClaudeDex** is a multi-module automated cryptocurrency trading bot supporting:
- **EVM chains**: Ethereum, BSC, Polygon, Arbitrum, Base, Avalanche, Fantom, Cronos, PulseChain, Monad
- **Solana**: Jupiter DEX integration
- **Futures**: Binance, Bybit (centralized exchanges)

The bot uses **PostgreSQL with TimescaleDB** for time-series data, **Redis** for caching, and runs in **Docker containers**.

---

## 2. Module Architecture

### 2.1 Orchestrator Pattern

The bot uses a **multi-process orchestrator pattern** where `main.py` manages separate module processes:

```
main.py (Orchestrator)
├── modules/dashboard/main_dashboard.py   (ALWAYS starts first, port 8080)
├── modules/dex_trading/main_dex.py       (EVM DEX trading)
├── modules/futures_trading/main_futures.py
├── modules/solana_trading/main_solana.py
├── modules/arbitrage/main_arbitrage.py
├── modules/copy_trading/main_copy.py
├── modules/sniper/main_sniper.py
└── modules/ai_analysis/main_ai.py
```

**Key Files:**
- `/home/user/claudedex/main.py` - Main orchestrator (lines 460-723)
- Environment variables control module enabling: `DEX_MODULE_ENABLED`, `FUTURES_MODULE_ENABLED`, etc.
- Dashboard is independent - starts first and stays running even if trading modules fail

### 2.2 Module Enable/Disable Flags

Set in `.env` or Docker environment:
```bash
DASHBOARD_MODULE_ENABLED=true      # Dashboard (default: true)
DEX_MODULE_ENABLED=true            # EVM DEX trading
FUTURES_MODULE_ENABLED=false       # Binance/Bybit futures
SOLANA_MODULE_ENABLED=false        # Solana trading
ARBITRAGE_MODULE_ENABLED=false     # Cross-DEX arbitrage
COPY_TRADING_MODULE_ENABLED=false  # Copy trading
SNIPER_MODULE_ENABLED=false        # Token sniping
AI_MODULE_ENABLED=false            # AI analysis
```

---

## 3. RPC/API Pool Engine

### 3.1 Overview

The **Pool Engine** provides intelligent RPC endpoint management with:
- Database-driven endpoint storage (`rpc_api_pool` table)
- Automatic rate limit detection and rotation
- Health checks and priority scoring
- Fallback to static URLs in `utils/constants.py`

**Key Files:**
- `/home/user/claudedex/config/pool_engine.py` - Core Pool Engine (1320 lines)
- `/home/user/claudedex/config/rpc_provider.py` - Convenience wrapper
- `/home/user/claudedex/utils/constants.py` - Static fallback RPCs

### 3.2 Pool Engine Database Tables

```sql
-- Main RPC/API pool table
rpc_api_pool (
    id, endpoint_type, provider_type, name, url, api_key,
    status, is_enabled, priority, weight, rate_limit_until,
    health_score, consecutive_failures, chain, supports_ws, ws_url,
    success_count, failure_count, avg_latency_ms, last_success_at, last_failure_at
)

-- Usage history for analytics
rpc_api_usage_history (
    id, endpoint_id, module_name, success, latency_ms, error_type, error_message, created_at
)

-- Provider type definitions
rpc_api_provider_types (
    provider_type, endpoint_type, chain, description, default_priority, is_required
)
```

### 3.3 Using the Pool Engine

```python
# Async usage (recommended)
from config.rpc_provider import RPCProvider

rpc_url = await RPCProvider.get_rpc('ETHEREUM_RPC')
await RPCProvider.report_success('ETHEREUM_RPC', rpc_url, latency_ms=100)
await RPCProvider.report_rate_limit('ETHEREUM_RPC', rpc_url, duration_seconds=300)

# Sync usage (for initialization)
rpc_url = RPCProvider.get_rpc_sync('ETHEREUM_RPC')
```

### 3.4 Provider Types

```
ETHEREUM_RPC, BSC_RPC, POLYGON_RPC, ARBITRUM_RPC, BASE_RPC,
AVALANCHE_RPC, FANTOM_RPC, CRONOS_RPC, PULSECHAIN_RPC, MONAD_RPC,
SOLANA_RPC, SOLANA_WS, HELIUS_API, JUPITER_API, GOPLUS_API, ETHERSCAN_API
```

### 3.5 Static Fallback RPCs

If Pool Engine fails, uses free public RPCs from `utils/constants.py`:
```python
Chain.ETHEREUM: ["https://eth.llamarpc.com", "https://ethereum.publicnode.com", ...]
Chain.BSC: ["https://bsc-dataseed1.binance.org", ...]
# etc.
```

---

## 4. Security & Secrets Management

### 4.1 Secrets Manager Architecture

**Priority order for credential lookup:**
1. Docker secrets (`/run/secrets/<key>`)
2. Database encrypted storage (`secure_credentials` table)
3. Environment variables (`.env` fallback)

**Key File:** `/home/user/claudedex/security/secrets_manager.py`

### 4.2 Encryption

- Encryption key stored in `.encryption_key` file (NOT in git)
- Uses **Fernet** symmetric encryption
- All sensitive values in database are encrypted at rest
- Encrypted values start with `gAAAAAB` (Fernet signature)

```python
from security.secrets_manager import secrets

# Get a credential (auto-decrypts if needed)
api_key = secrets.get('BINANCE_API_KEY')
api_key = await secrets.get_async('BINANCE_API_KEY')

# Store a credential (encrypts automatically)
await secrets.set('MY_KEY', 'value', category='api', is_sensitive=True)
```

### 4.3 Docker Secrets Setup

```bash
mkdir -p ./secrets && chmod 700 ./secrets
echo "bot_user" > ./secrets/db_user
echo "strong_password" > ./secrets/db_password
echo "redis_password" > ./secrets/redis_password
chmod 600 ./secrets/*
```

### 4.4 Database Credential Tables

```sql
-- Encrypted credentials storage
secure_credentials (
    id, key_name, display_name, description, encrypted_value, value_hash,
    category, subcategory, module, is_sensitive, is_encrypted, is_required,
    is_active, last_accessed_at, access_count, last_rotated_at
)

-- Access audit log
credential_access_log (
    id, credential_id, key_name, access_type, accessed_by, ip_address,
    success, error_message, created_at
)
```

---

## 5. Configuration Management

### 5.1 ConfigManager

Dynamic configuration from database with hot-reload support.

**Key File:** `/home/user/claudedex/config/config_manager.py` (1539 lines)

### 5.2 Configuration Types

```python
ConfigType.GENERAL          # Mode, dry_run
ConfigType.PORTFOLIO        # Balance, position sizes
ConfigType.RISK_MANAGEMENT  # Stop loss, take profit, daily limits
ConfigType.TRADING          # Slippage, fees
ConfigType.STRATEGIES       # Momentum, scalping, AI settings
ConfigType.CHAIN            # Enabled chains, liquidity thresholds
ConfigType.POSITION_MANAGEMENT  # Trailing stops
ConfigType.SOLANA           # Solana-specific settings
ConfigType.JUPITER          # Jupiter DEX settings
```

### 5.3 Configuration Priority

1. Database (`config_settings` table) - Highest priority
2. Environment variables (`.env`)
3. YAML config files (`./config/*.yaml`)
4. Default values in Pydantic schemas

### 5.4 Using ConfigManager

```python
from config.config_manager import ConfigManager

config_manager = ConfigManager()
await config_manager.initialize()
config_manager.set_db_pool(db_pool)

# Get typed config
trading_config = config_manager.get_trading_config()
chain_config = config_manager.get_chain_config()

# Get by key
value = config_manager.get('PRIVATE_KEY')

# Update config (persists to database)
await config_manager.update_config(
    ConfigType.TRADING,
    {'max_slippage_bps': 100},
    user='admin',
    reason='Increased slippage tolerance'
)
```

---

## 6. Database Architecture

### 6.1 Technology Stack

- **PostgreSQL 14** with **TimescaleDB** extension
- **asyncpg** for async database access
- Connection pooling with configurable min/max connections

**Key File:** `/home/user/claudedex/data/storage/database.py` (1022 lines)

### 6.2 Core Tables

```sql
-- Trading
trades (id, trade_id, token_address, chain, side, entry_price, exit_price,
        amount, usd_value, profit_loss, strategy, status, metadata)
positions (id, position_id, token_address, chain, entry_price, stop_loss,
           take_profit, amount, status, metadata)

-- Market Data (TimescaleDB hypertable)
market_data (time, token_address, chain, price, volume_24h, liquidity_usd,
             price_change_5m, price_change_1h, price_change_24h)

-- Configuration
config_settings (id, config_type, key, value, value_type, is_editable)
config_history (id, config_type, key, old_value, new_value, changed_by, reason)

-- Security
secure_credentials, credential_access_log, credential_categories

-- Pool Engine
rpc_api_pool, rpc_api_usage_history, rpc_api_provider_types
```

### 6.3 Database Connection

```python
from data.storage.database import DatabaseManager

db_config = {
    'DB_HOST': 'postgres',
    'DB_PORT': 5432,
    'DB_NAME': 'tradingbot',
    'DB_USER': 'bot_user',
    'DB_PASSWORD': 'password',
    'DB_POOL_MIN': 10,
    'DB_POOL_MAX': 20,
}

db_manager = DatabaseManager(db_config)
await db_manager.connect()

# Use transactions
async with db_manager.transaction() as conn:
    await conn.execute("INSERT INTO trades ...")
    await conn.execute("UPDATE portfolio ...")
```

---

## 7. Dashboard

### 7.1 Standalone Dashboard Architecture

The dashboard runs **independently** of trading modules to ensure access even when modules fail.

**Key Files:**
- `/home/user/claudedex/modules/dashboard/main_dashboard.py` - Standalone entry point
- `/home/user/claudedex/monitoring/enhanced_dashboard.py` - Dashboard endpoints & UI

### 7.2 Dashboard Features

- **Real-time monitoring** via WebSocket/SSE
- **Trading history** and analytics
- **Configuration management** (all ConfigTypes editable)
- **Pool Engine management** (add/edit/delete RPC endpoints)
- **Credentials management** (via secrets manager)
- **Module status** (online/offline indicators)

### 7.3 Dashboard Port Conflict Resolution

The orchestrator explicitly sets `DASHBOARD_MODULE_ENABLED=true` in the environment before starting trading modules. This prevents the DEX module from starting its own dashboard on port 8080.

```python
# In main.py orchestrator
if await dashboard.start():
    os.environ['DASHBOARD_MODULE_ENABLED'] = 'true'  # Critical!
```

---

## 8. Log Files

### 8.1 Log Directory Structure

```
logs/
├── orchestrator.log           # Main orchestrator logs
├── dashboard/
│   └── dashboard.log          # Standalone dashboard logs
├── dex_trading/
│   ├── stdout.log             # DEX module stdout (rotating, 10MB max)
│   └── stderr.log             # DEX module stderr
├── futures_trading/
├── solana_trading/
├── pool_engine/
│   ├── pool_engine.log        # Main Pool Engine activity
│   ├── pool_engine_errors.log # Errors only
│   ├── pool_engine_full.log   # Comprehensive debug logs
│   ├── pool_engine_rate_limits.log  # Rate limit events
│   └── pool_engine_health.log # Health check results
└── [other modules]/
```

### 8.2 Log Rotation

Logs use `RotatingFileHandler` with:
- Max 10MB per file
- 3-5 backup files
- Automatic rotation during health checks

---

## 9. Docker Infrastructure

### 9.1 Services

```yaml
services:
  postgres:       # TimescaleDB (port 5432)
  redis:          # Redis cache (port 6379)
  trading-bot:    # Main application (port 8080)
```

### 9.2 Key Docker Commands

```bash
# Start/rebuild (KEEPS data)
docker-compose up -d --build

# View logs
docker-compose logs -f trading-bot

# Stop (KEEPS data)
docker-compose down

# ⚠️ WIPE ALL DATA (use with caution!)
docker-compose down -v
```

### 9.3 Volumes

- `postgres-data`: Database persistence
- `redis-data`: Redis persistence
- `./logs:/app/logs`: Log files
- `./config:/app/config`: Configuration files
- `./.encryption_key:/app/.encryption_key:ro`: Encryption key

---

## 10. Key Constants

### 10.1 Trading Parameters (from `config/settings.py`)

```python
TRADING_MAX_SLIPPAGE_BPS = 50       # 0.5%
TRADING_DEX_FEE_BPS = 30            # 0.3%
MAX_GAS_PRICE_GWEI = 50             # Gas price limit
DEFAULT_STOP_LOSS = 0.05            # 5%
DEFAULT_TAKE_PROFIT = 0.10          # 10%
```

### 10.2 Supported Chains (from `utils/constants.py`)

```python
class Chain(IntEnum):
    ETHEREUM = 1
    BSC = 56
    POLYGON = 137
    ARBITRUM = 42161
    BASE = 8453
    AVALANCHE = 43114
    FANTOM = 250
    CRONOS = 25
    PULSECHAIN = 369
    MONAD = 41454
```

---

## 11. Common Issues & Solutions

### 11.1 Port 8080 Already in Use

**Cause**: Both standalone dashboard and DEX module trying to bind to port 8080.

**Solution**: Ensure `DASHBOARD_MODULE_ENABLED=true` is set in environment before starting DEX module. The orchestrator does this automatically as of commit `67fec4f`.

### 11.2 RPC Connection Failures

**Cause**: Expired API keys or rate-limited endpoints.

**Solution**:
1. Check Pool Engine table: `SELECT * FROM rpc_api_pool WHERE status != 'active';`
2. Update RPCs via dashboard or SQL script
3. Fallback to static RPCs in `utils/constants.py`

### 11.3 Encrypted Values Not Decrypting

**Cause**: Missing or mismatched encryption key.

**Solution**:
1. Ensure `.encryption_key` file exists and has correct permissions
2. Key must match what was used to encrypt values
3. Check `secrets.initialize(db_pool)` is called with database pool

### 11.4 Dashboard Can't Start Without DEX Module

**FIXED**: Dashboard is now decoupled and starts independently. See `modules/dashboard/main_dashboard.py`.

---

## 12. Development Workflow

### 12.1 Adding a New RPC Endpoint

```sql
INSERT INTO rpc_api_pool (
    endpoint_type, provider_type, name, url, chain, status, is_enabled, priority
) VALUES (
    'rpc', 'ETHEREUM_RPC', 'My RPC', 'https://example.com/rpc', 'ethereum', 'active', true, 100
);
```

### 12.2 Adding a New Configuration Setting

1. Add to Pydantic model in `config/config_manager.py`
2. Add to database `config_settings` table with:
   - `config_type`, `key`, `value`, `value_type`, `is_editable`
3. Dashboard will auto-detect and display it

### 12.3 Adding a New Trading Module

1. Create `modules/[name]/main_[name].py`
2. Add to `TradingBotOrchestrator` in `main.py`
3. Add environment variable `[NAME]_MODULE_ENABLED`
4. Add to `ModuleProcess.REQUIRED_SECRETS` if needed

---

## 13. Important Code Patterns

### 13.1 Getting RPC with Fallback

```python
# Preferred pattern
from config.rpc_provider import RPCProvider

rpc_url = await RPCProvider.get_rpc('ETHEREUM_RPC')
if not rpc_url:
    # Pool Engine unavailable, use static fallback
    from utils.constants import get_chain_rpc_url, Chain
    rpc_url = get_chain_rpc_url(Chain.ETHEREUM)
```

### 13.2 Getting Secrets

```python
from security.secrets_manager import secrets

# Sync (for initialization)
api_key = secrets.get('BINANCE_API_KEY')

# Async (preferred in async context)
api_key = await secrets.get_async('BINANCE_API_KEY')
```

### 13.3 Transaction-Safe Database Operations

```python
async with db_manager.transaction() as conn:
    await conn.execute("INSERT INTO trades ...")
    await conn.execute("UPDATE positions ...")
    # Auto-commit on success, auto-rollback on exception
```

---

## 14. Current Known Issues

*Add any current issues below this line:*

---

## Document History

- **2026-01-26**: Initial comprehensive documentation created
- **2026-01-26**: Dashboard decoupled from DEX module
- **2026-01-26**: Port 8080 conflict fix committed
