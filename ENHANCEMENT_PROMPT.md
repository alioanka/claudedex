# ClaudeDex Trading Bot - Enhancement Session Prompt

Use this prompt in a new Claude Code session to implement dramatic enhancements to the bot.

---

## PROMPT START (copy everything below this line)

---

I have a production cryptocurrency trading bot called **ClaudeDex** that I want to dramatically enhance with new profitable strategies and modules. Below is a complete analysis of the existing infrastructure, architecture requirements, and specific strategies to implement.

## Current Codebase Architecture

### Module Structure
The bot is modular with each module running as a **separate subprocess** managed by an orchestrator (`main.py`). Each module has:
- Its own **engine file(s)** in `modules/{module_name}/`
- Its own **main entry point** (e.g., `modules/solana_trading/main_solana.py`)
- **Separate log files**: `logs/{module_name}/{module}_trading.log`, `{module}_trades.log`, `{module}_errors.log` using `RotatingFileHandler` (10MB main, 5MB error, 10MB trades, with backup rotation)
- Trade logs use a **separate logger** with `propagate = False` for independence

### Module Registration Pattern (main.py orchestrator)
```python
# In TradingBotOrchestrator.__init__()
self.modules['module_name'] = ModuleProcess(
    name="ModuleName",
    script_path="modules/module_name/main_module.py",
    enabled_env_var="MODULE_NAME_MODULE_ENABLED",
    module_key="module_name"
)
```
Modules are enabled/disabled via `.env` flags. The orchestrator validates secrets before starting each module.

### Dashboard Architecture
Each module has its own dedicated dashboard pages:
- **Dashboard page**: `dashboard_{module}.html` - Real-time status and overview
- **Performance page**: `performance_{module}.html` - Analytics, win rate, P&L charts
- **Trade history page**: `trades_{module}.html` - Historical trade logs
- **Positions page**: `positions_{module}.html` - Active positions
- **Settings page**: `settings_{module}.html` - Full UI-based configuration

The dashboard runs as an **independent module** (`modules/dashboard/main_dashboard.py`) on port 8080. It connects directly to the database and does NOT depend on trading modules being running.

**Settings pages have TWO tabs:**
1. **Settings tab** - UI controls to modify all module configurations
2. **Guide tab** - Complete module documentation with all settings explained, default/recommended values, and detailed descriptions

**ALL settings are saved to the database** via the module's config manager. NO hardcoded settings. NO .env config (except module enable/disable flags and infrastructure URLs).

### Configuration Management
**File:** `config/config_manager.py` - `ConfigManager` class

```python
# Loads from DB (config_settings table) with priority:
# Database > Config Files > Environment Variables > Defaults
rows = await conn.fetch("""
    SELECT key, value, value_type
    FROM config_settings
    WHERE config_type = $1 AND is_editable = TRUE
""", config_type.value)
```

- Uses `ConfigType` enum: GENERAL, TRADING, STRATEGIES, SECURITY, DATABASE, API, MONITORING, ML_MODELS, RISK_MANAGEMENT, PORTFOLIO, CHAIN, DASHBOARD
- Sensitive values in `config_sensitive` table (encrypted)
- Change history tracked in `config_history` table
- Attribute-style access: `config.get('key')` or `config['key']`

### RPC Pool Engine
**File:** `config/pool_engine.py` - `PoolEngine` class (singleton)

```python
# Get best available RPC for a chain/provider type
endpoint = await pool_engine.get_endpoint(provider_type)
# Report health
pool_engine.report_success(endpoint)
pool_engine.report_failure(endpoint)
pool_engine.report_rate_limit(endpoint)
```

- Database table: `rpc_api_pool` with health status, priority, rate limit tracking
- Hourly health checks
- Usage history in `rpc_api_usage_history`
- Falls back to `.env` RPC URLs if DB not available
- **ALL modules MUST use PoolEngine** to get RPC endpoints - never hardcode or directly read from .env

### Security & Secrets
**File:** `security/secrets_manager.py` - `SecureSecretsManager` class (singleton)

```python
# Priority lookup: Docker secrets > Database > Environment
value = secrets_manager.get('PRIVATE_KEY')
```

- All wallet private keys, API keys, and sensitive data stored in `secure_credentials` table
- Encrypted at rest using **Fernet symmetric encryption**
- Wallet addresses are **derived from private keys** at runtime - never stored separately
- Access logged in `credential_access_log` table

### Database Schema (PostgreSQL + Redis cache)
**Core tables:** `trades`, `positions`, `market_data`, `alerts`, `token_analysis`, `performance_metrics`, `whale_wallets`, `mev_transactions`, `sentiment_logs`, `system_logs`
**Config tables:** `config_settings`, `config_sensitive`, `config_history`
**Security tables:** `secure_credentials`, `credential_access_log`
**RPC tables:** `rpc_api_pool`, `rpc_api_provider_types`, `rpc_api_usage_history`

All tables use:
- UUID-based IDs where applicable
- JSONB columns for flexible metadata
- Timestamp tracking (created_at, updated_at)
- Composite indexes for performance

### Current .env File Structure
```env
# MODULE ENABLE/DISABLE FLAGS (the ONLY config in .env besides infrastructure URLs)
DEX_MODULE_ENABLED=false
FUTURES_MODULE_ENABLED=false
SOLANA_MODULE_ENABLED=true
SNIPER_MODULE_ENABLED=false
AI_MODULE_ENABLED=false
ARBITRAGE_MODULE_ENABLED=true
COPY_TRADING_MODULE_ENABLED=false
DASHBOARD_MODULE_ENABLED=true

MODE=production
DRY_RUN=false
FUTURES_TESTNET=false

FLASHBOTS_RPC=https://relay.flashbots.net

# Flash loan contracts (deployed on-chain)
FLASH_LOAN_RECEIVER_CONTRACT=0xac16b6bfdc15d7e377b12165851bcfe553abdf62
FLASH_LOAN_RECEIVER_CONTRACT_ARB=0x6D143660b3aFfde7d8611F87d26adB36791e5003
FLASH_LOAN_RECEIVER_CONTRACT_BASE=0x8EAcF13DBf8Bd5832E3558997BAD16bBcCAffD21

# Multi-Chain RPC URLs
ETHEREUM_RPC_URLS="..."
BSC_RPC_URLS="..."
POLYGON_RPC_URLS="..."
ARBITRUM_RPC_URLS="..."
BASE_RPC_URLS="..."
MONAD_RPC_URLS="..."
PULSECHAIN_RPC_URLS="..."
FANTOM_RPC_URLS="..."
CRONOS_RPC_URLS="..."
AVALANCHE_RPC_URLS="..."

# Solana
SOLANA_RPC_URL=https://mainnet.helius-rpc.com/?api-key=...
SOLANA_RPC_URLS="..."
SOLANA_WS_URL=wss://mainnet.helius-rpc.com/?api-key=...

# Jupiter & Jito
JUPITER_API_URL=https://lite-api.jup.ag
JITO_TIP_ACCOUNT=96gYZGLnJYVFmbjzopPSU6QiEV5fGqZNyN9nmNhvrZU5
JITO_BLOCK_ENGINE_URL=https://mainnet.block-engine.jito.wtf
```

**Architecture diagram:**
```
.env (infrastructure only) --> Config Manager (loads from DB) --> Trading Engine (uses config)
                                      ^
                                      |
                               Settings UI (saves to DB)
```

---

## Existing Infrastructure (Already Built & Available)

### EVM Chains Supported (7+ mainnet)
| Chain | Chain ID | DEXs Available | Flash Loans |
|-------|----------|---------------|-------------|
| Ethereum | 1 | Uniswap V2/V3, SushiSwap | Aave V3 |
| BSC | 56 | PancakeSwap | - |
| Polygon | 137 | Uniswap V3, SushiSwap | Aave V3 |
| Arbitrum | 42161 | Uniswap V3, SushiSwap | Aave V3 |
| Base | 8453 | Uniswap V3 | Aave V3 |
| Avalanche | 43114 | Multi-DEX | - |
| Fantom | 250 | DEX integration | - |
| Monad | New L1 | Planned | - |
| PulseChain | EVM | Planned | - |
| Cronos | EVM | Planned | - |

### DEX Integrations
**EVM:**
- Uniswap V2 Router: `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D`
- Uniswap V3 SwapRouter: `0xE592427A0AEce92De3Edee1F18E0157C05861564` (uses Quoter for quotes, NOT getAmountsOut)
- SushiSwap Router: `0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F`
- PancakeSwap Router: `0x10ED43C718714eb63d5aA57B78B54704E256024E`

**Solana:**
- Jupiter V6 Aggregator (`modules/solana_strategies/jupiter_helper.py`) - Full quote/swap, rate limiting (0.8 RPS)
- Raydium AMM - Pool monitoring, graduation detection
- Orca Whirlpools - Referenced in monitoring
- Meteora DLMM - Referenced

### CEX Integrations
- **Binance Futures** (`modules/futures_trading/exchanges/binance_futures.py`) - FULLY IMPLEMENTED: USDT-M/COIN-M perpetuals, up to 125x leverage, isolated/cross margin, funding rate tracking
- **Bybit Futures** (`modules/futures_trading/exchanges/bybit_futures.py`) - PLACEHOLDER only, skeleton code

### Flash Loan Infrastructure
- **Aave V3 Flash Loans** - Deployed contracts on Ethereum, Arbitrum, Base
- Smart contracts: `contracts/FlashLoanArbitrage.sol`, `FlashLoanArbitrage_Arbitrum.sol`, `FlashLoanArbitrage_Base.sol`
- Pattern: Bot calls contract -> Aave flashLoanSimple -> Execute swaps -> Repay + 0.05% fee -> Keep profit

### MEV / Mempool Infrastructure
- **Mempool Monitor** (`data/collectors/mempool_monitor.py`) - Real-time pending tx monitoring, swap method detection (6 methods), sandwich attack detection, MEV opportunity scoring
- **MEV Protection** (`trading/executors/mev_protection.py`) - Flashbots bundle submission, 5 protection levels (NONE to MAXIMUM), gas randomization, decoy transactions, commit-reveal schemes
- **Private pool providers:** BloXroute, Blocknative, Ethermine

### Solana-Specific Infrastructure
- **Jupiter Helper** - Full aggregator with multiple API tiers (lite, public V6, ultra premium)
- **Drift Protocol** (`modules/solana_strategies/drift_helper.py`) - FULLY IMPLEMENTED: perpetual futures (SOL-PERP, BTC-PERP, ETH-PERP), funding rate monitoring, liquidation risk assessment, max 5x leverage
- **Pump.fun Launch Trading** (`trading/strategies/pumpfun_launch.py`) - Launch monitoring, bonding curve analysis, graduation detection (PRE_LAUNCH, EARLY, MID, LATE, GRADUATED phases), dev behavior analysis, sniper detection
- **Jito Bundles** - Infrastructure ready (tip account, block engine URL configured), Solana MEV protection

### ML/AI Stack
- **Ensemble Model** (`ml/models/ensemble_model.py`) - 9 models: LSTM (bidirectional, attention), Transformer (4 layers, 8 heads), XGBoost, LightGBM, Random Forest, Gradient Boosting, Isolation Forest
- **Rug Classifier** (`ml/models/rug_classifier.py`) - Rug probability scoring
- **Volume Validator** (`ml/models/volume_validator.py`) - Wash trading detection
- **AI Trading Engine** (`modules/ai_analysis/core/ai_trading_engine.py`) - LLM integration (OpenAI GPT + Claude API)
- **Sentiment Engine** (`modules/ai_analysis/core/sentiment_engine.py`) - Social media sentiment analysis

### Copy Trading & Whale Tracking
- **Copy Engine** (`modules/copy_trading/copy_engine.py`) - Real-time wallet monitoring (7 EVM chains + Solana), automatic trade replication, Jupiter for Solana execution
- **Whale Tracker** (`data/collectors/whale_tracker.py`) - Multi-chain detection, configurable thresholds ($250K-$1M), balance tracking, ERC-20 Transfer event monitoring

### Data Collection & Price Feeds
- **DexScreener** (`data/collectors/dexscreener.py`) - 11+ chains, real-time price/volume/liquidity, buy/sell ratios
- **CoinGecko** - Price feeds with 1-minute caching, used across multiple modules
- **Social Data** (`data/collectors/social_data.py`) - Twitter/X, Discord, Telegram, Reddit

### Monitoring & Observability
- Prometheus metrics (`observability/prometheus.yml`)
- Grafana dashboards (`observability/grafana/`)
- Alert rules (`observability/alerts.yml`)
- Telegram bot notifications (`monitoring/telegram_bot.py`)

---

## Currently Active Modules

| Module | Status | What It Does |
|--------|--------|-------------|
| **Solana Trading** | ACTIVE | Pump.fun token sniping with trailing stops, Jupiter swaps |
| **Arbitrage** | ACTIVE | Multi-chain EVM arbitrage (ETH, Base, Arbitrum) + Triangular arb with flash loans |
| **Dashboard** | ACTIVE | Web UI for all modules |

---

## STRATEGIES TO IMPLEMENT

### TIER 1: High ROI, Low Effort (Infrastructure Already Exists)

#### Strategy 1: CEX-DEX Arbitrage Engine
**Concept:** Compare Binance spot/futures prices against DEX AMM prices in real-time. CEX prices update faster than DEX pools - when Binance moves, DEX pools lag for seconds to minutes, especially on cheap L2s (Arbitrum, Base).

**What exists:** Binance API (full), DEX routers on 7+ chains, Aave flash loans, WebSocket feeds.

**What to build:**
- New module: `modules/cex_dex_arb/` with `cex_dex_engine.py`, `main_cex_dex.py`
- Real-time price comparison loop: Binance WebSocket stream vs on-chain DEX pool prices
- Execution path: Flash loan on Aave -> Buy cheap on DEX -> Sell on Binance (or vice versa)
- Focus chains: Arbitrum and Base (low gas, fast finality)
- Key pairs: ETH/USDC, WBTC/USDC, ARB/USDC, high-volume L2 tokens
- Must account for: bridge delays (if cross-chain), gas costs, slippage, Binance withdrawal fees
- Alternative: same-chain only (DEX price vs CEX price, execute on DEX only using flash loans to capture the reversion)

**Settings needed:**
- min_spread_threshold (default: 0.3%)
- max_position_size_usd
- chains_enabled (list of chains to monitor)
- pairs_to_monitor (list of token pairs)
- execution_mode (flash_loan / direct)
- price_staleness_threshold_ms (max age of price data)
- binance_api_key / secret (via secrets manager)

#### Strategy 2: Solana Cross-DEX Arbitrage Engine
**Concept:** Direct pool-to-pool arbitrage on Solana between Raydium, Orca, Meteora, and Jupiter routes. Near-zero gas (~$0.001/tx), sub-second finality. Jupiter aggregator doesn't always find the optimal route.

**What exists:** Jupiter helper, Raydium/Orca/Meteora integrations, Jito bundles for MEV protection.

**What to build:**
- New engine within Solana module or separate: `modules/solana_arb/solana_arb_engine.py`
- Direct pool state monitoring (not via Jupiter aggregator) for popular pairs
- Compare: Raydium CLMM pool price vs Orca Whirlpool price vs Meteora DLMM price
- Execute: Buy on cheapest DEX, sell on most expensive, using Jito bundles for atomicity
- Key pairs: SOL/USDC, SOL/USDT, mSOL/SOL, JitoSOL/SOL, popular memecoins with multi-DEX liquidity
- Volume-weighted pricing to avoid thin liquidity traps

**Settings needed:**
- min_spread_bps (default: 15 bps = 0.15%)
- max_trade_size_sol
- pairs_to_monitor
- dexes_enabled (raydium, orca, meteora)
- use_jito_bundles (default: true)
- jito_tip_lamports (default: 10000)
- scan_interval_ms (default: 500)

#### Strategy 3: Drift Funding Rate Arbitrage
**Concept:** When Drift Protocol funding rate is deeply negative, go long perp + short spot (delta-neutral). When positive, go short perp + long spot. Collect funding payments as pure yield.

**What exists:** `drift_helper.py` with full open/close/funding rate monitoring, Jupiter for spot hedging.

**What to build:**
- New strategy: `modules/solana_strategies/drift_funding_arb.py`
- Continuous funding rate monitoring on Drift (SOL-PERP, BTC-PERP, ETH-PERP)
- When |funding rate| > threshold: Open delta-neutral position (perp + opposite spot)
- Auto-rebalance when delta drifts beyond tolerance
- Close when funding rate normalizes
- Calculate: expected funding income vs position costs (spread, fees)

**Settings needed:**
- min_funding_rate_threshold (default: 0.01% per hour = ~87% APR)
- max_position_size_usd
- markets_enabled (SOL-PERP, BTC-PERP, ETH-PERP)
- delta_tolerance_pct (default: 2% - rebalance if delta drifts)
- max_leverage (default: 3x)
- auto_close_funding_threshold (default: 0.005% - close when funding normalizes)

#### Strategy 4: Enhanced Pump.fun Graduation Sniping
**Concept:** When a pump.fun token completes its bonding curve and migrates to Raydium, be the first buyer on Raydium. Tokens often see 2-10x spike as real DEX liquidity becomes available and aggregators discover them.

**What exists:** `pumpfun_launch.py` with graduation detection, Jupiter helper, Jito bundles.

**What to build:**
- Enhanced graduation detection with faster reaction time
- Pre-compute Raydium pool address from bonding curve parameters
- Use Jito bundle to land transaction in the same block as migration
- Graduated exit tiers: 25% at 2x, 25% at 3x, 25% at 5x, hold 25% moonbag
- Filter: Only snipe tokens with strong pre-graduation metrics (high holder count, good buy/sell ratio, dev sold <5%)

**Settings needed:**
- graduation_snipe_enabled (default: true)
- snipe_amount_sol (default: 0.1 SOL)
- min_holders_before_grad (default: 100)
- min_buy_sell_ratio (default: 1.5)
- max_dev_holding_pct (default: 5%)
- exit_tiers (JSON: [{pct: 25, target: 2x}, ...])
- use_jito_bundles (default: true)
- jito_tip_lamports (default: 50000 for priority)

---

### TIER 2: Medium Effort, Strong Profit Potential

#### Strategy 5: Aave Liquidation Hunter
**Concept:** Monitor unhealthy positions on Aave V3 across Ethereum, Arbitrum, Base. When health factor drops below 1.0, flash loan the repayment amount, liquidate the position, receive collateral + 5-15% liquidation bonus, repay flash loan, keep profit.

**What exists:** Flash loan contracts on 3 chains, multi-chain RPC, Web3 infrastructure.

**What to build:**
- New module: `modules/liquidation/` with `liquidation_engine.py`, `main_liquidation.py`
- Monitor Aave V3 lending pools for positions with health factor approaching 1.0
- Pre-calculate profitability: liquidation bonus - gas cost - flash loan fee (0.05%)
- Execute atomically via flash loan contract (may need new contract or extend existing)
- Multi-chain: ETH (highest value), Arbitrum (lower gas), Base (growing TVL)

**Settings needed:**
- chains_enabled (ethereum, arbitrum, base)
- health_factor_alert_threshold (default: 1.1 - start monitoring)
- health_factor_execute_threshold (default: 1.0 - execute liquidation)
- min_profit_usd (default: $50)
- max_gas_gwei (default: 50)
- aave_pool_addresses per chain
- collateral_tokens_whitelist

#### Strategy 6: Cross-Chain Arbitrage
**Concept:** Same token on different chains often has price discrepancies (e.g., WETH on Ethereum vs Arbitrum, USDC across chains). Buy cheap on one chain, bridge, sell expensive on another.

**What exists:** Multi-chain RPC configs, DEX execution on all chains.

**What to build:**
- New module: `modules/cross_chain_arb/` with `cross_chain_engine.py`
- Monitor prices of major tokens across all supported chains simultaneously
- Integrate fast bridge protocols (Stargate, Across, LayerZero) for quick settlement
- Account for: bridge fees, gas on both chains, bridge time, slippage
- Focus on stablecoins first (USDC, USDT - lower risk), then volatile assets

**Settings needed:**
- chains_enabled (list)
- tokens_to_monitor (WETH, USDC, USDT, WBTC, etc.)
- min_spread_after_fees_pct (default: 0.5%)
- bridge_provider (stargate, across, layerzero)
- max_bridge_time_minutes (default: 10)
- max_position_size_usd

#### Strategy 7: MEV Backrunning Engine
**Concept:** When a whale makes a large swap on a DEX (detected via mempool), the price moves. Immediately backrun the transaction to arbitrage the price impact against other pools or CEX prices. This is legal and non-harmful (unlike sandwiching).

**What exists:** `mempool_monitor.py` with swap detection, Flashbots bundle submission, DEX execution.

**What to build:**
- New strategy layer on top of mempool monitor
- Detect large pending swaps (>$10K value)
- Calculate expected price impact on the pool
- Find counter-arbitrage opportunity on another DEX/pool
- Bundle backrun transaction via Flashbots (same block, after the whale tx)

**Settings needed:**
- min_swap_size_usd (default: $10,000)
- target_dexes (uniswap_v2, uniswap_v3, sushiswap)
- max_gas_for_backrun_gwei
- min_profit_after_gas_usd (default: $20)
- flashbots_enabled (default: true)

---

### TIER 3: Higher Effort, New Capabilities

#### Strategy 8: Perpetual DEX Funding Rate Arb (EVM)
**Concept:** dYdX, GMX, Hyperliquid all have perpetual markets on EVM. Funding rates diverge across these venues. Open opposing positions on two perp DEXs to collect the funding rate differential.

**What to build:**
- Integrate GMX V2 (Arbitrum), Hyperliquid, dYdX V4
- Monitor funding rates across all venues
- Open delta-neutral positions across venues when rate differential is profitable

#### Strategy 9: Concentrated Liquidity Manager
**Concept:** Automate Uniswap V3 / Orca Whirlpool LP management. Dynamically adjust price ranges to maximize fee income while minimizing impermanent loss. This is yield farming, not trading.

**What to build:**
- LP position monitoring and rebalancing engine
- Range optimization based on volatility predictions (use existing ML models)
- Auto-compound fees

#### Strategy 10: Multi-Platform Token Launch Sniping
**Concept:** Extend pump.fun sniping to other launch platforms: Moonshot, Base stealth launches, new Solana launchpads. Different platforms have different graduation mechanics and profit opportunities.

**What to build:**
- Platform adapter pattern for multiple launch platforms
- Shared safety analysis (reuse existing rug classifier, volume validator)
- Platform-specific graduation detection

---

## Implementation Priority (Recommended Order)

**Phase 1 - Quick Wins (Week 1):**
1. Solana Cross-DEX Arbitrage (Strategy 2) - All integrations exist, near-zero gas
2. Drift Funding Rate Arbitrage (Strategy 3) - drift_helper.py is complete
3. Enhanced Pump.fun Graduation Sniping (Strategy 4) - Extends existing code

**Phase 2 - Medium Build (Week 2-3):**
4. CEX-DEX Arbitrage (Strategy 1) - Needs Binance WebSocket price stream
5. Aave Liquidation Hunter (Strategy 5) - Needs health factor monitoring
6. MEV Backrunning (Strategy 7) - Extends mempool monitor

**Phase 3 - Advanced (Week 4+):**
7. Cross-Chain Arbitrage (Strategy 6) - Needs bridge integration
8. Perpetual DEX Funding Arb (Strategy 8) - Needs new DEX integrations
9. Concentrated Liquidity Manager (Strategy 9) - New capability
10. Multi-Platform Launch Sniping (Strategy 10) - New platforms

---

## CRITICAL Architecture Requirements for All New Modules

Every new module MUST follow these patterns:

### 1. Separate Files & Engines
- Module directory: `modules/{module_name}/`
- Main entry: `modules/{module_name}/main_{module}.py`
- Core engine: `modules/{module_name}/core/{module}_engine.py`
- Separate log files: `logs/{module_name}/{module}_trading.log`, `{module}_trades.log`, `{module}_errors.log`
- Use `RotatingFileHandler` (10MB main, 5MB error, 10MB trades, backupCount=5)
- Trade logger must have `propagate = False`

### 2. Dashboard Integration
Each module must have its own:
- Dashboard page (`dashboard_{module}.html`) - Status overview
- Performance page (`performance_{module}.html`) - Analytics
- Trade history page (`trades_{module}.html`) - All trades
- Settings page (`settings_{module}.html`) with TWO tabs:
  - **Settings tab**: Full UI controls for all module configs
  - **Guide tab**: Complete documentation with setting explanations, default values, recommended values, descriptions
- All settings saved to DB via config manager, NOT hardcoded, NOT in .env

### 3. Configuration via DB
- Use `ConfigManager` to read/write settings
- All configs stored in `config_settings` table
- Sensitive configs in `config_sensitive` (encrypted)
- Settings UI writes to DB, engine reads from DB
- Support hot-reload where possible
- Change history tracked in `config_history`

### 4. RPC via Pool Engine
- **NEVER** hardcode RPC URLs or read them directly from .env
- Always use `PoolEngine.get_endpoint(provider_type)` to get working RPC
- Report success/failure/rate-limit back to pool engine
- Pool engine handles failover, health checks, and rotation automatically

### 5. Secrets via SecureSecretsManager
- All wallet PKs, API keys, sensitive data stored encrypted in `secure_credentials` table
- Wallet addresses derived from PKs at runtime
- Use `secrets_manager.get('KEY_NAME')` to retrieve
- Never log or expose sensitive values

### 6. Database Persistence
- All trades, positions, configs, blacklisted tokens, ignored tokens saved to DB
- Use existing `trades` and `positions` table patterns (see `data/storage/models.py`)
- Create module-specific tables if needed (e.g., `liquidation_opportunities`, `funding_rate_history`)
- Use JSONB metadata columns for flexible data

### 7. Orchestrator Registration
- Add module to `main.py` orchestrator with `ModuleProcess`
- Add enable/disable flag to `.env`: `{MODULE_NAME}_MODULE_ENABLED=false`
- Validate required secrets before starting

---

## Existing Module Files to Reference

When implementing new modules, study these existing patterns:

| What | Reference File |
|------|---------------|
| Module main entry | `modules/solana_trading/main_solana.py` |
| Engine pattern | `modules/solana_trading/core/solana_engine.py` |
| Arbitrage engine | `modules/arbitrage/arbitrage_engine.py` |
| Triangular arb | `modules/arbitrage/triangular_engine.py` |
| Dashboard setup | `modules/dashboard/main_dashboard.py` |
| Config manager | `config/config_manager.py` |
| Pool engine | `config/pool_engine.py` |
| Secrets manager | `security/secrets_manager.py` |
| DB models | `data/storage/models.py` |
| Logging setup | `modules/solana_trading/main_solana.py` (lines 48-113) |
| Jupiter helper | `modules/solana_strategies/jupiter_helper.py` |
| Drift helper | `modules/solana_strategies/drift_helper.py` |
| Flash loan contracts | `contracts/FlashLoanArbitrage.sol` |
| Pump.fun strategy | `trading/strategies/pumpfun_launch.py` |
| Mempool monitor | `data/collectors/mempool_monitor.py` |
| MEV protection | `trading/executors/mev_protection.py` |
| Binance futures | `modules/futures_trading/exchanges/binance_futures.py` |
| Copy trading | `modules/copy_trading/copy_engine.py` |
| Whale tracker | `data/collectors/whale_tracker.py` |

---

## Known Issues / Lessons Learned (Avoid These Mistakes)

1. **Uniswap V3 SwapRouter does NOT support `getAmountsOut`** - That's a V2 interface. V3 requires the Quoter contract for price quotes. Never mix V2 and V3 ABIs.

2. **Flashbots bundles returning "bundle_sent" does NOT mean inclusion** - Always verify bundle hash and check transaction receipts. Builders can silently ignore bundles.

3. **Thin liquidity pools give phantom spreads** - Scanning with tiny amounts (1 token) finds unrealistic spreads that vanish at real execution sizes. Always scan with flash-loan-scale amounts.

4. **Legacy `gasPrice` fails when base fee rises** - Always use EIP-1559: `maxFeePerGas` = 2x baseFee + priorityFee, `maxPriorityFeePerGas`, `type: 2`.

5. **Positions can get stuck indefinitely** - Always implement max hold time and stale price detection. Pump.fun trailing exit can bypass standard timeout logic.

6. **Pump.fun tokens can be scams/dumps** - Tighten entry filters: reject >10% price decline, require vol/liq ratio >0.5, minimum volume AND liquidity thresholds.

7. **Config comes from DB via config_manager** - Use attribute-style access. Never assume .env values are authoritative for trading parameters.

---

Please implement these strategies following the architecture requirements above. Start with the Phase 1 strategies (Solana Cross-DEX Arb, Drift Funding Rate Arb, Enhanced Graduation Sniping) and work through the phases in order. For each module, create the engine, dashboard pages (with settings + guide tabs), DB integration, and orchestrator registration.
