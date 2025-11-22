# 📦 Trading Modules System

## Overview

The modular architecture transforms the trading bot into a flexible, extensible system where each trading strategy or approach operates as an independent module.

## Architecture

```
claudedex/
├── modules/
│   ├── base_module.py          # Abstract base class for all modules
│   ├── integration.py          # Integration helper
│   ├── dex_trading/            # DEX spot trading module
│   │   ├── dex_module.py
│   │   └── __init__.py
│   ├── futures_trading/        # CEX futures module (Phase 2)
│   └── arbitrage/              # Arbitrage module (Phase 3)
├── core/
│   └── module_manager.py       # Central module orchestration
├── config/
│   ├── modules/                # Module-specific configs
│   │   ├── dex_trading.yaml
│   │   ├── futures_trading.yaml
│   │   ├── arbitrage.yaml
│   │   └── shared_risk.yaml
│   └── wallets.yaml            # Wallet separation config
└── monitoring/
    ├── module_routes.py        # Dashboard API routes
    └── dashboard/templates/
        └── modules.html        # Module management UI
```

## Core Components

### 1. BaseModule (Abstract Class)

All trading modules inherit from `BaseModule`:

```python
from modules.base_module import BaseModule, ModuleConfig, ModuleType

class MyModule(BaseModule):
    async def initialize(self) -> bool:
        # Setup resources
        pass

    async def start(self) -> bool:
        # Start operations
        pass

    async def stop(self) -> bool:
        # Stop operations
        pass

    async def process_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        # Handle trading opportunity
        pass

    async def get_positions(self) -> List[Dict]:
        # Return current positions
        pass

    async def get_metrics(self) -> ModuleMetrics:
        # Return performance metrics
        pass
```

### 2. ModuleManager

Central orchestration for all modules:

- **Module Registration**: Register/unregister modules
- **Lifecycle Management**: Start/stop/pause/resume modules
- **Health Monitoring**: Automatic health checks
- **Risk Coordination**: Shared risk management
- **Capital Allocation**: Manage capital distribution
- **Metrics Aggregation**: Unified portfolio view

```python
from core.module_manager import ModuleManager

# Create manager
manager = ModuleManager(
    config={'total_capital': 1000.0},
    db_manager=db,
    cache_manager=cache,
    alert_manager=alerts,
    risk_manager=risk
)

# Initialize and start
await manager.initialize()
await manager.start()

# Register module
manager.register_module(my_module)

# Control modules
await manager.enable_module('dex_trading')
await manager.disable_module('futures_trading')
```

### 3. Module Configuration

Each module has a YAML configuration file in `config/modules/`:

```yaml
# config/modules/dex_trading.yaml

module:
  name: "dex_trading"
  type: "dex_trading"
  enabled: true
  description: "DEX spot trading module"

capital:
  allocation: 500.0
  max_position_size: 100.0
  risk_per_trade: 0.02

positions:
  max_open: 5
  max_per_chain: 3

risk:
  stop_loss_pct: 0.05
  take_profit_pct: 0.10

strategies:
  enabled:
    - momentum
    - scalping
    - ai_strategy
```

### 4. Wallet Separation

Each module can have dedicated wallets for better tracking:

```yaml
# config/wallets.yaml

dex_trading:
  solana:
    address: "${SOLANA_WALLET_ADDRESS}"
    private_key_env: "SOLANA_PRIVATE_KEY"
    initial_balance: 500.0

futures_trading:
  binance:
    api_key_env: "BINANCE_FUTURES_API_KEY"
    api_secret_env: "BINANCE_FUTURES_API_SECRET"
    initial_balance: 300.0
```

## Integration

### Quick Start

```python
# In your main.py or startup code

from modules.integration import setup_modular_architecture

# Setup modules
module_manager = await setup_modular_architecture(
    engine=trading_engine,
    db_manager=db,
    cache_manager=cache,
    alert_manager=alerts,
    risk_manager=risk,
    dashboard=dashboard  # Optional
)

# Attach to engine
engine.module_manager = module_manager

# Start trading
await engine.start()
```

### Manual Integration

```python
from modules.integration import ModuleIntegration

# Create integration helper
integration = ModuleIntegration(
    trading_engine=engine,
    db_manager=db,
    cache_manager=cache,
    alert_manager=alerts,
    risk_manager=risk
)

# Setup modules
module_manager = await integration.setup_modules()

# Integrate with dashboard
if dashboard:
    integration.integrate_with_dashboard(dashboard, module_manager)
```

## Dashboard Access

Access the module management interface at:

**http://localhost:8080/modules**

Features:
- View all modules and their status
- Enable/disable modules
- Pause/resume operations
- View module metrics
- Monitor capital allocation
- Real-time position tracking

### API Endpoints

```
GET  /modules                              # Module management page
GET  /api/modules                          # Get all modules status
GET  /api/modules/{name}                   # Get specific module status
POST /api/modules/{name}/enable            # Enable module
POST /api/modules/{name}/disable           # Disable module
POST /api/modules/{name}/pause             # Pause module
POST /api/modules/{name}/resume            # Resume module
GET  /api/modules/{name}/metrics           # Get module metrics
GET  /api/modules/{name}/positions         # Get module positions
POST /api/modules/reallocate               # Reallocate capital
```

## Available Modules

### 1. DEX Trading Module ✅ (Phase 1 - LIVE)

**Status**: Implemented and ready
**Purpose**: Spot trading on decentralized exchanges

**Features**:
- Multi-chain support (Solana, EVM chains)
- Jupiter aggregator integration
- Momentum, Scalping, AI strategies
- MEV protection
- Dynamic route optimization

**Configuration**: `config/modules/dex_trading.yaml`

**Supported Chains**:
- Solana (Raydium, Orca, Jupiter)
- Ethereum (Uniswap, SushiSwap)
- Polygon, BSC, Arbitrum, Base

### 2. Futures Trading Module ⏳ (Phase 2 - PLANNED)

**Status**: Placeholder, to be implemented
**Purpose**: Short positions and hedging on CEX

**Features** (planned):
- Binance, Bybit, OKX integration
- Trend following strategies
- DEX hedge capability
- Funding rate arbitrage

**Configuration**: `config/modules/futures_trading.yaml`

### 3. Arbitrage Module ⏳ (Phase 3 - PLANNED)

**Status**: Placeholder, to be implemented
**Purpose**: Cross-DEX and CEX-DEX arbitrage

**Features** (planned):
- Spatial arbitrage (cross-DEX)
- Triangular arbitrage
- Cross-chain arbitrage
- CEX-DEX arbitrage

**Configuration**: `config/modules/arbitrage.yaml`

## Creating a New Module

### Step 1: Create Module Class

```python
# modules/my_module/my_module.py

from modules.base_module import BaseModule, ModuleConfig, ModuleType, ModuleMetrics

class MyTradingModule(BaseModule):
    def __init__(self, config: ModuleConfig, **kwargs):
        super().__init__(config, **kwargs)
        # Your initialization

    async def initialize(self) -> bool:
        # Setup your module
        return True

    async def start(self) -> bool:
        # Start trading operations
        self._running = True
        self.status = ModuleStatus.RUNNING
        return True

    async def stop(self) -> bool:
        # Stop trading operations
        self._running = False
        self.status = ModuleStatus.STOPPED
        return True

    async def process_opportunity(self, opportunity: Dict) -> Optional[Dict]:
        # Your trading logic
        pass

    async def get_positions(self) -> List[Dict]:
        # Return your positions
        return []

    async def get_metrics(self) -> ModuleMetrics:
        # Return your metrics
        return self.metrics
```

### Step 2: Create Configuration

```yaml
# config/modules/my_module.yaml

module:
  name: "my_module"
  type: "my_module"
  enabled: false
  description: "My custom trading module"

capital:
  allocation: 100.0
  max_position_size: 50.0

# Your custom settings
```

### Step 3: Register Module

```python
# In integration.py or your startup code

my_module = MyTradingModule(
    config=my_config,
    db_manager=db,
    cache_manager=cache,
    alert_manager=alerts
)

module_manager.register_module(my_module)
```

## Module Lifecycle

```
DISABLED ──enable()──> INITIALIZING ──initialize()──> STOPPED
                                                         │
                                                     start()
                                                         │
                                                         ▼
ERROR <────error────── RUNNING ──pause()──> PAUSED
  │                       │
  │                   stop()
  │                       │
  └──enable()──> STOPPING ──> STOPPED
```

## Monitoring & Health

### Automatic Health Checks

The module manager performs automatic health checks every 30 seconds:

```python
# Health check indicators
- Module responsiveness
- Error count
- Position validity
- Capital allocation
- Active task status
```

### Metrics Tracking

Each module tracks:

- **Performance**: PnL, win rate, profit factor, Sharpe ratio
- **Activity**: Total trades, active positions, capital usage
- **Health**: Uptime, error count, last trade time
- **Risk**: Drawdown, exposure, position limits

### Alerts

Automatic alerts for:
- Module errors
- Health check failures
- Capital exhaustion
- High drawdown
- Position limit reached

## Risk Management

### Shared Risk Limits

Configuration in `config/modules/shared_risk.yaml`:

```yaml
global:
  max_total_capital: 1000.0
  max_drawdown_pct: 0.20
  max_daily_loss: 100.0
  emergency_stop_loss: 200.0

cross_module:
  max_correlation: 0.7
  max_long_exposure: 800.0
  max_short_exposure: 400.0

circuit_breakers:
  pause_on_high_volatility: true
  pause_on_drawdown: true
  consecutive_loss_threshold: 3
```

### Per-Module Limits

Each module has its own risk parameters:

```yaml
risk:
  stop_loss_pct: 0.05
  take_profit_pct: 0.10
  max_drawdown_pct: 0.15
  trailing_stop_enabled: true
```

## Best Practices

1. **Start Small**: Test each module with small capital first
2. **Monitor Closely**: Watch module metrics in first 24-48 hours
3. **Gradual Scaling**: Increase capital allocation after proven performance
4. **Diversify**: Use multiple modules to spread risk
5. **Regular Review**: Check module performance weekly
6. **Keep Updated**: Update module configs based on market conditions

## Troubleshooting

### Module Won't Start

1. Check configuration file exists and is valid YAML
2. Verify wallet addresses are configured
3. Check logs for initialization errors
4. Ensure capital is allocated

### Module Shows Error Status

1. Check dashboard for error message
2. Review module logs
3. Verify API keys and wallets are valid
4. Check network connectivity

### Positions Not Showing

1. Verify module is running (not just enabled)
2. Check if module has capital allocated
3. Ensure position limits haven't been reached
4. Review module-specific filters

## Future Enhancements

- [ ] Module templates for quick creation
- [ ] Hot reload of module configurations
- [ ] Module marketplace for community modules
- [ ] A/B testing framework for strategies
- [ ] Module performance comparison tools
- [ ] Automatic capital rebalancing
- [ ] Module dependency management

## Support

For issues or questions:
1. Check logs in `/logs/modules/`
2. Review module configuration files
3. Check dashboard alerts
4. Consult main README.md

## Version History

- **v1.0.0** (2025-11-21): Initial module system implementation
  - Base module framework
  - Module manager
  - DEX trading module
  - Dashboard integration
  - Configuration system
  - Wallet separation
