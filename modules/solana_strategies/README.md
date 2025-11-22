# Solana Strategies Module

Advanced Solana-specific trading strategies module for ClaudeDex. This module focuses exclusively on Solana ecosystem opportunities with three specialized strategies.

## Overview

The Solana Strategies Module implements cutting-edge trading strategies unique to the Solana blockchain:

1. **Pump.fun Launch Trading** - Early entry on new token launches
2. **Jupiter Limit Orders** - Better execution with limit orders
3. **Drift Perpetuals** - Leveraged perpetual futures trading

## Strategies

### 1. Pump.fun Launch Trading

Monitors and trades new token launches on the Pump.fun platform.

**Features:**
- Real-time launch monitoring via WebSocket
- Bonding curve progress analysis (10-85% sweet spot)
- Developer behavior tracking (dev holdings, sells)
- Holder distribution analysis
- Sniper detection and filtering
- Graduated token support (post-Raydium migration)

**Risk Controls:**
- Quick stop loss (15%)
- Multi-level take profits (30%, 50%, 100%, 200%)
- Maximum hold time enforcement (1 hour default)
- Developer concentration limits
- Anti-bundled launch detection

**Entry Criteria:**
- Token age < 30 minutes
- Bonding progress: 10-85%
- Minimum 20 holders
- Buy/sell ratio > 1.5
- Dev holding < 5%
- Top 10 holders < 25%

**Configuration:**
```yaml
strategies:
  pumpfun:
    enabled: true
    max_token_age: 1800
    min_bonding_progress: 10.0
    max_bonding_progress: 85.0
    quick_stop_loss_pct: 0.15
    tp_levels: [0.30, 0.50, 1.00, 2.00]
```

### 2. Jupiter Limit Orders

Integrates with Jupiter's limit order functionality for better execution.

**Features:**
- Support/resistance level detection
- Smart limit order placement (1% below market)
- Automated stop loss and take profit orders
- Partial fill support
- Auto-cancel unfilled orders
- DCA (Dollar Cost Averaging) strategies
- Range trading capabilities

**Advantages:**
- Reduced slippage vs market orders
- Better entry prices
- Passive market making
- Off-hours trading
- No MEV exposure

**Use Cases:**
1. Better entry prices on trending tokens
2. Range trading in sideways markets
3. DCA into positions over time
4. Limit order exits for open positions
5. Reduce execution costs

**Configuration:**
```yaml
strategies:
  jupiter:
    enabled: true
    use_limit_orders: true
    limit_order_offset_pct: 0.01  # 1% better
    max_order_age: 3600
    partial_fill_enabled: true
    dca_enabled: false
```

### 3. Drift Protocol Perpetuals

Trades perpetual futures on Drift Protocol (Solana-native).

**Features:**
- Up to 20x leverage (configurable, default 5x)
- Both LONG and SHORT positions
- Funding rate arbitrage
- Cross-margin support
- Liquidation monitoring with 25% buffer
- Trailing stop loss
- Partial take profits (4 levels)

**Advantages over CEX Perpetuals:**
- Non-custodial (your keys, your coins)
- On-chain transparency
- Lower fees
- Solana speed (400ms transactions)
- Composability with DeFi protocols
- No KYC required

**Strategies:**
1. **Directional Trading** - Trend following with leverage
2. **Funding Rate Arbitrage** - Collect funding payments
3. **Hedging** - Hedge DEX positions
4. **Basis Trading** - Cash-and-carry arbitrage
5. **Delta Neutral** - Market making strategies

**Risk Management:**
- Conservative leverage (5x default)
- 25% liquidation buffer
- Trailing stop loss
- Multi-level take profits
- Auto-deleverage on losses

**Configuration:**
```yaml
strategies:
  drift:
    enabled: true
    max_leverage: 5
    funding_rate_enabled: true
    perp_stop_loss_pct: 0.08
    liquidation_buffer_pct: 0.25
    preferred_markets:
      - "SOL-PERP"
      - "BTC-PERP"
      - "ETH-PERP"
```

## Capital Allocation

The module has independent capital allocation:

```yaml
capital:
  allocation: 400.0  # Total module capital
  max_position_size: 80.0  # Max per position
  risk_per_trade: 0.02  # 2% risk
```

**Per-Strategy Limits:**
- Pump.fun: 2% per launch
- Jupiter: Based on support/resistance
- Drift: 2-3% with leverage

## Position Management

**Position Limits:**
```yaml
positions:
  max_open: 8  # Total across all strategies
  max_per_strategy: 3
  max_per_token: 1
```

**Exit Strategy:**
- Trailing stop loss (enabled by default)
- Multi-level take profits
- Maximum hold time (strategy-dependent)
- Auto-close on funding rate changes (Drift)

## Architecture

```
modules/solana_strategies/
├── __init__.py
├── solana_module.py          # Main module
└── README.md

trading/strategies/
├── pumpfun_launch.py          # 500+ lines
├── jupiter_limit_orders.py    # 600+ lines
└── drift_perpetuals.py        # 650+ lines

config/modules/
└── solana_strategies.yaml     # Configuration
```

## Integration

### With Module Manager

```python
from modules.integration import setup_modular_architecture

# Module manager auto-discovers solana_strategies
module_manager = await setup_modular_architecture(
    engine, db_manager, cache_manager, alert_manager, risk_manager
)

# Module is automatically registered and ready
```

### Manual Control

```python
# Enable module
await module_manager.enable_module("solana_strategies")

# Disable module
await module_manager.disable_module("solana_strategies")

# Get metrics
metrics = await module_manager.get_module_metrics("solana_strategies")
```

### Dashboard Control

Access via Settings > Modules tab:
- Enable/Disable module
- View real-time metrics
- Monitor positions
- Control individual strategies

## Monitoring

**Real-time Monitoring:**
- Pump.fun: Every 5 seconds (fast for launches)
- Jupiter: Every 30 seconds (order status)
- Drift: Every 15 seconds (positions, funding)

**Health Checks:**
- RPC endpoint connectivity
- Wallet balance monitoring
- Position liquidation risk
- Capital utilization
- Win rate tracking

## Performance Metrics

**Module-Level:**
- Total trades
- Win rate
- Total PnL
- Capital utilization
- Active positions

**Strategy-Level:**
- Per-strategy trades
- Per-strategy PnL
- Active positions
- Average hold time
- Success rate

## Risk Management

**Module-Level Controls:**
```yaml
risk:
  stop_loss_pct: 0.10
  take_profit_pct: 0.20
  max_drawdown_pct: 0.20  # Auto-pause at 20%
```

**Strategy-Specific:**
- Pump.fun: 15% quick SL
- Jupiter: 8% SL with 2:1 RR
- Drift: 8% SL (amplified by leverage)

**Safety Features:**
- Liquidation buffer monitoring (Drift)
- Auto-cancel stale orders (Jupiter)
- Blacklist detection (Pump.fun)
- Maximum hold time enforcement
- Capital allocation limits

## Configuration

### Basic Setup

1. Configure wallet:
```yaml
wallets:
  solana_strategies:
    solana:
      address: "${SOLANA_WALLET_ADDRESS}"
      private_key_env: "SOLANA_PRIVATE_KEY"
      initial_balance: 400.0
```

2. Adjust capital allocation:
```yaml
capital:
  allocation: 400.0  # Your capital
```

3. Enable strategies:
```yaml
strategies:
  pumpfun:
    enabled: true
  jupiter:
    enabled: true
  drift:
    enabled: true
```

### Advanced Configuration

**Pump.fun Tuning:**
- Adjust bonding curve range for more/fewer opportunities
- Modify holder requirements
- Change TP levels
- Adjust max hold time

**Jupiter Tuning:**
- Enable DCA for gradual entries
- Enable range trading
- Adjust limit order offset
- Configure partial fills

**Drift Tuning:**
- Increase leverage for more aggressive trading
- Enable funding rate arbitrage
- Add more markets
- Adjust liquidation buffer

## API Integration

### Pump.fun API
- Frontend API: `https://frontend-api.pump.fun`
- WebSocket: `wss://frontend-api.pump.fun/socket`
- Real-time launch events

### Jupiter API
- Quote API: `https://quote-api.jup.ag/v6`
- Limit Orders: `https://api.jup.ag/limit/v1`
- Order management

### Drift Protocol
- Program ID: `dRiftyHA39MWEi3m9aunc5MzRF1JYuBsbn6VPcn33UH`
- SDK integration
- On-chain positions

## Testing

### Testnet Configuration

```yaml
solana:
  rpc_primary: "https://api.devnet.solana.com"

strategies:
  drift:
    drift_rpc: "https://api.devnet.solana.com"
```

### Backtesting

Each strategy supports backtesting:

```python
# Backtest Pump.fun strategy
results = await pumpfun_strategy.backtest(
    historical_data=launch_history,
    initial_balance=Decimal("1000")
)
```

## Troubleshooting

**Common Issues:**

1. **No Pump.fun launches detected**
   - Check WebSocket connection
   - Verify filters aren't too restrictive
   - Ensure module is running

2. **Jupiter orders not filling**
   - Limit price may be too aggressive
   - Increase `limit_order_offset_pct`
   - Enable partial fills

3. **Drift liquidation warnings**
   - Reduce leverage
   - Increase liquidation buffer
   - Check position sizing

**Logs:**
```bash
# View module logs
tail -f logs/solana_strategies.log

# View strategy-specific logs
grep "Pump.fun" logs/solana_strategies.log
```

## Performance Expectations

**Pump.fun:**
- Win rate: 40-60% (high volatility)
- Average hold: 15-30 minutes
- Typical PnL: +30% to -15%
- Opportunities: 10-50 per day

**Jupiter:**
- Win rate: 60-70% (better entry)
- Fill rate: 50-80%
- Price improvement: 0.5-2%
- Reduced slippage costs

**Drift:**
- Win rate: 50-60%
- Average hold: 2-8 hours
- Typical PnL: +15% to -8%
- Funding collected: 10-30% APR

## Roadmap

**Near-term:**
- [ ] Add more Drift markets (BONK-PERP, WIF-PERP)
- [ ] Implement cross-strategy hedging
- [ ] Add Raydium CP pools support
- [ ] Enhance Pump.fun filters

**Long-term:**
- [ ] Mango Markets integration
- [ ] Phoenix DEX integration
- [ ] ML-based launch prediction
- [ ] Automated parameter optimization

## Resources

- [Pump.fun Documentation](https://pump.fun)
- [Jupiter Limit Orders](https://station.jup.ag/docs/limit-order/limit-order-integration)
- [Drift Protocol Docs](https://docs.drift.trade)
- [Solana Web3.js](https://solana-labs.github.io/solana-web3.js/)

## Support

For issues or questions:
- Check module logs
- Review configuration
- Test in testnet first
- Monitor dashboard metrics
