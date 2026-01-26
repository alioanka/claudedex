# Enhanced Pump.fun Trailing Stop Strategy

## Problem Statement

Current Pump.fun strategy limitations:
- Fixed 100% take profit cap - misses 10x-100x gains
- No trailing stops - SL doesn't follow price up
- No partial exits - can't lock in partial profits while holding remainder
- All-or-nothing exit logic

## New Strategy Design: "Moonbag Trailing" System

### Core Concept

Instead of fixed TP, use a **dynamic trailing stop loss** that:
1. Protects capital with initial tight SL (-20%)
2. Moves SL to breakeven after small gain (+15%)
3. Trails progressively as price rises
4. Takes partial profits at key levels while keeping a "moonbag"

### Strategy Tiers

#### Tier 1: Initial Entry (0% - 15% gain)
- **Stop Loss**: -20% (capital protection)
- **Action**: Hold full position
- **Goal**: Survive initial volatility

#### Tier 2: Breakeven Lock (+15% - 50% gain)
- **Stop Loss**: 0% (breakeven)
- **Action**: Hold full position
- **Goal**: Risk-free trade

#### Tier 3: First Profit Lock (+50% - 100% gain)
- **Stop Loss**: +25% (lock 25% profit)
- **Action**: Sell 25% of position at +50%
- **Goal**: Secure some profit, hold 75%

#### Tier 4: Acceleration Zone (+100% - 300% gain)
- **Stop Loss**: +50% (trails at 50% below peak)
- **Action**: Sell another 25% at +100% (2x)
- **Goal**: Hold 50% "moonbag" for potential 10x

#### Tier 5: Moon Mode (+300%+ gain)
- **Stop Loss**: Trails at 30% below peak
- **Moonbag**: Final 50% rides with trailing SL
- **Goal**: Capture 10x-100x while protecting gains

### Trailing Stop Logic

```
peak_price = MAX(current_price, peak_price)

if gain <= 15%:
    sl_price = entry_price * 0.80  # -20%
elif gain <= 50%:
    sl_price = entry_price * 1.00  # breakeven
elif gain <= 100%:
    sl_price = entry_price * 1.25  # +25%
elif gain <= 300%:
    sl_price = peak_price * 0.50   # 50% of peak
else:  # Moon mode
    sl_price = peak_price * 0.70   # 70% of peak (30% trail)
```

### Partial Exit Schedule

| Gain Level | Action | Position After |
|------------|--------|----------------|
| +50%  | Sell 25% | 75% remaining |
| +100% | Sell 25% | 50% remaining |
| +300% | Sell 25% | 25% moonbag |
| Trailing SL hit | Sell remaining | Position closed |

### Configuration Parameters

```python
PUMPFUN_TRAILING_CONFIG = {
    # Initial protection
    'initial_stop_loss_pct': -20.0,

    # Breakeven lock
    'breakeven_trigger_pct': 15.0,

    # Trailing tiers
    'tier_1_trigger': 50.0,   # First partial at 50%
    'tier_1_sell_pct': 25.0,  # Sell 25%
    'tier_1_sl_pct': 25.0,    # SL locks at +25%

    'tier_2_trigger': 100.0,  # Second partial at 100%
    'tier_2_sell_pct': 25.0,  # Sell another 25%
    'tier_2_trail_pct': 50.0, # Trail 50% below peak

    'tier_3_trigger': 300.0,  # Moon mode at 300%
    'tier_3_sell_pct': 25.0,  # Sell another 25%
    'tier_3_trail_pct': 30.0, # Trail 30% below peak

    # Position tracking
    'track_peak_price': True,
    'update_interval_seconds': 3,  # Faster updates for pump tokens
}
```

### Implementation Details

#### 1. Position Metadata Tracking

Add to position metadata:
```python
position.metadata = {
    'strategy': 'pumpfun_trailing',
    'peak_price': current_price,
    'current_sl_price': entry_price * 0.80,
    'current_sl_pct': -20.0,
    'tier_reached': 0,
    'partial_exits': [],
    'original_amount': amount,
    'remaining_amount': amount,
    'total_realized_pnl': 0.0
}
```

#### 2. Exit Condition Logic

Replace simple TP/SL check with tiered trailing logic:

```python
def check_trailing_exit(self, position):
    """Check trailing stop exit conditions"""
    entry_price = position.entry_price
    current_price = position.current_price
    meta = position.metadata

    # Update peak price
    if current_price > meta.get('peak_price', entry_price):
        meta['peak_price'] = current_price

    peak_price = meta['peak_price']
    gain_pct = ((current_price - entry_price) / entry_price) * 100
    peak_gain_pct = ((peak_price - entry_price) / entry_price) * 100

    # Calculate dynamic SL based on tier
    sl_price = self._calculate_trailing_sl(entry_price, peak_price, gain_pct, peak_gain_pct)
    meta['current_sl_price'] = sl_price

    # Check for SL hit
    if current_price <= sl_price:
        return True, 'trailing_stop', 1.0  # Exit all remaining

    # Check for partial exits
    if peak_gain_pct >= 300 and meta.get('tier_reached', 0) < 3:
        meta['tier_reached'] = 3
        return True, 'tier_3_partial', 0.25
    elif peak_gain_pct >= 100 and meta.get('tier_reached', 0) < 2:
        meta['tier_reached'] = 2
        return True, 'tier_2_partial', 0.25
    elif peak_gain_pct >= 50 and meta.get('tier_reached', 0) < 1:
        meta['tier_reached'] = 1
        return True, 'tier_1_partial', 0.25

    return False, None, 0.0

def _calculate_trailing_sl(self, entry, peak, gain_pct, peak_gain_pct):
    """Calculate trailing stop loss price"""
    if gain_pct <= 15:
        return entry * 0.80  # -20% initial SL
    elif peak_gain_pct <= 50:
        return entry  # Breakeven
    elif peak_gain_pct <= 100:
        return entry * 1.25  # +25% locked
    elif peak_gain_pct <= 300:
        return peak * 0.50  # 50% of peak
    else:
        return peak * 0.70  # 70% of peak (30% trail)
```

#### 3. Partial Exit Execution

```python
async def execute_partial_exit(self, position, exit_pct, reason):
    """Execute partial position exit"""
    exit_amount = position.remaining_amount * exit_pct

    # Execute sell for partial amount
    result = await self._execute_swap(
        token_mint=position.token_address,
        amount=exit_amount,
        side='sell'
    )

    if result['success']:
        # Update position
        position.metadata['remaining_amount'] -= exit_amount

        # Calculate partial PnL
        partial_pnl = (position.current_price - position.entry_price) * exit_amount
        position.metadata['partial_exits'].append({
            'timestamp': datetime.now().isoformat(),
            'amount': exit_amount,
            'price': position.current_price,
            'pnl': partial_pnl,
            'reason': reason
        })
        position.metadata['total_realized_pnl'] += partial_pnl

        logger.info(f"Partial exit: {exit_pct*100}% at {reason}, PnL: {partial_pnl:.4f} SOL")
```

### Risk Management

#### Position Sizing for Moonbag Strategy
- Initial position: 0.1 SOL per trade (configurable)
- Max concurrent positions: 5 (increased for moonbag tracking)
- Daily loss limit: 3.0 SOL (reduced due to moonbag capital lock)

#### Emergency Exits
- Force close all at -30% or 24-hour timeout
- Manual close available via dashboard

### Expected Outcomes

#### Conservative Scenario (50% of trades)
- Entry: 0.1 SOL
- Exit at trailing SL after +50%: ~0.125 SOL profit
- Result: +25% gain protected

#### Normal Scenario (40% of trades)
- Partial at +50%: 0.025 SOL * 1.5 = 0.0375 SOL
- Partial at +100%: 0.025 SOL * 2.0 = 0.05 SOL
- Trailing SL at +80%: 0.05 SOL * 1.8 = 0.09 SOL
- Total: ~0.18 SOL (+75% gain)

#### Moon Scenario (10% of trades)
- Partials at +50%, +100%, +300%
- Final 25% moonbag at trailing to 10x
- Total: ~0.5-1.0 SOL (+400-900% gain)

### Files to Modify

1. **`modules/solana_trading/core/solana_engine.py`**
   - `_check_exit_conditions()` - Add trailing logic
   - `_monitor_positions()` - Track peak prices
   - `_close_position()` - Support partial exits

2. **`modules/solana_trading/config/solana_config_manager.py`**
   - Add trailing config parameters

3. **`modules/solana_strategies/solana_module.py`**
   - Update Pump.fun strategy defaults

### Dashboard Integration

Update Solana dashboard to show:
- Peak price reached
- Current tier
- Partial exits taken
- Trailing SL level
- Moonbag remaining

### Migration Path

1. Add new config parameters with defaults
2. Implement trailing logic alongside existing TP/SL
3. Enable via config flag: `use_trailing_strategy: true`
4. A/B test with dry run mode
5. Gradually enable for live trades

---

## Quick Start Implementation

### Step 1: Add Config
Add to `solana_config_manager.py`:
```python
'trailing_strategy_enabled': True,
'trailing_config': PUMPFUN_TRAILING_CONFIG
```

### Step 2: Modify Position Tracking
Add `peak_price` tracking in `_update_position_prices()`

### Step 3: Replace Exit Logic
Replace fixed TP/SL with `check_trailing_exit()` function

### Step 4: Add Partial Exit Support
Implement `execute_partial_exit()` method

### Step 5: Test
Run with `DRY_RUN=true` to validate before live trading
