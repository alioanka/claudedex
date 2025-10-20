# 🔷 Solana Integration Guide for ClaudeDex

## 📋 Overview

This guide covers adding Solana support to your ClaudeDex trading bot using Jupiter aggregator.

---

## 🚀 Installation Steps

### 1. **Install Solana Dependencies**

```bash
# In your venv
pip install solders==0.18.1
pip install base58==2.1.1
```

Add to `requirements.txt`:
```
solders==0.18.1
base58==2.1.1
```

---

### 2. **Create Directory Structure**

```bash
mkdir -p trading/chains/solana
touch trading/chains/__init__.py
touch trading/chains/solana/__init__.py
```

---

### 3. **Add Files**

Create these files in `trading/chains/solana/`:

1. **`__init__.py`**:
```python
# trading/chains/solana/__init__.py
from .jupiter_executor import JupiterExecutor, JupiterQuote, JupiterRoute
from .solana_client import SolanaClient

__all__ = [
    'JupiterExecutor',
    'JupiterQuote',
    'JupiterRoute',
    'SolanaClient'
]
```

2. **`jupiter_executor.py`** - Copy from artifact above

3. **`solana_client.py`** - Copy from artifact above

---

### 4. **Update Configuration**

#### **Option A: Update existing config.yaml**

```yaml
# Add to your config.yaml
solana:
  enabled: true
  rpc_url: "https://api.mainnet-beta.solana.com"
  backup_rpcs:
    - "https://solana-api.projectserum.com"
    - "https://rpc.ankr.com/solana"
  
  private_key: "your_solana_private_key_here"  # Base58 or byte array
  
  jupiter_url: "https://quote-api.jup.ag/v6"
  max_slippage: 0.05  # 5%
  
  priority_fee: 5000  # Micro-lamports
  compute_unit_price: 1000
  compute_unit_limit: 200000
  
  min_liquidity: 5000  # USD
  max_position_size_sol: 5
  min_trade_size_sol: 0.1
```

#### **Option B: Use .env file**

Copy all `SOLANA_*` variables from the `.env` artifact above to your `.env` file.

---

### 5. **Apply Engine Patches**

Apply the patches from the `PATCH: core/engine.py` artifact:

1. Add imports at top
2. Initialize Solana executor in `__init__()`
3. Initialize in `initialize()` method
4. Update `_execute_opportunity()` with chain routing
5. Update `cleanup()` method

---

### 6. **Update DexScreener Discovery**

Update `data/sources/dexscreener.py` to include Solana:

```python
# In get_trending_pairs method, add Solana chain
chains = config.get('enabled_chains', 'ethereum,bsc,base,arbitrum,polygon,solana').split(',')

# Handle Solana chain ID
if chain.lower() == 'solana':
    chain_id = 'solana'  # DexScreener uses 'solana' not a number
```

---

## 🔑 Getting a Solana Wallet

### **Option 1: Generate New Wallet**

```python
# Run this once to generate a new wallet
from solders.keypair import Keypair
import base58

# Generate keypair
keypair = Keypair()

# Get private key in different formats
private_key_bytes = bytes(keypair)[:32]
private_key_base58 = base58.b58encode(private_key_bytes).decode('utf-8')
private_key_hex = private_key_bytes.hex()

print(f"Public Key: {keypair.pubkey()}")
print(f"Private Key (Base58): {private_key_base58}")
print(f"Private Key (Hex): {private_key_hex}")
print(f"Private Key (Array): {list(private_key_bytes)}")

# ⚠️ SAVE THE PRIVATE KEY SECURELY!
# Fund this address with SOL before trading
```

### **Option 2: Export from Phantom/Solflare**

1. Open Phantom wallet
2. Settings → Security & Privacy → Export Private Key
3. Enter password
4. Copy the private key (Base58 format)
5. Add to `.env` as `SOLANA_PRIVATE_KEY`

---

## 🧪 Testing

### **Test 1: Verify Installation**

```python
# test_solana_setup.py
import asyncio
from trading.chains.solana import JupiterExecutor

async def test():
    config = {
        'jupiter_url': 'https://quote-api.jup.ag/v6',
        'solana_rpc_url': 'https://api.mainnet-beta.solana.com',
        'solana_private_key': 'YOUR_KEY_HERE',
        'max_slippage': 0.05
    }
    
    executor = JupiterExecutor(config)
    await executor.initialize()
    
    # Test quote (SOL → USDC)
    quote = await executor.get_quote(
        input_mint='So11111111111111111111111111111111111111112',  # SOL
        output_mint='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  # USDC
        amount_lamports=100000000,  # 0.1 SOL
    )
    
    if quote:
        print(f"✅ Quote received!")
        print(f"Input: {quote.in_amount / 1e9} SOL")
        print(f"Output: {quote.out_amount / 1e6} USDC")
        print(f"Price Impact: {quote.price_impact_pct:.2f}%")
    else:
        print("❌ Failed to get quote")
    
    await executor.cleanup()

asyncio.run(test())
```

### **Test 2: Dry Run Trade**

```bash
# Update .env
DRY_RUN=true
SOLANA_ENABLED=true
ENABLED_CHAINS=solana

# Start bot
python main.py
```

Check logs for Solana opportunities and simulated trades.

---

## 📊 Popular Solana Token Mints

```python
# Common tokens for testing
TOKENS = {
    'SOL': 'So11111111111111111111111111111111111111112',
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
    'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
    'SRM': 'SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt',
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
}
```

---

## 🔍 Monitoring Solana Trades

### **Dashboard Updates**

The dashboard will automatically show Solana trades with:
- Chain: `SOLANA`
- Executor: `Jupiter 🔷`
- Transaction links to Solscan

### **Check Wallet Balance**

```python
from trading.chains.solana import SolanaClient

async def check_balance():
    client = SolanaClient(['https://api.mainnet-beta.solana.com'])
    await client.initialize()
    
    balance = await client.get_balance('YOUR_WALLET_ADDRESS')
    print(f"SOL Balance: {balance} SOL")
    
    await client.cleanup()
```

---

## ⚠️ Important Notes

### **Gas Fees (Rent + Priority)**
- Base transaction: ~5,000 lamports (0.000005 SOL)
- Priority fee: 5,000-50,000 lamports (configurable)
- Total per trade: ~$0.005-0.05 USD

### **Slippage**
- Jupiter auto-calculates optimal slippage
- Default: 500 bps (5%)
- Adjust in config for volatile tokens

### **RPC Rate Limits**
- Public RPC: ~5 requests/second
- Use paid RPC (Helius, QuickNode) for production
- Automatic failover to backup RPCs

### **Transaction Confirmation**
- Solana is much faster than EVM (~400ms vs 12s)
- Confirmed in 1-2 blocks typically
- Uses versioned transactions (v0) for efficiency

---

## 🐛 Troubleshooting

### **Issue: "Wallet not initialized"**
```bash
# Check private key format
python -c "import base58; print(len(base58.b58decode('YOUR_KEY')))"
# Should output: 32 or 64
```

### **Issue: "Quote failed"**
- Check internet connection
- Verify token mints are valid
- Ensure liquidity exists for pair
- Try different RPC endpoint

### **Issue: "Transaction failed"**
- Insufficient SOL for fees
- Slippage too low for volatile token
- Token has transfer restrictions
- Check transaction on Solscan

---

## 📚 Resources

- **Jupiter Docs**: https://station.jup.ag/docs/
- **Solana Docs**: https://docs.solana.com/
- **Solscan Explorer**: https://solscan.io/
- **Token List**: https://token.jup.ag/all

---

## ✅ Integration Checklist

- [ ] Install dependencies (`solders`, `base58`)
- [ ] Create directory structure
- [ ] Add Jupiter executor files
- [ ] Update config/env with Solana settings
- [ ] Apply engine.py patches
- [ ] Generate or import Solana wallet
- [ ] Fund wallet with SOL (~0.1 SOL minimum)
- [ ] Test quote fetching
- [ ] Run dry run test
- [ ] Monitor first real trade
- [ ] Update dashboard for Solana display

---

## 🎯 Next Steps

1. **Start with dry run** (`DRY_RUN=true`)
2. **Test with small amounts** (0.1 SOL)
3. **Monitor execution** (check Solscan)
4. **Scale up gradually**

**Ready to trade on Solana!** 🚀