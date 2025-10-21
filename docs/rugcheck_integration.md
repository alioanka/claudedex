# RugCheck Integration - Deployment Guide

## 📋 Prerequisites

```bash
# Ensure you have solders installed
pip list | grep solders
# Should show: solders==0.18.1

# If not installed:
pip install solders==0.18.1
```

---

## 🚀 Deployment Steps

### Step 1: Backup Current File
```bash
cd ~/claudedex
cp data/collectors/honeypot_checker.py data/collectors/honeypot_checker.py.backup
```

### Step 2: Apply Changes

Open the file and apply all 8 sections from the artifact:

```bash
nano data/collectors/honeypot_checker.py
```

**Changes to make:**

1. **Line 1-10**: Add `from solders.pubkey import Pubkey` to imports
2. **Line 358-369**: Update `_get_chain_id()` to include Solana
3. **Line 58-65**: Add Solana routing at top of `check_token()`
4. **After line 150**: Add `_check_solana_token()` method (~40 lines)
5. **After that**: Add `_check_rugcheck_summary()` method (~60 lines)
6. **After that**: Add `_calculate_solana_verdict()` method (~100 lines)
7. **Top of file**: Add `SAFE_SOLANA_TOKENS` constant
8. **In `__init__`**: Update session timeout

### Step 3: Verify Syntax

```bash
python3 -m py_compile data/collectors/honeypot_checker.py
echo $?  # Should output: 0 (success)
```

### Step 4: Restart Bot

```bash
docker compose restart trading-bot
```

### Step 5: Watch Logs

```bash
# Watch for Solana activity
docker compose logs -f trading-bot | grep -i solana

# You should see:
# ✅ HoneypotChecker initialized (EVM + Solana support)
# 🔗 Scanning SOLANA...
# ✅ Using relaxed checks for Solana token...
# 🎯 OPPORTUNITY: [TOKEN] on solana
```

---

## 🧪 Testing

### Test 1: Known Safe Token (BONK)
```bash
# Should pass immediately via whitelist
# BONK: DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263
```

Expected log:
```
✅ Whitelisted token: BONK
```

### Test 2: RugCheck API Response
```bash
# Test RugCheck API directly
curl -X 'GET' \
  'https://api.rugcheck.xyz/v1/tokens/DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263/report/summary' \
  -H 'accept: application/json'

# Should return score_normalised: 7 for BONK
```

### Test 3: Check Database for Solana Trades
```bash
# Wait 15-30 minutes after restart, then:
docker compose exec postgres psql -U claudedex -d claudedex -c \
  "SELECT COUNT(*) as solana_trades FROM trades WHERE chain = 'solana';"

# Expected: > 0 trades
```

### Test 4: Dashboard Verification
```bash
# Open dashboard
http://YOUR_VPS_IP:8080/dashboard

# Check "Open Positions" table
# Look for chain = "solana"
```

---

## 📊 Expected Behavior

### ✅ Good Outcomes:

1. **Whitelisted tokens** → Pass instantly
2. **Score 0-2** → Allow (minimal risk)
3. **Score 3-4** → Allow (low risk)
4. **Score 5-6** → Allow with warning (medium risk)

### ⚠️ Blocked Outcomes:

5. **Score 7-8** → Block (high risk)
6. **Score 9-10** → Block (critical risk)
7. **Critical risks** → Block immediately

### 🆕 New Token Handling:

- **Not found in RugCheck** → Allow with "high risk" warning
- **API timeout/error** → Allow with "unknown risk" (don't block trading)

---

## 🔍 Monitoring Commands

### Watch Solana Discovery:
```bash
docker compose logs -f trading-bot | grep -E "(Scanning SOLANA|Found.*SOLANA)"
```

### Watch Honeypot Checks:
```bash
docker compose logs -f trading-bot | grep -E "(RugCheck|solana token|Whitelisted)"
```

### Watch Trade Execution:
```bash
docker compose logs -f trading-bot | grep -E "(Jupiter|SOLANA.*POSITION|solana.*trade)"
```

### Check RugCheck Rate Limit:
```bash
docker compose logs trading-bot | grep "RugCheck rate limit remaining"
```

---

## 🐛 Troubleshooting

### Issue 1: Import Error (solders not found)
```bash
# Install in Docker container
docker compose exec trading-bot pip install solders==0.18.1
docker compose restart trading-bot
```

### Issue 2: Still No Solana Trades After 30 Minutes
```bash
# Check if Solana discovery working:
docker compose logs trading-bot | grep "Scanning SOLANA"

# Check if pairs found:
docker compose logs trading-bot | grep "Found.*SOLANA"

# Check if honeypot blocking:
docker compose logs trading-bot | grep -A5 "SOLANA"
```

### Issue 3: RugCheck API Rate Limited
```bash
# Check rate limit headers in logs:
docker compose logs trading-bot | grep "x-rate-limit"

# Rate limit is 15/min, we're using 12/min with buffer
# If hitting limit, reduce in honeypot_checker.py:
@rate_limit(calls=10, period=60.0)  # Lower from 12 to 10
```

### Issue 4: All Solana Tokens Showing "not_found"
```bash
# This is normal for very new/obscure tokens
# They will be allowed through with "high risk" warning
# Check if trades are still executing despite this
```

---

## 📈 Success Metrics

After 1 hour, you should see:

- ✅ **5-15 Solana pairs discovered** per scan cycle
- ✅ **1-3 Solana trades executed** (depending on market conditions)
- ✅ **Database has Solana entries**: `SELECT * FROM trades WHERE chain = 'solana' LIMIT 5;`
- ✅ **Dashboard shows Solana positions** in open positions table

---

## 🔄 Rollback (If Needed)

```bash
cd ~/claudedex
cp data/collectors/honeypot_checker.py.backup data/collectors/honeypot_checker.py
docker compose restart trading-bot
```

---

## 📞 Next Steps After Deployment

1. **Monitor for 1 hour** - Watch logs for Solana activity
2. **Verify first trade** - Check database and dashboard
3. **Adjust thresholds** - If too strict/loose, modify `_calculate_solana_verdict()`
4. **Review P&L** - After 24h, check Solana vs EVM performance
5. **Fine-tune scoring** - Adjust score thresholds based on results

---

## 🎯 Quick Verification Checklist

```bash
# 1. Check bot is running
docker compose ps

# 2. Check Solana enabled
grep "SOLANA_ENABLED" .env

# 3. Check RugCheck integration working
docker compose logs trading-bot | grep "RugCheck" | tail -20

# 4. Check for Solana trades
docker compose exec postgres psql -U claudedex -d claudedex -c \
  "SELECT token_symbol, entry_price, profit_loss FROM trades WHERE chain = 'solana' ORDER BY entry_time DESC LIMIT 5;"

# 5. Check dashboard
curl -s http://localhost:8080/api/stats | jq '.solana_trades'
```

---

## 📝 Configuration Tuning

### Make Solana Checks MORE Strict:
```python
# In _calculate_solana_verdict(), lower thresholds:
if score_normalised >= 6:  # Instead of 7
    risk_level = "high"
    is_honeypot = True
```

### Make Solana Checks LESS Strict:
```python
# Allow more medium-risk tokens:
if score_normalised >= 8:  # Instead of 7
    risk_level = "high"
    is_honeypot = True
```

### Increase LP Lock Weight:
```python
# In _calculate_solana_verdict(), add:
if lp_locked_pct > 50:  # If >50% LP locked
    score_normalised -= 2  # Reduce risk score by 2 points
```

---

## ✅ Success Indicators

You'll know it's working when you see:

```log
🔗 Scanning SOLANA... (min liquidity: $5,000)
✅ Found 12 pairs on SOLANA
🔬 Analyzing TOKEN_XYZ on solana
   RugCheck score: 4/10 (low risk)
   LP locked: 35.5%
✅ Token passed safety checks
🎯 OPPORTUNITY: TOKEN_XYZ on solana - Score: 0.820
🔷 Using Jupiter executor for Solana
📝 DRY RUN - OPENED POSITION: TOKEN_XYZ
💰 Position opened: +$10,000 (Paper)
```

🚀 **Happy Trading!**