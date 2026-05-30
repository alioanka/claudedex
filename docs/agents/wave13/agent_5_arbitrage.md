# Wave-13 Agent 5: Arbitrage Module Audit

## Root Cause: -10005 bps / "No Opportunities Ever Fire"

### Primary Bug (FIXED in commit 9d6bb76)

**Location:** `modules/arbitrage/arbitrage_engine.py` — `ARB_PAIRS_*` constants + `_check_arb_opportunity`

**Mechanism:**
19 of 31 scan pairs across all three chains had `token_out != 'WETH'`. The engine hardcodes
`borrow_amount = self.flash_loan_amount` in ETH wei (`10 * 10^18`). Every getAmountsOut call
uses borrow_amount as the input even when token_out is USDC (6 decimals). Querying
`getAmountsOut(10^18, [USDC_addr, token_addr])` is equivalent to asking for the output of
10^12 USDC — the call either reverts, returns 0, or returns a nonsensical value.
Consequence: `weth_returned ≈ 0`, spread = `(0 - 10^18) / 10^18 = -1.0 = -10000 bps`.
The -10005 bps figure is exactly `flash_loan_fee` making it -10005 not -10000.

**Broken pair types (pre-fix):**
- `('WETH', 'USDC')`, `('WETH', 'USDT')`, `('WETH', 'DAI')` — wrong direction; token_out is a stablecoin
- `('USDC', 'USDT')`, `('USDC', 'DAI')`, `('DAI', 'USDT')` — pure stablecoin, require USDC flash loan not WETH
- `('ARB', 'USDC')`, `('AERO', 'USDC')` — non-WETH borrow asset

**Valid pairs (correct before fix):** only `(token, 'WETH')` pairs like `('WBTC','WETH')`, `('LINK','WETH')` etc.
These were scanning correctly which is why logs showed realistic small negative spreads like `-0.6081%` for
`WETH uniswap_v2→sushiswap` (WBTC/WETH pair).

**Fix applied:**
1. Inverted `('WETH','X')` to `('X','WETH')` for USDC/USDT/DAI on all three chains.
2. Removed pure stablecoin pairs that require a USDC-denominated flash loan.
3. Added runtime guard in `_check_arb_opportunity` that rejects any pair where `token_out` is not the chain's WETH address, logs a clear diagnosis, and prevents future bad pairs from silently poisoning the scan.

Net result: valid pair count 12 -> 22 across all chains; all pairs now produce WETH-denominated spread calculations.

---

## Fee/Gas/Slippage Model Analysis

### Current model (post-fix, correct chains):

**Ethereum mainnet** (10 ETH borrow):
- Gas cost: `450k gas * 30 gwei * $3500/ETH = ~$47`, gas_frac = 0.135%
- Slippage: 0.4% (static default, ~0.2% per swap * 2 legs)
- Flash loan fee: 0.05% (already deducted from weth_returned via amount_owed)
- min_profit_threshold: 0.3% (adaptive, scales with gas spike multiplier)
- Total hurdle: raw_spread > 0.135% + 0.4% + 0.3% = 0.835%

**Arbitrum** (10 ETH borrow):
- Gas cost: `1.2M gas * 0.1 gwei * $3500/ETH = ~$0.42`, gas_frac = 0.0012%
- Slippage: 0.5% (L2 default)
- Total hurdle: raw_spread > 0.0012% + 0.5% + 0.3% = 0.801%

**Base** (10 ETH borrow):
- Gas cost: negligible (~$0.10-0.25)
- Slippage: 0.5%
- Total hurdle: raw_spread > ~0.8%

### Issue: hurdle rate vs DEX fee reality

V2 DEX fees are 0.3% per swap = 0.6% for two legs. For arbitrage to be profitable, a trader on one DEX must be moving the price by at least the fee differential plus our threshold. With 0.5% slippage + 0.3% min_profit, the engine needs 0.8% raw spread on top of DEX fees. Real cross-DEX spreads on liquid V2 pairs are typically 0.1-0.4%.

**Recommendation (not yet implemented — requires observation of real spreads first):**
- After the pair-direction bug is fixed, observe actual near-miss logs for `min_profit` reason.
- If spreads cluster around 0.2-0.4%, consider reducing `min_profit_threshold` from 0.3% to 0.1%.
- Slippage can be reduced dynamically: the `get_realized_slippage` learning system will kick in after first real fills.

---

## Other Issues Found

### Dead code: `amount_in` in `_check_arb_opportunity`
Line 1872: `amount_in = self._scan_amounts.get(token_symbol, self.flash_loan_amount)` is computed but never referenced in any price query. `borrow_amount` drives all getAmountsOut calls. Not a bug but misleading. Low priority to clean.

### TOKEN_SCAN_AMOUNTS_BASE missing tokens
LINK, UNI, AAVE, SUSHI, CRV, ARB, GMX, AERO, DEGEN are absent from `TOKEN_SCAN_AMOUNTS_BASE`. They fall back to `self.flash_loan_amount` for `amount_in` (dead code, no real effect). But if `amount_in` were ever used, this would inject wrong values for 6-decimal tokens. Can be addressed if `amount_in` is ever wired into scan queries.

### Triangular engine (21 routes, 0 opportunities)
Correctly gated by atomic-receiver guard (MB-05). The route math is sound — decimal handling is done per-token at line 635. The 0% hit rate is expected: execution is disabled, routes run in scan-only mode, and real 3-leg arb requires contract deployment. Not a bug.

### Solana arb (disabled)
`solana_enabled=false` in settings. Expected. Solana spatial arb via Jupiter is a different product track.

---

## DB Schema (not accessible — Docker not running in worktree)

Schema discovery SQL to run when DB is available:

```sql
-- Table schemas
\d arbitrage_runtime_stats
\d arbitrage_trades
\d arb_realized_slippage
\d arbitrage_positions

-- Runtime stats per chain (check staleness)
SELECT chain, updated_at, NOW() - updated_at AS age,
       stats->>'last_tick_at' AS last_tick,
       stats->>'last_error' AS last_error,
       stats->'near_miss_counters' AS near_misses
FROM arbitrage_runtime_stats
ORDER BY chain;

-- Near-miss breakdown (why no trades)
SELECT chain,
       stats->'near_miss_counters' AS counters
FROM arbitrage_runtime_stats;

-- Realized slippage learning status
SELECT chain, dex_pair, pair_symbol, sample_count, median_pct, p90_pct, updated_at
FROM arb_realized_slippage
ORDER BY chain, sample_count DESC;

-- Arbitrage trade history (last 24h)
SELECT chain, buy_dex, sell_dex, spread_pct, profit_loss_pct,
       entry_timestamp, status, is_simulated, tx_hash
FROM arbitrage_trades
WHERE entry_timestamp > NOW() - INTERVAL '24 hours'
ORDER BY entry_timestamp DESC;
```

---

## Top Enhancements by ROI (post pair-direction fix)

### 1. Add Uniswap V3 Quoter for price discovery (HIGH impact, MEDIUM effort)
The engine only has V2 routers. Uniswap V3 handles the vast majority of ETH mainnet and Base volume. Adding the V3 Quoter contract (0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6) enables querying V3 pools (multiple fee tiers: 0.05%, 0.3%, 1%). On Ethereum, Uniswap V3 + SushiSwap V2 spreads are significantly larger than V2+V2. Also add Aerodrome V2 quoter on Base (highest Base DEX volume by far).

### 2. Reduce flash_loan_amount for L2 chains (HIGH impact, LOW effort)
10 ETH ($25k) causes 0.5-1% price impact on Arbitrum/Base V2 pools (shallow liquidity, typically $50-200k TVL). Reducing to 1-2 ETH on Arbitrum and 0.5-1 ETH on Base would: (a) reduce price impact and thus improve scan accuracy, (b) lower slippage on execution, (c) reduce capital risk per trade. The 0.3% threshold remains valid; gas fraction stays near-zero on L2.

### 3. Add more DEX venues (MEDIUM impact, LOW effort)
- Ethereum: Add Curve (for stablecoin arb via 3pool, much tighter than V2), Balancer (for WBTC/WETH).
- Arbitrum: Add TraderJoe, Ramses (V2-compatible), Chronos.
- Base: Add Aerodrome V2 (dominant DEX), PancakeSwap Base.
Each additional DEX multiplies spread discovery surface area. Requires V2-compatible ABI or adapting the Quoter path.

### 4. Adaptive min_profit_threshold tuning (MEDIUM impact, LOW effort — data-dependent)
Current 0.3% min_profit_threshold + 0.5% slippage = 0.8% total hurdle. Once real spread data flows in post-fix, compare `profit_bps` in near-miss logs against actual fills. If median near-miss profit is 0.15-0.3%, lower threshold to 0.1% to capture more opportunities. Requires 48-72h of observation first.

### 5. Flashbots bundle submission for Ethereum mainnet (HIGH impact — already coded, needs endpoint key)
The `FlashbotsExecutor` is already implemented. Needs `FLASHBOTS_SIGNING_KEY` in secrets and a valid signing key. Without it, all ETH mainnet opportunities go to the public mempool (sandwich risk). On Arbitrum/Base, Jito-style MEV protection is not needed (sequencers control ordering), but the current code skips the opportunity entirely if Flashbots fails (MB-04 guard), which is correct behavior on mainnet.

---

## Cross-Module / Dashboard Handoffs

- **Dashboard**: Near-miss counters now populated with real `min_profit` / `raw_spread_negative` reasons after pair fix. The "Why no trades?" panel should show meaningful data.
- **Dashboard**: `/api/arbitrage/positions` still reads `arbitrage_positions WHERE status='open'` but engine never writes that table (atomic arb). The close buttons are 404. Recommend hiding them (noted in CLAUDE.md).
- **pool_engine**: Wave-11 dRPC 429 backoff is in place. If Base engine is still hitting rate limits, provision a non-public Alchemy/QuickNode Base key and add via pool_engine env. Log line names `RPC_ENV_KEY` / `RPC_PROVIDER_KEY` for operator.
- **risk_manager**: `validate_trade(token_in, amount)` called pre-execute. Verify risk manager thresholds allow $25k ETH-denominated trades — if max_trade_usd is set too low, every opportunity will be blocked at the risk gate.
- **Triangular engine**: Gated by MB-05 (atomic receiver contract). Deploy the 3-leg Solidity contract to activate. Route math is correct; gate is the only blocker.
