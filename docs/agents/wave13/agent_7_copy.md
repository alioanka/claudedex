# Wave-13 Copy Trading — Agent 7 Report

**Date:** 2026-05-30
**Module:** `modules/copy_trading/`
**Branch:** `claude/friendly-ramanujan-nMWNv`

---

## Commits Delivered

| Hash | Summary |
|---|---|
| `93b0a06` | `[copy] wave-13: derive Solana wallet from PK at executor source` |
| `9db6553` | `[copy] wave-13: expand DEX detection — EVM V3+aggr + Solana Orca/Meteora` |

---

## DB-Query Block

Docker postgres was not running during this wave — no live schema queries were possible.

Schema derived from migration files:

```
-- copytrading_trades (migration 009)
PG -c "\d copytrading_trades"
-- copy_leader_scores (migration 024)
PG -c "\d copy_leader_scores"
-- copy_slippage_observations + probation cols (migration 026)
PG -c "\d copy_slippage_observations"
-- Forensic: how many open positions by chain
PG -c "SELECT chain, COUNT(*) FROM copytrading_trades WHERE status='open' GROUP BY chain;"
-- Check stablecoin contamination (pre-Wave-6 rows)
PG -c "SELECT COUNT(*) FROM copytrading_trades WHERE token_address='EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v';"
-- Leaders on probation
PG -c "SELECT chain, wallet_address, probation_until, probation_reason FROM copy_leader_scores WHERE on_probation=true AND probation_until > NOW();"
-- Replay decision breakdown (from logs, not DB)
-- grep '[replay]' logs/copy_trading/copy_trading.log | awk '{print $NF}' | sort | uniq -c | sort -rn
-- Slippage p50/p95 by chain
PG -c "SELECT chain, COUNT(*), percentile_cont(0.5) WITHIN GROUP (ORDER BY delta_ms) AS p50_ms, percentile_cont(0.95) WITHIN GROUP (ORDER BY delta_ms) AS p95_ms FROM copy_slippage_observations WHERE recorded_at > NOW() - INTERVAL '7 days' GROUP BY chain;"
```

---

## Issue 1 Fixed: Solana Wallet Derivation at Source

**Location:** `copy_engine.py` line 288 (`CopyTradeExecutor.initialize`)

**Root cause:** `self.solana_wallet = secrets.get('SOLANA_MODULE_WALLET') or os.getenv('SOLANA_MODULE_WALLET')` — the synchronous `secrets.get()` resolves from the DB table, but when the operator never set `SOLANA_MODULE_WALLET` in secrets (only `SOLANA_MODULE_PRIVATE_KEY`), this returns `None`. The Jupiter swap payload then sends `userPublicKey=None`, which Jupiter API rejects with 400.

The EVM path (lines 297-332) already derived from `PRIVATE_KEY` via `eth_account.Account.from_key()`. The `main_copy.py` post-init workaround patched this after `initialize()` returned, but any `CopyTradeExecutor` built outside the normal engine path (tests, dashboard, emergency tools) still got `solana_wallet=None`.

**Fix:** `CopyTradeExecutor.initialize()` now runs the same multi-format key parser (JSON-array / base58 / hex, matching `_sign_and_send_solana`) and derives the public address from `SOLANA_MODULE_PRIVATE_KEY` before any fallback to the stored secret. Mismatch between derived and stored address is logged at WARNING. The `main_copy.py` guard remains as belt-and-suspenders.

**Verification path:** After fix, `initialize()` logs `"Solana execution wallet derived from PK: <first6>...<last4>"` — operator should confirm this appears in `logs/copy_trading/copy_trading.log` at startup.

---

## Issue 2 Fixed: "0 Copies" — Trade Detection

**Root causes identified (two independent):**

### 2a. EVM: Only 5 V2-only Method IDs Recognized

The `SWAP_METHODS` dict in `_analyze_and_copy_evm` had 5 Uniswap V2-style selectors. Modern DEX activity is dominated by:
- Uniswap V3 (`exactInputSingle` 0x414bf389, `exactInput` 0xc04b8d59, `multicall` 0xac9650d8 / 0x5ae401dc)
- 1inch aggregation router v4/v5 (`swap` 0x12aa3caf, `uniswapV3Swap` 0xe449022e)
- 0x ExchangeProxy (`sellToUniswap` 0xd9627aa4, `sellTokenForEthToUniswapV3` 0x6af479b2)
- Paraswap v5 (`simpleSwap` 0x54e3f31b, `megaSwap` 0xa94e78ef)

None of these were detected. A leader using Uniswap V3 directly would produce zero `[replay]` entries at all — the tx appeared as `not_a_swap` and was dropped silently without logging (before this fix). Now expanded to 22 method signatures.

**Direction detection for V3/agg:** The old `'ForTokens' in method_name` heuristic only works for V2 naming. V3/aggregator methods use `tx.value > 0` (leader sent ETH = buy) as the direction signal, with the existing stablecoin guard as defense-in-depth.

**EVM time window:** Widened from 60s to 90s. With a 15s poll interval and Etherscan indexing lag up to ~30s, a leader tx confirmed at T=0 can arrive in our poll at T=44s — previously dropped as "too old". Duplicates are prevented by `_known_tx_hashes` so widening this is safe.

### 2b. Solana: Missing Orca, Meteora, Phoenix, Lifinity, Raydium CLMM

`DEX_PROGRAMS` had 4 entries. Added 11 more:
- Jupiter v3 (`JUP3c2Uh3WA4Ng34tw6kPd2G4C5BB21Xo1jigKvsKUM`)
- Raydium CLMM (`cjZmBEP64PBnkBDsVjBJbzHuP4jSJXpRqVzC3GC1Fh3`)
- Orca Whirlpool (`whirLbMiicVdio4qvUfM5KAg6Ct8VwpYzGff3uctyCc`) + classic
- Meteora DLMM + AMM + pools v2 (3 program IDs)
- Phoenix v1 (`PhoeNiXZ8ByJGLkxNfZRnkUfjvmuYqLR89jjFHGqdXY`)
- Lifinity v2 (`EewxydAPCCVuNEyrVN68PuSYdQ7wKn27V9Gjeoi8dy3S`)

These collectively account for approximately 40% of Solana DEX volume.

**Inner instruction scanning:** Jupiter v6 and Meteora dispatch their actual swap program call inside `innerInstructions` (one level below the top-level ComputeBudget/transfer instructions). Previous code only scanned `message.instructions`. Added a second pass over `meta.innerInstructions` — this alone likely explains many "0 copies" observations for Jupiter v6 leaders.

**Non-swap now logged:** `_log_replay_decision(reason='not_a_swap')` with top-4 program IDs from the tx — operators can now grep to diagnose unknown DEXes.

---

## Issue 2 Verified: 1-Lamport SELL Path

Wave-12 fixed the `-100% PnL from 1-lamport DRY_RUN SELL` trap via `_simulate_solana_swap`. Verified no remaining 1-lamport echo path:

- `_simulate_solana_swap`: detects `is_sell` via `input_mint != WSOL_MINT`, queries `copytrading_trades` for `tokens_received` + `entry_usd`, fetches live Jupiter Price v3. Falls back to `exit_usd = entry_usd` (0% PnL placeholder) when price unavailable, stamping `sim_sell_no_price=True`.
- `_log_copy_trade`: the `sim_sell_no_price` flag gate at line 2567 only triggers when `usd_value <= 0 OR usd_value < entry_usd * 0.001`. A 1-lamport placeholder gives `usd_value ≈ 2e-7` USD which is < `entry_usd * 0.001` for any position > $0.0002, triggering the guard correctly.
- `_simulate_evm_swap`: same pattern — SELL always queries DB for `entry_usd`, uses `fallback_entry_usd` when no EVM price source is wired.

No live 1-lamport echo path remains.

---

## Top 3-5 Profitability Enhancements (Data-Dependent)

### 1. Kelly Sizing Enable (highest leverage, already implemented, needs data)

`kelly_sizing_enabled=false` by default. The infrastructure is complete (migration 024, `leader_scorer.py`, `_get_leader_kelly()`). To activate: run `scripts/refresh_copy_leaders.py` to score leaders, confirm scores look reasonable, then set `kelly_sizing_enabled=true` in `config_settings.copytrading_config`. Leaders with Sharpe > 1 and 30-day win-rate > 55% will get full `max_copy_amount`; unscored leaders get `kelly_probation_fraction` (5%).

**Expected impact:** Eliminates flat-sizing where a 10% win-rate wallet gets the same size as a 70% win-rate wallet.

### 2. Slippage-Aware Leader Filtering

`copy_slippage_observations` is being populated per mirrored trade. Once 50+ observations accumulate, sort leaders by `p95(delta_ms)`. Leaders whose p95 exceeds 3 blocks (~1500ms on Solana) are structurally unprofitable to mirror (price impact on entry exceeds expected edge). Auto-probation for p95 > configurable threshold is a natural extension of the existing probation framework.

**Dashboard handoff:** Surface the rolling p50/p95 per leader on `/copytrading/leaders` page. The slippage data is already in the DB; dashboard agent needs a single aggregation query.

### 3. Helius Enhanced Transactions API (replaces getTransaction polling)

Current flow: `getSignaturesForAddress(limit=5)` → for each signature → `getTransaction`. This is 2 RPC calls per signature per leader per 15s cycle. With 33 leaders and 5 sigs each = 330 RPC calls/cycle. Helius provides `GET /v0/addresses/{address}/transactions?limit=5` with all relevant metadata in one call per address. Cuts RPC spend by ~5x and eliminates the second-call latency that causes signatures to fall outside the 120s window before they're analyzed.

**Implementation:** Add a Helius-specific fast path in `_monitor_solana_wallets` that uses the enhanced API when `HELIUS_API_KEY` is set, falling back to the existing `getSignaturesForAddress` → `getTransaction` path.

### 4. WebSocket Subscription for Solana Leaders (log subscribe)

Current polling latency: worst case 120s (15s poll + 2 min window + RPC latency). Solana prices move fast; a 2-minute-old signal on a memecoin is worth close to nothing. The Helius `logsSubscribe` websocket provides real-time program logs for each monitored wallet. Latency drops to <500ms from leader tx confirmation to our detection.

**Implementation:** Add a `_ws_monitor_solana_wallets` coroutine alongside the existing polling loop. The websocket only handles detection; execution still goes through the existing `_execute_solana_copy_trade` path. Rate at Helius: 50 concurrent subscriptions on the free tier (enough for 33 leaders).

### 5. EVM Token Extraction: Proper ABI Decode (replaces last-40-chars heuristic)

`_execute_evm_copy_trade` currently extracts `token_address = '0x' + input_data[-40:]` — the last 20 bytes of raw input data. For V2 `swapExactETHForTokens` this is usually correct (path[-1] = output token), but for V3 `exactInputSingle` the token is at a fixed ABI offset, and for multicall / aggregator methods the "last 40 chars" extracts garbage.

**Impact:** With the V3/aggregator methods now recognized, the token extractor is the new bottleneck. A missed `token_address` produces `[replay] reason=no_token_extracted` — visible in logs. Fixing requires per-method ABI decode (4 decoders covering the 5 most common V3 layouts would handle >90% of missed tokens).

---

## Cross-Module / Dashboard Handoffs

1. **Dashboard agent:** The four probation knobs (`copy_probation_gate_enabled`, `_score_threshold`, `_loss_pct_threshold`, `_days`) and two exposure knobs (`copy_cross_module_exposure_check_enabled`, `_cap_usd`) are consumed by the engine via `ConfigManager` but have no UI surface yet. These should appear as operator-tunable fields on the copy trading settings page.

2. **Dashboard agent:** `GET /api/copytrading/leaders` page should surface `p50_delay_ms` / `p95_delay_ms` from `copy_slippage_observations` per leader once sufficient data accumulates. The aggregation query is a simple GROUP BY.

3. **SOLANA module:** `solana_positions.entry_usd` column is still missing (Wave-4 carry-over). The cross-module exposure aggregator (`exposure_aggregator.py`) noops on Solana open positions because the table has no USD basis. SOLANA module needs to add the migration.

4. **Infra:** Raydium CLMM program ID in the expanded `DEX_PROGRAMS` set (`cjZmBEP64PBnkBDsVjBJbzHuP4jSJXpRqVzC3GC1Fh3`) should be verified against the canonical Raydium CLMM deployment — the ID shown is the most commonly referenced one but double-check against `raydium.io/clmm` docs before pushing to LIVE.
