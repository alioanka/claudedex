# Wave-14 Backlog

All items are deferred from Wave-13. Items marked **data-dependent** require
live DRY_RUN or LIVE trade data before implementation is meaningful.
Items marked **infra-dependent** require external setup (contracts, API keys).

Effort: S = <1 day, M = 1-3 days, L = 3-7 days.
Impact: ROI/operational/security rating relative to current state.

---

## DEX

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| DEX-W14-1 | `_load_state` no-op — in-engine SL/rug gate not firing for restored positions after restart | M | HIGH | After restart `active_positions` is empty; position_service.py compensates for PnL display but the in-engine exit logic (stop-loss, rug-prob gate) does not fire for restored positions. Port the Wave-8 `_load_state` DB restore from the main checkout. |
| DEX-W14-2 | Train ML ensemble once 50+ closed trades exist | S | HIGH | **Data-dependent.** Run `python scripts/train_ensemble.py --days 90` on VPS after >=50 closed DEX trades exist. Then set `ml_retrain_enabled=true` in DB. Until trained, `predict_decoupled` falls back to heuristic. |
| DEX-W14-3 | Verify `/latest/dex/pairs/{chain}` returns fresh pairs with `pairCreatedAt` | S | MEDIUM | Strategy-4 uses this endpoint; unclear if it returns a `createdAfter`-filterable field. If not, the age filter kills all hits. |
| DEX-W14-4 | Buyer/seller ratio hard-reject filter | S | MEDIUM | `buyers_24h / (buyers_24h + sellers_24h) < 0.4` → bearish sign, hard-reject. Currently unused field in DexScreener response. Low implementation cost. |
| DEX-W14-5 | Adaptive min_score per chain (gas cost signal) | S | LOW | ETH mainnet pairs need higher conviction due to gas. Chain-specific `min_opportunity_score` reduces false positives on expensive chains. |

---

## ARBITRAGE

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| ARB-W14-1 | Uniswap V3 Quoter integration | L | HIGH | V3 handles the majority of ETH mainnet + Base volume. Add `quoteExactInputSingle` via V3 Quoter (0xb27308f9F90D607463bb33eA1BeBb41C27CE5AB6). Also add Aerodrome V2 quoter on Base. |
| ARB-W14-2 | Flashbots signing key | S | HIGH | **Infra-dependent.** `FlashbotsExecutor` is already coded. Add `FLASHBOTS_SIGNING_KEY` to secrets and provision a signing key. Without it, ETH mainnet arb goes to the public mempool (sandwich risk). |
| ARB-W14-3 | Reduce flash_loan_amount for L2 chains | S | HIGH | 10 ETH causes 0.5-1% price impact on shallow Arbitrum/Base V2 pools. Reduce to 1-2 ETH on Arbitrum, 0.5-1 ETH on Base. |
| ARB-W14-4 | Adaptive min_profit_threshold | S | MEDIUM | **Data-dependent.** After 48-72h of unblocked scan data, compare near-miss `profit_bps` distribution against fills. Lower from 0.3% to 0.1% if median near-miss clusters 0.15-0.3%. |
| ARB-W14-5 | Additional DEX venues | M | MEDIUM | ETH: Curve (stablecoin), Balancer (WBTC/WETH). Arbitrum: TraderJoe, Ramses. Base: Aerodrome V2, PancakeSwap Base. Each multiplies spread discovery surface. |

---

## SNIPER

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| SNIP-W14-1 | Jito bundle for snipe tx | L | HIGH | **Infra-dependent.** Current Solana snipe uses standard Jupiter swap. In competitive new-mint environment MEV bots front-run via Jito bundles. Wire `JITO_BLOCK_ENGINE_URL` in pool_engine + `submit_jito_bundle(txs, tip_lamports)` in trade_executor.py. |
| SNIP-W14-2 | Market-cap gate at entry | M | HIGH | **Data-dependent.** Fresh pump.fun mints with <$10k liquidity and <$5k initial market cap have ~90% rug rate. Add `min_market_cap_usd` + `min_pool_liquidity_usd` DB config keys. Calibrate thresholds from live data. |
| SNIP-W14-3 | Dynamic TP based on holder velocity | L | MEDIUM | If holder count grows >50/min in first 60s, increase TP 2x. Requires tracking `getTokenLargestAccounts` time series. Out of scope until holder-count data collection is added. |
| SNIP-W14-4 | Per-chain TP/SL differentiation | S | MEDIUM | pump.fun launches: high-vol (50% loss or 5x in minutes). Raydium V4: calmer. Add `pump_fun_take_profit_multiplier` config key. |
| SNIP-W14-5 | CHECK constraint on sniper_trades.profit_loss_pct | S | LOW | Agent 6 proposed `CHECK (profit_loss_pct BETWEEN -100 AND 300)`. Safe to add only AFTER cleaning up violating rows via `wave13_db_queries.sh` optional DELETE. Do not add before cleanup — ALTER TABLE fails if rows violate the constraint. |

---

## COPY_TRADING

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| COPY-W14-1 | Kelly sizing enable | S | HIGH | **Data-dependent.** Infrastructure is complete (migration 024, leader_scorer.py). Run `scripts/refresh_copy_leaders.py`, confirm scores are reasonable, then set `kelly_sizing_enabled=true`. Until then all leaders get flat sizing. |
| COPY-W14-2 | Helius Enhanced Transactions API | M | HIGH | Replace `getSignaturesForAddress`+`getTransaction` (2 RPC calls/sig) with Helius `/v0/addresses/{address}/transactions` (1 call per address). Cuts RPC spend 5x and reduces detection lag. |
| COPY-W14-3 | WebSocket subscription for Solana leaders | L | MEDIUM | Current polling latency worst-case 120s. Helius `logsSubscribe` provides <500ms detection. Add `_ws_monitor_solana_wallets` coroutine alongside polling loop. |
| COPY-W14-4 | EVM token extraction: proper ABI decode | M | MEDIUM | `'0x' + input_data[-40:]` heuristic works for V2 but extracts garbage for V3/aggregator methods now recognized (Wave-13). Add per-method ABI decoders for 5 most common V3 layouts. |
| COPY-W14-5 | Slippage-aware leader filtering | M | MEDIUM | **Data-dependent.** Once 50+ `copy_slippage_observations` accumulate, auto-probate leaders with p95(delta_ms) > configurable threshold (undetectable edge due to execution lag). |
| COPY-W14-6 | Raydium CLMM program ID verification | S | LOW | `cjZmBEP64PBnkBDsVjBJbzHuP4jSJXpRqVzC3GC1Fh3` in expanded DEX_PROGRAMS should be verified against `raydium.io/clmm` canonical docs before LIVE. |

---

## SOLANA

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| SOL-W14-1 | SOL-out verification on close | M | HIGH | After Jupiter sell, verify SOL balance increased by at least 80% of expected amount. If not, flag as partial fill and retry. Prevents silent value loss on bad routes. |
| SOL-W14-2 | Dynamic Jito tip based on opportunity size | S | HIGH | Current flat 50k lamports. For large PnL signals, bid 5% of expected profit (capped at configured max). Increases fill rate on high-competition mints. |
| SOL-W14-3 | pnl_pct clamp at monitoring update | S | MEDIUM | Current clamp is only at `_save_trade_to_db`. If monitoring calculates impossible `unrealized_pnl_pct`, the position may trigger incorrect TP exit. Add `max(min(...))` at the `position.unrealized_pnl_pct =` assignment in `_monitor_positions`. |
| SOL-W14-4 | pump.fun entry latency guard | S | MEDIUM | Add `pumpfun_min_age_seconds` config key (default 30s): skip tokens created less than this many seconds ago to skip the initial bot-sniping phase. |
| SOL-W14-5 | Wire `pool_engine.report_failure` on SolanaRpcException | S | LOW | Engine uses its own `_wallet_balance_error_streak` backoff but does not notify pool_engine. Wiring this lets the pool rotate to a different Helius endpoint. |

---

## FUTURES

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| FUT-W14-1 | Funding-rate carry strategy | L | HIGH | Add a parallel scan: when Bybit funding rate > `futures_min_funding_rate_pct` (suggest 0.05%/8h), enter the side that collects funding. Separate position cap for funding-carry positions. |
| FUT-W14-2 | Adaptive signal score threshold | S | MEDIUM | **Data-dependent.** After 48-72h of unblocked signals (volume gate fixed in Wave-13), observe near-miss counters. Lower `min_signal_score` from 2 to 1 if `confluence` dominates misses. |
| FUT-W14-3 | Per-symbol min signal score | S | LOW | Add `futures_min_score_{symbol}` config key support. BTC/ETH can justify higher conviction; alt perps less so. |

---

## AI

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| AI-W14-1 | `scripts/retrain_ai_calibration.py` | M | HIGH | **Data-dependent.** `ai_confidence_calibration` table accumulates (score, realized_pnl) rows on position close (requires `direct_trading=true` first). Once 100+ rows exist, train logistic regression on (abs_score, fgi, headline_count) → profitable_bool and replace raw LLM output with calibrated probability. |
| AI-W14-2 | Headline deduplication cache | S | MEDIUM | Cache analyzed headline sets by sorted SHA256; skip LLM call if identical batch analyzed within 2 cycles. Saves API cost and prevents stale news locking the score. |
| AI-W14-3 | Per-symbol sentiment routing | M | MEDIUM | Build keyword-to-symbol map; filter headlines per symbol before LLM call; run separate score per symbol. Currently ETH trades on BTC news. |
| AI-W14-4 | Fear & Greed Index integration | S | MEDIUM | `alternative.me/fng/` returns 0-100 daily composite. Use as second feature alongside LLM score for cross-asset confirmation. Zero extra LLM cost. |
| AI-W14-5 | `/ai/calibration` nav page | M | LOW | Dashboard page for `ai_confidence_calibration` table: scatter plot of (score vs realized_pnl), calibration curve, re-train trigger button. Requires AI-W14-1 data first. |

---

## DASHBOARD

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| DASH-W14-1 | `/pool/status` timeline page | M | MEDIUM | Show time-series of pool endpoint health scores, 429 counts, and rotation events. Data is available in `rpc_api_pool` history. Useful for Helius capacity planning. |
| DASH-W14-2 | `/ai/calibration` nav page | M | LOW | See AI-W14-5 above. |
| DASH-W14-3 | Per-leader slippage p50/p95 on leaders page | S | MEDIUM | **Data-dependent.** `copy_slippage_observations` data is already being collected. Add aggregation query to `/copytrading/leaders` once 50+ observations/leader exist. |

---

## INFRA

| ID | Item | Effort | Impact | Notes |
|----|------|--------|--------|-------|
| INFRA-W14-1 | `rpc_api_pool.api_key` column encryption | M | MEDIUM | Helius API keys stored plaintext in `api_key` column (Agent 9 R1). Encrypt at application layer using `security/encryption.py` before persisting; decrypt on load in `_load_from_database()`. Medium-severity for hosted VPS; low-severity if DB access is tightly controlled. |
| INFRA-W14-2 | Priority reset on rate-limit expiry | S | LOW | Agent 9 R2: after 18 consecutive 429s, endpoint priority reaches ~1000 and needs 180 successful calls to recover. Reset priority to base (100) when `rate_limit_until` expires. |
| INFRA-W14-3 | Targeted recovery probe for rate-limited endpoints | S | LOW | Agent 9 R3: health check interval is 3600s. Add an async targeted probe when `rate_limit_until` passes so recovered endpoints are re-promoted within seconds, not hours. |
| INFRA-W14-4 | `SOLANA_WSS` dedicated pool slot | M | MEDIUM | Agent 6 / Agent 9 handoff: WSS reconnect rotation derives WSS URL from HTTP endpoint hostname (works for Helius/QuickNode). Multi-provider setups may want separate WSS-specific keys. Add `SOLANA_WSS` provider type to pool_engine. |
