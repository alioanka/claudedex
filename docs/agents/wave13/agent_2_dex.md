# Wave-13 Agent 2 — DEX Module Quant/ML Review

## Commits in this wave

| Hash | Summary |
|------|---------|
| ce95e53 | [dex] wave-13: fix scoring/risk bugs + Pydantic model_dump |
| d782eec | [dex] wave-13: wire ML ensemble + port predict_decoupled + feature_builder |
| 1889364 | [dex] wave-13: functional auto-retrain loop + port train_ensemble.py |
| 7c7e991 | [dex] wave-13: fix EVM chain discovery — strategy 4 + wider boost scan |

---

## 1. Existing-edge audit

The DEX engine (`core/engine.py:TradingBotEngine`) runs a continuous scan-score-execute loop:

- **Discovery**: `DexScreenerCollector.get_new_pairs(chain, limit)` queries 3 strategies (boost list, profile list, quote-token search) per chain, filtered by liquidity/volume/age.
- **Scoring**: `_calculate_opportunity_score` — pure heuristic (volume, liquidity, price change, risk, age). No ML input.
- **ML path**: `EnsemblePredictor` exists but was never called in the hot path. Entry always used `ml_confidence=score`, `rug_probability=0.2` (flat constant), `pump_probability=score*0.8` — fully fabricated.
- **Sizing**: Half-Kelly with opportunity multiplier, clamped by min/max position size from config.
- **Execution**: `direct_dex.py` via Uniswap V2/V3, with real QuoterV2 impact estimation (Wave-3/4).

**Edge sources (genuine):**
- Momentum on new tokens (age < 24h, volume spike) — valid DEX-sniper edge when signal is real.
- Half-Kelly sizing over portfolio balance — appropriate for a volatile, fat-tailed asset class.
- QuoterV2 real-impact scoring for route selection (Wave-4) — prevents routing into thin pools.

---

## 2. Biases and bugs found (Wave-13)

### BUG 1 — Scoring rejects 100% of real opportunities (CRITICAL, FIXED)
**Location:** `core/engine.py:_calculate_opportunity_score` (liquidity block)

The vol/liq ratio hard-reject was `volume_to_liq_ratio < 2.0`. This means a pair needs 24h volume > 2x its liquidity to pass. Real active DEX pairs run 0.1x–1.0x vol/liq. For a $50k liquidity pool you needed $100k daily volume — which only happens at peak pump, not before entry. The log showed `0.2x` being rejected across every pair on every chain. Zero opportunities were found across 6 chains for many cycles.

**Fix:** Configurable minimum `min_vol_liq_ratio` (default 0.05 — ghost pool guard only). Active pairs now score continuously via a blended liq_score = 0.6 × depth_score + 0.4 × turnover_score.

### BUG 2 — Risk-failure rewarded (safety regression, FIXED)
**Location:** `core/engine.py:_calculate_opportunity_score` (risk block)

When `risk_score` was None/failed, the 0.20 weight was dropped from the denominator but not from the score. A token we could not safety-check normalized HIGHER than a token with known moderate risk. Now: None risk_score → hard reject (return 0.0), labeled "worst-case."

### BUG 3 — ML signals fabricated (ML path, FIXED)
**Location:** `core/engine.py:_analyze_opportunity`

The opportunity construction used `ml_confidence=score` (heuristic circular), `pump_probability=score*0.8`, `rug_probability=0.2` (constant — always passed the 0.5 rug gate). `EnsemblePredictor` was loaded but never called. Now wired via `_ml_predict_opportunity` + `predict_decoupled`.

### BUG 4 — Pydantic deprecation (runtime warning, FIXED)
**Location:** `modules/dex_trading/main_dex.py` (lines 483, 491, 562)

`config_model.dict()` is deprecated in Pydantic v2. Added `_model_to_dict()` helper (tries `model_dump()` first, falls back to `dict()`).

### BUG 5 — Auto-retrain loop is a dead loop (ML never improves, FIXED)
**Location:** `core/engine.py:_retrain_models` / `_should_retrain`

`_should_retrain()` always returned `False`. The loop would wait 24h, collect `{}`, check `False`, sleep again — forever. Now invokes `scripts/train_ensemble.py` as a subprocess with reloading of artifacts on success.

### BUG 6 — EVM chains return "No pairs found" (PARTIALLY FIXED)
**Location:** `data/collectors/dexscreener.py:get_new_pairs`

Strategy 1 only scanned top-10 global boosts (Solana-dominated). Strategy 3 searched by quote token name (WETH/USDC) but those pairs are old established pools that fail the 24h age filter. Fixed: Strategy 1 now scans top-30; added Strategy 4 using `/latest/dex/pairs/{chain}` to get chain-specific fresh pairs.

---

## 3. Missing features / data-dependent enhancements

### 3a. EVM chain age filter vs. discovery mismatch
The `_filter_pair` `max_age_hours=24` rejects all pairs older than 24h. Most meaningful EVM momentum trades happen in the 2h–12h window after listing, which is correct. However, Strategy 3's search endpoint returns established pairs, not new ones — so the age filter kills all Strategy 3 hits. Strategy 4 (new in this wave) retrieves chain-specific pairs, but DexScreener's `/latest/dex/pairs/{chain}` may also return old pairs. **Round 2 action**: verify what `/latest/dex/pairs/{chain}` actually returns (does it have `pairCreatedAt`?); may need `createdAfter` param if available.

### 3b. ML ensemble not yet trained
`predict_decoupled` is now wired but will always fall back to `heuristic_fallback` until artifacts exist in `models/`. Run `python scripts/train_ensemble.py --days 90` on the VPS after >= 50 closed trades exist. Check the log for `ML[ensemble]` vs `ML[heuristic_fallback]`.

### 3c. `_load_state` still a no-op
After restart, `active_positions` is empty. The price refresh loop in `position_service.py` compensates for PnL display, but the in-engine exit logic (stop-loss, rug-prob gate) doesn't fire for restored positions. **Round 2 action**: port the Wave-8 `_load_state` DB restore from the main checkout.

---

## 4. Proposed signals (ranked by expected impact)

1. **Fix scoring (this wave)** — zero opportunities found → hundreds of candidates per cycle. Impact: direct revenue unlock.
2. **Train ensemble (operational, requires data)** — replaces heuristic with real ML. When trained, `rug_probability` and `pump_probability` become real signals, enabling the full `TradingOpportunity.score` formula.
3. **Momentum composite signal** — add `price_change_1h * volume_change_24h` as a combined momentum feature. Pairs that pump AND have accelerating volume are the high-edge subset. Currently the scorer treats them independently.
4. **Seller/buyer ratio filter** — `buyers_24h / (buyers_24h + sellers_24h)` < 0.4 → strong bearish sign, hard-reject. Currently unused. Low implementation cost, real signal.
5. **Adaptive min_score based on chain** — ETH mainnet pairs need higher conviction (gas cost) than BSC/Base. Chain-specific `min_opportunity_score` would reduce false positives on expensive chains.

---

## 5. Profitability plan

**Edge source:** Momentum entry on new EVM tokens (< 24h) with genuine volume acceleration, filtered by real risk score, sized by half-Kelly.

**Fee/slippage/MEV cost model:**
- Uniswap V3 fee: 0.05%–1% (fee tier dependent; QuoterV2 returns the actual fee)
- Slippage: capped at `max_slippage_bps=50` (0.5%) per config
- Gas: modeled per-chain via `_CHAIN_MAX_GWEI_DEFAULTS`; real gas in Wave-2
- MEV: Flashbots on Ethereum, bloXroute on BSC (Wave-4)
- Total round-trip cost estimate: ~1%–2% on ETH, ~0.3%–0.5% on BSC/Base

**Expected Sharpe estimate:** Cannot estimate without live data. With the scoring bug fixed, DRY_RUN will accumulate closed trades. Run `train_ensemble.py --dry-run-no-save` after 50+ closed trades for CV AUC.

**Capital allocation:** Half-Kelly (already implemented): `kelly_fraction * 0.5 * opportunity_multiplier [0.8–1.2]`, clamped to `[min_position_usd, max_position_usd]`.

**Kill-switch conditions:**
- Global: `logs/.killswitch` — any process can write, all modules poll.
- Per-module: `logs/.pause_dex`.
- Drawdown: `risk_manager.py` has `max_drawdown_pct` — configure in DB.
- Rug gate: `rug_probability > 0.5` → position rejected before entry.

---

## 6. Cross-module / dashboard handoffs

- **Dashboard agent (Agent 10)**: Surface `metadata.ml_source` ('ensemble' vs 'heuristic_fallback') on the DEX positions page so the operator can see whether trained ML is active. Also surface `ensemble_version` from `config_settings(ml_models, ensemble_version)`.
- **Dashboard agent**: The `min_vol_liq_ratio` config key is now live and tunable. Add it to the Settings page under Trading section (default 0.05, range 0.01–1.0).
- **PM agent (Agent 1)**: `_load_state` is still a no-op in the worktree (no Wave-8 port). Open positions don't get in-engine exit monitoring after restart. Recommend adding to Wave-14 backlog.
- **PM agent**: The `ml_retrain_enabled`, `ml_retrain_interval_hours`, `ml_retrain_days`, `ml_retrain_min_trades` config keys are new. Need migration or DB seed.

---

## 7. DB-QUERY REQUEST BLOCK

Run these queries to understand current trade data quality and tune the ML pipeline:

```bash
# Schema discovery
PG -c "\d trades"
PG -c "\d config_settings"

# How many closed trades exist for ensemble training?
PG -c "SELECT chain, COUNT(*) as n, AVG(profit_loss_percentage) as avg_pnl, SUM(CASE WHEN profit_loss_percentage >= 20 THEN 1 ELSE 0 END) as pump_pos, SUM(CASE WHEN profit_loss_percentage <= -50 OR metadata->>'close_reason' = 'stop_loss_rapid' THEN 1 ELSE 0 END) as rug_pos FROM trades WHERE status='closed' AND metadata IS NOT NULL GROUP BY chain ORDER BY n DESC;"

# Scoring rejection breakdown — what's killing opportunities?
PG -c "SELECT metadata->>'close_reason' as reason, COUNT(*) as n, AVG(profit_loss_percentage) as avg_pnl FROM trades WHERE status='closed' GROUP BY reason ORDER BY n DESC LIMIT 20;"

# Current ML model version (is ensemble trained?)
PG -c "SELECT key, value, updated_at FROM config_settings WHERE config_type='ml_models';"

# Vol/liq ratio distribution on closed trades (validate our 0.05 threshold)
PG -c "SELECT ROUND((CAST(metadata->'pair'->>'volume_24h' AS numeric) / NULLIF(CAST(metadata->'pair'->>'liquidity_usd' AS numeric), 0))::numeric, 2) as vol_liq_ratio, COUNT(*) as n, AVG(profit_loss_percentage) as avg_pnl FROM trades WHERE status='closed' AND metadata->'pair'->>'volume_24h' IS NOT NULL GROUP BY 1 ORDER BY 1 LIMIT 30;"

# Trading config — what thresholds are live?
PG -c "SELECT key, value FROM config_settings WHERE config_type='trading' ORDER BY key;"

# Risk manager config
PG -c "SELECT key, value FROM config_settings WHERE config_type='risk_management' ORDER BY key;"

# Recent win/loss by chain (routing quality)
PG -c "SELECT chain, COUNT(*) as n, SUM(CASE WHEN profit_loss_percentage > 0 THEN 1 ELSE 0 END) as wins, AVG(profit_loss_percentage) as avg_pnl, MAX(profit_loss_percentage) as best, MIN(profit_loss_percentage) as worst FROM trades WHERE status='closed' AND created_at > NOW() - INTERVAL '30 days' GROUP BY chain ORDER BY n DESC;"
```

**How each query informs tuning:**
- Query 1 (`\d trades`): confirms `profit_loss_percentage` column exists and type
- Query 2 (closed count per chain): tells us if we have enough data (>= 50) to run train_ensemble.py; also shows pump/rug label class balance
- Query 3 (close_reason): reveals what our stop-loss/take-profit distribution looks like; if 90% are stop_losses the ensemble's rug label will have high signal
- Query 4 (ml_models): confirms whether ensemble_version exists (= trained) or not
- Query 5 (vol/liq ratio): validates whether our 0.05 ghost-pool threshold is appropriate — if winning trades cluster at > 0.1x it should stay; if they cluster at 0.05–0.1x we need to lower further
- Query 6 (trading config): confirms `min_opportunity_score` and `min_vol_liq_ratio` values live in DB
- Query 7 (risk config): confirms `max_position_size_usd`, `max_drawdown_pct` values
- Query 8 (win/loss by chain): identifies which chains are profitable so we can focus enabled_chains

---

*Report produced by Wave-13 Agent 2 (quant/ML/DEX). All code changes in `core/engine.py`, `ml/`, `data/collectors/dexscreener.py`, `modules/dex_trading/main_dex.py`, `scripts/train_ensemble.py`.*
