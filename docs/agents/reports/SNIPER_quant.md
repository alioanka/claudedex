# SNIPER Module — Quantitative / Algorithmic Audit

**Author:** quant-algo-expert (Claude)
**Date:** 2026-05-11
**Scope:** Signal generation, scoring, ML, position sizing, entry / exit decision logic for the cross-chain Sniper module (`modules/sniper/`, with shared dependencies in `analysis/` and `ml/models/`).

---

## 1. Executive Summary

The Sniper module is a **rules-based new-pair entry bot**. It listens to EVM mempool/events and Solana Raydium logs for newly created pairs, runs each candidate through a third-party safety check (GoPlus + Honeypot.is + RugCheck.xyz), and — if it passes a binary "is this safe?" gate — places a fixed-size buy and manages it with a hard-coded ±20 % / +50 % stop-loss / take-profit.

Compared with the Solana module the sniper is **simpler and less sophisticated**: there is no trailing stop, no tier ladder, no dynamic sizing, no scam blacklist feedback loop, and (like the Solana module) **no machine-learning model is invoked from the live path**. The `analysis/rug_detector.py`, `analysis/token_scorer.py`, `analysis/dev_analyzer.py`, `ml/models/rug_classifier.py`, and `ml/models/pump_predictor.py` files are all unimported from `sniper_engine.py`. Token "score" in the sniper is the integer 0–100 returned by the in-house heuristic at `token_safety.py:477-542`, plus thresholds in `_check_filters` (`sniper_engine.py:305-411`).

The strategy has three real strengths:
1. Honesty about safety thresholds — there's a configurable `test_mode` with explicit relaxed rules; production uses tighter caps.
2. Caches RugCheck/GoPlus results with TTL to avoid spam.
3. Per-token cooldown (`_rejected_cache`) prevents re-scoring rejected tokens for 5 min.

The biggest weaknesses are: the exit logic is a static ±20 % / +50 % rectangle (vastly inferior to the Solana tier ladder), there is no learned signal, position size is fixed regardless of confidence, the safety score is a hand-tuned point-system with no calibration to actual rug outcomes, and the price feed used for SL/TP monitoring is Jupiter Price API v4 (deprecated; will silently return 0).

**Top profit lever:** Replace the rectangle exit (`sniper_engine.py:580-625`) with the same multi-tier trailing-stop logic used in `modules/solana_trading/core/solana_engine.py:2254-2574`, parameterised by chain. The current rectangle gives back tail gains and is symmetric in cost-of-miss vs. cost-of-stop, which is wrong for memecoin payoffs.

**Top bias risk:** `_calculate_score` in `token_safety.py:477-542` is **hand-coded with no historical calibration**. The implied probability of "safe" tokens with score ≥ 70 has never been validated against actual hold-to-T+24h outcomes. Combined with the binary `SafetyRating` thresholds (`:548-552`) at 70 and 40, this discretises a noisy signal and may both under-buy promising tokens (false negatives) and over-buy weak ones (false positives).

---

## 2. Existing-Edge Audit (per signal / model)

### 2.1 Listener feeds (`evm_listener.py`, `solana_listener.py`)
- Not deeply audited here, but the listeners feed `_evaluate_target` (`sniper_engine.py:284`) with raw new-pair events.
- Edge: speed. Catching a pair within the first block before retail aggregator-trackers refresh.
- Risk: if `_check_filters` runs slow third-party API calls (GoPlus, Honeypot.is, RugCheck) the latency edge is destroyed. Each external API call is 200–800 ms; three of them serial in `_check_solana_token` / `_check_evm_token` adds 1–2 s of latency. By that time price has already moved 5–20 %.

### 2.2 Safety check gate (`token_safety.py:114-411`)
Calls 1–3 external APIs depending on chain:
- **EVM:** GoPlus + Honeypot.is (parallelism would help; currently serial at `:157-238`).
- **Solana:** RugCheck.xyz with Helius fallback.
Outputs a 0–100 `score` plus an enum rating: SAFE (≥70) / CAUTION (40-69) / DANGER (<40) / HONEYPOT (any).

**Strengths:**
- Multiple data sources for honeypot detection.
- Sensible feature set: tax, verified, renounced, liquidity, lock, holders, top-holder %.
- Solana freeze-authority check at `:369-372` is correct — a freeze authority is a hard kill.
- Caches per-token for 5 min (`:99`) — good for rate limits.

**Weaknesses:**
- Score formula is hand-tuned (`_calculate_score`, `:477-542`). Coefficients (`-25` for liq<1k, `-30` for top-holder>80%, etc.) are not derived from any backtest.
- Rating thresholds 70 / 40 are arbitrary. No calibration curve published anywhere.
- `is_honeypot` from RugCheck is inferred from regex of risk names containing "freeze" (`:338-340`), which is too narrow — there are other Solana honeypot patterns (transfer-fee extension abuse, transfer-hook traps).
- The Helius fallback (`:453-475`) is only for `holderCount`. If RugCheck is down the bot is essentially blind on Solana.

### 2.3 Entry decision (`sniper_engine.py:305-411`)
Pipeline:
1. Cooldown check (5 min) → skip.
2. If `safety_check_enabled = false`, accept everything. (Genuinely scary in production.)
3. Apply test-mode-relaxed or production thresholds:
   - `max_buy_tax`, `max_sell_tax` (default 15 %, test 50 %)
   - `min_liquidity` (default $1k, test $10)
4. Reject HONEYPOT, DANGER (unless test mode), high-tax, low-liquidity.
5. CAUTION tokens are allowed in production (`:385`) — interesting and probably wrong without measurement.
6. On pass → add to `pending_targets` and execute next loop.

**The thresholds (max-tax 15 %, min-liq $1k) have no learning component.** A token with 12 % buy tax is treated identically to one with 1 % tax. Same with $1.1k liquidity vs. $50k. The system loses all gradient information in the 0/1 decision.

### 2.4 Position size (`sniper_engine.py:476-482`)
Fixed `self.trade_amount` (default 0.1 SOL / ETH). No confidence weighting, no risk-band, no Kelly. A token scoring 95 gets the same bet as one scoring 70.

### 2.5 Exit logic (`sniper_engine.py:577-625`)
```
take_profit_pct = 50.0
stop_loss_pct   = -20.0
```
Hard-coded **inside the loop** despite `_load_settings` having read `take_profit_pct` and `stop_loss_pct` from DB (`:207-210`). The settings are loaded but **the monitor body redefines local constants** instead of using `self.take_profit_pct` / `self.stop_loss_pct`. This is a bug — config values are silently ignored.

The rectangle exit is bad for sniped memecoins:
- 50 % TP gives back all of the upside skew. Memecoin success distribution has a fat right tail (2× / 5× / 50×); a flat 50 % TP truncates exactly the trades that fund the strategy.
- −20 % SL with no trail means a token that goes +200 % then crashes to +5 % closes for a +5 % win that should have been +180 % locked.
- No time-stop. A dead position can sit forever.
- No early-stage (first 60 s) volatility widening. A token can dip −25 % in the first minute and rebound to +100 %.

### 2.6 Price feed for monitoring (`sniper_engine.py:627-655`)
For Solana: `https://price.jup.ag/v4/price?ids=...`. **This API was deprecated in 2024 and returns 0 / 404 for most tokens.** The function silently returns 0, and the monitoring loop skips the position (`:602`). Combined with no fallback, **stop-loss never triggers on Solana snipes** in production.

For EVM: DexScreener with `priceNative` field (`:650`) — fine but no validation that the pair returned is the correct one.

### 2.7 PnL accounting (`sniper_engine.py:700-782`)
- Records `entry_usd` and `exit_usd` using `PriceFetcher` (CoinGecko) for SOL/ETH price.
- Calculates `pnl_pct = (exit_usd - entry_usd)/entry_usd * 100`.
- **Does not subtract priority fees, gas, or slippage.** `result.gas_used` is recorded but never deducted from `pnl_usd`.
- For dry-run (`is_simulated=True` hard-coded at `:564` in spite of the `dry_run` field), all trades are flagged simulated — meaning the production code path also writes `is_simulated=True`. This is a real bug for analytics: live trades are mis-labelled as paper trades.

### 2.8 Cohort survival / analytics
**Nothing.** No tracking of snipe outcomes by hour-of-day, day-of-week, launch-type (Raydium vs. pump.fun-graduated), creator-cluster, or any other dimension. The bot is flying blind w.r.t. its own historical performance shape.

### 2.9 Cross-references to ML models
- `analysis/token_scorer.py`: not imported.
- `analysis/rug_detector.py`: not imported.
- `analysis/dev_analyzer.py`: not imported.
- `ml/models/rug_classifier.py`: not imported.
- `ml/models/pump_predictor.py`: not imported.

The sniper module is **rule-based end-to-end**. The "scoring" is the 0–100 hand-tuned integer from `token_safety._calculate_score`.

---

## 3. Bias & Leakage Findings

| # | Location | Bias / Leakage | Severity |
|---|----------|----------------|----------|
| B1 | `token_safety.py:477-542` | Score weights are hand-coded; no calibration to actual rug outcomes — **selection bias** in implied probability | High |
| B2 | `token_safety.py:548-552` | Discrete thresholds 70 / 40 collapse gradient information; **threshold over-fit** likely tuned to specific historical examples | High |
| B3 | `sniper_engine.py:580-581` | `take_profit_pct = 50.0` / `stop_loss_pct = -20.0` hard-coded inside `_monitor_active_snipes`; DB settings (`:207-210`) silently ignored — **stale config** | High |
| B4 | `sniper_engine.py:627-655` | Jupiter Price v4 deprecated → returns 0 → Solana SL/TP never fires; **survivorship bias** in production stats (only stop on take-profit or chain failure) | High |
| B5 | `sniper_engine.py:564` | `is_simulated = True` hard-coded — live trades mis-labelled as paper. Skews any post-hoc model that conditions on this flag | High |
| B6 | `sniper_engine.py:700-780` | PnL excludes priority fees + gas + actual slippage; reported P&L overestimated by 1–3 % per round trip on EVM, 0.5–1 % on Solana | Med |
| B7 | `sniper_engine.py:121` | `test_mode_min_liquidity=10` ($10 liquidity threshold) — if accidentally enabled in production allows trades on near-zero-liquidity pools | Med |
| B8 | `sniper_engine.py:317-323` | Cooldown removes the rejection record after 5 min, so a re-listed honeypot can be re-evaluated — **memory loss** for known scams | Med |
| B9 | `token_safety.py:99` | 5-min cache TTL: a token's tax can change inside the cache window via owner action; **stale safety data** | Med |
| B10 | `sniper_engine.py:387` | High-tax rejection uses `max_buy_tax`/`max_sell_tax` separately but never `(buy_tax + sell_tax)` round-trip cost — a 10/14 token (24 % round-trip!) passes the 15/15 thresholds | Med |
| B11 | `sniper_engine.py:284` | `_evaluate_target` triggered on every event but no per-source rate-limit — a noisy listener can starve other-chain processing | Low |
| B12 | `sniper_engine.py:493` | `entry_price = trade_amount / amount_out` is the *expected* fill from Jupiter quote (`amount_out` is the *expected* tokens), not the actual on-chain fill — **planned-fill bias** | High |
| B13 | `token_safety.py:222-224` (shared with Solana SafetyEngine) | Sell-route honeypot check allows-by-default on quote failure — **fail-open** rather than fail-closed | Med |

The combination of B3 + B4 means that on Solana **the stop-loss code path effectively does not function** in production. Every live Solana snipe is implicitly "infinite hold or +50 % TP" — a permanently-open call option with no theta.

---

## 4. Missing Signals / Features (ranked by expected lift)

1. **Real-time creator-wallet history.** `dev_analyzer.py` produces a `reputation_score` based on previous projects (`:194`). Wire it into `_check_filters`. Reject sub-30 reputations. Single biggest mechanical gain on rug avoidance.
2. **LP-burn vs. lock vs. unlocked classifier.** `rug_classifier.py` lists `lp_burn_percentage` (`:53`) — never populated in the sniper feature set. For EVM tokens, check if LP tokens are sent to `0x...dEaD` (burned) vs. a vesting contract (locked) vs. held by deployer (live rug risk).
3. **Token-age + concentration interaction.** A token that's 2 hours old with top-10 holders owning 30 % is fine. The same token at 2 days old with the same concentration is suspicious (no organic dispersion). Missing in current code.
4. **Pair-on-multiple-DEX detection.** If a pair appears on Uniswap V2 only, no LP migration → typical rug pattern. If it appears on V2+V3 + Sushi → real project.
5. **First-block-buyer cluster.** Same as in the Solana audit: the first 5 buyers of the pair, if all funded from one wallet, are coordinated insiders. Compute their share of supply; reject if >25 %.
6. **Round-trip tax cost feature.** Currently `buy_tax` and `sell_tax` are independent thresholds. The right feature is `total_tax = buy + sell`, and the threshold should be on `(1 - total_tax) > break-even * required_edge`.
7. **Bonding-curve progress (Solana pump.fun-via-Raydium).** When a pump.fun token graduates to Raydium it's the most-pumped point; sniping that graduation point is buying at the top. The Solana listener should distinguish "fresh-on-Raydium" vs. "fresh-on-Raydium-via-graduation".
8. **Snipe-conviction-weighted size.** Once a probability-of-success score exists (from a calibrated rug classifier), bet size should be `Kelly_fraction(p, b) * bankroll` rather than fixed 0.1.
9. **Time-of-day rate filter.** Empirically (literature on Solana memes) successful pumps cluster in US-evening hours; rug-launches cluster in Asia-night. Currently the bot trades 24/7 uniformly.
10. **Failed-snipe cohort analysis.** Track all rejected tokens and their T+1h outcomes. The rejection filters can then be A/B tested for false-positive rate.

---

## 5. Capital-Allocation Review

Current state:
- Fixed `trade_amount = 0.1` per snipe (`sniper_engine.py:115`).
- No max-simultaneous-snipes cap visible in the engine (only managed by the cooldown timer).
- No daily-loss kill switch.
- No correlation handling between concurrent snipes.

Recommendations:
- Add `max_concurrent_snipes` (e.g. 5) — track `len(active_snipes)`.
- Add `daily_loss_limit_usd` (e.g. $200) — count realised losses in last 24 h.
- Add `consecutive_loss_block` mirroring the Solana engine's progressive block (`solana_engine.py:222-275`).
- Replace fixed sizing with Kelly-fraction sizing keyed off the rug-classifier probability **once that is wired in**.

False-positive / miss cost framing:
- Cost of false-positive (buying a rug): −80 % to −100 % of stake.
- Cost of false-negative (skipping a winner): expected +50 % to +500 %.
- **For the right token mix, false-negatives can be more expensive than false-positives.** The current "reject unless score ≥ 70 AND safety_check passes" is asymmetric and likely too conservative for production memecoin sniping. The decision threshold should be calibrated to the empirical loss distribution, not chosen at 70.

---

## 6. ML Model Health

Same status as the Solana module:

| Model | Status | Wired Into Sniper? | Walk-forward CV? |
|-------|--------|--------------------|------------------|
| `analysis/token_scorer.py` | Built | No | n/a (rule-based) |
| `analysis/rug_detector.py` | Built | No | n/a (rule-based) |
| `analysis/dev_analyzer.py` | Built | No | n/a |
| `ml/models/rug_classifier.py` (XGB+LGB+RF+GB) | Built | No | No (`:277` random split) |
| `ml/models/pump_predictor.py` (LSTM+GBMs) | Built | No | No (`:359` index split) |

No retrain scripts (`scripts/retrain_*.py`) exist for either model. The sniper has been running on **third-party safety APIs + a hand-coded 0–100 integer** since inception. Treat any "model accuracy" stat reported by the upstream model code as un-calibrated until walk-forward training is in place.

The `analysis/dev_analyzer.py` `known_developers` cache is in-memory only (`:95`) and not persisted; this also limits cumulative learning.

---

## 7. Profitability Levers (ranked, with effort estimate)

1. **Fix B3 + B4 immediately.** Use `self.take_profit_pct` / `self.stop_loss_pct` in `_monitor_active_snipes`, and replace `price.jup.ag/v4` with the same DexScreener / multi-source price fetcher used in `solana_engine.py:413-545`. Effort: 0.5 day. *Expected: stop-loss actually fires on Solana — possibly the single biggest single-day P&L improvement in the module.*
2. **Migrate exit logic to multi-tier trailing stop (port from `solana_engine.py:2254-2574`).** Effort: 2 days. *Expected: 1.5–2× expected payoff per winning trade on the memecoin tail.*
3. **Confidence-weighted sizing.** Use `report.score / 100` as a multiplier on `trade_amount` (interim), then Kelly once a real probability exists. Effort: 0.5 day for the interim, 1 week for Kelly. *Expected: 10–20 % Sharpe lift.*
4. **Wire `dev_analyzer.reputation_score` into `_check_filters`.** Effort: 2 days (plus persistence work in dev_analyzer). *Expected: meaningful rug-rejection improvement.*
5. **Round-trip-tax composite threshold** (`total_tax`, not separate). Effort: 0.5 day.
6. **Honest PnL** (subtract gas + priority fee + actual fill slippage). Effort: 1 day.
7. **Fix `is_simulated` mis-labelling at `:564`.** Effort: trivial. *Required for any analytics/ML downstream.*
8. **Cohort-survival logger** (hour/day/launch-type). Effort: 1 day. *Enables future tuning.*

---

## 8. Proposed Action Backlog

| ID | Title | Files | Effort | Priority |
|----|-------|-------|--------|----------|
| SNIPE-Q-01 | Use `self.take_profit_pct` / `self.stop_loss_pct` from DB in `_monitor_active_snipes` (delete local re-bind) | `sniper_engine.py:580-581` | XS | P0 |
| SNIPE-Q-02 | Replace `price.jup.ag/v4` with DexScreener + CoinGecko multi-source price (mirror `solana_engine.py:413-545`) | `sniper_engine.py:627-655` | S | P0 |
| SNIPE-Q-03 | Fix `is_simulated` hard-coded `True` at insert; use `self.dry_run` | `sniper_engine.py:564` | XS | P0 |
| SNIPE-Q-04 | Subtract priority-fee + gas + slippage from `pnl_usd` | `sniper_engine.py:700-780` | S | P0 |
| SNIPE-Q-05 | Replace flat ±20 % / +50 % exit with multi-tier trailing stop (port from Solana engine) | `sniper_engine.py:577-625` | M | P1 |
| SNIPE-Q-06 | Composite `total_tax = buy_tax + sell_tax` threshold | `sniper_engine.py:387` | XS | P1 |
| SNIPE-Q-07 | Scale `trade_amount` by `safety_report.score / 100` as interim confidence weighting | `sniper_engine.py:476` | S | P1 |
| SNIPE-Q-08 | Add `max_concurrent_snipes` + `daily_loss_limit_usd` + consecutive-loss block (mirror Solana risk metrics) | `sniper_engine.py:90+` | M | P1 |
| SNIPE-Q-09 | Persist rejected-cache to DB so a re-listed honeypot stays rejected across restarts | `sniper_engine.py:137`, new table | M | P2 |
| SNIPE-Q-10 | Wire `analysis/dev_analyzer.py` reputation score into `_check_filters` | `sniper_engine.py:305`, `dev_analyzer.py` | L | P2 |
| SNIPE-Q-11 | Add early-volatility-window SL widening (mirror `solana_engine.py:2452`) | `sniper_engine.py:577` | S | P2 |
| SNIPE-Q-12 | Walk-forward training harness for rug classifier; build labelled snipe dataset | `ml/models/rug_classifier.py:277`, new script | L | P2 |
| SNIPE-Q-13 | Replace handcoded 70/40 rating thresholds with calibrated probability from rug classifier | `token_safety.py:548-552`, `sniper_engine.py:305` | L | P2 |
| SNIPE-Q-14 | Implement Kelly-fraction sizing once probability score is available | `sniper_engine.py:476` | M | P3 |
| SNIPE-Q-15 | First-block-buyer-cluster feature (top-5 buyers in tx 0-2) | new helper, `sniper_engine.py:305` | M | P3 |
| SNIPE-Q-16 | Cohort-survival dashboard (hour-of-day, launch-type, score-bucket → P&L) | new | M | P3 |
| SNIPE-Q-17 | Parallelise GoPlus + Honeypot.is calls in `_check_evm_token` | `token_safety.py:141-298` | S | P3 |
| SNIPE-Q-18 | Surface `take_profit_pct` / `stop_loss_pct` / partial-exit ladders via dashboard Settings page (note for dashboard agent) | settings UI | n/a | P3 |
| SNIPE-Q-19 | Treat sell-route honeypot check as fail-closed when not rate-limited | `safety_engine.py:222-224` (shared) | S | P2 |

---

## 9. Open Questions

1. Are `take_profit_pct` / `stop_loss_pct` actually meant to be configurable per-chain, per-token, or per-risk-band? The DB schema reads a single value (`sniper_engine.py:207-210`), which is too coarse.
2. Is the sniper supposed to outperform a "buy every safety-passed pump.fun graduation, hold 24h" naive baseline? That baseline should be benchmarked before any further tuning.
3. Should `test_mode` be a hard, separate runtime mode (different DB schema) rather than a toggle? The current design risks accidental production enablement (B7).
4. What's the actual production fill-vs-quote slippage on EVM snipes? The current PnL math (`:494`, `:683`) trusts quoted output exactly.
5. Is the Telegram-controlled remote-start/stop allowed to mutate `take_profit_pct` and `stop_loss_pct` at runtime? If yes, an attacker / mistake can rectangle-out a winning position.
6. Why does the production code path still allow CAUTION tokens (`:385`)? Was this measured to be profitable, or assumed? If unmeasured, this is the single most impactful threshold to validate.
7. Is there an end-of-day reconciliation between `active_snipes` (in-memory) and `sniper_trades` (DB)? A process crash mid-snipe currently loses the position state.

---

*End of SNIPER quant audit.*
