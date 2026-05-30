# PM Final Campaign Report — Wave 2

**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `ffeda0a` (campaign brief)
**Head commit:** `d2f555f` (T1 catalog campaign report)
**Window:** 2026-05-19
**Commits this wave:** 56
**Net diff:** 52 files changed, +7457 / -219
**Agents:** A1 (DEX) · A2 (ARB) · A3 (SOLANA) · A4 (SNIPER) · A5 (FUTURES) · A6 (AI) · A7 (COPY_TRADING) · T1 + T2 (Test Runner)
**Status:** All 9 agents reported. All 7 modules ship Wave-2. Operator action items below.

---

## 1. Executive summary (one page)

Wave 2 was the re-audit + enhancement sweep. Every module owner re-read their engine,
found the residuals Phase 1 had missed, shipped fixes in ≤200-LoC commits, then layered
on at least one operator-named enhancement. T1/T2 closed the loop with 38 new Test
Runner buttons so the operator can validate each fix from `/test_runner` without
pulling logs.

**What shipped (highlights):**
- DEX: 3 P0 crash bugs killed, web3 v6 API drift purged, per-chain gas caps + EIP-1559 hooks, route-quality scoring (`amount_out × (1−impact) − gas_cost_native`).
- ARBITRAGE: spatial-arb live-trade silent-drop NameError fixed (the engine literally could not execute *any* live arb until this commit), chain-aware PnL costs replace `$15 + 30%-of-spread` magic, hourly gas-budget tracker, dashboard `min_profit_spread` knob now honored.
- SOLANA: four residual MB-06 decimals hardcodes killed on close paths (was 10× oversell or 1000× undersell on BONK and modern launches), MB-15 Drift fail-closed guard suite, adaptive priority-fee + Jupiter quote TTL, opt-in ML rug-classifier gate.
- SNIPER: `processed`→`confirmed` two-stage readback cuts WSS commitment wait ~5s → ~300ms, safety-check exception cooldown, dual-source honeypot quorum (GoPlus + Honeypot.is), Birdeye SL/TP fallback, per-chain listener-health widget.
- FUTURES: discovered the `b1b8df9` leverage-cap fix only patched the dashboard wrapper — the subprocess entry was still defaulting `max_leverage=3` regardless of operator settings. Fixed end-to-end. Added funding-rate directional gate, ATR per-symbol sizing (opt-in), post-fill ISOLATED-margin verifier.
- AI: multi-provider quorum gate, confidence-calibration table + `/api/ai/calibration` reliability diagram, ε-greedy / UCB1 prompt-template bandit persisted to `ai_feature_store`.
- COPY_TRADING: built `wallet_discovery.py` (5-source rate-limited sweep — DexScreener, Birdeye, GMGN, Helius, manual) and `leader_scorer.py` (30/90-day Sharpe + hit-rate + drawdown composite); replaced constant `copy_ratio` with per-leader Kelly-fraction sizing; replay diagnostics log every gate decision; new `/copytrading/leaders` page.

**What's now safer:** every module gained at least one defense-in-depth or fail-closed guard. The ARB and DEX modules in particular went from "would crash on first profitable opportunity" to "can be live-flipped after a chain-by-chain smoke."

**What's blocked:** triangular arbitrage (waiting on atomic-receiver contract deploy — operator decision), Solana pump-predictor wiring (needs per-token rolling price buffer), the `pump_predictor.py` scaler-leakage fix (P1-08 re-classified; out of A3 scope), and the `_quote_v3` placeholder (V3 routing is fundamentally broken until Uniswap V3 QuoterV2 binding lands). None of these block a Wave-2 flip; all are wave-3 backlog.

---

## 2. P0 bugs caught and fixed (operator front-and-center)

These were silently breaking trades in production. Every one is now closed:

### DEX
- **`self.max_slippage` never initialized** on `DirectDEXExecutor` (`direct_dex.py:568`). Any swap where `order.slippage` was None/0 raised `AttributeError`. Silently bricked every order that didn't carry an explicit slippage. → `f7d7941`.
- **MEV `bundle_id` UnboundLocalError** (`mev_protection.py:232`). `bundle_id` was only assigned inside the Flashbots branch; the low-risk `ADVANCED` else-branch crashed every call. → `e872121`.
- **Flashbots on L2s** — `mev_protection: True` config sent BSC/Polygon/Arbitrum txs to the Flashbots relay (meaningless). Now per-chain gated; silent downgrade to private mempool on non-Ethereum. → `e872121`.
- **`ether_to_wei(order.amount)` mis-application** on the swap *input* leg (`direct_dex.py:594,601`). USDC/USDT (6-dec) overstated by 10¹², WBTC (8-dec) by 10¹⁰. Companion bug to MB-01 but on the input side. → `23d860d`.

### ARBITRAGE
- **NameError silently dropping every live opportunity** (`arbitrage_engine.py:1479`). The spatial-arb log line referenced `forward_output` / `final_output` — identifiers renamed in a prior refactor. Inside the `try/except` swallow. Net effect: with `DRY_RUN=false` the engine could not execute *any* spatial arb. → `9e6a7d1`.

### SOLANA
- **Four residual MB-06 decimals hardcodes** on close paths (`solana_engine.py:2197, 2217, 3339, 3761`) plus `int(actual_balance * (10**6))` at line 2229. For non-6-decimal tokens (BONK, modern memecoin launches) this caused 10× oversell or 1000× undersell. → `09a5c85`.

### FUTURES
- **Leverage-cap fix only patched the dashboard wrapper** (`futures_module.py:177-197`), not `main_futures.py:607-626` (the subprocess entry). The subprocess silently fell back to dataclass defaults (`max_leverage=3, max_positions=3, max_total_exposure=500`) regardless of operator settings. Symptom on production VPS: "Leverage 10x exceeds max 3x" with `futures_max_leverage=20` on the settings page. → `09a5c85` (FUT-RM-01) + `34399db` (startup assertion + DRY_RUN smoke).

---

## 3. Per-module fix + enhancement matrix

| Module | Re-verified MBs | Residual P0/P1 closed (commits) | Enhancements shipped | CLAUDE.md | New tests |
|---|---|---|---|---|---|
| DEX | MB-01 (partial), MB-02, P1-04 | `f7d7941`, `e872121`, `23d860d`, `48d5f20`, `162f711` | per-chain MEV gate, route-quality scoring, async `eth_gasPrice`, gas-randomization clamp | yes (`1367ed5`) | `tests/unit/test_dex_decimals.py` (`869eed3`) |
| ARBITRAGE | MB-03, MB-04, MB-05 (still gated), P1-04, P1-06 | `9e6a7d1`, `744ee48`, `4adcd29`, `8cf0143`, `89175d4` | per-chain `CHAIN_COST_PROFILE`, live gas-USD estimator, hourly gas-budget gate, adaptive `min_profit_bps`, DB-backed flash-loan receiver address | yes (`014384d`) | none net-new (engine smoke covered by api/db probes) |
| SOLANA | MB-06..MB-10, MB-15 | `09a5c85`, `661cee6`, `83df4ad`, `b1b358f` | Drift fail-closed guards (leverage/funding/oracle/Pyth confidence), adaptive priority-fee, Jupiter quote TTL, opt-in ML rug gate (`solana_ml_enabled`) | yes (`7e7cff5`) | none (gated by config flag) |
| SNIPER | MB-11..MB-14, Phase-2 WSS | `87c5523`, `77b22e7`, `6612be2`, `4adcd29`, `adee9c2`, `5a0a3e9` | dual-source honeypot quorum, Birdeye SL/TP fallback, per-chain listener-health widget | yes | 5 quorum unit tests in `tests/unit/test_sniper_new_paths.py` |
| FUTURES | MB-16, MB-17, MB-17b, MB-18, reconcile | `09a5c85` (FUT-RM-01), `34399db` (FUT-RM-02/03), `9aba5e8` (FUT-RM-05), `bcc7b91`+`7e9a47b`+`ad401c4` (FUT-RM-06), `ad401c4` (FUT-RM-07) | startup-assertion guard, funding-rate gate, ATR per-symbol sizing, post-fill ISOLATED verifier | yes (`2c907ca`) | `tests/unit/test_futures_risk_wiring.py` — 17 pin + 3 DRY_RUN smoke |
| AI | MB-19, MB-20, MB-21 | `8fe3671` (AI-CAMP-01), `6c3c058` (E1), `9e9f0db`+`c285074` (E2), `b26c0eb`+`db452d9` (E3) | multi-provider quorum, calibration table + reliability endpoint, prompt-bandit (ε-greedy + UCB1) | yes (`b427f67`) | none net-new (calibration write is best-effort) |
| COPY_TRADING | MB-22..MB-25 | n/a (Wave-2 was net-new rebuild) | `wallet_discovery.py` (`e833abb`), `leader_scorer.py` (`33ae524`), per-leader Kelly sizing (`2e59550`), replay diagnostics (`376f9be`), `/copytrading/leaders` page, retrain entrypoint (`91f838d`) | yes (`91f838d`) | none net-new (covered by 6 T2 probes) |

---

## 4. Migration ledger

Two migrations shipped this wave. Both numbered after Phase-1's `022_reset_per_module_dry_run.sql`.

| File | Origin commit | Author | Purpose | Status |
|---|---|---|---|---|
| `migrations/023_add_ai_confidence_calibration.sql` | `9e9f0db` | A6 (AI) | New `ai_confidence_calibration` table — per-trade `predicted_score`, `predicted_confidence`, `realized_pnl_pct`, `realized_won`, `quorum_required`, `closed_at`. Feeds `/api/ai/calibration`. | Applied at boot |
| `migrations/024_copy_leader_scores.sql` | `d129902` (renumbered) | A7 (COPY_TRADING) | New `copy_leader_scores` table — `(chain, wallet_address, source)` identity, 30/90d Sharpe + hit-rate + drawdown, composite `score`, `kelly_fraction`, JSONB `raw_metrics`. Index `(chain, score DESC)` for top-N reads. | Applied at boot |

**Collision note:** d129902 was authored as "migration 023" (commit message still reads "migration 023: copy_leader_scores table") at the same time 9e9f0db landed `023_add_ai_confidence_calibration.sql`. The collision was resolved by renumbering the copy migration to **024**. AI calibration kept the 023 slot because it landed first on disk; copy migration kept the commit body wording but the file on `HEAD` is `024_copy_leader_scores.sql`. No other migration numbers shifted; pre-Wave-2 migrations 009-022 are untouched.

---

## 5. New HTTP endpoints (this wave)

| Method | Route | Origin commit | T1/T2 probe |
|---|---|---|---|
| GET | `/api/ai/calibration` | `c285074` (A6) | `api_ai_calibration` (T2) |
| GET | `/copytrading/leaders` (page) | A7 | covered by `api_copytrading_leaders_list` |
| GET | `/api/copytrading/leaders` | A7 | `api_copytrading_leaders_list` (T2) |
| POST | `/api/copytrading/leaders/refresh` | A7 | `api_copytrading_leaders_refresh` (GET-against-POST → 405/403 = healthy) + paired `db_copy_leader_scores_post_refresh` (T2) |

Every new route has at least one Test Runner probe. The POST refresh route also has a paired DB probe to verify writes landed.

---

## 6. New DB tables / columns (this wave)

| Table | Migration | Probes |
|---|---|---|
| `ai_confidence_calibration` (new) | 023 | `db_ai_calibration_table` (schema), `db_ai_calibration_sample` (90-day roll-up + miscalibration check) |
| `copy_leader_scores` (new) | 024 | `db_copy_leader_scores_table` (schema), `db_copy_leader_scores_top` (top-10 reader), `db_copy_leader_scores_by_source` (source-distribution / rate-limit smell test), `db_copy_leader_scores_post_refresh` (fresh_5m count after manual refresh) |

No column-only changes shipped this wave. Both new tables have schema and content probes.

---

## 7. Test Runner growth (66 → 104 in PM brief; actual 68 → 106 by `^\s+"id":` count)

T1 added 22 entries across DEX/ARB/SOLANA/SNIPER. T2 added 16 across FUTURES/AI/COPY_TRADING. Total = **38 new** (one of the SNIPER block landed in T2 commit `913c720` due to a workspace race; attribution is in `TEST_RUNNER_T1_CAMPAIGN.md`).

### T1 (22 entries)

**DEX (6)** — `script_dex_decimals_unit_tests`, `script_dex_web3_v6_imports`, `script_dex_mev_unbound_check`, `db_dex_recent_trades_24h`, `db_dex_settings_keys`, `db_dex_open_positions`.

**ARBITRAGE (5)** — `db_arb_cost_profile_keys`, `db_arb_flash_loan_receiver_secrets`, `db_arb_recent_pnl_costs`, `api_arb_settings_get`, `script_arb_nameerror_regression`.

**SOLANA (5)** — `db_solana_adaptive_priority_fee`, `db_solana_drift_guards`, `db_solana_ml_rug_gate`, `db_solana_recent_trades_decimals`, `db_solana_position_size_caps`.

**SNIPER (6)** — `db_sniper_processed_hit_ratio`, `db_sniper_safety_check_errors`, `db_sniper_wss_carry_over`, `db_sniper_quorum_outcomes_30m`, `api_sniper_timing_per_chain`, `script_sniper_listener_widget_present`.

### T2 (16 entries)

**FUTURES (5)** — `db_futures_leverage_caps`, `db_futures_funding_gate`, `db_futures_atr_sizing`, `db_futures_isolated_enforce`, `api_settings_futures_post_wave2`.

**AI (5)** — `db_ai_calibration_table`, `db_ai_calibration_sample`, `db_ai_quorum_bandit_config`, `api_ai_calibration`, `db_ai_bandit_state`.

**COPY_TRADING (6)** — `db_copy_leader_scores_table`, `db_copy_leader_scores_top`, `db_copy_leader_scores_by_source`, `api_copytrading_leaders_list`, `api_copytrading_leaders_refresh`, `db_copy_leader_scores_post_refresh`.

Four new bash scripts under `scripts/`:
- `scripts/dex_web3_v6_smoke.sh`
- `scripts/dex_mev_unbound_check.sh`
- `scripts/arb_nameerror_check.sh`
- `scripts/sniper_listener_widget_check.sh`

All are POSIX bash with `set -euo pipefail`, honor `CLAUDEDEX_REPO_ROOT` (default `/app` for Docker), exit non-zero on drift, complete in <100 ms. Every catalog row is read-only — no mutations to live state, no DRY_RUN flag flips.

---

## 8. Live-readiness verdict per module

Phase 1 left every module flagged "AMBER → GREEN candidate." Wave 2 status:

| Module | Pre-wave | Post-wave | Operator next step |
|---|---|---|---|
| DEX | AMBER → GREEN candidate | **GREEN candidate** (clean to flip after chain-by-chain smoke) | Run T1 `db_dex_settings_keys` + `db_dex_recent_trades_24h` probes; mainnet canary one DEX at a time |
| ARBITRAGE | AMBER → GREEN candidate | **GREEN candidate (spatial only)** — was effectively broken pre-wave (A2-01) | Spatial: run `script_arb_nameerror_regression` + `db_arb_cost_profile_keys`; chain-by-chain canary. Triangular: stays gated — needs atomic-receiver contract deploy (operator decision) |
| SOLANA | AMBER → GREEN candidate | **GREEN candidate** | Verify `db_solana_recent_trades_decimals` after first live trade; keep `drift_enabled=false` and `solana_ml_enabled=false` until canary |
| SNIPER | AMBER → GREEN candidate | **AMBER (operator step pending)** — operator must flip `safety_check_enabled=true` before LIVE. Code is ready | Flip the flag in DB (Phase-2 explicitly forbids agents from touching it); verify LIVE-mode startup guard refuses to boot if flip skipped |
| FUTURES | AMBER → GREEN candidate | **GREEN candidate** (only after pre-wave the leverage cap was silently wrong) | Run `db_futures_leverage_caps` + `api_settings_futures_post_wave2`; set `FUTURES_RISK_ASSERT_HARD=1` for canary |
| AI | AMBER → GREEN candidate | **GREEN candidate** | Quorum + bandit stay off by default; flip after calibration page shows good Brier on first 50 closed trades |
| COPY_TRADING | AMBER → GREEN candidate | **GREEN candidate (paper)** — quant rebuild is fresh, needs ~7 days of `copy_leader_scores` rows before live | Trigger `POST /api/copytrading/leaders/refresh`, then run `db_copy_leader_scores_post_refresh`; stay DRY_RUN until composite scores stabilize |

DASHBOARD module was not in Wave-2 scope.

---

## 9. Carry-over for Wave 3

These were called out by agents as deferred-by-design or out-of-scope this wave:

**DEX**
- `_quote_v3` placeholder returns `int(amount * 0.997)` — V3 routing is structurally broken until a real Uniswap V3 QuoterV2 binding lands.
- EIP-1559 `maxFeePerGas` / `maxPriorityFeePerGas` on Ethereum (still using legacy `gasPrice` Type-0).
- `_estimate_price_impact` linear-extrapolation breaks for V3 concentrated liquidity at tick boundaries.
- `_apply_time_delays` bypasses `nonce_lock` (race on concurrent DEX txs from same wallet).
- `_path_has_liquidity` returns hardcoded `True`.

**ARBITRAGE**
- Triangular atomic-receiver contract deploy (operator approval required).
- Per-DEX realized-slippage learning to replace static `default_slippage_pct` (needs ~7d of honest PnL rows now that 8cf0143 stopped writing magic numbers).
- Live `_gas_spend_usd_hour` tile in monitoring dashboard.

**SOLANA**
- **P1-08 re-classification (A3 finding):** The original look-ahead label-bug claim was wrong on closer read — X uses `scaled_data[i-lookback:i]`, y uses `price[i] vs price[i-1]`. The *actual* leakage is `scaler.fit_transform(data)` over the entire dataset before splitting (standardization leakage). Fix is in `ml/models/pump_predictor.py:201`; out of A3 scope per worktree restriction.
- Pump-predictor wiring deferred — needs per-token rolling price buffer (~60-bar window) maintained by engine monitor loop.
- Jito bundle path not lifted from `arbitrage/solana_engine.py:297` into `trading/chains/solana/jito_bundle.py`.

**SNIPER**
- Pyth-feed wiring for blue-chip mints (extension point in `_get_token_price`; deferred because Pump.fun mints have no feed-id).
- `/api/sniper/timing` per-chain GROUP BY in `monitoring/enhanced_dashboard.py` (client-side widget delivers operator visibility without backend contention).
- `safety_check_enabled=true` DB flip — operator-only step.

**FUTURES**
- Hourly funding-cost realized-vs-predicted dashboard widget.
- Per-symbol leverage table (`max_leverage_overrides`).
- Auto-deleverage trigger wiring (`should_auto_deleverage()` exists but no caller).
- Telegram alert on FUT-RM-07 emergency-close.

**AI**
- AI-Q-05..AI-Q-18 from Phase-1 audit (calibrated booster wrap, LSTM rolling buffer, EnsembleModel feature decoupling, token_scorer weight learning).

**COPY_TRADING**
- Slippage-decay tracker (leader fill price vs ours).
- Per-leader probation table & re-entry gate (CT-Q-09).
- Cross-module exposure aggregator (if DEX is long PEPE and a copied leader also buys PEPE, current code doubles up).

---

## 10. Operator quick-start

After pulling Wave 2:

```bash
# 1. Pull
git pull origin claude/create-expert-agents-JFSF5

# 2. Rebuild + restart (postgres volume stays — never `docker volume prune`)
docker compose up -d --build trading-bot

# 3. Verify both migrations applied
#    Look in logs/main/main.log for the migration runner log lines mentioning
#    023_add_ai_confidence_calibration.sql and 024_copy_leader_scores.sql.

# 4. Open dashboard, log in, go to /test_runner
```

In `/test_runner`, click these buttons in order to validate the wave:

**Round 1 — sanity (run all, expect green):**
- `script_dex_web3_v6_imports` — DEX module imports cleanly under web3 v6
- `script_dex_mev_unbound_check` — UnboundLocalError fix held
- `script_arb_nameerror_regression` — spatial-arb log-line identifiers correct
- `script_sniper_listener_widget_present` — per-chain widget DOM present

**Round 2 — DB schema (expect green):**
- `db_ai_calibration_table` — migration 023 applied, 8 columns present
- `db_copy_leader_scores_table` — migration 024 applied, 11 columns present
- `db_dex_settings_keys` — DEX max_slippage / gas-ceiling / mev_protection rows seeded
- `db_arb_cost_profile_keys` — ARB chain_cost_profile + gas_budget_usd_per_hour seeded
- `db_solana_drift_guards` + `db_solana_ml_rug_gate` — guards seeded, opt-in flags = false
- `db_futures_leverage_caps` + `db_futures_funding_gate` — FUTURES wave-2 keys seeded

**Round 3 — live HTTP surface:**
- `api_ai_calibration` — `/api/ai/calibration` returns 200 with empty buckets pre-trades
- `api_copytrading_leaders_list` — `/api/copytrading/leaders` returns cached ranked list
- `api_settings_futures_post_wave2` — `/api/settings/futures` round-trips wave-2 keys

**Round 4 — operator-priority:**
- Hit `POST /api/copytrading/leaders/refresh` from the UI (admin-only). Then click `db_copy_leader_scores_post_refresh` and confirm `fresh_5m > 0`. This proves wallet_discovery actually wrote.

**DO NOT** flip any of these without a canary first:
- `safety_check_enabled = true` (SNIPER — operator-only step before LIVE)
- `drift_enabled = true` (SOLANA — keep off until Drift canary planned)
- `solana_ml_enabled = true` (SOLANA — needs trained RugClassifier artefact)
- `atr_sizing_enabled = true` (FUTURES — recommended after first canary)
- `bandit_enabled = true` / `quorum_required = true` (AI — flip after calibration page shows good Brier on 50+ closed trades)
- `FUTURES_RISK_ASSERT_HARD = 1` (recommended for canary, hard-fails on any cap mismatch)

`DRY_RUN=true` stays everywhere by default. The kill-switch (`logs/.killswitch` / `/api/bot/emergency-exit`) is unchanged and still authoritative.

---

## Appendix A — Commit ledger by module

Full `git log --oneline claude/create-expert-agents-JFSF5 ffeda0a..HEAD` is in `git`. Selected operator-relevant commits:

**PM:** `0200aeb`, `d2f555f` (this report's siblings are `DEX_CAMPAIGN.md`, `ARBITRAGE_CAMPAIGN.md`, `SOLANA_CAMPAIGN.md`, `SNIPER_CAMPAIGN.md`, `FUTURES_CAMPAIGN.md`, `AI_CAMPAIGN.md`, `TEST_RUNNER_T1_CAMPAIGN.md`, `TEST_RUNNER_T2_CAMPAIGN.md`. COPY_TRADING_CAMPAIGN.md was rolled into `91f838d` CLAUDE.md instead of a standalone file.)

**DEX (A1):** `28c484e`, `f7d7941`, `e872121`, `23d860d`, `48d5f20`, `162f711`, `a40f69a`, `869eed3`, `1367ed5`.

**ARBITRAGE (A2):** `9e6a7d1`, `744ee48`, `4adcd29`, `8cf0143`, `89175d4`, `014384d`.

**SOLANA (A3):** `09a5c85`, `661cee6`, `83df4ad`, `b1b358f`, `7e7cff5`.

**SNIPER (A4):** `e4b8025`, `87c5523`, `77b22e7`, `6612be2`, `adee9c2`, `5a0a3e9` (R4 quorum bundled into `4adcd29`).

**FUTURES (A5):** `34399db`, `9aba5e8`, `bcc7b91`, `7e9a47b`, `ad401c4`, `2c907ca` (FUT-RM-01 leverage propagation bundled into `09a5c85` due to inflight rebase).

**AI (A6):** `6acb48b`, `8fe3671`, `6c3c058`, `9e9f0db`, `c285074`, `b26c0eb`, `db452d9`, `b427f67`.

**COPY_TRADING (A7):** `d129902`, `33ae524`, `e833abb`, `376f9be`, `2e59550`, `91f838d`.

**Test Runner (T1/T2):** `cbe5910`, `5fb94d1`, `00e8f6c`, `9642626`, `941911b`, `913c720`, `16e75c4`, `3a76af5`, `d2f555f`.

---

**Wave 2 closed. Operator: pull, restart, run the four rounds of probes above, decide on LIVE flip per module.**
