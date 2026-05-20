# PM_FINAL_WAVE4 — Wave-4 Carry-over Close-out

**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `3d1dd43` (Wave-3 close, post-PM_FINAL_WAVE3)
**Date:** 2026-05-20
**Scope:** 5 module agents (DEX / SOLANA / FUTURES / AI / COPY_TRADING) + 2 Test Runner agents (T1-W4 + T2-W4).
**Wave-4 commits:** 28 on the campaign branch.

## 1. Executive summary

Wave-4 was a pure carry-over wave — every item on the Wave-3 close-out's "deferred" list either shipped or moved one rung closer to ship. No new module audits, no rebuilds, no new strategies. Three patterns held:

- **Every new feature is flag-gated and default-OFF** except the two COPY safety rails (probation + cross-module exposure cap, both default-TRUE), which can only refuse trades — they never originate one.
- **Every new HTTP endpoint has a Test Runner probe** (Section 3 below).
- **No flags flipped to LIVE this wave.** `DRY_RUN=true` everywhere. Killswitch unchanged. `SNIPER_SAFETY_CHECK_ENABLED=false` operator-DB toggle remains operator-only.

What's safer now:

- **DEX**: V3 price-impact estimate stopped lying. The Wave-3 fix bound QuoterV2 for *quoting*; Wave-4 binds it for *impact estimation* too, killing the linear extrapolation that under-counted impact on tight V3 ranges by 5–15×. New `max_price_impact_bps` (default 200) refuses sign-off above the cap. BSC swaps got a real private-tx path via bloXroute BDN (`blxr_private_tx`), gated behind the `bloxroute_enabled` flag.
- **SOLANA**: Jito bundle path wired into the engine itself (the Wave-3 shared helper is now actually called). Pump-predictor warmup pre-fetches 60×1m bars per active token at startup via Birdeye so the gate doesn't spend the first 30 minutes of uptime in cold-start refuse-mode.
- **FUTURES**: FUT-RM-07 emergency-close (Wave-2 silent reversion) now emits a critical Telegram alert when triggered. New funding-cost forecast widget + `/api/futures/funding-forecast` endpoint reads `futures_funding_payments` (migration 029 from Wave-3) and projects 24h forward cost per (symbol, side).
- **AI**: Calibrated booster inference path (AI-Q-05) shipped behind `ai_calibrated_predictions_enabled` (default FALSE) — operator flips once calibration table has ≥50 samples. Quorum agreement-rate observability: outcome rows persisted to `ai_feature_store`, exposed via `/api/ai/quorum-metrics?hours=24`, charted on `dashboard_ai.html`.
- **COPY_TRADING**: Per-leader probation table (CT-Q-09) — bench leaders for `probation_days` after a -25%+ mirrored-trade or a score-drop under 30 (with ≥10 trades). Cross-module exposure aggregator (CT-Q-12) — refuses BUYs when the per-token open USD across DEX+SNIPER+SOLANA+COPY+AI would exceed `$5000` (default cap). Both gates default-TRUE; SELLs never gated. Migration `030_copy_probation_gate_defaults.sql`.

What's still deferred — see Section 7.

## 2. Per-module deliverables ledger

### DEX — 3 deliverables
| # | Deliverable | Commits |
|---|---|---|
| 1 | V3 price-impact via chunked QuoterV2 round-trip + refusal gate | `115cb35` |
| 2 | bloXroute BSC private-tx (plumbing + submit body) | `6f96075`, `a6c3a89` |
| 3 | `modules/dex_trading/CLAUDE.md` Wave-4 block + `DEX_WAVE4.md` | `7b2da32`, `424d8c6` |

New flags + defaults:
- `max_price_impact_bps` = `200` (refuse-above-cap; behaviour-additive)
- `bloxroute_enabled` = `false`
- `bloxroute_bsc_endpoint` = `https://api.blxrbdn.com`
- `bloxroute_auth_header` = `None` (also reads `BLOXROUTE_AUTH_HEADER` env)

Migrations: none (config-additive only).

### SOLANA — 3 deliverables
| # | Deliverable | Commits |
|---|---|---|
| 1 | Jito bundle wiring into engine (`_execute_swap_via_jito` + `_open_position` call-site + `shutdown` close) | `d6a4a8c`, SOLANA portion of `a6c3a89` |
| 2 | Pump-predictor warmup pre-fetch (Birdeye 60×1m bars per token, spot fallback) | `3edd27e`, `c69f575` |
| 3 | `modules/solana_trading/CLAUDE.md` Wave-4 block + `SOLANA_WAVE4.md` | `63884d8` |

New flags + defaults:
- `solana_jito_bundle_enabled` = `false`
- `solana_jito_tip_lamports` = `50_000` (vs arbitrage's 10k — 50k is the documented competitive landing floor)

Migrations: none.

### FUTURES — 3 deliverables
| # | Deliverable | Commits |
|---|---|---|
| 1 | FUT-RM-07b Telegram alert on emergency-close (flag plumbing + dispatcher wire-up) | `34e6c95`, `6f66608`, `68b20fb` |
| 2 | FUT-RM-09b funding-cost forecast widget + `/api/futures/funding-forecast` endpoint | `142250b`, `b805626`, `7b2da32` |
| 3 | CLAUDE.md Wave-4 block | `7b2da32` |

New flags + defaults:
- `futures_telegram_emergency_close_enabled` = `true` (fail-soft if Telegram not configured)

Migrations: none (re-uses `029_add_futures_funding_payments.sql` from Wave-3).

### AI — 3 deliverables
| # | Deliverable | Commits |
|---|---|---|
| 1 | AI-Q-05 calibrated-booster inference path (`fit_and_persist_calibration` + `calibrated_predict_proba` + `_load_calibrated_models`) | `16b7dab`, `68b20fb`, AI portion of `a6c3a89` |
| 2 | Quorum observability (`_record_quorum_outcome` / `_persist_quorum_outcome` → `ai_feature_store.metadata.quorum_outcome` + `/api/ai/quorum-metrics?hours=24` endpoint + dashboard agreement-rate widget) | `c7e4a27`, `d0a6356`, `50bd9c3` |
| 3 | `modules/ai_analysis/CLAUDE.md` Wave-4 block | `ccb4689` |

New flags + defaults:
- `ai_calibrated_predictions_enabled` = `false` (flip after calibration table has ≥50 samples)

Migrations: none (`ai_confidence_calibration` already shipped Wave-2 migration 023; quorum outcomes ride in existing `ai_feature_store.metadata` JSONB).

### COPY_TRADING — 3 deliverables
| # | Deliverable | Commits |
|---|---|---|
| 1 | CT-Q-09 per-leader probation table + auto-trigger (loss-pct + score-drop) | `f5953cd`, `ba4d15d`, `142250b` (score-side) |
| 2 | CT-Q-12 cross-module exposure aggregator + engine gate | `f5953cd`, `d0a6356` (rolled in by orchestrator) |
| 3 | `modules/copy_trading/CLAUDE.md` Wave-4 block + `COPY_WAVE4.md` | `e87e3b1` |

New flags + defaults (all `copytrading_config`; engine accepts bare AND `copy_`-prefixed keys):
- `copy_probation_gate_enabled` = `true` (safety rail)
- `copy_probation_score_threshold` = `30`
- `copy_probation_loss_pct_threshold` = `25`
- `copy_probation_days` = `7`
- `copy_cross_module_exposure_check_enabled` = `true` (safety rail)
- `copy_cross_module_exposure_cap_usd` = `5000`

Migrations: **`030_copy_probation_gate_defaults.sql`** — idempotent. Updates Wave-3 migration-026 default loss-pct from 15 → 25 ONLY when the row is still at the unmodified 15 (operator overrides survive). Also seeds the six `copy_*`-prefixed alias keys (`ON CONFLICT DO NOTHING`).

## 3. New HTTP endpoints introduced this wave

| Method | Path | Handler | Test Runner probe |
|---|---|---|---|
| GET | `/api/ai/quorum-metrics` | `api_get_ai_quorum_metrics` | `api_ai_quorum_metrics` |
| GET | `/api/futures/funding-forecast` | `api_futures_funding_forecast` | `api_futures_funding_forecast` |

Two new `router.add_get(...)` lines confirmed via `git diff 3d1dd43..HEAD`. Both endpoints have a Test Runner catalog probe. No new `add_post` / `add_put` / `add_delete` routes this wave.

## 4. New DB columns / migrations

**Migrations shipped this wave: 1 (030).**

| Migration | Purpose |
|---|---|
| `030_copy_probation_gate_defaults.sql` | Realign Wave-3 probation default from 15% → 25% loss-pct + seed six `copy_*`-prefixed alias keys. Idempotent. Operator overrides preserved. |

**Tables touched by Wave-4 code (no new columns this wave, just new rows / JSONB extensions):**

- `config_settings` — UPDATE + 6 INSERTs (migration 030)
- `copy_leader_scores` — Wave-3 migration-026 columns (`on_probation`, `probation_until`, `probation_reason`, `probation_set_at`) now consumed by Wave-4 engine
- `ai_feature_store.metadata` JSONB — new key `quorum_outcome` (no column add; JSONB extension)
- `futures_funding_payments` (migration 029, Wave-3) — read by new funding-forecast endpoint

**Carry-over (NOT shipped this wave):**
- `solana_positions.entry_usd` — COPY's `exposure_aggregator` would consume this column to count live Solana open positions against the $5000 cross-module cap. Until it lands, the aggregator under-counts open Solana exposure (it still picks up SOL-closed trades via `copytrading_trades` / `solana_trades` which carry USD basis). The COPY-W4 agent explicitly tagged this as a SOLANA-owned schema add and shipped fail-soft handling. See Section 7.

## 5. Test Runner growth

**Catalog size: 99 → 121 entries (+22).** Verified by `diff` of `monitoring/test_runner_routes.py` at `3d1dd43` vs `HEAD`.

### New entries by module

**DEX (3)** — added by T1-W4 (`7b85324`, `34215fb`):
- `db_dex_v3_quoter_addresses`
- `db_dex_bloxroute_config`
- `script_dex_quoter_v2_addresses_present`

**SOLANA (3)** — added by T1-W4:
- `db_solana_jito_flag`
- `db_solana_pump_predictor_flag`
- `script_solana_jito_helper_present`

**FUTURES (5)** — added by T1-W4 (`34215fb`):
- `api_futures_funding_forecast`
- `db_futures_funding_payments_recent`
- `db_futures_telegram_alert_flag`
- `script_futures_funding_forecast_widget_present`
- `script_futures_fut_rm_07b_notify_helper`

**AI (7)** — added by T2-W4 (`0c6d1af`, `0dd5126`, `2880bec`):
- `api_ai_quorum_metrics`
- `db_ai_quorum_outcomes`
- `db_ai_calibrated_predictions_flag`
- `db_ai_calibrated_model_artefacts`
- `script_ai_calibrated_helper_present`
- `script_ai_quorum_persist_present`
- `script_ai_quorum_widget_present`

**COPY_TRADING (4)** — added by T2-W4 (`d656b63`):
- `db_copy_probation_thresholds`
- `db_copy_probation_state`
- `db_copy_exposure_breakdown`
- `db_copy_cross_module_cap`

Per-module probe-type coverage is uniform: every new endpoint has an `api_*` probe, every new flag has a `db_*` probe reading `config_settings`, every new helper path has a `script_*` grep probe.

## 6. Live-readiness verdict per module

Wave-3 left all 7 modules at **AMBER → GREEN candidate**. Wave-4 did **NOT** move any module to GREEN — each one still requires operator-side production verification (a real fill on testnet/mainnet with `DRY_RUN=false`), which is operator scope, not agent scope. What Wave-4 did do is widen the safety net so the eventual GREEN promotion has more guardrails:

| Module | Pre-W4 | Post-W4 | Net change |
|---|---|---|---|
| DEX | AMBER → GREEN candidate | AMBER → GREEN candidate | + V3 impact no longer under-counted (eliminates the worst slippage surprise vector); + BSC private-tx path. Both default-safe. |
| ARBITRAGE | AMBER → GREEN candidate | AMBER → GREEN candidate | No W4 work (Wave-3 closed all in-scope items). |
| SOLANA | AMBER → GREEN candidate | AMBER → GREEN candidate | + Jito wired (default-OFF); + warmup removes 30-min cold-start hole when pump-predictor flipped on. |
| SNIPER | AMBER → GREEN candidate | AMBER → GREEN candidate | No W4 work (Wave-3 closed Pyth feed + resolution chain). Final operator action — flip `safety_check_enabled=true` in DB — still operator-only. |
| FUTURES | AMBER → GREEN candidate | AMBER → GREEN candidate | + Telegram alert on FUT-RM-07 emergency-close (default-TRUE, fail-soft); + 24h funding-cost forecast widget. |
| AI | AMBER → GREEN candidate | AMBER → GREEN candidate | + Calibrated booster inference (default-OFF; awaits calibration-table maturity); + quorum observability for operator confidence-building. |
| COPY_TRADING | AMBER → GREEN candidate | AMBER → GREEN candidate | + Probation gate (default-TRUE); + cross-module exposure cap (default-TRUE). Both REDUCE the set of trades the module will fire — strictly defensive. |

**No module is downgraded.** The two default-TRUE additions (COPY probation + cross-module cap) only REFUSE trades; they cannot originate one. Every other addition is flag-gated OFF.

## 7. Carry-over for Wave 5

| Item | Owner | Why deferred |
|---|---|---|
| **DEX V3 tick-boundary impact refinement** | smartcontract-web3-expert | Wave-4 chunked QuoterV2 round-trip matches what Uniswap's own frontend does; per-tick walk (`slot0` + `liquidityNet`) is more accurate for orders crossing 10+ ticks. Not blocking. |
| **DEX bloXroute submit-path production verification** | operator | The submit body shipped (`a6c3a89`) + 7 unit tests green, but no live BSC swap has used the path yet (`bloxroute_enabled` default false; operator must supply `BLOXROUTE_AUTH_HEADER`). Smoke-test on testnet before flipping. |
| **SOLANA `solana_positions.entry_usd` column** | smartcontract-web3-expert (SOLANA) | Blocks COPY's `exposure_aggregator` from counting live Solana open positions against the $5000 cross-module cap. Until shipped, aggregator under-counts open Solana exposure; per-module COPY cap + RiskManager remain the safety net. |
| **FUTURES live Telegram payload testing** | operator | Dispatcher wired + flag default-TRUE + fail-soft path verified, but no real emergency-close has fired in production. First mainnet FUT-RM-07 trigger will be the live test. |
| **AI calibrated-prediction operator flip** | operator | `ai_calibrated_predictions_enabled` default FALSE pending ≥50 samples in `ai_confidence_calibration` table. AI-Q-06 trainer pipeline already writes both `calibrated_<name>.pkl` and `<name>_calibrated.{pkl,joblib}` artefacts; flip is just a `UPDATE config_settings` once the table is mature. |
| **COPY EVM live PnL backfill** | operator | Wave-2/3 design assumed the operator was actively running COPY-on-Solana with one set of leader wallets. DEX/AI write EVM positions to `trades` / `ai_trades`; the aggregator reads them, but no EVM leaders were configured during Wave-4 testing. First EVM leader add will be the live verification. |

Out-of-scope items that the campaign brief explicitly never promised:
- ARBITRAGE triangular path live execution (atomic-receiver contract deploy is operator-approval gated — scope cut by brief, not defect).
- SOLANA Drift perp leg as default-on (MB-15 designed it as feature-flag from inception).
- SNIPER `safety_check_enabled=true` DB flip (operator gate by brief).

## 8. Operator quick-start (Wave 4)

```bash
# 1. Pull
git fetch origin claude/create-expert-agents-JFSF5
git checkout claude/create-expert-agents-JFSF5
git pull origin claude/create-expert-agents-JFSF5

# 2. Rebuild trading-bot (NEVER prune the postgres volume)
docker compose up -d --build trading-bot
docker logs -f trading-bot   # ctrl-c after 30s; verify clean startup

# 3. Apply migration 030 (idempotent; safe to re-run)
docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" tradingbot \
  -f /migrations/030_copy_probation_gate_defaults.sql

# 4. Confirm probation defaults landed
docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" tradingbot \
  -c "SELECT key, value FROM config_settings WHERE config_type='copytrading_config' AND key LIKE 'copy_probation%';"
```

After login, in the dashboard's Test Runner page click through the 22 new entries listed in Section 5 — they are grouped by module prefix (`db_dex_*`, `db_solana_*`, `db_futures_*` / `api_futures_*` / `script_futures_*`, `db_ai_*` / `api_ai_*` / `script_ai_*`, `db_copy_*`). Expected outcomes:

- All `db_*` probes return rows (config_settings, table-exists checks).
- `api_ai_quorum_metrics` returns `{ "agreement_rate": <number>, "samples": <number> }` — fresh deploys will show `samples: 0` until the AI module ships a few signals.
- `api_futures_funding_forecast` returns an empty list until at least one row lands in `futures_funding_payments` from the Wave-3 funding-cost worker.
- All `script_*` probes confirm helper files are on disk.

**Default behaviour is unchanged from Wave-3** for every module except COPY_TRADING, where the two safety-rail gates (probation + cross-module cap) are now active. If the operator wants pre-Wave-4 COPY behaviour:

```sql
UPDATE config_settings SET value='false' WHERE config_type='copytrading_config'
  AND key IN ('copy_probation_gate_enabled', 'copy_cross_module_exposure_check_enabled');
```

(Not recommended — these gates only refuse demonstrably bad bets.)

## 9. DRY_RUN reminder

`DRY_RUN` stays TRUE everywhere. No flags flipped to LIVE this wave. Killswitch unchanged. `SNIPER_SAFETY_CHECK_ENABLED=false` remains a Phase-2 operator-DB toggle (re-enable before any live SNIPER run). `core.dry_run.should_skip_live(...)` continues to gate every live-write path in every module — Jito bundle path is no exception (`JupiterHelper.execute_swap` honors the same gate, so DRY_RUN short-circuits before any bundle is signed).

---

**Wave 4 closed. 28 commits. +22 Test Runner entries. 1 migration. 2 new HTTP endpoints. Zero LIVE flags flipped.**
