# PM_FINAL_WAVE5 — Wave-5 Bug-fix + Observability + Test Runner UX Close-out

**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `c9adaa8` (Wave-4 close, post-PM_FINAL_WAVE4)
**Date:** 2026-05-20
**Scope:** 5 module agents (COPY / ARB / FUT / AI / Dashboard-Test-Runner) + 1 P0 fix (`6fe0a36` engine entry_price).
**Wave-5 commits:** 23 on the campaign branch.

## 1. Executive summary

Wave-5 was **not** a carry-over wave. It was driven by operator-reported reality after Wave-4:

- COPY positions showed wrong PnL (P0 — see Section 2).
- ARB and AI both scanned hot but fired zero trades; operators had no instrumentation to ask *why*.
- FUT lost real money on a 4-trade 25% win-rate session (3 SL hits on 4%-ATR alts at 10x; static 2% SL too tight).
- Test Runner had become noisy (120+ entries with no way to filter / search / re-run a subset).

Five themes shipped in response:

- **One P0 fix** — `6fe0a36` corrects the per-token USD price written to `copytrading_trades.entry_price`. Pre-fix value was the native-SOL spot price, which made every Solana copy position's unrealized PnL meaningless. Five open positions need backfill (see Section 2).
- **Observability over silence** — every module that "scans but doesn't fire" now emits a structured skip-reason ledger (`[arb-skip]`, `[ai-skip]`), a `/api/<module>/diagnostics` JSON snapshot, and a dashboard "Why no trades?" collapsible panel with a colour-coded status pill.
- **Live unrealized PnL** — COPY hot-wallets / dashboard / performance / trades pages now surface mark-to-market on every Solana open position, badged "Live" vs "Realized". DEX EVM PnL parity is Wave-6 (operator confirmed only Solana matters this wave).
- **FUTURES risk hardening** — confluence gate (FUT-RM-15), ATR-scaled SL/TP (FUT-RM-16), per-symbol consecutive-loss cool-off (FUT-RM-17), and a global leverage default drop 10x → 5x (FUT-RM-18, migration 031). Together these make the 25%-win-rate session structurally less likely to recur.
- **Test Runner UX** — tag system (`must`/`new`/`p0`/`flaky`/`expected-empty`), coloured pills next to every card title, sticky toolbar with search box + filter chips + Run Filtered button, and per-section recent history.

**Default behaviour changes this wave:**

- New futures positions now default to **5x** leverage (was 10x). Operator overrides per-symbol via FUT-RM-08 table; per-account customisations on `futures_default_leverage` survive (migration 031 only updates when value is still legacy 10).
- ATR-scaled SL/TP is **default-ON** (`futures_atr_dynamic_sl_tp_enabled=true`). Float `atr_sl_min_pct=1.5` is the floor so behaviour can never become *less* protective than the old static 2% on low-vol pairs.
- Per-symbol cool-off after **2 consecutive losses** is **default-ON** (`futures_post_loss_cooloff_threshold=2`, `futures_post_loss_cooloff_minutes=240`).
- Confluence gate **default-ON** at min 2 of 4 indicators (`futures_min_signal_confluence_count=2`).
- AI quorum auto-pass when only one provider key is loaded — fixes a latent footgun where `quorum_required=true` + single key = silent zero-trade.

Everything else is observation-only or operator-toggle.

## 2. P0 — `6fe0a36` engine `entry_price` per-token USD fix

**What broke:** `copy_engine.py` was writing the native SOL spot price into `copytrading_trades.entry_price` instead of the per-token USD price. Every Solana copy position's mark-to-market PnL was therefore comparing today's per-token USD price against yesterday's SOL price — garbage in, garbage out. Discovered when operator opened `/copytrading/dashboard` and saw five open positions all showing nonsense PnL.

**Fix:** `6fe0a36 [copy] P0: engine writes per-token USD price into entry_price (was native SOL price)`. The engine now resolves and persists the per-token USD price at fill time. New `tokens_received` column populated alongside. Going-forward correctness is restored at the next BUY.

**Operator action required for the 5 pre-fix open positions:**

```bash
docker exec trading-bot python scripts/backfill_copy_tokens_received.py --force
```

The backfill walks the five rows with `tokens_received IS NULL`, re-resolves the per-token USD price at the original `entry_timestamp` via the same provider chain the engine uses now, and writes both `entry_price` and `tokens_received`. `--force` is required because the script defaults to dry-run.

This is the only wave-5 item that requires an operator-side data-touch command. Everything else in Section 6 is config flips or dashboard navigation.

## 3. Per-module deliverables ledger

### COPY_TRADING — 3 deliverables (1 is the P0)

| # | Deliverable | Commits |
|---|---|---|
| 1 | **P0:** engine writes per-token USD price into `entry_price` + new `tokens_received` column populated | `6fe0a36` |
| 2 | `_enrich_copytrading_pnl` helper + live unrealized PnL on `/api/copytrading/trades`, dashboard hero-stat split (realized vs live), legacy-pending-backfill badge | `537a27b`, `54f8739` |
| 3 | Hot-wallets discovery page shows real per-wallet stats with live unrealized PnL | `395fe68` |

What the operator sees post-fix:
- `/copytrading/dashboard` hero P&L pill now shows the realized/live split (e.g. `+$12.40 (R: +$3.10 + Live: +$9.30)`) and badges legacy unbackfilled rows.
- `/copytrading/trades` rows badge each open Solana position with mark-to-market unrealized.
- `/copytrading/wallets` per-leader stats are populated from the real `copy_leader_scores` cache instead of the previous zero-placeholder.

**EVM live PnL parity is Wave-6.** Operator confirmed: only Solana copy positions are currently in play, so EVM enrichment was scope-cut from this wave to avoid touching DEX/AI executor paths in the same batch.

### ARBITRAGE — 4 commits

| # | Deliverable | Commits |
|---|---|---|
| 1 | Near-miss logging in opportunity evaluation path | `d2e1019` |
| 2 | `GET /api/arbitrage/diagnostics` endpoint | `012887a` |
| 3 | "Why no trades?" collapsible panel on `/arbitrage/dashboard` | `4fd9b33` |
| 4 | Latent dashboard crash fix: `_record_near_miss` helper that `d2e1019` referenced but never defined | `4d08a92` |

Skip-reason vocabulary persisted to `arbitrage_runtime_stats.near_misses` (rolling 50-deep deque, cross-process readable): `raw_spread_negative` (sampled 1:120), `min_profit` (sampled 1:40), `daily_cap`, `cooldown`, `gas_budget`, `risk_manager`, `risk_manager_error`. Same surface returns `stale=true` when the snapshot is older than 10 minutes — so the operator can tell engine-not-running apart from no-opportunities.

`4d08a92` is the recovery commit for a latent crash: `d2e1019` called `self._record_near_miss(...)` without defining the helper, which would have crashed every opportunity scan the first time a rejection happened. Worth calling out because it could easily have shipped silent.

### FUTURES — 6 commits (FUT-RM-15 / 16 / 17 / 18 + migration 031 + CLAUDE.md)

| # | Deliverable | Commits |
|---|---|---|
| 1 | **FUT-RM-15** — multi-indicator confluence gate (min 2 of 4: RSI/MACD/BB/EMA) | `e3af3eb` |
| 2 | **FUT-RM-16** — ATR-scaled SL/TP per symbol (`SL = max(atr_sl_min_pct, atr_sl_multiplier × ATR%)`, `TP1 = atr_tp_rr_ratio × SL`) | `8e79b08` |
| 3 | **FUT-RM-17** — per-symbol consecutive-loss cool-off (2 losses → 4h bench for that symbol; winning trade clears counter) | `69862c1`, `ac7a57b` |
| 4 | **FUT-RM-18** — default leverage **10x → 5x** | `57a6a2c` |
| 5 | Migration `031_reduce_futures_default_leverage.sql` (idempotent; operator overrides survive) | `57a6a2c` |
| 6 | CLAUDE.md Wave-5 block | `809c14b` |

Wave-5 was operator-triggered: 4 trades, 25% win rate, –$14.99 realized. Three SL hits on AAVE/FIL/NEAR shorts (–20% each on 10x), one TP1 on SOL long (+18%). Audit pinned two structural causes — static 2% SL too tight for 4%-ATR alts at 10x, and the +4 `signal_score` bar clearing on a single strong indicator. Wave-5 closes both.

New flag defaults (all `futures_config`; per-symbol overrides via existing FUT-RM-08 table win):
- `futures_min_signal_confluence_count` = `2` (0 = disabled)
- `futures_atr_dynamic_sl_tp_enabled` = `true`
- `futures_atr_sl_multiplier` = `1.5`
- `futures_atr_sl_min_pct` = `1.5` (floor — guarantees behaviour ≥ old static 2% on low-vol pairs)
- `futures_atr_tp_rr_ratio` = `2.0`
- `futures_post_loss_cooloff_threshold` = `2`
- `futures_post_loss_cooloff_minutes` = `240`
- `futures_default_leverage` = `5` (was 10; migration-031-conditional)

### AI — 5 commits

| # | Deliverable | Commits |
|---|---|---|
| 1 | Skip-reason ledger — `SentimentEngine._record_skip()` emits `[ai-skip] reason=<gate> conf=<n> sentiment=<n>` for every rejected signal | `28efc1a` |
| 2 | `GET /api/ai/diagnostics?hours=24` — joins sentiment_logs + ai_trades + the [ai-skip] log tail; returns signals/trades/action_rate/buy-sell-hold split/rejection Counter/recent skips/effective_config (REDACTED)/operator hint | `71988d2` |
| 3 | "Why no trades?" collapsible panel on `/ai/dashboard` with colour-coded status pill (green firing / red no-trades / amber action-needed / grey idle) | `c64890c` |
| 4 | Quorum auto-pass — when `quorum_required=true` but only one provider key is loaded, 1/1 agreement is trivially satisfied; single-provider score passes through unchanged with a one-time startup log line | `e271ad1` |
| 5 | CLAUDE.md Wave-5 block | `4468a11` |

**ROOT CAUSE confirmed by the new diagnostic for "50 signals, 0 trades":** operator never flipped `direct_trading=true` in `/ai/settings`. The skip-reason ledger captures every rejected signal as `[ai-skip] reason=direct_trading_off conf=<n> sentiment=<n>`; the diagnostic endpoint elevates this case to the top of the `hint` field; the dashboard pill goes amber with a one-line "Master enable is OFF — flip `direct_trading=true` on /ai/settings" CTA.

Known reason enum on `[ai-skip]`: `direct_trading_off` (#1 cause), `confidence_below_threshold`, `zero_sentiment`, `position_exists`, `cooldown_active`, `risk_rejected`, `exchange_unavailable`, `execution_failed`.

### Dashboard / Test-Runner — 4 commits

| # | Deliverable | Commits |
|---|---|---|
| 1 | Tag system on the test catalog: `must`, `new`, `p0`, `flaky`, `expected-empty` | `1e751c3` |
| 2 | Visual tag pills next to each test card title | `f9a1efe` |
| 3 | Sticky toolbar HTML: search box, filter chips, count chips | `a99e78c` |
| 4 | Wire toolbar: search + filter chips + Run Filtered button + per-section recent history | `8af28ca` |

Operator workflow gain: from "scroll through 120+ cards by hand" to "click `must` chip → see 12 cards → click Run Filtered". Per-section recent history makes triage cycles faster — most-recent result for each test is now adjacent to the run button.

## 4. New HTTP endpoints this wave

| Method | Path | Handler (`enhanced_dashboard.py`) | Test Runner probe |
|---|---|---|---|
| GET | `/api/arbitrage/diagnostics` | `api_get_arbitrage_diagnostics` (line 10305) | **MISSING — Wave-6 carry-over for T1/T2** |
| GET | `/api/ai/diagnostics` | `api_get_ai_diagnostics` (line 13735) | **MISSING — Wave-6 carry-over for T1/T2** |

Both endpoints are live in `enhanced_dashboard.py` (`router.add_get` calls confirmed at lines 933 and 971). Neither has a `monitoring/test_runner_routes.py` catalog entry yet — a `grep -n "diagnostics" monitoring/test_runner_routes.py` returns zero hits. This is the **one place Wave-5 deviated from the post-W4 convention** ("every new HTTP endpoint has a Test Runner probe"). T1/T2 ran the Test Runner UX overhaul instead of catalog adds this wave; the two probes are explicitly Wave-6 carry-over.

No new `add_post` / `add_put` / `add_delete` routes this wave.

## 5. New DB columns / migrations

**Migrations shipped this wave: 1 (031).**

| Migration | Purpose |
|---|---|
| `031_reduce_futures_default_leverage.sql` | Lower the DB-seeded `futures_leverage.default_leverage` from 10 → 5 ONLY when the value is still the legacy 10. Operator customisations survive. Pairs the pydantic/env/engine default drop in `57a6a2c`. |

**Tables touched (no new columns beyond what the P0 fix consumes):**

- `config_settings` — UPDATE only via migration 031 (no INSERTs).
- `copytrading_trades` — `tokens_received` column now populated by the engine + backfill script (column itself shipped pre-Wave-5; W5 just starts writing to it correctly).

## 6. Operator action items (in execution order)

1. **Pull + rebuild.**
   ```bash
   git fetch origin claude/create-expert-agents-JFSF5
   git checkout claude/create-expert-agents-JFSF5
   git pull origin claude/create-expert-agents-JFSF5
   docker compose up -d --build trading-bot
   docker logs -f trading-bot   # 30s sanity-check
   ```

2. **Apply migration 031 (idempotent).**
   ```bash
   docker exec trading-postgres psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" tradingbot \
     -f /migrations/031_reduce_futures_default_leverage.sql
   ```

3. **Recover the 5 pre-fix COPY open positions.**
   ```bash
   docker exec trading-bot python scripts/backfill_copy_tokens_received.py --force
   ```
   Verify post-run by reloading `/copytrading/dashboard` — the hero P&L pill should now show a meaningful realized/live split; legacy-pending badge should drop to 0.

4. **Unblock AI trade execution.** Open `/ai/settings`, flip `direct_trading=true`, save. The Wave-5 diagnostic at `/ai/dashboard` "Why no trades?" panel will go from amber to green once the next signal fires and opens a position.

5. **Inspect ARB live diagnostics.** Open `/arbitrage/dashboard` and expand the "Why no trades?" panel. The panel surfaces per-chain cards + colour-coded near-miss table; the header alone tells you top rejection reason + STALE flag without expanding.

6. **Tune FUT Wave-5 knobs.** Open `/futures/settings`:
   - confirm `default_leverage = 5` (was 10);
   - adjust `min_signal_confluence_count` (default 2) if the bar feels too high;
   - leave `atr_dynamic_sl_tp_enabled = true`;
   - leave `post_loss_cooloff_threshold = 2` / `post_loss_cooloff_minutes = 240` unless you want a tighter or looser bench.

7. **Try the new Test Runner.** Open `/test_runner`, click the `must` chip (or `new` for the most-recent additions), then click `Run Filtered`. Per-section recent history shows the previous run result next to each card.

## 7. Wave-6 carry-over

| Item | Owner | Why deferred |
|---|---|---|
| Test Runner probes for `/api/arbitrage/diagnostics` and `/api/ai/diagnostics` | T1 / T2 (Test Runner agents) | Wave-5's Test Runner work went into UX (tags, sticky toolbar, search, Run Filtered, per-section history); catalog adds for the two new endpoints did not land. Both endpoints are live and operator-usable directly; only the in-dashboard one-click probe is missing. |
| **SOLANA `solana_positions.entry_usd` migration** | smartcontract-web3-expert (SOLANA) | Still pending from Wave-4. Blocks COPY's `exposure_aggregator` from counting live Solana open positions against the $5000 cross-module cap; also blocks COPY's Solana enrichment from reaching positions held outside the `copytrading_trades` table. Per-module COPY cap + RiskManager remain the safety net. |
| **DEX EVM live PnL** | quant-algo-expert + backend-devops-expert | Operator confirmed only Solana copy positions matter this wave, so EVM enrichment was scope-cut to avoid touching DEX/AI executor paths in the same batch as the COPY P0. Re-open in Wave-6 once the Solana enrichment has run in production for a few days. |
| **COPY EVM live PnL parity with Solana** | quant-algo-expert | Same scope cut — Solana-only this wave. EVM rows still display realized PnL only; "Live" badge does not appear on EVM positions. |
| **FUTURES live Telegram payload test** | operator | Carry-over from Wave-4. Dispatcher wired + flag default-TRUE + fail-soft path verified, but no real FUT-RM-07 emergency-close has fired in production. First mainnet trigger will be the live test. |
| **AI calibrated-prediction operator flip** | operator | Carry-over from Wave-4. `ai_calibrated_predictions_enabled` default FALSE pending ≥50 samples in `ai_confidence_calibration`. AI-Q-06 trainer pipeline already writes both `calibrated_<name>.pkl` and `<name>_calibrated.{pkl,joblib}` artefacts; flip is a single `UPDATE config_settings` once the table is mature. |

Out-of-scope items the campaign brief never promised (no change vs Wave-4): ARB triangular path live execution (atomic-receiver deploy is operator-approval gated), SOLANA Drift perp default-on (MB-15 designed feature-flag from inception), SNIPER `safety_check_enabled=true` flip (operator gate by brief).

## 8. Live-readiness verdict

All 7 modules remain **AMBER → GREEN candidate**. Wave-5 did not promote any module to GREEN — each one still requires operator-side production verification under `DRY_RUN=false`. What Wave-5 did do:

- **COPY** is now safer to verify: pre-fix Solana copy PnL was garbage. Post-fix + post-backfill, the operator can read live PnL meaningfully for the first time.
- **ARB / AI** went from "scans hot, fires nothing, silent" to "scans hot, fires nothing, here's exactly why" — which removes the largest impediment to operator confidence.
- **FUT** structural risk profile got tighter: confluence + ATR + cool-off + 5x default. The 25%-win-rate session is structurally less likely.

No module is downgraded. No flag was flipped to LIVE this wave. `DRY_RUN=true` everywhere. `core.dry_run.should_skip_live(...)` continues to gate every live-write path; the COPY P0 fix sits *before* the broadcast in the same call frame as the existing `should_skip_live` check.

## 9. DRY_RUN reminder

`DRY_RUN` stays TRUE everywhere. Killswitch unchanged. `SNIPER_SAFETY_CHECK_ENABLED=false` remains a Phase-2 operator-DB toggle. All Wave-5 behavioural changes (confluence gate, ATR SL/TP, cool-off, 5x leverage default) take effect only when the operator subsequently flips a module to live — they're config-resolved at trade time, not at startup, so the operator can A/B them by toggling and observing.

---

**Wave 5 closed. 23 commits. 1 migration (031). 2 new HTTP endpoints (probe-coverage deferred). 1 P0 (`6fe0a36` engine entry_price) requiring a one-shot backfill script. Zero LIVE flags flipped.**
