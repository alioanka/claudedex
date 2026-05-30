# PM_FINAL_WAVE6 — Wave-6 AI Key-Resolution P0 + COPY Semantics Hardening + UX

**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `1014d51` (Wave-5 close, post-PM_FINAL_WAVE5)
**Date:** 2026-05-21
**Scope:** 4 module agents (AI / COPY / ARB-Ops / Dashboard) + 1 foreground orchestrator bug-fix batch (`5a857e0`, `f81144f`).
**Wave-6 commits:** 20 on the campaign branch (T1/T2 Test-Runner coverage is being dispatched in parallel after this report).

## 1. Executive summary — headline P0

**The AI subprocess silently produced zero signals for ~4 months.**

`main_ai.py` was calling `secrets.get('ANTHROPIC_API_KEY')` **before** the `db_pool` was initialised. With no pool, the centralised `secrets_manager` stayed in bootstrap mode — the encrypted DB lookup never fired — and the `os.getenv` fallback returned `None`. The AI provider then loaded with no API key, every sentiment cycle no-op'd, no signals were emitted, no `[ai-skip]` entries were even logged. The dashboard "Why no trades?" panel from Wave-5 showed an empty diagnostic because the engine itself was not producing the inputs.

**Fix:** `cca8d94 [ai] main_ai: load API keys via secrets_manager AFTER db_pool init`. The subprocess now constructs the DB pool first, hands it to `secrets_manager`, and only then resolves provider keys — same ordering every other subprocess already used. `3939a20` swapped `ai_provider` to `secrets.get_async` so the DB path actually fires on subsequent reads.

**Operator action:** restart the AI subprocess. Signals will resume within ~15 min (first sentiment cycle after restart). The Wave-6 per-tick liveness log (`b675ab1`) and `subprocess_health` block in `/api/ai/diagnostics` (`49400d6`) confirm the engine is alive even before its first signal lands.

The implication is significant: AI was **effectively non-functional for the entire Phase-3 LIVE-readiness window**. No data was lost (no trades were attempted), but every "AI scans hot, fires nothing" observation in W3/W4/W5 was conflating the diagnostic-tracked rejections (which were real for the few signals that did get through during testing windows) with the much-larger silent-zero-signal floor. See Section 5 for the corresponding LIVE-flip checklist amendment.

## 2. Per-module Wave-6 deliverables ledger

### AI — 5 commits (`cca8d94`, `3939a20`, `b675ab1`, `49400d6`, `9d32bc4`)

| # | Deliverable | Commits |
|---|---|---|
| 1 | **P0:** `main_ai` resolves API keys via `secrets_manager` AFTER `db_pool` init (was bootstrap-mode silent failure) | `cca8d94` |
| 2 | `ai_provider` switches to `secrets.get_async` so the DB lookup path actually fires on per-call resolution | `3939a20` |
| 3 | `SentimentEngine` emits a per-tick liveness log (`[ai-tick] cycle=<n> elapsed=<ms>`) — diagnostic survives zero-signal cycles | `b675ab1` |
| 4 | `/api/ai/diagnostics` + `/api/ai/stats` now include a `subprocess_health` block (key-resolution status, last-tick age, provider-init age) using `secrets_manager` directly so the dashboard tells you when the engine is alive but silent vs dead | `49400d6` |
| 5 | `modules/ai_analysis/CLAUDE.md` Wave-6 block — documents the key-resolution ordering invariant + subprocess-health surface | `9d32bc4` |

Net result for the operator: `/ai/dashboard` distinguishes three states post-restart — **green** (engine alive + keys resolved + signals firing), **amber** (engine alive + keys resolved + no signals from market conditions — the Wave-5 surface), **red** (engine alive but keys not resolved — the W6 P0 case). Pre-W6, all three collapsed to a single silent amber.

### COPY — 5 commits + 1 in-flight helper (`2d30f3c`, `2466ec0`, `f0fb2bd`, `0770912`, `662f456`, helper `f81144f`)

| # | Deliverable | Commits |
|---|---|---|
| 1 | **In-flight helper:** `STABLECOIN_MINTS` constant set (USDC / USDT / USDH / WSOL Solana + EVM equivalents on ETH/BSC/BASE/ARB/OP/POLY) + `_has_open_copy_position(leader, mint)` helper + refusal counter on `runtime_stats` | `f81144f` |
| 2 | **Engine semantic rewrite:** `copy_engine.py` BUY/SELL detection switched from "did leader transact this mint" to **delta-based** (positive Δ-balance = BUY, negative Δ-balance = SELL). Stablecoin guard refuses USDC/USDT/USDH/WSOL as the *traded* token regardless of which side of the swap they appear on. `dashboard.copytrading` + `test_runner_routes` updated to consume new ledger keys. | `2d30f3c` |
| 3 | Trades UI: tokens-held column + entry/now per-token USD price columns + Copy/Birdeye/Solscan icon trio on every row | `2466ec0` |
| 4 | Positions + dashboard UI: same icon trio + tokens-held enrichment (parity with trades page) | `f0fb2bd` |
| 5 | `/api/copytrading/stats` exposes `stablecoin_refusals` + `leader_sold_we_dont_hold` counters | `0770912` |
| 6 | `modules/copy_trading/CLAUDE.md` Wave-6 block — documents BUY/SELL semantic + stablecoin guard | `662f456` |

**Behavioural change:** prior to W6, when a leader sold a position from token X back to USDC, the engine could mis-classify the USDC-receipt as a BUY of USDC and attempt to copy-buy USDC. Post-W6, two independent gates kill this:

1. The delta-based detector sees the leader's USDC balance went up *because* their X balance went down — and reads this as a SELL of X, not a BUY of USDC.
2. The stablecoin guard refuses to *ever* treat USDC/USDT/USDH/WSOL as a copyable BUY target, regardless of which heuristic the rest of the engine picked.

New replay-log reasons surfaced on the ledger and on `/api/copytrading/stats`:
- `stablecoin_not_tradeable` — guard fired
- `leader_sold_we_dont_hold` — `_has_open_copy_position` returned `False` so the SELL signal was dropped (we never bought it; nothing to close)

### Ops — 5 commits (`87f3ef5`, `7538a35`, `e32cf63`, `a2849b7`, `dafc929`)

| # | Deliverable | Commits |
|---|---|---|
| 1 | Backfill script: retry Jupiter 429 with exponential backoff (5/10/20/40s) + per-row summary line | `87f3ef5` |
| 2 | Backfill script: Birdeye fallback when Jupiter has no price for the historical timestamp | `7538a35` |
| 3 | `scripts/arb_engine_health.py` — operator diagnostic that prints `last_tick_at` / `last_error` / `last_error_at` from `arbitrage_runtime_stats` (alive vs dead in one command) | `e32cf63` |
| 4 | `arbitrage_engine` surfaces `last_tick_at` / `last_error` / `last_error_at` via `arbitrage_runtime_stats`; dashboard reads them | `a2849b7` |
| 5 | `modules/arbitrage/CLAUDE.md` Wave-6 block — documents new health-surface keys + restart-flag IPC pattern | `dafc929` |

The backfill changes mean `scripts/backfill_copy_tokens_received.py --force` now picks up a sixth row that was failing the Wave-5 run because Jupiter rate-limited the call (silent skip pre-W6, retry+Birdeye-fallback post-W6).

### Dashboard — 4 commits (`2500f5f`, `9d32bc4` tz-part, `2d30f3c` server-part, `f01399a`)

| # | Deliverable | Commits |
|---|---|---|
| 1 | Shared `dashboard/static/js/timezone.js` — `parseUtcTimestamp` / `formatLocalDateTime` / `formatTimeAgo`; loaded from `base.html` so every page has the helper | `2500f5f` |
| 2 | Server-side `_iso_utc()` helper added; every naive datetime returned from `/api/copytrading/*` is now serialized as a UTC-marked ISO string so the JS helper can convert deterministically | part of `2d30f3c` (`monitoring/enhanced_dashboard.py`) |
| 3 | All 6 Copy Trading templates (dashboard / trades / positions / performance / leaders / wallets) routed through the JS helper — operator sees local-timezone timestamps everywhere | part of `9d32bc4` (`dashboard/templates/*_copytrading.html`) |
| 4 | All 8 sibling templates (AI / ARB / FUT / SOL / SNIPER dashboards + trades) routed through the JS helper | `f01399a` |
| 5 | New Test Runner probe `script_timezone_helper_present` — fails if `timezone.js` is missing or `base.html` no longer loads it | added with `2d30f3c` |

Reach: 14 templates total now display all timestamps in operator-local timezone with deterministic UTC parsing on the wire. No backend timezone work required — server stays UTC, browser does the conversion.

## 3. Foreground orchestrator fixes (`5a857e0` + `f81144f`)

Two operator-reported bug batches landed alongside the agent work, **not** as part of any single module campaign — the PM thread shipped them directly because each one crossed module boundaries:

`5a857e0 [copy+futures] 4 operator-reported bugs`:
- **`/copytrading/positions` Close button** — engine looked up the wrong table when resolving the position-id click; rewired to `copytrading_trades`. Now functional.
- **`/copytrading/dashboard` Reconcile P&L button** — CSRF token was missing on the POST, request 403'd silently. Token now injected from the base template.
- **`/copytrading/dashboard` Total P&L card** — layout regression from Wave-5's realized/live split (the second line overflowed). Fixed by tightening the flex layout.
- **`/copytrading/discovery` Hot Wallets** — filter + sort controls were inert. Wired up; default sort is now last-trade-recency.
- **`/futures/settings`** — Wave-5 knobs (FUT-RM-15..18) had no UI surface; settings page now exposes confluence count + ATR multipliers + cool-off threshold/minutes + default leverage as editable fields.

`f81144f [copy]` — see Section 2; landed as a PM-foreground commit because it's the helper primitive set the COPY engine commit (`2d30f3c`) was about to consume.

## 4. Migrations + endpoints

**No migrations added this wave.** W5 already shipped migration `031`; W6 made no schema changes.

**No new HTTP endpoints this wave.** W5 already shipped `/api/arbitrage/diagnostics` and `/api/ai/diagnostics`; W6 *enriched* those existing endpoints (the AI `subprocess_health` block, the ARB `last_tick_at`/`last_error` keys, the COPY `stablecoin_refusals` counters on `/api/copytrading/stats`). The Wave-6 changes are surgical — keep that clear when reading the diff.

## 5. Live-readiness verdict

All 8 modules remain **AMBER → GREEN candidate**. No module is downgraded; no module is promoted. `DRY_RUN=true` stays everywhere; `core.dry_run.should_skip_live(...)` still gates every live-write path.

**Important amendment to the LIVE-flip checklist:** the W6 AI P0 means AI was effectively non-functional for the entire Phase-3 LIVE-readiness window — every prior "AI is amber but ready" judgement was made against a silent engine. The corrective measure for the LIVE-flip checklist:

> **Add a "credential-resolution smoke test" before any module is flipped to LIVE.** For each subprocess, immediately after startup, log the result of `secrets_manager.get_async(<every_key_the_module_needs>)` with the value redacted to `"<set>"` / `"<missing>"`. A missing required key must hard-fail the subprocess, not silently fall back to `os.getenv`. The Wave-6 `subprocess_health` block on `/api/ai/diagnostics` is the prototype — replicate the pattern in DEX / ARB / SOLANA / SNIPER / FUT / COPY before any of them flips.

This is a Wave-7 deliverable (see Section 6).

## 6. Wave-7 carry-over

**Immediate next task (already dispatched in parallel with this report):**

| Item | Owner | Why |
|---|---|---|
| T1 + T2 Test Runner coverage of the W6 diagnostics enrichments (AI `subprocess_health`, ARB `last_tick_at`/`last_error`, COPY `stablecoin_refusals` counters) | T1 / T2 (Test Runner agents) | W6 enriched endpoints but did not add corresponding probes |
| T1 + T2 Test Runner coverage of the W6 BUY/SELL semantics (delta-based detector + stablecoin guard) | T1 / T2 | Engine semantic rewrite needs golden-fixture coverage before any LIVE flip |

**Beyond that, ranked:**

1. **SOLANA `solana_positions.entry_usd` migration** — still pending from W4. Blocks COPY's `exposure_aggregator` from counting non-`copytrading_trades` Solana positions against the $5000 cross-module cap. Now particularly relevant because COPY's W6 BUY/SELL semantic exposes a cleaner Solana-position picture; the missing entry_usd column is what keeps cross-module exposure from being authoritative.
2. **EVM live PnL parity in COPY** — operator hasn't seen EVM copy positions in production yet, but DEX and AI **do** write EVM rows. As soon as the operator runs a non-Solana copy trade, the EVM-PnL gap will be visible. Deferred from W5 + W6 because the operator was still on Solana-only.
3. **DEX V3 tick-walk refinement** — W3 shipped V3 impact estimation; production traffic shows it occasionally over-estimates impact on illiquid ticks. Refinement is a smartcontract-web3 agent task.
4. **DEX bloXroute production smoke-test** — W4 shipped bloXroute BSC; never verified in mainnet production traffic.
5. **AI calibrated-prediction operator flip** — `ai_calibrated_predictions_enabled` still default-FALSE pending ≥50 calibration samples in `ai_confidence_calibration`. Now blocked by the W6 P0 fix — the calibration table will not accumulate samples until the AI subprocess is restarted. Operator can flip once the sample threshold is met; trainer pipeline (AI-Q-06) is ready.
6. **Credential-resolution smoke test rollout** to DEX / ARB / SOLANA / SNIPER / FUT / COPY (see Section 5 amendment).

**Operator-side carry-over items unchanged from W5:** FUTURES live Telegram payload test (still no mainnet emergency-close has fired), SNIPER `safety_check_enabled=true` flip (operator gate by brief).

## 7. Operator quick-start

```bash
git pull origin claude/create-expert-agents-JFSF5
docker compose up -d --build trading-bot

# Re-run the W5 backfill — W6's 429 retry + Birdeye fallback picks up the 6th row that
# silently failed under the W5 Jupiter rate-limit
docker exec trading-bot python scripts/backfill_copy_tokens_received.py --force

# /ai/dashboard should populate within ~15 min (first sentiment cycle after restart);
# subprocess_health block will go green once Anthropic key resolves from the DB
# /copytrading/* timestamps now display in operator local timezone
# /copytrading/positions Close button works
# /copytrading/dashboard Reconcile P&L button works
# Future copy trades will NEVER buy USDC/USDT/USDH/WSOL when the leader is selling
```

ARB engine health (one-shot diagnostic):

```bash
docker exec trading-bot python scripts/arb_engine_health.py
```

Prints `last_tick_at` + `last_error` + `last_error_at` + a STALE flag if the engine has been silent >10 min — same surface the dashboard reads via `arbitrage_runtime_stats`.

## 8. DRY_RUN + killswitch reminder

`DRY_RUN` stays TRUE everywhere. `logs/.killswitch` continues to be polled by every BaseModule subprocess. `SNIPER_SAFETY_CHECK_ENABLED=false` remains a Phase-2 operator-DB toggle. The W6 changes only affect (a) which signals get emitted at all (AI key fix), (b) which signals are accepted as copyable (COPY BUY/SELL + stablecoin guard), and (c) what the operator sees on the dashboard (timezone + icons + diagnostics). Nothing flipped to LIVE.

---

**Wave 6 closed. 20 commits. 0 migrations. 0 new HTTP endpoints (enrichments only). 1 P0 (`cca8d94` AI key-resolution ordering) with a single restart as recovery action. Zero LIVE flags flipped.**
