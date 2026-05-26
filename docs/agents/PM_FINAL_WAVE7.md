# PM FINAL — Wave 7 verification gate (2026-05-26)

Branch `claude/create-expert-agents-JFSF5`. Verification method: code read +
`python -m py_compile` (no docker / DB / live bot available; Postgres on VPS
only). DRY_RUN remained TRUE everywhere; no LIVE flip, no force-push, no amend.

## VERDICT: SHIP (DRY_RUN data collection) — with the punch-list below

All 22 reported issues are fixed in code and verified. The PM closed 5
trivial-but-load-bearing gaps the agents left (CSRF close-fetch headers,
futures analytics column mismatch, telegram token carry-over, AI funding
note, solana portfolio double-count). After these, the operator will collect
CORRECT DRY_RUN data for ALL 7 modules. Remaining open items are
enhancement-grade (ML quality, DEX idle health) and do NOT corrupt data.

---

## Per-issue table

| # | Issue | Commit(s) | Status | Evidence |
|---|---|---|---|---|
| 1 | /full-dashboard inaccurate, "ENABLED (no health)", empty charts | 34ddaf4, 5a65264 | VERIFIED | tz crash fixed (`_as_utc` L88, 10 uses); charts use `_unified_closed_trades`; health fallbacks added |
| 2 | /analytics fake PnL (Sniper), no entry/exit, identical size/PnL | 16c8867, c3bcc6d, **PM eb969d0** | VERIFIED+FIXED-BY-PM | Sniper `_model_dry_run_exit_pct` (55/30/15 dist, seeded), ±20% entry jitter; analytics reads real per-trade columns; PM fixed futures empty-in-analytics |
| 3 | main dashboard charts only DEX/Futures/Solana | 5a65264, c161cd4, **PM eb969d0** | VERIFIED+FIXED-BY-PM | `api_performance_charts` + summary use `_unified_closed_trades` (all 7); PM fixed futures dropped from unified set |
| 4 | /dex/dashboard stuck GRAIL position; ML review | 7f12036, 87f16d9 | VERIFIED | `DexPositionService.price_refresh_loop` (DB-first, started in `main_dex.run` L940-942); stale-flag after 5 fails/30min; ML findings punch-listed (carry-over B) |
| 5 | /dex/positions close 503 (`/api/position/close`) | 7f12036, 635d55e | VERIFIED | `api_close_position` routes integer `trades.id` → `logs/.close_dex_<id>`; poller `close_flag_loop` (L298) matches byte-for-byte |
| 6 | ALL modules close buttons must work | 635d55e, 921fa65, **PM ea223da** | VERIFIED+FIXED-BY-PM | backend routing all correct; PM fixed missing CSRF header on Solana/Futures/Sniper close fetches (the real HTTP-layer break) |
| 7 | /futures poor perf — strategy enhancement | d805d37, 1bb8a4e, 284dcfd, **PM eb969d0** | VERIFIED+FIXED-BY-PM | MACD signal-line fix, FUT-RM-19 edge gate, FUT-RM-20 candle throttle, FUT-RM-21 regime gate; PM fixed futures showing empty in /analytics+charts so operator can SEE the data |
| 8 | /solana Active Positions empty despite 7 open | c9cfb1f | VERIFIED | `_reconcile_positions_on_startup` treats DRY_RUN sim positions as recoverable, not phantom (L2357-2360) |
| 9 | /solana close "CSRF token missing or invalid" | **PM ea223da** | FIXED-BY-PM | NOT in any Wave-7 commit. `positions_solana.html` close fetch had no `X-CSRF-Token`; double-submit middleware 403'd it. PM added `window.withCsrfHeaders` |
| 10 | /solana unrealistic exit price/PnL (+495424%, ORCA $6788) | 94d3eee | VERIFIED | DexScreener price now filters `baseToken.address == mint` (L474-477) before max-liquidity pick — fixes base/quote identity bug |
| 11 | Solana Drift enabled, zero activity + guide | def2957, 35c6c41 | VERIFIED | Drift DRY_RUN synthesizes funding + emits activity; setup guide in CLAUDE.md |
| 12 | /sniper 93% WR, 400+ open, funding | 16c8867, 95e1605 | VERIFIED | realistic exit dist (issue 2); `max_active_positions=500` enforced (L429, L817); `_compute_funding_recommendation` surfaced |
| 13 | /sniper/trades empty | e00dc72, 635d55e | VERIFIED | engine writes `sniper_trades`; `api_get_sniper_trades` reads it (no status filter, JSON-safe casts L10104-10141); page fetches `/api/sniper/trades?limit=2000` |
| 14 | /arbitrage dashboard dead since Feb 7 | e7e61ca | VERIFIED | `_realized_slip_*` attrs init in `EVMArbitrageEngine.__init__` (L1033-1036), inherited by ETH/ARB/Base subclasses (no `__init__` override) |
| 15 | surface wallet/exchange per module incl AI+Copy | 9a19815, 4ecafa9, 95e1605, 1ddf4ee, 35c6aad, 7934423, e96df31, **PM a601289** | VERIFIED+FIXED-BY-PM | `/api/funding/accounts` covers all 7; PM made AI note explicit (delegates to futures account) |
| 16 | ARB millions of `_realized_slip_refreshed_at` AttributeError | e7e61ca | VERIFIED | same as #14 — attrs initialized |
| 17 | Copy HELIUS/ETHERSCAN "NOT SET"; "Solana RPC rate limited" | 1d9ab68, 35c6aad | VERIFIED | secrets re-init out of bootstrap w/ db_pool (L199-201), `get_async` post-db_pool (L263); prefers Helius RPC (L269-271) |
| 18 | dashboard_errors.log naive/aware datetime crash | 34ddaf4, 80c1ddf, c3bcc6d | VERIFIED | `_as_utc`/`_iso_utc` normalize all subtractions; `pd.to_datetime(..., utc=True)` in charts |
| 19 | Futures TELEGRAM_BOT_TOKEN not set | 8a105b2 | VERIFIED | pre-warms `secrets._cache` via `get_async` before constructing controller (L770-777) |
| 20 | Orchestrator scored=7 but only 2 to_live recs | 237d10b | VERIFIED | `run_tick` iterates all 7 `_MODULE_QUERIES`; under-traded modules emit `not_ready` 'hold'; supersede-then-insert = 1 row/module/tick. Module wired in main.py L554 |
| 21 | logs/sniper growing | a6abd4e | VERIFIED | per-trade/per-candidate lines moved to DEBUG; main handler 10MB×5→×3 |
| 22 | Solana `Error getting wallet balance` + RAPID CRASH spam | 00a3a52 | VERIFIED | `_get_wallet_balance` throttled (15s TTL) + fail-soft cache + 30s log throttle; rapid-crash guarded by `current_price > 0` (L3031) so feed-miss ≠ false crash |

Carry-over C (shared telegram sync-token) fixed by PM (e42de7a).
`_db_portfolio` solana double-count fixed by PM (b716f66).

---

## Close-routing match matrix (issue 6, both sides confirmed)

| Module | Dashboard writes/calls | Module reads | Match |
|---|---|---|---|
| DEX | `logs/.close_dex_<id>`; `<id>`=int `trades.id` (validated open, non-Solana) | `close_flag_loop` polls `.close_dex_*`, `close_position(int(raw_id))` by `trades.id` | YES |
| COPY | `logs/.close_copy_<trade_id>` | `_process_close_flag_files` polls `.close_copy_*`, looks up `copytrading_trades` | YES |
| SNIPER | DB UPDATE `sniper_trades SET status='closed'` (single by `token_address`, all by status) | engine writes `sniper_trades`; retires in-mem snipe on monitor tick | YES (table fixed in 635d55e; was `trades WHERE strategy='sniper'` = 0 rows) |
| SOLANA | `POST :8082/close-position {mint}` | `close_position_handler` reads `mint` → `_close_position(mint)` | YES |
| FUTURES | `POST :8081/position/close {symbol}` | `close_position_handler` reads `symbol` | YES |
| ARB | none (close buttons removed) | atomic, no positions | YES (921fa65) |
| ORCH/AI | none (no close button) | meta-modules, no positions | YES |

**CSRF (PM ea223da):** Solana/Futures/Sniper close fetches were missing the
`X-CSRF-Token` header → 403 on every close regardless of correct backend
routing. PM added global `window.withCsrfHeaders` in `base.html` and applied
it. This was the actual cause of issue 9 and the HTTP-layer half of issue 6.

---

## Wallet / identity coverage matrix (issue 15)

| Module | Source | Surfaced via /api/funding/accounts | Secret-safe |
|---|---|---|---|
| DEX | `:8085/health .wallet_address` | yes (`evm_wallet`, shared EOA) | public addr only |
| SOLANA | `:8082/health .wallet_address` | yes (`solana_wallet`) | public addr only |
| FUTURES | `:8081/health exchange/network/api_key_fingerprint` | yes (`exchange`) | masked `****last4` only |
| SNIPER | `sniper_runtime_stats.stats` | yes (wallet/solana/evm) | public addr only |
| ARB | `arbitrage_runtime_stats.stats.wallet_address/.chain` | yes (`evm_wallet` + chains) | public addr only |
| COPY | `config_settings('copytrading_diagnostics')` | yes (evm/solana execution wallet) | public addr only |
| AI | delegated (no own wallet) | yes — PM made it explicit: delegates to FUTURES executor → trades land on FUTURES exchange account (delegates_to, execution_exchange/network) | n/a |

No private keys / keypairs / full API keys are surfaced anywhere. Confirmed
by grep over the Wave-7 diff (only masked fingerprints + "NOT SET/SET" log
lines + public addresses).

---

## Safety-invariant check results

| Invariant | Result |
|---|---|
| DRY_RUN defaults remain TRUE | PASS — sniper L118/L252, copy L166/L698, solana L1124, all default true |
| No `dry_run=False` / DRY_RUN default flip in Wave-7 diff | PASS — grep empty |
| `safety_check_enabled` not re-enabled / `SNIPER_SAFETY_CHECK_ENABLED` untouched | PASS — default True, LIVE-refuse guard intact (L283-291); no toggle in diff |
| No secrets/keys printed or surfaced | PASS — only masked fingerprints + public addresses |
| All changed .py `py_compile` clean | PASS — 19 Wave-7 files + 4 PM-touched files all compile |

---

## PM commits this gate

1. `ea223da` [pm] CSRF token on Solana/Futures/Sniper close fetches (issues 9 + 6)
2. `eb969d0` [pm] map futures_trades real columns in analytics + unified charts (issues 2/3/7)
3. `e42de7a` [pm] telegram: re-resolve token via get_async in initialize() (carry-over C)
4. `a601289` [pm] funding panel: AI delegates to futures account (issue 15)
5. `b716f66` [pm] dedupe solana_trades in analytics portfolio total

---

## PUNCH-LIST (still open — none block DRY_RUN data correctness)

### P1 — recommended next wave (data QUALITY, not correctness)
- **B. core/engine.py ML dead-weight (DEX entry).** `EnsemblePredictor` /
  `DecisionMaker` loaded but never consulted; `ml_confidence` is the heuristic
  re-labeled; `rug_probability=0.2` hardcoded; risk-analysis FAILURE is
  REWARDED (the 20% risk weight is dropped from num+denom so an un-checkable
  token scores as if risk were perfect and can pass `min_score`); `_load_state()`
  is a no-op. The DEX stuck-position SYMPTOM is masked by `position_service.py`
  (verified — DB-first refresh is independent of `active_positions`), but the
  entry-quality problem is real. Owner: shared scorer in `core/engine.py`
  (`_calculate_opportunity_score`). Recommend a dedicated wave; do NOT hot-patch.

### P2 — operator-visibility polish
- **C. residual telegram exposure.** PM made the shared controller re-resolve
  the token via `get_async` in `initialize()` (e42de7a), which covers AI / ARB /
  COPY / FUTURES uniformly. Low severity (Telegram is a control convenience,
  not a data path). Verify on the VPS that the "TOKEN not set" line is gone for
  AI and ARB.
- **D. DEX live-but-idle health = "no health".** When DEX runs but isn't
  trading there's no heartbeat row, so the dashboard shows "no health". Cosmetic;
  add a lightweight heartbeat/last_tick surface (mirror ARB's `last_tick_at`).
- **futures_trades has no `status` column** (only stores closed trades). PM's
  analytics fix maps this to `status_filter='TRUE'`. If a future migration adds
  `status`/open rows to `futures_trades`, revisit the override in
  `analytics_routes._col_overrides` and `_unified_closed_trades`.

### Notes for the operator
- Futures DRY_RUN data will now render on /analytics and the all-7 charts
  (was silently empty due to the column mismatch — that is why issue 7's
  -$59.99/39%WR never showed). Re-check after the dashboard restart.
- Sniper / Solana modeled DRY_RUN exits are POPULATION MODELS for data
  realism, not backtests — treat modeled win rate as illustrative.
