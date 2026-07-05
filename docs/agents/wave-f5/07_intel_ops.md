# Wave-F5 Report 07 — Intelligence & Ops / Advisory / Observability Modules

Scope: verify all 21 intel/ops modules are actually working (operator: "not sure Intelligence&Ops
modules are working correctly"). Evidence: `logs/<module>/` tails + pattern counts, `logs/orchestrator.log`
supervisor status, dashboard screenshots `screenshots/screencapture-*-module-<name>-*.png` (taken
2026-07-05 ~23:0x local). Log snapshot ends ~2026-07-05 19:26; "last activity" below is relative to that.

**Headline: no crash loops anywhere.** `logs/orchestrator.log` shows every subprocess RUNNING with
`Restarts: 0` (advisor: 1) through 19:26. The feared execution_gateway "crash loop" (960-line stderr)
is a false alarm — see below. Real problems: **treasury has produced zero output for 20 days**
(missing `secrets.initialize`), which also starves yield_treasury; **intent_solver ingest is 100%
dead** (CoW 403 every minute since Jun 15); **smart_money ingests nothing on ethereum/arbitrum**;
**basis_desk is unverifiable** (silent for 20 days, 0 rows); **param_tuner is reward-starved**;
**catalyst_calendar** lost its DeFiLlama source (HTTP 402).

## Status table

| Module | Verdict | Last activity (log) | Evidence |
|---|---|---|---|
| meta_controller | **WORKING** | 19:14 (15-min tick) | `meta tick: scored=7 actuated=0 autopilot=False`; panel: 14,077 meta_decisions (672/24h), 1,912 calibration rows |
| orchestrator_ai | **WORKING** | 19:23 (5-min tick) | `recommendations_inserted: 7` per tick, `errors: []`, 0 ERROR lines; solana→to_live conf=1.00, others hold (honest: 0 closed trades) |
| portfolio_allocator | **WORKING** | 18:50 (hourly) | `proposals_written: 7, reserve_pct≈10, orchestrator_budgets_found: 1, errors: []` every tick |
| regime_allocator | **WORKING** | 18:55 (hourly) | `regime=range_compression … proposals=7` every tick; BTC/ETH vol/ER inputs fresh from free klines |
| sentinel | **WORKING** | 19:26 (1-min tick) | `sentinel tick: anomalies=0 … autopilot=False`; panel: 43 anomalies total (loss_velocity, silent_module, DAI depeg) — detectors demonstrably fire; sentinel_actions=0 rows is by design (autopilot OFF) |
| param_tuner | **DEGRADED** | 18:50 (hourly) | Alive 20 days but `proposed=0` on every tick since Jun 16; only 2 proposals ever; bandit arms show `pulls: 0` |
| execution_quality | **WORKING** | 18:59 (30-min tick) | `TCA tick: modules=7 trades≈330 new_rows=7-13 bad_rows=0`; panel: 18,163 tca_trade_costs (349/24h), 7,049 scorecards. Caveat: `quote_coverage_pct=0`, fee/gas bps all 0 for solana — decomposition is degenerate (rows flow, cost model inputs missing) |
| treasury | **BROKEN** | 19:24 (ticking but idle) | 5,730 of 5,735 ticks = `no wallet addresses discovered … idle tick`; `observed=0/0` since first tick Jun 15 21:48; panel: treasury_snapshots **0 rows / 0 total ever** |
| market_data_warehouse | **WORKING** | 19:22 (5-min tick) | `ingest tick: candles=54-64 … errors=0`; funding `series` rows periodically (e.g. 15:40-16:00 series=3-6); 0 WARN/ERROR in whole log |
| catalyst_calendar | **DEGRADED** | 18:51 (hourly) | Ticks fine but sources dead: `defillama_unlocks: 0` always (HTTP 402 at startup, hard-unavailable), `binance_announcements: 0` since ~Jun 21 (was 2/tick Jun 20); only `static_macro: 1` upserted — and the static list expires 2026-12-31 |
| options_vol | **WORKING** | 19:26 (5-min tick) | BTC/ETH index/chain/ATM-IV/RV computed every tick (`[BTC] index=62676 chain=739 expiries=13`); panel: 11,578 options_vol_surface rows (572/24h). 0 suggestions is BY DESIGN: `no hedge needed (net_delta=$163, threshold=$1000)`. Only 41 warnings in 2.3MB log (transient Deribit timeouts) |
| yield_treasury | **DEGRADED** (blocked upstream) | 19:22 (15-min tick) | Every tick: `no fresh idle data (treasury module snapshots required) — idle tick`, `0/0 venues advised`. Pure downstream casualty of the treasury break — unblocks with the treasury fix |
| execution_gateway | **WORKING** (idle library, by design) | 18:26 (hourly health check) | **NOT a crash loop**: the 960-line `stderr.log` is PoolEngine hourly `Health check complete: 73 checked, ~60 healthy` INFO lines routed to stderr; 0 ERROR/Traceback; single process since Jun 15 21:48. 0 `execution_gateway_sends` rows expected — no module is wired to it yet (per root CLAUDE.md) |
| clmm_lp | **WORKING** (shadow) | 19:25 (5-min tick) | Continuous `[reject] … net_apr<10%` evaluation; 26 `[propose] … simulated (shadow_mode)` + `[close:out_of_range/max_age]` lifecycle events; panel: 16 clmm_shadow_positions. Caveats: watchlist has shrunk to ~1 solana pool + 1 eth pool; APR/IL components are absurd (fee=3,908%, il=6,387%) — sim math needs a sanity pass before anyone trusts the shadow APRs |
| intent_solver | **BROKEN** ingest (module PARKED by design) | 19:25 (1-min tick) | `CoW mainnet: HTTP 403 on /mainnet/api/v1/auction` **28,589 times — every minute since Jun 15 21:48**; UniswapX disabled by default (`uniswapx_enabled=false`); panel: intent_fill_opportunities 0 rows ever. Doc verdict was PARK IT, but as long as it runs it should not 403-spam 28k lines |
| basis_desk | **DEGRADED** (silent/unverifiable) | 19:20 (5-min tick) | 5,708 of 5,716 log lines are `Loaded 22 basis_desk settings`; ZERO engine output ([advise]/[confirming]) in 20 days; panel: basis_carry_suggestions 0 rows ever. Cannot distinguish "no carry clears the 44bps cost model" (plausible in range_compression) from "both venue fetchers failing": fetch exceptions are swallowed silently |
| stat_arb | **WORKING** (shadow) | 09:02 Jul 5 (event-driven; spread state updated hourly) | `[enter:…] … sim=True skip=shadow_mode` / `[exit:mean_revert|hard_z_stop]` events daily; panel: 73 stat_arb_trades (3/24h), 89,280 stat_arb_spread_state rows (4,275/24h). Notes: many `[exit:pair_retest_fail] z=None` (cointegration retest churn — worth tuning), every line logged twice (duplicate handler) |
| smart_money | **DEGRADED** | 19:23 (5-min tick) | base chain healthy: 4,796 swaps ingested Jul 5, 130,885 wallet_events, 570 wallet_scores. But **ethereum & arbitrum = 0 swaps across all 228 ticks on Jul 5** (`ingest[ethereum]: pairs=8 swaps=0`, `ingest[arbitrum]: pairs=1 swaps=0`), and **smart_money_signals = 0 rows ever** — the emit path has never been exercised |
| pool_engine | **DEGRADED** | 19:26 (continuous) | Serving all modules, hourly health checks + keep-alive running. But HELIUS/SOL endpoints are saturated: rate-limit counters at ~67,400 per endpoint, 3,871 `All HELIUS endpoints rate-limited` warnings in recent window; 6-18 of 73 endpoints persistently unhealthy (Monad #2, POL #4 at 11+ consecutive failures) |
| notifications | **WORKING** | 19:03 (event-driven) | Telegram `SENT` lines for futures (1,419), solana (478), dashboard (238), ai (49); 0 errors in log |

Cross-cutting cosmetic finding: every intel/ops dashboard panel shows runtime badge **UNKNOWN**
even though the supervisor shows all processes RUNNING — the dashboard's runtime-status probe does
not reach these modules' health ports / DB heartbeats for the badge. Worth one dashboard fix, since
it is exactly what made the operator "not sure the modules are working".

---

## BROKEN / DEGRADED details

### 1. treasury — BROKEN (zero output for 20 days)
- **Root cause:** `modules/treasury/main_treasury.py` never initializes the secrets manager.
  Every other module calls `secrets.initialize(db_pool)` after connecting (e.g.
  `modules/options_vol/main_options_vol.py:333`, `modules/arbitrage/main_arbitrage.py:356`);
  treasury does not (its DB init block at `modules/treasury/main_treasury.py:95-113` goes straight
  from `asyncpg.create_pool` to PoolEngine). So `_derive_evm_address()` /
  `_derive_solana_address()` in `modules/treasury/core/treasury_engine.py:107-146` call
  `secrets.get('PRIVATE_KEY')` against an uninitialized manager → None; the exception/miss is
  logged only at DEBUG (`treasury_engine.py:119`). The env fallbacks (`WALLET_ADDRESS`,
  `SOLANA_WALLET`, `SOLANA_MODULE_WALLET`) are unset in the container → `discover_wallets()`
  returns `[]` → warning at `treasury_engine.py:393` every tick.
- **Evidence:** `logs/treasury/treasury.log` — first tick Jun 15 21:48:50 already
  `no wallet addresses discovered … idle tick`; 5,730/5,735 ticks identical;
  `treasury_errors.log` is 888KB of the same line. Panel: `treasury_snapshots 0 rows / 0 total`.
- **Fix:** in `modules/treasury/main_treasury.py` after the pool is created (~line 113, after
  `logger.info("   DB pool connected …")`), add
  `from security.secrets_manager import secrets; secrets.initialize(pool)` (fail-soft try/except,
  same pattern as options_vol). Secondary: raise `treasury_engine.py:119` derivation failure from
  DEBUG to one-shot WARNING so this class of failure is visible next time.
- **Blast radius:** yield_treasury is fully idle because of this (below); the gas_low /
  hot_wallet / ledger_unbacked alert surface has been blind for the entire live-flip window.

### 2. yield_treasury — DEGRADED (starved by treasury)
- **Root cause:** requires fresh `treasury_snapshots` rows to compute idle capital; table has 0
  rows ever. Log: every 15-min tick `no fresh idle data (treasury module snapshots required)`.
- **Fix:** none in this module — fix treasury (above). Re-check after one treasury tick writes
  snapshots.

### 3. intent_solver — BROKEN ingest (and log spam)
- **Root cause:** CoW orderbook returns HTTP 403 on every call. `_JsonClient.request_json`
  (`modules/intent_solver/clients.py:42-57`) sends requests with the default aiohttp User-Agent
  and no headers; CoW's Cloudflare front rejects that (403 = blocked, not rate-limit). UniswapX
  is off by default (`uniswapx_enabled=false`, `main_intent_solver.py:151`), so CoW is the only
  source → zero data since Jun 15.
- **Evidence:** `logs/intent_solver/intent_solver.log` — 28,589 × `CoW mainnet: HTTP 403 on
  /mainnet/api/v1/auction`, one per minute from 2026-06-15 21:48 to snapshot end; panel:
  `intent_fill_opportunities 0 rows / 0 total`.
- **Fix (if the parked scaffold is kept running at all):** (a) set a real `User-Agent` header in
  `clients.py` `session.request(...)` (~line 55); (b) suppress repeats after first failure per
  run, like catalyst_calendar's `hard-unavailable; suppressing further attempts` pattern;
  (c) cheaper alternative honoring the PARK-IT verdict: set `INTENT_SOLVER_MODULE_ENABLED=false`.

### 4. smart_money — DEGRADED (2 of 3 chains ingest nothing; signal path never exercised)
- **Root cause (eth/arb swaps=0):** free-tier RPC rejects `eth_getLogs` over address+topic ranges
  (the known Ankr limitation, acknowledged in code comment
  `modules/smart_money/core/flow_engine.py:105-109`). The failure is **silent**:
  `chain_scanner.scan_chain` returns `[], cursor` when `_rpc(...)` yields None
  (`modules/smart_money/core/chain_scanner.py:301-302`) with no log line, so the tick reports
  `pairs=8 swaps=0` with no error. Arbitrum additionally has near-empty pair discovery
  (`pairs=1` vs 8-10 elsewhere) — the DexScreener liquidity/volume filter leaves ~1 pair.
- **Evidence:** Jul 5 totals — base 4,796 swaps/228 ticks; ethereum 0/228; arbitrum 0/228.
  All 570 wallet scores and all 130,885 wallet events on the panel are `base`.
  `smart_money_signals`: 0 rows ever (`signals=0` on every tick in the whole log).
- **Fix:** point ethereum/arbitrum at a getLogs-capable endpoint (Alchemy/dRPC) via
  `/settings/rpc-api` per the code's own comment — config change, not code. Code hardening:
  log a one-shot WARNING at `chain_scanner.py:301` when getLogs persistently returns None for a
  chain. Signal emission (cluster ≥3 wallets/45min, ≥2 scored ≥0.55) has never fired even on
  healthy base data — after the RPC fix, if signals stay at 0 for another week, revisit
  `cluster_min_wallets`/`min_wallet_score` (migration 133 defaults) because an emit path that
  never emits is untested code.

### 5. basis_desk — DEGRADED (20 days silent, unverifiable data path)
- **Root cause:** observability gap. `_cycle()` gathers venue quotes with
  `return_exceptions=True` and never logs the per-venue result
  (`modules/basis_desk/main_basis_desk.py:154-159`); non-actionable plans log only at DEBUG
  (`:193`). So a run where both venue fetchers throw every cycle is log-identical to a run where
  funding never clears the 44bps cost model. `venues_ok`/`quotes_evaluated` exist only in the
  in-memory `/status` stats (port 8103), which nothing captured.
- **Evidence:** `logs/basis_desk/basis_desk.log` — after startup Jun 15, 5,708 lines of
  `Loaded 22 basis_desk settings from database` and nothing else; panel:
  `basis_carry_suggestions 0 rows / 0 total`. (Zero actionable carry over 20 days of
  range_compression is plausible — |funding| must exceed ~7-15bps/interval — but it must be
  provable from logs.)
- **Fix:** add one INFO summary per cycle in `_cycle()` after the gather (~line 159):
  `logger.info("basis tick: venues_ok=%d/%d quotes=%d actionable=%d best=%s", ...)` and log
  gather exceptions at WARNING. Then re-verdict after 24h: venues_ok=0 → broken fetch;
  venues_ok=2 with 0 actionable → working-by-design.

### 6. param_tuner — DEGRADED (alive but reward-starved; no proposals since Jun 16)
- **Root cause:** the 3 seeded tunables (mig 129) target `sniper max_hold_minutes`,
  `ai confidence_threshold`, `dex min_vol_liq_ratio` — three modules with ~0 closed trades in
  every 6h reward window (orchestrator_ai confirms: "0 closed trades in the 24h window" for
  dex/ai/sniper). `compute_reward` requires `reward_min_trades` (default 5) closed trades
  (`modules/param_tuner/core/tuner_engine.py:201-209`), so rewards almost never accrue
  (`rewarded=1` a handful of times in 20 days), arms sit at `pulls: 0`, and
  `propose()` (`modules/param_tuner/core/bandit.py:154+`) keeps selecting the current arm.
- **Evidence:** every hourly tick since Jun 16: `tick: tunables=3 rewarded=0 proposed=0`;
  panel: 2 param_proposals total (both `explore`, Jun 16), bandit state `LAST_REWARD_AT`
  2026-06-16 (sniper) / never (ai, dex).
- **Fix:** register tunables for the modules that actually close trades (solana, futures) in the
  `param_tunables` registry (config/DB seed, mig-129 pattern), and/or lower
  `reward_min_trades` for the quiet modules. No code defect — the bandit is correctly refusing
  to learn from no data; it is pointed at the wrong knobs.

### 7. catalyst_calendar — DEGRADED (2 of 3 sources dead)
- **Root cause:** DeFiLlama emissions endpoint now paywalled — startup log
  `fetch https://api.llama.fi/emissions -> HTTP 402 (hard-unavailable; suppressing further
  attempts this run)` (correctly suppressed, one line per run). Binance announcements returned
  rows through Jun 20 (`binance_announcements: 2`) and 0 since Jun 21 — CMS feed/params likely
  changed; no error is logged for an empty-but-200 response.
- **Evidence:** every hourly tick since Jun 21: `collected=1 upserted=1 …
  sources={'defillama_unlocks': 0, 'binance_announcements': 0, 'static_macro': 1}`.
- **Fix:** replace the unlocks source (DefiLlama unlocks API alternative or scrape) in
  `modules/catalyst_calendar/core/`; verify the Binance CMS query still returns items and log a
  WARNING when a previously-productive source returns 0 for >48h. Reminder: `static_macro`
  fixture EXPIRES 2026-12-31 (root CLAUDE.md) — after the other two sources' death this module
  currently contributes one static row per tick.

### 8. pool_engine — DEGRADED (Helius saturation; 13-18 endpoints down)
- **Root cause:** Solana request volume (sniper WSS + solana + smart_money + treasury reads)
  exceeds the HELIUS/SOL key pool: per-endpoint rate-limit counters at ~67,000 events
  (`Rate limited: HELIUS_API - SOL #1..3 / HELIUS_API_KEY … count=67460`), and
  `All HELIUS_API endpoints rate-limited; using least-penalized …` 3,871 times in the recent
  window — the anti-starvation fallback is doing its job but every Solana consumer is being
  throttled. Separately, 6-18/73 endpoints are persistently unhealthy (`Monad #2`, `POL #4` at
  11+ consecutive failures) and never recover.
- **Fix:** add Helius keys / raise plan, or shed Solana read load (smart_money solana is already
  skipped; sniper getTransaction commitment waits are the known heavy consumer); prune or
  replace the permanently-failing Monad/POL endpoints in the DB pool so health checks stop
  re-testing dead URLs.

### 9. execution_quality — WORKING, one caveat (not counted against verdict)
- `quote_coverage_pct=0` and `avg_fee/gas/extra_bps=0` on every solana scorecard row: TCA rows
  are being written from trades that carry no quoted-cost inputs, so the "quoted-vs-realized
  decomposition" is currently realized-only. Worth wiring quote capture before using scorecards
  to green-light clmm_lp/basis_desk (their docs defer to "until TCA proves quality").

### 10. execution_gateway — WORKING; log-routing nit
- The suspected crash loop is disproven: `stderr.log` (960 lines) contains only hourly
  `PoolEngine - INFO - Health check complete: 73 checked …` lines from Jun 15 to Jul 5 —
  0 ERROR/Traceback, timestamps strictly hourly from a single process. Nit: PoolEngine's INFO
  logger writes to stderr in this module; route it to the module log to keep stderr clean.

### Minor / cosmetic
- **Duplicate log lines** in stat_arb, clmm_lp, basis_desk engines (every engine line appears
  twice): the `StatArbModule.Engine`-style sub-loggers get the same handlers as the parent AND
  propagate (`main_basis_desk.py:56-59` pattern). Set `sub_logger.propagate = False` or stop
  re-adding handlers. Doubles log volume (intent_solver's 28k-line and options_vol's 16k-line
  logs are partly this).
- **Dashboard UNKNOWN badges** on all intel/ops module panels while the supervisor shows
  RUNNING — the per-module runtime-status endpoint doesn't resolve these modules' health
  (ports 8091-8105 / DB heartbeats). This is the single highest-leverage fix for operator
  confidence in this module group.
