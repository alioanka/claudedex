# Wave-F5 Final Report — operator issue closure + deploy instructions

Date: 2026-07-06. Branch: `claude/friendly-ramanujan-nMWNv` (81 wave commits + 4 PM-gate commits).
PM verification gate: migrations 137-144 audited (idempotent, no number collisions, balanced
transactions, no unintended flag flips), all 48 changed Python files compile, zero duplicate
method definitions in the multi-agent hotspot `monitoring/enhanced_dashboard.py`, cross-agent
seams verified (llm_budget signatures, pool_engine rotation accessors, deleted-template
references, mig-139 key consumption), and two api-key log-leak defects found and fixed during
the gate itself (`f7ad031`, `9092796`).

Root-cause evidence for every claim below: `docs/agents/wave-f5/01..08`.

---

## Issue-by-issue closure

### 1. "DEX stopping somehow, no trades for weeks"
- **Root cause** (report 01, RC-D1): three OOM kills on 2026-06-15 exhausted `main.py`'s
  restart budget (`max_restarts=3`); the 1h-uptime budget reset lived *inside* `restart()`,
  which the permanent-failure latch prevented from ever being called again — dead for 20 days
  with 26,228 "failed permanently" log lines.
- **Fixed**: `bf92fea` (recoverable latch: `maybe_reset_restart_budget` hoisted before the
  gate + `logs/.restart_<module>` force-restart flag polled every 5s), `c7b1515` (bot
  mem_limit 4g → 6g so the OOM trigger has headroom), `28e6b56` (numpy.float64 JSON crash on
  trade close).
- **You do**: nothing beyond deploy. If a module ever shows "failed permanently" again:
  `touch logs/.restart_dex` (or any module key) — no container restart needed.
- **Watch (48h)**: `logs/orchestrator.log` — DEX should show RUNNING with restarts resetting
  after 1h of uptime; DEX dashboard banner "engine data unavailable" must be gone.

### 2. "Arbitrage zero trades"
- **Root cause** (report 01, RC-A1/A2): a dead Ankr ETH key returned 401 for 11 straight days
  and the error was swallowed into the liquidity blacklist ("no liquidity"), with the endpoint
  pinned at startup; separately, the 10 ETH scan size produced ~-162bps of price impact on
  thin V2 pools, drowning the real 1-30bps divergence.
- **Fixed**: `25b0952` (auth failures classified as INFRA: rotate endpoint, never blacklist,
  hourly WARN, `rpc_health` in runtime stats), `68b660d` (1 ETH scan size via mig 142 +
  price-impact pre-filter + hourly `SPREAD DISTRIBUTION` log), `9092796` (PM gate: redact
  endpoint URLs from the new infra-error logs — Ankr keys are path-based).
- **You do**: provision fresh EVM RPC URLs per `docs/RPC_API_KEYS_GUIDE.md`.
- **Watch**: hourly `SPREAD DISTRIBUTION` lines in `logs/arbitrage/arbitrage.log` and
  `rpc_health.auth_failing=false` on `/api/arbitrage/diagnostics`.
- **HONEST VERDICT — downgraded to AMBER**: executable legs are V2-only against a ~65bps+
  structural cost floor. Expected edge ≈ 0 until a V3 execution leg ships in the receiver
  contract. Keep SHADOW; judge by whether the spread summary ever prints a positive median.

### 3. "Futures losing money"
- **Root cause** (report 02): PF 0.55 over 691 trades with **zero** take-profit exits — TP1 at
  2.0x the SL distance was unreachable inside the 4h hold; plus dead-tape entries
  (0.09-0.24x volume), shorting into RSI 30-33, and benched bleeders re-entering after 24h.
- **Fixed**: `c96e00e` + `3e047ca` + mig 139 (`0a199ab`): `atr_tp_rr_ratio` 1.0, hard
  `min_volume_ratio` 0.25, `blocked_entry_hours_utc` 3-5, `short_min_rsi` 35, bench 48h /
  win-rate 0.48, early trailing-stop arm at +0.75%, fee-aware entry log, DRY_RUN funding
  honesty note.
- **You do**: nothing — all conditional seeds, operator overrides preserved.
- **Watch**: `/futures/performance` exit-reason mix — take-profit exits must appear at all;
  daily PnL trend. Do NOT judge before 2+ weeks of fresh DRY_RUN (see expectations below).

### 4. "AI module barely trades / P&L looks wrong"
- **Root cause** (report 02): the bot-wide 10-call LLM budget was drained by the advisor at
  midnight, starving AI to ~1 stale reading/day; separately the close path wrote the LONG
  PnL formula for SHORTs, sign-inverting every closed SHORT row (dashboard showed +$18.30;
  truth was negative).
- **Fixed**: `928695c` (cap 25→150 + per-module reservations ai=40/kap=30 + advisor
  ceiling 60), `453bfdf` (side-aware profit_loss at close + unrealized), `3dbb4eb` (mig 139
  idempotent historical SHORT backfill — recomputes from prices, safe to re-run), `b8abafe`
  (CryptoCompare key-gated, per-signal position cap = 1 — no more BTC+ETH+SOL baskets on one
  score), mig 139 exit geometry (SL -2 / TP 2.5 / 48h) + confirmation filter enabled
  (filter-only, zero LLM spend).
- **You do**: optionally add a CryptoCompare key; nothing else.
- **Watch**: `logs/ai_analysis/ai.log` cycle cadence (should be ~hourly, not daily) and
  `/ai/performance` totals now matching the trade table signs.

### 5. "Solana P&L is fake (+526,608%)"
- **Root cause** (report 03): one poisoned upstream price source (real price × ~4,900-5,200)
  in the DexScreener→CoinGecko→Jupiter chain fed unvalidated scan/entry prices; 138 closes
  ≥ +1,000%, 112 mirror -99.98% stops, 11.8% of opens poisoned.
- **Fixed**: `a9324b0` (all scan/entry prices through PriceValidator, seed/drop on
  open/close), `c2e514c` (hard price jumps quorum-gated via cross-source corroboration),
  `fa9a28e` (poisoned trades voided, not pinned), mig 140 Part A (`d28dbbb`: historical rows
  tagged `metadata.excluded` — kept for audit, filtered from every dashboard aggregate).
- **You do**: nothing.
- **Watch**: `/solana/performance` totals should drop to honest small numbers after the
  migration; grep `logs/solana_trading/` for `REJECTED` PriceValidator lines (they should
  appear — that is the guard working).

### 6. "Sniper always loses + trades page empty"
- **Root cause** (report 03): the Solana listener had been a silent zombie since 2026-06-15
  (RPC verification failed once → gave up forever; 0 pools over 16.9M scans) while the
  dashboard said healthy; the trades page JS read a nonexistent `filter-side` element →
  TypeError → permanent "0 of 0"; and the loss population was driven by `test_mode` relaxed
  safety + oversized settings.
- **Fixed**: `27a8b41`+`aac3cd3` (self-healing listener supervisor, loud `rpc_auth_failed`
  status, daily-loss halt `sniper_max_daily_loss_usd` + entry cooldown), `1339b6e` (trades
  page filter), `485c9c1` (multi-key Helius rotation in listener + safety checker), mig 140
  Part B hard tuning (liquidity 25k, taxes 5%, safety score 70, 0.05 SOL size, 25-position
  cap, 240min time-stop), `f7ad031` (PM gate: api-key redaction in listener logs).
- **You do**: **create the Helius accounts** — the listener detects nothing without a working
  key (see deploy steps).
- **Watch**: `sniper_runtime_stats.listener_status` must read `ok` (not `rpc_auth_failed`);
  `/sniper/trades` renders rows; `daily_loss_halted` / `entry_cooldown_skipped` counters.

### 7. "Wallet discovery finds only my already-tracked wallets"
- **Root cause** (report 04): not a dedup bug — every external candidate source returned
  zero rows (Helius sampling flaw: a wallet needed ≥5 swaps inside ONE 100-tx snapshot;
  Etherscan/Birdeye starved by 429s/missing keys), and discovery degraded by design to a
  local fallback built from your own tracked wallets, presented as discovery output. The v3
  discovery engine (mig 135) had never run: its flag was still false.
- **Fixed**: `80289ca` (cross-sweep fee-payer accumulation + Helius daily call budget),
  `fff2dec` (helius_tokens source), `e84a813` (smart_money_scores EVM source), `de5f678`
  (pool-address-as-wallet quarantine), `f6092c8`+`151542b` (per-call-batch key rotation +
  per-key budget scaling), `956ed04`+`c5f6a18` (per-source diagnostics + honest
  already_tracked/fallback marking in the UI), `770c862` (mig 141 conditional activation of
  discovery + shadow sim — both non-trading), `5c4a493` (AI-Trader adaptations: shadow MTM,
  crowding penalty, leader holdings reconcile — advisory only).
- **You do**: create 3-4 Helius + 2 Etherscan + 1 Birdeye accounts (deploy steps). A
  discovered wallet is still never traded without your approval.
- **Watch**: `/copytrading/discovery` — new candidates must carry a real source label, not
  `fallback`; `copy_helius_budget` day counter staying under 500.

### 8. "Polymarket not working well, no visual pages"
- **Root cause** (report 05): 70.6% of signals pinned score=1.0 (ranking nothing), momentum
  was in-play sports noise flip-flopping within minutes, the one "risk-free arb" in 20 days
  was a dead-book false positive, the live path ran DEX honeypot analysis on CLOB token ids,
  and there were no dedicated dashboard pages.
- **Fixed**: `3974ff4` (book-honesty gates, score normalization, new-market age check, flip
  cooldown; mig 137), `d931d1f` (forward-outcome tracking: price snapshots + LATE-marked
  1h/6h/24h signal outcomes), `d8bbb2d` (two-leg atomic arb lifecycle, polymarket-native
  exposure-cap risk gate replacing the DEX RiskManager call — fail-closed; shadow_mode +
  live_execution_enabled + should_skip_live chain unchanged), `a139f7b`+`d46e710` (five
  dedicated `/polymarket/*` pages incl. outcome-horizon scorecard), `1c51301` + 
  `docs/POLYMARKET_GUIDE.md` (operator guide).
- **You do**: nothing now. Live trading remains OFF behind the dual-flag chain.
- **Watch**: `/polymarket/performance` outcome scorecard filling in as horizons elapse.
- **HONEST VERDICT**: shadow-first with an edge-proof gate. Do not consider live until the
  outcome scorecard shows a signal type with positive forward returns over meaningful sample
  size (weeks of data).

### 9. "Advisor gives no advice (US / metals / FX / BIST)"
- **Root cause** (report 06): five compounding defects — a Midas `await` bug killed all fund
  advice; the risk gate rejected the ADVICE itself when the per-channel sim cap (75/75,
  saturated by 365-day sims) was full; NaN in JSON killed the advice-history/simulations
  pages; the BIST universe was 88% garbage tickers and Fonoloji price endpoints went HTTP 451;
  KAP budget-starved stamps were never retried.
- **Fixed**: `8b3ed36` (await fix), `83b0fe8` (full sim channel demotes the sim only, never
  the advice), `ddbfdf3` + mig 138 (NaN-safe JSON + row cleanup + sim horizon 365→90),
  `8a2cdac`+`a37fd7e`+`b52347c` (451 circuit breaker, borsapy source swap baked into the
  image, curated bist50 default), `3ec33f9` (KAP bounded re-queue), `ce356f8` (session cookie
  re-issue — the intermittent 401s after 1h).
- **You do**: nothing; rebuild picks up borsapy (deploy steps).
- **Watch**: advisor panel — advice rows appearing per channel again; one aggregated
  all-sources-failed WARN per cycle at most in `logs/advisor/`.

### 10. "Not sure the Intelligence & Ops modules are working correctly"
- **Root cause** (report 07): treasury produced zero output for 20 days (secrets manager
  never initialized → wallet derivation failed silently, also starving yield_treasury);
  intent_solver 403-hammered the CoW API 28,589 times; param_tuner's three tunables targeted
  modules with ~0 closed trades (rewards never fired); smart_money silently ingested nothing
  on eth/arbitrum; all 16 intel-ops dashboard badges read UNKNOWN (a lost method def).
- **Fixed**: `03f7369` (treasury secrets init), `236a3b2` (intent_solver browser UA + hourly
  dead-source backoff + mig 143 poll 60s→3600s), `8edc9df` (mig 143 param_tuner retarget at
  futures/solana knobs that close trades — shadow posture unchanged), `07ebd31` (smart_money
  hourly WARN when getLogs is blocked), `553b448` (basis_desk per-cycle venue summaries),
  `c501036` (`_aux_runtime_spec` restored — badges work again).
- **You do**: nothing.
- **Watch**: `treasury_snapshots` gaining rows; control-center intel-ops table showing real
  statuses instead of UNKNOWN; `logs/intent_solver/` quiet (hourly, not per-minute).

### 11. "Dashboard settings are a total mess"
- **Root cause** (report 08): four overlapping "home" pages disagreeing on portfolio totals
  ($8.9M vs $882 vs $1,982), the "DEX Settings" page was the old app-global settings page
  with 7 of 14 tabs not DEX at all and 1 hard-broken, plus orphan pages and dead routes.
- **Fixed**: `6fd6919` (home = `/control-center`; `/`, `/full-dashboard` 302), `4b51bf5`
  (`/module-control`, `/pro-controls` → `/modules` hub), `3a62951` (orchestrator+allocation →
  `/proposals` inbox), `dbd4446` (DEX settings rebuilt as 6 real tabs + Guide), `c96bd3a`
  (DEX standard 5-page set with 302 aliases), `983318f` (orphan/dead
  cleanup incl. legacy `monitoring/dashboard.py`), `dd249a8` (retire `/global-settings` into
  typed `/config/<type>` editors), `8fb567b` (full redirect table in `docs/dashboards.md`).
- **You do**: re-bookmark `/control-center` (old bookmarks 302 anyway).
- **Watch**: any 404 in browser devtools while navigating — all legacy paths should 302.

### 12. Multi-key API rotation (operator request)
- **Delivered**: numbered env/secrets pattern (`HELIUS_API_KEY`, `HELIUS_API_KEY_2..9`, same
  for ETHERSCAN/BIRDEYE/GOPLUS/1INCH/JUPITER keys and `<CHAIN>_RPC_URL[_N]`); each slot is
  its own rotating pool_engine endpoint with 429 cool-down; accessors
  `get_api_key()`/`report_key_*` (`f2b28d9`, `3be041b`), consumers wired in copy + sniper
  (`f6092c8`, `151542b`, `485c9c1`), BIRDEYE_API provider type for the dashboard dropdown
  (mig 144, `17a0be6`), offline unit tests (`1c099b8`), operator guide (`4a99d98`,
  `docs/RPC_API_KEYS_GUIDE.md`).

---

## Deploy steps (in order)

1. `git pull` on the VPS, then `docker compose up -d --build` — the build bakes borsapy in;
   migrations 137-144 run automatically at startup (idempotent; safe to re-run).
2. **Create the API accounts** per `docs/RPC_API_KEYS_GUIDE.md`: 3-4 Helius, 2 Etherscan,
   2 Alchemy, 1-2 Infura, 1 Birdeye. Put them in `.env` as numbered vars (or the dashboard
   `/settings/rpc-api` page) and restart. This is the single highest-leverage action in the
   whole wave — sniper detection, copy discovery, and arbitrage quoting are all currently
   key-starved.
3. Flip **nothing** else. No live flag changed; every module stays DRY_RUN/shadow.

## What to watch in the first 48h

| Surface | Healthy looks like |
|---|---|
| `logs/orchestrator.log` | all modules RUNNING; no "failed permanently" older than 1h |
| `sniper_runtime_stats.listener_status` | `ok`; pools detected > 0 |
| `logs/arbitrage/arbitrage.log` | hourly `SPREAD DISTRIBUTION` lines; no `auth_failing` |
| `/copytrading/discovery` | candidates with real source labels, not `fallback` |
| `/ai/performance` | totals match the trade table; ~hourly cycles in `ai.log` |
| `/solana/performance` | honest totals; PriceValidator REJECTED lines in logs are GOOD |
| `/futures/performance` | take-profit exits appearing in the exit-reason mix |
| Advisor panel | advice rows per channel; simulations page loads (no NaN parse error) |
| Control-center intel-ops table | statuses populated, treasury snapshots accruing |
| `/polymarket/performance` | outcome scorecard rows appearing after 1h/6h/24h horizons |

## Honest expectations (do not skip)

- **Futures + sniper tuning is hypothesis, not result.** The mig 139/140 parameters are
  derived from 20 days of losing data; they need **2+ weeks of fresh DRY_RUN** before any
  judgment, and the A/B discipline notes in the module CLAUDE.md files apply (change nothing
  else during the window).
- **Polymarket needs outcome-tracking data** before any live consideration — the scorecard
  starts empty and only LATE-marks horizons; expect 1-2 weeks before signal-type verdicts.
- **Arbitrage has ~0 expected edge until V3 execution legs ship** in the flash-loan receiver
  contract. The Wave-F5 fixes make its telemetry honest; they do not create edge.

## Known residual risks (not fixed in this wave)

1. **DEBUG-level exception logs can embed keyed URLs.** Many fail-soft `logger.debug(f"...: {e}")`
   lines (copy discovery, helius fetch paths) would include a full Helius URL if the aiohttp
   exception carries it. INFO file handlers filter DEBUG today, so exposure requires an
   operator to enable DEBUG. Proper fix: a global `logging.Filter` that redacts
   `api-key=`/path-key patterns — recommended next wave. (The INFO/WARN/ERROR-level leaks
   found during the gate were fixed: `f7ad031`, `9092796`.)
2. **`/api/rpc-pool/endpoints` returns full keyed URLs** to authenticated dashboard users.
   Pre-existing and arguably intended (it IS the key-management UI), but a `mask_urls=true`
   default would be safer if the dashboard is ever exposed beyond the operator.
3. **Pre-existing migration filename collision `002_add_config_tables.sql` /
   `002_add_module_support.sql`** (both from 2025; runner tracks by full filename so both
   apply — cosmetic, but renumbering would churn `schema_migrations`; leave as-is).
4. **mem_limit 6g is a mitigation, not a diagnosis** of the Jun-15 OOM kills. If DEX dies
   again with restarts climbing, capture `docker stats` before touching the restart flag.
5. **Import-time verification of dashboard modules was environment-limited** in the gate
   (pandas/pyotp absent in the verify sandbox); py_compile + AST duplicate-def scan passed
   on all 48 files. First container start after deploy is the real import test — watch
   `logs/dashboard/stderr.log` for the first minute.
