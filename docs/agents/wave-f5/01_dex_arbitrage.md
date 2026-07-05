# Wave-F5 Root-Cause Report: DEX stopped for weeks / Arbitrage zero trades

Analysis window: runtime logs 2026-06-15 → 2026-07-05 (committed by operator), dashboard screenshots 2026-07-05.
Read-only analysis; no code changed.

---

## 1. DEX module: "stopping somehow, no trades for weeks"

### Verdict
The DEX subprocess has been **dead since 2026-06-15 22:53 UTC** (20 days at screenshot time). It was killed
(most likely container-cgroup OOM) three times on 2026-06-15, exhausted `main.py`'s restart budget
(`max_restarts = 3`), and hit an **unrecoverable permanent-failure latch** in the orchestrator. It was never
restarted again. The 2026-06-15 EnsemblePredictor fix (commit `4e2c187`) **is deployed and working** — the
stopping cause is different.

### Timeline (from `logs/orchestrator.log` + `logs/dex_trading/*`)

| Time (2026-06-15) | Event |
|---|---|
| 21:48:44 | Orchestrator starts all 28 modules (`✅ Started 28/28 module(s)`) |
| 21:50:44 | `⚠️ DEX Trading is not running!` → restart #1 (PID 225). **Financial Advisor died at the same moment** (21:50:50) |
| 21:52:55 | `⚠️ DEX Trading is not running!` → restart #2 (PID 355) |
| 21:53→22:52 | PID 355 healthy: 3 DRY_RUN trades (SPACEMOON, MIZU), `ML[ensemble] conf=1.000` scoring lines, position monitoring. MEM ≈ 1058 MB |
| 22:52:53 | `⚠️ DEX Trading is not running!` → restart #3 (PID 1692). **Financial Advisor died again at the same time** |
| 22:53:17–22:53:47 | PID 1692 initializing; last app log 22:53:28 (`HoneypotChecker initialized`); last stderr = TensorFlow loading (`cudart_stub.cc` at 22:53:47). Then **silence — no exception, no shutdown log** |
| 22:53:54 | Orchestrator status report: `DEX Trading - STOPPED` |
| 22:54:19 | `❌ DEX Trading has failed permanently` — repeated **every 60s for 20 days (26,228 lines)** through 2026-07-05 19:28 |

Dashboard confirms (screenshots 2026-07-05): DEX dashboard banner "Standalone Dashboard - Historical Data
Only / DEX engine data unavailable"; 3 orders stuck `OPEN` since Jun 15 (SPACEMOON ×2, MIZU); newest closed
positions Jun 14.

### Root causes, ranked by confidence

**RC-D1 (HIGH): permanent-failure latch is unrecoverable — the restart-budget reset can never fire once the cap is hit.**
- `main.py:282` — `self.max_restarts = 3`.
- `main.py:467` — `RESTART_COUNT_RESET_AFTER = 3600` (1h uptime resets budget). The reset lives **inside**
  `restart()` (`main.py:468-476`).
- `main.py:823` — the health monitor gates **before** calling restart:
  `if module.restart_count < module.max_restarts: await module.restart() else: "failed permanently"`.
  Once `restart_count == 3`, `restart()` is never invoked again, so the uptime-based reset is unreachable
  forever. Dead module stays dead until a human intervenes.
- Near-miss detail: PID 355 was stable **3593 s** (21:53:00 → 22:52:53) — **7 seconds short** of the 3600 s
  reset window. Financial Advisor, stable 3805 s, got `♻️ Financial Advisor stable for 3805s — resetting
  restart budget from 1` (orchestrator.log 22:54:19) and survived; DEX did not.
- The one built-in recovery path — `logs/.restart_dex` flag handled by `_restart_flag_monitor`
  (`main.py:911-952`), which calls `module.restart()` directly and *would* reset the budget (uptime now ≫ 1h)
  — was **never used**: `grep -c "Restart flag" orchestrator.log` = 0.

**RC-D2 (MEDIUM-HIGH): the underlying kills are container OOM during ML/TensorFlow startup.**
- All three deaths are **silent** (no traceback in TradingBot.log / stderr; 252-line stderr ends mid-TF-init)
  — consistent with SIGKILL, not a Python crash.
- Deaths #1 and #3 killed **DEX and Financial Advisor simultaneously** — the two largest non-dashboard
  processes (DEX 1058 MB, Advisor 887–1040 MB, Dashboard 805 MB). Correlated multi-process silent death is
  the OOM-killer signature (kernel picks the biggest RSS in the cgroup).
- `docker-compose.yml:86` — `mem_limit: 4g`, with a comment sizing it for "the current SNIPER+ARBITRAGE
  workload (steady-state ~1.3 GB)". The deployment now runs **28 subprocesses**; the 21:53 status report sums
  to ≈4.3–4.8 GB with DEX's TF/Torch/sklearn import spike on top. Deaths #1/#2 happened ≤120 s after start —
  exactly the TF model-load window; death #3 (22:53:47) was mid-TF-load.
- Verification for operator: `docker inspect trading-bot --format '{{.State.OOMKilled}}'` won't show child
  kills; check `dmesg | grep -i "killed process"` on the VPS or `memory.events` in the container cgroup.

**RC-D3 (CONFIRMED, secondary): closed-trade DB updates fail with `numpy.float64` serialization error.**
- `TradingBot_errors.log` 22:18:51 & 22:28:23:
  `❌ Failed to update trade in database: Type is not JSON serializable: numpy.float64` —
  `core/engine.py:2505 (_close_position)` → `data/storage/database.py:470 (orjson.dumps(value))`.
- Effect: DRY_RUN closes don't persist their update → contributes to the stuck-`OPEN` rows visible on the
  dashboard and skews trade-history stats. Not a stopping cause, but every close hits it.

**Ruled out:** the EnsemblePredictor load crash (`_initialize_default_models` / LSTM input_dim). Run 3 logs
show the ensemble loading and scoring (`🤖 ML[ensemble] conf=1.000 pump=… rug=…`, 87 mentions), zero
ensemble-related tracebacks. Commit `4e2c187` reached the deployment.

### Fixes (ranked by ROI)

1. **`main.py` health monitor — make the latch recoverable** (~5 lines): at `main.py:820-826`, always call
   `await module.restart()` and let `restart()` itself apply the reset-then-cap logic (it already does, in
   the right order at lines 468-480). Additionally: log "failed permanently" **once** with a
   state-transition flag, not every 60 s (26k spam lines); consider `max_restarts` 3→5 and
   `RESTART_COUNT_RESET_AFTER` 3600→1800 so a 1-hour-stable module can't miss the window by seconds.
2. **Immediate operator recovery (no code):** `touch logs/.restart_dex` inside the container — the
   `_restart_flag_monitor` path bypasses the health-monitor gate and will reset the budget (uptime > 1h).
3. **Memory headroom:** raise `docker-compose.yml` `mem_limit` to 8g (VPS permitting) **or** disable the
   default-OFF expansion modules that are currently enabled (orchestrator shows all 28 running), **and**
   stagger heavy-module starts (DEX + Advisor both import ~1 GB of ML libs at t=0). Longer-term: lazy-load
   TF in `modules/dex_trading` so a restart doesn't spike.
4. **Serialization fix:** in `data/storage/database.py:470` use
   `orjson.dumps(value, option=orjson.OPT_SERIALIZE_NUMPY)` (or cast in `core/engine.py:2505`), then
   reconcile the 3 stuck `OPEN` trades from Jun 15.
5. **Alerting:** surface "module latched permanently" as a dashboard banner + sentinel anomaly (the
   `silent module` detector evidently didn't page anyone for 20 days — verify sentinel covers DEX).

---

## 2. Arbitrage module: "not even 1 trade for weeks"

### Verdict
The process has been **up and scanning the whole time** (PID 18, Restarts: 0, Jun 15 → Jul 5). Zero trades is
the product of two independent root causes: **(1)** the pinned Ankr Ethereum RPC key returns
`401 Unauthorized` for ~60% of the window and the engine misclassifies that as "no liquidity" instead of
rotating endpoints; **(2)** even in the healthy windows, the strategy as configured has a **structural cost
floor of ~162 bps against real cross-DEX divergence of 1–30 bps** — every scanned spread in 3 weeks was
negative, so the funnel dies before any (even simulated/DRY_RUN) trade is recorded. Recording simulated
trades only happens after an opportunity passes the gates — last simulated trade: **2026-05-28** (dashboard).

### Evidence — the funnel (from `logs/arbitrage/arbitrage.log`, 70k lines)

Funnel counts over Jun 15 → Jul 5:
- 22,647 five-minute STATS lines — **every single one** `Opportunities: 0 | Executed: 0`.
- 12,938 of them report `Pairs w/Liquidity: 0/N` (RPC-blind periods).
- 2,006 `Best spread:` samples: min −3.72%, p25 −1.633%, median −1.625%, p75 −1.622%, **max −1.618% — zero
  positive samples in 3 weeks**.
- 695 `[arb-skip] reason=raw_spread_negative` lines (sampled 1:120 → ≈83k actual rejections);
  e.g. `2026-06-15 21:54:11 [arb-skip] reason=raw_spread_negative pair=WBTC/WETH buy_dex=uniswap_v3
  sell_dex=uniswap_v2 profit_bps=-162.97`.
- Dashboard "Why no trades?" panel: **top reason `thin_pool_artifact` (75,960×)** — the >500 bps
  upper-spread sanity cap (`arbitrage_engine.py:2563-2576`) rejecting garbage quotes from near-empty pools.
- `arbitrage_trades.log` is empty; last recorded (simulated) trade on the dashboard: 28.05.2026.

### Root causes, ranked by confidence

**RC-A1 (CONFIRMED): Ankr Ethereum RPC key `47ce353a…` is dead/over-quota for most of the window; the
engine treats HTTP 401 as "no liquidity" and never rotates the endpoint.**
- 5,043 `401 Client Error: Unauthorized` lines. Per-day pattern is a **monthly-quota signature**:
  - healthy: Jun 15–19 (liquidity present every day)
  - 401s start **Jun 19 13:29**, continuous through **Jun 30** (`Pairs w/Liquidity: 0/N` for 11 straight days)
  - healthy again **Jul 1–Jul 4** (quota reset; e.g. Jul 4 `Pairs w/Liquidity: 23/23`, `42/42`)
  - 401s resume **Jul 4 11:44**, fully blind Jul 5. → the key's monthly credits burn out in days.
- `arbitrage_errors.log` (2 lines total, Jun 19 + Jul 4):
  `❌ RPC connection may be dead: 401 Client Error: Unauthorized for url: https://rpc.ankr.com/eth/47ce353a…`
- Note: DEX's honeypot checker uses a **different, working** Ankr key (`85e89188…`) — the arb key is
  separately provisioned in pool_engine/DB and is the broken one.
- Code path: `main_arbitrage.py:158/183/206` fetches the RPC **once at startup** via
  `RPCProvider.get_rpc('ETHEREUM_RPC')` and `initialize()` pins it (`arbitrage_engine.py:1771-1773`).
  Per-DEX quote failures are swallowed into `buy_errors` (`arbitrage_engine.py:2409-2413`), logged as
  `⚠️ [SUSHI] No buy prices from any DEX` (line 2432), then **blacklisted as a liquidity problem**
  (`arbitrage_engine.py:2434-2436` → `⛔ [SUSHI_liquidity] Blacklisted for 60min (no liquidity)`).
  Endpoint rotation exists but only fires on `'429' / 'too many requests' / 'rate limit'` in the main-loop
  exception handler (`arbitrage_engine.py:2005-2011` → `_rotate_rpc_endpoint`, line 2049). A 401 during
  quoting never reaches it, never calls `pool_engine.report_failure`, never rotates. The module reports
  "Trading Active / Scanning" on the dashboard while effectively blind for 11 consecutive days.

**RC-A2 (HIGH): strategy economics — the scanned venue set structurally cannot produce a positive round-trip.**
- `Best spread` is the full round-trip (`arbitrage_engine.py:2597-2612`): borrow `flash_loan_amount`
  (default **10 ETH**, `arbitrage_engine.py:1071-1083`) → buy leg → sell leg → minus Aave 5 bps. It embeds
  2×30 bps V2 fees + 5 bps flash fee + **round-trip price impact of 10 ETH on thin V2 pools** ≈ constant
  −162 bps. Median −1.625% with p25–p75 spread of only 1 bp over 3 weeks + repeated
  `⚠️ STALE DATA: Best spread unchanged for 15min` warnings = the scanned uniswap_v2/sushiswap (+baseswap,
  camelot) pools for majors are **dead pools with no flow**; real cross-DEX divergence on liquid pairs is
  1–30 bps (the engine's own comment, line 2489).
- Executable legs are **V2-only**: the V3 quoter is price-discovery only, enabled solely on Ethereum
  (`arb_v3_quoter_enabled_arbitrum/base: false` in the startup settings dump, arbitrage.log line 10), and
  `_check_arb_opportunity` deliberately downgrades a V3 best-buy to the best V2 quote
  (`arbitrage_engine.py:2455-2470`). Where the liquidity actually lives (V3 / aggregators / Aerodrome
  concentrated), the module cannot execute.
- Conclusion: `Opportunities: 0` is the *correct* output of this configuration. The gates
  (`min_profit_spread 0.2%`, `min_net_spread_bps_* = 0`, econ dormancy) are not the blockers — nothing ever
  gets near them. This is why DRY_RUN records nothing: simulated fills are only recorded downstream of an
  opportunity passing `raw_spread` + cost gates.

**RC-A3 (MEDIUM, confirmed by dashboard): triangular path disabled.** Dashboard shows `Triangular DISABLED`
badge; per root CLAUDE.md this is the documented scope cut (atomic-receiver contract gate), and
`TRIANGULAR ARB STATS` lines show `Opportunities: 0` as well. Not a defect, but it removes the only path
that doesn't depend on cross-venue V2 divergence.

### Fixes (ranked by ROI)

1. **RPC auth-failure handling (loss-surface + fill-rate):** in `_check_arb_opportunity`
   (`modules/arbitrage/arbitrage_engine.py:2409-2436`), inspect `buy_errors` for `401/403/Unauthorized`:
   (a) do **not** call `_update_liquidity_blacklist(..., has_liquidity=False)` for auth errors —
   blacklisting a market for an infra failure poisons 60 min of scanning per pair;
   (b) call `pool_engine.report_failure(...)` and `_rotate_rpc_endpoint()` (line 2049) immediately;
   (c) extend the main-loop matcher (`arbitrage_engine.py:2005-2011`) to include `'401'`,
   `'unauthorized'`, `'403'`. Also add a periodic (e.g. hourly) re-resolve of the endpoint from pool_engine
   instead of pinning one URL for the process lifetime — this is the pool_engine contract the rest of the
   repo follows.
2. **Replace the dead key (operational, no code):** remove/replace Ankr key `47ce353a…` for
   `ETHEREUM_RPC` (and the sushiswap/baseswap-hitting ARB/BASE entries) via the dashboard RPC/API config;
   seed at least one non-Ankr fallback per chain in pool_engine. Budget check: current scan cadence
   (`scan_interval: 2`s × 3 chains × N pairs × N routers × 2 legs, individual `eth_call`s) burned a monthly
   quota in ~19 days, then in 4 days after reset — batch quotes via Multicall3 to cut RPC calls ~10×
   before buying a bigger plan.
3. **Make the economics winnable or park the strategy:**
   - Reduce scan/borrow size: DB `flash_loan_amount` 10 → 1 ETH cuts round-trip price impact several-fold
     (the −162 bps floor is size-dependent); the notional gate math still holds.
   - Enable the V3 execution leg (Wave-18 TODO in code, `arbitrage_engine.py:2455-2470` caveat block) and
     flip `arb_v3_quoter_enabled_arbitrum/base` to true — V2-only execution on 2026 liquidity is the core
     structural problem.
   - Honest assessment for the operator: HTTP-polling spatial arb on majors against public mempools loses
     to MEV bots on latency; without the atomic receiver + private order flow
     (`modules/execution_gateway` exists but nothing is wired to it), expected LIVE edge is ≈0. Consider
     keeping it SHADOW and judging by whether *any* positive raw spread ever appears after fixes 1–3.
4. **Observability honesty:** dashboard "Trading Active / Scanning" while 100% of quotes 401 is misleading.
   Emit a distinct `rpc_auth_failure` near-miss reason (`_record_near_miss`, `arbitrage_engine.py:1400`) and
   a sentinel anomaly when `Pairs w/Liquidity == 0` for > N hours, so an 11-day blind spell pages someone.

---

## 3. Tuning / enhancement notes

- **`main.py` status spam:** 26,228 identical "failed permanently" ERROR lines and per-minute "not running"
  warnings buried the one signal that mattered. Rate-limit repeated module-state errors to state
  transitions.
- **Restart-budget window:** 3600 s reset vs 60 s health-check granularity means a module that dies hourly
  (like an OOM-cycling DEX) sits exactly on the boundary — DEX missed it by 7 s. Reset on `uptime >
  RESET_AFTER - health_check_interval` or track cumulative stable time.
- **Arb blacklist semantics:** `⛔ [X_liquidity] Blacklisted for 60min (no liquidity)` fires on *any* quote
  failure; split into `no_liquidity` vs `rpc_error` buckets so stats (`Pairs w/Liquidity: 0/N`) stop
  conflating market state with infra state.
- **DEX stuck-OPEN reconciliation:** add a startup reconcile in `modules/dex_trading` (mirror the sniper
  restart-reconcile pattern from mig 107) so orders left `OPEN` by a killed process are closed/flagged on
  next boot.
- **Capacity:** with 28 subprocesses in one 4 GB container, any module restart that imports TF is a
  Russian-roulette OOM. Either raise the ceiling, trim enabled modules, or move DEX (the only TF-heavy
  trading module) to its own memory budget.
