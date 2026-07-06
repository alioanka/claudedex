# ARBITRAGE Module
## What it does
Spatial (cross-DEX) and triangular EVM arbitrage with flash-loan funding (Aave V3). Spatial is production-grade; the triangular path is currently gated behind the MB-05 atomic-receiver guard.
## Entry point
`modules/arbitrage/main_arbitrage.py` — launched as a subprocess by `main.py` when `ARB_MODULE_ENABLED=true`. Engines: `modules/arbitrage/arbitrage_engine.py` (spatial), `triangular_engine.py` (gated), `solana_engine.py` (cross-chain helper).
## Key config (DB-backed via `ConfigManager`)
- `flash_loan_amount` — flash-loan size in ETH-equivalent per leg
- `chain_config.tokens` / `routers` / `arb_pairs` — selected via `EVM_DEX_ROUTING` (MB-24)
- `flash_loan_env_key` — name of env var holding the deployed receiver contract address (resolved via `security/secrets_manager` first, env fallback)
- `rpc_url` — overrides `PoolEngine` selection when set
- `min_profit_spread` — minimum profit gate before broadcast (accepts fraction `0.005` or percent `0.5`; wired wave-2 A2-06)
- `gas_budget_usd_per_hour` — hourly rolling cap on USD gas spend per engine; new executions are refused once exceeded (default `$50`; wave-2 A2-07)
- `arb_rpc_refresh_minutes` — periodic pool_engine endpoint re-resolve cadence, 0 disables (default 15; Wave-F5 RC-A1, mig 142)
- `arb_max_price_impact_bps` — reserve-math one-way price-impact pre-filter at configured `flash_loan_amount` (default 50; Wave-F5 RC-A2, mig 142). Note mig 142 also rewrites an untouched `flash_loan_amount` default `10` → `1` ETH.
## Wallet / Account identity
All three chain engines (`ETHArbitrageEngine` / `ARBArbitrageEngine` / `BaseArbitrageEngine`) share ONE signer wallet:
- Private key: secrets-manager key `PRIVATE_KEY` (DB-backed via `security/secrets_manager`, Fernet-decrypted in `_get_decrypted_key`; env `PRIVATE_KEY` fallback). The stored `WALLET_ADDRESS` is IGNORED — the address is always DERIVED from the private key in `initialize()` (`eth_account.Account.from_key`) into `self.wallet_address`. So ETH/ARB/Base all sign from the SAME EOA; fund that one address with native gas on each chain.
- Receiver (flash-loan callback) contract, per chain (resolved via secrets_manager first, env fallback):
  - ETH: `FLASH_LOAN_RECEIVER_CONTRACT_ETH` (fallback `FLASH_LOAN_RECEIVER_CONTRACT`)
  - ARB: `FLASH_LOAN_RECEIVER_CONTRACT_ARB`
  - Base: `FLASH_LOAN_RECEIVER_CONTRACT_BASE`
  The EOA must be the owner of the receiver contract on each chain (it signs + pays gas; the contract receives the Aave callback).
- Surfaced for the dashboard: `arbitrage_runtime_stats.stats.wallet_address` + `.chain` (public address only, never the key). Visible per chain via `/api/arbitrage/diagnostics`. `null` until `initialize()` runs.

## Positions / close path (issue 6)
Spatial arbitrage is ATOMIC: each opportunity is a single flash-loan tx (borrow -> buy leg -> sell leg -> repay) that is all-or-nothing. The engine holds NO standing/closeable positions and exposes no `open_position`/`close_position`/`get_positions`. Every fill is written to `arbitrage_trades` with `status='closed'` and `entry_timestamp == exit_timestamp` at write time (`arbitrage_engine.py` ~line 2579). A reverted tx leaves no position — only burnt gas.
- DASHBOARD ACTION (for dashboard agent): the `/api/arbitrage/positions` endpoint reads `arbitrage_positions WHERE status='open'`, but the engine NEVER writes that table, so it always returns `[]`. The `positions_arbitrage.html` close buttons POST to `/api/arbitrage/position/close` and `/api/arbitrage/positions/close-all` which are NOT registered routes (404). Recommend HIDING/disabling the arbitrage close buttons rather than wiring a no-op close path. There is no failed-leg residue to clean up (atomic revert).

## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_arbitrage` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Hourly gas-budget gate: `_gas_budget_check_and_charge` (A2-07) — auto-cooldown on USD spend.
- Effect: `should_skip_live` returns `True` -> live-write gates return their `_simulate_*` path.
## Restart pattern
- `logs/.restart_arbitrage` — flag file picked up by `main.py::_restart_flag_monitor` within 5s. Wave-F5 RC-D1: the flag now explicitly CLEARS the permanent-failure latch and restart budget before dispatching `ModuleProcess.restart()`, and the health monitor evaluates the 1h-uptime budget reset in its gate path — a latched module recovers within an hour on its own instead of staying dead until a human intervenes. Same IPC shape as the kill-switch and pause flags so dashboard, scripts, and operators all share one mechanism.
- Operator-friendly diagnostic before restarting: `python scripts/arb_engine_health.py` — read-only triage that prints arbitrage_runtime_stats freshness per chain, the last fired trade, all flag-file states, the tail of `arbitrage_errors.log`, and a VERDICT line that points at the restart flag when the engine is dead.
## Logs
`logs/arbitrage/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(token_in, amount)` called inside the spatial-arbitrage execute path in `arbitrage_engine.py` (search for `# P1-06: pre-execute risk gate`). Injected via `set_risk_manager()` method on the engine.
## Per-chain cost profile (wave-2)
`CHAIN_CONFIGS[chain_id]` carries `flash_loan_gas_limit`, `fallback_gas_gwei`, `default_slippage_pct`, `flash_loan_fee_pct`, `is_l2`. Consumed by:
- `_check_arb_opportunity` net-spread gate (replaces hardcoded 0.5% A2-03)
- `_log_arb_trade` PnL accounting (replaces hardcoded $15 gas / 0.6% slippage A2-02 / A2-05)
- `_gas_cost_usd_per_tx` live gas oracle (1s cached, USD-denominated)
- `_gas_spike_multiplier` adaptive `min_profit_bps` curve (1.0–2.5x baseline)
## Live-trade readiness
AMBER (spatial code path is sound, but see the Wave-F5 honest economics verdict below: V2-only execution against 1–30 bps real divergence ≈ 0 expected edge — keep SHADOW).
Closed pre-wave: MB-03 (DAI typo), MB-04 (one-legged broadcast), MB-05 (triangular gated), secrets_manager wiring (`b20f56a`), pool_engine sweep (`a21ec41`).
Closed wave-2: **A2-01** (NameError crash in opportunity log path — every live trade was silently dropped), **A2-02 / A2-03 / A2-05** (hardcoded gas/slippage replaced with per-chain live profile), **A2-04** (receiver address via secrets_manager), **A2-06** (dashboard min-profit knob now honored), **A2-07** (hourly gas-budget tracker).
Closed wave-5: **near-miss observability** (`d2e1019` engine, `012887a` API, `4fd9b33` UI) — every rejected opportunity now emits a structured `[arb-skip] reason=<gate> profit_bps=<n>` log line + gets pushed into a rolling 50-deep deque persisted to `arbitrage_runtime_stats.near_misses` so the dashboard "Why no trades?" panel can render cross-process. Reasons covered: `raw_spread_negative` (sampled 1:120), `min_profit` (sampled 1:40), `daily_cap`, `cooldown`, `gas_budget`, `risk_manager`, `risk_manager_error`. Same commit fixed a latent dashboard-startup crash where `/api/arbitrage/diagnostics` was registered without a handler.
Triangular path remains entry-disabled by atomic-receiver guard — explicit scope cut pending contract deploy, not a defect.

## Diagnostic surfaces
- `/api/arbitrage/diagnostics` — per-chain cost profile, scan/found/executed counters, last 20 near-misses, last 10 fired trades, snapshot liveness (>10 min = `stale=true`).
- `/arbitrage/dashboard` "Why no trades?" collapsible panel — header carries top rejection reason + STALE flag without expanding; expanded view shows per-chain cards + color-coded near-miss table.
- Engine log grep: `grep '\[arb-skip\]' logs/arbitrage/arbitrage.log` returns one structured line per rejected opportunity.
- W6 subprocess health surface (`arbitrage_runtime_stats.stats`):
  - `last_tick_at` — ISO timestamp stamped at the top of every scan iteration in `run()`. A fresh `last_tick_at` with a stale `updated_at` means the engine is alive but its persist call is wedged.
  - `last_error` / `last_error_at` — error type + truncated message captured inside the run-loop `except`. Surfaces the actual cause (AttributeError, RPC drop, pool exhausted, ...) on the very next `/api/arbitrage/diagnostics` poll, no log-tailing required.
  - Persisted once at engine startup (startup-marker write) before the first 5-min `_log_stats_if_needed` tick, so a freshly-restarted engine is visible to the API within seconds rather than 5 minutes.
## Wave-F5 fixes (2026-07-06) — "zero trades for weeks" root causes
Report: `docs/agents/wave-f5/01_dex_arbitrage.md`.
- **RC-A1 RPC-auth blindness**: HTTP 401/403 during quoting was swallowed into `buy_errors` and the pair blacklisted as "no liquidity"; the endpoint was pinned at startup, so a dead Ankr ETH key left the chain quote-blind for 11 straight days while the dashboard said "Scanning". Now `_is_rpc_infra_error` (401/403/unauthorized/'api key') routes to `_handle_rpc_infra_failure`: never touches the liquidity blacklist, reports `pool_engine.report_failure`, rotates the endpoint (debounced 60s), WARNs hourly, records a `rpc_infra_error` near-miss, and the endpoint is re-resolved every `arb_rpc_refresh_minutes` regardless. `arbitrage_runtime_stats.stats.rpc_health` surfaces `endpoint_host` (host only, never the key), `infra_fail_count`, `auth_failing` (infra failure within 15 min).
- **RC-A2 economics**: mig 142 shrinks the untouched `flash_loan_amount` default 10 → 1 ETH (conditional `WHERE value='10'`); a reserve-math pre-filter skips quoting any DEX whose WETH reserve implies >`arb_max_price_impact_bps` (50) one-way impact at the configured size (derived from the existing TVL probe — zero extra RPC calls); an hourly per-chain `SPREAD DISTRIBUTION` log line (median/p25/best, sample count) makes "zero positive samples" visible at a glance.

### Honest economics verdict (operator-facing)
Executable legs are **V2-only** (the flash-loan receiver only supports V2 `swapExactTokensForTokens`; the V3 quoter is price-discovery only and a V3 best-buy is downgraded to the best V2 quote). Real cross-DEX divergence on liquid majors is **1–30 bps**, while the structural round-trip cost floor is ~65 bps+ (2×30 bps V2 fees + 5 bps Aave flash fee) BEFORE price impact and gas — the observed 3-week median best spread of **-162 bps** was 10-ETH size impact on dead V2 pools, not a formula bug. `Opportunities: 0` is the correct output of this venue set. **Expected edge ≈ 0 until (a) the V3 execution leg ships in the receiver contract and (b) quoting moves from 2s HTTP polling to event-driven** (HTTP polling against public mempools loses to MEV bots on latency; `modules/execution_gateway` exists but nothing is wired to it). Keep SHADOW and judge by whether the hourly spread summary ever prints a positive median after the Wave-F5 fixes.

## Wave-11 fixes (2026-05-28)
- **`self.chain` AttributeError** — single typo in the SLOW-SCAN warning at `arbitrage_engine.py:1754` referenced `self.chain` instead of the canonical `self.chain_name` set in `EVMArbitrageEngine.__init__`. ARB log spammed every few seconds. Renamed to `self.chain_name`.
- **Base RPC 429 / dRPC rate-limit** — when the engine's run-loop exception path sees `429` / `Too Many Requests` / `rate limit` in the error string, it now applies **jittered exponential backoff** (30s -> 60s -> 120s -> 300s) keyed off a `_rate_limit_streak` counter, AND calls `pool_engine.report_rate_limit(...)` so the offending URL is demoted in rotation. Streak resets to zero on any successful scan iter. Operator-facing log line names the chain and points at `RPC_ENV_KEY` / `RPC_PROVIDER_KEY` so the fix (provision an Alchemy/Infura/Quicknode key for Base and add it via the pool_engine .env load) is obvious. No new env key REQUIREMENTS — if no better RPC is configured, the engine just slows down instead of hammering.

## See also
- Phase 1 audit reports: `docs/agents/reports/ARBITRAGE_*.md` (smartcontract / quant / analyst).
- Wave-2 audit: `docs/agents/reports/ARBITRAGE_CAMPAIGN.md`.
- Canonical engine API: `docs/engines.md`.
