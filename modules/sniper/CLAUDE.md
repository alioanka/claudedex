# SNIPER Module
## What it does
New-pool / new-token sniper across EVM chains and Solana. Watches for liquidity events, runs token-safety checks, then buys with tight TP/SL.
## Entry point
`modules/sniper/main_sniper.py` — launched as a subprocess by `main.py` when `SNIPER_MODULE_ENABLED=true`. Engine: `modules/sniper/core/sniper_engine.py` (delegates to `trade_executor.py`, `evm_listener.py`, `solana_listener.py`, `token_safety.py`).
## Key config (DB-backed via `ConfigManager`)
- `trade_amount` — entry size per snipe (chain-native units)
- `slippage` — per-snipe slippage cap (default 10.0%)
- `priority_fee` — Solana compute-unit priority fee for snipe txs
- `max_buy_tax` / `max_sell_tax` — reject tokens exceeding these tax ceilings
- `min_liquidity` — minimum pool liquidity gate (chain-native)
- `take_profit_pct` / `stop_loss_pct` — read from DB at `sniper_engine.py:208-211` (no longer shadowed after MB-12)
- `target_chain` — EVM chain id / `solana` for routing
- `max_active_positions` — emergency brake; reject new candidates at `_evaluate_target` once `len(active_snipes) >= cap` (default 500, seeded by migration 016)
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_sniper` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> trade executor returns simulated fill.
## Logs
`logs/sniper/` is the single source of truth:
- `sniper.log` / `sniper.log.{1..5}` — structured logger output (10MB cap, 5 rotations)
- `sniper_errors.log` / `.{1..3}` — ERROR-level only (5MB cap, 3 rotations)
- `stdout.log`, `stderr.log` / `.{1..3}` — captured by the parent
  `RotatingLogFile` in `main.py` (10MB cap, 3 rotations)

The subprocess used to install its own `StderrToRotatingFile` that
double-wrote stderr to both `logs/sniper/stderr.log` AND fd 2 (where
the parent then captured it into `logs/sniper_module/`). Removed; the
parent's rotation is now the only stderr writer. If you see a
`logs/sniper_module/` directory on disk it's stale — safe to `rm -rf`.
## Primary risk-policy gate
Per-module local risk: `TokenSafetyChecker` (`modules/sniper/core/token_safety.py`) gates each candidate on tax / liquidity / honeypot heuristics before submission; no cross-module `RiskManager.validate_trade` call yet (P1 follow-up).
## Live-trade readiness
AMBER → GREEN candidate (volume-validated; pending production verification).
MB-11..MB-14 closed. Phase 1 WSS infrastructure shipped end-to-end on both
Solana (Pump.fun + Raydium V4) and EVM (PairCreated logs). 22 hours of
DRY_RUN data over 87,747 trades validates:
- **Volume**: WSS throughput is **2.6× polling** (63k vs 24k trades).
  Both AMMs firing via `logsSubscribe` with strict opcode pre-filter.
- **Pipeline**: end-to-end functional. No algorithmic failures.
- **Synthetic-close fallback**: 84k positions retired cleanly via
  `dry_run_no_price_feed` path; no orphan accumulation post-fix.
- **Block-time anchoring**: works at listener layer. Flag now
  propagated to `sniper_trades.metadata.block_time_anchored` so the
  underlying timing is SQL-filterable.

Per-event latency advantage is now **verifiable** via the
`detect_to_rpc_receipt_ms` marker shipped with the timing
re-architecture:
- `t_rpc_receipt` is stamped by each listener at notification arrival
  (`logsSubscribe` push for WSS, `getSignaturesForAddress` / `eth_getLogs`
  response for polling) BEFORE any `getTransaction` commitment wait.
- Delta `(t_rpc_receipt - t_detect)` = pure detection staleness
  (block production → process receipt), independent of the 3-13s
  RPC confirmation wait that previously dominated `total_ms`.
- Surfaced in `sniper_trades.metadata.timing.detect_to_rpc_receipt_ms`,
  aggregated p50/p95 per detection_path at `/api/sniper/timing`, and
  rendered on `/sniper/performance` as a highlighted yellow card.
  Historical rows (pre-marker) show "—".

WSS concurrency: first VPS run showed `detect_to_rpc_receipt_ms`
~1.9s for WSS — not the expected ~200ms. Root cause: the WSS loop
awaited `_check_pool_transaction` inline, serializing all incoming
notifications behind the prior message's 3-13s commitment wait. The
`rpc_receipt` stamp of message N+1 was therefore taken AFTER message
N's getTransaction had completed, contaminating the metric with
queue-depth delay.

Fix: `_process_wss_candidate` runs the slow `_check_pool_transaction`
+ decode + queue work as an `asyncio.create_task()` bounded by a
`asyncio.Semaphore(SNIPER_WSS_CONCURRENCY, default=16)`. The hot
WSS loop now stamps `rpc_receipt_perf`, logs the candidate, adds
to `sig_set`, and dispatches — never blocks on RPC. In-flight task
peak is exposed via `sniper_runtime_stats.solana_listener.wss_inflight_peak`
so operators can see semaphore saturation; total dispatched via
`wss_dispatched`. Tune `SNIPER_WSS_CONCURRENCY` upward if the peak
hits the cap during normal bursts (be aware of RPC rate limits).

Before going LIVE (not DRY_RUN), one DB-ops flip still required:
1. Re-enable safety filter: `safety_check_enabled=true` in DB (was
   disabled for Phase 2 volume measurement). Engine now REFUSES to
   start if `DRY_RUN=false` and `safety_check_enabled=false` — see
   `_load_settings` LIVE-trading safety guard — so a forgotten flip
   raises `RuntimeError` instead of silently buying honeypots.

Shipped pre-LIVE fixes:
- Active-positions cap (`max_active_positions`, default 500, seeded by
  migration 016). Gated at `_evaluate_target` (cheapest exit, before
  safety cost) with belt-and-suspenders gate at `_execute_snipe`.
  `sniper_runtime_stats.stats` surfaces `active_positions` and
  `max_active_positions`. Per-window cap-rejection log throttled to
  once per minute.
- Solana price fetcher for new mints. `_get_token_price` now falls
  through Jupiter Price v2 → Jupiter `/quote` (live route data, works
  as soon as a pool exists even if Price v2 has not indexed yet).
  15s per-mint cache bounds quote RPS during the 1s monitor tick.
  `jupiter_quote_fallback_hits` counter exposed in runtime stats.
  Pool-derived fallback (read AMM reserves directly) deferred — Jupiter
  quote covers Pump.fun and Raydium V4 launches in practice.
- Block-time anchoring flag (`block_time_anchored`) propagated through
  `_log_snipe_to_db` into `sniper_trades.metadata` JSONB. Detection
  paths anchored via on-chain `blockTime` are now SQL-filterable.
- LIVE-trading safety guard at `_load_settings`: raises `RuntimeError`
  if the engine starts with `DRY_RUN=false` AND
  `safety_check_enabled=false`. Belt-and-suspenders for the human DB
  flip below.

Commits delivering Phase 2 validation surface:
afbc2c5 (Phase 0 instrumentation) + 1a8010b (DB timing persistence) +
93899c2/c8debf6 (Solana WSS) + d1e6108 (EVM WSS mirror) +
e4db485 (Pump.fun WSS) + 7e4987e (strict opcode filter) +
54d26e0 (commitment fix) + e43e34d (block-time anchor) +
fe45df0 (synthetic-close) + d0890e3 (timing dashboard) +
2faa707 (docker-compose env mount + log rotation).
## See also
- Phase 1 audit reports: `docs/agents/reports/SNIPER_*.md` (smartcontract / quant / analyst).
- Canonical engine API: `docs/engines.md`.
