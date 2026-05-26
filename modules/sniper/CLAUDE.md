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
- `take_profit_pct` / `stop_loss_pct` — read from DB in `sniper_engine.py:_load_settings` (no longer shadowed after MB-12)
- `target_chain` — EVM chain id / `solana` for routing
- `max_active_positions` — emergency brake; reject new candidates at `_evaluate_target` once `len(active_snipes) >= cap` (default 500, seeded by migration 016)
- `gas_buffer_usd` — per-cap-slot gas/fee reserve used by the funding recommendation (default 0.50; see "## Wallet funding")
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_sniper` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> trade executor returns simulated fill.
## Logs
`logs/sniper/` is the single source of truth:
- `sniper.log` / `sniper.log.{1..3}` — structured logger output (10MB cap, 3 rotations)
- `sniper_errors.log` / `.{1..3}` — ERROR-level only (5MB cap, 3 rotations)
- `stdout.log`, `stderr.log` / `.{1..3}` — captured by the parent
  `RotatingLogFile` in `main.py` (10MB cap, 3 rotations)

### Log-volume control (Wave 7)
The high-frequency informational lines are now logged at DEBUG (filtered
out by the INFO file handler) so `logs/sniper/` stops growing from
per-trade / per-candidate spam:
- WSS init-candidate (`solana_listener.py`, thousands/hr) — DEBUG.
- Per-trade engine lines: `EXECUTING SNIPE`, `SNIPE SUCCESS`, `TX:`,
  `Entry value:`, `Exiting position`, `EXIT SUCCESS`, modeled-exit — DEBUG.
Kept at INFO/WARN: `TARGET ACQUIRED`, `STOP LOSS`, `TAKE PROFIT`, cap
rejections (already 1/min throttled), and all ERROR lines. The throughput
counts that those DEBUG lines used to convey are preserved in
`sniper_runtime_stats` (`wss_dispatched`, `positions_synthetic_closed`,
etc.). Main rotating handler tightened 10MB x5 -> 10MB x3 in
`main_sniper.py`. To temporarily see the DEBUG firehose, set the
`SniperEngine`/`SolanaListener` loggers to DEBUG in `main_sniper.py`.

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
- **Block-time anchoring**: works at listener layer for both chains.
  Flag propagated to `sniper_trades.metadata.block_time_anchored` so
  the underlying timing is SQL-filterable. Solana anchors via
  `getTransaction.result.blockTime`; EVM polling anchors via
  `eth.get_block(blockNumber).timestamp` with a bounded LRU cache
  (`_block_ts_cache_max=200`), and EVM WSS reuses the same cache so
  bursts get real block-time for free without blocking the WSS loop.

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

## Wave-2 enhancements (2026-05-19 audit)
Per `docs/agents/reports/SNIPER_CAMPAIGN.md`. Six residual items
closed; each in its own commit on `claude/create-expert-agents-JFSF5`:

- **R1** processed→confirmed two-stage commitment readback
  (`_check_pool_transaction`). Stage-1 `processed` typical 200-400ms,
  fall back to `confirmed` only on miss. New counters
  `processed_hit` / `processed_miss_fallback` expose hit ratio in
  `sniper_runtime_stats.solana_listener` and on the per-chain
  listener-health dashboard widget.
- **R2** safety-check exception cooldown + `safety_check_errors`
  counter. Stops the busy-loop that burned GoPlus/Honeypot.is
  rate-limit budget on token addresses that already failed once.
  Timing outcome `rejected_safety_error` distinct from
  `rejected_safety` so /api/sniper/timing can separate API outages
  from legitimate honeypot rejections.
- **R3** preserved `wss_dispatched` / `wss_inflight_peak` /
  `block_time_anchored` / `block_time_missing` across the 1-minute
  stats-window reset. Without this the dashboard counters read 0
  after the first window flip even while the WSS hot loop was
  dispatching candidates.
- **R4** dual-source honeypot quorum
  (`TokenSafetyChecker._quorum_honeypot_decision`). GoPlus +
  Honeypot.is verdicts combined: both-agree honors the verdict,
  single-source trusts the only one available, disagreement defaults
  fail-safe (treat as honeypot). Per-source verdicts tagged in the
  report with `[GP]` / `[HP]` so the dashboard surfaces which oracle
  flagged. 5 new unit tests cover all five truth-table branches.
- **R5** Solana SL/TP price-feed redundancy. Birdeye `/defi/price`
  added as tertiary fallback (Jupiter Price v2 → Jupiter /quote →
  Birdeye) so a Jupiter brown-out doesn't synthetically-close every
  active position simultaneously. New `birdeye_fallback_hits`
  counter. Pyth deliberately NOT wired — Pump.fun mints have no Pyth
  feed-id; future extension for blue-chip mints.
- **R6** per-chain listener-health dashboard widget on
  `/sniper/performance`. Side-by-side SOLANA + EVM cards: processed-
  hit ratio, WSS notifications/dispatched/in-flight-peak vs cap,
  block-time anchored ratio (EVM), top-3 rejection buckets. Pure
  consumer of `/api/sniper/stats`; no backend change required.

Commits: e4b8025 (campaign report), 87c5523 (R3),
77b22e7 (R2), 6612be2 (R1), 4adcd29 (R4 — bundled),
adee9c2 (R5), 5a0a3e9 (R6).

## Wave-3 enhancements (2026-05-19)
Per `docs/agents/reports/SNIPER_WAVE3.md`. One Wave-2-deferred item
closed; matches PM mission item "Pyth-feed wiring for blue-chip mints".

- **W3-1** Pyth Hermes blue-chip price feed. New
  `modules/sniper/core/pyth_feed.py` exposes `pyth_client` singleton
  with per-feed TTL cache (3s), process-wide 100ms throttle, 2s HTTP
  timeout, fail-soft on every error. New
  `modules/sniper/core/pyth_feed_ids.py` maps 13 Solana blue-chip
  mints (SOL, USDC, USDT, ETH, WBTC, JUP, WIF, BONK, PYTH, RAY, ORCA,
  JTO, JLP) to their Pyth feed-ids. `_get_token_price` resolution
  order on Solana is now:
      Pyth (if mapped) → Jupiter Price v2 → Jupiter /quote → Birdeye
  Feature-flag `sniper_pyth_feeds_enabled` defaults TRUE (Pyth is
  free + independent of Jupiter). Pump.fun mints have no feed-id so
  `get_pyth_feed_id` returns None and the chain falls through
  unchanged — no extra HTTP on the hot path for new launches.

  Profitability lever: kills the residual Jupiter single-point-of-
  failure cascade for blue-chip SL/TP decisions. New counter
  `pyth_fallback_hits` preserved across the 1-min stats reset.
  10 unit tests added in `tests/unit/test_sniper_new_paths.py`
  (feed-id map, hex format, parse_price, cache, helper integration,
  feature-flag gating, stats shape). 7 pure-python pass in the
  sandbox; 3 engine-import tests run on CI.

Commits: d89b1c4 (helper + ids), f2e95d3 (engine wiring — bundled
into ARB commit due to concurrent index race), 5c00192 (tests).

## DRY_RUN simulation — what it models, what it does NOT (Wave 7)
DRY_RUN is the current default and the only validated mode. Earlier the
sim had no exit price feed, so every DRY_RUN trade retired flat
(exit_price == entry_price, P&L = 0) or sold at a constant `*1.1`. That
produced the "$0.84 on every win" / identical-size / ~93%-win-rate
artifact the operator saw on `/sniper/dashboard` and `/analytics`. That
was fabricated, not collected, data. What the sim now does:

- **Entry size varies.** `_execute_snipe` jitters the entry amount ±20%
  around `trade_amount`, seeded by the mint hash (reproducible). `entry_usd`
  / `entry_price` derive from `result.amount_in`, so sizes differ per trade.
  LIVE always uses the exact configured `trade_amount` (no jitter).
- **Exit P&L is MODELED, not measured.** When the monitor loop has a real
  price (Pyth/Jupiter/Birdeye), it computes real P&L and exits on TP/SL as
  before. When the mint has NO obtainable price (the common Pump.fun case),
  `_close_position_synthetic` now draws a realistic %P&L from
  `_model_dry_run_exit_pct`: ~55% loss (clustered toward SL), ~30% chop,
  ~15% winner (toward TP), seeded by mint hash, bounded by the DB
  `take_profit_pct` / `stop_loss_pct` caps. `_simulate_sell` in
  `trade_executor.py` uses the same distribution shape for the
  has-a-price-but-DRY_RUN path. Exit price now differs from entry, P&L and
  win rate span a believable spread.
- **What it does NOT model:** real fills, real slippage/price-impact, real
  gas, MEV, or the true post-launch price path of any specific mint. It is a
  population-level outcome model for data realism, not a backtest. Treat the
  modeled win rate as illustrative, not predictive. For a true historical
  backtest, replay archived prices — out of scope here.

## Wallet / Account identity (issue 15)
Sniper is multi-chain. Keys are resolved in `sniper_engine.py` wallet init
via `security` secrets (encrypted) with `.env` fallback — never plain reads
of a private key:
- **Solana (primary, Jupiter):** private key `SOLANA_MODULE_PRIVATE_KEY`,
  public address `SOLANA_MODULE_WALLET`. **SHARED with the `solana_trading`
  module** — both modules use the exact same `SOLANA_MODULE_*` keys
  (confirmed in `modules/solana_trading/config/solana_config.py`). Funding
  this one Solana wallet funds BOTH sniper and solana_trading. Plan position
  caps and balance across both modules accordingly.
- **EVM:** private key `PRIVATE_KEY` (fallback `EVM_PRIVATE_KEY`), public
  address `WALLET_ADDRESS` (fallback `EVM_WALLET_ADDRESS`).
- The resolved PUBLIC address is surfaced (never the key) on the diagnostics
  surface `sniper_runtime_stats.stats`: `wallet_address` (Solana-preferred),
  plus explicit `solana_wallet_address` / `evm_wallet_address`.

## Wallet funding (issue 12)
With 400+ open positions the operator needs a concrete fund number.
`_compute_funding_recommendation` (snapshotted into `sniper_runtime_stats`)
exposes:
- `open_notional_usd` = `SUM(entry_usd)` over `sniper_trades WHERE status='open'`
  — capital already committed to open positions.
- `avg_entry_usd` = `AVG(entry_usd)` over the same rows (falls back to
  `trade_amount` if no open rows yet).
- `recommended_funding_usd` =
  `open_notional_usd` + `(max_active_positions - open_count) * avg_entry_usd`
  + `max_active_positions * gas_buffer_usd`.
  i.e. cover the open book, plus headroom to fill every remaining cap slot at
  the average size, plus a per-slot gas/fee reserve. Worst-case sizing so a
  fully-saturated cap can't run the wallet dry mid-snipe.
- `gas_buffer_usd_per_slot` echoes the buffer used (DB key `gas_buffer_usd`,
  default $0.50; Solana fees are ~$0.01 so this is conservative).

**Is 400+ open expected?** Yes, given the cap. `max_active_positions`
defaults to 500 and IS enforced — gated at `_evaluate_target` (cheap exit
before safety cost) and re-checked at `_execute_snipe`, both via
`_effective_active_count()` = `max(in-memory, COUNT(*) FROM sniper_trades
WHERE status='open')` so DB orphans across restarts still count. 400+ is
normal accumulation under a 500 cap when the modeled/real exits don't retire
positions as fast as new launches arrive. To reduce the standing count,
lower `max_active_positions` and/or set `max_hold_minutes` (time-stop) in DB.

## Sniper trades persistence — table contract (issues 13 + 6, for the dashboard agent)
The engine persists ONLY to **`sniper_trades`** (migration
`010_add_sniper_ai_tables.sql`). It NEVER writes the generic `trades` table.
- **Entry** (`_log_snipe_to_db`): INSERT with `status='open'` (or `'failed'`),
  `side='buy'`, `is_simulated=should_skip_live(...)`, real `entry_usd`.
- **Exit** (`_log_exit_to_db` / `_close_position_synthetic`): UPDATE the row
  to `status='closed'` with `exit_price`, `exit_usd`, `profit_loss`,
  `profit_loss_pct`, `exit_reason`.
- **Status values:** `'open'`, `'closed'`, `'failed'`. (In-memory `data['status']`
  uses `'active'`/`'buying'`/`'closed'` but those never reach the DB; DB is
  always `open`/`closed`/`failed`.)
- **Key columns** read by the dashboard: `trade_id, token_address, chain,
  side, entry_price, exit_price, amount, entry_usd, exit_usd, profit_loss,
  profit_loss_pct, safety_score, safety_rating, status, exit_reason,
  is_simulated, entry_timestamp, exit_timestamp, entry_tx_hash, exit_tx_hash,
  metadata`.

**Issue 13 (`/sniper/trades` empty):** the dashboard READ handlers
(`api_get_sniper_trades`, `api_get_sniper_positions`) already query
`sniper_trades` with the right columns/status, so the engine side is correct
and IS persisting rows. An empty page is therefore a dashboard-side
filter/render issue (HTML/JS), which the dashboard agent owns — NOT an
engine persistence gap.

**Issue 6 (close position) — ENGINE-SIDE FINDING, dashboard agent must fix
the query:** close is triggered purely via DB UPDATE from the dashboard
(`/api/sniper/position/close`, `/api/sniper/positions/close-all`) — there is
no flag-file and no in-process call to the sniper engine (it runs as a
standalone subprocess, not registered with `module_manager`). BUT those two
handlers `UPDATE trades SET status='closed' ... WHERE strategy='sniper'`,
which matches ZERO rows because the engine writes `sniper_trades`, not
`trades`. So close-all currently silent no-ops. **Fix (dashboard agent):**
change both handlers to `UPDATE sniper_trades SET status='closed',
exit_timestamp=NOW(), exit_reason='manual_close' WHERE token_address=$1 AND
status='open'` (and the close-all variant without the token filter). The
engine's in-memory `active_snipes` will retire the position on its next
monitor tick; no engine change needed for DRY_RUN.

## See also
- Phase 1 audit reports: `docs/agents/reports/SNIPER_*.md` (smartcontract / quant / analyst).
- Canonical engine API: `docs/engines.md`.
