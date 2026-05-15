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
`logs/sniper/` — main, errors, trades (rotating handler).
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

Per-event latency advantage is **unverifiable** with current timing
markers: `getTransaction` commitment='confirmed' wait (~3-13s) dominates
end-to-end measurement (`detect→broadcast_done`). Real detection-time
delta (~100ms WSS vs ~7.5s polling avg cycle) is buried under the
RPC-confirmation noise. Isolating it requires re-anchoring t_detect on
RPC-receipt time, not wall-clock — separate future work.

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
