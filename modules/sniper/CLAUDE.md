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
- **Block-time anchoring**: works at listener layer. Flag propagation
  to `sniper_trades.metadata` pending (5-line follow-up fix; underlying
  timing IS anchored, just not SQL-filterable).

Per-event latency advantage is **unverifiable** with current timing
markers: `getTransaction` commitment='confirmed' wait (~3-13s) dominates
end-to-end measurement (`detect→broadcast_done`). Real detection-time
delta (~100ms WSS vs ~7.5s polling avg cycle) is buried under the
RPC-confirmation noise. Isolating it requires re-anchoring t_detect on
RPC-receipt time, not wall-clock — separate future work.

Before going LIVE (not DRY_RUN), three production-grade fixes required:
1. Solana price fetcher for new mints (Jupiter quote + pool-derived
   fallback). Without this, the `_check_pool_transaction`
   "no price feed" path keeps firing for new Pump.fun launches.
2. Sniper active-positions cap (e.g. `max_active_positions=500`) as
   emergency brake against runaway accumulation seen in DRY_RUN
   stress test (10k+ positions in 22h).
3. Re-enable safety filter: `safety_check_enabled=true` in DB (was
   disabled for Phase 2 volume measurement).

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
