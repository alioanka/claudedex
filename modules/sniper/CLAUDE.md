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
AMBER (Phase 1 code-complete; Phase 2 operational validation pending).
MB-11..MB-14 closed (amount_out_min, hardcoded TP/SL, `is_simulated` flag, deprecated price feed).
Latency-reduction plan (`docs/agents/reports/SNIPER_LATENCY_PLAN.md`) shipped end-to-end:
Phase 0 instrumentation (`afbc2c5`) + DB persistence (`1a8010b`) +
Solana WSS skeleton (`93899c2`) + queue wire-up (`c8debf6`) +
EVM WSS mirror (`d1e6108`) + dashboard P50/P95 panel (`d0890e3`).
To validate Phase 2: set `SNIPER_LISTENER_MODE=wss` and
`SNIPER_EVM_LISTENER_MODE=wss` in testnet, let snipes accumulate
for ~1 week, watch `/api/sniper/timing` panel. Target: WSS P50 total
in 100-500ms range (Solana) / 50-200ms (EVM) with meaningful
`sample_count` and parity with polling on success rate. When met,
bump to GREEN candidate and retire the polling backstop.
Phase 3 (mempool watching) deferred per the plan.
## See also
- Phase 1 audit reports: `docs/agents/reports/SNIPER_*.md` (smartcontract / quant / analyst).
- Canonical engine API: `docs/engines.md`.
