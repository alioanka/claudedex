# SNIPER Module

## What it does
New-pool / new-token sniper across EVM chains and Solana. Watches for liquidity
events, runs token-safety checks, then buys with tight TP/SL.

## Entry point
`modules/sniper/main_sniper.py` — launched as a subprocess by `main.py` orchestrator when `SNIPER_MODULE_ENABLED=true`.
Engine: `modules/sniper/core/sniper_engine.py` (delegates to `trade_executor.py`, `evm_listener.py`, `solana_listener.py`, `token_safety.py`).

## Key config (DB-backed via `ConfigManager`)
- `trade_amount` — entry size per snipe (chain-native units)
- `slippage` — per-snipe slippage cap (default 10.0%)
- `priority_fee` — Solana compute-unit priority fee for snipe txs
- `max_buy_tax` / `max_sell_tax` — reject tokens exceeding these tax ceilings
- `min_liquidity` — minimum pool liquidity gate (chain-native)
- `take_profit_pct` / `stop_loss_pct` — read from DB at `sniper_engine.py:208-211` (no longer shadowed after MB-12)
- `target_chain` — EVM chain id / "solana" for routing

## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by every BaseModule subprocess via `core.dry_run.start_killswitch_poller`)
- Per-module: `logs/.pause_sniper` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`)
- Effect: `should_skip_live` returns `True` -> trade executor returns simulated fill.

## Logs
`logs/sniper/` — main, errors, trades (rotating handler).

## Primary risk-policy gate
Per-module local risk: `TokenSafetyChecker` (`modules/sniper/core/token_safety.py`)
gates each candidate on tax / liquidity / honeypot heuristics before submission; no cross-module `RiskManager.validate_trade` call yet (P1 follow-up).

## Live-trade readiness
AMBER. MB-11..MB-14 closed (amount_out_min, hardcoded TP/SL, `is_simulated` flag, deprecated price feed). Structural detection latency (15-120s vs competitors' 50-400ms) remains as P1.

## See also
Phase 1 audit reports: `docs/agents/reports/SNIPER_*.md` (smartcontract / quant / analyst).
