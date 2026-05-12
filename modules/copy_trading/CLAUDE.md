# COPY_TRADING Module
## What it does
Mirrors on-chain trades from configured leader wallets across EVM chains and Solana. Watches for new tx hashes, decodes swap legs, and replays them at operator-capped size.
## Entry point
`modules/copy_trading/main_copy.py` — launched as a subprocess by `main.py` when `COPY_MODULE_ENABLED=true`. Engine: `modules/copy_trading/copy_engine.py` (combined leader-watcher + `CopyTradeExecutor`).
## Key config (DB-backed via `ConfigManager`; leader list persisted in DB)
- `targets` — list of `(chain, wallet_address, label)` leader rows loaded at `_load_settings`
- `max_copy_amount` — USD-equivalent ceiling per copied trade (default 100.0)
- `copy_ratio` — percent of leader size to mirror (default 10)
- `_wallet_cooldown_seconds` — per-leader rate limit (default 300)
- `solana_rpc_url` / `web3_provider` — resolved via `RPCProvider` / `PoolEngine`; `.env` fallback honoured
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_copy_trading` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` at `copy_engine.py:263` -> `_simulate_solana_swap` / EVM equivalent.
## Logs
`logs/copy_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
Per-module local risk: `max_copy_amount` cap + per-leader cooldown enforced inline in `copy_engine.py`. Engine now exposes `get_positions()` / `close_position()` so emergency-stop reaches COPY positions (MB-25). No cross-module `RiskManager.validate_trade` call yet (P1 follow-up).
## Live-trade readiness
AMBER. MB-22 (fake Solana SELL), MB-23 (`* 1e18` unit-bug), MB-24 (per-chain DEX routing), MB-25 (emergency-stop integration) closed; BaseModule conversion done.
## See also
- Phase 1 audit reports: `docs/agents/reports/COPY_TRADING_*.md` (quant / analyst / backend).
- Canonical engine API: `docs/engines.md`.
