# COPY_TRADING Module
## What it does
Mirrors on-chain trades from configured leader wallets across EVM chains and Solana. Watches for new tx hashes, decodes swap legs, and replays them at operator-capped size.

**Wave-2 (2026-05-19) — quant rebuild.** Discovery layer rebuilt from scratch (operator-flagged broken). New `wallet_discovery.py` + `leader_scorer.py` modules + `copy_leader_scores` table (migration 024). Engine now wires Kelly-fraction sizing per leader (feature-flagged off) and emits structured replay-decision diagnostics at every gate.
## Entry point
`modules/copy_trading/main_copy.py` — launched as a subprocess by `main.py` when `COPY_MODULE_ENABLED=true`. Engine: `modules/copy_trading/copy_engine.py` (combined leader-watcher + `CopyTradeExecutor`). Discovery + scoring: `wallet_discovery.py` + `leader_scorer.py` (called from the dashboard and `scripts/refresh_copy_leaders.py`).
## Key config (DB-backed via `ConfigManager`; leader list persisted in DB)
- `targets` — list of `(chain, wallet_address, label)` leader rows loaded at `_load_settings`
- `max_copy_amount` — USD-equivalent ceiling per copied trade (default 100.0)
- `copy_ratio` — percent of leader size to mirror (default 10)
- `_wallet_cooldown_seconds` — per-leader rate limit (default 300)
- `solana_rpc_url` / `web3_provider` — resolved via `RPCProvider` / `PoolEngine`; `.env` fallback honoured
- **Wave-2 new tunables (`config_settings.copytrading_config`):**
  - `kelly_sizing_enabled` (bool, default `false`) — when on, `_get_leader_kelly` multiplies the `max_copy_amount` cap by the leader's persisted `kelly_fraction` (0..0.25)
  - `kelly_probation_fraction` (float, default `0.05`) — fraction used for unscored / stale-scored leaders; clamped 0..0.25
  - `kelly_staleness_days` (float, default `7.0`) — score-row TTL before falling back to probation
  - `leader_score_weights` (JSON, optional) — operator override for `leader_scorer.DEFAULT_WEIGHTS`
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_copy_trading` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` at the top of `copy_solana_swap` / `copy_evm_swap` in `copy_engine.py` -> `_simulate_solana_swap` / `_simulate_evm_swap`.
## Logs
`logs/copy_trading/` — main, errors, trades (rotating handler). Wave-2 added a `[replay]` prefix on every gate decision (cooldown / position_cap / unsupported_chain / risk_gate / no_token_extracted / amount_too_small / no_position_to_close / success / error) — greppable for forensics.
## Primary risk-policy gate
- Cross-module: `core.risk_manager.RiskManager.validate_trade(token, amount)` called in both `copy_solana_swap` and `copy_evm_swap` at `copy_engine.py` immediately after the `should_skip_live` check, before broadcast. Solana amount is `lamports / 1e9` (SOL units); EVM amount is `wei / 1e18` (ETH units) so the gate sees consistent native-token magnitudes across both paths. Injected via `set_risk_manager()` on the executor by the outer `CopyTradingEngine.initialize`; fail-soft if RiskManager construction fails.
- Per-module local: `max_copy_amount` cap + per-leader cooldown + global open-position cap (`max_active_positions`, default 50, seeded by migration 017) enforced inline. `_at_position_cap` runs a bounded `SELECT COUNT(*)` against `copytrading_trades WHERE status='open'` just before each BUY broadcast; SELLs are never gated because they reduce exposure. Engine exposes `get_positions()` / `close_position()` so emergency-stop reaches COPY positions (MB-25).
- **Wave-2 Kelly-fraction layer (feature-flagged):** `_get_leader_kelly(chain, wallet)` reads `copy_leader_scores.kelly_fraction` (computed by `leader_scorer.compute_score`, quarter-Kelly cap mirrors `core/decision_maker.py:712`). Multiplies the operator-set USD cap by this fraction so a 100-trade Sharpe-2 leader gets full size while a 0-history wallet pasted from Twitter gets the probation fraction.
## Wave-2 discovery + scoring pipeline
- `modules/copy_trading/wallet_discovery.py` — 5-source rate-limited sweep (DexScreener, Helius, Birdeye, GMGN, on-chain `copytrading_trades`). Mock-mode flag for offline tests. Network-safe (every fetch bounded by `asyncio.wait_for` + try/except). Always returns within `request_timeout_s` even with zero keys configured.
- `modules/copy_trading/leader_scorer.py` — pure-function scoring: walk-forward 30d/90d PnL + annualised Sharpe + Bayesian-shrunk hit rate + bell-curve hold-time + cumulative-PnL drawdown → bounded 0..100 composite. Quarter-Kelly fraction from shrunk hit-rate and 1.5 payoff assumption.
- `migrations/023_copy_leader_scores.sql` — persistent score cache with `(chain, wallet_address)` UNIQUE + `score DESC` index.
- `scripts/refresh_copy_leaders.py` — operator-facing entrypoint; safe for cron. Honors Docker secrets / env. Supports `--mock` and per-chain filtering.
- Dashboard: `GET /copytrading/leaders` page + `GET /api/copytrading/leaders` JSON + admin-only `POST /api/copytrading/leaders/refresh` to trigger a sweep.
## Live-trade readiness
AMBER → GREEN candidate (pending production verification). MB-22 (fake Solana SELL), MB-23 (`* 1e18` unit-bug), MB-24 (per-chain DEX routing), MB-25 (emergency-stop integration) closed; BaseModule conversion done; secrets_manager wiring (`b20f56a`) and pool_engine sweep (`a21ec41`) landed. Wave-2 discovery rebuild + Kelly sizing + replay diagnostics shipped 2026-05-19.
## See also
- Phase 1 audit reports: `docs/agents/reports/COPY_TRADING_*.md` (quant / analyst / backend).
- Wave-2 quant audit: section 2 of `docs/agents/reports/COPY_TRADING_quant.md` — the CT-Q-01 / CT-Q-02 backlog drove the rebuild.
- Canonical engine API: `docs/engines.md`.
