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

## Wave-4 (2026-05-20) — risk-gate hardening (CT-Q-09 + CT-Q-12)
- **CT-Q-09 per-leader probation table.** Migration `030_copy_probation_gate_defaults.sql` plus `_is_leader_on_probation` + `_maybe_set_probation` helpers in `copy_engine.py`. Engine BUY path consults `copy_leader_scores.on_probation` + `probation_until` -- if benched, refuses the BUY with `[replay] reason=probation`. SELLs are NEVER gated (they reduce exposure). Auto-trigger: a closed mirrored trade with PnL <= `-probation_loss_pct_threshold`% benches the source leader for `probation_days`. Score-based auto-bench in `leader_scorer.upsert_score` (CT-Q-09b): scores < `probation_score_threshold` with >=10 trades also trigger. Re-entry is automatic on `probation_until` expiry. New flags (all `copytrading_config`, also accept bare keys for backwards compat):
  - `copy_probation_gate_enabled` (bool, default `true`)
  - `copy_probation_score_threshold` (number, default `30`)
  - `copy_probation_loss_pct_threshold` (number, default `25`)
  - `copy_probation_days` (number, default `7`)
- **CT-Q-12 cross-module exposure aggregator.** New `modules/copy_trading/exposure_aggregator.py` (`get_exposure_usd`, `get_exposure_breakdown_usd`). Sums per-token open-position USD across DEX (`trades.usd_value`), SNIPER (`sniper_trades.entry_usd`), SOLANA (`solana_positions` if it has `entry_usd`), COPY (`copytrading_trades.entry_usd`), AI (`ai_trades.entry_usd`) for the same chain. Engine BUY path consults via `_check_cross_module_exposure`; refuses if `existing + intended > cap` with `[replay] reason=cross_module_cap` and a per-module breakdown in the extra dict. Default cap **$5000**. Fail-soft: any DB error returns (allow=True, 0.0, {}) so the existing per-module caps remain the safety net. New flags:
  - `copy_cross_module_exposure_check_enabled` (bool, default `true`)
  - `copy_cross_module_exposure_cap_usd` (number, default `5000`)

Replay-log gate vocabulary expanded to: `probation`, `cross_module_cap` (in addition to Wave-1/2/3 gates `cooldown`, `position_cap`, `unsupported_chain`, `risk_gate`, `no_token_extracted`, `amount_too_small`, `no_position_to_close`, `success`, `error`).

### Wave-4 carry-over (not shipped this wave)
- `solana_positions.entry_usd` column. The aggregator currently noops on Solana open positions because the table has no USD basis (only `amount_sol`). SOLANA module owns the migration; until then the aggregator falls back to summing only the SOL-closed trades stored in `copytrading_trades` and `solana_trades` (which DO have USD basis).
- Dashboard surfacing of the new flags. The settings page should expose the four probation knobs + two exposure knobs as operator-tunable fields. Engine consumes via `ConfigManager` already; UI lift remains for the dashboard agent.

## Wave-5 (2026-05-20) — dashboard PnL surfacing
**Problem.** Operator reported that every Copy Trading page (`/copytrading/discovery`, `/copytrading/wallets`, `/copytrading/performance`, `/copytrading/trades`, `/copytrading/dashboard`) showed `Total PnL +$0.00` even though they had 5 open mirrored positions on 2 leaders (`5pziQHHK` x4, `Coyadnds` x1). Root cause: `copytrading_trades.profit_loss` is only populated on close — for OPEN positions it stays at 0, and every dashboard query was summing that column directly.

**Fix.** New `_enrich_copytrading_pnl(raw_rows)` helper in `monitoring/enhanced_dashboard.py` that augments every row with:
- `realized_pnl` = the `profit_loss` column verbatim (closed trades only)
- `unrealized_pnl` = `live_jupiter_price * metadata.tokens_received - entry_usd` for OPEN Solana rows that have `tokens_received` populated
- `pnl_pending` = `True` for legacy pre-`6fe0a36` rows where `tokens_received` is missing AND `entry_price` is within +/-5% of `native_price_at_trade` (the schema-bug signature). UI shows "PnL pending" instead of fabricating $0.
- `profit_loss` is rewritten to `realized + unrealized` so every existing template that reads only `t.profit_loss` shows the right number with zero JS surgery.

**Endpoints rewired (all in `monitoring/enhanced_dashboard.py`):**
- `GET /api/copytrading/trades` — every row gets the enriched fields; aggregate `stats.realized_pnl` + `stats.unrealized_pnl` exposed.
- `GET /api/copytrading/wallets` — per-wallet aggregation in ONE query + ONE batched price call. Adds `realized_pnl`, `unrealized_pnl`, `pending_count` per wallet.
- `GET /api/copytrading/stats` — appends an OPEN-positions enrichment pass; folds unrealized PnL into both `total_pnl` and `live_pnl`. Adds `pnl_pending_count`.
- `GET /api/copytrading/discover` (local fallback path) — hot wallets card now shows live PnL via a follow-up enrichment pass on per-wallet OPEN rows.

**Templates updated** to show `(live)` vs `(realized)` suffix + legacy-row badge: `wallets_copytrading.html`, `trades_copytrading.html`, `performance_copytrading.html`, `dashboard_copytrading.html`, `discovery_copytrading.html`.

**Legacy-row detection.** A row is "legacy" (pre-6fe0a36 schema bug, where `entry_price = native_SOL_price` instead of `entry_price = USD_per_token`) when BOTH:
1. `metadata.tokens_received` is missing or 0
2. `entry_price` is within +/-5% of `native_price_at_trade` (the schema-bug always wrote the EXACT native price into entry_price; fresh post-fix rows write USD-per-token which is almost never within 5% of the SOL price for a real token).

Operator runs `python scripts/backfill_copy_tokens_received.py --force` to retroactively populate `metadata.tokens_received` on legacy rows. After backfill, the helper computes real unrealized PnL on those rows automatically.

**Performance.** A single dashboard refresh hits Jupiter Price v3 at most once thanks to the existing 30s TTL `_token_price_cache`. Fail-soft: any price-fetch error leaves `unrealized_pnl=0` and `pnl_pending=True` so the UI doesn't lie.

## Wave-6 (2026-05-21) — BUY/SELL detection rewrite + stablecoin guard

**Operator-reported critical bug.** Leader wallet `Coyadnds...HBjuBLh` SELL tx (token outflow + SOL/USDC inflow) was misread as a BUY of USDC. Five rows in `copytrading_trades` with `token_address = EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v` (USDC mint) are direct evidence -- we never should have held USDC as a copy-position.

**Root cause.** `_execute_solana_copy_trade` iterated `postTokenBalances` and stopped at the FIRST non-WSOL mint with a non-zero delta. When USDC iterated first, it was returned as `token_mint` with `is_buy=True` (positive delta).

**Fix (commit `2d30f3c`).** Delta-based detector:
1. Compute per-mint NET delta = `sum(post.uiAmount) - sum(pre.uiAmount)` across all owner rows.
2. Partition into `base_deltas` (mint NOT in `STABLECOIN_MINTS`) vs `quote_deltas` (USDC/USDT/USDH/WSOL).
3. Traded token = `base_deltas` mint with largest `|delta|`. Stablecoin mints are NEVER returned as the traded token.
4. `is_buy = base_delta > 0` (leader received tokens). Quote-side total is a cross-check warning, base is truth.
5. Defense-in-depth: refusal log `[replay] reason=stablecoin_not_tradeable` + `_stablecoin_refusals++` if detector ever picks a stable.

**SELL-side rule.** Only mirror if WE hold the token in EITHER on-chain SPL balance OR an open `copytrading_trades` row (via `_has_open_copy_position`). Otherwise log `[replay] reason=leader_sold_we_dont_hold` and skip. The previous `no_position_to_close` branch only checked on-chain balance -- broken under DRY_RUN where SPL=0 even after a simulated BUY.

**EVM parallel guard.** `_execute_evm_copy_trade` refuses BUYs whose target is in `EVM_STABLECOIN_ADDRESSES` (USDC / USDT / DAI / BUSD / WETH / WBNB / WMATIC on Ethereum, Base, Arbitrum, Optimism). Same `stablecoin_not_tradeable` reason string for symmetric forensics.

**New constants.**
- `STABLECOIN_MINTS` (set) -- USDC + USDT + USDH + WSOL on Solana. Members never become `token_mint`.
- `EVM_STABLECOIN_ADDRESSES` (lowercased set) -- EVM analogue. Members refuse on BUY.

**New replay-log reasons.** `stablecoin_not_tradeable`, `leader_sold_we_dont_hold` -- both join the existing Wave-1..4 vocabulary (`cooldown`, `position_cap`, `unsupported_chain`, `risk_gate`, `no_token_extracted`, `amount_too_small`, `no_position_to_close`, `probation`, `cross_module_cap`, `success`, `error`).

**Stats endpoint (commit `0770912`).** `/api/copytrading/stats` now tails `logs/copy_trading/main.log` (bounded 512 KB read) and exposes:
- `stablecoin_refusals` -- count of `[replay] reason=stablecoin_not_tradeable` lines.
- `leader_sold_we_dont_hold` -- count of `[replay] reason=leader_sold_we_dont_hold` lines.
Caveat: these reset on log rotation. They're forensic counters, not settled metrics.

**UI enrichment (commits `2466ec0` + `f0fb2bd`).** `/copytrading/trades` + `/copytrading/positions` + `/copytrading/dashboard` row rendering now shows: tokens-held (from `metadata.tokens_received`), entry $ / now $ per-token, copy-to-clipboard icon button on the token address, Birdeye link, and Solscan / Etherscan link. All three pages inline the same `copyTokenAddress` helper -- self-contained, no shared JS dependency.

## See also
- Phase 1 audit reports: `docs/agents/reports/COPY_TRADING_*.md` (quant / analyst / backend).
- Wave-2 quant audit: section 2 of `docs/agents/reports/COPY_TRADING_quant.md` — the CT-Q-01 / CT-Q-02 backlog drove the rebuild.
- Canonical engine API: `docs/engines.md`.
