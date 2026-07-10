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

## Wave-7 (2026-05-26) — secrets ordering + Helius RPC + wallet identity

**Issue 17a — keys logged as NOT SET (same class as AI commit `cca8d94`).**
`main_copy.py` read `secrets.get('ETHERSCAN_API_KEY')` / `HELIUS_API_KEY`
at startup BEFORE the db_pool was created and BEFORE
`secrets.initialize(db_pool)`. The secrets manager was still in bootstrap
mode, so it fell through to `os.getenv` (None for DB-only operators) and
logged `ETHERSCAN_API_KEY: NOT SET` / `HELIUS_API_KEY: Not set` even
though both live in the encrypted `secure_credentials` DB table. Fix
(mirrors `cca8d94`): connect DB first -> `secrets.initialize(db_pool)` ->
resolve via `get_async` (sync `get()` short-circuits inside a running
event loop) -> `.env` fallback retained. Startup now logs the keys as
`SET`.

**Issue 17b — "Solana RPC rate limited - backing off" spam.** The copy
monitor (`_monitor_solana_wallets`, 15s loop) was hitting a public Solana
RPC because the Helius key wasn't resolved (17a) and Helius is registered
under PoolEngine provider type `HELIUS_API`, not `SOLANA_RPC`. RPC
resolution path now:
1. `CopyTradeExecutor.initialize()` (async, post-db_pool): resolve
   `HELIUS_API_KEY` via `secrets.get_async`; if present, build
   `https://mainnet.helius-rpc.com/?api-key=<key>` and PREFER it.
2. Fallback: `RPCProvider.get_rpc('SOLANA_RPC')` (PoolEngine) -> `.env`
   `SOLANA_RPC_URL`.
3. `CopyTradingEngine.initialize()` adopts the executor's resolved
   `solana_rpc_url` (the engine's `__init__` resolves it SYNCHRONOUSLY
   pre-bootstrap and can't see the DB-stored key; the monitor reads the
   ENGINE's `self.solana_rpc_url`, so the value is mirrored across).
4. The 429-rotation handler no longer rotates AWAY from Helius to a
   public endpoint (that would just 429 again).

**Issue 6 — close path (VERIFIED end-to-end).** `_process_close_flag_files`
polls `logs/.close_copy_<trade_id>` (dashboard manual-close IPC), looks up
the position in `copytrading_positions` FIRST then falls back to
`copytrading_trades WHERE status='open'` (`from_trades_table=True`) —
positions actually live in `copytrading_trades`. `close_position` routes
via the executor's `copy_solana_swap` / `copy_evm_swap`, which under
DRY_RUN short-circuit through `should_skip_live` -> `_simulate_*`. On
success both tables are marked `status='closed'`. Confirmed wired and
DRY_RUN-safe.

## Wallet / Account identity
The copy module executes mirrored trades from the bot's OWN execution
wallets — DISTINCT from the leader `targets` it copies. Public addresses
only; private keys/keypairs are never logged or surfaced.
- **EVM execution wallet** (`self.executor.evm_wallet`): public address
  from secrets key `WALLET_ADDRESS`; signer key from secrets key
  `PRIVATE_KEY` (Fernet-decrypted in `_get_decrypted_key`, env fallback).
  ONE EOA signs across all supported EVM chains (Ethereum / Base /
  Arbitrum / Optimism) — fund that address with native gas on each chain.
- **Solana execution wallet** (`self.executor.solana_wallet`): public
  address from secrets key `SOLANA_MODULE_WALLET`; keypair from secrets
  key `SOLANA_MODULE_PRIVATE_KEY`.
- These are the bot's wallets that BROADCAST copies, NOT the leader
  wallets in `targets` (those are observed, never controlled).
- **Surfaced for the dashboard:** `CopyTradingEngine.get_execution_wallets()`
  returns the public addresses + their source secret-key names;
  `_persist_execution_wallets()` (called from `initialize()`) writes them
  to `config_settings(config_type='copytrading_diagnostics')` keys
  `evm_execution_wallet` / `solana_execution_wallet` (public address only,
  empty string until resolved). Dashboard agent: read these to show the
  operator which wallet funds copy trades per chain.

## Wave-19 (2026-06-03) — Solana fallback-poll RPC routing + cadence
**Operator-reported (15h):** `Solana RPC rate limited in fallback poll - backing off` every ~5 min; `EVM Copies: 0 | Solana Copies: 0` for 33 wallets. Public Solana endpoints (`rpc.ankr.com/solana`, `free.rpcpool.com`, `api.mainnet-beta`) were `unhealthy` while the operator's Helius endpoints were `active`/high-priority.

**Root cause.** The Helius JSON-RPC endpoint is registered under pool_engine provider type `HELIUS_API`, NOT `SOLANA_RPC`. `get_endpoint('SOLANA_RPC')` therefore only ever returned the rate-limited PUBLIC endpoints. The fallback poll (`_poll_wallet_sigs`) POSTed `getSignaturesForAddress` to a **pinned, static** `self.solana_rpc_url`; whenever the Helius key wasn't resolved synchronously at `__init__`, that pinned value was a public endpoint, and the old 429 handler rotated only within the public `SOLANA_RPC` pool — so it could never escape to Helius.

**How the RPC is now resolved (fallback poll).** `_poll_wallet_sigs` no longer uses the pinned URL. It calls `_resolve_solana_rpc()` fresh **per cycle**:
1. `pool_engine.get_endpoint('HELIUS_API')` — the healthy Helius JSON-RPC URL (`https://mainnet.helius-rpc.com/?api-key=...`) is a valid JSON-RPC endpoint and is preferred when present.
2. `pool_engine.get_endpoint('SOLANA_RPC')` — public pool, used only when no Helius key.
3. `self.solana_rpc_url` — last-resort static value.
Every call reports `report_success` / `report_failure` / `report_rate_limit` back to pool_engine under the resolved provider type so health/priority is honoured each cycle. The fast path (`_fetch_wallet_txs_helius`, `api.helius.xyz/v0` enhanced-tx REST) is unchanged and still preferred; the fallback poll only fires on Helius transport failure.

**Cadence knobs (migration 068, `copytrading_config`).**
- `copy_poll_interval_s` (float, default `15`, clamp 5..300) — full monitor-cycle cadence; replaces the hardcoded 15 s `asyncio.sleep`.
- `copy_request_spacing_s` (float, default `0`, clamp 0..5) — global minimum gap between consecutive outbound Solana RPC calls, enforced across the wallet fan-out via `_space_solana_request()` so N wallets don't fire one synchronized burst.
Pre-existing related knobs still apply: `copy_max_concurrent_wallets` (default 5), `copy_max_signal_age_s` (default 5), `copy_cursor_lookback_minutes` (default 15).

**Rate-limit WARNING throttle.** Confirmed working: `_fallback_rl_throttle_s=300` demotes repeats to DEBUG for 5 min; the WARNING now names the endpoint (Helius vs public) so a single line per 5 min is not masking total failure.

**Honest sustainability assessment (33 wallets).** The PRIMARY load is the Helius **enhanced-tx REST** path (one `GET /v0/addresses/{w}/transactions` per wallet per cycle), NOT JSON-RPC `getSignaturesForAddress` — the fallback poll only runs when Helius REST fails. At the default 15 s cadence, 33 wallets = ~2.2 REST calls/s sustained (bursty up to `copy_max_concurrent_wallets=5` concurrent). Helius free/developer tier (~10 req/s, 100k credits/day) makes this borderline: 33 wallets × 5760 cycles/day ≈ 190k calls/day, which EXCEEDS a 100k/day free credit budget. Recommendation: on the free tier either raise `copy_poll_interval_s` to ~30 s (halves daily calls to ~95k) or trim the watchlist to ~17 wallets; a paid Helius plan (Developer 10M credits/mo) sustains 33 wallets comfortably at 15 s. This change FIXES the routing (Helius is now actually used) but does not raise the operator's Helius quota — if the free tier is exhausted, the symptom shifts from "always 429 on public RPC" to "429 on Helius once daily credits run out," which the per-endpoint WARNING now makes visible.

## Telegram alerts (wave-24)
The copy engine emitted nothing to Telegram beyond the startup/shutdown banner.
`modules/copy_trading/copy_alerts.py:CopyTelegramAlerts` is an engine-routed
helper (mirrors `futures_alerts`/`solana_alerts`): every send goes through
`TelegramNotificationEngine.notify('copy', category, ...)` so it carries the
`[COPY]` header and lands in topic 18. Fail-soft + DRY_RUN-safe. **Wire-in
(owning web3 agent):** construct `CopyTelegramAlerts()` in `CopyTradingEngine`
and call `send_copy_alert(CopyTradeAlert(action='buy'|'sell', token=..., chain=...,
leader=..., amount_usd=..., is_simulated=self.dry_run))` at the
`_execute_evm_copy_trade` / `_execute_solana_copy_trade` success points and at
close. Until wired, COPY still appears in the periodic summary/dashboard topics
built from `copy_trades`. See `docs/TELEGRAM_SETUP.md`.

## See also
## Wave-9 honest-scoring audit (2026-05-26)
Audited `leader_scorer.py` + the `copy_engine.py` entry path for the two
DEX-scorer defects fixed in Wave-8 (`core/engine.py`). Both ABSENT. No
code changed; no gate loosened; DRY_RUN untouched.

- **PATTERN 1 (a failed safety/risk check is silently rewarded) — ABSENT.**
  `leader_scorer.compute_score` normalizes weights over a FIXED set of 5
  components (pnl/sharpe/hit_rate/hold/drawdown) and ALWAYS computes all 5
  — no component is conditionally dropped from the denominator, so the
  "missing term inflates the normalized score" defect cannot occur. Every
  metric helper degrades a missing/insufficient input toward 0 (Sharpe→0
  under 3 samples, hold→0, dd→0), and `sample_credit` shrinks the whole
  composite toward 0 for low-trade leaders — missing data LOWERS the score,
  the opposite of the DEX bug. `_kelly_from_metrics` returns 0 (refuse to
  size) below 5 trades or break-even. Walk-forward enforced via
  `_filter_window` on `exit_timestamp` (no look-ahead). On the engine side,
  the primary `RiskManager.validate_trade` gate in `copy_solana_swap` /
  `copy_evm_swap` FAILS CLOSED (refuses the swap on exception) — exemplary.
- **PATTERN 2 (fabricated ML/confidence constants feeding real gates) —
  ABSENT.** No fabricated probability/confidence fields. `_simulate_*`
  return only tx-hash/amount (no fake PnL). Close-path PnL is computed from
  real `entry_usd`/`exit_usd` via live `PriceFetcher.get_price` (CoinGecko);
  the only constants are last-resort NATIVE-token price fallbacks on total
  API failure, which don't feed any gate.
- **Punch-list (operator sign-off, NOT fixed):** the SECONDARY refinement
  gates `_is_leader_on_probation` + `_check_cross_module_exposure` are
  fail-SOFT (a DB error returns "allow"). Superficially PATTERN-1-shaped,
  but deliberate and defensible: both are layered ON TOP of the
  fail-CLOSED `RiskManager.validate_trade` + per-module caps + position
  cap, and can only further-restrict — a lookup failure cannot bypass the
  primary safety gate. Left as documented design; flagged so the operator
  is aware of the asymmetry vs the sniper/DEX fail-closed posture.
  `_get_leader_kelly` returns 0.0 (shrinks size) on DB failure — already
  conservative.

- Phase 1 audit reports: `docs/agents/reports/COPY_TRADING_*.md` (quant / analyst / backend).
- Wave-2 quant audit: section 2 of `docs/agents/reports/COPY_TRADING_quant.md` — the CT-Q-01 / CT-Q-02 backlog drove the rebuild.
- Canonical engine API: `docs/engines.md`.

## COPY v3 (2026-06-13) — profitable-wallet discovery + shadow-copy simulator (migration 135)

### What shipped
1. **`wallet_profitability.py`** — pure FIFO realized-PnL wallet scorer (0..100). Metrics over trailing windows only: realized PnL, Bayesian-shrunk win rate, profit factor (capped 10), daily-PnL consistency (annualised-Sharpe sigmoid), max drawdown of the cumulative realized curve, hold-time bell fit, recency, token-notional diversification. Penalties: `wash_penalty` (sub-60s near-zero-PnL round-trip share) and `lucky_penalty` (best trade > 60% of gross wins) multiply the composite down; `sample_credit` shrinks low-sample wallets toward 0. **No-look-ahead guarantee:** events after `as_of` are dropped (strict mode raises — asserted in the self-test); only round-trips whose SELL leg falls inside the window are scored; open lots contribute NOTHING; sells with no known basis are ignored, never fabricated. Self-test: `python -m modules.copy_trading.wallet_profitability`.
2. **`discovery_v3.py`** — multi-source candidate sweep + scoring + proposal. Sources (`copy_v3_sources`): `smart_money` (READ-ONLY `smart_money_wallet_scores`/`smart_money_wallet_events`, mig 133 — USD-priced swap events, the highest-quality feed), `onchain` (our own `copytrading_trades` per source_wallet — realized, USD-exact), `leader_scores` (v2 `copy_leader_scores` re-ranked under the realized-only scorer), `dexscreener` (v2 fetcher reuse — addresses only, ranks low until history exists), `rpc_solana` (bounded Helius enhanced-tx enrichment, auto-added when a Helius key resolves; SOL-numeraire pricing flagged `pnl_basis='sol_numeraire'` — relative ranking honest, absolute USD noisy). Writes the FULL ranked universe to `copy_discovered_wallets` (provenance + score breakdown); wallets passing `copy_v3_min_score`/`min_trades`/`min_realized_pnl_usd` + penalty ceilings are upserted into `copy_leader_candidates` as **`pending`** (never overwrites an operator's approved/rejected decision).
3. **`shadow_simulator.py`** — paper-copies candidate/leader wallets to measure COPY-performance before promotion. Fixed-notional paper BUY at `leader_price*(1+(slippage+fee)bps)`, proportional exit mirroring on leader SELLs (fraction of leader's tracked open qty), realized booked on sells, unrealized marked at last observed event price (stale between events — documented limitation). Persists `copy_shadow_fills` / `copy_shadow_positions` / `copy_shadow_equity` (ALL `is_simulated=true`) with a per-wallet `copy_shadow_cursors` ingest cursor (events processed strictly in order; future events refused). Event sources: `smart_money_wallet_events` (read-only) + discovery's Helius enrichment via `record_events` (dedup on source_ref). Self-test: `python -m modules.copy_trading.shadow_simulator`.
4. **`read_api.py`** — read-only query layer for the dashboard agent: `get_discovery_leaderboard`, `get_candidates`, `get_shadow_leaderboard`, `get_shadow_equity_curve`, `get_shadow_fills`, `get_v3_overview`. Fail-soft, no mutations.
5. **Engine v3** (`copy_engine.py`): main loop gains fail-soft `_maybe_run_discovery_v3` (gated by the previously-unconsumed mig 092 `copy_auto_discovery_enabled`, default false) and `_maybe_tick_shadow_sim` (`copy_shadow_sim_enabled`, default false). `_get_score_multiplier` (`copy_score_sizing_enabled`, default false) multiplies the kelly-capped size by clamp(score/100, floor, 1.0) at both EVM and Solana BUY sizing points — **can only shrink** below existing caps (unscored leaders size at the floor; lookup failure → 1.0, static caps remain the net). `_track_peak_and_decay` persists `copy_leader_scores.peak_score` every refresh; with `copy_score_decay_demotion_enabled` (default false) a score ≥ `copy_score_decay_pct`% below peak (min-trades gated) benches the leader via the EXISTING probation mechanism (BUY-only, SELLs never gated, auto re-entry).

### Promotion-approval gate (the headline safety property)
A discovered wallet is NEVER traded silently. The only paths to live copying:
- **Operator approval**: review the `pending` row in `copy_leader_candidates`, then manually add the wallet to `target_wallets` (unchanged v2 flow).
- **Auto-promote (DOUBLE-gated, default does nothing)**: requires `copy_v3_auto_promote_enabled=true` AND `copy_v3_auto_promote_max_leaders > 0` (seeded **0**) AND ≥ `copy_v3_auto_promote_min_shadow_fills` simulated fills with **positive** realized shadow PnL. Promotion appends to `target_wallets`, stamps `reviewed_by='auto_promote'`, and WARN-logs. Every downstream gate (BUY-only `RiskManager.validate_trade`, `should_skip_live`, probation, cooldown, position/leader/cross-module caps, dedup) still applies to promoted leaders exactly as to manual ones.

### Key config (all `copytrading_config`, seeded mig 135, default-safe)
`copy_v3_sources`, `copy_v3_window_days` (30), `copy_v3_min_score` (60), `copy_v3_min_trades` (10), `copy_v3_min_realized_pnl_usd` (500), `copy_v3_max_lucky_share` (0.6), `copy_v3_max_wash_penalty` (0.4), `copy_v3_max_candidates_per_sweep` (25), `copy_v3_max_rpc_enrich_wallets` (5), `copy_v3_auto_promote_enabled` (**false**), `copy_v3_auto_promote_max_leaders` (**0**), `copy_v3_auto_promote_min_shadow_fills` (10), `copy_shadow_sim_enabled` (**false**), `copy_shadow_sim_interval_s` (300), `copy_shadow_sim_notional_usd` (100), `copy_shadow_fee_bps` (30), `copy_shadow_slippage_bps` (50), `copy_shadow_max_adds_per_position` (3), `copy_shadow_max_wallets` (40), `copy_score_sizing_enabled` (**false**), `copy_score_sizing_floor` (0.25), `copy_score_decay_demotion_enabled` (**false**), `copy_score_decay_pct` (40), `copy_score_decay_min_trades` (10). Dashboard agent: surface these on the settings page; render `/copytrading/discovery_v3` from `copy_discovered_wallets` + `copy_leader_candidates` and `/copytrading/simulator` from `copy_shadow_equity` + `copy_shadow_fills` via `read_api.py`.

### Honesty section — what copy trading can and cannot do
- **Copying is reactive and laggy.** We see a leader's swap at best one poll cycle (~15s) plus indexing lag after it lands; in memecoin time that is an eternity. The shadow simulator's slippage knob models this only crudely — real copy fills are systematically WORSE than the leader's fill, which is exactly why the simulator measures copy-performance, not wallet-performance.
- **Alpha decays as wallets get crowded.** A publicly-rankable profitable wallet attracts copiers; their flow front-runs ours and fades the edge. Expect discovered-wallet performance to mean-revert; the decay-demotion knob exists because scores SHOULD fall.
- **On-chain PnL attribution is noisy.** FIFO over observed swaps misses transfers in/out, airdrops, LP positions, cross-wallet activity, and (for `sol_numeraire` rows) uses a single current SOL price. Wash-trade detection is heuristic. Treat scores as a screening rank, not an audited track record — that is why the shadow simulator and the operator-approval gate sit between discovery and live money.
- **Survivorship bias at the source.** Leaderboard-style feeds (DexScreener, smart-money scores) only ever show wallets that already won; the scorer's trailing-window + penalty design mitigates but cannot eliminate this.

## Wave-F5 (2026-07-06) — discovery revival + AI-Trader adaptations (migration 141)

**Operator complaint (second time): "wallet discovery is finding my already-tracked wallet always."** Root cause (docs/agents/wave-f5/04_copy_aitrader.md): every EXTERNAL candidate source returned 0 rows, so discovery degraded — by design — to a LOCAL fallback built only from `target_wallets` + `copytrading_trades.source_wallet` (the already-tracked wallets), and the UI presented that fallback as real discovery. Meanwhile the v3 engine (mig 135) had NEVER run because `copy_auto_discovery_enabled` was still `false`. Source failures were logged at DEBUG only, so the degradation was invisible.

### Discovery source table (v3, `copy_v3_sources`)
| Source | Chain | Free? | Status | Notes |
|---|---|---|---|---|
| `smart_money` | EVM | yes (mig 133) | best USD-priced feed | needs `smart_money_wallet_events`; EVM-only in v1 |
| `smart_money_scores` | EVM | yes | **NEW (F5)** | reads `smart_money_wallet_scores` ≥ `copy_sm_min_score`, proposes top-N EVM wallets DIRECTLY into `copy_leader_candidates` (bypasses the realized-PnL scorer — the smart_money module already scored them); fail-soft if table absent |
| `helius_tokens` | Solana | yes | **NEW (F5)** | top-volume tokens (DexScreener) → recent SWAP fee-payers per token (Helius, budget-capped) → accumulated ACROSS sweeps in `copy_discovery_feepayers` → candidate once cumulative swaps ≥ `discovery_min_swaps`. This is the structurally-sound Solana feed the old code lacked |
| `onchain` | any | yes | recycles tracked | our own `copytrading_trades` — realized, USD-exact, but only already-tracked wallets |
| `leader_scores` | any | yes | recycles v2 output | re-ranks `copy_leader_scores` |
| `dexscreener` | any | yes | **candidate-address only** | see quarantine note below |
| `rpc_solana` | Solana | Helius | enrichment | per-candidate Helius enhanced-tx, SOL-numeraire pricing; auto-added when a Helius key resolves |

**DexScreener pool-address bug fixed.** `wallet_discovery.fetch_dexscreener_top_traders` used to harvest `pairAddress` (an AMM pool contract) and a non-existent `info.deployerAddress` as "wallets" — junk that scored ~0 and polluted candidates. DexScreener has no public top-traders REST API, so that adapter now returns `[]` (mock rows preserved). The legitimate volume signal it CAN provide is exposed by the new `fetch_dexscreener_token_pools` and consumed by `helius_tokens`.

### Helius quota discipline
The Helius key is shared with the copy monitor's per-wallet poll, so discovery is a good tenant: exponential backoff on 429 + a persistent daily call budget (`helius_daily_call_budget`, seed 500) tracked in `copy_helius_budget` (one row per UTC day). Every discovery Helius request checks the budget BEFORE firing and consumes one unit. The old "wallet must appear ≥5 times inside ONE 100-tx snapshot" sampling flaw is replaced by cross-sweep fee-payer accumulation (`copy_discovery_feepayers`): low-frequency wallets accumulate over days until they clear `discovery_min_swaps`.

### Honest-fallback semantics (no more silent degradation)
- **Dashboard `/api/copytrading/discover`**: response now carries `source_status` (per-source candidate counts + failure reasons, e.g. `helius: no_helius_key`, `birdeye: no_birdeye_key`) and `already_tracked_count`. Local-fallback rows are flagged `fallback=true` + `already_tracked=true` so the UI never presents an already-tracked wallet as a new find. Discovery is allowed to honestly return zero.
- **Module (`wallet_discovery` v2 + `discovery_v3`)**: per-source failures are now WARNING (rate-limited 1/source/hour) instead of DEBUG; every sweep logs a per-source summary (`per-source={...}, external=N, local=M`). When EVERY external source returns 0 and only local/recycled wallets remain, ONE honest WARNING fires and those candidates are marked `fallback=true`.

### Config knobs (all `copytrading_config`, mig 141 seeds)
`helius_daily_call_budget` (500), `helius_tx_sample` (100), `discovery_min_swaps` (3), `copy_sm_min_score` (0.6), `copy_reconcile_minutes` (360). Mig 141 also CONDITIONALLY flips `copy_auto_discovery_enabled` and `copy_shadow_sim_enabled` `'false'→'true'` **only WHERE still the seeded `'false'`** (operator overrides preserved). Both gate read-only discovery writes and simulated shadow fills — **neither is a live-execution flag; no order path is enabled.** A discovered wallet is still traded only after operator approval (`copy_leader_candidates`) or the pre-existing double-gated auto-promote (default OFF, `max_leaders=0`).

### AI-Trader (HKUDS) adaptations
- **Shadow mark-to-market** (`shadow_simulator.mark_to_market`, called each `tick`): re-prices open `copy_shadow_positions` via Jupiter Price v3 (free) and writes a fresh `copy_shadow_equity` snapshot each cycle, so equity curves are honest between leader events (Solana marks refreshed; other chains keep last price — documented limitation).
- **Crowding penalty** (`wallet_profitability.crowding_penalty_from_followers` + optional `crowding` arg to `score_wallet`): inverted follower-density soft-cap (crowding = alpha decay). Pure, default 0, can only shrink the composite.
- **Leader holdings reconciliation** (`copy_engine._maybe_reconcile_leaders`, every `copy_reconcile_minutes`): snapshots each Solana leader's current SPL holdings (`getTokenAccountsByOwner`) vs our mirrored open positions, logs `[reconcile] reason=leader_exited` drift. **Advisory only — never trades, never auto-closes** (RPC failure is not misread as a full exit). NOT worth porting from AI-Trader: 1:1 mirroring (regression vs fractional-Kelly), LLM-in-the-loop trade decisions, the FastAPI signal marketplace.

### Operator notes (key-gated, fail-soft)
- **`ETHERSCAN_API_KEY` unlocks EVM discovery.** Without it the copy engine's EVM monitor is disabled, so `smart_money` / `smart_money_scores` EVM candidates would be leaders the engine cannot mirror — set the key, or drop those two sources from `copy_v3_sources`.
- **`BIRDEYE_API_KEY` unlocks the Birdeye source.** Absent → `source_status` reports `birdeye: no_birdeye_key` and the source is skipped (never a hard failure).
- **Helius free tier**: at 33 wallets/15s the copy monitor alone can exceed 100k credits/day; discovery's daily budget is deliberately small (500) so it does not compound the monitor's burn. Raise `copy_poll_interval_s` or upgrade the plan if 429s persist.

## Wave-F6 (2026-07-10) — discovery actually ON + Helius honesty & cadence (migration 149)

**Root cause closed (docs/agents/wave-f6/01_rate_limiting.md + 04_advisory_sweep.md):** the Wave-F5 discovery revival NEVER ran — mig 141's conditional `UPDATE … WHERE value='false'` did not match the stored value, so `copy_auto_discovery_enabled` / `copy_shadow_sim_enabled` stayed off and `_maybe_run_discovery_v3` / `_maybe_tick_shadow_sim` returned immediately for weeks (operator symptom: "discovery finds only my wallet", 0 copies, zero `[discovery-v3]` log lines).

### Discovery + shadow sim are now ON BY DEFAULT (non-trading)
Mig 149 sets both flags to `'true'` **unconditionally** (`INSERT … ON CONFLICT DO UPDATE`). This is deliberate — the conditional flip already failed once. Safety unchanged: discovery only writes candidates to `copy_leader_candidates` for **operator approval**; the shadow simulator only writes `is_simulated=true` paper rows. Neither is a live-execution flag; auto-promote stays double-gated OFF (`copy_v3_auto_promote_enabled=false`, `max_leaders=0`); every live gate (`should_skip_live`, RiskManager, DRY_RUN) is untouched. Operators can still turn either flag off in DB; the engine re-reads them every cycle. Verify liveness by watching for `[discovery-v3] sweep:` lines.

### Cadence defaults (mig 149 seeds; conditional UPDATE preserves operator overrides)
| key | old | new | why |
|---|---|---|---|
| `copy_helius_rps` | 8 | **2** | the Helius account is SHARED with sniper+solana; 8 rps from copy alone nearly consumed the ~10 rps free ceiling (90k pool-side 429 events in ~2 days) |
| `copy_poll_interval_s` | 15 | **30** | halves daily Helius calls; the staleness gate (`_effective_signal_age_s`) self-adjusts to the poll cadence so no signal is spuriously rejected |

The shared `HELIUS_API` token bucket is now **re-configured whenever the knob changes** (was configure-once per process, so runtime edits silently never applied). Note: the ~40 s Solana **price**-poll called out in the rate-limit report lives in `modules/solana_trading` (its position monitor), not in this module — flagged to the solana owner, not changed here.

### Helius multi-key honesty (pool_engine + budget)
- `config/pool_engine.py` now DEDUPS `HELIUS_API` endpoints by the embedded account key at load and logs the distinct count at startup (e.g. `HELIUS_API: 1 distinct key across 5 endpoint name(s)` — WARN when names > keys). Five names over one key was rotation theater: one quota bucket hit five ways.
- `discovery_v3` interprets `helius_daily_call_budget` (500) **per DISTINCT account key**, no longer per endpoint name — the old counting multiplied the budget 5x against one free-tier account.
- **Operator action for real headroom (the actual fix):** create DISTINCT Helius accounts and store their keys as `HELIUS_API_KEY_2..4` in `secure_credentials` (see `docs/RPC_API_KEYS_GUIDE.md`). Until the startup line reports more than 1 distinct key, rotation adds zero capacity — the dedup/cadence work makes the pool honest and cuts demand ~3-4x, it does not raise quota.
