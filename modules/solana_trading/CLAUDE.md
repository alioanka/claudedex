# SOLANA Module
## What it does
Solana spot trading via Jupiter aggregator with trailing-stop ladder, plus optional pump.fun launch sniping and Drift perp leg. Canonical module dir for all SOL strategies.
## Entry point
`modules/solana_trading/main_solana.py` — launched as a subprocess by `main.py` when `SOLANA_MODULE_ENABLED=true`. Engine: `modules/solana_trading/core/solana_engine.py`. The sibling dir `modules/solana_strategies/` holds helper utilities only (`jupiter_helper.py`, `drift_helper.py`, a duplicate `solana_config_manager.py`); it intentionally has no `CLAUDE.md`.
## Key config (DB-backed via `solana_config_manager.py`)
- `position_size_sol` — base SOL committed per Jupiter entry
- `jupiter_slippage_bps` — Jupiter quote slippage cap (default 50)
- `priority_fee_lamports` — Solana compute-unit priority fee (default 1_000_000)
- `stop_loss_pct` / `take_profit_pct` — Jupiter trailing-stop bounds
- `drift_enabled` / `drift_leverage` — Drift perp leg toggle and leverage cap
- `pumpfun_max_positions` — pump.fun concurrent position ceiling
- `adaptive_priority_fee_enabled` (default `False`) — when True, JupiterHelper bids `getRecentPrioritizationFees` p75 (clamped to `adaptive_priority_fee_min_lamports`..`adaptive_priority_fee_max_lamports`, 5s cache). Off keeps legacy structured-dict default.
- `jupiter_quote_max_age_s` (default 10s) — execute_swap refetches quotes older than this before signing (mirrors `jupiter_executor.py` P1 fix).
- `solana_ml_enabled` (default `False`) — when True, `_open_position` runs the lazy-loaded `ml/models/rug_classifier.RugClassifier` and refuses entry above `solana_ml_max_rug_prob` (default 0.40). Fail-soft: no trained model → gate idle (refuse-to-predict, MB-19 pattern).
- `drift_max_leverage` / `drift_max_funding_pct_annual` / `drift_oracle_deviation_max_pct` / `drift_min_oracle_conf_bps` — MB-15 client-side guards inside `DriftHelper.open_position`. Drift `place_perp_order` is gated by all four AND the killswitch / pause flag. Drift stays toggle-OFF by default.
## Wallet / Account identity
- The Solana trading keypair is loaded from the secrets-manager key **`SOLANA_MODULE_PRIVATE_KEY`** (`security/secrets_manager.secrets.get_async('SOLANA_MODULE_PRIVATE_KEY')`, with a plain-`os.getenv` fallback for bootstrap and Fernet auto-decrypt when the stored value is still encrypted). Accepts JSON-array, base58, or hex key formats; `solana_engine.py:~1353` `_get_solana_private_key`.
- The resulting **public address** is derived once at load time (`self.wallet_pubkey = str(self.wallet.pubkey())`, `solana_engine.py:~1426`) and is the wallet the module trades from. This is the address the operator must FUND before flipping LIVE.
- Surfaced for the dashboard as `wallet_address` on `engine.get_health()` (→ `/health`, `/stats` health block on port `SOLANA_HEALTH_PORT`, default 8082). Public address ONLY — the private key/keypair is never exposed on any surface.
- To find the address without the bot running: decrypt `SOLANA_MODULE_PRIVATE_KEY` and derive the pubkey, or read it from the running module's `/health`.

## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_solana` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> Jupiter execute returns simulated route.
## Logs
`logs/solana_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
- Cross-module: `core.risk_manager.RiskManager.validate_trade(token_mint, amount_sol)` called in `_open_position` at `solana_engine.py:~3296` immediately before every Jupiter swap broadcast. Injected via `set_risk_manager()` from `main_solana.py`; engine is fail-soft if `RiskManager` construction fails (logs a warning, continues without the gate). Only entries are gated; exits always allowed.
- Per-module local: position-count ceiling + per-strategy SL/TP percent enforced inside the engine's close-path.
## Drift perp strategy — setup & usage guide
**What it is.** Drift Protocol is an on-chain perpetual-futures DEX on Solana. The module trades a funding-carry signal: when a market's annualized funding rate exceeds a threshold it takes the side that COLLECTS funding (SHORT when funding is positive — longs pay shorts — LONG when negative). Wiring lives in `solana_engine.py` `_init_drift` / `_scan_drift_opportunities`; the on-chain SDK calls are in `modules/solana_strategies/drift_helper.py` (`DriftHelper`).

**Dependencies / collateral.**
- `pip install driftpy` (optional dep; without it Drift runs in DRY_RUN-simulated mode only and cannot trade live).
- A funded Drift user account: deposit USDC collateral to the Drift sub-account owned by the wallet behind `SOLANA_MODULE_PRIVATE_KEY` (see "Wallet / Account identity"). No collateral -> the leverage guard refuses every entry (account_value=0 fail-closed).
- Uses the same wallet/keypair as Jupiter spot — no separate Drift wallet.

**Config keys (DB-backed, `solana_drift` section).**
- `drift_enabled` (default `False`) — toggles the strategy into the run loop.
- `drift_markets` (default `SOL-PERP,BTC-PERP,ETH-PERP`) — comma list; engine maps names to Drift indices via `DRIFT_MARKET_INDEX` (SOL-PERP=0, BTC-PERP=1, ETH-PERP=2).
- `drift_leverage` (default 5) — multiplier applied to `position_size_sol` for the perp base size.
- MB-15 client-side guards (read via `cm.get()`): `drift_max_leverage` (3.0), `drift_max_funding_pct_annual` (50.0), `drift_oracle_deviation_max_pct` (1.0), `drift_min_oracle_conf_bps` (500). All four fail CLOSED inside `DriftHelper.open_position`.
- Scan tunables (engine constants): `_drift_scan_interval_s` (60s per-market throttle), `_drift_funding_signal_pct` (10%/yr signal threshold).

**How to enable.** Set `drift_enabled=true` (and the desired `drift_markets`) in DB config; ensure `SOLANA_MODULE_ENABLED=true`. Keep `DRY_RUN=true` until verified.

**How to verify it's working (DRY_RUN).** Watch `logs/solana_trading/`. On startup you should see `🔶 DriftHelper chain not connected ... DRY_RUN simulated Drift activity will still run` (no driftpy/collateral) or `✅ DriftHelper initialized`. Then once per `_drift_scan_interval_s` per market: a `🎯 Drift signal: <market> funding=...` line followed by `✅ [DRY_RUN] Drift <SHORT|LONG> <market> opened: DRY_RUN_DRIFT_...`. In DRY_RUN, if chain funding is unavailable the engine synthesizes a funding value (logged `🔶 [DRY_RUN] ... using simulated ...%/yr`) so the path runs end-to-end without a funded account. Entries are gated by `RiskManager.validate_trade` and the killswitch/pause file.

**Going live.** Install driftpy, deposit USDC collateral, confirm `✅ DriftHelper initialized` (NOT the "chain not connected" warning), then flip the module to LIVE. In LIVE the helper enforces all four MB-15 guards and the engine disables Drift entirely if it cannot reach chain (refuses to trade perps blind). Exit/close of perp legs is `DriftHelper.close_position(market_index)` (opposite-side market order); the spot trailing-stop ladder does NOT manage perp exits.

## Live-trade readiness
AMBER → GREEN candidate (Jupiter spot; pending production verification). MB-06..MB-10 closed (decimals, co-signers, priority fee, restart reconciliation, DRY_RUN gate); secrets_manager wiring (`8b4ee7d`) and pool_engine sweep (`a21ec41`) landed. Campaign wave-2 additions:
- **MB-06 residuals (`09a5c85`)** — `_get_token_balance` now resolves decimals on-chain via `core.units.get_spl_decimals` + parsed `tokenAmount.decimals`; the emergency-close path no longer hardcodes `(10 ** 6)` for the Jupiter sell amount.
- **MB-15 Drift hardening (`661cee6`)** — `DriftHelper.open_position` enforces dry-run / killswitch / pause, funding-rate sanity, oracle deviation + Pyth confidence, and a leverage cap. All four fail closed. Drift stays toggle-OFF by default.
- **Adaptive priority fee + quote freshness TTL (`83df4ad`)** — JupiterHelper can bid `getRecentPrioritizationFees` percentile (default p75) and refetches stale Jupiter quotes before signing. Both opt-in; config wired in `8cf0143` bundle.
- **P1-07 ML rug gate (`b1b358f`)** — `_open_position` runs `RugClassifier.predict` when `solana_ml_enabled=True`; refuses entry above `solana_ml_max_rug_prob`. Fail-soft when no trained model.

Wave-4 additions:
- **Jito bundle wiring (`d6a4a8c` + `a6c3a89`)** — `solana_jito_bundle_enabled` flag (default `False`). When True, the engine submits the signed Jupiter swap + a tip tx via `trading/chains/solana/jito_bundle.JitoClient` and falls back to vanilla Jupiter on bundle-rejection / rate-limit / timeout. Tip lamports configurable via `solana_jito_tip_lamports` (default 50_000 — the documented competitive floor; arbitrage's 10k default lands far less reliably). Every attempt + outcome (SEND / LANDED / REJECTED / SKIPPED-rate-limited / ERROR / fell-back) is logged at INFO so operator can see uptake.
- **Pump-predictor warmup pre-fetch (`3edd27e`)** — engine startup pre-fills the per-token `TokenPriceBuffer` with the last hour of 1-min bars via Birdeye `defi/history_price` when `BIRDEYE_API_KEY` is configured, otherwise seeds a single Jupiter spot price per active token. Tokens warmed = `config_manager.jupiter_tokens` ∪ reconciled-from-DB position mints. Per-token skipped when the buffer is already full. The predictor gate itself stays opt-in via `solana_pump_predictor_enabled` (default `False`); the warmup just removes the 30-min cold-start window once the gate is flipped on.

Outstanding (follow-ups, not blocking):
- `pump_predictor.py:170-215` scaler `fit_transform` over full dataset = standardization leakage (independent of the "look-ahead" label characterization in P1-08, which on re-read is a legitimate next-bar binary label, not X→y leakage).
- Birdeye history is gated on operator-supplied `BIRDEYE_API_KEY`; without it the warmup only seeds 1 bar/token (not 60). Jupiter Price v3 has no history endpoint so this is the cheapest data source today.

## See also
- Phase 1 audit reports: `docs/agents/reports/SOLANA_*.md` (smartcontract / quant / analyst).
## Wave-9 honest-scoring audit (2026-05-26)
Audited the entry/scoring path for the two DEX-scorer defects fixed in
Wave-8 (`core/engine.py`). Both ABSENT here. No code changed; no gate
loosened; DRY_RUN untouched.

- **PATTERN 1 (a failed safety/risk check is silently rewarded) — ABSENT.**
  The Solana entry decision is NOT a normalize-by-weight score; it is a
  chain of hard boolean gates in `_open_position` (symbol/scam-name/
  blacklist/holder/dev-holding/mcap/liquidity/momentum/vol-liq/sell-
  pressure, sniper_engine sections 1-5) each returning `False` to reject,
  plus the cross-module `RiskManager.validate_trade` gate, plus the
  opt-in ML rug + pump gates. `filter_token` is likewise a boolean.
  Nothing divides by a sum-of-weights so nothing inflates when a term is
  missing. The two ML gates (`_ml_rug_probability` /
  `_pump_predict_probability`) are layered ON TOP of the still-active hard
  checks; their `try/except` "continue without it" falls back to those
  hard checks, not to "treat as safe". `calculate_dynamic_position_size`
  affects SIZE only (bounded `[min, 2x]`), not the entry gate.
- **PATTERN 2 (fabricated ML/confidence constants feeding real gates) —
  ABSENT.** `_ml_rug_probability` / `_pump_predict_probability` are
  HONEST: they return `None` (refuse-to-predict) when no trained model is
  loaded, and the caller falls back to the heuristic gates — they never
  fabricate a probability. The only hardcoded heuristics are
  `signal_strength`/`trend_strength` feeding `calculate_dynamic_position_size`
  (transparent sizing math, not surfaced as model output, feeds no gate).
- **Punch-list (operator sign-off, NOT fixed):** the hard gates use the
  idiom `if value and value < threshold` — missing/zero metadata SKIPS the
  check rather than failing closed. This is a "missing-data-treated-as-
  benign" posture, NOT either DEX pattern, and tightening it to fail-closed
  would change rejection behavior materially. Flagged for review, left
  as-is.

- Wave-2 campaign report: `docs/agents/reports/SOLANA_CAMPAIGN.md`.
- Wave-4 report: `docs/agents/reports/SOLANA_WAVE4.md`.
- Canonical engine API: `docs/engines.md`.

## Wave-F5 fake-PnL chain closed (2026-07-06)
Per `docs/agents/wave-f5/03_solana_sniper.md`. The five-bug wrong-denomination
fake-PnL chain (poisoned quote = real USD × ~5000 → fake +526,608% TP exits,
mirror −99.98% stops, +1,176 SOL fabricated on the dashboard) is closed. The
`PriceValidator` (`core/price_validator.py`) is now the single price-trust
authority. **Contract:**
- **Validate at fetch, everywhere.** `_scan_jupiter_opportunities` and the
  `_open_position` metadata/DexScreener entry price now route through
  `_get_token_price` (validated). A candidate entry that disagrees with the
  validated quote beyond the hard ratio is discarded — a poisoned quote can no
  longer fire a fake momentum BUY or seed a poisoned entry.
- **Seed on open/reconcile, drop on close.** `price_validator.seed(mint,
  entry_price)` anchors the mint at entry (and on restart reconcile);
  `.drop(mint)` clears it on full close.
- **Quorum for hard jumps (bug 3).** A `>hard_jump_ratio` (5x) move can no
  longer auto-confirm by single-source repetition or via the `lastgood_ttl_s`
  quiet window. Under `solana_price_quorum_required=true` (default) only
  cross-source agreement (≥2 independent providers within `CONSISTENCY_BAND`)
  accepts it; `_get_token_price` fetches a second provider on a held hard jump
  to corroborate genuine moves. Otherwise it keeps HOLDING the last-good price
  (a held price can never fire TP/SL).
- **Void, don't pin (bug 4).** `_save_trade_to_db` no longer pins exit to 50×
  entry and books `notional×49` phantom profit. An implausible exit/entry is
  recomputed from the last VALIDATED price (or flat-voided), tagged
  `metadata.excluded=true` + `exclusion_reason`, raw values kept for audit.
- **Dashboard honesty (bug 5).** `/api/solana/trades` and the module-overview
  aggregate exclude `metadata.excluded=true` rows from PnL/win-rate (separate
  `excluded_count`). tz-aware ISO timestamps no longer emit `+00:00Z`
  (killed the "NaNm" position ages).
- **Data repair.** Migration 140 tags historical poisoned rows (pnl_pct ≥ 2000
  and their mirror −99.98% stops on the same mints) excluded; idempotent,
  audit-preserving.
- Config knobs: `solana_price_{soft,hard}_jump_ratio`, `solana_price_{jump,
  hard_jump}_confirmations`, `solana_price_lastgood_ttl_s`,
  `solana_price_pending_window_s`, `solana_price_quorum_required`.

## Wave-F6 entry-side fake-PnL fix + Drift repair (2026-07-10)
Per `docs/agents/wave-f6/03_trading_sweep.md` (SOLANA CRIT + Drift P1).
- **Entry-price corroboration (the ENTRY half of the F5 fix).** F5 secured
  the exit path, but 38/194 jupiter opens still booked wrong-denomination
  entries (RAY $3310 vs $0.68 → 7 fake −99.98% stops + 16 $0 time-exits).
  Two additive gates in `solana_engine.py`:
  - **Cold start never self-seeds.** `_get_token_price` now requires a
    SECOND independent provider (via `_fetch_alt_price`: CoinGecko →
    Jupiter → DexScreener, always ≠ primary) within `hard_jump_ratio`
    before the FIRST quote for a mint may be trusted/seeded as last-good;
    otherwise the quote is discarded and the price cache evicted (retried
    next poll). Covers the momentum-signal scan path too.
  - **JUPITER entries corroborate before opening.** `_open_position`
    re-checks the chosen entry price against a second source
    (`_cold_start_corroborated`); second source missing or >hard_jump_ratio
    divergence → the entry is SKIPPED (logged `failed cross-source
    corroboration`), no position, no seed. Pump.fun mints are exempt
    (no second source exists at t=0; their PnL was ≈flat).
  No knob — the gate is strictly stricter and fail-safe. If DexScreener +
  CoinGecko + Jupiter price APIs are all unreachable, jupiter entries pause
  (by design) until one recovers.
- **Drift KeyError fixed.** `drift_helper.py` passed the Solana cluster
  name `mainnet-beta` to `DriftClient`; driftpy keys its configs as
  `mainnet`/`devnet` → `KeyError` at init, Drift never connected. Now maps
  cluster-style names to the driftpy key (`DRIFT_ENV` env override,
  default `mainnet`).
- **RiskManager Drift-block WARN rate-limited.** `⛔ Drift <mkt> blocked by
  RiskManager: Insufficient liquidity` fired every scan per market (10k+
  lines — token-style liquidity validation against a perp market name).
  Block unchanged (fail-closed); WARN now ≤1/hour per market, repeats at
  DEBUG. Follow-up (not this wave): a perp-appropriate risk gate so Drift
  signals aren't structurally blocked by DEX-pool liquidity checks.
