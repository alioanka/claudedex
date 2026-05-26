# DEX Module
## What it does
Spot trading on EVM DEXes (Uniswap V2/V3, SushiSwap, PancakeSwap) across Ethereum, BSC, Polygon, Arbitrum, Base. Routes through `trading/executors/direct_dex.py` with optional Flashbots MEV protection.
## Entry point
`modules/dex_trading/main_dex.py` — launched as a subprocess by `main.py` when `DEX_MODULE_ENABLED=true`. Live engine is the shared `core/engine.py:TradingBotEngine` (constructed in `main_dex.py:initialize`). NOTE: `modules/dex_trading/dex_module.py` (`DexTradingModule`) is a BaseModule wrapper that is NOT wired into the live engine path — the subprocess uses `TradingBotEngine` directly. Editing `dex_module.py` has no runtime effect on trading.
## Key config (DB-backed via `ConfigManager`)
- `max_slippage_bps` — per-trade slippage cap (default 50)
- `max_gas_price` — gwei ceiling for tx submission (default 50)
- `mev_protection` — toggle Flashbots private-bundle path (default True)
- `supported_dexs` — list of router names enabled for routing
- `jupiter_routing` — kept for future SOL-leg routing; currently EVM only
## Kill switch
- Global: `logs/.killswitch` (written by `scripts/emergency_stop.py` or `/api/bot/emergency-exit`; polled by BaseModule subprocesses via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_dex` (written by dashboard pause/resume; read by `core.dry_run.is_module_paused`).
- Effect: `should_skip_live` returns `True` -> live-write gates in `direct_dex.py:64,276,1013` return their `_simulate_*` path.
## Logs
`logs/dex_trading/` — main, errors, trades (rotating handler).
## Primary risk-policy gate
`core.risk_manager.RiskManager.validate_trade(token, amount)` — wired through the shared `trading/trading_engine.py` order-execution path. Per-executor caps (`max_slippage_bps`, `max_gas_price`) enforced inline at `trading/executors/direct_dex.py:88,567`.
## Live-trade readiness
AMBER → GREEN candidate (pending production verification). MB-01 (decimals — both legs now via `core.units.to_raw_evm`), MB-02 (Flashbots EIP-191) and P1-04 (`pool_engine` RPC unification) all closed.
## Wave-2 hardening (campaign `claude/create-expert-agents-JFSF5`)
- P0: `DirectDEXExecutor.max_slippage` initialized (was AttributeError when `order.slippage` unset).
- P0: `MEVProtectionLayer.protect_transaction` `bundle_id` defaulted to `None` (was `UnboundLocalError` on ADVANCED+low-risk path).
- P0: MB-01 input-side — `amount_in_raw` now uses `to_raw_evm(chain, token_in, amount)` instead of `ether_to_wei` (USDC/USDT/WBTC fix).
- P1: web3 v6 API drift — `toChecksumAddress` → `to_checksum_address`, `isAddress` → `is_address`, `isConnected` → `is_connected`, PoA middleware import made version-safe.
- P1: per-chain gas-price ceiling (`_CHAIN_MAX_GWEI_DEFAULTS` + `chain_max_gas_gwei` config override). Single 50-gwei cap was wrong on Polygon/L2s.
- P1: `_get_optimal_gas_price` runs `eth_gasPrice` via `loop.run_in_executor` (was blocking the async loop).
- P1: `mev_protection._apply_gas_randomization` clamps post-randomization gasPrice to `max_gas_price` ceiling.
- P1: Flashbots ADVANCED-tier gate now requires `chain ∈ {ethereum, eth, mainnet}` — silently falls through to private-mempool routing on BSC/Polygon/Arb/Base (Flashbots relay does not service those chains).
- Enhancement: `get_best_quote` ranks by `_score_quote = amount_out * (1 - impact) - gas_cost_native`, not raw headline output. Routes to the DEX with the best NET fill.
- Enhancement: `tests/unit/test_dex_decimals.py` — 5 regression tests for MB-01 (USDC 6 dec, WBTC 8 dec, WETH 18 dec sanity) + scoring (prefers lower gas, prefers lower impact).
## Wave-3 hardening (campaign `claude/create-expert-agents-JFSF5`)
- `_quote_v3` real Uniswap V3 QuoterV2 binding (was `int(amount * 0.997)` placeholder). Iterates fee tiers {100, 500, 3000, 10000} for single-hop, falls back to `quoteExactInput(bytes,uint256)` for multi-hop. QuoterV2 addresses seeded per chain (ETH/POLY/ARB/BASE/OP/BSC), overridable via `config['v3_quoter_addresses']`. Returns 0 (not the placeholder) when no pool exists — so route ranking treats V3 as "no liquidity" instead of silently winning. Tests: `tests/unit/test_dex_quoter_v3.py`.

## Wave-4 hardening (campaign `claude/create-expert-agents-JFSF5`)
- `_estimate_price_impact` now does a real chunked QuoterV2 round-trip (was linear extrapolation from a 0.1% sample). Quotes the actual size and a 1%-of-size probe; impact = `(eff_tiny - eff_actual) / eff_tiny`. Both legs flow through `_simulate_swap`, which already routes V3 paths through the real QuoterV2. 1% is the smallest probe that still produces a non-degenerate quote on 6-dec USDC/USDT majors. Tests: `tests/unit/test_dex_price_impact.py`.
- New `max_price_impact_bps` config (default **200 bps** — matches the Uniswap-frontend "high impact" warning). `get_best_quote` drops any DEX candidate whose impact exceeds the cap; when nothing passes it returns None instead of signing a tx that would eat the entire slippage tolerance. Override via `config['max_price_impact_bps']`.
- `MEVProtectionLayer._attempt_bloxroute_bsc` — bloXroute BDN private-tx routing on BSC (Flashbots is Ethereum-only, so BSC swaps previously had no private path). Submits `blxr_private_tx` JSON-RPC to `https://api.blxrbdn.com` with `Authorization: <CLOUD_API_KEY>` header. Gated by `chain == 'bsc' and config.get('bloxroute_enabled', False)`. Auth header read from `config['bloxroute_auth_header']` (encrypted-secret friendly) or `BLOXROUTE_AUTH_HEADER` env. Returns None on missing-header / HTTP error / RPC error / timeout — `protect_transaction` then falls back to the public-mempool send path so the tx still ships. Endpoint overridable via `config['bloxroute_bsc_endpoint']`. Ethereum Flashbots branch is unchanged. Tests: `tests/unit/test_mev_bloxroute_bsc.py`.

## Wallet / Account identity (issue 15)
EVM-only module (no SOL leg yet). Single EVM wallet is shared across all enabled
EVM chains (Ethereum, BSC, Polygon, Arbitrum, Base, Optimism, Avalanche) — same
address, fund per-chain native gas token (ETH / BNB / MATIC / etc.).
- **Private key**: secret `PRIVATE_KEY` (encrypted in DB `config_sensitive` via
  `security/encryption.py`; Fernet-encrypted `gAAAAAB...` values are decrypted in
  `main_dex.py:initialize` with `ENCRYPTION_KEY`). `.env` fallback only.
- **Public address**: secret/env `WALLET_ADDRESS` if set, otherwise DERIVED from
  the decrypted `PRIVATE_KEY` via `eth_account.Account.from_key(...).address`
  (`main_dex.py:_resolve_wallet_address`). Stored on `app.wallet_address`.
- **Surfaced** (public address only — NEVER the private key): DEX health server
  `GET :8085/health` and `GET :8085/stats` both return a `wallet_address` field
  (plus `dry_run` on `/health`). Dashboard can proxy these for display.

## Manual close — flag-file IPC (issues 5 + 6, engine side)
The dashboard process cannot call the engine directly (separate process), so the
close button uses flag-file IPC mirroring copy-trading.
- **Flag file**: `logs/.close_dex_<id>` (empty file; presence is the trigger).
- **`<id>` format**: the integer SERIAL `trades.id` — exactly the value the
  dashboard already returns as `position['id']` from `api_open_positions` /
  `api_dex_positions`. NOT `trade_id` (the UUID text column).
- **Engine side**: `modules/dex_trading/position_service.py:DexPositionService`.
  `close_flag_loop()` polls every 15s, `close_position(id)` looks the row up by
  `id` in the `trades` table (DB-first; works even after a subprocess restart
  when the engine's in-memory `active_positions` is empty), marks it
  `status='closed'` with `exit_price` (fresh quote → last-known → entry),
  `profit_loss`, `exit_timestamp`. Under DRY_RUN / kill-switch / pause it is a
  SIMULATED close (`metadata.closed_simulated=true`). Flag deleted after each
  attempt. **Dashboard agent: write `logs/.close_dex_<trades.id>` to close.**

## OPEN-position price refresh (issue 4, stuck position)
`DexPositionService.price_refresh_loop()` (started in `main_dex.py:run`) is the
source of truth for OPEN-position price/PnL on `/dex/*`. Every 30s it refreshes
`metadata.current_price` + `profit_loss` + `profit_loss_percentage` for every
OPEN `trades` row on the DEX chains, querying live price by
`metadata.pair.pair_address` (→ `dexscreener.get_pair_data`) with a
`get_token_price` fallback. **Root cause of the frozen GRAIL position:** the
shared engine only refreshes positions held in its in-memory `active_positions`
dict, and `core/engine.py:_load_state()` is a no-op — so after a restart NO open
position gets a price update and PnL freezes at entry indefinitely. This loop is
DB-first and independent of `active_positions`. If price fetch fails
`>=5` times or no update for `>=30min`, the row is flagged
`metadata.price_stale=true` + `metadata.price_stale_reason` (token likely
unroutable/illiquid) instead of silently freezing the last price.

## ML / entry-scoring review (issue 4 ML) — FINDINGS (shared infra, not edited)
The DEX entry decision is `core/engine.py:_calculate_opportunity_score` +
`TradingOpportunity.score`. This is SHARED infra used by AI/sniper too, so per
campaign rules it is documented here and NOT rewritten mid-campaign. Recommend
the owning agent address:
1. **The ML ensemble is dead weight for entries.** `EnsemblePredictor`
   (`ml/models/ensemble_model.py`) is loaded + retrained but NEVER consulted for
   entry. `pump_predictor.py` / `rug_classifier.py` / `volume_validator.py` are
   not called either. Entry is a pure heuristic (volume/liquidity/price/risk/age).
2. **`ml_confidence` is mislabeled.** `main_dex`-engine sets `ml_confidence=score`
   (the heuristic), and `pump_probability=score*0.8`, `rug_probability=0.2`
   (hardcoded constant — `rug_classifier` exists but is unused). So
   `TradingOpportunity.score` is a circular re-derivation of the heuristic, not an
   independent ML signal. `DecisionMaker` (`core/decision_maker.py`) is
   constructed but never invoked in the entry path.
3. **Risk-analysis failure is REWARDED (loss surface).** When
   `risk_manager.analyze_token` raises (caught → `risk_score=None`), the 20% risk
   weight is dropped from BOTH numerator and denominator, and the score is
   normalized `score/weights` over the smaller denominator — so a token we could
   NOT safety-check scores as if risk were perfect and can pass `min_score`.
   A missing safety signal should bias the score DOWN, not up. Recommend: treat
   `risk_score is None` as `overall_risk=1.0` (max risk) rather than omitting the
   term, or hard-reject when risk analysis fails.
Conservative DEX-side guards were NOT added because `DexTradingModule.process_opportunity`
is not wired into the live `TradingBotEngine` path (see Entry point note) — the
fix must land in the shared scorer to have effect.

## DRY_RUN data-quality notes
- OPEN DEX positions live in `trades` (`status='open'`, `side='buy'`); there is no
  `current_price` column — live price for open positions is carried in
  `metadata.current_price` (which is what the dashboard reads). The refresh loop
  above keeps it current; closed-trade PnL is the `profit_loss` column.
- Simulated closes set `metadata.closed_simulated=true` so DRY_RUN fills are
  distinguishable from real ones in audit.

## Wave-8 ML + risk-failure + state-restore (campaign `claude/create-expert-agents-JFSF5`)
The three issues flagged in the "ML / entry-scoring review" section above were
fixed in the shared scorer `core/engine.py` (the LIVE DEX entry decision via
`main_dex.py:725` → `TradingBotEngine.run()`). DRY_RUN untouched; no safety gate
loosened — every change can only make a gate stricter or more honest.

- **DEFECT 1 — risk-failure was rewarded → now penalized.**
  `_calculate_opportunity_score` previously added the 0.20 risk term only `if
  risk_score`, dropping that weight from the normalization denominator on
  failure so an un-safety-checkable token scored HIGHER. Now a missing/failed
  risk assessment is treated as WORST-CASE and the scorer **rejects outright**
  (returns 0.0) — an unverifiable honeypot/rug signal must never raise the
  score (most conservative option for a safety gate). The opportunity-construction
  default also changed from the latent-crash `RiskScore(overall_risk=0.5)`
  (`overall_risk` is a read-only @property, not a ctor arg → would TypeError) to a
  real all-1.0 worst-case `RiskScore` (now an unreachable branch).

- **DEFECT 2 — ML ensemble now consulted; heuristic fallback labeled honestly.**
  The opportunity used to fabricate `ml_confidence=heuristic_score`,
  `pump_probability=score*0.8` and a flat `rug_probability=0.2` that silently
  passed the `>0.5` rug gate. `_analyze_opportunity` now calls
  `EnsemblePredictor.predict_decoupled(token, chain, features)` (features mapped
  from data already gathered — no invented pipeline). A result is used ONLY if
  trustworthy: no `error` key and not the degenerate all-0.5 untrained
  passthrough (returned when the unfitted `RobustScaler.transform` raises — the
  state whenever NO model artifacts exist on disk, which is the case in this
  environment). Otherwise it falls back to the heuristic, tagged
  `metadata.ml_source='heuristic_fallback'` (vs `'ensemble'`), and NEVER reports a
  fake high `ml_confidence`. Fallback `rug_probability` is now score-coupled
  (`0.25..0.6`, inverse to the heuristic score) instead of the free optimistic
  `0.2` — weak tokens approach the 0.5 rug gate on their own merit. The real
  ensemble branch activates once artifacts ship via `scripts/retrain_models.py`.

- **DEFECT 3 — `_load_state` restores open positions (was `try: pass`).**
  On restart it restores `trades WHERE status='open' AND side='buy' AND chain IN
  dex_chains` into `engine.active_positions`, shaped like the live entry path
  (`entry_price`/`amount` as `Decimal`). Each restored position carries
  `trade_id` = the INTEGER `trades.id`, so the engine's close
  (`db.update_trade(id, ...)`) targets the EXACT row `position_service.py`
  manages — no duplicate/orphan rows; tagged `metadata.restored_from_db=True`.
  `DexPositionService.price_refresh_loop` remains the DB-first source of truth for
  OPEN-position price/PnL; the restored in-memory dict re-enables in-engine exit
  logic, the rug-prob exit gate and `get_stats`. Both write the same fresh quote
  to the same row, so they converge (last-writer-wins). Fail-soft: a bad load
  logs and leaves `active_positions` untouched — never crashes startup.

- **Same-pattern follow-ups — RESOLVED in Wave-9.** The "ML fabricated /
  risk-failure rewarded" pattern was audited in sniper/solana/copy and found
  ABSENT in all three (commit `50e37ea`; see `docs/agents/PM_FINAL_WAVE9.md`).
  The engine placeholder stubs were made honest (commit `37e2cfe`):
  `_check_smart_contract` now returns `verified=False` / `status='unknown'`
  (was a false `verified=True` safety signal) and the consuming gate default was
  hardened to `False`; `_analyze_holder_distribution` returns
  `{'concentrated': None, 'status': 'unknown'}`; the random-feature stub
  `_extract_features` (returned `np.random.rand(10)`, zero live callers) was
  DELETED. `_check_developer_reputation` stays a neutral `0.5` no-op (no gate
  reads it) and is labelled unimplemented. Remaining open item: the real ML
  ensemble stays in honest `heuristic_fallback` until trained artifacts exist —
  see `ml/CLAUDE.md` for the activation runbook + the retrain-pipeline gap.

## See also
- Phase 1 audit reports: `docs/agents/reports/DEX_*.md` (smartcontract / quant / analyst).
- Wave-2 campaign report: `docs/agents/reports/DEX_CAMPAIGN.md`.
- Wave-3 campaign report: `docs/agents/reports/DEX_WAVE3.md`.
- Wave-4 campaign report: `docs/agents/reports/DEX_WAVE4.md`.
- Canonical engine API: `docs/engines.md`.
