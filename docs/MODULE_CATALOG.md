# ClaudeDex Module Catalog

One-screen index of every module. Enable flag pattern: `<MODULE>_MODULE_ENABLED`
in `.env` (authoritative list in `main.py`). All modules default DISABLED except
dashboard (and DEX, which docker-compose defaults to `true`). Ports marked `*`
are probe-only (no bound health server). Full detail per module:
`modules/<dir>/CLAUDE.md`. See also `docs/DEPLOYMENT_GUIDE.md` and
`docs/OPERATIONS_GUIDE.md`.

Type legend: **T** = trade-executing (live-capable), **A** = advisory/meta
(writes proposals/signals only), **O** = observability/data (read-only).

## Trading modules

| Module | Purpose | Port | config_type | Output tables | Type | Risk note |
|---|---|---|---|---|---|---|
| dex_trading | EVM DEX spot (UniV2/V3, Sushi, Pancake; 5 chains) | 8085 | dex_config | `trades` (legacy) | T | LIVE gated by DRY_RUN + RiskManager fail-closed at broadcast; set `live_max_execute_retries=1` pre-LIVE |
| futures_trading | Binance/Bybit perps, multi-strategy, ATR SL/TP | 8081 | futures sections | `futures_trades`, `futures_positions`, `futures_funding_payments` | T | Momentum stack empirically weak — entries suppressed by default (mig 066); own `FuturesRiskManager` |
| solana_trading | Jupiter spot + optional pump.fun snipe + Drift perp leg | 8082 | solana_config | `solana_positions`, `solana_trades` | T | Drift and pump.fun sub-flags default OFF; RiskManager entry-only |
| sniper | Multi-chain new-pool/token sniping, safety filters | 8083* | sniper_config | `sniper_trades` | T | DRY_RUN exit P&L is MODELED, not measured; flip `safety_check_enabled=true` pre-LIVE |
| arbitrage | Spatial cross-DEX arb, Aave V3 flash loans | 8084* | arbitrage_config | `arbitrage_trades` | T | `live_execution_enabled=false` default; triangular path scope-cut; hourly gas-budget cap |
| ai_analysis | LLM sentiment/news signals, optional futures execution | 8087* | ai_config | `ai_trades`, `sentiment_logs`, `ai_feature_store` | T | `direct_trading=false` default (shadow); LLM spend capped via `core/llm_budget` |
| copy_trading | Mirrors leader wallets (EVM + Solana), Kelly sizing | 8088* | copytrading_config | `copytrading_trades`, `copy_leader_scores` | T | BUY-only caps + probation + cross-module exposure cap; leader scoring has survivorship-bias risk |
| stat_arb | Statistical pairs mean-reversion on liquid USDT perps | 8104 | stat_arb | `stat_arb_trades`, `stat_arb_spread_state` | T | Only expansion-wave module with a WIRED live path (Bybit ccxt) — dual-flag gated OFF; hard z-stop = tail-risk control; pair-break risk is real |
| polymarket | Prediction-market arb + momentum (Gamma API) | 8089 | polymarket_config | `polymarket_trades`, `polymarket_signals` | T | Shadow-first; LIVE CLOB behind dual flags + RiskManager; binary oracle-resolution risk |

## Advisory / meta layers

| Module | Purpose | Port | config_type | Output tables | Type | Risk note |
|---|---|---|---|---|---|---|
| dashboard | Web UI + REST/WS control plane, emergency stop | 8080 | env/DB | writes `config_settings` only | A | Admin auth + CSRF; writes the killswitch on emergency-exit |
| advisor | Multi-market advice (crypto/US/BIST/FX/Midas), ADVICE-ONLY | 8086 | advisor_config | `advisor_advice`, `advisor_sim_positions`, `advisor_portfolio` | A | Zero execution; sim positions are paper per channel |
| meta_controller | Scores module performance, ACTIVATE/KEEP/PAUSE decisions | 8090 | meta_config | `meta_decisions`, `meta_calibration` | A | Autopilot (`meta_autopilot_enabled=false`) only writes pause files; calibration is retrospective only |
| orchestrator_ai | DRY_RUN performance → to_live/to_dry recommendations | none | env knobs | `orchestrator_recommendations` | A | Operator approves explicitly; no health server (DB heartbeats) |
| portfolio_allocator | Capital-allocation proposals (fractional Kelly, 5-40%) | none | env knobs | `portfolio_allocations` | A | Proposals only; nothing auto-executes them; no health server |
| regime_allocator | Vol-regime classification + per-module weight proposals | 8091 | regime_allocator | `regime_snapshots`, `regime_allocation_proposals` | A | Free kline sources can stale; tick skipped on dual-source failure |
| sentinel | Minutes-scale anomaly watchdog (depeg, silent module, loss velocity...) | 8094 | sentinel | `sentinel_anomalies`, `sentinel_actions` | A | Autopilot default OFF; when ON can only write/clear its OWN pause files; futures/solana/copy lack heartbeats in v1 |
| param_tuner | Bandit knob-tuning PROPOSALS for whitelisted keys | 8101 | param_tuner | `param_proposals`, `param_bandit_state` | A | Risk/leverage/live/secret keys hard-excluded; counterfactual rewards overfit in shadow — judge out-of-sample only |

## Observability / data / scaffold (default OFF, no order paths)

| Module | Purpose | Port | config_type | Output tables | Type | Risk note |
|---|---|---|---|---|---|---|
| execution_quality | TCA: quoted-vs-realized cost per trade + module scorecards | 8092 | execution_quality | `tca_trade_costs`, `tca_scorecards` | O | Zero market risk; read-only over trade ledgers |
| treasury | Phase-1 wallet/gas/inventory observer | 8093 | treasury | `treasury_snapshots` | O | Never signs/transfers/decrypts keys; Phase 2 (top-up/sweep) does not exist |
| market_data_warehouse | OHLCV + funding archive (Binance/Bybit free) | 8095 | market_data_warehouse | `market_candles`, `market_series` | O | Failure = wasted disk, never lost money |
| catalyst_calendar | Unlocks/listings/macro forward-event feed | 8096 | catalyst_calendar | `catalysts` | O | Static macro list EXPIRES 2026-12-31 — emits nothing after |
| options_vol | Deribit BTC/ETH hedging advisor (puts/collars, premium-capped) | 8097 | options_vol | `options_vol_suggestions`, `options_vol_surface` | A/T | BUY legs live-capable behind dual flags + RiskManager + monthly premium budget; SELL legs hardcoded record-only |
| yield_treasury | Idle-capital carry advisor (Aave/jitoSOL/stETH math) | 8098 | yield_treasury | `yield_treasury_advice` | A | Live deposit path NOT BUILT; honest low upside vs smart-contract risk — deliberately advice-only |
| execution_gateway | MEV-aware EVM send-policy LIBRARY (private flow + fallback) | 8099 | execution_gateway | `execution_gateway_sends` | O | Never originates trades, never holds keys; no caller wired yet; per-chain private-send flags default false |
| clmm_lp | CLMM LP range proposer (UniV3/Orca), net-of-IL APR | 8100 | clmm_lp | `clmm_shadow_positions` | A | v1 has NO mint/rebalance/burn code; IL + toxic-flow caveats — approve only against HODL-benchmark accounting |
| intent_solver | CoW/UniswapX fill-opportunity recorder — PARKED scaffold | 8102 | intent_solver | `intent_fill_opportunities` | O | ZERO live code; flags are documented no-ops; research verdict: do not build the filler |
| basis_desk | Delta-neutral funding/basis carry advisor (full hedged costing) | 8103 | basis_desk | `basis_carry_suggestions` | A | Live two-leg wiring intentionally not implemented; small default notional ($200) if ever wired |
| smart_money | On-chain smart-wallet flow scorer (forward-return, no look-ahead) | 8105 | smart_money | `smart_money_wallet_events`, `_scores`, `_signals` | O | Zero broadcast code; EVM only in v1; crowding soft-cap on signals |

`modules/backtest_replay/` (no port, no subprocess): counterfactual replay via
dashboard `POST /api/backtest/replay`; read-only; rejected-trade P&L is biased
(no market impact / fill assumptions).
