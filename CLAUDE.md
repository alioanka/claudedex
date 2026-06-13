# ClaudeDex Trading Bot
Multi-strategy crypto trading bot. Each strategy runs as an independent subprocess managed by `main.py`; each can be enabled/disabled via `.env` flags.
## Modules
Enable flags follow the pattern `<MODULE>_MODULE_ENABLED` in `.env` (see `main.py` for the authoritative list). Default state is DISABLED unless noted.

### Trading modules
| Module | Dir | Entry | Verdict |
|---|---|---|---|
| DEX | `modules/dex_trading/` | `main_dex.py` | AMBER → GREEN candidate (P1-04 pool_engine sweep closed in a21ec41); default ON in docker-compose |
| ARBITRAGE | `modules/arbitrage/` | `main_arbitrage.py` | AMBER → GREEN candidate (spatial; triangular path explicitly gated by atomic-receiver contract — scope cut, not defect) |
| SOLANA | `modules/solana_trading/` | `main_solana.py` | AMBER → GREEN candidate (Jupiter spot; MB-15 Drift is a toggleable feature, not a blocker) |
| SNIPER | `modules/sniper/` | `main_sniper.py` | AMBER → GREEN candidate (Phase 2 WSS volume = 2.6× polling over 22h/87k trades; per-event latency unverifiable under getTransaction commitment wait; pre-LIVE fixes shipped — active-positions cap + Jupiter quote price fallback + block_time_anchored DB propagation + LIVE-mode safety-filter startup guard; final step: flip safety_check_enabled=true in DB) |
| FUTURES | `modules/futures_trading/` | `main_futures.py` | AMBER → GREEN candidate (Bybit V5 surface complete; reconcile + position normalizer + restart-cap detection shipped) |
| AI | `modules/ai_analysis/` | `main_ai.py` | AMBER → GREEN candidate (executor delegation + secrets + prompt-injection sanitization closed) |
| COPY_TRADING | `modules/copy_trading/` | `main_copy.py` | AMBER → GREEN candidate (MB-22..MB-25, BaseModule conversion, pool_engine + secrets all closed) |
| STAT_ARB | `modules/stat_arb/` | `main_stat_arb.py` | SHADOW-FIRST statistical pairs mean-reversion on liquid USDT perps (Bybit via ccxt); only expansion-wave module with a WIRED live order path — gated behind shadow_mode=false AND live_execution_enabled=true AND should_skip_live AND RiskManager; hard z-stop is the tail-risk control; mig 132 |
| POLYMARKET | `modules/polymarket/` | `main_polymarket.py` | SHADOW-FIRST prediction markets (Gamma API read-only; YES+NO<1 arb + momentum signals recorded simulated; LIVE CLOB gated OFF behind shadow_mode=false AND live_execution_enabled=true AND should_skip_live AND RiskManager; mig 101) |

### Advisory / meta layers (zero order paths unless noted)
| Module | Dir | Entry | Verdict |
|---|---|---|---|
| DASHBOARD | `modules/dashboard/` | `main_dashboard.py` | AMBER → GREEN candidate (security + operational P0s + reconcile-state surface + RESTART OVER-CAP banner); default ENABLED |
| ADVISOR | `modules/advisor/` | `main_advisor.py` | ADVICE-ONLY, no trade execution (Wave-20+; CRYPTO/US/BIST/FX/Midas signals) |
| META_CONTROLLER | `modules/meta_controller/` | `main_meta_controller.py` | ADVISORY-BY-DEFAULT self-deciding layer: scores rolling DRY+LIVE performance, writes ACTIVATE/KEEP/PAUSE to `meta_decisions`; autopilot gated behind DB `meta_autopilot_enabled=false`, and when enabled only writes/clears `logs/.pause_<module>` (never killswitch, never trades); migs 112-113 |
| ORCHESTRATOR_AI | `modules/orchestrator_ai/` | `main_orchestrator_ai.py` | ADVISORY-ONLY: scores DRY_RUN performance, writes to_live/to_dry/hold recommendations to `orchestrator_recommendations`; operator approves explicitly; no health server (DB heartbeats) |
| PORTFOLIO_ALLOCATOR | `modules/portfolio_allocator/` | `main_portfolio_allocator.py` | ADVISORY-ONLY: per-module capital-allocation proposals (`portfolio_allocations`, fractional-Kelly, 5-40% bounds, 10% reserve); no feedback loop auto-executes them; no health server |
| REGIME_ALLOCATOR | `modules/regime_allocator/` | `main_regime_allocator.py` | ADVISORY-ONLY: vol-regime classification (Binance/Bybit free klines) + per-module weight proposals to `regime_allocation_proposals`; migs 118-119 |
| SENTINEL | `modules/sentinel/` | `main_sentinel.py` | ADVISORY-BY-DEFAULT minutes-scale anomaly watchdog (depeg, price divergence, silent module, full rejection, loss velocity, correlated drawdown → `sentinel_anomalies`); autopilot (`sentinel_autopilot_enabled=false`) can only write/clear its own `logs/.pause_<module>` files; mig 122 |
| PARAM_TUNER | `modules/param_tuner/` | `main_param_tuner.py` | SHADOW-ONLY bandit knob-tuner: writes PROPOSALS to `param_proposals`; risk/leverage/live/secret keys hard-excluded; `auto_apply_enabled=false` default; counterfactual rewards = overfit risk, acceptance must be out-of-sample; mig 129 |

### Observability / data / scaffold modules (default OFF, no order paths)
| Module | Dir | Entry | Verdict |
|---|---|---|---|
| EXECUTION_QUALITY | `modules/execution_quality/` | `main_execution_quality.py` | Read-only TCA: quoted-vs-realized cost decomposition per trade + per-module scorecards (`tca_trade_costs`, `tca_scorecards`); mig 120 |
| TREASURY | `modules/treasury/` | `main_treasury.py` | Phase-1 OBSERVE-ONLY wallet/gas/inventory monitor (`treasury_snapshots`); never signs/transfers; mig 121 |
| MARKET_DATA_WAREHOUSE | `modules/market_data_warehouse/` | `main_market_data_warehouse.py` | OHLCV + funding-rate archive from free sources (`market_candles`, `market_series`); pure data; mig 123 |
| CATALYST_CALENDAR | `modules/catalyst_calendar/` | `main_catalyst_calendar.py` | Forward-catalyst feed: unlocks (DefiLlama), listings (Binance CMS), macro dates (`catalysts`); static macro list EXPIRES 2026-12-31; mig 124 |
| OPTIONS_VOL | `modules/options_vol/` | `main_options_vol.py` | Deribit BTC/ETH hedging ADVISOR (protective puts/collars, premium-capped, `options_vol_suggestions` simulated); BUY legs live-capable behind the full dual-flag gate chain, SELL legs hardcoded record-only; mig 125 |
| YIELD_TREASURY | `modules/yield_treasury/` | `main_yield_treasury.py` | Idle-capital carry ADVISOR (`yield_treasury_advice`); live deposit path NOT BUILT (`live_deposit_path_not_built` even with all flags open); honest low upside vs smart-contract risk; mig 126 |
| EXECUTION_GATEWAY | `modules/execution_gateway/` | `main_execution_gateway.py` | MEV-aware EVM send-policy LIBRARY (private order flow + public fallback, nonce/gas policy, kill-switch at send boundary); never originates trades, never holds keys; no module wired to it yet; mig 127 |
| CLMM_LP | `modules/clmm_lp/` | `main_clmm_lp.py` | SHADOW-ONLY concentrated-liquidity LP proposer (`clmm_shadow_positions`, net-of-IL APR); no mint/rebalance/burn code in v1 (`live_path_not_implemented`); IL/toxic-flow caveats — deferred until TCA proves quality; mig 128 |
| INTENT_SOLVER | `modules/intent_solver/` | `main_intent_solver.py` | PARKED scaffold (doc verdict: PARK IT): records simulated CoW/UniswapX fill opportunities (`intent_fill_opportunities`); ZERO live code — flags are documented no-ops; mig 130 |
| BASIS_DESK | `modules/basis_desk/` | `main_basis_desk.py` | Delta-neutral funding/basis carry ADVISOR (Bybit+Binance, full hedged-structure costing, `basis_carry_suggestions` simulated); live two-leg wiring intentionally not implemented; mig 131 |
| SMART_MONEY | `modules/smart_money/` | `main_smart_money.py` | Read-only on-chain smart-money flow scorer (forward-return wallet scores, no-look-ahead; `smart_money_signals` advisory); EVM only in v1; zero broadcast code; mig 133 |

`modules/backtest_replay/` is not a subprocess — it is invoked synchronously via the dashboard `POST /api/backtest/replay` (read-only counterfactual replay).

Each module has its own `CLAUDE.md` with entry point, config keys, kill-switch paths, log location, and risk-gate hooks.
## Health-port map
`/health` = liveness, `/status` = stats. `*` = probe-only default (no bound server; dashboard falls back to DB heartbeats). `orchestrator_ai` and `portfolio_allocator` bind no health server. Override env pattern: `<MODULE>_HEALTH_PORT`. Only 8080 is published by docker-compose — probe the rest from inside the container.

| Port | Module | Port | Module |
|---|---|---|---|
| 8080 | dashboard | 8093 | treasury |
| 8081 | futures | 8094 | sentinel |
| 8082 | solana | 8095 | market_data_warehouse |
| 8083 | sniper* | 8096 | catalyst_calendar |
| 8084 | arbitrage* | 8097 | options_vol |
| 8085 | dex | 8098 | yield_treasury |
| 8086 | advisor | 8099 | execution_gateway |
| 8087 | ai* | 8100 | clmm_lp |
| 8088 | copy_trading* | 8101 | param_tuner |
| 8089 | polymarket | 8102 | intent_solver |
| 8090 | meta_controller | 8103 | basis_desk |
| 8091 | regime_allocator | 8104 | stat_arb |
| 8092 | execution_quality | 8105 | smart_money |
## Run flows
Run-from-CLI: `python main.py` launches every module whose `*_MODULE_ENABLED=true` flag in `.env` as a subprocess. Each subprocess writes to its own `logs/<module>/` directory.

Run-from-dashboard: the dashboard runs independently on port 8080 and does NOT require trading modules to be running. Authenticate at `/login`, then use per-module enable/disable/pause controls under `/modules`. Emergency-stop button at the top of every page hits `/api/bot/emergency-exit`.
## Safety primitives
- `core/dry_run.py` — `should_skip_live(module_dry_run, *, module, account)` returns `True` iff module DRY_RUN is set, the global kill switch is set, or the module is paused via `logs/.pause_<module>`.
- `logs/.killswitch` — flag file written by `scripts/emergency_stop.py` or the dashboard's `/api/bot/emergency-exit`. Polled by every BaseModule subprocess via `core.dry_run.start_killswitch_poller`.
- `core/risk_manager.py` — `RiskManager.validate_trade(token, amount)`. Called by ARB, AI, SOLANA (entry-only), and COPY_TRADING (BUY-only) execution paths before broadcast. FUTURES has its own `FuturesRiskManager.validate_new_position`.
- `config/pool_engine.py` — `PoolEngine.get_endpoint(provider_type)` is the single source of RPC URLs across all on-chain modules.
## Wave-13 status (2026-05-30)
| Module | Wave-13 changes |
|---|---|
| DEX | AMBER→GREEN: BUG P0 scoring fixed (vol/liq threshold 2.0→0.05, was blocking 100% of candidates); ML ensemble wired (artifacts needed); auto-retrain loop fixed; EVM chain discovery strategy-4 added. Mig 036 seeds min_vol_liq_ratio + ml_retrain keys. |
| ARBITRAGE | AMBER→GREEN: -10005bps bug fixed (19/31 pairs had wrong direction, WETH borrow vs non-WETH token — now 22 valid pairs). Pair-direction runtime guard added. |
| SOLANA | AMBER→GREEN: datetime tz crash fixed in _save_trade_to_db; pnl_pct clamped to [-100,2000]% at write. Mig 034 alters timestamps to TIMESTAMPTZ. |
| SNIPER | AMBER→GREEN: absurd PnL bug fixed (USD/native unit mismatch in monitor); WSS 429 detection + RPC rotation. Mig 037 adds compound index + max_hold_minutes seed. Wave-13 PM: RiskManager wired (entry-only, fail-soft). |
| FUTURES | AMBER→GREEN: volume gate demoted to diagnostic-only (was blocking 100% of signals at 0.80x threshold vs live 0.17-0.76x range); log noise fixed; testnet log clarified. |
| AI | AMBER→GREEN: Claude model 404 fixed (haiku-latest→sonnet-20241022, now DB-configurable); dead news source dropped; confidence threshold lowered 0.50→0.35 to match observed signal distribution. |
| COPY_TRADING | AMBER→GREEN: Solana wallet derived from PK at executor source (was None → Jupiter 400); EVM V3+aggregator + Solana Orca/Meteora detection added (was missing 22 method IDs and 11 DEX programs). |
| DASHBOARD | AMBER→GREEN: DEX ml_source badge + ensemble version; copy probation/exposure knobs; AI model IDs + confidence decimal fix + health badge; pool fallback tier badge. |
| INFRA | AMBER→GREEN: pool_engine anti-starvation fallback + exp backoff; secrets_manager idempotent re-init; migration 035 seeds Solana/BASE/FANTOM public RPC fallbacks. |
## Recovery wave addendum (2026-06-11)
Salvage/repair wave on top of Waves 14-25: sniper revival baseline (mig 089), futures FUT-RM-27 per-symbol tiering + rolling gate (mig 088), copy leader lifecycle (mig 092), arbitrage economics gates (LIVE execution default OFF behind `live_execution_enabled`), Solana pump.fun LIVE gating via `should_skip_live` + `pumpfun_live_enabled` knob (mig 096), Solana PriceValidator wired into `_get_token_price`, DEX week-1 tuning seeds + DB-first exit watchdog — watchdog closes are DRY_RUN-only (mig 095), dashboard per-module runtime-status endpoint + honest runtime badges, advisor Fonoloji activation auto-prefer (mig 083). Stale-worktree clobber from 0cb4c3a repaired in 46092cd. Note: migration numbers 084-087, 090-091, 093-094 are intentionally unused; `min_net_spread_bps_<chain>` / `live_execution_enabled` arb keys have no seed migration yet (code defaults are fail-safe OFF).
## Enhancement wave 2 addendum (2026-06-11)
DEX entry-side tunables consumed (`trading.max_trades_per_day` + `chain_weights`; position SL/TP now read DB `risk_management.stop_loss_pct`/`take_profit_pct`). Dashboard pause writes the SHORT engine key (`logs/.pause_<short>`); `/api/trades/history` scoped to DEX legacy `trades` table, fail-soft. Arbitrage econ-gate seeds shipped in mig 097 (closes the "no seed migration yet" note above). Solana mig 100 drops the dead `('solana_pumpfun','enabled')` row with a guarded operator-intent copy. NEW POLYMARKET module (mig 101). NEW default-OFF strategies, both seeded by mig 102: futures funding-carry v2 (`futures_funding_carry_enabled`, entries route through `_open_position` so every risk gate incl. `validate_new_position` applies) and AI cross-source confirmation (`ai_confirmation_signal_enabled`, filter-only — can only skip entries, never adds them; pure tape math, no LLM spend). Health-port map: 8080 dashboard, 8081 futures, 8082 solana, 8083 sniper*, 8084 arbitrage*, 8085 dex, 8086 advisor, 8087 ai*, 8088 copy_trading*, 8089 polymarket (*probe-only defaults — those modules bind no health server; dashboard falls back to DB heartbeats). COPYTRADING probe default moved 8085→8088 (was colliding with DEX); AI probe default moved 8086→8087 (was colliding with advisor).
## Live-readiness wave addendum (2026-06-12)
Live-flip hardening across all trading modules (migs 103-113; 114 intentionally unused; all new seeds preserve DRY_RUN-safe defaults — no live-execution flag defaults to true). DEX: LIVE entry/exit path made real (RiskManager.validate_trade fail-closed at broadcast, kill-switch/pause at the broadcast boundary, native-unit orders, DRY_RUN plumbing wired — flipping to LIVE was previously a silent no-op; knobs in mig 105; manual close routes through on-chain sell). SOLANA: late-confirmation rescue window (mig 106) + Drift risk-gate tuple bug fixed (gate always passed). SNIPER: LIVE Solana sends confirmed before booking, restart reconcile + exit-retry backoff (mig 107), failed partial-take no longer mutates position state, EVM leg quoted amount_out + approve-before-sell. ARBITRAGE: tx receipt confirmed before booking LIVE fills — reverted = gas burnt, not profit (mig 108); shadow/killswitch gates enforced in solana_engine. FUTURES: close-path honesty — never paper-close LIVE positions while orders are blocked (mig 109); ccxt open_long/open_short fix. COPY: execution tunables + durable idempotency + EVM exit mirroring + BUY-only caps (mig 110). AI: legacy-path LLM budget gate (core/llm_budget.try_consume) + per-trade notional ceiling (mig 111). NEW META_CONTROLLER module (migs 112-113, see table). Dashboard v4: control-center page (`/api/control-center/overview`, `/api/performance/cross-module`, `/api/meta/decisions`) + advisor sim performance panel (`/api/advisor/performance`, migs 103-104). Health port 8090 = meta_controller (extends the wave-2 port map).
## Module-expansion wave addendum (2026-06-13)
Sixteen new default-OFF subprocess modules registered in `main.py` (flags in `.env`, all `false`), migs 118-133 (+ mig 135 copy-trading v3 discovery/shadow-sim; 114-117 and 134 intentionally unused). Two groups: advisory/meta (regime_allocator migs 118-119, sentinel 122, param_tuner 129) and observability/data/strategy (execution_quality 120, treasury 121, market_data_warehouse 123, catalyst_calendar 124, options_vol 125, yield_treasury 126, execution_gateway 127, clmm_lp 128, intent_solver 130, basis_desk 131, stat_arb 132, smart_money 133). Design invariants verified against code 2026-06-13 (zero doc drift found): every module polls `logs/.killswitch` + `logs/.pause_<module>`; no seed flips any live flag to true; modules with a live concept use the dual-flag chain (`shadow_mode=true` AND `live_execution_enabled=false` defaults) + `should_skip_live` + RiskManager. Live-capability honesty: only STAT_ARB has a wired live order path (gated OFF); OPTIONS_VOL BUY legs are live-capable behind the full chain (SELL legs record-only); CLMM_LP/BASIS_DESK/YIELD_TREASURY terminate in `live_path_not_implemented`-class skip reasons even with all gates open; INTENT_SOLVER is a parked scaffold with zero live code; the rest are read-only. Actuation honesty: sentinel + meta_controller autopilots (both default OFF in DB) can only write/clear pause files; param_tuner auto-apply (default OFF) can only change whitelisted non-risk config keys. Health ports 8091-8105 extend the map (full table above). Skeptical carry-notes from the research docs stand: clmm_lp and yield_treasury have low/negative expected edge until TCA + HODL-benchmark accounting prove otherwise; param_tuner counterfactual rewards overfit in shadow; intent_solver is documented to explain why NOT to build it.
## See also
- Deployment from clean host: `docs/DEPLOYMENT_GUIDE.md`
- Day-to-day operation: `docs/OPERATIONS_GUIDE.md`
- One-screen module index: `docs/MODULE_CATALOG.md`
- Module research/skeptic docs: `docs/agents/NEW_MODULE_IDEAS.md`, `docs/agents/NEW_STRATEGY_BACKLOG.md`
- Phase 1 module audits: `docs/agents/reports/<MODULE>_*.md`
- Wave-13 module reports: `docs/agents/wave13/agent_*.md`
- Wave-14 backlog: `docs/agents/wave13/WAVE14_BACKLOG.md`
- Master backlog: `docs/agents/MASTER_BACKLOG.md`
- Multi-agent plan: `docs/agents/PLAN.md`
