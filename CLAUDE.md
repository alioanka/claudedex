# ClaudeDex Trading Bot
Multi-strategy crypto trading bot. Each strategy runs as an independent subprocess managed by `main.py`; each can be enabled/disabled via `.env` flags.
## Modules
| Module | Dir | Entry | Verdict |
|---|---|---|---|
| DEX | `modules/dex_trading/` | `main_dex.py` | AMBER → GREEN candidate (P1-04 pool_engine sweep closed in a21ec41) |
| ARBITRAGE | `modules/arbitrage/` | `main_arbitrage.py` | AMBER → GREEN candidate (spatial; triangular path explicitly gated by atomic-receiver contract — scope cut, not defect) |
| SOLANA | `modules/solana_trading/` | `main_solana.py` | AMBER → GREEN candidate (Jupiter spot; MB-15 Drift is a toggleable feature, not a blocker) |
| SNIPER | `modules/sniper/` | `main_sniper.py` | AMBER → GREEN candidate (Phase 2 WSS volume = 2.6× polling over 22h/87k trades; per-event latency unverifiable under getTransaction commitment wait; pre-LIVE fixes shipped — active-positions cap + Jupiter quote price fallback + block_time_anchored DB propagation + LIVE-mode safety-filter startup guard; final step: flip safety_check_enabled=true in DB) |
| FUTURES | `modules/futures_trading/` | `main_futures.py` | AMBER → GREEN candidate (Bybit V5 surface complete; reconcile + position normalizer + restart-cap detection shipped) |
| AI | `modules/ai_analysis/` | `main_ai.py` | AMBER → GREEN candidate (executor delegation + secrets + prompt-injection sanitization closed) |
| COPY_TRADING | `modules/copy_trading/` | `main_copy.py` | AMBER → GREEN candidate (MB-22..MB-25, BaseModule conversion, pool_engine + secrets all closed) |
| DASHBOARD | `modules/dashboard/` | `main_dashboard.py` | AMBER → GREEN candidate (security + operational P0s + reconcile-state surface + RESTART OVER-CAP banner) |
| ADVISOR | `modules/advisor/` | `main_advisor.py` | ADVICE-ONLY, no trade execution (Wave-20+; CRYPTO/US/BIST/FX/Midas signals; default DISABLED, opt-in via `ADVISOR_MODULE_ENABLED=true`; health port 8086) |
| POLYMARKET | `modules/polymarket/` | `main_polymarket.py` | SHADOW-FIRST prediction markets (Gamma API read-only; YES+NO<1 arb + momentum signals recorded simulated; LIVE CLOB gated OFF behind shadow_mode=false AND live_execution_enabled=true AND should_skip_live AND RiskManager; default DISABLED via `POLYMARKET_MODULE_ENABLED=true` opt-in; health port 8089; mig 101) |

Each module has its own `CLAUDE.md` with entry point, config keys, kill-switch paths, log location, and risk-gate hooks.
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
## See also
- Phase 1 module audits: `docs/agents/reports/<MODULE>_*.md`
- Wave-13 module reports: `docs/agents/wave13/agent_*.md`
- Wave-14 backlog: `docs/agents/wave13/WAVE14_BACKLOG.md`
- Master backlog: `docs/agents/MASTER_BACKLOG.md`
- Multi-agent plan: `docs/agents/PLAN.md`
