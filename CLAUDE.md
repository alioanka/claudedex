# ClaudeDex Trading Bot
Multi-strategy crypto trading bot. Each strategy runs as an independent subprocess managed by `main.py`; each can be enabled/disabled via `.env` flags.
## Modules
| Module | Dir | Entry | Verdict |
|---|---|---|---|
| DEX | `modules/dex_trading/` | `main_dex.py` | AMBER → GREEN candidate (P1-04 pool_engine sweep closed in a21ec41) |
| ARBITRAGE | `modules/arbitrage/` | `main_arbitrage.py` | AMBER → GREEN candidate (spatial; triangular path explicitly gated by atomic-receiver contract — scope cut, not defect) |
| SOLANA | `modules/solana_trading/` | `main_solana.py` | AMBER → GREEN candidate (Jupiter spot; MB-15 Drift is a toggleable feature, not a blocker) |
| SNIPER | `modules/sniper/` | `main_sniper.py` | AMBER → GREEN candidate (Phase 2 WSS volume = 2.6× polling over 22h/87k trades; per-event latency unverifiable under getTransaction commitment wait; pre-LIVE remaining: re-enable safety filter — active-positions cap + Jupiter quote price fallback shipped) |
| FUTURES | `modules/futures_trading/` | `main_futures.py` | AMBER → GREEN candidate (Bybit V5 surface complete; reconcile + position normalizer + restart-cap detection shipped) |
| AI | `modules/ai_analysis/` | `main_ai.py` | AMBER → GREEN candidate (executor delegation + secrets + prompt-injection sanitization closed) |
| COPY_TRADING | `modules/copy_trading/` | `main_copy.py` | AMBER → GREEN candidate (MB-22..MB-25, BaseModule conversion, pool_engine + secrets all closed) |
| DASHBOARD | `modules/dashboard/` | `main_dashboard.py` | AMBER → GREEN candidate (security + operational P0s + reconcile-state surface + RESTART OVER-CAP banner) |

Each module has its own `CLAUDE.md` with entry point, config keys, kill-switch paths, log location, and risk-gate hooks.
## Run flows
Run-from-CLI: `python main.py` launches every module whose `*_MODULE_ENABLED=true` flag in `.env` as a subprocess. Each subprocess writes to its own `logs/<module>/` directory.

Run-from-dashboard: the dashboard runs independently on port 8080 and does NOT require trading modules to be running. Authenticate at `/login`, then use per-module enable/disable/pause controls under `/modules`. Emergency-stop button at the top of every page hits `/api/bot/emergency-exit`.
## Safety primitives
- `core/dry_run.py` — `should_skip_live(module_dry_run, *, module, account)` returns `True` iff module DRY_RUN is set, the global kill switch is set, or the module is paused via `logs/.pause_<module>`.
- `logs/.killswitch` — flag file written by `scripts/emergency_stop.py` or the dashboard's `/api/bot/emergency-exit`. Polled by every BaseModule subprocess via `core.dry_run.start_killswitch_poller`.
- `core/risk_manager.py` — `RiskManager.validate_trade(token, amount)`. Called by ARB and AI execution paths before broadcast. FUTURES has its own `FuturesRiskManager.validate_new_position`.
- `config/pool_engine.py` — `PoolEngine.get_endpoint(provider_type)` is the single source of RPC URLs across all on-chain modules.
## See also
- Phase 1 module audits: `docs/agents/reports/<MODULE>_*.md`
- Master backlog: `docs/agents/MASTER_BACKLOG.md`
- Multi-agent plan: `docs/agents/PLAN.md`
