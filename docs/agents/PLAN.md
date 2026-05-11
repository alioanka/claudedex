# Multi-Agent Enhancement Plan — ClaudeDex

Branch: `claude/create-expert-agents-JFSF5`

## Specialists

| Agent | Role | Owns |
|---|---|---|
| `smartcontract-web3-expert` | EVM/Solana on-chain, MEV, RPC, flash-loans | `modules/{dex_trading,solana_trading,solana_strategies,sniper,arbitrage}`, `trading/executors`, `trading/chains/solana`, `data/collectors/{chain_data,mempool_monitor,honeypot_checker,token_sniffer}.py`, `contracts/` |
| `quant-algo-expert` | Strategies, ML, signals, backtesting | `modules/ai_analysis`, `trading/strategies`, `ml/`, `analysis/` |
| `market-trading-analyst` | Risk, sizing, P&L, live-readiness | `modules/{futures_trading,copy_trading}`, `core/{risk_manager,portfolio_manager,decision_maker}.py`, cross-module risk policy |
| `backend-devops-expert` | Dashboard, infra, DB, secrets, RPC pool | `modules/dashboard`, `config/`, `security/`, `monitoring/`, `observability/`, `data/storage/`, `Docker*`, `kubernetes/`, `main.py` |
| `pm-architect-qa` | Coordination, docs, tests, dead-code cleanup | `docs/`, all `CLAUDE.md`, `tests/`, refactor sequencing |

## Operating rules
1. All commits on `claude/create-expert-agents-JFSF5`. Tag with `[smartcontract] / [quant] / [analyst] / [backend] / [pm] <module>: <change>`.
2. **Small batches**: each commit ≤ ~200 lines net, single logical change. Push after each commit.
3. **No two specialists edit the same file in the same batch.** PM agent schedules.
4. **Gap analysis first, code second.** Every module gets a report under `docs/agents/reports/<module>_<agent>.md` before code changes land.
5. **Profitability gate**: any new/changed strategy states edge, expected cost (fees+slippage+gas+MEV), kill-switch, and capital rule.
6. **Secrets**: must move from `.env` to encrypted DB (`config_sensitive`) via `security/encryption.py`. Only module-enable flags + infra URLs stay in `.env`.
7. **RPC**: every chain client must use `config/pool_engine.py` — no direct `os.getenv('*_RPC_URL')` in module code.
8. **Live-readiness checklist** (from `market-trading-analyst.md`) must be green per module before flipping `DRY_RUN=false`.

## Module ownership matrix

| Module | Primary | Secondary |
|---|---|---|
| `DEX_MODULE` | smartcontract-web3-expert | quant-algo-expert, market-trading-analyst |
| `FUTURES_MODULE` | market-trading-analyst | quant-algo-expert, backend-devops-expert |
| `SOLANA_MODULE` | smartcontract-web3-expert | quant-algo-expert |
| `SNIPER_MODULE` | smartcontract-web3-expert | quant-algo-expert (scoring) |
| `AI_MODULE` | quant-algo-expert | backend-devops-expert |
| `ARBITRAGE_MODULE` | smartcontract-web3-expert | quant-algo-expert (math), market-trading-analyst (risk) |
| `COPY_TRADING_MODULE` | market-trading-analyst | quant-algo-expert |
| `DASHBOARD_MODULE` | backend-devops-expert | pm-architect-qa |

## Phases

### Phase 0 — bootstrap (this batch)
- [x] Define 5 expert agents (`.claude/agents/*.md`)
- [x] Create this PLAN.md
- [ ] Create `docs/agents/reports/` skeleton (lazily, as reports land)

### Phase 1 — gap analysis (one batch per module pair, small commits)
Per module, the assigned specialists each drop a `<module>_<agent>.md` report and the PM agent merges them into `<module>_combined.md` with a ranked action list. No code yet.

Order (4 batches):
1. `DEX_MODULE`, `ARBITRAGE_MODULE`
2. `SOLANA_MODULE`, `SNIPER_MODULE`
3. `FUTURES_MODULE`, `AI_MODULE`
4. `COPY_TRADING_MODULE`, `DASHBOARD_MODULE`

### Phase 2 — infrastructure (must precede module rewrites)
- Secrets migration: `.env` → encrypted `config_sensitive` (backend agent)
- RPC pool refactor: remove all direct RPC env reads in modules (backend + smartcontract)
- DB-backed config for all module settings (backend agent)
- Dashboard scaffolding: per-module pages + Settings/Guide tabs (backend agent)

### Phase 3 — module enhancement (per-module, ranked by ROI from Phase-1 reports)
For each module: ≤ 5 commits per session, each addressing one ranked finding. Live-readiness checklist must pass before module is flipped from `DRY_RUN=true` to `false`.

### Phase 4 — new strategies / modules (only after Phase 3 stable)
Quant agent proposes; analyst gates risk; smartcontract agent implements execution; backend agent wires dashboard + config; PM agent docs and tests.

### Phase 5 — documentation, dead-code removal, hardening
PM agent owns. `CLAUDE.md`, `docs/{architecture,engines,logging,dashboards,staging,deployment,issues,runbook}.md`, and `*.backup` / duplicate-engine cleanup.

## Profitability playbook (the answer to user item 8)
Documented separately in `docs/runbook.md` once Phase 2 is done. Outline:
- Capital tiers: how much risk per module, scaling rule.
- Daily routine: pre-market checks, mid-day P&L review, end-of-day reconciliation.
- Weekly: model retrain, leader pool refresh (copy-trading), strategy parameter review.
- Monthly: backtest re-validation, contract & API key rotation.
- Red flags: drawdown thresholds, when to freeze, when to flatten.
