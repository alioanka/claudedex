# PM Campaign Plan — Module Audit & Enhancement Wave 2

**Branch:** `claude/create-expert-agents-JFSF5`
**Base commit:** `ffeda0a` ([docs] multi-agent campaign brief)
**PM:** pm-architect-qa
**Date:** 2026-05-19
**DRY_RUN:** stays TRUE for every module (operator must explicitly flip)

## Context snapshot
All 8 modules are currently flagged "AMBER → GREEN candidate" in their CLAUDE.md.
The Phase 1 P0/P1 list (`docs/agents/MASTER_BACKLOG.md`) shows MB-01..MB-33 closed.
This wave (campaign brief at `docs/agents/CAMPAIGN_BRIEF.md`) is a re-audit + enhancement
sweep. Each module agent re-reads its module, ships fixes for residual issues, and
adds at least one operator-named enhancement.

## Constraint note (PM thread tooling)
The PM thread in this harness does not have a Task / Agent tool exposed, so the PM
cannot directly fork the 7 module sub-agents itself. Instead the PM publishes the
fully-formed briefs below; the parent orchestrator (or a wrapper that has `Task`
permitted) consumes this file and dispatches each module agent with the appropriate
`subagent_type`, `worktree` isolation, and `run_in_background=true`.

Each brief is self-contained — copy-paste-ready.

## Hard rules every module agent must honor
- Branch: `claude/create-expert-agents-JFSF5` only. Never main.
- Each commit ≤ 200 LoC net, descriptive message tagged with module + change.
- Push after each commit (stop-hook complains about unpushed commits).
- Never `--no-verify`, `--amend`, `--force-push`, or `docker volume prune`.
- DRY_RUN stays TRUE. Never flip a module's DRY_RUN flag to LIVE.
- `SNIPER_SAFETY_CHECK_ENABLED=false` is Phase-2 only — do NOT re-enable in this wave.
- All deliverables land at:
  - `docs/agents/reports/<MODULE>_CAMPAIGN.md` — findings + fix log
  - Each fix or enhancement = its own commit
  - Module CLAUDE.md updated to reflect new state
- No two agents may rewrite the same shared file in the same batch.
  Shared files (PM coordinates writes to these):
  - `core/risk_manager.py`
  - `core/dry_run.py`
  - `core/decision_maker.py`
  - `core/portfolio_manager.py`
  - `config/pool_engine.py`
  - `monitoring/test_runner_routes.py`
  - `dashboard/templates/test_runner.html`
  - `modules/base_module.py`

## Per-module briefs

### A1 — DEX (smartcontract-web3-expert)
**Worktree:** `worktrees/dex`
**Working files:**
- `modules/dex_trading/` (entry `main_dex.py`, engine `dex_module.py`)
- `trading/executors/direct_dex.py`
- `trading/executors/mev_protection.py`
- `dashboard/templates/dex/*.html`
- `modules/dex_trading/CLAUDE.md`
**Mission:**
1. Re-audit MB-01 (decimals), MB-02 (Flashbots EIP-191), P1-04 (pool_engine sweep)
   — verify they hold under current code.
2. Look for residual issues: divergent slippage/gas/DRY_RUN across executor entrypoints,
   missing `pool_engine` calls, missing block-tag on `min_amount_out`, gas oracle
   staleness on L2s.
3. Enhancements to consider:
   - Multi-DEX route quality scoring (Uniswap V3 vs Sushi vs Pancake)
   - MEV-protection toggle per chain (Flashbots only on Ethereum; bloXroute on BSC)
   - Per-trade decimals-correct `min_amount_out` audit with test coverage
4. Update `modules/dex_trading/CLAUDE.md` and write `docs/agents/reports/DEX_CAMPAIGN.md`.

### A2 — ARBITRAGE (smartcontract-web3-expert)
**Worktree:** `worktrees/arbitrage`
**Working files:**
- `modules/arbitrage/` (`arbitrage_engine.py`, `triangular_engine.py`, `solana_engine.py`)
- `trading/executors/` (shared with A1 — coordinate; A2 may only **read**)
- `dashboard/templates/arbitrage/*.html`
- `modules/arbitrage/CLAUDE.md`
**Mission:**
1. Re-verify MB-03 (DAI typo), MB-04 (one-legged broadcast), MB-05 (placeholder amountIn).
2. Confirm triangular path is still gated by the atomic-receiver guard. Do NOT enable
   it without operator approval — the contract isn't deployed.
3. Residual issues: spatial-arb PnL accounting (`P1-05` hardcoded $15 gas + 0.6% slippage),
   cross-module risk_manager wiring (`P1-06`), Aave V3 flash-loan receiver address
   resolution under `flash_loan_env_key` vs DB-backed secrets.
4. Enhancements: per-chain L2 gas/slippage table; hourly gas-budget tracker;
   `min_profit_bps` adaptive curve based on observed gas spike rate.
5. `docs/agents/reports/ARBITRAGE_CAMPAIGN.md` + CLAUDE.md update.

### A3 — SOLANA (smartcontract-web3-expert)
**Worktree:** `worktrees/solana`
**Working files:**
- `modules/solana_trading/` (`core/solana_engine.py`, helpers)
- `modules/solana_strategies/` (`jupiter_helper.py`, `drift_helper.py`)
- `trading/chains/solana/jupiter_executor.py`
- `dashboard/templates/solana/*.html`
- `modules/solana_trading/CLAUDE.md`
**Mission:**
1. Re-verify MB-06..MB-10 (decimals, co-signers, prio-fee, restart reconcile, DRY_RUN gate).
2. MB-15 Drift hardening — add explicit DRY_RUN gate, leverage cap, oracle-deviation guard,
   funding-rate sanity. Drift stays toggle-off by default.
3. Enhancements: Jito bundle path (toggleable), Jupiter quote freshness TTL,
   `priority_fee_lamports` adaptive controller tied to recent network percentile.
4. ML wiring (`P1-07`): wire `pump_predictor` / `rug_classifier` into `_open_position`
   under a `solana_ml_enabled` flag (default false). Audit `P1-08` look-ahead label bug.
5. `docs/agents/reports/SOLANA_CAMPAIGN.md` + CLAUDE.md update.

### A4 — SNIPER (smartcontract-web3-expert)
**Worktree:** `worktrees/sniper`
**Working files:**
- `modules/sniper/` (`core/sniper_engine.py`, `core/trade_executor.py`,
  `core/evm_listener.py`, `core/solana_listener.py`, `core/token_safety.py`)
- `data/collectors/` (mempool monitor, honeypot checker, token sniffer)
- `dashboard/templates/sniper/*.html`
- `modules/sniper/CLAUDE.md`
**Mission:**
1. Re-verify MB-11..MB-14 closures + the Phase-2 WSS volume gains.
2. Residual: per-event latency unverifiable under `getTransaction` commitment wait —
   propose & implement bounded `confirmed`-commitment readback with fallback.
3. Active-positions cap behavior + Jupiter quote price fallback regression tests.
4. Enhancements: dual-source safety check (Honeypot.is + Goplus) with quorum,
   Solana SL price-feed redundancy (Jupiter quote + Pyth + Birdeye fallback),
   per-chain latency dashboard widget.
5. DO NOT flip `safety_check_enabled=true` — that's an operator step.
6. `docs/agents/reports/SNIPER_CAMPAIGN.md` + CLAUDE.md update.

### A5 — FUTURES (market-trading-analyst)
**Worktree:** `worktrees/futures`
**Working files:**
- `modules/futures_trading/` (`core/futures_engine.py`, `futures_risk_manager.py`,
  `exchanges/`, `config/futures_config_manager.py`)
- `dashboard/templates/futures/*.html`
- `modules/futures_trading/CLAUDE.md`
**Mission:**
1. Verify the leverage-cap fix the operator referenced (commit `b1b8df9`) actually
   propagates `futures_max_leverage` → `max_leverage` end-to-end (see
   `futures_module.py:179-186`). Add a startup assertion + a smoke test that proves
   the runtime `FuturesRiskManager.max_leverage` matches DB config.
2. Re-verify MB-16..MB-18 + reconcile observability + Bybit/Binance normalizer.
3. Confirm the bot can place an order in DRY_RUN end-to-end on both Binance and
   Bybit adapters (smoke test only — no live calls).
4. Enhancements: funding-rate strategy gate (skip new longs when funding > X bps),
   per-symbol position sizing tied to ATR, isolated-margin assertion at order time
   (defense in depth on MB-17).
5. `docs/agents/reports/FUTURES_CAMPAIGN.md` + CLAUDE.md update.

### A6 — AI (quant-algo-expert)
**Worktree:** `worktrees/ai`
**Working files:**
- `modules/ai_analysis/` (`core/sentiment_engine.py`, `core/ai_provider.py`,
  `core/ai_trading_engine.py`)
- `trading/strategies/ai_strategy.py`
- `ml/` (scaler/rug/pump training scripts)
- `dashboard/templates/ai/*.html`
- `modules/ai_analysis/CLAUDE.md`
**Mission:**
1. Re-verify MB-19 (load-or-refuse scaler) closure including the 27-feature canonical
   layout + outcome backfill on close.
2. Re-verify MB-20 (executor delegation) and MB-21 (prompt-injection sanitization).
3. Enhancements:
   - Strategy generator: small Q-learning bandit selecting prompt template from a
     pinned set; persist exploration logs to `ai_feature_store`.
   - Confidence calibration: track LLM confidence vs realized outcome, ship a
     calibration plot endpoint.
   - Multi-provider quorum: when both `openai` and `anthropic` configured, require
     agreement above a threshold before acting.
4. `docs/agents/reports/AI_CAMPAIGN.md` + CLAUDE.md update.

### A7 — COPY_TRADING (quant-algo-expert)
**Worktree:** `worktrees/copy_trading`
**Working files:**
- `modules/copy_trading/` (`copy_engine.py`, `main_copy.py`)
- New: `modules/copy_trading/wallet_discovery.py`, `leader_scorer.py` (operator
  named these as broken — likely never existed)
- `dashboard/templates/copy_trading/*.html`
- `modules/copy_trading/CLAUDE.md`
**Mission (highest-priority feature work this wave):**
1. Re-verify MB-22..MB-25 closures.
2. **REBUILD wallet discovery** — operator flagged this as broken/missing.
   - New `wallet_discovery.py`: pull top wallets per chain from public data
     sources (DexScreener, Birdeye, GMGN, etc.) with rate-limited HTTP, cache to DB.
   - New `leader_scorer.py`: score wallets on (a) 30-day realized PnL,
     (b) Sharpe, (c) hit-rate, (d) avg hold time, (e) drawdown. Persist scores
     in a `copy_leader_scores` table (write migration `018_copy_leader_scores.sql`).
   - Surface top-N scored leaders in the dashboard `/copy_trading/leaders` page.
3. Leader-trade replay diagnostics: log why each detected leader tx was or was
   not mirrored (size cap, cooldown, risk-manager rejection, chain unsupported).
4. Enhancement: Kelly-fraction sizing per leader based on their score.
5. `docs/agents/reports/COPY_TRADING_CAMPAIGN.md` + CLAUDE.md update.

### (No A8) — DASHBOARD
Not in this wave per operator brief. Dashboard agents (`backend-devops-expert`)
will be spawned as T1/T2 after module work completes, primarily for Test Runner
integration of new endpoints + smoke probes.

## Sequencing & conflict map

| Shared file | A1 | A2 | A3 | A4 | A5 | A6 | A7 |
|---|---|---|---|---|---|---|---|
| `trading/executors/direct_dex.py` | RW | r | — | — | — | — | — |
| `trading/executors/mev_protection.py` | RW | r | — | — | — | — | — |
| `trading/chains/solana/jupiter_executor.py` | — | — | RW | r | — | — | — |
| `core/risk_manager.py` | r | r | r | r | r | r | r |
| `core/dry_run.py` | r | r | r | r | r | r | r |
| `core/portfolio_manager.py` | r | r | — | — | RW | — | — |
| `config/pool_engine.py` | r | r | r | r | — | — | r |
| `monitoring/test_runner_routes.py` | — | — | — | — | — | — | — (T1/T2) |
| `dashboard/templates/test_runner.html` | — | — | — | — | — | — | — (T1/T2) |
| `modules/base_module.py` | r | r | r | r | r | r | r |

RW = may edit; r = read-only; — = no access.
Any agent needing to RW a row marked `r` must publish a brief to PM_PLAN.md
addendum and wait for PM ACK before committing.

## After all module agents commit — T1 / T2 brief

**T1, T2 — Test Runner agents (backend-devops-expert)**
**Worktrees:** `worktrees/test_runner_1`, `worktrees/test_runner_2`
**Mission:**
1. `git log claude/create-expert-agents-JFSF5 ffeda0a..HEAD --oneline` — enumerate
   every fix/feature commit.
2. For each commit, add a row in `monitoring/test_runner_routes.py::TEST_CATALOG`
   with: name, module, category (api/db/probe), command or async function, expected
   pass criterion.
3. Add bash test scripts under `scripts/` where catalog rows reference a script.
4. Update `dashboard/templates/test_runner.html` only if the existing grid can't
   hold the new categories.
5. Every new HTTP endpoint introduced this wave gets:
   - one api probe (returns 2xx and expected schema)
   - one db probe if it writes (row exists post-call)
6. Report at `docs/agents/reports/TEST_RUNNER_CAMPAIGN.md`.

Split: T1 owns DEX / ARB / SOLANA / SNIPER. T2 owns FUTURES / AI / COPY_TRADING.

## PM cadence
- After this commit lands: append first status line to
  `docs/agents/reports/PM_STATUS.md`.
- Every ~30 min while agents run: another PM_STATUS.md line.
- All-modules-done: write `docs/agents/reports/PM_FINAL.md`.
