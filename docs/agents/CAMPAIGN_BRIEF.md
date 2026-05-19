# Multi-Agent Module Audit & Enhancement Campaign

**Launched:** 2026-05-19
**Branch:** `claude/create-expert-agents-JFSF5`
**Status:** ACTIVE — work continues until operator says stop.

## Mission
Every trading module gets a dedicated 20-year experienced agent. Each module agent:
1. **Reviews** every engine file, dashboard page, settings page, log, and DB table touching their module.
2. **Reports** issues found in `docs/agents/reports/<MODULE>_CAMPAIGN.md`.
3. **Fixes** each issue with a small commit (≤200 LoC) on `claude/create-expert-agents-JFSF5`.
4. **Enhances** the module: missing strategies, tuned parameters, new behaviors, better UX. Examples:
   - COPY_TRADING wallet-discovery / good-wallet-scorer is broken — rebuild it
   - FUTURES funding-rate strategy gating
   - SNIPER WSS reliability + safety filter timing
   - ARBITRAGE triangular path correctness
   - SOLANA Drift integration toggle
   - AI prompt-injection hardening + strategy generator
   - DEX MEV-protection + multi-DEX routing quality
5. **Documents** all changes in module CLAUDE.md.

Module agents have authority to **redesign entire engines** subject to PM approval.

## Agents

| Agent | Module | Expertise |
|---|---|---|
| A1 | DEX | smartcontract-web3-expert |
| A2 | ARBITRAGE | smartcontract-web3-expert |
| A3 | SOLANA | smartcontract-web3-expert |
| A4 | SNIPER | smartcontract-web3-expert |
| A5 | FUTURES | market-trading-analyst |
| A6 | AI | quant-algo-expert |
| A7 | COPY_TRADING | quant-algo-expert |
| **PM** | **All** | **pm-architect-qa** — 20-yr QA + SW + smart-contract + Web3 + PM |
| T1, T2 | Test Runner | backend-devops-expert (spawned by PM after module work) |

## Rules
- **Branch:** Only `claude/create-expert-agents-JFSF5`. Never main.
- **Commits:** ≤200 LoC each, descriptive message + Claude-Code session footer.
- **No --no-verify, no --amend, no force-push.**
- **DRY_RUN must stay true** for every module unless operator explicitly enables LIVE.
- **SNIPER_SAFETY_CHECK_ENABLED=false** is Phase-2 only — re-enable before live.
- **No volumes prune** on docker — postgres data lives inside trading-postgres.
- **Push after each commit batch.** Stop-hook complains otherwise.
- **Document everything** in CLAUDE.md.

## Workflow
1. PM agent reads this brief, allocates a worktree-isolation strategy for each module agent.
2. Module agents work in parallel; report back to PM via individual report files.
3. PM reviews each report, approves enhancements, sequences commits to avoid conflicts.
4. After all 7 modules are GREEN, PM spawns T1/T2 to add Test Runner coverage for every new fix/feature.
5. Continuous autonomous loop until operator types "stop".

## Module campaign deliverables (per module)
- `docs/agents/reports/<MODULE>_CAMPAIGN.md` — findings + fix log
- Fixed bugs (each its own commit)
- New strategies / behaviors / settings (each its own commit)
- Updated `modules/<module>/CLAUDE.md`
- New Test Runner entries (handled by T1/T2 at end)
