# COPY_TRADING Wave-4 (CT-Q-09 + CT-Q-12)

**Branch:** `claude/create-expert-agents-JFSF5`
**Date:** 2026-05-20
**Agent:** A7 (quant-algo-expert)
**Carry-over closed:** CT-Q-09 (per-leader probation) + CT-Q-12 (cross-module exposure aggregator).

## Wave-4 commits (this carry-over)

| # | SHA | Subject | LoC |
|---|---|---|---|
| 1 | `f5953cd` | CT-Q-09 + CT-Q-12 scaffold: migration 030 + engine probation + cross-module gate | 211 |
| 2 | `ba4d15d` | CT-Q-09 follow-up: probation-loss trigger in `_log_copy_trade` | 21 |
| 3 | `142250b` | CT-Q-09b: `leader_scorer.upsert_score` score-based auto-bench (>=10 trades) | ~40 (multi-module) |
| 4 | `d0a6356` | CT-Q-12 file: `exposure_aggregator.py` + engine `_check_cross_module_exposure` | 322 (rolled into AI-tag commit by orchestrator) |
| 5 | this | CLAUDE.md + WAVE4 report | doc-only |

Some Wave-4 commits were authored under unrelated `[ai]`/`[wave-4]` tags by the orchestrator's auto-commit pass; the `git log -- modules/copy_trading/*` view shows the real provenance.

## CT-Q-09 — per-leader probation table

**Schema (migration 026 + 030).** `copy_leader_scores` columns: `on_probation BOOLEAN`, `probation_until TIMESTAMPTZ`, `probation_reason TEXT`, `probation_set_at TIMESTAMPTZ`. Migration 030 aligns the default loss-pct threshold from the conservative 15% in 026 to the operator-approved 25%, and seeds the `copy_`-prefixed alias keys.

**Engine gate** (`copy_engine._is_leader_on_probation` → both BUY paths):
```
if on_probation AND probation_until > NOW():
    refuse BUY  -> [replay] reason=probation
SELLs always allowed (they reduce exposure)
```
Re-entry is automatic on expiry (the gate treats `probation_until <= NOW()` as "not on probation" without needing the row to be cleared).

**Auto-trigger sources.**
1. *Loss-pct trigger* (`_log_copy_trade` SELL branch): if `profit_loss_pct <= -copy_probation_loss_pct_threshold`, calls `_maybe_set_probation(reason="loss_<pct>pct")` for `copy_probation_days`.
2. *Score-drop trigger* (`leader_scorer.upsert_score`): if `score < copy_probation_score_threshold` AND `trade_count_30d >= 10`, UPDATE sets the probation columns atomically with `reason="score_<v>_lt_<thr>"`. The 10-trade floor prevents a single bad fill from benching a new leader.

`_maybe_set_probation` uses `GREATEST(existing, NEW())` so re-calling extends the probation window rather than overwriting -- longest cooldown wins.

**Tunables** (all `config_settings.copytrading_config`, engine accepts both bare and `copy_`-prefixed forms):
| Key | Type | Default | Purpose |
|---|---|---|---|
| `copy_probation_gate_enabled` | bool | `true` | Master switch |
| `copy_probation_score_threshold` | number | `30` | Score below which auto-bench fires |
| `copy_probation_loss_pct_threshold` | number | `25` | Abs PnL% loss that benches the leader |
| `copy_probation_days` | number | `7` | Bench duration |

## CT-Q-12 — cross-module exposure aggregator

**Design note.** Five independent trading modules can open positions in the same token without seeing each other. A leader posting a moonshot on Twitter can trigger AI (sentiment), SNIPER (new-pair detector), DEX (momentum) AND COPY (wallet mirror) to all enter the same bag simultaneously. With per-module $100 caps that's a $500 unhedged position the operator never sized for; with Kelly multipliers in play it can be much larger.

**New module:** `modules/copy_trading/exposure_aggregator.py`
- `async get_exposure_usd(chain, token_address, db_pool) -> float` — total USD exposure summed across:
  - DEX → `trades.usd_value` (status='open')
  - SNIPER → `sniper_trades.entry_usd` (status='open')
  - SOLANA → `solana_positions.entry_usd` (when the column exists; noop otherwise -- see carry-over)
  - COPY → `copytrading_trades.entry_usd` (status='open')
  - AI → `ai_trades.entry_usd` (status='open')
- `async get_exposure_breakdown_usd(...)` — per-module dict variant for diagnostic logging.

**Token matching.** Chain-scoped (`chain = $1`), case-insensitive on EVM addresses (`lower()` both sides). Each per-module sum is independently fail-soft: a missing table / missing column / type mismatch logs at DEBUG and returns 0.0 -- the aggregator returns the partial sum from modules that did respond rather than blocking trading.

**Engine wiring** (`copy_engine._check_cross_module_exposure`):
```
intended_usd = copy_amount_native * native_price
allow, existing, breakdown = aggregator.check(...)
if existing + intended > cap:
    refuse BUY  -> [replay] reason=cross_module_cap extra={existing, intended, cap, breakdown}
```
The breakdown is only computed on the refusal path (the happy path is one SUM per module, not two).

**Tunables.**
| Key | Type | Default | Purpose |
|---|---|---|---|
| `copy_cross_module_exposure_check_enabled` | bool | `true` | Master switch |
| `copy_cross_module_exposure_cap_usd` | number | `5000` | Per-token cap across all modules |

## Profitability checklist

| Item | Wave-4 entry |
|---|---|
| Edge source | Risk-reduction: probation cuts off bad leaders sooner; exposure cap prevents un-hedged fanout into the same token. |
| Modelled costs | None added (these are GATES, not trades). They REDUCE expected slippage + drawdown by refusing already-bad bets. |
| Sharpe / hit-rate impact | Expected: + on Sharpe (lower variance from concentrated single-token blowups), neutral-to-positive on hit-rate (probation removes bottom-quartile leaders before they fire more losers). |
| Capital-allocation rule | Quarter-Kelly per leader (unchanged from Wave-2); now further gated by per-token total cap. |
| Kill-switch condition | Both gates have an enable flag. Engine fail-soft if DB lookup fails -- per-module caps + RiskManager remain. |
| Validation | Walk-forward by construction: `copy_leader_scores.score` only consumes trades whose `exit_timestamp` is in-window; the aggregator only reads OPEN positions (no look-ahead into closed-trade outcomes). |

## Carry-over for Wave-5 (deferred)

- **`solana_positions.entry_usd`.** SOLANA owns the schema add. Until then the aggregator under-counts open Solana exposure (it picks up SOL-closed trades in `solana_trades`, but live Solana positions held in `solana_positions` are skipped). Workaround: COPY's own Solana open positions ARE counted via `copytrading_trades`.
- **Dashboard surfacing.** The four probation knobs + two exposure knobs need to land on the Settings page. Engine reads from DB already; UI lift is for the dashboard agent.
- **Test coverage.** Unit tests for `exposure_aggregator` (DB-mocked) and probation lifecycle (trigger → expiry → re-entry). T1/T2 wave.
