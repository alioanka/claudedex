# EXECUTION_QUALITY Module (TCA)

## What it does
Read-only **Transaction Cost Analysis** observer. Every tick it reads the
trailing window of closed trades from every trading module's own table
(DEX `trades`, `solana_trades`, `futures_trades`, `arbitrage_trades`,
`sniper_trades`, `copytrading_trades`, `ai_trades`), decomposes
quoted-vs-realized execution cost per trade, and persists:
- one **per-trade** row to `tca_trade_costs` (idempotent — unique
  `(module, trade_ref)`, re-scores are no-ops), and
- one **per-module scorecard** row to `tca_scorecards` per tick
  (time-series; dashboard reads the latest per module).

**PURELY OBSERVES.** It never places a trade, never writes
`logs/.killswitch` or any `logs/.pause_*` flag, never touches an order path,
and uses no paid LLM. The only DB writes are the two `tca_*` tables.

## Entry point
`modules/execution_quality/main_execution_quality.py` — launched by `main.py`
when `EXECUTION_QUALITY_MODULE_ENABLED=true` (default **false**). Health server
on port 8092 (`EXECUTION_QUALITY_HEALTH_PORT`): `GET /health` (liveness) and
`GET /status` (latest tick summary). Cost math: `core/cost_model.py` (pure,
self-tested: `python -m modules.execution_quality.core.cost_model`).
Fetch/normalize/persist loop: `core/tca_engine.py` (offline-smoke-testable;
table names drift-checked against
`orchestrator_ai.core.orchestrator_engine._MODULE_QUERIES`, warn-only).

## Cost formulas (every number operator-recomputable)
All metrics are COSTS in bps of trade notional; positive = lost to execution,
negative = price improvement.
```
fee_bps   = fee_usd   / notional_usd * 10000
gas_bps   = gas_usd   / notional_usd * 10000
extra_bps = extra_usd / notional_usd * 10000     # flash-loan fee, tips, modeled
                                                 # slippage already deducted upstream
entry slippage (long buy / short sell entry), vs the CAPTURED quote:
  buy leg : (realized - quoted) / quoted * 10000   # paid more  = cost
  sell leg: (quoted - realized) / quoted * 10000   # received less = cost
exit leg uses the opposite direction of the entry leg.
total_cost_usd    = fee + gas + extra + adverse slippage legs (favorable
                    slippage is reported but never charged as cost)
total_cost_bps    = total_cost_usd / notional_usd * 10000   # KNOWN components only
cost_to_gross_pct = total_cost_usd / |gross_pnl_usd| * 100
mev_suspect       = quote-measured entry slippage >= sandwich_suspect_bps
```
Honesty rules: a trade with no captured quote gets NULL slippage, never a
guess (`quote_covered=false`; coverage % is on every scorecard). The DEX
table's recorded `slippage` column is used as a clearly-flagged fallback
(`entry_slippage_source='recorded_no_quote'`) that never trips the MEV flag.
Unit assumptions (e.g. legacy `trades.gas_fee` treated as USD) are written to
the `components` JSON instead of being hidden.

## Quote capture (coverage grows over time)
The engine looks for these metadata keys on each trade row (first hit wins):
entry quote — `quoted_entry_price`, `expected_entry_price`, `expected_price`,
`quote_price`, `quoted_price`; exit quote — `quoted_exit_price`,
`expected_exit_price`; gas — `gas_cost_usd`, `gas_usd`, `gas_cost`,
`gas_fee_usd`; fees — `fees_usd`, `fee_usd`, `total_fees_usd`; extras —
`flash_loan_cost`, `jito_tip_usd`, `priority_fee_usd`, `tip_usd`.
Modules that start writing a quoted price at decision time become
slippage-measurable with no change here.

## Key config (DB-backed, config_type='execution_quality'; migration 120)
| Key | Default | What it does |
|---|---|---|
| `tick_interval_seconds` | 1800 | Seconds between TCA cycles (slow loop) |
| `lookback_hours` | 24 | Closed-trade window scored each tick |
| `max_rows_per_module` | 500 | Per-tick fetch cap per module table |
| `min_trades_for_scorecard` | 3 | Below this, scorecard is written but breaches are NOT evaluated |
| `include_simulated` | true | Score DRY_RUN fills too (LIVE vs sim always distinguishable) |
| `sandwich_suspect_bps` | 150 | Quote-measured entry slippage >= this flags `mev_suspect` |
| `fee_warn_bps` | 30 | Scorecard breach threshold: avg fee cost |
| `gas_warn_bps` | 50 | Scorecard breach threshold: avg gas cost |
| `slippage_warn_bps` | 50 | Scorecard breach threshold: avg entry slippage |
| `total_cost_warn_bps` | 100 | Scorecard breach threshold: avg total cost |

Env fallbacks (pre-migration boot): `EXECUTION_QUALITY_TICK_INTERVAL`,
`EXECUTION_QUALITY_LOOKBACK_HOURS`.

## Kill switch
- Global: `logs/.killswitch` — tick skipped (and the standard poller runs).
- Per-module: `logs/.pause_execution_quality` — tick skipped.

## Logs
`logs/execution_quality/` — `execution_quality.log`, `execution_quality_errors.log`.

## DB tables (migration 120)
- `tca_trade_costs` — one row per scored trade; unique `(module, trade_ref)`.
- `tca_scorecards` — one row per module per tick; `breaches` JSON lists
  threshold violations (empty = clean).

## Dashboard surface (owned by the dashboard agent)
Read-only + fail-soft: latest `tca_scorecards` per module (avg_total_cost_bps,
quote_coverage_pct, mev_suspect_count, breaches) and a per-trade drill-down
from `tca_trade_costs`. Hide the panel when the tables are absent.

## Isolation / safety
Reuses read-only: `core/dry_run.py` (killswitch poll), orchestrator_ai's
schema map (drift check only), the `*_trades` tables. Writes only
`tca_trade_costs` and `tca_scorecards`. Never imports a trading executor,
never signs anything, zero market risk. Safe to run always-on.
