# META_CONTROLLER Module

## What it does
Self-deciding / self-improving **advisory** layer. Reads every trading module's
rolling DRY_RUN + LIVE performance, scores each module's health/edge, and writes
a transparent **ACTIVATE / KEEP / PAUSE** decision per module to `meta_decisions`.
**ADVISORY BY DEFAULT** — it only writes decision rows. When
`meta_autopilot_enabled=true` it may additionally actuate decisions by
writing/clearing `logs/.pause_<module>` (the same short key the engines poll),
subject to a per-module dwell guard. It **never places a trade, never touches
`logs/.killswitch`, and never edits a risk gate.**

## Entry point
`modules/meta_controller/main_meta_controller.py` — launched by `main.py` when
`META_CONTROLLER_MODULE_ENABLED=true` (default **false**). Health server on port
8090 (`META_CONTROLLER_HEALTH_PORT`). Decision math: `core/health_scorer.py`
(pure, self-tested: `python -m modules.meta_controller.core.health_scorer`).
Engine/loop/persistence/actuation: `core/meta_engine.py`.

## Decision math (every number is operator-derivable — no black box)
Per module, two tracks (DRY and LIVE) are scored separately then blended:
```
score = 0.25*win_rate + 0.25*sharpe + 0.20*expectancy + 0.15*profit_factor + 0.15*(1-drawdown_frac)
health = live_weight*live_score + (1-live_weight)*dry_score   # LIVE dominates when both exist
confidence = min(1, closed_trades / n_target)
```
Decision rules, in order: (1) `< min_trades` → KEEP (conf 0, never actuates);
(2) HARD LOSS: LIVE window loss `< -pause_loss_usd` (or 2× on a dry-only track)
→ PAUSE regardless of score; (3) score band with hysteresis —
`health ≤ pause_score` → PAUSE, `health ≥ activate_score` AND paused → ACTIVATE,
else KEEP. Confidence below `min_confidence` downgrades any PAUSE/ACTIVATE to an
advisory KEEP. The per-module schema map is single-sourced from
`orchestrator_ai.core.orchestrator_engine._MODULE_QUERIES` so the two meta
layers never drift on table/column names.

## Self-improvement loop
Each tick `meta_engine._record_calibration` scores how well PAST decisions
matched the forward realized PnL that followed them (PAUSE was right if the next
window lost money; KEEP/ACTIVATE right if it did not) and records a hit-rate +
per-module detail to `meta_calibration`. This is bookkeeping the operator reads
to judge the controller and tune `meta_config` thresholds — it actuates nothing
on its own. (Bounded auto-tuning of thresholds is a documented follow-up, kept
proposal-only for now by design.)

## Key config (DB-backed, config_type='meta_config'; migration 112)
| Key | Default | What it does |
|---|---|---|
| `meta_autopilot_enabled` | false | Master autopilot. false = advisory only |
| `tick_interval_seconds` | 900 | Decision cadence |
| `lookback_hours` | 24 | Performance + calibration window |
| `autopilot_dwell_minutes` | 720 | Min minutes between two actuations of the same module |
| `pause_score` | 0.35 | Health ≤ this → PAUSE |
| `activate_score` | 0.55 | Health ≥ this (if paused) → ACTIVATE (gap = hysteresis) |
| `min_confidence` | 0.5 | Below → downgrade actuation to advisory KEEP |
| `pause_loss_usd` | 25.0 | Hard-loss PAUSE threshold (2× on dry-only) |
| `min_trades` | 10 | Window floor below which decision is always KEEP |
| `live_weight` | 0.7 | LIVE blend weight when both tracks exist |
| `n_target` | 50 | Trades at which confidence saturates |

## Kill switch
- Global: `logs/.killswitch` — tick skipped; autopilot never actuates while present.
- Per-module: `logs/.pause_meta_controller` — tick skipped.

## Logs
`logs/meta_controller/` — `meta_controller.log`, `meta_controller_errors.log`.

## DB tables
- `meta_decisions` (migration 112) — one row per module per tick; dashboard reads the latest per module.
- `meta_calibration` (migration 113) — self-improvement hit-rate history.

## Dashboard surface (owned by the dashboard agent)
Read-only + fail-soft: latest `meta_decisions` per module (decision + health +
confidence + reason) and the `meta_calibration` hit-rate trend. Hide the panel
when the tables are absent.

## Isolation / safety
Reuses read-only: `core/dry_run.py` (killswitch poll), `orchestrator_ai`'s schema
map, the `*_trades` tables. Writes only `meta_decisions`, `meta_calibration`, and
(autopilot only) `logs/.pause_<module>` flags. Never imports a trading executor,
never signs anything.

## New-module / engine backlog (deep-thought ideas, NOT built — operator triage)
1. **Cross-venue funding & basis desk** — a unified module that nets perp funding
   (Bybit/Binance, already partly in futures funding-carry v2) against spot/DEX
   inventory to run genuinely delta-neutral carry. Edge: funding is a persistent,
   measurable risk premium; the missing piece is the automated spot hedge leg.
   Feasibility: medium (needs a spot execution venue wired); risk: low (market-neutral).
2. **On-chain "smart-money" flow follower** — generalize copy_trading beyond named
   leaders to cluster-detected accumulation (wallets that repeatedly front profitable
   moves), scored by realized forward return. Edge: information, not prediction.
   Feasibility: medium (data-collector heavy); risk: medium (crowded-trade decay).
3. **Volatility-regime allocator** — a meta-engine that sizes EACH module's capital
   to the current vol regime (trend modules up in expansion, mean-revert/market-make
   up in compression), feeding portfolio_allocator. Edge: regime-conditional sizing
   beats static allocation. Feasibility: high (pure analytics on existing data);
   risk: low (advisory sizing only). **Highest ROI / lowest risk of the three.**
