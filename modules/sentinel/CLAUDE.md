# SENTINEL Module (ADVISORY-BY-DEFAULT anomaly watchdog)

## What it does
Cross-module **anomaly detector + optional auto-freeze advisor**. One tick
(default 60s): collect a two-source price board (Coinbase + Kraken free public
REST), per-module heartbeat ages from `*_runtime_stats`, candidate-flow
counters, and rolling-window PnL per module; run six pure-threshold detectors;
persist graded anomalies to `sentinel_anomalies` with refire dedup. It watches
the minutes-scale failure modes no per-module gate sees. It **never trades,
never touches `logs/.killswitch`, never edits a risk gate**.

Boundary vs meta_controller: meta_controller scores PERFORMANCE over days;
sentinel detects ANOMALIES over minutes. Both actuate (when enabled) only
through the identical pause-file mechanism.

## Entry point
`modules/sentinel/main_sentinel.py` — launched by `main.py` when
`SENTINEL_MODULE_ENABLED=true` (default **false**). Health server on port 8094
(`SENTINEL_HEALTH_PORT`): `/health` liveness, `/status` last-tick summary.
Engine: `core/sentinel_engine.py`; pure detectors in `core/detectors.py`
(self-tested: `python -m modules.sentinel.core.detectors`); price feed in
`core/price_board.py`. Per-module trade-table schema is single-sourced from
`modules.orchestrator_ai.core.orchestrator_engine._MODULE_QUERIES` so the meta
layers never drift on table/column names.

## Detectors (each individually DB-toggleable)
| Detector | Fires on | Severity ladder |
|---|---|---|
| `stable_depeg` | stablecoin off $1 by `depeg_warn_bps` (50) / `depeg_critical_bps` (150) | WARNING → CRITICAL |
| `price_divergence` | Coinbase-vs-Kraken reference-asset gap > `divergence_warn_bps` (100) / `divergence_critical_bps` (300) | WARNING → CRITICAL |
| `silent_module` | heartbeat age > `heartbeat_warn_factor` (3x) / `heartbeat_critical_factor` (10x) expected refresh | WARNING → CRITICAL |
| `full_rejection` | a module sees ≥ `rejection_min_seen` (25) candidates and rejects 100% | WARNING |
| `loss_velocity` | rolling loss rate > `loss_velocity_warn_usd_per_hr` (15) / `loss_velocity_critical_usd_per_hr` (50) over `loss_velocity_window_minutes` (60) | WARNING → CRITICAL |
| `corr_drawdown` | ≥ `corr_drawdown_min_modules` (3) modules each losing > `corr_drawdown_module_loss_usd` (5); CRITICAL past `corr_drawdown_critical_total_usd` (100) | WARNING → CRITICAL |

Heartbeat coverage (v1): `dex` (dex_runtime_stats, 150s), `sniper` (120s),
`ai` (1800s), `arbitrage` (per-chain, 300s). Modules WITHOUT a runtime_stats
heartbeat (futures, solana, copy_trading) are **out of scope for the
silent-module detector in v1**.

## Autopilot (freeze path — OFF by default)
- Gate: DB `sentinel_autopilot_enabled` (default **false**). Advisory-only
  until the operator flips it.
- When ON: only CRITICAL anomalies from the freeze-eligible set
  (`loss_velocity`, `correlated_drawdown` — asserted in detectors self-test)
  may freeze the offending module(s) by writing `logs/.pause_<module>` — the
  SAME flag every engine polls via `core.dry_run.is_module_paused`. A pause
  freezes ENTRIES; modules can still exit positions.
- Dwell-guarded per module (`autopilot_dwell_minutes`, 360). Sentinel clears
  ONLY pause files it wrote itself (marker text "sentinel autopilot FREEZE")
  once the anomaly has stayed clear for `unfreeze_clear_minutes` (60).
- Every freeze/unfreeze is audited in `sentinel_actions`.

## Key config (DB-backed, config_type='sentinel'; migration 122)
| Key | Default | What it does |
|---|---|---|
| `sentinel_autopilot_enabled` | false | Master actuation gate (advisory-only when false) |
| `tick_interval_seconds` | 60 | Tick cadence |
| `autopilot_dwell_minutes` | 360 | Min gap between freezes of the same module |
| `unfreeze_clear_minutes` | 60 | Anomaly-clear time before sentinel removes its own pause |
| `anomaly_refire_minutes` | 30 | Dedup window — persisting condition bumps `last_seen_at`/`fire_count` instead of new rows |
| `alive_window_hours` | 24 | Module considered in-scope if it traded within this window |
| `price_fetch_timeout_seconds` | 8 | Price-board HTTP timeout |
| `depeg_*`, `divergence_*`, `silent_module_*`, `full_rejection_*`, `loss_velocity_*`, `corr_drawdown_*` | see table above | Per-detector enable flags + thresholds |

Env knobs: `SENTINEL_MODULE_ENABLED` (gate, default false),
`SENTINEL_HEALTH_PORT` (8094).

## Kill switch
- Global: `logs/.killswitch` — tick skipped (module only READS the flag;
  poller via `core.dry_run.start_killswitch_poller`).
- Per-module: `logs/.pause_sentinel` — tick skipped.

## Logs
`logs/sentinel/` — `sentinel.log` (INFO), `sentinel_errors.log` (ERROR+).

## DB tables (migration 122)
- `sentinel_anomalies` — one open row per (detector, subject, severity) with
  refire dedup (`fire_count`, `last_seen_at`).
- `sentinel_actions` — audit log of every autopilot freeze/unfreeze.

## Isolation / safety
Fail-soft everywhere: any per-detector or per-module error is logged and
skipped; the tick continues. No paid LLM — pure thresholds the operator can
re-derive by hand. Never imports an executor, never reads private keys, never
writes `logs/.killswitch`. The ONLY thing it can ever actuate (autopilot ON)
is the per-module pause file, and it can only remove pauses it created.
