# ClaudeDex Logging Reference

Per-module log file layout, rotation policy, level conventions, and
log-shipping guidance. Pairs with `docs/runbook.md` (where to look
during incidents) and `docs/engines.md` (which engines emit which logs).

## 1. Directory layout

Every trading module writes to its own subdirectory under `logs/`:

```
logs/
├── ai_analysis/
│   ├── ai_analysis.log         # main
│   ├── ai_analysis_errors.log  # level >= ERROR
│   └── ai_analysis_trades.log  # trade events (audit trail)
├── arbitrage/
├── copy_trading/
├── dex_trading/
├── futures_trading/
├── sniper/
├── solana_trading/
├── dashboard/                  # HTTP access + auth events; no trades log
└── pool_engine/                # RPC pool health + rotation events
```

The directory name matches each module's BaseModule `name` field and the
`logs/.pause_<module>` flag-file convention (see `docs/runbook.md` §2).
The three filenames are produced by `setup_logger` in
`monitoring/logger.py:964-997`.

## 2. Rotation policy

All log files use Python's `RotatingFileHandler` with identical defaults
(`monitoring/logger.py:778-779`):

| Setting | Value |
|---|---|
| `maxBytes` | 10 MB (`10 * 1024 * 1024`) |
| `backupCount` | 10 |
| Compression | none (rotated `.1`..`.10` suffixes; uncompressed) |

Per module that means at most 33 files (main + 10, errors + 10, trades + 10),
roughly 330 MB before the oldest rotation is overwritten. To lower the
ceiling, change `max_file_size` via the `config_settings` table — do not
edit the source.

## 3. Level conventions

| Level | Use |
|---|---|
| `DEBUG` | Hot-loop noise. Off by default in production; opt-in per module. |
| `INFO` | Normal lifecycle (start, stop, fill, position open/close, killswitch detected). |
| `WARNING` | Recoverable issue worth investigating later (slippage over budget, LLM refused a sanitized headline, risk validator rejected a trade). |
| `ERROR` | Module-level failure that did not cascade (single RPC failure, single tx revert). |
| `CRITICAL` | Process-level halt (engine dead, killswitch tripped, OOM, DB unreachable). |

Risk-validator rejections should log at `WARNING`, not `ERROR`: they are
the system working as designed. Reserve `ERROR` for genuine unrecoverable
per-call failures.

## 4. Trade-log isolation (`propagate = False`)

The trade-event helpers `log_trade_entry`, `log_trade_exit`, and
`log_portfolio_update` each set `trades_logger.propagate = False`
(`monitoring/logger.py:1161`, `:1225`, `:1290`).

Effect:
- Trade events do NOT propagate to the root logger.
- They do NOT appear in `*_trading.log` or `*_errors.log`.
- They write ONLY to the dedicated trades log.

Reason: the trades log is the operator-visible audit trail. Mixing it
with verbose lifecycle / debug noise destroys auditability. Operators
shipping logs to compliance or analytics should route `*_trades.log`
to a separate index from `*_trading.log` and `*_errors.log`.

## 5. Log-shipping guidance

Production deployments should ship `logs/` off-box for compliance and
postmortem use. Recommended tenancy:

- `*_trades.log` -> long-term analytical store (S3, GCS, Loki tenant).
  Audit trail; retain at least 1 year.
- `*_trading.log` and `*_errors.log` -> short-term operational store
  (~90 days). Level-based routing should send `*_errors.log` to alerts.
- `logs/pool_engine/` -> operational store; use to investigate RPC
  provider outages.
- `logs/dashboard/` -> security store; auth events live here.

Do NOT ship `logs/.killswitch` or `logs/.pause_*` flag files. They are
operational state, not events.

### Sample fluent-bit pattern

```
[INPUT]
    Name    tail
    Path    /var/log/claudedex/*/*_trades.log
    Tag     trade
    Parser  json

[INPUT]
    Name    tail
    Path    /var/log/claudedex/*/*_errors.log
    Tag     error
    Parser  json

[OUTPUT]
    Name    forward
    Match   trade
    Host    <audit-store>

[OUTPUT]
    Name    forward
    Match   error
    Host    <alert-pipeline>
```

## 6. Where to look during an incident

| Symptom | First log |
|---|---|
| Module won't start | `logs/<module>/<module>_errors.log` first line (config / init failure) |
| Trade rejected silently | `logs/<module>/<module>.log` — look for `risk_manager rejected` |
| Slippage abnormal | `logs/<module>/<module>_trades.log` (compare fill vs quote) |
| Killswitch tripped | `logs/.killswitch` content shows reason + ts + pid |
| RPC errors recurring | `logs/pool_engine/` (look for the affected provider_type) |
| Login failed / WS rejected | `logs/dashboard/` |

## See also
- `docs/runbook.md` — incident response
- `docs/engines.md` — engine API
- Per-module `modules/<name>/CLAUDE.md`
