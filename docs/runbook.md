# ClaudeDex Operator Runbook

Incident response and routine operational procedures for the
ClaudeDex bot. Use alongside `docs/engines.md` (engine API) and
each `modules/<name>/CLAUDE.md`.

## 1. Emergency stop

### Quick path (halt new entries; existing positions stay open)
```bash
python scripts/emergency_stop.py
```
Writes `logs/.killswitch` (a flag file every BaseModule subprocess
polls every ~1s) AND POSTs `/api/bot/emergency-exit`. Either
channel alone is sufficient.

### Web UI
Click the red "EMERGENCY STOP" button at the top-right of any
dashboard page. Confirms via modal, then POSTs
`/api/bot/emergency-exit`, which writes the flag file AND walks
every registered module calling `close_position()` on each open
position.

### What it does
- Sets the process-local global kill switch
  (`core.dry_run.set_global_kill_switch(True)`).
- Writes `logs/.killswitch` with a JSON payload (reason, ts, pid).
- BaseModule subprocesses' poller (auto-started via
  `__init_subclass__`) detects the file and flips THEIR local
  kill switch.
- Every `should_skip_live(...)` call thereafter returns `True`;
  live-broadcast paths divert to simulated equivalents.
- Existing positions stay open. To also flatten:
  `python scripts/close_all_positions.py` (same endpoint, which
  flattens).

### To resume
Delete `logs/.killswitch` AND restart the orchestrator (or
specific modules). The kill switch is intentionally
sticky-by-restart.

## 2. Per-module pause / resume

Cosmetic-pause was the entire MB-30 bug. The fix uses the same
flag-file pattern.

### Pause one module without halting others
- Dashboard: per-module pause toggle on the module's tile
  (admin-gated).
- CLI:
  ```bash
  echo '{"reason":"manual","ts":"'$(date -u +%Y-%m-%dT%H:%M:%SZ)'"}' \
    > logs/.pause_<module>
  ```
  Where `<module>` is one of: `dex_trading`, `arbitrage`,
  `solana_trading`, `sniper`, `futures_trading`, `ai_analysis`,
  `copy_trading`.

### Resume
Delete `logs/.pause_<module>` (the dashboard `/resume` route
does this).

### Effect
`is_module_paused(module)` returns `True`; `should_skip_live(...)`
returns `True` for that module only; its live-write gates simulate.

## 3. Secrets rotation

### One-time initial setup
```bash
python scripts/init_auth.py
```
Creates the admin user with a `secrets.token_urlsafe(24)` random
password printed ONCE to stdout (NOT to logs). Record it
immediately. If the admin user already exists with the
known-leaked `admin123` hash, `init_auth.py` detects and
force-rotates.

### Routine rotation (admin password)
```bash
python scripts/unlock_admin.py                  # random + printed once
python scripts/unlock_admin.py --password X     # explicit override
```

### Per-secret rotation (API keys, wallet keys, RPC keys)
All sensitive credentials live in the `config_sensitive` table
(encrypted via `security/encryption.py`). To rotate:
1. Insert/update via the dashboard `/settings/credentials` UI
   (admin-gated, CSRF-protected).
2. Or `python scripts/migrate_credentials_to_db.py --key <KEY> --value <VAL>`.
3. Subprocesses pick up new values on next restart. For
   zero-downtime rotation, restart each affected module.

## 4. RPC pool health investigation

`config/pool_engine.py` tracks endpoint health (success / failure
counts, rate-limit windows, rolling latency) and rotates
automatically.

### Check status
- Dashboard: `/settings/rpc_api` shows per-endpoint health.
- CLI: query the `rpc_api_pool` table directly. Relevant columns:
  `status` (`active` / `rate_limited` / `unhealthy` / `disabled`),
  `priority`, `weight`, `health_score`, `consecutive_failures`,
  `success_count`, `failure_count`, `rate_limit_until`,
  `rate_limit_count`, `last_rate_limit_at`, `last_success_at`,
  `last_failure_at`, `avg_latency_ms`.

### Investigate a misbehaving provider
1. Tail `logs/pool_engine/*.log` for the affected `provider_type`.
2. Inspect the latest failure rows in `rpc_api_usage_history`
   (`error_type`, `error_message`, `latency_ms`) joined on
   `rpc_api_pool.id`. The pool table does NOT store a `last_error`
   column; failures live in usage history.
3. Manually demote: update `priority` in the DB (higher = lower).
4. Health checks run automatically on a 1h interval; or trigger
   via `await pool.run_health_checks()` from a Python REPL.

## 5. Common failure modes per module

### DEX (`modules/dex_trading/`)
- **Symptom:** all swaps revert. First check token decimals; MB-01
  (`trading/executors/direct_dex.py:560`) routes through
  `core/units.to_raw_evm`. If a new token's decimals fetch fails,
  swap reverts. Second: Flashbots auth; MB-02 fixed
  (`mev_protection.py:388`) the EIP-191 signature path. If
  MEV-protected sends fall back to public mempool unexpectedly,
  check Flashbots relay rejection in logs.

### ARBITRAGE (`modules/arbitrage/`)
- **Symptom:** spatial arb skips opportunities silently. Check
  `RiskManager.validate_trade` rejections (P1-06, commit
  `47998c1`): circuit-breaker tripped, daily-loss cap, hourly
  gas burn.
- **Symptom:** triangular shows zero trades. By design (MB-05):
  entry-guarded pending atomic-receiver contract.

### SOLANA (`modules/solana_trading/`)
- **Symptom:** positions wiped on restart. MB-09 reconciliation
  reads `solana_positions` on startup. If rows are missing, check
  `_save_position_to_db` errors in `logs/solana_trading/`.
- **Symptom:** Drift perps unprotected. MB-15 deferred; Drift
  helper has no DRY_RUN gate yet. Disable Drift strategies until
  follow-up lands.

### SNIPER (`modules/sniper/`)
- **Symptom:** Solana snipes never stop-out. MB-14 swapped the
  price feed (`api.jup.ag/price/v2`); check endpoint reachability.
- **Symptom:** detection latency feels unchanged. P1-12 follow-up;
  no WebSocket / Geyser subscription yet, polling is still 15s+.

### FUTURES (`modules/futures_trading/`)
- **Symptom:** bot fails to start. MB-16
  (`futures_engine.py:240`); ensure `DRY_RUN` env var is set.
- **Symptom:** Bybit live trade fails. MB-17b shipped V5 helpers
  but `get_balance`, `get_position`, `close_position`,
  `get_all_positions` are still placeholder stubs.

### AI (`modules/ai_analysis/`)
- **Symptom:** no signals. MB-19; strategy refuses to predict
  until a fitted scaler exists at `models/ai_strategy_scaler.pkl`.
  Run a (forthcoming) `scripts/train_ai_strategy_scaler.py`.
- **Symptom:** adversarial sentiment swings. MB-21 sanitization
  is in `_analyze_with_llm` and `_analyze_with_claude`; verify the
  headlines passing them. Suspicious patterns log at `WARNING`.

### COPY_TRADING (`modules/copy_trading/`)
- **Symptom:** detected wrong-chain copies. MB-24 added per-chain
  routing for ethereum / bsc / polygon / base. Arbitrum, Optimism,
  Avalanche return a clean "unsupported" error.
- **Symptom:** Solana SELL not closing. MB-22; verify the executor
  reaches `_get_solana_token_balance` and the Jupiter call. If
  balance is 0, "no position to close" is logged and the DB row
  is left open.

### DASHBOARD (`modules/dashboard/`)
- **Symptom:** PANIC SELL button does nothing. Verify the
  endpoint reaches `module_routes.py:bot_emergency_exit` (MB-31
  fixed the dash / underscore URL mismatch).
- **Symptom:** WebSocket disconnects. MB-26 requires an
  authenticated session cookie; re-login if `connect` rejects.

## 6. Log paths

Per-module logs all live under `logs/`:
- `logs/<module>/<module>_trading.log` (main, 10MB rotating, 5
  backups)
- `logs/<module>/<module>_errors.log` (errors only, 5MB rotating)
- `logs/<module>/<module>_trades.log` (trade events, 10MB
  rotating; separate logger with `propagate=False`)

Cross-cutting:
- `logs/pool_engine/` — RPC pool health, rate limits
- `logs/dashboard/` — auth events, request access logs

## 7. See also
- `docs/engines.md` — engine public API
- `docs/agents/MASTER_BACKLOG.md` — outstanding follow-ups
- Per-module `modules/<name>/CLAUDE.md`
</content>
</invoke>