# ClaudeDex Staging & Promotion Workflow

Operator guide for promoting modules from paper-trading to live capital.
Pairs with `docs/runbook.md` (incident response) and `docs/engines.md`
(engine API). See also `docs/OPERATIONS_GUIDE.md` (shadow → DRY_RUN → LIVE
path for ALL modules incl. the expansion wave) and `docs/DEPLOYMENT_GUIDE.md`
section 9 (pre-LIVE punch-list).

## 0. Three-stage promotion path

Every module passes through:

1. **DRY_RUN** — full strategy logic runs but every live-broadcast gate
   (`core.dry_run.should_skip_live`) returns True. Fake order IDs / tx
   hashes are stamped onto trades. No real capital touched.
2. **Testnet / canary** — connect to a non-mainnet venue or run on
   mainnet with a hard capital cap (≤ 1% of book) and hour-level
   monitoring.
3. **Production** — mainnet at the operator's target allocation.

Never skip stage 2. Every module's verdict in
`docs/agents/MASTER_BACKLOG.md` is AMBER, not GREEN.

## 1. What DRY_RUN looks like per module

Set `DRY_RUN=true` in `.env`. The parser
(`core/dry_run.py:74` `resolve_dry_run_env`) is **default-safe**: a
missing OR unparseable value resolves to `True`. Live trading requires
the explicit string `false`/`0`/`no`/`off`.

| Module | Live-write path that short-circuits | Fake-fill marker |
|---|---|---|
| DEX | `direct_dex` skips `eth.send_raw_transaction`; routes return a simulated receipt | `DRY_RUN_<sha_first16>` |
| ARBITRAGE | `arbitrage_engine._execute_flash_swap` / `triangular_engine` short-circuit and log `Flash Swap Executed (DRY RUN)` | logged, no on-chain tx |
| SOLANA | `JupiterExecutor` and `solana_engine` close-path simulate via Jupiter quote | `DRY_RUN_<uuid_hex16>` (close path: `DRY_RUN_CLOSE_<uuid_hex16>`) |
| SNIPER | `trade_executor._simulate_buy` / `_simulate_sell` gated by `should_skip_live` (MB-13) | `DRY_RUN_<fake_hash16>` + `is_simulated=true` on `sniper_trades` row |
| FUTURES | `_execute_binance_futures` / `_execute_bybit_futures` short-circuit to `_simulate_trade` | `DRY_RUN_<sha_first12>` |
| AI | `sentiment_engine._simulate_trade` runs before delegating to the futures executor | `DRY_RUN_<fake_hash12>` |
| COPY_TRADING | `copy_engine._simulate_evm_swap` / `_simulate_solana_swap` | `DRY_RUN_<fake_hash16>` |
| DASHBOARD | n/a — doesn't trade. Always operational. | — |

A red `LIVE TRADING` / blue `DRY-RUN` badge sits top-right on every
dashboard page (MB-32). Confirm the badge before moving to stage 2.

## 2. Testnet / canary config

### Env flags
- `FUTURES_TESTNET=true` — Binance/Bybit futures route to testnet
  endpoints. See `futures_engine.py:_initialize_config` (~:255) for the
  precedence chain (env var > DRY_RUN=false safety default > database
  `general_config.testnet`).
- `DASHBOARD_HTTPS=false` (LOCAL DEV ONLY) — disables the `secure`
  session-cookie flag in `auth_routes.py` and `auth/csrf.py`. Leave
  **unset** (defaults to `true`) for any staging environment behind a
  TLS-terminating proxy.
- `DASHBOARD_CORS_ORIGINS` — comma-separated origin allowlist
  (`enhanced_dashboard.py:136`). Default `http://localhost:8080` is
  dev-only; set the production origin before exposing the dashboard.

No bare `BYBIT_TESTNET` / `BINANCE_TESTNET` env flag exists —
per-venue testnet selection is driven by `FUTURES_TESTNET` and the
database `general_config.testnet` row. Testnet *credentials* are
separate env vars (`BINANCE_TESTNET_API_KEY`, `BYBIT_TESTNET_API_KEY`).

Per the `.env.example` architecture notes, all sizing/risk knobs
(`max_copy_amount`, `position.position_size_usd`, `pumpfun_max_positions`,
`max_buy_tax`, …) live in the database and are edited via the
`/settings`, `/futures/settings`, `/solana/settings`, `/dex/settings`
pages — not via `.env`.

### Canary on mainnet
For modules whose venue has no useful testnet (DEX, ARBITRAGE, SOLANA,
SNIPER, COPY_TRADING — all real EVM / real Solana), canary = mainnet
with a hard cap:

- Set a small capital cap in the relevant settings page:
  - DEX / SNIPER / ARBITRAGE — `position.position_size_usd` (futures
    settings page exposes the analogous key for FUTURES).
  - SOLANA — `pumpfun_max_positions` plus per-strategy size key.
  - COPY_TRADING — `max_copy_amount` (default 100.0 USD; see
    `modules/copy_trading/CLAUDE.md`).
- Restrict to a single chain to bound the on-chain surface.
- Run for at least 24h with hour-level dashboard monitoring.
- Watch the per-module trades log AND `logs/.killswitch` for unexpected
  halts.

## 3. Per-module live-readiness checklist

Source: `docs/agents/MASTER_BACKLOG.md` verdict matrix. Every module is
AMBER; clearing these per-module items moves it to GREEN.

### DEX (AMBER)
- [ ] P1-04: every chain client routes through `config/pool_engine.py`
  (today: direct `Web3(HTTPProvider(...))` in places). Outstanding.
- [ ] Live SL/TP fallback path tested end-to-end on a tiny mainnet
  position.

### ARBITRAGE (AMBER)
- [ ] Triangular path: keep the MB-05 entry-guard until the atomic
  receiver contract ships.
- [ ] `RiskManager.validate_trade` (P1-06) hourly gas-burn breaker
  calibrated to the operator's wallet balance.

### SOLANA (AMBER)
- [ ] MB-15: Drift perp helper needs DRY_RUN gate + leverage cap.
  Disable Drift strategies via config until then.
- [ ] Confirm `solana_positions` reconcile (MB-09) ran on first start
  and matches on-chain balance.
- [ ] Cross-module `RiskManager.validate_trade` wired (Phase 2 #5
  scaffolded; not yet called from the SOLANA engine).

### SNIPER (AMBER)
- [ ] P1-12: WebSocket / Geyser real-time listener is a structural
  follow-up. Latency stays 15s+ until then — SNIPER is structurally
  late.
- [ ] Verify `should_skip_live` is consulted in every path that calls
  `_execute_evm_buy` / `_execute_solana_buy`.
- [ ] Cross-module `RiskManager.validate_trade` (P1 follow-up).

### FUTURES (AMBER)
- [ ] `FuturesRiskManager` injected at start (`main_futures.py:initialize`).
  Confirm logs show "FuturesRiskManager injected — MB-17 entry
  validator is active".
- [ ] Bybit `get_balance` / `get_position` / `close_position` /
  `get_all_positions` still placeholders — keep `FUTURES_TESTNET=true`
  until those land.
- [ ] Mark-price feed (MB-18) reachable for every traded symbol.

### AI (AMBER)
- [ ] MB-19: trained scaler `models/ai_strategy_scaler.pkl` exists.
  Until then, AI emits zero signals (startup warning logged).
- [ ] `direct_trading=true` only after canary on testnet for ≥48h.
- [ ] LLM provider daily-spend cap (`ai_provider.py:160`) set
  conservatively.

### COPY_TRADING (AMBER)
- [ ] Cross-module `RiskManager.validate_trade` integration (P1
  follow-up; currently only the local `max_copy_amount` cap).
- [ ] `module_manager` cross-process registration so the dashboard
  panic button reaches COPY (follow-up).
- [ ] `EVM_DEX_ROUTING` covers the operator's target chains;
  Arbitrum/Optimism/Avalanche return "unsupported" today.

### DASHBOARD (AMBER → ready for staging)
- [ ] `DASHBOARD_HTTPS` unset OR `true`, AND TLS terminated at the
  reverse proxy.
- [ ] Default admin password rotated (verify `scripts/init_auth.py`
  printed a fresh password on first run; don't reuse).
- [ ] `DASHBOARD_CORS_ORIGINS` set to the production origin.

## 4. Promotion procedure (per-module)

1. **DRY_RUN smoke** — start with `DRY_RUN=true`; watch the trades log
   for 1h. Verify simulated fills carry the expected fake-hash marker
   from the table in §1.
2. **Testnet (if available)** — flip `FUTURES_TESTNET=true` and the
   module's enable flag; run overnight; check
   `logs/<module>/<module>_errors.log` is quiet.
3. **Canary on mainnet** — set the per-module sizing key in the
   settings UI to ≤ 1% of book; flip `DRY_RUN=false`; monitor hour-level
   for 24h.
4. **Lift cap** — gradually raise the sizing key toward the operator's
   target allocation.

Every step is reversible — see §5.

## 5. Rollback

### Halt without flatten (investigate, leave positions open)
```bash
python scripts/emergency_stop.py
```
Writes `logs/.killswitch`. Every BaseModule subprocess flips its kill
switch within ~1s. Existing positions stay open.

### Halt and flatten
```bash
python scripts/emergency_stop.py && python scripts/close_all_positions.py
```
Or click the dashboard "EMERGENCY STOP" button.

### Revert a bad commit
1. `git log --oneline` — identify the offending commit.
2. `git revert <sha>` on the staging branch.
3. Restart the affected module (subprocesses don't hot-reload; rolling
   restart from the orchestrator is required).
4. Document in the on-call channel.

### Resume
Delete `logs/.killswitch` AND restart the affected processes. The kill
switch is intentionally sticky-by-restart so silent recovery never
re-enables live trading.

## See also
- `docs/runbook.md` — incident response
- `docs/engines.md` — engine API
- `docs/agents/MASTER_BACKLOG.md` — outstanding follow-ups
- Per-module: `modules/<name>/CLAUDE.md`
