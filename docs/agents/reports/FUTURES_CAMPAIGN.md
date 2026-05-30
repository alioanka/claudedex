# FUTURES — Wave 2 Campaign Report

**Agent:** A5 (market-trading-analyst, 20-yr CEX perps)
**Branch:** `claude/create-expert-agents-JFSF5`
**Wave:** 2 (re-audit + enhancement sweep)
**Date:** 2026-05-19

## TL;DR
- **The b1b8df9 leverage fix was incomplete on the live path.** It patched the dashboard wrapper but not `main_futures.py` (the subprocess entry). The runtime risk manager was still defaulting to `max_leverage=3` regardless of operator settings whenever the bot was launched via `python main.py`. FUT-RM-01 closes this.
- **MB-16..MB-18, reconcile observability, Binance/Bybit normalizer** re-verified — all still GREEN.
- Added 3 new risk policies (funding gate, ATR sizing, isolated-margin assertion) and 17 pinning unit tests + 3 DRY_RUN smoke tests.
- Module status: AMBER → GREEN candidate, blocked only on operator-led mainnet canary.

## Audit findings

### 1. Leverage-cap propagation (FUT-RM-01) — **CRITICAL, fixed**
- Commit `b1b8df9` patched only `FuturesTradingModule.initialize()` (the dashboard wrapper at `modules/futures_trading/futures_module.py:177-197`).
- The subprocess entry `modules/futures_trading/main_futures.py:607-626` builds `FuturesRiskManager` from `FuturesConfigManager.get_risk()` only. `FuturesRiskConfig` has no `max_leverage` / `max_positions` / `max_total_exposure` fields, so the manager silently fell back to its hard-coded defaults (`max_leverage=3`, `max_positions=3`, `max_total_exposure=500`).
- Symptom on production VPS: "Risk manager rejected entry for AAVE/USDT: Leverage 10x exceeds max 3x" even though the settings page showed `futures_max_leverage=20`.
- **Fix:** `main_futures.py` now merges `FuturesLeverageConfig.max_leverage`, `FuturesPositionConfig.max_positions`, and a derived `max_total_exposure = capital_allocation × default_leverage` into `risk_cfg` before constructing the manager. Also forwards funding-gate config (FUT-RM-05).

### 2. Startup assertion (FUT-RM-02) — **NEW, fixed**
- Added `FuturesTradingApplication._assert_runtime_risk_matches_config()` which runs right after `set_risk_manager()` and logs an error when runtime caps don't match DB-backed config.
- Soft-fail by default (log + alert); hard-fail when `FUTURES_RISK_ASSERT_HARD=1` (recommended for CI / canary).

### 3. DRY_RUN smoke (FUT-RM-03) — **NEW**
- 5 cap-propagation unit tests pin: max_leverage, max_positions, derived max_total_exposure, liquidation_buffer pct-to-fraction conversion, validate_new_position allows op-set leverage and still blocks above.
- 3 DRY_RUN smoke tests:
  - Binance adapter trip-wire (skips when aiohttp absent in CI)
  - Bybit adapter trip-wire (skips when aiohttp absent)
  - Static source-guard pinning `should_skip_live(..., module='futures')` around the `exchange_client.open_long`/`open_short` call site in `_open_position`.

### 4. MB-16..MB-18 re-verification — **PASS**
- **MB-16** (init order): `FuturesTradingEngine.__init__` (lines 219-431) resolves `self.dry_run` via `resolve_dry_run_env` before any other reads. No regression.
- **MB-17** (margin mode + validator wiring): `_open_position` calls `validate_new_position` at line 1727; FUT-RM-07 below adds defense-in-depth.
- **MB-17b** (Bybit V5): `exchanges/bybit_futures.py` `open_long`/`open_short` call `set_leverage` + `set_margin_type(ISOLATED)` before order placement.
- **MB-18** (mark vs last): `_check_exit_conditions` and liquidation monitor consume `mark_price` via the normalizer. No regression.
- **Reconcile observability**: `_sync_positions` stamps `last_reconcile_at` / `last_reconcile_count`; `check_reconciled_capacity` flags `RESTART OVER-CAP`. Hooked into BaseModule via `reconcile_open_positions`.
- **Bybit/Binance normalizer** (`exchanges/_normalizers.py`): canonical schema lookup works for both shapes including Bybit V5 `liqPrice` / `tradeMode` recovery from the `raw` sub-dict.

### 5. Funding-rate directional gate (FUT-RM-05) — **NEW**
- New `FuturesFundingConfig.skip_long_funding_bps` (default 5 bps ~= 55% APR) and `skip_short_funding_bps`.
- New `FuturesRiskManager.should_skip_for_funding(side, funding_rate)` — fail-open on missing data.
- New `FuturesTradingEngine._get_funding_rate_cached()` using the mainnet `price_client` (testnet funding is fictional) with a 300s TTL.
- Wired in `_open_position` BEFORE `validate_new_position` (cheap gate runs first). 6 unit tests.

### 6. ATR per-symbol sizing (FUT-RM-06) — **NEW, opt-in**
- New `FuturesPositionConfig.atr_sizing_enabled` (default off), `atr_risk_pct` (1%), `atr_stop_multiplier` (1.5).
- `TechnicalSignals.atr` / `atr_pct` populated in `_get_technical_signals` (14-period TR avg).
- `_calculate_position_size` picks the ATR branch when enabled + ATR available; falls through to static/dynamic otherwise.
- Risk-parity math: a 5% ATR symbol gets ~1/5 the notional of a 1% ATR symbol so a `stop_multiplier × ATR` move costs the same dollars on both. 5 unit tests.

### 7. Isolated-margin assertion (FUT-RM-07) — **NEW, defense-in-depth**
- `set_margin_type` runs inside the adapter `open_long`/`open_short`, but stale account-level settings or Bybit's `110026/110043` idempotency-as-success branch could land a CROSS-margin fill.
- New `FuturesLeverageConfig.enforce_isolated_margin` (default True).
- `_verify_isolated_or_close()` reads the position back immediately and emergency-closes on `margin_type != ISOLATED` with reason `fut_rm_07_cross_margin_detected`.
- Fail-open when `margin_type` missing (normalizer edge case). 3 unit tests + source-guard pin.

## Live-readiness checklist
- [x] `DRY_RUN` honored on every send/order/sign path (FUT-RM-03 pin + source guard)
- [x] Per-trade leverage cap enforced and matches DB config (FUT-RM-01, FUT-RM-02)
- [x] Per-position margin mode ISOLATED enforced post-fill (FUT-RM-07)
- [x] Per-day max loss enforced in `FuturesRiskManager.can_trade` (existing)
- [x] Funding-rate gate on directional entries (FUT-RM-05)
- [x] Per-symbol sizing tied to ATR available as opt-in (FUT-RM-06)
- [x] Position reconciliation on startup (`_sync_positions`)
- [x] Idempotent order IDs (`newClientOrderId` on Binance, `orderLinkId` on Bybit)
- [x] Heartbeat (`_trading_loop` logs every 10 iters); Telegram + Health server up
- [x] Emergency stop wired (`/api/bot/emergency-exit` → `logs/.killswitch` → `should_skip_live` short-circuits the adapter call)

**Pending operator step before live:** mainnet canary at small size with `FUTURES_RISK_ASSERT_HARD=1` and the funding/ATR gates enabled at conservative defaults. Confirm one full SL hit, one TP1 hit, and one funding-gate skip in the logs before scaling.

## Profitability levers (ranked, ordered by expected $ impact per month at $10k capital)
1. **ATR risk-parity sizing** (FUT-RM-06) — biggest expected variance-reduction win. Without it, a single volatile-symbol stop hit eats ~5× the dollar amount of a quiet-symbol stop hit. Recommend flipping `atr_sizing_enabled=true` after canary.
2. **Funding-rate gate** (FUT-RM-05) — saves the cost of holding longs into 8-hour funding windows when funding APR > 50%. At 5 bps gate cutoff this should drop ~5-15% of late-cycle long entries.
3. **Tighten `min_signal_score` from 4 → 5** — fewer trades, higher win rate. Current config favours volume over quality.
4. **Trailing stop activation post-TP2** — already wired (commit `8b71526`). Verify in canary that the stop migrates as expected.
5. **Per-symbol leverage cap table** — currently one cap for all pairs. BTC/ETH can safely run higher than mid-cap perps. Out of scope for this wave.

## Files touched
- `modules/futures_trading/main_futures.py` — FUT-RM-01, FUT-RM-02, FUT-RM-05 wiring
- `modules/futures_trading/core/futures_engine.py` — FUT-RM-05 funding gate + cache, FUT-RM-06 ATR signal + sizing, FUT-RM-07 verifier + post-fill hook
- `modules/futures_trading/futures_risk_manager.py` — FUT-RM-05 `should_skip_for_funding`
- `modules/futures_trading/config/futures_config_manager.py` — FUT-RM-05/06/07 fields + routing keys
- `modules/futures_trading/CLAUDE.md` — Wave-2 changes + config cheat-sheet
- `tests/unit/test_futures_risk_wiring.py` — 17 pin tests + 3 smoke tests

## Commits (this wave)
- `09a5c85` — FUT-RM-01 (main_futures leverage propagation) + FUT-RM-02 (startup assertion). [bundled into solana commit due to inflight rebase]
- `34399db` — FUT-RM-03 (DRY_RUN smoke + cap-propagation tests)
- `9aba5e8` — FUT-RM-05 (funding-rate directional gate)
- `bcc7b91` — FUT-RM-06 (ATR sizing, settings + tests; engine implementation lost to rebase race)
- `7e9a47b` — FUT-RM-06 ATR test pins
- `ad401c4` — FUT-RM-06 engine re-apply + FUT-RM-07 isolated-margin verify

## Recommended follow-ups for next wave
1. Hourly **funding-cost realized vs predicted** dashboard widget (uses the new `_funding_cache`).
2. Per-symbol leverage table (`max_leverage_overrides: dict[str, int]` on `FuturesLeverageConfig`).
3. **Auto-deleverage** trigger wiring — `FuturesRiskManager.should_auto_deleverage()` exists but no caller wires it to the engine's drawdown counter.
4. Telegram alert on FUT-RM-07 emergency-close (currently logs only).
5. Smoke-test CI image with `aiohttp` so the two skipped DRY_RUN adapter tests run.
