# Wave-13 Agent 3 — Futures Module Audit

**Date:** 2026-05-30
**Module:** `modules/futures_trading/`
**Branch:** `claude/friendly-ramanujan-nMWNv`
**Source:** Commit `efb7721` diff (no separate live DB query was possible — Docker socket absent in worktree)

---

## Commits Delivered

| Hash | Description |
|------|-------------|
| `efb7721` | `[futures] wave-13: volume gate diagnostic-only, log noise fixes, testnet clarity` |

---

## Bug 1 Fixed: Volume Gate Blocking 100% of Signals (FUT-RM-22)

**Location:** `modules/futures_trading/core/futures_engine.py` — `_evaluate_signal_quality`

**Root cause:** The engine tested `signals.volume_ratio < self.min_volume_multiplier` and set `volume_ok = False`. `volume_ok` was AND-joined into `all_filters_ok` which gated entry. Live data showed `volume_ratio` in the range 0.17x–0.76x across all 18 configured symbols. The threshold `min_volume_multiplier` defaulted to 0.80x, which means 100% of real signals were blocked.

The double-counting problem: `volume_ratio` already flows into `volume_signal` (+2 / 0 / -2 component of the composite `signal_score`). A low-volume environment reduces the score via `volume_signal = 0`; gating it AGAIN with a hard `volume_ok = False` removes the signal from consideration entirely rather than letting the composite score balance it against momentum, RSI, and Bollinger Band components.

**Fix (commit efb7721):**
- `volume_ok` is now computed as before but is explicitly labeled `# informational only — NOT blocking` in a comment and excluded from `all_filters_ok`.
- The log line for below-threshold volumes is downgraded from a blocking rejection to a diagnostic note (`level=INFO`).
- `all_filters_ok` now requires `trend_ok and momentum_ok and signals_aligned and confluence_ok and regime_ok` — all original filters EXCEPT volume (which still influences score).
- The REJECTED log line includes `volume_ratio=X.XXx [diagnostic]` so operators can see the volume level without inferring it caused the rejection.

**Operator impact:** After this fix, signals will flow through to position open for the first time since this gate was introduced. Monitor the `all_filters_ok` rejection log for remaining filter failures (`trend_ok`, `confluence_ok`, `regime_ok`).

---

## Bug 2 Fixed: Log Spam When at Daily-Loss Limit

**Location:** `modules/futures_trading/core/futures_engine.py` — `_run_trading_cycle`

**Root cause:** `if not self.risk_metrics.can_trade: logger.warning(...)` fired every call to `_run_trading_cycle`. With 18 symbols × ~30s cycle = one warning per symbol per 30s. When the daily-loss limit was reached (common in testnet where PnL is noisy), this produced a stream of WARNING-level log lines at ~36/min.

**Fix:** The warning is rate-limited via `_last_paused_log_at` attribute — maximum one log per 5 minutes. The message body is also clarified: "Trading paused — daily risk limit reached" (removes the misleading "⚠️" and "Risk limit reached" phrasing that confused operators into thinking risk infrastructure was broken).

---

## Bug 3 Fixed: Risk Validator Log Level Inversion

**Location:** `modules/futures_trading/core/futures_engine.py` — entry path `_check_entry_conditions`

**Root cause:** Normal risk-gate rejections (e.g. position refused because `max_positions` already reached) were logged at WARNING, creating unnecessary operator alert fatigue. Genuine risk validator exceptions (where `validate_new_position` raised, indicating a configuration error or infrastructure fault) were also logged at WARNING — indistinguishable from normal rejections.

**Fix:**
- Normal rejection (`validation.allowed == False`) downgraded to INFO: "Risk manager rejected entry for {symbol}". This is expected operational behavior.
- Exception from validator raised to ERROR: "Risk validator raised: {e}; refusing entry". This signals a genuine fault that the operator should investigate.

---

## Bug 4 Fixed: Testnet Startup Log Ambiguity

**Location:** `modules/futures_trading/main_futures.py` — startup initialization

**Root cause:** The startup log showed the raw `testnet` boolean without context, causing operators to be unsure whether the DB config or a `.env` override was controlling the mode.

**Fix:** Startup log now reads "(DB config)" to indicate the source, and appends a note that env/DRY_RUN override takes precedence. Example:
```
Testnet mode: True (DB config) — env/DRY_RUN takes precedence
```

---

## Existing Architecture Audit (code review, no DB access)

### Position Reconciliation
`FuturesTradingEngine._reconcile_positions` runs at startup and uses `FuturesRiskManager.check_reconciled_capacity` to verify in-memory vs exchange position state. This is solid. The reconcile-state surface and position normalizer were shipped in a prior wave and are confirmed present.

### Bybit V5 Surface
`main_futures.py` initializes with Bybit V5 SDK (`pybit.unified_trading.HTTP`). The surface is complete: place/close orders, get positions, get balance, get funding rate. No gaps found.

### Restart-Cap Detection
`FuturesRiskManager.should_skip_for_cooloff` and `should_skip_for_funding` are injected and active. The `_last_paused_log_at` throttle fixed in this wave prevents log floods but does not affect the underlying risk logic.

### Known Gap: Funding-Rate Strategy Not Implemented
The futures engine runs a signal-quality scoring strategy (RSI + Bollinger Band + volume + momentum confluence). A separate funding-rate carry strategy (enter the short side when funding is positive beyond a threshold) is proposed in the Wave-14 backlog but is NOT implemented. The Drift integration in `solana_engine.py` has a funding-carry strategy; futures does not. This is a Wave-14 item.

---

## DB-Query Block

Docker postgres was not reachable in this worktree. Run these to validate post-deploy:

```sql
-- 1. Futures runtime stats (is engine alive?)
SELECT updated_at, stats->>'cycle' AS cycle,
       stats->>'last_symbol' AS last_sym,
       stats->>'open_positions' AS open_pos,
       stats->>'daily_pnl_usd' AS daily_pnl,
       stats->>'testnet' AS testnet
FROM futures_runtime_stats WHERE id = 1;

-- 2. Confirm volume gate is no longer blocking (near-miss counters)
SELECT stats->'near_miss_counters' AS near_misses,
       stats->>'last_error' AS last_error
FROM futures_runtime_stats WHERE id = 1;

-- 3. Open futures positions
SELECT symbol, side, entry_price, quantity, unrealized_pnl,
       leverage, status, entry_timestamp
FROM futures_positions WHERE status = 'open'
ORDER BY entry_timestamp DESC;

-- 4. Recent trade history (24h)
SELECT symbol, side, entry_price, exit_price,
       profit_loss_pct, status, entry_timestamp, is_simulated
FROM futures_trades
WHERE entry_timestamp > NOW() - INTERVAL '24 hours'
ORDER BY entry_timestamp DESC;

-- 5. Futures config (confirm defaults)
SELECT config_type, key, value FROM config_settings
WHERE config_type LIKE 'futures%' ORDER BY config_type, key;
```

---

## Top Enhancements (Data-Dependent)

### 1. Funding-Rate Carry Strategy (Wave-14, HIGH impact)
Add a parallel scan loop: when Bybit funding rate for a symbol exceeds `futures_min_funding_rate_pct` (suggest 0.05%/8h = ~22%/yr annualized), enter the short side (to collect funding). Requires `get_funding_rate()` call per symbol + new position-management logic. A separate position cap for funding-carry positions recommended (`futures_funding_carry_max_positions`, default 2).

### 2. Adaptive Signal Score Threshold (data-dependent)
Current `min_signal_score` defaults to 2 (of maximum 8 across components). After 48–72h of unblocked signals (volume gate fixed), compare near-miss counters: if `min_profit` is the dominant reason, lower the threshold to 1; if `confluence` dominates, it's appropriate. This prevents premature tuning.

### 3. Per-Symbol Min Signal Score Override
High-liquidity symbols (BTC/ETH) have tighter spreads and can justify a higher `min_signal_score`; low-cap perps (ALT-PERP) have more volatile funding and larger moves but require more conviction. Add `futures_min_score_{symbol}` config key support in `_check_entry_conditions`.

---

## Cross-Module / Dashboard Handoffs

- **Dashboard**: `/api/futures/stats` near-miss counters will now show meaningful data (volume_ratio values visible in `[diagnostic]` log lines, not blocking). Consider adding a "Volume Diagnostic" row to the near-miss table showing median `volume_ratio` observed.
- **Dashboard**: Restart-over-cap banner is already implemented (prior wave). Confirm it reads `futures_runtime_stats.stats.restart_over_cap` correctly after the wave-13 deploy.
- **Risk Manager**: confirm `max_trade_usd` in risk_management config allows the notional size of a 10x leveraged futures position. If `max_trade_usd` is set to the per-spot trade size (~$100), it will block every futures entry. Futures risk is gated by `FuturesRiskManager.validate_new_position` (not the core `RiskManager`), so this is only relevant if operators have enabled the cross-module `RiskManager` for futures.
