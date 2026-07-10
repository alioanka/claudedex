# Wave-F6 — Post-F5 Trading-Module Validation Sweep

Scope: DEX, FUTURES, SOLANA, SNIPER. Logs span **2026-07-07 → 2026-07-09**
(~2.5 days, not a month — logs rotated fast). Wave-F5 deployed 2026-07-07 ~08:19
(all four modules restarted cleanly at that timestamp; orchestrator shows
`Restarts: 0` for DEX/Futures/Solana/Sniper through 20:38 on 07-09 — the
restart-latch never re-tripped). All modules run **DRY_RUN**.

Method: tailed each `*.log` / `*_errors.log` / `stderr.log*`, grepped Traceback/ERROR,
counted per-day, verified each F5 fix's new log lines, and quantified entries/exits/PnL
post-deploy.

---

## DEX — F5 restart-latch fix: **ENGAGED / HOLDING**

**F5 fixes verified**
- Restart-latch recoverable: **HELD**. Single PID since 07-07 08:19:16, zero restarts
  in orchestrator.log over the whole window (prior wave: DEX was dead 20 days).
- ML ensemble loading: **YES** — 756+ `🤖 ML[ensemble] conf=1.000 pump=… rug=…` lines
  (engine.py:983); no ensemble tracebacks.
- numpy close-fix: **WORKING** — 0 `not JSON serializable: numpy.float64` in current
  logs; 58 trades closed cleanly (`✅ Trade NNNN closed in database`).

**Activity (07-07 → 07-09)**: tokens_analyzed 8655, opportunities 4100, total_trades 58,
successful 84 / failed 32, `total_profit -0.62` (DRY_RUN). 58 DRY-RUN closes booked.

**NEW issues**
1. **[HIGH] DEX has taken zero NEW entries since 07-09 07:21** — 3254 lines
   `⚖️ Chain 'solana' weighted 0 in trading.chain_weights — skipping NEW entry …`.
   Every discovered opportunity is Solana (all EVM chains log `⚠️ No pairs found on
   ETHEREUM/BSC/BASE/MONAD/PULSECHAIN` — engine.py:672), and Solana is weighted 0, so
   100% of opportunities are skipped. DEX ran DRY-RUN trades on 07-07 (34 by 21:00) then
   went inert once chain_weights zeroed Solana. Either restore a non-zero Solana weight
   or fix EVM pair discovery so the weighted chains actually produce candidates.
   Evidence: TradingBot.log skip counts per rotation (0 before 07-09T07:21, 707 in the
   next segment). Fix: review `trading.chain_weights` seed + EVM discovery in
   `_monitor_new_pairs` (engine.py:638-672).
2. **[LOW] RugCheck 400 spam** — 39 `RugCheck API 400 … "unable to generate report"`
   for pump.fun mints in TradingBot_errors.log (honeypot_checker.py:392). Fail-soft, but
   noisy; suppress/backoff for pump mints RugCheck can't score.
3. **[NOISE] stderr = 4102× identical sklearn UserWarning** ("X does not have valid
   feature names, but RobustScaler was fitted with feature names"). Entire 704K stderr is
   this one line. Harmless; fit the scaler on a DataFrame or pass `.values` consistently
   to silence.

---

## FUTURES — F5 geometry rr=1.0 + gate seeds: **ENGAGED, but edge NOT recovered**

**F5 fixes verified**
- Geometry/gate seeds loaded: **YES** — trailing arm fires (`🎚️ TSL armed early at
  +0.78% (arm 0.75%)`), rolling gate benches symbols (`🪑 FUT-RM-27 rolling gate BENCHED
  …`), min score 4 rejects (`❌ REJECTED: Bullish but weak (score 2 < 4)`), volume gate
  demoted to diagnostic-only.
- Clean process: stderr 0 tracebacks; no crashes.

**Activity (37 closes post-07-07)**: WR 24% (9 win / 28 loss), gross +14.05 / −25.39,
**PF 0.55**, net ≈ −11.3. Daily finals: 07-07 −4.73, 07-08 +2.93, 07-09 −9.54.
Consecutive-loss circuit breaker engaged 07-09 evening (paused entries after 11 straight
losses).

**NEW issues**
1. **[HIGH] rr=1.0 did NOT produce TP exits — still 0 `take_profit` closes.** Exit-reason
   mix: time_limit 17, SL Hit 10, TSL Hit 7, Signal 3, **take_profit 0**. The F5 premise
   (TP1 unreachable at rr=2.0) is only half-fixed: TP1 still never fires; realized gains
   now come solely from TSL Hit (7). PF is unchanged from the F5-era 0.55. Module is
   bleeding in DRY_RUN. Evidence: `futures_trades.log` reason histogram. Needs the CLAUDE.md
   2-week window, but the early read is negative — recommend re-examining `atr_tp_rr_ratio`
   /TP1 distance vs realized MFE, not just trailing.
2. **[MED] 43.6% cumulative win rate on 1426 lifetime trades, PF-negative** (DAILY STATS
   line). The rolling gate benched nearly every symbol at startup (trailing net < −5 on
   10-19 trades each). The universe is structurally unprofitable under current signal —
   gate is working as a loss-limiter, not a profit engine.

---

## SOLANA — F5 fake-PnL / PriceValidator quorum: **ENGAGED on exit path; NEW entry-path hole**

**F5 fixes verified**
- PriceValidator wired on monitoring/exit path: **WORKING**. stderr shows 9436
  price-validator lines — 1506 `REJECTED unconfirmed …x jump`, 28 `CONFIRMED …x jump`
  via cross-source quorum, 7902 holding last-good. Example: HOBBES 70.9x crash held on the
  single-source read then correctly CONFIRMED once coingecko+dexscreener agreed
  (price_validator.py:257-268).
- No absurd PnL: **max +512%, only 1 close >500%, 0 ≥2000%** (prior wave had +2000% pins).
- Reconcile seeds validator from DB entry (solana_engine.py:2528). Poisoned exits get the
  void/clamp treatment (solana_engine.py:2610-2646).

**Activity (07-07 → 07-09)**: 474 opens, 754 closes, WR 67.2%, net **−$99.45**.
By strategy: pumpfun 560 closes −$6.53 (≈flat, fine); **jupiter 194 closes −$92.93** (the
entire loss).

**NEW issues**
1. **[CRITICAL] Jupiter ENTRY prices poisoned by wrong-denomination quotes — 38/194
   jupiter opens booked at implausible entry.** Examples (real px in parens): RAY $3310
   ($0.68), JTO $3232 (~$2.9), ORCA $6061 (~$3.5), PYTH $216 (~$0.10), KMNO $100 ($0.02),
   RENDER $7542 ($1.55). Consequences: **7 fake −99.98% `stop_loss` closes** (−$13–14 each)
   when the real price is later quorum-accepted, plus **16 `jupiter_time_exit` at $0**
   (entry==exit held for 3h — dead capital slots). This is the same denomination class the
   F5 fix targeted, but on the **entry** path.
   Root cause: PriceValidator only does *relative* jump detection. On the first read of a
   jupiter token there is no seeded last-good, so a poisoned quote seeds *itself* as
   last-good and is accepted; the F5 cross-check at
   **solana_engine.py:4449-4463** compares `metadata['price']` against a second
   `_get_token_price()` — but both derive from the same poisoned source, so `hi/lo≈1` and
   the guard passes. Real prices then look like a hard jump *down* and get held → time-exit
   at 0, or occasionally confirmed → −99.98%.
   Fix: for the fixed jupiter token universe (`config_manager.jupiter_tokens`), seed an
   **absolute** plausibility band (or require cross-source corroboration on the FIRST read
   before booking) so a $3310 RAY quote is rejected regardless of anchor. Touch
   `_open_position` entry-price block (solana_engine.py:4442-4463) + PriceValidator
   cold-start seeding (price_validator.py, and the `_corroborate_price` path at
   solana_engine.py:4036).
2. **[LOW] Drift init crash at startup** — `KeyError: 'mainnet-beta'` at
   `modules/solana_strategies/drift_helper.py:194` (driftpy `configs[env]`), 1 traceback.
   Fail-soft (DRY_RUN sim continues), but Drift chain never connects. Map env
   `mainnet-beta`→driftpy's expected key.

---

## SNIPER — F5 self-healing listener: **ENGAGED; module inert — 0 entries in 3 days**

**F5 fixes verified**
- Self-healing listener + RPC rotation: **WORKING**. Listener alive the whole window,
  detecting pools continuously (179,533 queued), 414× `🔄 Rotated to new RPC endpoint`,
  rate-limit backoff functioning, **zero `rpc_auth_failed` zombie** (prior wave: silent
  zombie 20 days). WSS pump_fun + raydium_v4 both live.
- SafetyChecker/analyzer running: **YES** — 85,561 pools analyzed.

**Activity**: **0 entries / 0 passed (0.0%)** across the entire window. Rejection tally:
**LowBSR 85,122 (99.5%)**, Danger 434, Honeypots 4, everything else 0.

**NEW issues**
1. **[CRITICAL] BSR gate rejects ~100% of candidates — module cannot enter.** 85,122 of
   85,561 analyzed rejected for Low Buy/Sell Ratio. Zero `birdeye` mentions in any sniper
   log → `_get_buy_sell_ratio()` returns `None` for every token → with age-floor active and
   `sniper_fail_closed_missing_bsr=True`, the gate fail-closes and rejects.
   This is the same 100%-block class as the Wave-13 DEX vol/liq threshold and futures volume
   gate. Evidence: SNIPER STATS lines in `stderr.log.1`; gate at
   **sniper_engine.py:1109-1141** (default `sniper_min_buy_sell_ratio=1.5` line 182,
   `sniper_fail_closed_missing_bsr=True` line 233).
   Fix (pick one): (a) provision a working Birdeye key so BSR data actually returns; (b) for
   t=0 snipe intent, fail-**open** on missing BSR (the module's whole point is buying before
   trade history exists — fail-closed defeats it); or (c) lower/disable the age floor so the
   fail-closed branch doesn't arm. Without one of these the sniper is a pure pool-detector
   with no execution.

---

## Cross-module summary of NEW post-F5 issues (ranked)

| # | Module | Sev | Issue | Fix location |
|---|--------|-----|-------|--------------|
| 1 | SNIPER | CRIT | BSR fail-closed + no Birdeye data → 99.5% rejected, 0 entries | sniper_engine.py:1109-1141 |
| 2 | SOLANA | CRIT | Jupiter entry-price denomination poison (38/194 opens; −$93) | solana_engine.py:4442-4463 |
| 3 | DEX | HIGH | Solana weighted 0 + EVM "no pairs" → 0 new entries since 07-09 07:21 | engine.py:638-672 + chain_weights seed |
| 4 | FUTURES | HIGH | rr=1.0 still yields 0 TP exits; PF stuck at 0.55 | atr_tp_rr / TP1 distance |
| 5 | SOLANA | LOW | Drift init KeyError 'mainnet-beta' | drift_helper.py:194 |
| 6 | DEX | LOW | RugCheck 400 spam for pump mints | honeypot_checker.py:392 |
| 7 | DEX | NOISE | 4102× sklearn feature-name UserWarning fills stderr | ML scaler input |

**F5-fix verdicts:** DEX restart-latch **ENGAGED**; Futures geometry **ENGAGED (edge not
recovered)**; Solana PriceValidator **ENGAGED (exit path only — entry hole open)**; Sniper
self-healing listener **ENGAGED (but module inert on BSR gate)**. No process crashed or
re-latched; the two CRITICAL findings are execution-blocking, not stability.
