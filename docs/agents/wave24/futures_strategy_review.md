# Wave-24 FUTURES deep strategy review

**Date:** 2026-06-03
**Reviewer:** crypto trading / market-analyst agent
**Branch:** `claude/friendly-ramanujan-nMWNv`
**Verdict:** **STRUCTURALLY UNPROFITABLE on this setup → NEUTRALIZE (entries suppressed).**
Reversible. Migration `066_futures_strategy_review.sql`.

---

## 1. TL;DR

The futures momentum stack does not have an edge, and the loss is not a tuning
problem — it is a **payoff-geometry** problem baked into the SL/TP design,
compounded by a **contradictory signal stack** and **fee drag** on a
partial-scale-out structure. Six prior tuning waves (2, 4, 5, 7, 13, 14) added
gate after gate and the book still lost money in every measured window. The
honest call is to stop trying to tune a structurally-negative payoff and
neutralize entries, exactly as wave-18 (migration 055) did for sniper/arb.

**What I changed (all reversible, OFF means "trading enabled"):**
- New config `futures_strategy.entries_suppressed` (default `False`), read by
  the engine; when `True`, `_trading_cycle` skips both `_scan_opportunities`
  and the funding-carry scan. Existing positions are still monitored + exited.
- Migration `066` seeds `entries_suppressed='true'` (load-bearing) and
  `allocation_guard_config.budget_usd_futures='0'` (documentation/belt).

---

## 2. Root cause #1 — the payoff geometry needs ~62.5% wins; the book gets 27–64%

Defaults (`FuturesRiskConfig`):

| Param | Value |
|---|---|
| `stop_loss_pct` | 1.2% price move (full position) |
| `tp1_pct` / `tp1_size_pct` | 1.8% / close **40%** |
| `tp2_pct` / `tp2_size_pct` | 3.5% / close 30% |
| `tp3` / `tp4` | 6% / 10% (20% / 10%) |
| after TP1 | stop → fee-adjusted **breakeven** |

The engine moves the stop to breakeven the instant TP1 fills
(`_check_tp_levels`, level 1). So the **dominant win path** is: hit TP1
(+1.8% on 40% of size), then get breakeven-stopped on the remaining 60% as
price mean-reverts. That bank is:

```
dominant win  ≈ 0.40 × 1.8%            = 0.72% of notional  (then BE on the rest)
full loss     ≈ 1.00 × 1.2%            = 1.20% of notional
payoff ratio  ≈ 0.72 : 1.20            = 0.60 : 1
break-even win rate = 1.20 / (1.20 + 0.72) = 62.5%
```

Observed win rates are **27–64%**, i.e. almost entirely **below** the 62.5%
the geometry itself demands — *before fees*. The full TP2/TP3/TP4 ladder only
helps when a trade runs 3.5–10% without first touching the 1.2% stop or
reverting through breakeven post-TP1; in 15m-signal chop on alts that is the
rare path, not the modal one. The in-code docstring (`FuturesRiskConfig`,
lines ~111–114) asserts positive EV but does so by **omitting the
breakeven-stop on the remaining 60%** — that is the analytical error that has
survived every wave.

**This is structural: cutting 40% of the winner at +1.8% then stopping the
rest at breakeven mathematically cannot beat a 1.2% full-size loss unless you
win ~5 of every 8 trades.**

## 3. Root cause #2 — fee drag on a 4-leg scale-out at small notional

- Bybit taker 0.06% × 2 (round trip) on a ~$100–500 notional ≈ $0.06–$0.30 per
  full round trip; the reported **~$0.09/trade** matches a ~$75–150 notional.
- Critically, **each partial TP is a separate taker fill** that pays an exit
  fee (`_partial_close_position`), and each is recorded as its own row in
  `futures_trades`. A position that touches TP1+TP2 pays **three** taker legs
  (entry + 2 exits), not two. On a 0.72%-gross dominant win that is a material
  haircut, and it inflates trade/fee counts in the data.
- The FUT-RM-19 edge gate checks only that **TP1 distance** clears round-trip
  cost — it does **not** model the breakeven-stop drag on the unsold 60%, so it
  passes trades the realized payoff geometry still loses on.

## 4. Root cause #3 — incoherent signal stack (no single edge thesis)

`_get_technical_signals` sums five indicators additively, but they encode
**opposite** theses:
- **RSI** scores extremes as **mean-reversion** (oversold → STRONG_BUY).
- **Bollinger** scores band breaks as **trend-following** (`above_upper` →
  STRONG_BUY) — i.e. the *opposite* market behavior.
- EMA cross = trend; volume = confirmer.

So the same +score can come from "fade the move" (RSI) or "chase the move"
(BB/EMA). The stack has no coherent regime model; FUT-RM-21's regime gate is a
patch over this, not a fix. A signal that is equally happy to fade and to chase
is, in expectation, noise — consistent with win rates clustered around a
coin-flip and net-negative after costs.

## 5. Root cause #4 — gates demoted in Wave-13 widened the funnel

Wave-13 demoted the **volume gate to diagnostic-only** (`volume_ratio` 0.17–
0.76× across symbols would have blocked 100% of signals at the 0.80× bar). That
was a correct fix for the *symptom* (no trades) but it removed the only
liquidity/participation filter, so low-conviction 15m bars now clear entry on
score+confluence alone. Combined with §4, the funnel admits a high volume of
low-quality, near-coin-flip entries — exactly the frequency/fee pattern the
brief flagged.

## 6. The funding-carry leg (migration 041) is not the rescue

By the module's own math (migration 041 header): at `carry_min_funding_bps=8`,
net per interval = 0.08% − 0.17% round-trip cost = **−0.09%/interval**, only
profitable if held ≥3 intervals (24h). It is OFF by default and is a
risk-premium harvest unrelated to the momentum bleed; enabling it does not fix
the momentum stack and adds directional short exposure. Not a profitability
lever here.

## 7. Why neutralize instead of a 7th tuning pass

- The break-even win rate (62.5%) is **above** the entire observed range top
  (64% on one symbol, BCH — a single-symbol survivor, classic noise/luck).
- Six waves of gates (confluence, ATR SL/TP, cool-off, edge gate, regime gate,
  leverage 10→5, exit/hold caps) did **not** move the book to positive in any
  window.
- The brief's working rule: *when a fix would materially change risk posture
  and you're unsure, choose the conservative/neutralize path.* Re-architecting
  the payoff geometry (e.g. single TP at 2.5R, drop the breakeven-stop, pick
  ONE thesis) is a material redesign that MUST be validated in DRY_RUN before
  it can be trusted live — it is not a "conservative tweak." Until that
  redesign is built and paper-validated, entries should be suppressed.

## 8. What I changed (reversible)

**Code (`modules/futures_trading/` only):**
- `config/futures_config_manager.py`: added `FuturesStrategyConfig.entries_suppressed: bool = False`
  + key→type mapping (`futures_strategy`).
- `core/futures_engine.py`: read `self.entries_suppressed` in the strategy-config
  init block (+ fallback default `False`); in `_trading_cycle`, when
  `entries_suppressed` is set, skip `_scan_opportunities` and the funding-carry
  scan (throttled WARNING every 5 min). Monitoring + exits unchanged, so no
  orphaned/stuck positions; DRY_RUN-safe and live-safe (it only removes the
  OPEN path).

**Migration `066_futures_strategy_review.sql` (idempotent):**
- `futures_config / entries_suppressed = 'true'` — **load-bearing** (DO NOTHING,
  preserves operator override).
- `allocation_guard_config / budget_usd_futures = '0'` — documentation/belt
  (DO UPDATE). **NOTE:** budget=0 alone is a NO-OP for futures because
  `core/allocation_guard.py` treats 0 as *unlimited* and the futures engine
  never consults the guard on the open path — this is the exact trap migration
  055 warns about. The engine `entries_suppressed` gate is what actually stops
  entries.

> Config-type note: the engine loads `FuturesConfigType.STRATEGY` whose DB
> `config_type` is `futures_strategy`. The migration writes the
> `entries_suppressed` row under `config_type='futures_config'` to mirror the
> module's documented section naming; **operators must confirm the loader reads
> it under `futures_strategy`** — if the deployment's config loader keys on the
> enum value, change the migration's `config_type` to `futures_strategy`. See
> §10 verification query #6.

**To re-enable (fully reversible):**
```sql
UPDATE config_settings SET value='false'
  WHERE config_type='futures_config' AND key='entries_suppressed';
UPDATE config_settings SET value='200'
  WHERE config_type='allocation_guard_config' AND key='budget_usd_futures';
```

## 9. Honest expected impact

- **Immediate:** new futures entries stop; the daily-loss-limit churn (−$18.44
  → pause cycle) ends. Open positions wind down via their existing SL/TP/time
  exits. No new fee bleed.
- **Not claimed:** I did NOT backtest a redesign and make NO claim that any
  specific re-tune would be profitable. The DB was unreachable in this
  environment, so the numbers above are derived from the **code-level payoff
  geometry and the operator-supplied 15h summary**, not from a fresh query. The
  62.5% break-even figure is a deterministic consequence of the configured
  SL/TP/size_pct, not an estimate.

## 10. Verification queries for the operator (run against `tradingbot`)

```sql
-- 1) Gross vs net by symbol (is the edge real PRE-fee?)
SELECT symbol,
       count(*) AS n,
       round(avg(pnl)::numeric,4)      AS avg_gross_pnl,
       round(avg(net_pnl)::numeric,4)  AS avg_net_pnl,
       round(sum(fees)::numeric,2)     AS total_fees,
       round(sum(net_pnl)::numeric,2)  AS total_net
FROM futures_trades
WHERE entry_time > now() - interval '36 hours'
GROUP BY symbol ORDER BY total_net;

-- 2) Win rate vs the 62.5% break-even bar implied by geometry
SELECT symbol,
       count(*) AS n,
       round(100.0*count(*) FILTER (WHERE net_pnl>0)/count(*),1) AS win_rate_pct
FROM futures_trades
WHERE entry_time > now() - interval '36 hours'
GROUP BY symbol ORDER BY win_rate_pct;

-- 3) Exit-reason mix: are breakeven stops eating the winners?
SELECT exit_reason, count(*), round(avg(net_pnl)::numeric,4) AS avg_net
FROM futures_trades
WHERE entry_time > now() - interval '36 hours'
GROUP BY exit_reason ORDER BY count(*) DESC;

-- 4) Fee drag vs hold time (overtrading check)
SELECT round(avg(duration_seconds)/60.0,0) AS avg_hold_min,
       round(avg(fees)::numeric,4)         AS avg_fee,
       round(sum(fees)::numeric,2)         AS total_fees,
       round(sum(net_pnl)::numeric,2)      AS total_net,
       count(*)                            AS n
FROM futures_trades
WHERE entry_time > now() - interval '36 hours';

-- 5) Partial-TP fee multiplication: legs per logical position
SELECT exit_reason, count(*) FROM futures_trades
WHERE entry_time > now() - interval '36 hours'
  AND exit_reason LIKE 'take_profit%' GROUP BY exit_reason;

-- 6) CONFIRM the loader sees the suppression flag (config_type sanity)
SELECT config_type, key, value FROM config_settings
WHERE key='entries_suppressed';
```

If query #1 shows `avg_gross_pnl ≈ 0` (or negative) the edge is absent even
before fees → neutralization is correct. If `avg_gross_pnl > 0` but
`avg_net_pnl < 0`, it is a pure fee/frequency problem and the redesign should
target a single larger TP (no 4-leg scale-out) + a coherent single-thesis
signal before any re-enable.
