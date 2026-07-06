# Wave-F5 / 03 — SOLANA fake PnL + SNIPER losses & empty trade history

Scope: `modules/solana_trading/`, `modules/sniper/`, dashboard surfaces for both.
Evidence: `logs/solana_trading/solana_trades.log` (2026-06-15 → 07-05), `logs/sniper/sniper.log`,
screenshots `screencapture-...-solana-{performance,trades,positions}-*.png`, `...-sniper-{dashboard,performance,trades,settings}-*.png`.
Analysis date 2026-07-06. READ-ONLY — no code changed.

---

## A. SOLANA — current fake-PnL mechanism (root cause)

**Scale.** In `solana_trades.log`: **138 CLOSE rows with pnl_pct ≥ +1,000%** (max +526,608%; 127 `jupiter`
`take_profit`, 11 `pumpfun` partial exits), **112 CLOSEs at ≈ −99.98%**, and **449 of 3,794 OPENs (11.8%)
opened at a poisoned price** — RAY×134, PYTH×115, JTO×107, ORCA×91 (+2 KMNO). Bug is ACTIVE (last fake row
2026-07-05 07:02:23, ACM +1009%).

**Signature.** Poisoned quotes are the real USD price × ~4,900–5,200 (stable intra-day, drifts slowly across
days), e.g. RAY opens oscillate `0.6278 → 3118.7 → 0.6332 → 3155.75` on 06-16 (log lines 04:04, 12:11). The
same multiplier hits pump.fun mints (BUCKINGHAM exit 0.2459 vs entry 4.671e-05 = 5,264×, six identical
+526,339% partial-exit rows 07-03 11:13), so it is ONE upstream source in the shared multi-source chain
(DexScreener → CoinGecko → Jupiter v3, `SolanaEngine`/`JupiterClient.get_price`,
`modules/solana_trading/core/solana_engine.py:476-513`), not a per-token mapping error. The validator's
REJECTED/CONFIRMED WARN lines that would name the source have rotated out of `solana_trading.log` (it only
covers the final day) — see fix #7 (retention).

**Defect chain (five compounding bugs):**

1. **Scan/entry path bypasses PriceValidator.**
   `solana_engine.py:3599` (`_scan_jupiter_opportunities`) calls `self.jupiter_client.get_price(token_mint)`
   RAW — not `_get_token_price()` (the validated wrapper at `:3930`). The raw price feeds the momentum signal
   (`:3625-3640`) — a 5,000× quote is "+499,900% momentum" → instant BUY — and is passed as
   `metadata={'price': price}` which `_open_position` prefers as the entry price (`:4310-4321`; the validated
   call at `:4336` is the LAST fallback). Result: RAY position OPENED at $3,434.82 (log 2026-07-03 11:22:13),
   ORCA entry $6,128.29.
2. **PriceValidator never seeded or dropped.** Zero call sites for `price_validator.seed()` / `.drop()` in the
   engine, despite `price_validator.py` docstring ("seeded from entry price on open / reconcile"). First quote
   for an unanchored mint is auto-accepted (`price_validator.py:176-185`), and the 600s `lastgood_ttl_s` expiry
   auto-accepts anything after a quiet period.
3. **Persistent wrong quotes defeat the jump-confirmation.** A wrong-denomination feed echoes the same value
   every poll; 3 consecutive readings within the 30% band → **CONFIRMED and accepted**
   (`price_validator.py:211-231`), i.e. ~15s at the 5s cache/poll cadence. The validator only stops one-poll
   glitches. Once confirmed, monitor pnl (`solana_engine.py:2827-2848`) hits TP → fake +500k% exit; the
   reverse flip (poisoned entry → real price confirmed) produces the −99.98% `stop_loss` rows. Third class:
   entry AND exit both poisoned → sane-looking rows at absurd prices (ORCA 6128.29→5987.24, −2.30%,
   log 07-03 11:13:17) that no ratio guard can ever see.
4. **Save-time guard converts fake exits into +2000% "wins" instead of voiding them.**
   `_save_trade_to_db` (`solana_engine.py:2551-2616`) pins exit to 50× entry, recomputes
   `pnl_sol = notional × 49` (≈ **+8.5–9.2 SOL per fake trade**), clamps pnl_pct to +2000, tags
   `metadata.excluded=true` — but still writes a WINNING row.
5. **Dashboard ignores the `excluded` tag.** `/api/solana/trades`
   (`monitoring/enhanced_dashboard.py:9888-9897`) selects `solana_trades` with **no metadata filter** (doesn't
   even SELECT `metadata`); the module-overview aggregate (`:2112-2123`) likewise. `/solana/performance` and
   `/solana/trades` are computed client-side from that feed. The control-center aggregate (`:3700`) DOES
   filter — hence pages contradict each other.

**Observed dashboard damage (screenshots 07-05):** Solana Performance: Total P&L **+1176.38 SOL ($235,276)**,
PF 25.84, best trade +2000.0%, max drawdown **−134.5%**, "best tokens" RAY +359 SOL / PYTH +315 / JTO +277 /
ORCA +205 — exactly the four poisoned watchlist mints. ~127 pinned rows × ~8.6 SOL ≈ **1,090 of the
+1,176 SOL is fabricated**; the −134.5% drawdown is arithmetic over the corrupted equity curve. Solana Trades
page top rows: JTO entry $0.7775 → exit $38.8750 (=50× pin), +9.1773 SOL, +2000.00%.
Also poisoned: in-engine `risk_metrics.daily_pnl_sol` accumulates the RAW fake pnl (`:5068`), and the trades
JSON log is written pre-guard (`_log_trade`, `:2176-2188`).

**"NaNm" ages (separate small bug):** `opened_at.isoformat() + 'Z'` on tz-AWARE datetimes emits invalid ISO
`2026-07-05T18:41:08+00:00Z` — `solana_engine.py:5088-5089` (trade log) and `:5168` (positions API) → JS
`Date` = NaN → every AGE cell on /solana/positions shows "NaNm" (screenshot).

### Solana fixes (ranked by ROI)

1. **Validate at fetch, everywhere** — make `_scan_jupiter_opportunities:3599` and the `_open_position`
   metadata/DexScreener price steps (`:4310-4332`) go through `_get_token_price`; validate any
   `metadata['price']` against the validated quote before using it as entry.
2. **Wire `price_validator.seed(mint, entry_price)` on open/reconcile and `.drop(mint)` on close** (as the
   validator docstring already promises).
3. **Cross-source quorum instead of repetition-confirmation for exits**: a >hard-ratio jump may only be
   ACCEPTED when ≥2 independent sources agree within ~20%; single-source persistence should keep HOLDING the
   last-good price (a held price can never fire TP/SL). Config: keep `solana_price_hard_jump_ratio=5.0`, add
   `solana_price_quorum_required=true`.
4. **Void, don't pin**: in `_save_trade_to_db`, an implausible ratio should write `pnl_sol=0 / pnl_pct=0` (or
   `status='voided'`) + excluded tag — never `notional×49` profit. Also void when ENTRY price is implausible
   vs a fresh quote (catches the both-sides-poisoned class).
5. **Dashboard**: add `WHERE NOT COALESCE((metadata->>'excluded')::boolean,false)` to the `:9888` query and
   `:2112` aggregate (mirrors `:3700`); backfill-tag the 07-03-and-earlier pinned rows if metadata is NULL.
6. **Fix timestamps**: drop the `+ 'Z'` suffix for tz-aware datetimes (`:5088`, `:5089`, `:5168`).
7. **Log retention**: keep ≥3 rotations of `solana_trading.log` so validator WARN lines survive long enough to
   identify the offending source; add a `price_source` field to the trades JSON CLOSE record.

---

## B. SNIPER — loss diagnosis, empty history, tuning

### B1. What actually happened (quantified)

- **The module has been a zombie since 2026-06-15 21:49.** On restart the Solana listener logged
  `❌ RPC verification failed - check your RPC URL` (`sniper.log` 21:49:03; RPC=Alchemy, WSS=Helius) and then
  **0 pools detected across 16.9M monitor scans over 3 weeks** (`Monitor status: 16926000 scans, 0 EVM pairs,
  0 Solana pools` through 07-05). EVM sniping disabled by config; LIVE budget 0 (`budget_usd_sniper=0`).
  Last trades closed 06-16 00:03. Operator paused it 07-05 (`logs/.pause_sniper`).
- **Loss numbers (all DRY_RUN, pre-06-16):** all-time **−$4,898.11** over 24,405 filtered closed snipes,
  WR 37.43% (dashboard tile ← filtered aggregate `enhanced_dashboard.py:12278-12288`); last-1000 sample
  **−$536.56**, PF 0.56, Sharpe −3.02, avg win $1.84 vs avg loss $1.95, best +$15.34 / worst −$6.62
  (performance page). EV ≈ **−$0.20 to −$0.54 per trade**. Exit reasons: STOP_LOSS dominant, then
  `dry_run_no_price_feed`, `phantom_price_dry_run`, PARTIAL_TAKE, TAKE_PROFIT, TIME_STOP.
- **Honesty caveat:** DRY_RUN exit P&L is **MODELED**, not measured (`_model_dry_run_exit_pct` ≈ 55% loss /
  30% chop / 15% win; `_simulate_sell` clamps [−99,+200]%). The −$4,898 says "buying unfiltered pump.fun flow
  loses at the population base rate" — which is exactly what the config allowed: **TEST_MODE ON** (relaxed
  safety: `min_liq` effectively $0, up to 50% tax, DANGER-rated tokens accepted — settings page red warning),
  gates at min_holders=10, min_age=30s, min_score=40, on ALL chains, cap 500 concurrent positions.
- **Loss shape:** TP 50% / SL 20% with 50%-partial-take at +20% caps winners at ~$1.84 average while stops run
  the full −20% → PF 0.56 is structural under this asymmetry + 37% hit rate.

### B2. Why /sniper/trades is EMPTY (root cause — template JS crash)

`dashboard/templates/trades_sniper.html:636` (also `:724`, `:768`, `:785`):
`document.getElementById('filter-side').value` — **no `filter-side` element exists in the markup**; it was
replaced by `filter-chain` (`:435`), which the JS never reads. `getElementById` returns `null` → TypeError on
every `applyFilters()` call → `filteredTrades`/`displayedTrades` never populate → permanent
"Trade History ( 0 of 0 )" and $0.00 tiles. The API side is healthy: `api_get_sniper_trades`
(`monitoring/enhanced_dashboard.py:12594-12690`) returns up to 2000 rows, and dashboard logs contain zero
"Error getting sniper trades" lines. Fix: add a `filter-side` select (or read `filter-chain` and filter on
`trade.chain`, which the API already returns).

**Page-contradiction explainer** (−$4,898 vs −$537 vs 0): one table, three windows — all-time filtered
aggregate (dashboard tile) vs last-1000-trades sample (performance page) vs the JS crash above (trades page).
Label the performance page "last 1000 trades" or aggregate server-side.

### B3. Hard tuning package (config_type='sniper_config' unless noted)

**P0 — before any tuning matters:** (1) fix the Solana listener RPC/WSS credentials (Alchemy `RPC
verification failed`, Helius WSS) — the module currently detects nothing; (2) fix B2; (3) re-validate on real
price feeds — do not tune against the synthetic exit model.

| key | current | proposed | rationale |
|---|---|---|---|
| `test_mode` | true | **false** | kills the $0-liquidity / 50%-tax / DANGER-token acceptance that defines the loss population |
| `safety_check_enabled` | true | true (verify in DB) | LIVE startup guard requires it; keep dual-source honeypot quorum active |
| `min_liquidity` | 1000 | **25000** | sub-$25k pools are the 55%-loss bucket; entry gate is the cheapest edge |
| `max_buy_tax` / `max_sell_tax` | 10 / 10 | **5 / 5** | tax is a direct haircut on both legs |
| `sniper_min_holder_count` | 10 | **50** | 10 holders ≈ deployer + bundled wallets |
| `sniper_min_token_age_seconds` | 30 | **180** | aligns with W17 delayed-entry `sniper_min_entry_age_seconds=180`; skips insta-rugs |
| `sniper_min_safety_score` | 40 | **70** | 40 admits most of the DANGER band |
| `sniper_min_buy_sell_ratio` | 1.5 | **2.0** | demand confirmation before entry |
| `sniper_max_dev_holding_pct` | 30 | **15** | dev >15% = exit-liquidity risk |
| `trade_amount` | 0.1 | **0.05** (SOL) | halve per-snipe exposure until PF > 1 on real data |
| `max_active_positions` | 500 | **25** | 500 is a Phase-2 data-collection cap, not a trading cap; bounds worst-case notional to ~25×entry |
| `max_hold_minutes` | 0 (off) | **240** | time-stop retires zombies instead of `dry_run_no_price_feed` limbo |
| `take_profit_pct` / `stop_loss_pct` | 50 / 20 | **100 / 15** | pump.fun edge is long-tailed; smaller stop, bigger tail capture |
| `sniper_partial_take_pct` | 20 | **30** | stop capping winners at +20% while stops run −20% |
| `sniper_partial_take_size_pct` | 50 | 50 (keep) | de-risk half, trail the rest (`sniper_trail_after_partial=true`) |
| `target_chain` | all | **solana** | EVM listener disabled anyway; focus one venue |
| NEW `sniper_max_daily_loss_usd` | — (stub: `api_sniper_trading_status` at `enhanced_dashboard.py:12692` is hardcoded) | **50** | real daily kill; wire into engine + status endpoint |
| NEW `sniper_loss_cooldown_minutes` | — | **30 after 3 consecutive stops** | stops loss-streak bleed during hostile tape |
| NEW per-mint re-entry cooldown | — | **60 min** | prevents re-sniping the same rug |

Cross-module gap (unchanged from Phase-1): sniper still has no `RiskManager.validate_trade` call on the
snipe path (`modules/sniper/CLAUDE.md` "Primary risk-policy gate") — wire it before any LIVE flip, alongside
the existing `safety_check_enabled` startup guard.

---

*Wave-F5 agent 03 (smartcontract-web3). No code or config was modified; all fixes above are proposals.*
