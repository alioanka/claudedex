# Wave-F5 / 05 — POLYMARKET module: diagnosis, gaps, dedicated-dashboard spec

Read-only analysis, 2026-07-05. Operator ask: *"not working well and there are no visual
pages/dashboard. I need a full polymarket module ready for real trading, with full guide
and visual realtime charts/graphs dashboard."*

Sources: `logs/polymarket/polymarket.log` (63,220 lines, 2026-06-15 21:48 → 2026-07-05 19:25,
~19.9 days, no rotation yet at 6.2 MB), `modules/polymarket/*`, `migrations/101_polymarket_module.sql`,
`monitoring/enhanced_dashboard.py`, screenshots `screencapture-...-module-polymarket-2026-07-05-22_55_22.png`
and `screencapture-...-config-polymarket-config-2026-07-05-22_56_03.png`.

---

## (a) Current-state diagnosis — the numbers

**Process health: good. Signal usefulness: poor. Trading output: effectively zero.**

| Metric | Value | Evidence |
|---|---|---|
| Uptime | ~20 days continuous, zero crashes/restarts | single "Module Starting" banner at log line 1 |
| ERROR-level lines | **0** in 20 days | `polymarket_errors.log` is 0 bytes |
| Gamma API health | 9 timeout events (all clustered 2026-06-18 00:46–04:10) + 4 "No markets fetched" cycles; otherwise clean | `Gamma fetch failed (/markets): TimeoutError` |
| Momentum signals | **19,742** log lines ≈ **990/day** (daily range 703–1,254) | `grep -c "\[momentum\]"` |
| new_market signals | **14,930** ≈ **750/day** (654–842) | `grep -c "\[new_market\]"` |
| Risk-free arb signals | **1 in 20 days** — and it is a false positive (see BUG-1) | line 53600 |
| DB signal rows | 35,828 total, 1,479/24h (matches log rates) | module-polymarket screenshot |
| Simulated trades recorded | **1 total, 0 in last 24h** (`polymarket_trades`) | screenshot |
| Live orders | 0 (shadow_mode=true, live_execution_enabled=false, DRY-RUN badge) | config screenshot |

**Signal quality is not good, and is unmeasurable by design:**

- **Score saturation:** 24,470 of ~34,672 signals (**70.6%**) log `score=1.0`. `move_term`
  caps at 2× `momentum_min_move_frac` and `volume_term` at 4× `momentum_min_volume_24h_usd`
  (`strategies.py:96,125`) — nearly everything that passes the floor pins the ceiling. A score
  that is 1.0 seventy percent of the time ranks nothing.
- **Momentum = in-play sports noise.** The top-volume-200 universe is dominated by live MLB /
  tennis / esports markets; a 5-cent move inside one 60 s poll is routine *during a game* and
  meaningless as tradable momentum. The 24h signal table in the screenshot is almost entirely
  "Pirates vs Nationals / Rays vs Astros / Wimbledon". Direction flip-flops within minutes:
  market **2788061** logged `dir=NO yes=0.565` (19:01), `NO yes=0.275` (19:06), `NO yes=0.195`
  (19:11), then `dir=YES yes=0.215` (19:21) on 2026-07-05.
- **No outcome tracking.** Momentum/new_market signals are written and forgotten — no forward
  return at +1h/+24h, no hit rate, no realized-edge column. 34k signals in 20 days have produced
  zero evidence of edge either way. Control-center honestly reports `pnl_available: False`
  (`enhanced_dashboard.py:3782`).
- **new_market mislabels.** "New" = absent from the *previous top-200 snapshot*
  (`strategies.py:97-101`), so any old market re-entering the volume window is flagged: e.g.
  `new_market 1931118 yes=0.011`, `2798678 yes=0.9995`, "Will LeBron James play…" at yes=0.004
  in the screenshot. WATCH advice on a 99.95%-priced market is useless; near-resolved markets
  (yes ≥0.99 / ≤0.01) are not excluded.

**Operator-visible surface:** only the *generic* `/module/polymarket` panel
(`_PANEL_TRADING_MODULES`, `enhanced_dashboard.py:3000-3016`) — two raw table dumps
(`polymarket_signals`, `polymarket_trades`) with a horizontal scrollbar, plus the generic
`/config/polymarket_config` key editor. **Zero charts, zero market view, zero PnL, zero gate-chain
visualization.** Every other trading module has a dedicated page set (`dashboard_futures.html`,
`positions_*`, `trades_*`, `performance_*`, `settings_*`). The operator's complaint is accurate.

---

## (b) Bugs found (evidence + file:line)

**BUG-1 (P0 for live) — Arb detector fires on untradeable stale mid-prices; the only arb signal
in 20 days was a false positive.**
`detect_risk_free_arb` (`modules/polymarket/strategies.py:19-62`) uses Gamma `outcomePrices`
(mid/last), only rejects `yes<=0 or no<=0`, and checks **no liquidity floor and no best_bid/best_ask**
(both parsed and available in `gamma_client.py:98-99` but ignored). The one firing —
`2026-07-02 16:19:06 [arb] 2756683 edge=2950.0bps pair_cost=0.695` on "Matteo Berrettini vs.
Arthur Fils: Total Sets O/U" — is a decided/dead tennis book where mids drift; a 29.5% "risk-free"
edge on a real book does not survive minutes, let alone sit at one detection in 20 days. The
docstring itself admits it (`strategies.py:4-6`: "shadow edges are an upper bound and the live
path MUST re-quote against the CLOB order book") — but the live path does **not** re-quote.

**BUG-2 (P0 for live) — The "risk-free arb" live path would place a single-leg directional bet.**
`main_polymarket.py:176-184` calls `executor.execute(..., outcome='BOTH',
token_id=sig['details'].get('yes_token_id'), price=sig['yes_price'], ...)` — one call, YES token
only. The executor live branch (`executor.py:129-141`) builds **one** `OrderArgs` and posts **one**
GTC order. With all gates open this buys only the YES leg at the YES mid — that is not an arb, it
is an unhedged position, on a signal that BUG-1 shows is usually a stale book. The simulated record
has the same flaw: one row at `price=yes_price`, the NO-leg cost is unrecoverable from the ledger.

**BUG-3 (P1) — RiskManager gate is semantically wrong for Polymarket.**
`executor.py:55` calls `RiskManager.validate_trade(token_id, size_usd)`;
`core/risk_manager.py:1097+` runs `analyze_token(token_address)` — honeypot/liquidity analysis
built for EVM DEX tokens. A CLOB token id (a ~77-digit numeric string) is not a token address;
the gate will either error (→ `risk_manager_error`, permanently blocking live) or produce
garbage. Also `RiskManager` is constructed with `config={}, portfolio_manager=None`
(`main_polymarket.py:325`) so no portfolio-aware limits exist. Net effect today: fail-closed
(safe) but unusable as the real live gate.

**BUG-4 (P1) — Score formula has no discriminating power.** 70.6% of signals at score=1.0
(see above; `strategies.py:79-83,96,125-126`). `momentum_min_score=0.3` filters almost nothing.

**BUG-5 (P2) — new_market false positives + no near-resolution filter**
(`strategies.py:97-101,123`): membership test against the previous top-200 snapshot ≠ "new";
`yes in (0.0,1.0)` only excludes exact 0/1, so 0.9995/0.011 markets signal freely.

**BUG-6 (P2) — Duplicate log lines from sub-loggers.** `main_polymarket.py:52-57` attaches the
parent's handlers to `PolymarketModule.Gamma`/`.Executor` without `propagate=False`; records
propagate to the parent's identical handlers → every Gamma warning is written twice (visible as
`uniq -c` = 2 on each timeout line). Inflates counts and file size.

**BUG-7 (P2) — `momentum_min_volume_24h_usd` is consumed (`main_polymarket.py:195`, code default
5000) but never seeded in migration 101** (19 keys; it is absent from the config screenshot) —
invisible and untunable from the UI.

**BUG-8 (P3) — config editor shows locale-comma floats** (`0,05`, `0,3` in the config screenshot).
If an operator ever saves a comma form, `load_config` (`main_polymarket.py:82-84`) keeps it a
string and `float("0,05")` in `_cycle` raises → every cycle errors (fail-soft, but signals stop).
Display/validation hazard.

**BUG-9 (P3) — minor engine hygiene:** new `aiohttp.ClientSession` per request
(`gamma_client.py:152`); throttle map wholesale `clear()` at >5000 entries resets all throttles
at once (`main_polymarket.py:122-123`); `/status` is an alias of `/health` rather than a richer
stats payload (`main_polymarket.py:246`).

---

## (c) Live-readiness gap list (ranked)

The gate chain (shadow_mode → live_execution_enabled → should_skip_live → RiskManager →
py-clob-client + key) is correctly fail-safe, `py-clob-client==0.17.5` is in the Docker image
(`Dockerfile:89`), and mig-101 contract addresses are correct. Everything after the gate is
missing or wrong:

1. **Executable pricing (blocker).** Re-quote the CLOB order book (best ask, depth for size)
   before any order; drop Gamma mids from the execution decision entirely. Fixes BUG-1.
2. **Two-leg atomic-ish arb execution (blocker).** Both legs (YES ask + NO ask), FOK/FAK order
   type, leg-fill mismatch handling (unwind or hold-to-resolution policy). Fixes BUG-2.
3. **Order lifecycle (blocker).** `post_order` response is fire-and-forget today
   (`executor.py:139-141`). Need: order status polling, fill/partial-fill booking, cancel on
   timeout, open-order registry, restart reconciliation against the CLOB API
   (`get_orders`/`get_trades`).
4. **Wallet/funding path (blocker).** No USDC.e balance check; no one-time approvals (USDC →
   CTF Exchange, CTF ERC-1155 setApprovalForAll); no `signature_type`/`funder` support —
   `ClobClient(host, key, chain_id)` (`executor.py:86`) is EOA-L1 only, so a Polymarket-UI
   (proxy) wallet cannot trade without signature_type 1/2 + funder address. Needs an operator
   setup script + config keys.
5. **Polymarket-specific risk gate (blocker).** Replace/augment `validate_trade` (BUG-3) with:
   per-market exposure cap, total module exposure cap, daily loss cap, max open markets,
   market end-date buffer (no entries < N hours to resolution), min liquidity/depth floor,
   price-band sanity (reject <0.02 / >0.98). Only per-trade `max_position_size_usd=50` exists.
6. **Resolution & settlement (blocker for PnL).** Nothing watches `closed`/`umaResolutionStatus`;
   winning tokens are never redeemed (CTF `redeemPositions`); no realized-PnL column → the
   performance page has nothing to show. Needs a resolution watcher + redemption step (or
   sell-before-resolution policy) + `polymarket_positions` table with mark-to-market.
7. **Position model.** Only a trade ledger exists; no open-position aggregation, no per-market
   net exposure, no reconciliation of DB vs on-chain/CLOB balances on restart.
8. **Prove edge before live.** Add forward-outcome scoring of shadow signals (yes_price at
   +1h/+24h/resolution vs direction) and a proper shadow-PnL sim for arb (both-leg cost from
   book snapshots). Fix score saturation (BUG-4), exclude in-play sports or make
   `category_filter` the default posture, exclude near-resolved markets (BUG-5). Going live on
   the current signal stream would be uninformed gambling.
9. **Secrets/ops.** `POLYMARKET_PRIVATE_KEY` via secrets_manager is wired (`executor.py:73-79`);
   document rotation + derive-API-creds behavior; add `signature_type`/`funder_address` config
   keys; seed `momentum_min_volume_24h_usd` (BUG-7).
10. **Hygiene.** BUG-6/8/9; add real `/status` stats payload for the dashboard.

---

## (d) Dedicated dashboard spec — `/polymarket/*` page set

**Reuse (all existing):** templates extend `dashboard/templates/base.html` (Chart.js 4.4.0 already
loaded via CDN, `base.html:28`); reference implementation `dashboard_futures.html` — canvases +
`new Chart(...)` + `fetch('/api/...')` + `setInterval(...,30000)` polling + runtime badge from
`/api/modules/<key>/runtime-status` (`dashboard_futures.html:680,817,1615,1639`);
`dashboard/static/js/charts.js` ChartManager (`createPnLChart`, `createWinRateChart`);
routes registered in `monitoring/enhanced_dashboard.py` next to the futures block
(`:1613-1617` fallback pattern, API routes `:1362-1370` pattern). Update `_HELP_TRADING_MODULES`
`panel` from `/module/polymarket` → `/polymarket/dashboard` (`enhanced_dashboard.py:3617`).

**New data source needed for real charts:** a per-cycle snapshot table
`polymarket_market_snapshots(market_id, question, category, yes_price, no_price, best_bid,
best_ask, liquidity, volume_24h, ts)` written by the engine for the top-N (e.g. 50) watched
markets each poll (engine already holds them in `_cycle`; ~50 rows/min, add retention/downsample
job). Without it, YES/NO price charts can only be reconstructed from `polymarket_signals`, which
is sparse and biased to signal moments.

### Pages

**1. `/polymarket/dashboard` — Overview (template `dashboard_polymarket.html`)**
- Header: runtime badge (enabled/running/paused/DRY-RUN) + **gate-chain card** rendering each of
  the 5 live gates green/red (shadow_mode, live_execution_enabled, killswitch/pause/dry_run, risk
  gate, clob client+key) — data from `/api/polymarket/overview`.
- Stat tiles: signals 24h (by type), markets seen last cycle, Gamma last_error/last_fetch age,
  simulated trades, live orders, cycles, arb signals all-time.
- Chart A — *Signals per hour, stacked bar* (momentum / new_market / arb), 24h–7d selector.
  Source: `SELECT date_trunc('hour', created_at), signal_type, COUNT(*) FROM polymarket_signals`.
- Chart B — *Arb edge over time, scatter* (edge_bps vs created_at, point size = liquidity) with
  a threshold line at `min_arb_edge_bps`. Source: signals where `signal_type='risk_free_arb'`.
- Chart C — *Score distribution histogram* (exposes BUG-4 saturation to the operator).
- Table: last 20 signals (reuses generic panel query).

**2. `/polymarket/markets` — Market catalog + realtime prices**
- Sortable/filterable table of watched markets (question, category, YES, NO, YES+NO pair cost,
  spread best_bid/ask, liquidity, vol24h, end_date, signals-24h count) from
  `/api/polymarket/markets` (latest snapshot per market).
- Inline YES-price sparklines; click → detail drawer with full-width *YES/NO price line chart*
  (dual series, 0–1 y-axis) + liquidity/volume subchart from
  `/api/polymarket/market/{id}/history?hours=24` (snapshot table).
- "Pair cost < 1" column highlighted when below `1 - buffer` (visual arb radar).

**3. `/polymarket/signals` — Signal stream & quality**
- Filters: type / direction / min score / market search; paged table from
  `/api/polymarket/signals?hours=&type=&min_score=`.
- Chart: *signal outcome scatter* — score vs forward YES-price move at +1h/+24h (needs gap-8
  outcome scoring job; ship the panel with an honest "outcome tracking pending" empty state).
- Per-market signal concentration bar (top 15 market_ids by signal count).

**4. `/polymarket/positions` — Open shadow/live positions**
- Requires the position model (gap 7). Table: market, outcome, entry price, size, current
  YES/NO mark (snapshots), unrealized PnL, hours-to-end_date, resolution status, is_simulated.
- Donut: exposure by category; bar: per-market exposure vs cap. Buttons (live era): close /
  cancel open orders — POST endpoints mirroring `/api/futures/position/close` (`:1364`).

**5. `/polymarket/performance` — PnL & scorecard**
- *Cumulative simulated PnL line* (mark-to-market from snapshots + resolution settlements once
  gaps 6–7 land) via `charts.js createPnLChart`; win-rate donut (`createWinRateChart`);
  realized-vs-expected edge bar for arb trades; daily PnL bars.
- Until settlement exists: show trade ledger stats + explicit "no realized PnL yet — shadow
  accounting requires resolution watcher" banner (keeps the honesty convention of
  `pnl_available=False`, `enhanced_dashboard.py:3757-3782`).

**6. `/polymarket/settings`** — keep the existing generic `/config/polymarket_config` editor;
add a link plus a read-only "effective gate state" strip (same component as page 1).

### API endpoints (all fail-soft, `enhanced_dashboard.py` conventions)
- `GET /api/polymarket/overview` — gate chain + `:8089/status` proxy + tile numbers.
- `GET /api/polymarket/signal-stats?hours=` — hourly buckets by type.
- `GET /api/polymarket/signals?hours=&type=&min_score=&limit=`.
- `GET /api/polymarket/markets` / `GET /api/polymarket/market/{id}/history?hours=`.
- `GET /api/polymarket/trades?limit=` / `GET /api/polymarket/positions`.
- `GET /api/polymarket/performance?days=`.
- Realtime = 15–30 s `setInterval` polling (futures pattern); `websocket.js` exists if push is
  wanted later, but polling matches every sibling page.

---

## (e) Operator guide outline (`docs/POLYMARKET_GUIDE.md` — to be written in the fix wave)

1. **What this module is** — shadow-first prediction-market engine; strategy definitions
   (YES+NO<1 arb, momentum advice); what "simulated" means; where data lands
   (`polymarket_signals`, `polymarket_trades`, mig 101).
2. **Running in shadow** — `POLYMARKET_MODULE_ENABLED=true`, health `:8089/health`, log paths,
   pause/killswitch, reading the new dashboard pages.
3. **Tuning** — every `polymarket_config` key with defaults and effects; recommended posture:
   `category_filter` to exclude in-play sports, raise `momentum_min_move_frac`, near-resolution
   exclusion; how `shadow_record_interval_s` throttles rows.
4. **Proving edge before live** — read the signal-outcome panel; acceptance criteria (e.g. arb
   detections re-quoted against real books, momentum hit-rate > 55% over N≥200 out-of-sample
   signals) before touching any live flag.
5. **Going live checklist (in order)** — dedicated Polygon wallet; fund POL (gas) + USDC.e;
   proxy vs EOA decision (`signature_type`/`funder`); one-time approvals (USDC→CTF Exchange,
   CTF setApprovalForAll); store `POLYMARKET_PRIVATE_KEY` in secrets manager; verify
   `clob_client_unavailable` no longer appears in skip_reasons; set risk caps
   (per-market/total/daily-loss) LOW; flip `shadow_mode=false`, then `live_execution_enabled=true`;
   first trade at minimum size; watch order-lifecycle panel.
6. **Live operations** — order states, partial fills, cancel flow, position reconciliation on
   restart, resolution/redemption procedure, fee accounting.
7. **Incident response** — pause file vs killswitch vs `live_execution_enabled=false` (which to
   use when), what each stops, how to verify nothing is resting on the book after a halt.
8. **Known limitations** — Gamma mids vs CLOB asks, in-play sports noise, resolution (UMA)
   risk, single-venue concentration.

---

## Verdict

Infrastructure and safety gating: solid (20 days, zero crashes, zero ERRORs, gates verified
fail-safe). Strategy layer: not producing tradable signal (1 arb in 20 days = a false positive;
~1,700 momentum/new_market rows/day of mostly in-play sports noise with a saturated score and no
outcome tracking). Live path: present but **wrong** (single-leg "arb", mid-price execution, no
order lifecycle, no settlement, mismatched risk gate) — flipping the gates today would lose money
mechanically, independent of signal quality. Recommended sequence: fix BUG-1/2 + snapshot table +
outcome scoring → ship the `/polymarket/*` dashboard (all reusable patterns exist) → run a
measured shadow month → only then execute the going-live checklist.
