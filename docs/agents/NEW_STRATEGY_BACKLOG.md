# New-Strategy Backlog — researched, prioritized, skeptical

Date: 2026-06-12. Author: market-trading-analyst agent.
Scope: candidate NEW profitable modules/strategies for ClaudeDex, ranked by
(expected edge durability x feasibility on EXISTING infra / build cost).
Every entry states the edge thesis, why it might NOT work (decay/competition),
data + infra needed, and where it slots in. Companion to the meta_controller
CLAUDE.md 3-idea backlog; idea #3 there (volatility-regime allocator) is now
BUILT (`modules/regime_allocator/`, migs 118-119) and is removed from this list.

Ground rules applied to every candidate:
- Must respect the existing safety stack (should_skip_live, killswitch,
  RiskManager / FuturesRiskManager, DRY_RUN-first, shadow-first).
- Sized for a small book (low 4-figures USD): capacity and fee drag matter
  more than headline Sharpe. Strategies that only work above ~$1M are cut.
- No martingale, no dynamically widening stops, no unbounded inventory.

---

## P1 — build next

### 1. Cross-venue funding & basis desk (delta-neutral carry)
- **Edge thesis:** Perp funding is a persistent, directly observable risk
  premium — longs pay shorts most of the time in bull tape and the sign flips
  are themselves tradable. The bot already harvests HALF this trade
  (futures funding-carry v2, mig 102, entries via `_open_position`); the
  missing piece is the **automated spot hedge leg** so the position is
  genuinely delta-neutral instead of "short perp and hope". Net edge per
  position must clear `funding * notional - taker_fees * 2 -
  expected_slippage - liquidation_premium` (sizing rule already in the
  futures module's policy).
- **Why it can fail / decay:** Funding compresses exactly when everyone
  crowds the carry; exchange outages mid-position turn a neutral book
  directional; spot borrow for the short-spot direction is not always
  available. The premium is real but cyclical — expect long flat stretches.
- **Data/infra needed:** Bybit V5 spot order endpoint (same keys, ccxt
  already wired), funding-rate history table (exists: mig 029
  `futures_funding_payments`), a reconciliation loop asserting |perp qty -
  spot qty| < epsilon every tick, kill-to-flat on hedge-leg failure.
- **Feasibility: HIGH** (one new executor leg + a pairing reconciler).
  **Risk: LOW-MED** (market-neutral by construction; main risks are
  operational: leg-out, funding flip, fee drag).
- **Slot-in:** extend `modules/futures_trading/strategies/funding_carry.py`
  into a two-leg strategy; risk gates stay inside
  `FuturesRiskManager.validate_new_position`. Shadow-first like Polymarket.

### 2. Statistical pairs / cross-sectional mean reversion on liquid perps
- **Edge thesis:** Cointegrated or beta-stable alt pairs (e.g. L1 vs L1,
  DOGE/SHIB-style same-narrative pairs) mean-revert on hours-days horizons;
  cross-sectional version (short the day's biggest beta-adjusted outlier vs
  basket) is the classic small-book market-neutral trade. Fits the
  regime_allocator loop perfectly: this is THE module to upweight in
  `range_compression`.
- **Why it can fail / decay:** Crypto cointegration is regime-fragile —
  pairs break on idiosyncratic news (unlock, hack, listing) and the loss on a
  broken pair is fat-tailed. Must hard-stop on spread z-score blowout
  (NO averaging down — repo rule), cap per-pair notional, and re-test
  cointegration on a rolling window with a kill criterion.
- **Data/infra needed:** Bybit klines (free, already used), a pair-scanner
  job (pure math, offline-testable like `health_scorer`), two-leg
  reduceOnly/ISOLATED execution via existing futures engine.
- **Feasibility: HIGH** (all data + execution exists). **Risk: MED**
  (tail risk on pair breaks; mitigated by hard z-stop + per-pair caps).
- **Slot-in:** new strategy file under `modules/futures_trading/strategies/`,
  default-OFF DB flag, entries through `_open_position` so every existing
  gate applies. DRY_RUN season of 4+ weeks before any live flip.

### 3. On-chain smart-money flow follower (generalized copy)
- **Edge thesis:** Copy_trading already mirrors NAMED leaders. Generalize to
  cluster-detected accumulation: score wallets by survivorship-bias-free
  realized forward return (the leader-lifecycle plumbing from mig 092 is
  reusable), follow the top decile's NEW positions with fractional-Kelly
  caps. This is information flow, not prediction — the edge is being 10
  minutes behind smart money instead of 2 days behind CT.
- **Why it can fail / decay:** Crowded — Nansen/Arkham followers compress
  the window; wallet clustering has false positives (CEX hot wallets, market
  makers); leaders rotate wallets. Score decay must be measured (the
  copy module's 30/90/180d Sharpe + concentration scoring applies directly).
- **Data/infra needed:** heavy collector work — wallet-label clustering over
  existing RPC pool (`config/pool_engine.py`), forward-return labeler,
  candidate-pool table. No paid API strictly required but Helius/Alchemy
  quotas will be the binding constraint.
- **Feasibility: MED** (collector-heavy). **Risk: MED** (crowded-trade decay,
  data quality). **Slot-in:** discovery engine inside
  `modules/copy_trading/` feeding the existing leader-scoring pipeline.

## P2 — worth a design doc, not a build yet

### 4. Liquidation-cascade liquidity provision (perps)
- **Edge:** resting reduce-risk limit bids below visible liquidation clusters
  (Bybit publishes liq data; open-interest + leverage tiers approximate the
  map) capture forced-flow discounts. Small-book friendly: it is capacity-
  constrained per event but events repeat weekly.
- **Skepticism:** the map is approximate; a cascade that blows THROUGH your
  bid is the same trade with the sign flipped. Needs strict per-event max
  loss + immediate hard stop, and honest measurement of fill toxicity.
- **Feasibility MED / Risk MED-HIGH.** Slot-in: futures module strategy with
  its own breaker; never sized above per-symbol tier caps (mig 088).

### 5. Polymarket <-> perp basis/event hedge
- **Edge:** "BTC above X by date" markets frequently misprice vs the
  option-implied / perp-realizable probability; the bot already has the
  Gamma reader + shadow ledger (mig 101). Hedging a YES/NO position with
  small perp delta converts a gamble into a basis trade.
- **Skepticism:** CLOB depth is thin; fees + spread eat most small edges;
  resolution risk is binary and oracle-dependent. Strictly shadow-first,
  and the live gate already requires four independent switches.
- **Feasibility MED / Risk MED.** Slot-in: `modules/polymarket/strategies.py`
  new strategy + a delta line item in the futures book.

### 6. Stablecoin / LST depeg monitor + reversion
- **Edge:** episodic but fat: USDe/USDT, stETH/ETH style dislocations revert
  unless the issuer is actually dead. A monitor that alerts + (live-gated)
  buys small clips beyond a threshold with hard time/loss stops is cheap to
  run and dovetails with the arbitrage module's venue plumbing.
- **Skepticism:** the tail case (it was a REAL depeg) is exactly when you
  lose; position caps must assume 100% loss. Months of nothing between
  events — meta_controller will (correctly) score it as idle.
- **Feasibility HIGH / Risk MED (tail).** Slot-in: `modules/arbitrage/`
  side-strategy, default OFF, alert-first.

## P3 — researched, explicitly NOT recommended now

### 7. CEX-DEX latency arbitrage
True latency arb (race the CEX print to the AMM quote) is an HFT war:
colocated players + private orderflow + builders win it. The bot's existing
spatial arb with econ gates (migs 097/108) already captures the slow tail of
this; building a dedicated latency module would burn gas and dev time against
competitors we cannot beat. Revisit only if a private-builder relationship
materializes.

### 8. Options / vol-surface desk (Deribit)
Selling vol premium is profitable until it very much is not; small-book
margin requirements + 24/7 gamma risk + a whole new venue integration make
this the worst effort/edge ratio on the list. The regime classifier built in
this wave provides the main thing options would have given us (a vol-state
signal) for free.

### 9. LP / JIT liquidity provisioning
JIT on Uniswap v3 is a builder/MEV-infrastructure game (same war as #7);
passive LP is short-vol-without-the-premium (impermanent loss ≈ selling
straddles for less than fair value). Neither fits this stack.

---

## How this interacts with what was built this wave
The REGIME_ALLOCATOR (built; `modules/regime_allocator/`) is the routing
layer for the P1 items: funding/basis desk and stat-pairs are the
`*_compression` beneficiaries, the existing momentum modules own
`trend_expansion`, and `chop_expansion` shifts weight to reserve + arb. Each
new module lands with a `*_trades` table + heartbeat so meta_controller and
portfolio_allocator score it with zero extra wiring.

Build order recommendation: **#1 funding/basis desk -> #2 stat-pairs ->
re-evaluate #3 after measuring copy-module live decay.** Everything ships
DRY_RUN/shadow-first with a measured 4+ week paper season and the standard
live-readiness checklist (DRY_RUN honored on every send path, per-trade/hour/
day loss caps, startup reconciliation, idempotent order IDs, dashboard
heartbeat, emergency-stop reachable) before any live flip.
