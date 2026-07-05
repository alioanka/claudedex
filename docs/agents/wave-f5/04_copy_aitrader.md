# Wave-F5 / 04 — COPY discovery root cause + HKUDS/AI-Trader review

Analyst: market-trading-analyst (read-only wave; no code changed).
Scope: operator complaint "wallet discovery is not working again. it is finding my already tracked wallet always" + adaptability review of https://github.com/HKUDS/AI-Trader.

---

## A. Root cause: discovery only ever returns the already-tracked wallets

**Verdict: not a dedup/seen-set bug. Every external candidate source is returning
zero rows, and every discovery path then degrades — by design — to a "local
fallback" whose only inputs are the operator's own `target_wallets` plus
`copytrading_trades.source_wallet` (i.e., the wallets already being tracked).
The fallback is then presented on the UI as discovery output.** Meanwhile the
real fix that shipped for this exact complaint (v3 discovery, mig 135) has
never executed once, because its gate flag is still off.

### A.1 The path the operator actually sees (dashboard "Wallet Discovery" page)

`/copytrading/discovery` → `GET /api/copytrading/discover`
(`dashboard/templates/discovery_copytrading.html:646`, handler
`monitoring/enhanced_dashboard.py:15156`). Source cascade:

1. **Helius** (`_discover_wallets_helius`, `enhanced_dashboard.py:15578`) —
   pulls the **last 100 SWAP txs** from two DEX program addresses (Jupiter v6,
   Raydium V4; `:15621`), counts swaps per fee payer, then filters
   `stats['trades'] < min_trades` (`:15672`, UI default **5**;
   `discovery_copytrading.html:437`). Two independent kill modes:
   - **Sampling flaw:** 100 txs on Jupiter v6 span ~seconds of chain time; a
     wallet must appear ≥5 times inside that snapshot to survive. Essentially
     only MEV/arb bots ever can, and usually nobody does → 0 candidates.
   - **Helius is hard-429'd anyway:** any non-200 → `continue` (`:15624-15626`).
     Log evidence below shows the Helius key is rate-limited around the clock,
     so this source frequently returns nothing even before the filter.
2. **Birdeye** (`:15216`, `_discover_wallets_birdeye:15486`) — skipped: no
   `BIRDEYE_API_KEY` configured (no key in `.env.example` deployment, no
   "Discovered N wallets via Birdeye API" line anywhere in logs).
3. **Local fallback** (`:15231-15239`, `_discover_wallets_local_fallback:15707`)
   — explicitly built from `copytrading_trades` source wallets +
   `config_settings.copytrading_config.target_wallets`. **This always fires,
   and by construction it can only return already-tracked wallets.** The UI
   renders these rows in the same card as real discoveries (the
   `data_source='operator_targets+onchain'` field exists in the JSON but the
   page does not make "this is your own wallet" visually loud).

Result: the operator clicks "Find New Wallets" and gets back the 4 wallets
they already track (screenshot `screencapture-...-copytrading-wallets-2026-07-05-22_54_37.png`:
TRACKED WALLETS 4; `H4Ln48XG…`, `D3zbRTT8…`, `5pziQHHK…`, `Coyadnds…`).

### A.2 The v2 sweep (`/api/copytrading/leaders/refresh`) has the same disease

`modules/copy_trading/wallet_discovery.py::discover_and_score` (line 548):

- `fetch_operator_targets` (`:358`) **always seeds the tracked wallets** into
  the candidate set ("the operator already vouched for them").
- `fetch_onchain_local` (`:419`) ranks `copytrading_trades.source_wallet` —
  again only tracked wallets.
- `fetch_dexscreener_top_traders` (`:163`) is **structurally broken**: it
  harvests `pairAddress` — an AMM **pool contract**, not a trader wallet —
  plus a `info.deployerAddress` field DexScreener does not actually return
  (`:204-222`). `_looks_like_wallet` (`:477`) cannot tell a pool address from
  a wallet on either chain, so at best this source injects junk pool addresses
  that score ~0 (no trade history) and never rank. DexScreener has no public
  top-traders REST API; this adapter can never work as written.
- `fetch_gmgn_smart_money` (`:267`) — gmgn.ai aggressively Cloudflare-blocks
  non-browser clients; failure is swallowed at DEBUG (`_safe_get_json`,
  `:155,159`), so it silently returns [].
- `fetch_birdeye_top_traders` (`:225`) — no key → `return []` (`:233-234`).
- `fetch_helius_active` (`:305`) — same 100-tx program-snapshot approach as
  A.1, same 429 exposure.

So the scored output = tracked wallets (which have history in
`copytrading_trades` and therefore nonzero scores) on top, junk/nothing below.
"Discovery finds my already tracked wallet always" — exactly.

### A.3 The v3 engine (mig 135) — the shipped fix — has NEVER run

`copy_engine.py:1772 _maybe_run_discovery_v3` gates on
`copy_auto_discovery_enabled` (mig 092 seed, **default false**;
`copy_engine.py:1352,2091-2096`).

**Log evidence** (`logs/copy_trading/copy_trading.log`, 9,223 lines,
2026-06-15 → 2026-07-05):

- `grep -c 'discovery-v3\|sweep' copy_trading.log` → **0**. The
  `[discovery-v3] sweep: N candidates…` summary line (`discovery_v3.py:586-591`)
  never appears once in 20 days → the sweep never executed → the flag is still
  false in the DB.
- Even if flipped on today, three of its five sources are compromised:
  - `smart_money` (its best feed) is **EVM-only in v1** and the copy module's
    EVM leg is currently down — settings screenshot shows
    "ETHERSCAN_API_KEY is not currently set. EVM wallet monitoring is
    disabled." Candidates would be discovered on chains the engine can't watch.
    (smart_money itself is healthy: `logs/smart_money/smart_money.log:22418`
    "tick: ingested=60 marked=0 **scored=497** signals=0".)
  - `onchain` + `leader_scores` recycle tracked wallets / v2 output (A.2).
  - `dexscreener` reuses the broken v2 fetcher (`discovery_v3.py:210-231`).
  - `rpc_solana` Helius enrichment needs a working Helius quota — see A.4.
- Qualification bar `copy_v3_min_trades=10` + `copy_v3_min_realized_pnl_usd=500`
  (`discovery_v3.py:119-120,554-560`) means candidates **without event
  history** (all dexscreener rows) can never be proposed — documented as
  "honest, not a defect", but it means the only proposable wallets are ones
  with history in our DB = tracked wallets, unless smart_money/rpc_solana
  actually deliver events.

### A.4 The operational kill: Helius quota is exhausted 24/7

- `grep -c '429|rate limited' copy_trading.log` → **3,386 lines** in 20 days.
- Rate-limited **0.2 s after startup**: `copy_trading.log:28`
  (2026-06-15 21:49:03,498 "Solana RPC rate limited in fallback poll (Helius
  endpoint) - backing off") and still going at the end of the log
  (`:9222`, 2026-07-05 19:22:44).
- The same key serves the copy monitor's per-wallet enhanced-tx REST poll, the
  dashboard discovery path, and v3's `rpc_solana` enrichment. While it 429s,
  **every Helius-backed discovery source dies at `status != 200`** and the
  cascade lands on the local fallback. (Wave-19 already predicted this failure
  mode; with 4 wallets at 15 s the copy poll alone is ~23k calls/day before
  sniper/solana/dashboard share the key.)

### A.5 Why the operator sees no diagnostics

Every source failure in both v2 and v3 is logged at **DEBUG**
(`wallet_discovery.py:155,159`; `discovery_v3.py:96,230,310,489-490,536,541,570,577`)
and the deployed log level is INFO — so "sources queried / candidates found /
filters applied" is invisible. The log contains literally zero discovery
lines. Silent degradation is the reason this is the second time ("again") the
operator has hit the same wall without a trail.

---

## B. Concrete fix (ranked; smallest honest change first)

1. **Stop the fallback masquerading as discovery (dashboard, ~15 lines).**
   In `api_copytrading_discover` / `_discover_wallets_local_fallback`
   (`enhanced_dashboard.py:15231,15707`): exclude wallets already present in
   `target_wallets` from the returned list (or tag rows
   `already_tracked=true` and render a distinct "You already track this"
   badge + a visible banner "External sources returned 0 candidates —
   reason: helius_429 / no_birdeye_key"). Discovery must be allowed to
   honestly return **zero**.
2. **Raise source diagnostics DEBUG→WARNING with counts (v2+v3, ~10 lines).**
   One line per source per sweep: `source=helius candidates=0 reason=http_429`.
   This alone would have made the root cause operator-visible a month ago.
3. **Fix the Helius quota starvation (config-only).** Either raise
   `copy_poll_interval_s` (mig 068 knob) to 30–60 s and confirm the sniper /
   solana modules aren't burning the same key, or upgrade the Helius plan.
   Until 429s stop, no Helius-backed source can work. Verify with the
   per-endpoint WARNING already shipped in Wave-19.
4. **Enable the v3 pipeline — it was built for exactly this complaint.**
   Set `copy_auto_discovery_enabled=true` (mig 092 key) so
   `_maybe_run_discovery_v3` (copy_engine.py:1772) runs daily, and set
   `copy_shadow_sim_enabled=true` so proposals accumulate shadow fills.
   Pair with (3), otherwise `rpc_solana` enrichment 429s out.
5. **Give v3 a Solana-native candidate feed (the real gap, ~1 day).**
   The only structurally sound free Solana source is the one v3 already has —
   Helius enhanced-tx — but pointed at **token pages, not program snapshots**:
   take the day's top-volume tokens (DexScreener token endpoints DO work for
   tokens), pull each token's recent SWAP txs, aggregate fee payers **across
   sweeps into `copy_discovered_wallets`** (persistent accumulation fixes the
   100-tx-snapshot sampling flaw), then let `wallet_profitability.score_wallet`
   rank on realized PnL. Add as a new `helius_tokens` source in
   `discovery_v3.py`; delete or quarantine the pair-address DexScreener
   adapter (`wallet_discovery.py:204-222`) — it can never produce a wallet.
6. **EVM leg:** restore `ETHERSCAN_API_KEY` (settings page shows it unset) so
   smart_money's 497 scored EVM wallets become actionable copy candidates;
   otherwise drop `smart_money` from `copy_v3_sources` to avoid proposing
   leaders the engine cannot mirror.

Non-goals: do NOT loosen `copy_v3_min_score/min_trades/min_realized_pnl_usd`
to force output — proposing unscoreable wallets is how copy books die. Fix the
candidate supply, keep the bar.

---

## C. HKUDS/AI-Trader review

### C.1 What it is

Two overlapping identities in one MIT-licensed repo:

1. **Research benchmark** ("AI-Trader: Benchmarking Autonomous Agents in
   Real-Time Financial Markets", arXiv:2512.10971): LLM agents (DeepSeek,
   MiniMax-M2, etc.) autonomously trade NASDAQ-100 / SSE-50 / crypto through a
   standardized MCP toolchain; live mark-to-market leaderboard decides the
   winner. Headline finding from their own report: **most LLM agents lose
   money and show weak risk management**; only 1–2 models were consistently
   profitable in any market.
2. **Agent-native copy-trading platform** (current README): FastAPI backend +
   React frontend + `skills/` (`ai4trade`, `copytrade`, `tradesync`) that any
   coding agent (Claude Code, Codex, Cursor…) can drive. Agents publish
   signals (`position` / `trade` / `realtime` types) via
   `POST /api/signals/realtime`; followers browse `GET /api/signals/feed`
   ranked by return rate / win rate / subscriber count and mirror **1:1** via
   `POST /api/signals/follow`. Paper trading ($100k simulated), Polymarket
   paper mode, broker sync (Binance/Coinbase/IBKR).

### C.2 The two things the operator asked about

- **Wallet discovery: NOTHING to port.** AI-Trader has **zero on-chain wallet
  tracking or wallet discovery**. Its "discovery" is browsing a feed of agents
  registered on its own platform — a curated, self-reported universe. It does
  not solve, or even face, our problem (finding profitable anonymous wallets
  from raw chain data).
- **Copy trading: ClaudeDex is strictly ahead.** Their mirroring is 1:1
  notional, "fully automatic copying with no custom sizing options", no
  documented risk controls, no slippage model, no probation, no promotion
  gate. Our stack (fractional-Kelly capped sizing, probation, cross-module
  exposure caps, shadow simulator, operator-approval promotion) is what they
  would need to build. Porting their copy engine would be a regression.

### C.3 Adaptable components, ranked

| # | Idea | Worth it? | Concrete ClaudeDex integration point |
|---|------|-----------|--------------------------------------|
| 1 | **Continuous mark-to-market scoring of leaders** (their leaderboard re-marks live, ours marks shadow positions only "at last observed event price — stale between events", a documented limitation) | **Yes — best single port.** | `modules/copy_trading/shadow_simulator.py`: add a periodic re-mark tick that prices open `copy_shadow_positions` via the dashboard's existing Jupiter Price v3 fetcher (30 s TTL cache in `monitoring/enhanced_dashboard.py`) and writes fresh `copy_shadow_equity` points. Makes shadow equity curves and the `copy_v3_auto_promote_min_shadow_fills` gate honest between leader events. |
| 2 | **Subscriber-count as a leader metric — inverted, as a crowding penalty.** They rank leaders UP by follower count; on-chain, crowding = alpha decay (our own CLAUDE.md honesty note). | Yes, cheap heuristic. | `modules/copy_trading/wallet_profitability.py`: add `crowding_penalty` — from `smart_money_wallet_events`, count distinct wallets buying the same token within N seconds after the leader; many fast followers → shave score. Feeds the existing decay-demotion knob. |
| 3 | **Signal-type taxonomy (`position` / `trade` / `realtime`) + 5-min position snapshots.** Their tradesync separates "state sync" from "event sync"; our monitor is event-only, so a missed tx = silently wrong leader state. | Yes, robustness. | `copy_engine.py` Solana monitor: add a low-frequency (5–10 min) leader **holdings snapshot** (Helius balances or `getTokenAccountsByOwner`) reconciled against our inferred leader position; divergence → log `[replay] reason=leader_state_drift` and pause mirroring that leader. Budget-friendly at 4 leaders. |
| 4 | **Challenge/variant evaluation** (same strategy, parameter variants, identical live MTM scoring) | Maybe, later. | `modules/param_tuner/` acceptance: run proposal-vs-baseline as parallel shadow variants scored by the same MTM harness — directly answers the documented "counterfactual rewards = overfit risk" caveat (out-of-sample by construction). |
| 5 | Agent-readable `SKILL.md` interface docs per module | Cosmetic. | Optional: a machine-readable per-module capability manifest under `docs/`; low value, we already have per-module CLAUDE.md. |

### C.4 Explicitly NOT worth porting

- **LLM-in-the-loop trade decisions / multi-agent trading arena.** Their own
  technical report shows most LLM agents are unprofitable with weak risk
  management. ClaudeDex's AI module already confines LLM spend behind a
  budget gate and a confirmation-filter design; do not expand LLM authority.
- **1:1 copy mirroring** — regression vs fractional-Kelly caps (see C.2).
- **The platform layer itself** (FastAPI signal marketplace, points/rewards,
  broker OAuth): solves distribution for a public product, not P&L for a
  single-operator bot. Pure surface area.
- **Alpha Vantage/yfinance equities data, Polymarket paper mode** — we already
  have a POLYMARKET module and a market-data warehouse; nothing new.

### C.5 License

MIT — fully compatible; ClaudeDex may copy code or ideas with attribution
preserved in copied files. Nothing in this review recommends verbatim code
copying anyway (the valuable parts are design patterns).

---

## Evidence index

- `logs/copy_trading/copy_trading.log` — 0 discovery lines in 9,223 lines
  (2026-06-15→07-05); 3,386 Helius rate-limit lines; `:28` 429 at t+0.2 s.
- `logs/smart_money/smart_money.log:22418` — smart_money scoring 497 EVM wallets.
- `monitoring/enhanced_dashboard.py:15156,15578,15621,15672,15707` — discover
  cascade, snapshot sampling flaw, local fallback.
- `modules/copy_trading/wallet_discovery.py:163-222,267,358,419,477` — v2
  source defects (pair-address bug, GMGN, operator-target seeding).
- `modules/copy_trading/discovery_v3.py:475-592` — v3 sweep (never run);
  `copy_engine.py:1352,1772,2091` — the off-by-default gate.
- Screenshots: `copytrading-wallets` (4 tracked), `copytrading-settings`
  ("ETHERSCAN_API_KEY is not currently set. EVM wallet monitoring is disabled").
