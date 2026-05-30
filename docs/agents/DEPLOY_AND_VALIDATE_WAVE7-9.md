# Deploy & Validate — Wave 7-9 campaign (operator runbook)

Date: 2026-05-26 · Branch: `claude/create-expert-agents-JFSF5` · Audience: VPS operator.

This is the single consolidated runbook for deploying the Wave 7 + Wave 8 +
Wave 9 campaign to your VPS and validating it. It assumes you already run the
stack via `docker compose` with the `trading-postgres` container and the
standalone dashboard on `:8080`.

> **DRY_RUN STAYS TRUE.** Nothing in this runbook enables LIVE trading. Do not
> flip `DRY_RUN`, do not re-enable `SNIPER_SAFETY_CHECK_ENABLED`, and never run
> `docker ... prune --volumes`. The campaign's job is to make your DRY_RUN data
> *honest and complete* for all 7 modules + dashboard — go live only after you
> have reviewed that data and explicitly decide to.

Source verification: `docs/agents/PM_FINAL_WAVE7.md`, `PM_FINAL_WAVE8.md`,
`PM_FINAL_WAVE9.md`, the per-module `CLAUDE.md` files, and `ml/CLAUDE.md`.

---

## 1. One-time deploy steps (in order)

### Step 1 — Pull the branch on the VPS

```bash
cd /path/to/claudedex
git fetch origin
git checkout claude/create-expert-agents-JFSF5
git pull --ff-only origin claude/create-expert-agents-JFSF5
```

Confirm HEAD is the campaign tip and the tree is clean before restarting
anything. (There is a stale `git stash` on this branch — leave it alone; it is
not part of the deploy.)

### Step 2 — Run migration `032_add_dex_runtime_stats.sql`

Wave 8 added the `dex_runtime_stats` heartbeat table (single row, id=1) so the
dashboard can tell a LIVE-but-idle DEX engine apart from a dead one. The
migration is idempotent (`CREATE TABLE IF NOT EXISTS` + `INSERT ... ON CONFLICT
(id) DO NOTHING`), so it is safe to re-run.

**Preferred — the migration runner** (tracks applied versions in the
`migrations` table, applies the numbered series in order, skips already-applied
files):

```bash
python scripts/migrate_database.py
```

**Manual fallback** — mirror the established `docker exec trading-postgres`
psql pattern used elsewhere in the docs. The DB user is read from the same
Docker secret the app uses; the database is `tradingbot`:

```bash
docker exec -i trading-postgres \
  psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot \
  < migrations/032_add_dex_runtime_stats.sql
```

Verify the table exists:

```bash
docker exec trading-postgres \
  psql -U "$(docker exec trading-postgres cat /run/secrets/db_user)" -d tradingbot \
  -tAc "SELECT id, updated_at FROM dex_runtime_stats;"
```

> Without 032 the heartbeat UPSERT fails fail-soft (harmless) and
> `/full-dashboard` keeps showing DEX "ENABLED (no health)" when idle. The rest
> of Wave 8 is unaffected.

### Step 3 — Restart subprocesses (which restart picks up which fixes)

The dashboard runs independently on `:8080` and does NOT require any trading
subprocess to be running. The trading modules are launched as subprocesses by
`main.py` per their `*_MODULE_ENABLED=true` flags. Restart order does not
matter, but **all three of the following restarts are required** to pick up the
campaign:

| Restart | Picks up |
|---|---|
| **ALL trading modules** (restart `main.py` / the `trading-bot` container) | Wave 8/9 engine fixes live in the shared `core/engine.py` (DEX risk-failure penalty, ML ensemble + honest `heuristic_fallback`, `_load_state` open-position restore, honest safety stubs, ensemble feature-contract). DEX heartbeat (`main_dex.py::_status_reporter`) starts writing `dex_runtime_stats`. |
| **DASHBOARD process specifically** (`:8080`) | Wave 7 dashboard fixes live HERE: tz-comparison crash fix, all-7-module charts via `_unified_closed_trades`, Funding/Accounts panel, close-button CSRF header (`window.withCsrfHeaders`), close-button routing, analytics real-column mapping, RESTART OVER-CAP banner. Plus Wave 8 dashboard-side 150s DEX heartbeat freshness fallback. |
| **AI subprocess specifically** | The AI module's keys now resolve via `get_async` (Wave 6 fix carried through), but the subprocess has been **stalled / silent since ~late January** because it read keys before `db_pool` init. It needs an explicit restart to actually start ticking. After restart, watch for the per-cycle liveness line (see §2, issue group "AI"). |

If you run the whole stack in one container, a single `docker compose restart
trading-bot` (plus a dashboard restart if it is a separate service) covers all
three. If modules run as separate processes, restart each. Confirm each
subprocess re-spawned and is writing to its own `logs/<module>/` directory.

---

## 2. Per-module DRY_RUN validation checklist (the 22 original issues)

For each issue, the one-line confirmation you can check on the dashboard or in
the logs after the restarts above. All checks are DRY_RUN-safe.

### Dashboard / cross-module

| # | Issue | Confirm it's fixed |
|---|---|---|
| 1 | `/full-dashboard` inaccurate, "ENABLED (no health)", empty charts | `/full-dashboard` shows all 7 modules WITH health (DEX no longer stuck at "no health" once 032 + restart land) and charts render. |
| 3 | Main dashboard charts only DEX/Futures/Solana | `/` and `/full-dashboard` performance charts now show ALL 7 modules (unified closed-trades set). |
| 18 | `dashboard_errors.log` naive/aware datetime crash | `tail logs/dashboard/errors.log` — no `can't subtract offset-naive and offset-aware datetimes` errors. |
| 2 | `/analytics` fake PnL, no entry/exit, identical size/PnL | `/analytics` shows real per-trade entry/exit/size/PnL columns; Sniper rows are no longer all identical `$0.84`. |
| 7 | `/futures` poor perf — strategy enhancement | Futures DRY_RUN data now RENDERS on `/analytics` and the all-7 charts (was silently empty from a column mismatch). See §4(b) for the new RM gates. |
| 15 | Surface wallet/exchange per module incl AI+Copy | The Funding / Accounts panel (`/api/funding/accounts`) shows each module's wallet/exchange — see the map in §3. |
| 6 | ALL modules' close buttons must work | Close buttons on DEX/Solana/Futures/Sniper/Copy positions pages succeed (no 403). Root cause was a missing `X-CSRF-Token` header — now added globally. |

### ARBITRAGE

| # | Issue | Confirm it's fixed |
|---|---|---|
| 14 | `/arbitrage` dashboard dead since Feb 7 | `/arbitrage/dashboard` shows fresh activity again; the engine no longer crashes on init. |
| 16 | ARB millions of `_realized_slip_refreshed_at` AttributeError | `grep _realized_slip_refreshed_at logs/arbitrage/errors.log` returns nothing new (attrs now initialized in `EVMArbitrageEngine.__init__`). |

### SOLANA

| # | Issue | Confirm it's fixed |
|---|---|---|
| 8 | `/solana` Active Positions empty despite 7 open | `/solana/dashboard` Active Positions shows the open (DRY_RUN sim) positions; they are no longer discarded as phantom on startup reconcile. |
| 9 | `/solana` close "CSRF token missing or invalid" | Solana close button succeeds (CSRF header now sent). |
| 10 | `/solana` unrealistic exit price/PnL (+495424%, ORCA $6788) | `/solana/trades` exit prices are realistic (DexScreener price now matched by `baseToken.address == mint`). |
| 11 | Solana Drift enabled, zero activity | In DRY_RUN, `logs/solana_trading/` shows `🎯 Drift signal: ...` then `✅ [DRY_RUN] Drift ... opened` lines (funding synthesized when chain data absent). LIVE Drift needs extra setup — see §4(c). |
| 22 | Solana `Error getting wallet balance` + RAPID CRASH spam | `logs/solana_trading/` no longer floods balance errors / RAPID CRASH (balance throttled + cached; crash guarded by `current_price > 0`). |

### SNIPER

| # | Issue | Confirm it's fixed |
|---|---|---|
| 2/12 | `/sniper` 93% WR, 400+ open, identical `$0.84` PnL | `/sniper/dashboard` win rate is no longer a flat 93% with identical $0.84 PnL; open positions are capped at `max_active_positions=500`. (Modeled exits are a population MODEL for data realism, not a backtest — treat WR as illustrative.) |
| 13 | `/sniper/trades` empty | `/sniper/trades` lists trades (engine writes `sniper_trades`; page reads it). |
| 21 | `logs/sniper` growing | `logs/sniper/` stops ballooning (per-trade/per-candidate lines moved to DEBUG; rotation tightened). A stale `logs/sniper_module/` dir is safe to `rm -rf`. |

### DEX

| # | Issue | Confirm it's fixed |
|---|---|---|
| 4 | `/dex/dashboard` stuck GRAIL position; ML review | `/dex/dashboard` position price/PnL refreshes (DB-first `price_refresh_loop`); stale flag after 5 fails/30 min. ML review actioned in Wave 8 (see §4(a)). |
| 5 | `/dex/positions` close 503 | DEX close button succeeds; backend writes `logs/.close_dex_<trades.id>` and the engine poller matches it. |

### FUTURES

| # | Issue | Confirm it's fixed |
|---|---|---|
| 7 | (strategy) — see cross-module table | Validate the new edge/throttle/regime gates in DRY_RUN — see §4(b). |
| 19 | Futures `TELEGRAM_BOT_TOKEN not set` | `logs/futures_trading/` no longer logs "TELEGRAM_BOT_TOKEN not set" (token pre-warmed via `get_async`). Re-check AI + ARB too (§4 note). |

### COPY

| # | Issue | Confirm it's fixed |
|---|---|---|
| 17 | Copy HELIUS/ETHERSCAN "NOT SET"; "Solana RPC rate limited" | `logs/copy_trading/` shows `ETHERSCAN_API_KEY: SET` / `HELIUS_API_KEY: SET` (if you stored them) and prefers the Helius RPC over the rate-limited public endpoint. |

### ORCHESTRATOR

| # | Issue | Confirm it's fixed |
|---|---|---|
| 20 | Orchestrator scored=7 but only 2 `to_live` recs | The orchestrator recommendations surface ONE current row per module for all 7 (under-traded modules show a `not_ready` 'hold' instead of going silent). |

---

## 3. Funding / Accounts map (what to fund for live mode)

This is the wallet/exchange identity per module, drawn from the per-module
`CLAUDE.md` identity sections. Only public addresses / masked fingerprints are
ever surfaced — private keys and keypairs are never exposed. **This is the
authoritative list of what you fund before you ever consider LIVE.**

| Module | Account / wallet | Secret keys | Notes |
|---|---|---|---|
| **ARBITRAGE** | ONE shared EOA across ETH / ARB / Base | `PRIVATE_KEY` (address DERIVED from key; stored `WALLET_ADDRESS` is ignored) | Fund the SAME address with native gas on each of the three chains. |
| **DEX** | Single EVM wallet | `WALLET_ADDRESS` + `PRIVATE_KEY` (address derived if `WALLET_ADDRESS` unset) | EVM-only (no SOL leg). |
| **SOLANA** | Solana keypair | `SOLANA_MODULE_WALLET` + `SOLANA_MODULE_PRIVATE_KEY` | **SHARED with SNIPER** — same keypair. |
| **SNIPER** | Same Solana keypair as SOLANA (primary); separate EVM wallet for EVM snipes | Solana: `SOLANA_MODULE_PRIVATE_KEY` / `SOLANA_MODULE_WALLET`. EVM: `PRIVATE_KEY` (fallback `EVM_PRIVATE_KEY`) / `WALLET_ADDRESS` (fallback `EVM_WALLET_ADDRESS`) | One Solana wallet funds BOTH sniper and solana_trading — plan position sizing across both. |
| **FUTURES** | Binance OR Bybit exchange account (selected by `futures_general.exchange`) | Mainnet `{BINANCE,BYBIT}_API_KEY` / `_API_SECRET`; testnet `{BINANCE,BYBIT}_TESTNET_API_KEY` / `_TESTNET_API_SECRET` | Testnet vs mainnet via `engine.testnet` (env `FUTURES_TESTNET` override → DRY_RUN=false defaults MAINNET → else DB value). No on-chain wallet. |
| **COPY** | OWN EVM execution wallet + OWN Solana execution wallet (the bot's broadcasting wallets) | EVM: `WALLET_ADDRESS` + signer key. Solana: `SOLANA_MODULE_WALLET` + keypair | DISTINCT from the leader `targets` it observes — leaders are watched, never controlled. |
| **AI** | No own wallet — DELEGATES execution to the FUTURES exchange account | (none of its own) | AI trades land on the FUTURES exchange account via the canonical futures executor. |
| **ORCHESTRATOR** | No wallet — advisory only | (none) | Meta-module: writes recommendation rows, signs nothing, holds no positions. Funding N/A. |

---

## 4. Operator-driven follow-ups (NOT done — need your VPS / data)

These are open items the campaign could not complete in a no-DB/no-live
environment. None of them block correct DRY_RUN data collection.

### (a) ML ensemble activation — gated on a training pipeline

The DEX ensemble runs in **honest `heuristic_fallback`** today because no
trained artifacts exist on disk, and `scripts/retrain_models.py` does NOT yet
train the ensemble (it trains 3 separate models to `ml/models/`, while the
engine reads `models/`). Until you build a DB → canonical-82-feature labeled
training pipeline, DEX keeps using the heuristic fallback (this is correct and
fail-soft, not broken).

- Full activation runbook + KNOWN GAP: **`ml/CLAUDE.md`**.
- Confirm provenance any time: DEX log line reads `ML[heuristic_fallback]`
  today; it will read `ML[ensemble]` (and `metadata.ml_source == 'ensemble'`)
  once trained artifacts ship to `models/` and the DEX module is restarted.
- Keep DRY_RUN = true while validating the ensemble signal.

### (b) Futures strategy enhancements are THEORY — validate in DRY_RUN first

The Wave 7 Futures fixes (MACD signal-line fix + the FUT-RM-19/20/21 gates) are
theoretically sound bleed-reduction but were NOT backtested in this environment.
Before live, watch DRY_RUN and confirm:

- **FUT-RM-19** (fee+funding minimum-edge gate, `min_net_edge_pct` default
  0.30%) — new rejection log lines appear for trades that can't clear
  round-trip costs.
- **FUT-RM-20** (one-entry-per-candle throttle) — repeated same-candle entries
  stop.
- **FUT-RM-21** (regime gate, `block_counter_trend_entries` default ON) —
  counter-trend entries are blocked in clear trends.
- Net check: trade frequency should DROP without starving legitimate signals.
  If signals are over-suppressed, tune the keys in the Futures Wave-7 config
  cheat-sheet (`modules/futures_trading/CLAUDE.md`) rather than going live.

### (c) Drift LIVE prerequisites (Solana)

Drift stays toggle-OFF by default and runs synthesized activity in DRY_RUN. For
LIVE Drift you additionally need: `pip install driftpy` AND USDC collateral
deposited to the Drift sub-account owned by the `SOLANA_MODULE_PRIVATE_KEY`
wallet (same wallet as Jupiter spot). With no collateral the leverage guard
fails closed and refuses every entry.

### (d) DEX live-but-idle health = cosmetic

After 032 + restart, an idle DEX engine reports health via the
`dex_runtime_stats` heartbeat (≤150s freshness). If you still briefly see "no
health" right after restart, it is cosmetic — it clears on the next ~60s
heartbeat write.

---

## 5. Safety reminders (do not skip)

- **DRY_RUN stays TRUE** until you explicitly decide to go live, module by
  module, after reviewing the now-honest DRY_RUN data.
- **Do NOT re-enable `SNIPER_SAFETY_CHECK_ENABLED`** as part of this deploy.
- **Never run `docker ... prune --volumes`** — that destroys the Postgres data
  volume (`postgres-data`) and your trading history.
- The kill switch (`logs/.killswitch`), per-module pause flags
  (`logs/.pause_<module>`), and the dashboard emergency-stop button
  (`/api/bot/emergency-exit`) remain your live safety primitives — verify the
  emergency-stop button still works after the dashboard restart.
- No private keys / keypairs / full API keys are surfaced on any dashboard or
  log surface — only public addresses and masked `****last4` fingerprints. If
  you ever see a full secret in a log, stop and investigate.
