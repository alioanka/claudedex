# Wave-F6 Final Report — external-audit adjudication + repair, PM gate, deploy instructions

Date: 2026-07-10. Branch: `claude/friendly-ramanujan-nMWNv`.
Wave range: `7ba25c8` (exclusive) → HEAD. Five fixer merges (security RBAC, core safety,
solana/sniper, copy/rate-limit, advisor/futures/polymarket/arb) + migrations 147-150 + the
PM gate commits below.

This wave adjudicated an external GPT-5.6 audit (`docs/agents/wave-f6/06_gpt56_adjudication.md`;
the audit reviewed a stale checkout ~40 F5 commits behind this branch — every finding was
re-verified against current code before action). Root-cause evidence: `docs/agents/wave-f6/01..05`.

---

## PM verification gate (this document's authority)

1. **Migration audit 147-150** — PASS. Numbers unique, no collision with 137-146. Idempotent:
   147/148 `ON CONFLICT DO NOTHING`; 149 `ON CONFLICT DO UPDATE SET value='true'` for the two
   intended activations + conditional `UPDATE ... WHERE value IN (old)` then `INSERT ON CONFLICT
   DO NOTHING` for cadence seeds (operator overrides preserved); 150 Part A a re-runnable
   `NOT COALESCE(excluded)`-guarded jsonb tag, Part B conditional UPDATE + INSERT ON CONFLICT.
   Transactions: 147/149 explicit BEGIN/COMMIT (balanced); 148/150 single-purpose statements
   wrapped atomically by the runner (`scripts/migrate_database.py:168` `async with conn.transaction()`).
   **No live/paid flag flipped to true** — grep of every seed confirms no `shadow_mode` /
   `live_execution_enabled` / `dry_run` / autopilot / auto_apply flip. The ONLY activations are
   `copy_auto_discovery_enabled` + `copy_shadow_sim_enabled` (mig 149, both non-trading:
   read-only discovery writes candidates for operator approval; shadow sim writes
   `is_simulated=true` rows). Neither can place an order.

2. **Compile + duplicate-method AST sweep** — PASS. Every `.py` in `git diff --name-only
   7ba25c8..HEAD` compiles (the two `py_compile` failures are the intentional deletions —
   root `main_dex.py` and `scalping copy.py`). AST child-scope duplicate-def scan (setter/
   overload/register-aware) clean across all changed files, including the multi-touched
   hotspots `monitoring/enhanced_dashboard.py`, `config/pool_engine.py`, `core/engine.py`,
   `core/risk_manager.py`, and `main_futures.py` / `main_solana.py` / `main_ai.py`.

3. **Security lock-out check** — PASS with a documented operator caveat.
   - **Account role**: the sole account-bootstrap path (`scripts/init_auth.py::_ensure_admin`)
     creates username `admin` with role `admin`. Migration 001 deliberately seeds NO default
     admin. `AuthService.create_user` and the admin-panel `api_create_user` both default NEW
     accounts to `viewer`. So the operator's primary/bootstrap account is `admin` (passes every
     gate). **Residual risk (low)**: if the operator's day-to-day login was manually created as
     a `viewer`/`operator`, the new write-floor changes behavior. Verify + fix SQL below.
   - **Auth self-service never bricked**: `/login`, `/api/auth/login`, `/api/auth/logout` are in
     the middleware `public_routes`; `/api/auth/change-password` + `/api/auth/logout` are in
     `MUTATION_ALLOWED_ANY_ROLE`, so the VIEWER write-floor never blocks password change or
     logout. Confirmed in `auth/middleware.py`.
   - **CORS**: `enhanced_dashboard.py` builds the aiohttp-cors allowlist from
     `DASHBOARD_CORS_ORIGINS` (defaults `http://localhost:8080`), filters out `*`, never
     combines wildcard+credentials. Same-origin requests carry no `Origin` mismatch and are
     unaffected. Operator already has `DASHBOARD_CORS_ORIGINS` set.

4. **DRY_RUN-aware contract gate** — DONE (committed `e8a543a`). `core/engine.py`
   `_final_safety_checks` step 6: warn-only in DRY_RUN (paper data keeps flowing), fail-closed
   in LIVE; `trading.block_unverified_contracts` (mig 147 default true) overrides to warn-only
   in both modes. mig 147 comment + description updated to match.

5. **Cross-agent seams** — PASS. (a) `start_killswitch_poller()` added once each to
   futures/solana/ai mains; the poller is a per-process idempotent singleton
   (`core/dry_run.py:184`) — no double-start, imports resolve. (b) pool_engine Helius dedup
   (`_extract_api_key`: api_key column, else `api-key` from URL) and copy's
   `discovery_v3._helius_key_count` (`api_key` else `_current_helius_url_key` regex on
   `api-key=`) fingerprint by the SAME account key — consistent. (c) the global VIEWER
   write-floor is middleware, so it catches the ungated POST routes in `module_routes.py`
   (`/api/modules/{m}/start|enable|disable|pause|resume`, `/api/modules/reallocate`) and
   `test_runner_routes.py` — they surface as UNGATED in the route-authz self-test (no
   admin/operator distinction) but are viewer-protected exactly as designed. (d) mig 147 seeds
   `('trading','block_unverified_contracts')`; `core/engine.py` reads
   `config.get('trading',{}).get('block_unverified_contracts', True)` — key name matches, same
   pattern as other working `trading.*` reads.

6. **Documented follow-ups (recorded, NOT fixed)** — see "Known residual risks" below.

7. **Root CLAUDE.md** — Wave-F6 addendum added (2026-07-10).

8. **This report.**

---

## Issue-by-issue closure

### 1. Rate-limiting / Helius saturation (90,365 events in ~2 days)
- **Root cause** (report 01, adjudication #12): ONE real Helius free-tier account registered
  under FIVE endpoint names (env key, secrets key, DB rows SOL #1..#3 / Helius #5 — all with
  the SAME key embedded in the URL, `api_key` column NULL). Round-robin "rotation" spread one
  ~10 rps quota across five names — no extra capacity, and the starved-fallback path fired on
  every name. copy's daily budget counted the five names as five accounts and multiplied its
  Helius allowance 5× against one quota. `copy_helius_rps=8` alone nearly saturated the ceiling.
- **Fixed**: `1844c12` (pool_engine `_dedup_endpoints_by_key` + distinct-key startup log),
  `c6bd0eb` (copy `_helius_key_count` counts by distinct account key), `5a5874c` (copy rps 8→2,
  poll 15→30s, live limiter reconfigure), mig 149 (cadence seeds).
- **Operator must**: populate `HELIUS_API_KEY_2`, `HELIUS_API_KEY_3`, `HELIUS_API_KEY_4` +
  `BIRDEYE_API_KEY` in `/credentials` (or `.env`), restart, and confirm the startup log line
  reads `HELIUS_API: N distinct key(s)` with **N > 1**. Until then the fix is honesty + load
  reduction only — it does NOT create capacity.
- **Watch (48h)**: pool-side rate-limit event count should drop sharply; `HELIUS_API: N
  distinct key(s)` in startup logs; `[discovery-v3]` + `CopyShadowSim` log lines now appear.

### 2. Arbitrage profitability / phantom PnL
- **Root cause** (report 02, adjudication #11): 100% of arbitrage PnL (+1085 7D / +3250
  all-time, 100% win, Sharpe 3.00) was the triangular DRY_RUN branch — gross quote-time spread,
  zero gas/fees, ~2400× notional inflation (1 CRV mislabeled as 1 ETH), on a path that cannot
  execute live (MB-05 atomic-receiver guard).
- **Fixed**: `cc87b95` (triangular DRY fills book net-of-gas + self-tag `metadata.excluded=true`
  at insert), mig 150 Part A (`4f35676`, back-tag pre-fix history; rows KEPT for audit).
- **Operator must**: nothing. **Honest expectation: PARK IT.** Executable legs are V2-only vs a
  ~65bps cost floor against 1-30bps real divergence — expected edge ≈ 0 until a V3 receiver leg
  ships. Keep arbitrage live OFF (already default).
- **Watch (48h)**: control-center arbitrage PnL should reset toward 0 (fabricated wins excluded);
  the hourly `SPREAD DISTRIBUTION` log is the only honest edge signal.

### 3. Security cluster (P0 — adjudication #2/#4/#5)
- **Root cause**: ~45 mutating dashboard routes carried auth-only (any role could trade/configure/
  CRUD RPC endpoints incl. key-bearing URLs); wildcard CORS with credentials let any site read
  authenticated responses with the victim cookie; public bind on `0.0.0.0:8080` over plain HTTP.
- **Fixed**: `0796806` (`require_operator` gate + RBAC audit logs + VIEWER write-floor +
  route-authz self-test), `d2b1de5` (RBAC gates on all mutating enhanced_dashboard routes),
  `8b8e323` (admin-gate RPC pool CRUD), `c6b50fc` (CORS allowlist from `DASHBOARD_CORS_ORIGINS`),
  `f85ada5` (401→/login redirect), `07da5be`/`5b2af07` (docs: RBAC matrix + public-exposure
  warning).
- **Operator must**: **firewall / VPN / TLS-reverse-proxy port 8080** — code cannot fix a public
  bind decision. Confirm your login account is `admin` (verify SQL below).
- **Watch (48h)**: startup log `Route-authz self-test: ...`; `RBAC allow/deny actor=... ` lines
  on privileged actions; no legitimate operator action returns 403.

### 4. Copy discovery dead for weeks
- **Root cause** (report 01/04, adjudication #17): the Wave-F5 mig 141 conditional flip
  (`WHERE value='false'`) never matched the stored value; discovery + shadow-sim returned
  immediately for three weeks (operator: "discovery finds only my wallet, 0 copies").
- **Fixed**: `8e6545a` (mig 149 UNCONDITIONAL enable — both non-trading), `07da5be` docs.
- **Operator must**: nothing to enable; approve discovered leaders explicitly before any live copy.
- **Watch (48h)**: `copy_discovered_wallets` / `copy_leader_candidates` populate beyond your own
  wallet; `CopyShadowSim` paper rows accrue.

### 5. Core safety (adjudication #6/#15)
- **Root cause**: `RiskManager.validate_trade` + `calculate_position_size` called
  `wallet_manager.get_available_balance()` — a method `WalletSecurityManager` does not have →
  AttributeError → permanent veto on EVERY validate_trade-gated live entry (DEX, solana, sniper,
  arb, copy, ...). Separately, the DEX contract-verification gate was warn-only in all modes; and
  DEX `successful_trades` was incremented at OPEN and again at profitable close (`successful >
  total`, corrupting win_rate).
- **Fixed**: risk_manager switched to the fail-soft `self.get_available_balance()`
  (wallet→portfolio→0 fallback); `2fbd9d8`+`e8a543a` (DRY_RUN-aware contract gate + mig 147);
  `536b9b2` (stop double-counting wins at entry).
- **Operator must**: nothing.
- **Watch (48h)**: DEX/solana/sniper `validate_trade` no longer logs "Validation error" on every
  entry; DEX win_rate no longer shows successful > total.

### 6. Solana / Sniper (report 03)
- **Fixed**: `5e365c4` (solana entry-price cross-source corroboration), `63ebdd4` (Drift driftpy
  env KeyError + rate-limit RM-block WARN), `8c6497b`/`8c6545a` sniper BSR fallback mode (mig 148,
  default `skip_gate` — a down Birdeye BSR source was rejecting 99.5% of candidates; now skips
  ONLY the BSR gate, all other safety gates run), stale BSR comment corrected (`dda9938`).
- **Operator must**: provide real Helius keys (item 1) — sniper detects nothing without them.
- **Honest expectation**: sniper needs weeks of DRY_RUN with real keys before any live judgment.

### 7. Futures geometry (report 03)
- **Fixed**: mig 150 Part B `max_hold_minutes` 240→480 (`4f35676`) — rr=1.0 (F5 mig 139) still
  produced 0 take_profit exits in 3 days (17/37 closes were time_limit). ONE geometry lever.
- **Honest expectation**: needs 2+ weeks fresh DRY_RUN before judging; do NOT stack more geometry
  changes inside that window.

### 8. Advisor (report 04)
- **Fixed**: `c66d14c` — Kronos forecaster synthesises OHLC from close-only klines before
  inference (was failing on close-only sources).

### GPT-5.6 adjudication outcome
Of 20 findings: several FALSE_POSITIVE_STALE (dirs "missing", arb gate, sniper poller — all
already present on this branch); the genuinely-new P0/P1 bugs (RBAC gap, wildcard CORS,
kill-switch blind spots in futures/solana/ai, `validate_trade` permanent veto, DEX win
double-count, dead broken files) were ALL fixed this wave. Roadmap items (immutable release
identity, typed pre-trade reservation, execution-gateway adoption, data-quality SLOs) recorded,
not built. Full ruling: `docs/agents/wave-f6/06_gpt56_adjudication.md`.

---

## OPERATOR ACTION LIST (do exactly this, nothing else)

1. **Deploy**: `git pull && docker compose up -d --build`. The entrypoint runs
   `scripts/migrate_database.py`, which applies migrations 147-150 (idempotent — safe to re-run).
2. **Multi-key Helius/Birdeye**: put `HELIUS_API_KEY_2`, `HELIUS_API_KEY_3`, `HELIUS_API_KEY_4`
   and `BIRDEYE_API_KEY` into `/credentials` (dashboard Secure Credentials) or `.env`, then
   restart. **Confirm** the startup log shows `HELIUS_API: N distinct key(s)` with **N > 1**.
   Setup guide: `docs/RPC_API_KEYS_GUIDE.md`. This is the ONLY thing that creates real rate-limit
   headroom.
3. **Firewall port 8080** (VPN / security group / TLS reverse proxy). The dashboard binds
   `0.0.0.0:8080` over plain HTTP by design; auth + CSRF + RBAC protect it, but do not expose the
   control plane to the open internet.
4. **Verify your dashboard role is `admin`** (avoids the viewer write-floor):
   ```sql
   SELECT username, role, is_active FROM users ORDER BY id;
   ```
   If the account you log in with is NOT `admin`, grant it (replace `<username>`):
   ```sql
   UPDATE users SET role = 'admin', updated_at = NOW() WHERE username = '<username>';
   ```
   If you have no admin at all, re-run the bootstrap (prints a one-time password):
   `python scripts/init_auth.py` (uses `DATABASE_URL`).
5. **Apply nothing else.** No manual SQL beyond the above; no flag flips.

---

## 48-hour watch list

- `HELIUS_API: N distinct key(s)` — N must be > 1 after step 2; pool rate-limit events drop.
- `[discovery-v3]` + `CopyShadowSim` log lines appear (discovery revived); `copy_discovered_wallets`
  grows beyond your own wallet.
- Control-center arbitrage PnL resets toward 0 (phantom wins excluded); hourly `SPREAD
  DISTRIBUTION` is the honest signal.
- `Route-authz self-test` + `RBAC allow/deny` lines present; no legitimate operator action 403s.
- `validate_trade` stops logging "Validation error" on every gated entry.
- Futures: watch for the FIRST take_profit exit under the 480-min hold; do not re-tune for 2+ weeks.
- Kill-switch: `touch logs/.killswitch` should now halt futures/solana/ai live writes too (test
  once in DRY_RUN).

---

## Honest expectations

- **Arbitrage — park it.** V2-only executable legs vs a ~65bps cost floor; expected edge ≈ 0
  until a V3 receiver leg ships. Live stays OFF.
- **Futures & sniper — weeks of DRY_RUN.** Futures geometry needs 2+ weeks fresh data under the
  new hold window; sniper needs real Helius keys before it detects anything, then weeks of paper.
- **DEX still idle** — the 0-entries state is collector-side, NOT fixed this wave (see below).
- Everything remains DRY_RUN / shadow-first. No module's live flag was flipped.

---

## Known residual risks / follow-ups (item 6 — NOT fixed this wave)

1. **DEX EVM pair discovery is collector-side.** `data/collectors/dexscreener.py` discovery
   strategies scan GLOBAL boost/profile lists (Solana-dominated) and mostly return established
   pools that fail the 24h max-age filter, so EVM chains discover zero new pairs — this, not a
   chain_weights misconfig, is why DEX shows 0 entries. Fix = a chain-scoped EVM discovery source
   in the collector. Until then DEX stays idle even with all gates open.
2. **Solana position-monitor cadence.** `modules/solana_trading/` position monitor polls at
   ~40s; it should back off to 90-120s to cut Helius load. Outside copy's scope this wave.
3. **Drift perp RiskManager gate.** The Drift perp path reuses the spot-token liquidity gate;
   it needs a perp-appropriate check (open-interest / perp depth), not spot liquidity.
4. **Viewer-default account creation.** `create_user`/`api_create_user` default `role='viewer'`;
   this is safe with the write-floor but an admin creating operators via the UI must set the role
   explicitly.
5. **Roadmap (adjudication #20)**: immutable release identity (git SHA + image digest at startup),
   typed pre-trade risk reservation replacing the mixed-unit `amount` param, wiring modules onto
   the `execution_gateway` library (mig 127), data-quality SLOs + DEGRADED strategy state.
