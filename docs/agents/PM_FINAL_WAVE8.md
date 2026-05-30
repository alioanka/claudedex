# PM FINAL — Wave 8 verification gate (2026-05-26)

Branch `claude/create-expert-agents-JFSF5`. Verification method: code read +
`python -m py_compile` (no docker / DB / live bot; Postgres on VPS only).
DRY_RUN remained TRUE everywhere; no LIVE flip, no force-push, no amend.
Closes Wave-7 punch-list item **B** (the DEX entry ML/risk dead-weight) plus the
P2 **D** (DEX live-but-idle "no health").

## VERDICT: SHIP (DRY_RUN data collection)

All five Wave-8 commits are correct and SAFE. No PM code change was required —
the highest-priority concern (DEX entry starvation) does NOT occur, for the
reason proven below. The operator will collect CORRECT DEX DRY_RUN data after
this wave once migration 032 is applied.

## KEY DETERMINATION — does Wave 8 starve DEX entries? NO.

The DEFECT-1 hard-reject (`return 0.0` when `risk_score` is falsy in
`_calculate_opportunity_score`, engine.py:3742-3748) is reached **almost never**,
because the real fail-soft lives one layer down:

- `risk_score` becomes `None` ONLY at engine.py:902-904, and ONLY when the
  `asyncio.gather(..., return_exceptions=True)` element for
  `risk_manager.analyze_token(...)` is an `Exception`.
- `core/risk_manager.py::analyze_token` (L439-512) wraps its whole body in
  `try/except` and **returns a valid worst-case `RiskScore(all-1.0,
  confidence=0.1)` on ANY failure** (L500-512). It also converts each per-sub-task
  exception to `1.0` (max risk) via `return_exceptions=True` (L472-474). So an
  external-API hiccup on many tokens yields a high-risk RiskScore, NOT a raised
  exception.
- Therefore, in normal operation an un-assessable token flows through the normal
  `if risk_score and hasattr(...)` branch with `overall_risk ≈ 1.0`, so
  `risk_component ≈ 0.0` contributes 0 to the score — a HEAVY PENALTY — but the
  token can still score on volume / liquidity / price / age. This is exactly the
  "worst-case penalty, not entry kill-switch" behaviour the gate brief asked for.
  It already exists; the hard-reject only fires if `analyze_token` raises BEFORE
  its own try-block (a coroutine-scheduling pathology, effectively never).

PM decision: **leave the hard-reject as-is.** It is a conservative belt-and-
suspenders guard for the pathological case and cannot starve DEX in practice.
Converting it to a soft penalty would be redundant (the soft penalty is already
provided by `analyze_token`) and would remove the only guard against the
raise-before-try case. Documented rather than changed.

## Per-defect results

| # | Defect | Commit | Status | Evidence |
|---|---|---|---|---|
| 1 | Risk-failure was REWARDED in scorer | 031a1bc | VERIFIED | `_calculate_opportunity_score` no longer drops the 0.20 risk weight from the denominator on failure; treats unknown risk as worst-case and `return 0.0` (engine.py:3742-3748). Worst-case construction is VALID: `RiskScore(liquidity_risk=1.0…market_risk=1.0, confidence=0.0)` passes all 8 required fields (risk_manager.py:39-46); `overall_risk` is a read-only `@property` (risk_manager.py:84-122) so it is NOT a ctor arg — the old `RiskScore(overall_risk=0.5)` would have raised TypeError; the new build does not. Does NOT starve DEX (see determination above). |
| 2 | ML fabricated; flat rug=0.2 silently passed gate | fd66a0e | VERIFIED | `_ml_predict_opportunity` (engine.py:808-865) consults `EnsemblePredictor.predict_decoupled` and uses it ONLY if `_is_trustworthy_ml_result` passes (no `error` key; not the all-0.5/conf≤0.1 untrained passthrough — engine.py:784-806). With no model artifacts the LIVE path is the heuristic fallback, labeled `ml_source='heuristic_fallback'` and surfaced in the entry log (engine.py:947-950). Fallback `rug_probability = max(0.25, min(0.6, 0.6 - 0.4*hs))`: NOT a fabricated 0.2, NOT so high it rejects everything. Tokens reaching this branch already cleared `min_score` (default 0.25), so `hs ≥ 0.25` → rug ≤ 0.5; the gate is strict `> 0.5` (engine.py:2939-2944), so a marginal token sits exactly at the boundary and PASSES, stronger tokens get lower rug. Composite `score` weights (engine.py:89-97) unchanged. |
| 3 | `_load_state` was a no-op | 1abef40 | VERIFIED | Restores `trades WHERE status='open' AND side='buy' AND chain IN _DEX_STATE_CHAINS` (engine.py:3000-3105). `_DEX_STATE_CHAINS` (engine.py:2995-2998) == `position_service._DEX_CHAINS` (position_service.py:33-36) exactly — no cross-module/orphan rows. Restored dict shape matches the live entry path (engine.py:1215-1233): `entry_price`/`amount` as `Decimal`, `entry_time` as `datetime` — so the monitoring arithmetic at engine.py:1715/1718 (`Decimal * Decimal`) and 1724 (`datetime` subtraction) is type-correct. `trade_id` = INTEGER `trades.id` (row['id']); the engine close path reads `position.get('trade_id')` and calls `db.update_trade(trade_id, …)` (engine.py:2164-2197), and `position_service._update_trade(int(trade_id))` targets the same `id` — coexistence confirmed, last-writer-wins converges. Fail-soft: method-level try/except (engine.py:3103-3105) and called inside `initialize()` (engine.py:391) — a bad load cannot crash startup. |
| 4 | DEX live-but-idle heartbeat | 6f4a9d7 | VERIFIED | Write interval ~60s (`_status_reporter` → `_persist_heartbeat` then `sleep(60)`, main_dex.py:935/976/980-1029); dashboard freshness `age <= 150` = 2.5× write interval (enhanced_dashboard.py). Age computed **entirely in SQL** (`EXTRACT(EPOCH FROM (NOW() - updated_at))::int`) over a `TIMESTAMPTZ` column — no Python naive/aware subtraction, so Wave-7 issue 18 is NOT reintroduced. Migration 032 idempotent (`CREATE TABLE IF NOT EXISTS` + `INSERT … ON CONFLICT (id) DO NOTHING`). No table collision: heartbeat = `dex_runtime_stats`, `_load_state` = `trades`. `_persist_heartbeat` fail-soft (debug log, swallowed). `self.wallet_address`/`self.is_dry_run` exist (main_dex.py:388-389; DRY_RUN defaults TRUE). |
| 5 | dex CLAUDE.md docs | 97cb33c | VERIFIED | Wave-8 section is accurate and honest (documents the hard-reject choice, the score-coupled fallback rug, the `_load_state` coexistence, and the Wave-9 follow-ups). No code touched. |

## Safety-invariant check results

| Invariant | Result |
|---|---|
| DRY_RUN defaults remain TRUE | PASS — main_dex.py:389 defaults true; no flip in diff |
| No gate LOOSENED (`max_rug_prob`/`min_opportunity_score`/`min_liquidity`/score weights) | PASS — `max_rug_prob=0.5` (engine.py:2939) unchanged; `min_opportunity_score` default 0.25 (engine.py:923) unchanged; `min_liquidity` (engine.py:2922) unchanged; composite weights (engine.py:89-97) unchanged. DEFECT 1 can only make the gate STRICTER; DEFECT 2 fallback rug (0.25-0.6) is STRICTER than the old flat 0.2 |
| Kill-switch / `safety_check_enabled` / `SNIPER_SAFETY_CHECK_ENABLED` | PASS — untouched in Wave-8 diff |
| No secrets surfaced | PASS — heartbeat snapshot carries only public `wallet_address` + counters |
| All changed `.py` `py_compile` clean | PASS — core/engine.py, modules/dex_trading/main_dex.py, monitoring/enhanced_dashboard.py (+ risk_manager.py, position_service.py) all compile |
| Migration 032 idempotent | PASS — `IF NOT EXISTS` + `ON CONFLICT DO NOTHING` |

## PM commits this gate

None. Verification only; no real gap found that warranted a code change.

## Operator action required

- Apply **migration 032** (`migrations/032_add_dex_runtime_stats.sql`) on the VPS
  before/with the next DEX subprocess + dashboard restart. Without it the
  heartbeat UPSERT fails (fail-soft, harmless) and `/full-dashboard` keeps showing
  DEX "ENABLED (no health)" when idle — the rest of Wave 8 is unaffected.
- After restart, DEX DRY_RUN entry data is now HONEST: `metadata.ml_source`
  distinguishes a real ensemble signal from the heuristic fallback (currently
  always `heuristic_fallback` — no model artifacts on disk), and risk-failed
  tokens are penalized/rejected rather than rewarded. Open positions survive a
  subprocess restart (`_load_state`).

## Wave-9 punch-list (left by the quant agent; NOT regressions, do NOT block ship)

- **Same ML-fabrication / risk-failure pattern likely in sniper / solana / copy**
  entry scoring — audit each independently (Wave 8 only fixed the shared DEX
  scorer in `core/engine.py`).
- **`DecisionMaker` (`core/decision_maker.py`) is still never invoked** in the
  entry path — it is constructed but unused.
- **Engine placeholder stubs return fixed values** and feed the heuristic:
  `_check_developer_reputation` (0.5), `_check_smart_contract` (`verified=True`),
  `_analyze_liquidity_depth`, `_analyze_holder_distribution`, and
  `_extract_features` (`np.random.rand(10)`, now unused by the new ML path).
  These bound the heuristic's quality ceiling; replace with real collectors.
- The real `EnsemblePredictor` branch is exercised only once trained artifacts
  ship via `scripts/retrain_models.py` (none exist in this environment).
