# PM Final Verification — Wave 9

Date: 2026-05-26 · Branch: `claude/create-expert-agents-JFSF5` · HEAD pre-fix: `31b3cdf`
Method: code read + AST analysis + `python -m py_compile`. No DB, no live bot, DRY_RUN untouched.

VERDICT: **SHIP.** No defect in Agent A's audit or Agent B's code changes halts DEX entries
or loosens a safety gate. One trivial documentation mismatch fixed by PM. Two items deferred
to Wave 10 (ensemble-artifact pipeline; DecisionMaker integration).

---

## Agent A — `50e37ea` (sniper/solana/copy honest-scoring audit) — VERIFIED (docs-only)

`git show --stat 50e37ea` = 3 CLAUDE.md files, 95 insertions, 0 deletions. **No code touched.**
Spot-checked the four central "ABSENT" claims against the cited code; all hold:

- (a) `sniper/core/token_safety.py::_calculate_score` (548) is **purely subtractive** — starts
  at 100, subtracts penalties, `return max(0, min(100, int(score)))`. There is NO
  normalize-by-weight denominator, so the Wave-8 "dropped failed check inflates score" pattern
  is structurally impossible. ABSENT — confirmed.
- (b) `sniper/core/sniper_engine.py::_check_filters` (473): the `except` block (587) logs,
  cooldowns the token, bumps `safety_check_errors`, and `return False` — **fail-safe**. Also
  honeypot/DANGER/high-tax/low-liquidity each `return False`. ABSENT — confirmed.
- (c) `solana_trading/core/solana_engine.py::_ml_rug_probability` (1780) /
  `_pump_predict_probability` (1852): every no-model / no-data / unfitted-scaler branch
  `return None` (refuse-to-predict). Callers (3869, 3894) act only `if prob is not None`, are
  feature-flagged OFF by default, and on `None` the upstream hard boolean gates stay
  authoritative. Missing model is benign, never rewarded. ABSENT — confirmed.
- (d) `copy_trading/leader_scorer.py::compute_score` (327): `components` always builds all 5
  entries; `composite = sum(components[k] * w[k] for k in components)` iterates the fixed dict —
  no component dropped. Sparse data shrinks the score via `sample_credit`; sparse Kelly → 0.
  ABSENT — confirmed.

No new instance of either Wave-8 pattern found in the three modules. Nothing to fix.

---

## Agent B `#1` — `37e2cfe` (honest placeholder stubs) — VERIFIED

**THE KEY DETERMINATION: does flipping the contract-gate default `True`→`False` halt DEX
entries? NO — it only changes a log line. SAFE.**

`core/engine.py::_final_safety_checks` step 6 (2962-2966):
```python
contract_safety = opportunity.metadata.get('contract_safety', {})
if not contract_safety.get('verified', False):
    logger.warning("   ⚠️  Contract not verified - proceeding with caution")
# (no return False; flow continues to `return True`)
```
This is a **WARN-only** branch with no `return False`. Flipping the default makes the caution
*fire* when verification is absent/unknown (which is now always, since `_check_smart_contract`
returns `verified=False`/`status=unknown`), but the function still proceeds to `return True`.
Entry behaviour is **unchanged**; only logging is stricter/more honest.

Second consumer of `contract_safety`: it is passed to `_calculate_opportunity_score(...,
contract_safety=...)` (919) but the scorer body (3711) **never reads the parameter** — its
components are volume 30% / liquidity 35% / price_change 10% / risk 20% (RiskScore;
hard-rejects on missing) / age 5%. So the now-honest `verified=False` stub does **not** lower
the score either. Conclusion: `_check_smart_contract` returning `unknown` cannot starve DEX
entries through any path.

Other stubs (grep-confirmed no gate reads them):
- `_check_developer_reputation` → 0.5 neutral. Computed (888), unpacked into `dev_reputation`
  (896), then never referenced again — not even persisted. No gate. Confirmed.
- `_analyze_holder_distribution` → `{'concentrated': None, 'status':'unknown'}`. Stored in
  metadata (986) only; `concentrated` appears nowhere except its own return/docstring. No gate.
  Confirmed.
- `_extract_features` (returned `np.random.rand(10)`) **deleted**. `grep -n _extract_features
  core/engine.py` → only two **comment** hits (2227 docstring referencing the *different* class
  `AIStrategy._extract_features`; 3181 the deletion note). **Zero callers.** Confirmed; the live
  ML path is `_build_ml_feature_dict` → `EnsemblePredictor.extract_features`.

Net: the contract gate got **stricter** (honest), no other gate changed, no entries halted.

---

## Agent B `#2` — `31b3cdf` (ensemble feature contract) — VERIFIED

**FEATURE-CONTRACT MATCH VERDICT: EXACT MATCH (field-for-field, order-for-order).**

AST comparison of `ENSEMBLE_FEATURE_NAMES` (82 names, all unique) vs the 82 scalar `.get(...)`
keys in `extract_features`, sorted by true source position:
`ORDER IDENTICAL (source-sorted): True`. Set difference both directions: empty. Section counts
8+7+5+6+7+7+10+6+5+5+8+4+4 = 82. **No silent-garbage column remap possible.**

- `retrain` (1164): `X = X.reindex(columns=ENSEMBLE_FEATURE_NAMES, fill_value=0.0)` at **1198**,
  BEFORE `self.scaler.fit_transform(X)` at **1201**. `self.feature_names` set to canonical order
  (1205). Missing canonical cols → 0.0 + warning; extra cols dropped. train==inference by
  construction.
- Inference (`_predict_from_features`, 953) scales the same `extract_features` vector via the
  same scaler (`self.scaler.transform`, 971). Same order on both sides — confirmed.
- **Fail-soft intact (two degenerate paths, both rejected):**
  1. Unfitted scaler → `NotFittedError` inside `_predict_from_features`; caught at 1120, returns
     neutral `PredictionResult(pump=0.5, rug=0.5, confidence=0.1)`. `predict_decoupled` wraps it
     (no `error` key). Engine `_is_trustworthy_ml_result` rejects via the all-neutral check
     (engine.py 804: `|pump-0.5|<1e-9 and |rug-0.5|<1e-9 and conf<=0.1`) → heuristic fallback.
  2. `predict_decoupled`'s own `except` (798) returns the neutral dict **with** `error` key →
     rejected at engine.py 796. Either way: never crashes, never fabricates a confident signal.
     `ml_source='heuristic_fallback'`.
- **KNOWN GAP accurately documented:** `grep EnsemblePredictor scripts/retrain_models.py` → zero
  hits. The script trains only Pump/Rug/Volume models to `ml/models/`; it does NOT write the
  `models/` ensemble artifacts. `ml/CLAUDE.md` records this as the gating open item. Accurate,
  not silently broken.

---

## FIXED-BY-PM — `ml/CLAUDE.md` filename mismatch (trivial doc-vs-code)

`ml/CLAUDE.md` referenced `feature_names.json` in 4 places (lines 35, 52, 79, 93), but the
actual code reads/writes **`features.json`**:
- `save_models()` (ensemble_model.py:1331): `open(self.model_dir / "features.json", 'w')`
- `load_models()` (ensemble_model.py:384): `features_path = self.model_dir / "features.json"`

Save/load are internally consistent (round-trip works) and the data-correctness guarantee does
NOT depend on this filename — it depends on the verified `extract_features` == reindex order. So
this is a **doc bug, not a functional bug**: an operator following the activation runbook would
look for `models/feature_names.json` and not find it. PM fixed `ml/CLAUDE.md` to say
`models/features.json` (doc-coherence is PM-owned). Grep confirmed no other `.py` reads either
filename. Code left untouched (it works) — see Wave-10 punch-list for the cosmetic code-comment.

---

## Safety invariants

- Only gate-constant change in Wave 9: `_final_safety_checks` `contract_safety.get('verified',
  True)` → `('verified', False)` = **STRICTER**. No `min_score` / `max_rug_prob` /
  `min_liquidity` / `dry_run` / kill-switch / `min_opportunity_score` constant changed.
- `_check_smart_contract` return changed from false-positive `{verified:True}` → honest
  `{verified:False, status:unknown}` — removes an optimistic signal, adds none.
- Ensemble diff: only `fill_value=0.0` reindex — no threshold touched.
- DRY_RUN, `logs/.killswitch`, `SNIPER_SAFETY_CHECK_ENABLED` files untouched by all 3 commits.

## py_compile

`python -m py_compile core/engine.py core/decision_maker.py ml/models/ensemble_model.py
scripts/retrain_models.py` → **PY_COMPILE_OK** (clean).

---

## Wave-10 punch-list (open, non-blocking)

1. **Ensemble-retrain artifact pipeline (gates ensemble activation).** Build a
   DB→canonical-82-feature trainer that constructs `EnsemblePredictor`, calls `retrain` +
   `update_models`/`save_models` to `models/`, and stamps a `model_version` row. Today the
   ensemble can only run in `heuristic_fallback`. (Owner: quant/ML.)
2. **DecisionMaker integration deferred** until the ensemble is actually activated (no value
   wiring an untrained-fallback signal deeper). (Owner: quant.)
3. **Cosmetic (ML agent):** `ensemble_model.py:1204` inline comment still says
   "feature_names.json"; the code uses `features.json`. Optionally rename the artifact to
   `feature_names.json` in BOTH `save_models` (1331) and `load_models` (384) for clarity, OR fix
   the comment. Round-trip works either way; non-blocking. (Owner: ML.)
