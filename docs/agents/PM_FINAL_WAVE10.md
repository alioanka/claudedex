# PM Final Verification — Wave-10 Ensemble Trainer + Parity Refactor

- Date: 2026-05-26
- Branch: `claude/create-expert-agents-JFSF5`
- Verified commits: `66535e9` (parity refactor), `0f7efae` (train_ensemble.py), `ff44b90` (ml/CLAUDE.md)
- Method: static read + git-history diff + programmatic parity check + `python -m py_compile`.
  No DB and no heavy ML deps in this sandbox, so training itself was NOT run — only
  structural/static checks.

## 1. Parity refactor (core/engine.py — LIVE DEX path) — VERDICT: SHIP

The body of `TradingBotEngine._build_ml_feature_dict` was moved verbatim into the
module-level pure function `ml/feature_builder.py::build_ml_feature_dict(pair, risk_score, patterns)`.

Evidence it is a TRUE, logic-preserving move:

- `git diff 66535e9^..ff44b90 -- core/engine.py` shows exactly: one import added
  (`from ml.feature_builder import build_ml_feature_dict`, line 33), the docstring
  updated, and the entire mapping body replaced by
  `return build_ml_feature_dict(pair, risk_score, patterns)` (line 740).
- The removed lines and the lines in `ml/feature_builder.py` are character-for-character
  identical (same keys, same `.get(..., 0) or 0` coercions, same
  `pair.get('liquidity_usd') or pair.get('liquidity') or 0` fallback, same
  `(age_hours)/24.0`, same `if risk_score is not None and hasattr(risk_score,'liquidity_risk')`
  gate, same `risk_data`/`holder_data` getattr set, same `isinstance(patterns, dict)` gate).
- The call site (line 787) still invokes `self._build_ml_feature_dict(pair, risk_score, patterns)`
  with identical arguments; the wrapper method signature is unchanged.

Programmatic spot-check (PM-run, in this sandbox): transcribed the ORIGINAL body from
`git show 66535e9^:core/engine.py` into a reference function and compared dict equality
(incl. key sets) against the new function across 4 input shapes:
1. full pair + `SimpleNamespace` risk + patterns dict (all 8 sub-dicts present),
2. empty pair + `None` risk + `None` patterns (risk_data/holder_data/pattern_data absent),
3. liquidity-fallback (`liquidity` key) + partial risk obj missing `liquidity_risk`
   (hasattr gate = False) + `{}` patterns,
4. falsy-zero coercions + patterns set to a non-dict (`pattern_data` absent).

Result: `ALL MATCH: True` for all 4 cases. The agent's "byte-identical across 4 shapes"
claim is corroborated. **The live DEX feature vector is UNCHANGED.** No renamed key, no
changed default, no dropped field.

Fail-soft ML path is intact and untouched by this wave: `_is_trustworthy_ml_result` →
heuristic_fallback gate at lines 789-808 is not modified; with no artifacts on disk the
engine still falls back (`ml_source='heuristic_fallback'`).

## 2. Trainer safety + correctness (scripts/train_ensemble.py) — VERDICT: SHIP

(a) Single DB write confirmed. `grep -niE "insert|update|delete|upsert|conn.execute"` over
the file finds exactly ONE write statement: the UPSERT into
`config_settings(config_type='ml_models', key='ensemble_version')` in `record_version()`
(lines 276-289). The only other DB touch is the READ-ONLY SELECT in `fetch_rows()`
(lines 238-247: `SELECT metadata, profit_loss_percentage FROM trades WHERE status='closed'
AND metadata IS NOT NULL AND created_at >= $1`). No write to `trades`, `positions`,
`dry_run`, or any gate config. `head_out.update(...)` at line 198 is a Python dict update,
not SQL.

(b) Feature parity by construction. `row_to_feature_dict()` (lines 87-103) calls the SAME
`build_ml_feature_dict`, wrapping the stored `metadata.risk_score` DICT in a
`SimpleNamespace` (line 100) so the engine's `getattr`/`hasattr` risk branch fires
identically. Then `build_dataframe()` calls the SAME `EnsemblePredictor.extract_features`
(line 127) → canonical 82-vector. No copied/divergent mapping.

(c) Labels match retrain()'s columns. `derive_labels()` (lines 70-81):
`rug_label = pnl<=-50 OR close_reason=='stop_loss_rapid'`; `pump_label = pnl>=20`;
`returns = pnl/100.0`. The DataFrame is given columns `pump_label`, `rug_label`, `returns`
(lines 137-139). `EnsemblePredictor.retrain()` (ensemble_model.py:1176-1179) drops
`['label','pump_label','rug_label','returns']` from X and reads `pump_label`/`rug_label`
(and `returns` at line 1236). Column names MATCH exactly. retrain() also reindexes X to
`ENSEMBLE_FEATURE_NAMES` (82 names, count verified) before fitting, so train order ==
inference order.

(d) Flag safety. `--dry-run-no-save` returns 0 at line 419-422 BEFORE `retrain`/save/version
(line 426+) — writes nothing. `--mock` uses `synthetic_rows()` only, never calls
`fetch_rows`/`record_version` (DB write explicitly skipped at lines 454-457).

(e) Fail-soft. Every failure path returns a distinct non-zero code without partial writes:
DB fetch fail=2, <50 rows=3, ImportError(deps)=4, feature/label assembly fail=5,
retrain empty=6, train/save fail=7, version-write fail=8 (artifacts already valid),
KeyboardInterrupt=130. Artifacts + version are written only after a successful fit.

(f) DB connection. `_db_params()` uses `security.docker_secrets.get_db_credentials()`
(secret-file aware), reading keys `host/port/name/user/password` — which match
`get_db_credentials()`'s return shape. Falls back to env vars if the import fails.
`sys.path` is set to repo root (line 53). In the `trading-bot` container (WORKDIR /app,
PYTHONPATH /app, asyncpg + heavy ML deps present) this resolves correctly.

## 3. Safety invariants + compile

- `git diff 66535e9^..ff44b90` grepped for `dry_run|killswitch|kill_switch|min_score|`
  `max_rug_prob|should_skip_live|validate_trade|emergency`: only documentation prose and
  the `--dry-run-no-save` flag name appear. NO trading gate / kill-switch / threshold
  constant changed.
- Files touched in the whole range: `core/engine.py`, `ml/CLAUDE.md`, `ml/feature_builder.py`,
  `scripts/train_ensemble.py` only. No smart contracts, DB schemas, or trading logic.
- `python -m py_compile core/engine.py ml/feature_builder.py scripts/train_ensemble.py ml/models/ensemble_model.py`
  → ALL OK.

## 4. PM fix

None required. The refactor is a proven pure move and the trainer is safe and correct;
there is no real, trivial gap to fix.

## 5. Operator caveats before `docker exec trading-bot python scripts/train_ensemble.py --days 90`

- Inspect first: run `--dry-run-no-save --days 90` to see row count, class balance, and CV
  AUC before committing artifacts. 671 rows is small; CV numbers (not in-sample) are the
  honest read.
- The feature builder only populates a SUBSET of the 82 canonical features (price/volume/
  liquidity/market/time/risk/holder/trend_strength). The remaining columns (technical
  indicators, social, mempool, whale, contract flags) get extract_features' own defaults
  (0, RSI=50). retrain() will print a "filled with 0.0" warning for any canonical column
  absent from the DataFrame — review it; large gaps mean an under-fed model even though it
  activates. This is identical between train and inference, so it is not skew — but it does
  bound predictive quality.
- After a successful train, artifacts land in repo-root `models/` (the engine's default
  `model_dir`). Restart the DEX module so `load_models()` re-reads them, then confirm the
  per-opportunity log flips from `ML[heuristic_fallback]` to `ML[ensemble]`.
- Keep DRY_RUN = true. Activating a trained ensemble must not be paired with enabling LIVE.
  The trainer is offline/analytical and never alters DRY_RUN or places trades.

## Final verdicts

- (a) Engine parity refactor: **SHIP** — live DEX feature vector verified unchanged.
- (b) train_ensemble.py: **SHIP** — single non-trading DB write, parity by construction,
  labels match, fail-soft, DRY_RUN untouched.
