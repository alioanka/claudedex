#!/usr/bin/env python3
"""Train the DEX EnsemblePredictor from CLOSED `trades` rows and persist the
artifacts the live engine reads (`models/*.pkl`, `models/scaler.pkl`,
`models/features.json`), then stamp a model version in `config_settings`.

WHY THIS EXISTS
---------------
`scripts/retrain_models.py` trains three *separate* legacy models into
`ml/models/` and NEVER builds an `EnsemblePredictor` nor writes the `models/`
artifacts the engine loads — so it could never flip the live ML path off its
`heuristic_fallback`. This script closes that gap. It is the canonical ensemble
retrain entrypoint (retrain_models.py is superseded for the ensemble, kept for
the legacy single models).

FEATURE PARITY (the critical invariant)
---------------------------------------
The live inference path builds its feature dict via
`ml.feature_builder.build_ml_feature_dict(pair, risk_score, patterns)` then
`EnsemblePredictor.extract_features(dict)` -> the canonical 82-vector. This
trainer calls the SAME shared function on each historical row's
`metadata.pair` / `metadata.risk_score` / `metadata.patterns`, then the SAME
`extract_features`. The training matrix is therefore identical-by-construction
to what the engine produces live — no train/inference skew.

SAFETY
------
- OFFLINE / analytical. Places NO trades. Touches NO trading gate or DRY_RUN.
- The ONLY DB write is a single UPSERT into
  `config_settings(config_type='ml_models', key='ensemble_version')`.
- Fail-soft: any error exits non-zero with a clear message and writes nothing
  partial (artifacts + version row are written only after a successful fit).

USAGE
-----
  python scripts/train_ensemble.py                 # train on last 90d, save + version
  python scripts/train_ensemble.py --days 30
  python scripts/train_ensemble.py --dry-run-no-save   # train + report, write nothing
  python scripts/train_ensemble.py --mock          # no DB; synthetic rows, full pipeline
"""
import argparse
import asyncio
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

# Project root on path so `ml.*` / `security.*` import like the other scripts.
sys.path.append(str(Path(__file__).parent.parent))

from ml.feature_builder import build_ml_feature_dict  # pure, no heavy deps

# Repo-root `models/` is what the engine reads (EnsemblePredictor default
# model_dir is "models/", repo-root relative — NOT ml/models/).
MODEL_DIR_DEFAULT = "models/"

# ----- Label thresholds (documented in the deliverable report + ml/CLAUDE.md).
RUG_PNL_THRESHOLD = -50.0          # profit_loss_percentage <= -50  => catastrophic
RUG_CLOSE_REASON = "stop_loss_rapid"  # rapid-dump close => rug regardless of pnl
PUMP_PNL_THRESHOLD = 20.0          # profit_loss_percentage >= +20  => strong winner


# --------------------------------------------------------------------------- #
# Labels
# --------------------------------------------------------------------------- #
def derive_labels(pnl_pct: float, close_reason: Optional[str]) -> Tuple[int, int, float]:
    """Return (pump_label, rug_label, returns) from outcome fields.

    rug_label  = 1 if pnl <= -50 OR close_reason == 'stop_loss_rapid'.
    pump_label = 1 if pnl >= +20 (captures take_profit / take_profit_rapid /
                 high_volatility winners).
    returns    = pnl_pct / 100.0 (fraction). Used only by the (gated) NN heads.
    """
    cr = (close_reason or "").strip()
    rug = 1 if (pnl_pct <= RUG_PNL_THRESHOLD or cr == RUG_CLOSE_REASON) else 0
    pump = 1 if pnl_pct >= PUMP_PNL_THRESHOLD else 0
    return pump, rug, pnl_pct / 100.0


# --------------------------------------------------------------------------- #
# Row -> feature dict (uses the SHARED inference builder for parity)
# --------------------------------------------------------------------------- #
def row_to_feature_dict(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Build the nested feature dict from a stored `trades.metadata` JSONB blob.

    The stored `metadata.risk_score` is a *dict*; the shared builder accesses
    risk sub-scores via getattr/hasattr (it was written for the engine's
    RiskScore *object*). We wrap the dict in a SimpleNamespace so the exact
    same code path runs — no copy, no skew.
    """
    pair = metadata.get("pair") or {}
    risk_dict = metadata.get("risk_score") or {}
    patterns = metadata.get("patterns") or {}
    # Wrap dict -> attribute-access object so the shared builder's getattr/
    # hasattr branch (risk_data / holder_data) fires exactly as it does live.
    risk_obj = SimpleNamespace(**risk_dict) if isinstance(risk_dict, dict) else None
    if not isinstance(patterns, dict):
        patterns = {}
    return build_ml_feature_dict(pair, risk_obj, patterns)


def build_dataframe(rows: List[Tuple[Dict[str, Any], float]]):
    """Assemble the labeled training DataFrame.

    rows: list of (metadata_dict, profit_loss_percentage).
    Columns = 82 canonical feature names (canonical order) + pump_label +
    rug_label + returns. Heavy imports (pandas, ensemble) are local so the
    pure/mock-prep stages and py_compile work without the bot image.
    """
    import numpy as np
    import pandas as pd
    from ml.models.ensemble_model import EnsemblePredictor, ENSEMBLE_FEATURE_NAMES

    predictor = EnsemblePredictor({"model_dir": MODEL_DIR_DEFAULT})

    feat_matrix: List[np.ndarray] = []
    pump_labels: List[int] = []
    rug_labels: List[int] = []
    returns: List[float] = []

    for metadata, pnl_pct in rows:
        feat_dict = row_to_feature_dict(metadata)
        vec = predictor.extract_features(feat_dict)  # canonical 82-vector
        close_reason = (metadata or {}).get("close_reason")
        pump, rug, ret = derive_labels(float(pnl_pct), close_reason)
        feat_matrix.append(vec)
        pump_labels.append(pump)
        rug_labels.append(rug)
        returns.append(ret)

    X = np.vstack(feat_matrix) if feat_matrix else np.empty((0, len(ENSEMBLE_FEATURE_NAMES)))
    df = pd.DataFrame(X, columns=list(ENSEMBLE_FEATURE_NAMES))
    df["pump_label"] = pump_labels
    df["rug_label"] = rug_labels
    df["returns"] = returns
    return df, predictor


# --------------------------------------------------------------------------- #
# Cross-validated metrics (small data: be honest, don't overfit-claim)
# --------------------------------------------------------------------------- #
def cv_metrics(df) -> Dict[str, Any]:
    """Stratified-CV AUC / precision / recall per head with a small,
    regularized XGBoost using scale_pos_weight for imbalance. Returns plain
    floats so the result is JSON-serialisable for the version row.

    671 rows is small -> we report CV (not in-sample) numbers and never claim
    more than the data supports. Folds drop to a safe minimum when the positive
    class is tiny.
    """
    import numpy as np
    from ml.models.ensemble_model import ENSEMBLE_FEATURE_NAMES

    X = df.reindex(columns=list(ENSEMBLE_FEATURE_NAMES), fill_value=0.0).values
    out: Dict[str, Any] = {}
    try:
        from sklearn.model_selection import StratifiedKFold
        from sklearn.preprocessing import RobustScaler
        from sklearn.metrics import roc_auc_score, precision_score, recall_score
        import xgboost as xgb
    except Exception as e:  # noqa: BLE001
        return {"error": f"cv-skipped (dep missing): {e}"}

    for head in ("pump_label", "rug_label"):
        y = df[head].values.astype(int)
        pos = int(y.sum())
        neg = int(len(y) - pos)
        head_out: Dict[str, Any] = {"n_pos": pos, "n_neg": neg}
        if pos < 2 or neg < 2:
            head_out["note"] = "too few of one class for CV"
            out[head] = head_out
            continue
        n_splits = max(2, min(5, pos, neg))
        spw = (neg / pos) if pos else 1.0
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        aucs, precs, recs = [], [], []
        for tr, te in skf.split(X, y):
            scaler = RobustScaler()
            Xtr = scaler.fit_transform(X[tr])
            Xte = scaler.transform(X[te])
            clf = xgb.XGBClassifier(
                n_estimators=120, max_depth=3, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=2.0,
                scale_pos_weight=spw, eval_metric="logloss",
                use_label_encoder=False, random_state=42,
            )
            clf.fit(Xtr, y[tr])
            proba = clf.predict_proba(Xte)[:, 1]
            pred = (proba >= 0.5).astype(int)
            if len(set(y[te])) > 1:
                aucs.append(float(roc_auc_score(y[te], proba)))
            precs.append(float(precision_score(y[te], pred, zero_division=0)))
            recs.append(float(recall_score(y[te], pred, zero_division=0)))
        head_out.update({
            "cv_folds": n_splits,
            "scale_pos_weight": round(float(spw), 3),
            "auc_mean": round(float(np.mean(aucs)), 4) if aucs else None,
            "precision_mean": round(float(np.mean(precs)), 4) if precs else None,
            "recall_mean": round(float(np.mean(recs)), 4) if recs else None,
        })
        out[head] = head_out
    return out


# --------------------------------------------------------------------------- #
# DB access (mirrors migrate_database.py: Docker secrets, env fallback)
# --------------------------------------------------------------------------- #
def _db_params() -> Dict[str, Any]:
    try:
        from security.docker_secrets import get_db_credentials
        c = get_db_credentials()
        return {
            "host": c["host"], "port": int(c["port"]), "database": c["name"],
            "user": c["user"], "password": c["password"],
        }
    except Exception:
        return {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": int(os.getenv("DB_PORT", "5432")),
            "database": os.getenv("DB_NAME", "tradingbot"),
            "user": os.getenv("DB_USER", "bot_user"),
            "password": os.getenv("DB_PASSWORD", ""),
        }


async def fetch_rows(days: int) -> List[Tuple[Dict[str, Any], float]]:
    """SELECT closed trades with non-null metadata within `days`."""
    import asyncpg
    params = _db_params()
    print(f"📊 Connecting to {params['user']}@{params['host']}:{params['port']}/{params['database']}")
    conn = await asyncpg.connect(timeout=15, **params)
    try:
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        recs = await conn.fetch(
            """
            SELECT metadata, profit_loss_percentage
            FROM trades
            WHERE status = 'closed'
              AND metadata IS NOT NULL
              AND created_at >= $1
            """,
            cutoff,
        )
    finally:
        await conn.close()
    rows: List[Tuple[Dict[str, Any], float]] = []
    for r in recs:
        md = r["metadata"]
        if isinstance(md, str):
            try:
                md = json.loads(md)
            except Exception:
                continue
        if not isinstance(md, dict):
            continue
        pnl = r["profit_loss_percentage"]
        if pnl is None:
            continue
        rows.append((md, float(pnl)))
    return rows


async def record_version(version_payload: Dict[str, Any]) -> None:
    """UPSERT the model version into config_settings(ml_models, ensemble_version).

    This is the ONLY DB write the script performs. It does not touch trades,
    dry_run, or any trading-behaviour config.
    """
    import asyncpg
    conn = await asyncpg.connect(timeout=15, **_db_params())
    try:
        await conn.execute(
            """
            INSERT INTO config_settings
                (config_type, key, value, value_type, description, is_editable, requires_restart)
            VALUES ('ml_models', 'ensemble_version', $1, 'json',
                    'DEX ensemble model version (written by scripts/train_ensemble.py)',
                    FALSE, TRUE)
            ON CONFLICT (config_type, key)
            DO UPDATE SET value = EXCLUDED.value,
                          value_type = 'json',
                          updated_at = NOW()
            """,
            json.dumps(version_payload),
        )
    finally:
        await conn.close()


# --------------------------------------------------------------------------- #
# Synthetic data for --mock
# --------------------------------------------------------------------------- #
def synthetic_rows(n: int = 400) -> List[Tuple[Dict[str, Any], float]]:
    """Generate ~n realistic metadata rows mirroring the production JSONB shape
    (pair + risk_score + close_reason) with plausibly-correlated pnl so labels
    aren't degenerate. Mirrors the close_reason distribution from the operator's
    DB."""
    rng = random.Random(1234)
    reasons = (
        ["stop_loss"] * 49 + ["take_profit"] * 19 + ["trailing_stop"] * 11 +
        ["high_volatility"] * 8 + ["stop_loss_rapid"] * 6 + ["time_limit"] * 5 +
        ["take_profit_rapid"] * 2
    )
    rows: List[Tuple[Dict[str, Any], float]] = []
    for _ in range(n):
        cr = rng.choice(reasons)
        if cr in ("take_profit", "take_profit_rapid"):
            pnl = rng.uniform(20, 133)
        elif cr == "high_volatility":
            pnl = rng.uniform(-30, 60)
        elif cr == "trailing_stop":
            pnl = rng.uniform(5, 40)
        elif cr == "stop_loss_rapid":
            pnl = rng.uniform(-100, -45)
        elif cr == "time_limit":
            pnl = rng.uniform(-15, 15)
        else:  # stop_loss
            pnl = rng.uniform(-60, -2)
        pair = {
            "price_usd": rng.uniform(1e-6, 5.0),
            "price_change_1h": rng.uniform(-25, 25),
            "price_change_24h": rng.uniform(-60, 80),
            "volume_24h": rng.uniform(1e3, 5e6),
            "volume_change_24h": rng.uniform(-90, 300),
            "buy_sell_ratio": rng.uniform(0.4, 2.5),
            "liquidity_usd": rng.uniform(2e3, 8e5),
            "age_hours": rng.uniform(1, 720),
            "market_cap": rng.uniform(5e4, 5e7),
            "fdv": rng.uniform(5e4, 8e7),
            "chain": rng.choice(["ethereum", "base", "bsc"]),
        }
        risk_score = {
            "liquidity_risk": rng.uniform(0, 60),
            "developer_risk": rng.uniform(0, 50),
            "contract_risk": rng.uniform(0, 40),
            "volume_risk": rng.uniform(0, 70),
            "holder_risk": rng.uniform(0, 60),
            "honeypot_risk": rng.uniform(0, 20),
            "top_10_holders_percentage": rng.uniform(10, 90),
            "whale_concentration": rng.uniform(0, 10),
            "unique_holders": rng.uniform(20, 5000),
        }
        rows.append(({"pair": pair, "risk_score": risk_score,
                      "patterns": {"trend_strength": rng.uniform(-1, 1)},
                      "close_reason": cr}, pnl))
    return rows


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def _git_sha() -> Optional[str]:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).parent.parent), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return None


def _class_counts(df) -> Dict[str, int]:
    return {
        "n_samples": int(len(df)),
        "pump_pos": int(df["pump_label"].sum()),
        "rug_pos": int(df["rug_label"].sum()),
    }


async def run(args) -> int:
    print("=" * 64)
    print("DEX ENSEMBLE TRAINER" + ("  [MOCK]" if args.mock else ""))
    print("=" * 64)

    # 1) Gather rows.
    if args.mock:
        rows = synthetic_rows(400)
        print(f"🧪 Mock mode: generated {len(rows)} synthetic metadata rows")
    else:
        try:
            rows = await fetch_rows(args.days)
        except Exception as e:  # noqa: BLE001
            print(f"❌ DB fetch failed: {e}")
            return 2
        print(f"📥 Fetched {len(rows)} closed trades (last {args.days}d)")

    if len(rows) < 50:
        print(f"❌ Not enough rows to train ({len(rows)} < 50). Aborting; nothing written.")
        return 3

    # 2) Build labeled DataFrame via the SHARED feature path (parity).
    try:
        df, predictor = build_dataframe(rows)
    except ImportError as e:
        print(f"❌ Heavy ML deps unavailable in this environment: {e}")
        print("   (xgboost/lightgbm/sklearn/torch/pandas live in the bot image — "
              "run this on the operator's VPS / inside the container.)")
        return 4
    except Exception as e:  # noqa: BLE001
        print(f"❌ Feature/label assembly failed: {e}")
        return 5

    counts = _class_counts(df)
    print(f"\n📊 Rows used: {counts['n_samples']}")
    print(f"   pump_label positives: {counts['pump_pos']} "
          f"({100*counts['pump_pos']/max(1,counts['n_samples']):.1f}%)")
    print(f"   rug_label  positives: {counts['rug_pos']} "
          f"({100*counts['rug_pos']/max(1,counts['n_samples']):.1f}%)")

    # 3) Cross-validated metrics (honest, regularized, imbalance-aware).
    metrics = cv_metrics(df)
    print("\n📈 Cross-validated metrics (StratifiedKFold, scale_pos_weight):")
    print(json.dumps(metrics, indent=2))

    if args.dry_run_no_save:
        print("\n🛈 --dry-run-no-save: trained-CV report only. "
              "NO artifacts, NO version row written.")
        return 0

    # 4) Fit full ensemble + persist artifacts to models/.
    try:
        new_models = await predictor.retrain(df)
        if not new_models:
            print("❌ retrain() returned no models — nothing saved.")
            return 6
        await predictor.update_models(new_models)  # -> save_models()
    except Exception as e:  # noqa: BLE001
        print(f"❌ Train/save failed: {e}")
        return 7

    model_dir = Path(predictor.model_dir)
    written = sorted(p.name for p in model_dir.glob("*") if p.is_file())
    print(f"\n💾 Artifacts written to {model_dir.resolve()}:")
    for name in written:
        print(f"   - {name}")
    if "scaler.pkl" not in written or "features.json" not in written:
        print("⚠️  Expected scaler.pkl + features.json among artifacts — verify above.")

    # 5) Record version (the ONLY DB write).
    version_payload = {
        "version": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(),
        "n_samples": counts["n_samples"],
        "pump_pos": counts["pump_pos"],
        "rug_pos": counts["rug_pos"],
        "cv_metrics": metrics,
        "source": "mock" if args.mock else f"trades(closed, {args.days}d)",
    }
    if args.mock:
        print("\n🧪 Mock mode: skipping config_settings version write "
              "(no DB). Payload that WOULD be written:")
        print(json.dumps(version_payload, indent=2))
    else:
        try:
            await record_version(version_payload)
            print(f"\n🗂  Recorded version in config_settings(ml_models, ensemble_version): "
                  f"{version_payload['version']}")
        except Exception as e:  # noqa: BLE001
            print(f"⚠️  Artifacts saved but version write failed: {e}")
            print("   (Artifacts are valid; re-record the version manually if needed.)")
            return 8

    print("\n✅ SUMMARY: rows=%d pump_pos=%d rug_pos=%d artifacts=%d version=%s" % (
        counts["n_samples"], counts["pump_pos"], counts["rug_pos"],
        len(written), version_payload["version"]))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description="Train the DEX EnsemblePredictor from closed trades")
    p.add_argument("--days", type=int, default=90, help="Days of closed trades to use (default 90)")
    p.add_argument("--mock", action="store_true",
                   help="No DB: synthetic rows, full fit+save pipeline (offline self-test)")
    p.add_argument("--dry-run-no-save", action="store_true",
                   help="Train + report CV metrics only; write NO artifacts and NO version")
    args = p.parse_args()
    try:
        return asyncio.run(run(args))
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted — nothing partial written.")
        return 130


if __name__ == "__main__":
    sys.exit(main())
