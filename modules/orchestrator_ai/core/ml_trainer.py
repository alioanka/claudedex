"""Baseline ML trainer for the orchestrator's confidence calibration.

Reads orchestrator_training_data (view from migration 019), fits a
logistic regression on (features → operator_agreed), and saves the
model + metadata to data/orchestrator_ai_model.pkl.

This is intentionally minimal: a single binary classifier on the
operator's approve/reject signal. Once we have a few hundred labeled
examples we can swap in gradient boosting + per-recommendation-type
models. Right now the goal is just the plumbing — the script runs,
the artifact lands, and the orchestrator can OPTIONALLY load it
later to override its hard-coded scorer weights.

Usage (from inside trading-bot container):
    python -m modules.orchestrator_ai.core.ml_trainer

Or from the dashboard as a Test Runner script entry.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import pickle
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import asyncpg

logger = logging.getLogger("orchestrator_ai.ml_trainer")

FEATURE_KEYS = [
    "closed_trades",
    "total_pnl_usd",
    "live_trades",
    "btc_24h_change_pct",
    "win_rate",
    "pnl_signal",
    "volume_factor",
    "regime_signal",
    "sharpe_signal",
    "sharpe",
    "score",
]
MIN_TRAINING_EXAMPLES = 30


@dataclass
class TrainingReport:
    timestamp: str
    n_examples: int
    n_positive: int   # approved
    n_negative: int   # rejected
    feature_columns: List[str]
    weights: Dict[str, float]
    bias: float
    accuracy: float
    skipped: bool
    skip_reason: Optional[str] = None


async def _load_training_data(pool) -> List[Tuple[Dict[str, Optional[float]], bool]]:
    """Returns [(features_dict, operator_agreed_bool), ...]"""
    cols_sql = ", ".join(FEATURE_KEYS)
    sql = (
        f"SELECT {cols_sql}, operator_agreed FROM orchestrator_training_data "
        "WHERE operator_agreed IS NOT NULL"
    )
    async with pool.acquire() as conn:
        rows = await conn.fetch(sql)
    out: List[Tuple[Dict[str, Optional[float]], bool]] = []
    for r in rows:
        feats = {k: (float(r[k]) if r[k] is not None else None) for k in FEATURE_KEYS}
        out.append((feats, bool(r["operator_agreed"])))
    return out


def _features_to_vector(feats: Dict[str, Optional[float]]) -> List[float]:
    """Replace None with 0.0 (a feature absent is treated as neutral).
    The same projection is used at inference."""
    return [feats.get(k) or 0.0 for k in FEATURE_KEYS]


def _fit_logistic(X: List[List[float]], y: List[bool]) -> Tuple[List[float], float, float]:
    """Tiny gradient-descent logistic regression. ~200 lines of
    sklearn-free implementation so the model trains inside the
    trading-bot container without dragging in sklearn at import time.

    Returns (weights, bias, train_accuracy).
    """
    import math
    n_features = len(X[0])
    weights = [0.0] * n_features
    bias = 0.0
    lr = 0.05
    n_iter = 200
    n = len(X)
    for _ in range(n_iter):
        # Forward pass + gradient accumulation
        grad_w = [0.0] * n_features
        grad_b = 0.0
        for xi, yi in zip(X, y):
            z = bias + sum(w * x for w, x in zip(weights, xi))
            # Clamp for numerical safety
            z = max(-30.0, min(30.0, z))
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - (1.0 if yi else 0.0)
            for j in range(n_features):
                grad_w[j] += err * xi[j]
            grad_b += err
        for j in range(n_features):
            weights[j] -= lr * grad_w[j] / n
        bias -= lr * grad_b / n
    # Train accuracy
    correct = 0
    for xi, yi in zip(X, y):
        z = bias + sum(w * x for w, x in zip(weights, xi))
        pred = z > 0.0
        if pred == yi:
            correct += 1
    return weights, bias, correct / n if n else 0.0


async def train_and_save(
    pool, output_dir: str = "data", report_only: bool = False,
) -> TrainingReport:
    """End-to-end: load → fit → save. Returns a TrainingReport.

    If there are fewer than MIN_TRAINING_EXAMPLES rows, returns a
    report with skipped=True and skip_reason — no model written.
    """
    rows = await _load_training_data(pool)
    n_total = len(rows)
    n_pos = sum(1 for _, y in rows if y)
    n_neg = n_total - n_pos
    ts = datetime.utcnow().isoformat()
    if n_total < MIN_TRAINING_EXAMPLES:
        return TrainingReport(
            timestamp=ts, n_examples=n_total, n_positive=n_pos, n_negative=n_neg,
            feature_columns=FEATURE_KEYS, weights={}, bias=0.0, accuracy=0.0,
            skipped=True,
            skip_reason=f"Need {MIN_TRAINING_EXAMPLES} labeled examples; only {n_total} present",
        )
    if n_pos == 0 or n_neg == 0:
        return TrainingReport(
            timestamp=ts, n_examples=n_total, n_positive=n_pos, n_negative=n_neg,
            feature_columns=FEATURE_KEYS, weights={}, bias=0.0, accuracy=0.0,
            skipped=True,
            skip_reason="Single-class data — need both approved and rejected examples",
        )
    X = [_features_to_vector(f) for f, _ in rows]
    y = [agreed for _, agreed in rows]
    weights, bias, acc = _fit_logistic(X, y)
    weights_dict = dict(zip(FEATURE_KEYS, weights))
    report = TrainingReport(
        timestamp=ts, n_examples=n_total, n_positive=n_pos, n_negative=n_neg,
        feature_columns=FEATURE_KEYS, weights=weights_dict, bias=bias,
        accuracy=acc, skipped=False,
    )
    if not report_only:
        out_path = Path(output_dir) / "orchestrator_ai_model.pkl"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "wb") as f:
            pickle.dump({
                "feature_columns": FEATURE_KEYS,
                "weights": weights,
                "bias": bias,
                "trained_at": ts,
                "n_examples": n_total,
                "accuracy": acc,
            }, f)
        # Also drop a JSON sidecar so an operator can inspect without
        # unpickling.
        with open(out_path.with_suffix(".json"), "w") as f:
            json.dump(asdict(report), f, indent=2)
    return report


async def _main(args) -> int:
    db_host = os.getenv("DB_HOST", "postgres")
    db_port = int(os.getenv("DB_PORT", "5432"))
    db_name = os.getenv("DB_NAME", "tradingbot")
    db_user = (
        Path("/run/secrets/db_user").read_text().strip()
        if Path("/run/secrets/db_user").exists()
        else os.getenv("DB_USER", "tradingbot")
    )
    db_pass = (
        Path("/run/secrets/db_password").read_text().strip()
        if Path("/run/secrets/db_password").exists()
        else os.getenv("DB_PASSWORD", "")
    )
    pool = await asyncpg.create_pool(
        host=db_host, port=db_port, database=db_name,
        user=db_user, password=db_pass, min_size=1, max_size=2,
    )
    try:
        report = await train_and_save(
            pool, output_dir=args.output_dir, report_only=args.report_only,
        )
    finally:
        await pool.close()
    print(json.dumps(asdict(report), indent=2))
    return 0 if not report.skipped else 1


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="data")
    parser.add_argument("--report-only", action="store_true",
                        help="Don't save the pkl — just print the report.")
    args = parser.parse_args()
    sys.exit(asyncio.run(_main(args)))
