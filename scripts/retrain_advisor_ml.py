"""
scripts/retrain_advisor_ml.py — Retrain the Advisor ML confidence model.

Standalone operator script.  Pulls closed advisor_sim_positions from DB,
trains a LightGBM win/loss classifier, and writes artifacts to models/.
Also UPSERTs advisor_ml_version into config_settings.

Usage
-----
    # Inspect (no artifacts written, no DB write):
    python scripts/retrain_advisor_ml.py --dry-run

    # Standard retrain:
    python scripts/retrain_advisor_ml.py

    # Custom artifact directory:
    python scripts/retrain_advisor_ml.py --model-dir /data/advisor_ml

    # Bypass min-samples gate (e.g. during testing with synthetic data):
    python scripts/retrain_advisor_ml.py --force

Validation method
-----------------
Walk-forward, time-ordered 80/20 split.  No shuffle.  Test metrics are
out-of-sample by construction.  See AdvisorMLModel.retrain() docstring.

Config keys read
----------------
  advisor_ml_min_samples  (default 50)
  advisor_ml_retrain_days (default 7)
  advisor_ml_blend_weight (default 0.6)

Artifacts written
-----------------
  models/advisor_ml_model.pkl
  models/advisor_ml_scaler.pkl
  models/advisor_ml_features.json
  config_settings (config_type='advisor_config', key='advisor_ml_version')
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Allow running from repo root.
sys.path.insert(0, str(Path(__file__).parent.parent))


async def main(dry_run: bool, model_dir: Path, force: bool) -> None:
    import asyncpg  # type: ignore[import]
    from modules.advisor.core.advisor_ml import AdvisorMLModel, _fetch_training_rows

    conn_kwargs = dict(
        host=os.getenv("DB_HOST", "localhost"),
        port=int(os.getenv("DB_PORT", 5432)),
        database=os.getenv("DB_NAME", "tradingbot"),
        user=os.getenv("DB_USER", "trading"),
        password=os.getenv("DB_PASSWORD", "trading123"),
    )

    print("Advisor ML Retrainer")
    print("=" * 60)

    pool = await asyncpg.create_pool(**conn_kwargs, min_size=1, max_size=3)

    # Load advisor_config from DB.
    async with pool.acquire() as conn:
        rows = await conn.fetch(
            "SELECT key, value FROM config_settings WHERE config_type='advisor_config'"
        )
    config = {r["key"]: r["value"] for r in rows}

    ml = AdvisorMLModel(config=config, model_dir=model_dir)

    if force:
        # Override min_samples gate by patching config.
        original_min = ml.config.get("advisor_ml_min_samples", "50")
        ml.config["advisor_ml_min_samples"] = "1"
        print(f"[force] Overriding min_samples from {original_min} to 1")

    # Fetch data to show counts.
    print("\nFetching closed sim positions...")
    try:
        rows_data = await _fetch_training_rows(pool)
    except Exception as exc:
        print(f"ERROR fetching training data: {exc}")
        await pool.close()
        sys.exit(1)

    n = len(rows_data)
    print(f"Found {n} closed sim positions with pnl_pct and joined advice.")

    win_count = sum(1 for r in rows_data if float(r.get("pnl_pct") or 0) > 0)
    loss_count = n - win_count
    print(f"Win/Loss: {win_count}/{loss_count} ({win_count/n*100:.1f}% win rate)" if n > 0 else "No data.")

    min_samples = int(ml.config.get("advisor_ml_min_samples", 50))
    print(f"Min samples gate: {min_samples}")

    if dry_run:
        print(f"\n[dry-run] Would train on {n} samples.")
        print(f"[dry-run] Artifacts would be written to: {model_dir}")
        print("[dry-run] No changes made.")
        await pool.close()
        return

    if n < min_samples and not force:
        print(
            f"\nNot enough samples ({n} < {min_samples}). "
            "Skipping retrain.\n"
            "Collect more closed sim positions first, or use --force to bypass."
        )
        await pool.close()
        return

    print("\nRetraining model...")
    success = await ml.retrain(pool)
    if success:
        print(f"\nRetrain complete. Artifacts: {model_dir}")
        print(f"Model version: {ml._model_version}")
        print(f"Trained on {ml._n_training_samples} samples.")
        print("\nNext steps:")
        print("  1. Restart the advisor module to load the new model.")
        print("  2. Enable ML in DB:")
        print("       UPDATE config_settings SET value='true'")
        print("       WHERE config_type='advisor_config'")
        print("         AND key='advisor_ml_enabled';")
    else:
        print("\nRetrain failed or skipped. Check logs for details.")

    await pool.close()


def cli() -> None:
    parser = argparse.ArgumentParser(
        description="Retrain the Advisor ML confidence model from closed sim outcomes."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show data counts without training or writing artifacts.",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("models"),
        help="Directory for model artifacts (default: ./models).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass min-samples gate. Use for testing; do NOT use in production.",
    )
    args = parser.parse_args()
    asyncio.run(main(dry_run=args.dry_run, model_dir=args.model_dir, force=args.force))


if __name__ == "__main__":
    cli()
