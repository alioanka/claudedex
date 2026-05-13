"""
Feature-store write-side helpers.

Persists feature vectors per signal-generation cycle into ai_feature_store
(migrations/014). Training scripts query this table for real-data fits
(follow-up; today they fit on _generate_synthetic_* blocks).

Best-effort by design: write failures NEVER block signal generation.
"""

import json
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np

logger = logging.getLogger("feature_store")


def _sanitize(obj: Any) -> Any:
    """Convert numpy types into JSON-serialisable Python primitives."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(v) for v in obj]
    return obj


async def write_feature_row(
    db_pool,
    *,
    token_address: Optional[str],
    chain: str = "unknown",
    feature_vector: Dict[str, Any],
    side: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Best-effort write. Returns the new row id, or None on failure.

    Never raises — feature-store outages must not gate the live signal path.
    """
    if db_pool is None:
        return None
    try:
        async with db_pool.acquire() as conn:
            row_id = await conn.fetchval(
                """
                INSERT INTO ai_feature_store
                    (token_address, chain, feature_vector, side, metadata)
                VALUES ($1, $2, $3::jsonb, $4, $5::jsonb)
                RETURNING id
                """,
                token_address,
                chain,
                json.dumps(_sanitize(feature_vector)),
                side,
                json.dumps(_sanitize(metadata or {})),
            )
            return row_id
    except Exception as e:
        logger.debug(f"feature-store write failed (non-fatal): {e}")
        return None


async def update_outcome(
    db_pool,
    *,
    row_id: int,
    outcome: Dict[str, Any],
) -> bool:
    """Backfill outcome_label on a previously-written row. Called by the
    trade-close hook (separate follow-up). Best-effort."""
    if db_pool is None or row_id is None:
        return False
    try:
        async with db_pool.acquire() as conn:
            await conn.execute(
                "UPDATE ai_feature_store SET outcome_label = $1::jsonb WHERE id = $2",
                json.dumps(_sanitize(outcome)),
                row_id,
            )
            return True
    except Exception as e:
        logger.debug(f"feature-store outcome update failed (non-fatal): {e}")
        return False


async def load_labeled_features(
    database_url: str,
    *,
    feature_key: str,
    limit: int = 5000,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Pull labeled features from ai_feature_store for offline training.

    Args:
        database_url: asyncpg-style DSN.
        feature_key:  Which key inside feature_vector JSONB to extract.
                      Today AIStrategy writes 'scaler_v1' and 'raw_v1'; rug
                      and pump shapes ('rug_v1', 'pump_v1') aren't written
                      yet — those queries will return None until AIStrategy
                      grows additional write hooks.
        limit:        Max rows to pull (ORDER BY timestamp DESC).

    Returns:
        (X, pnl_pct_vec, won_vec) on success, or None on connect/query
        failure or fewer than 10 matching rows. Never raises.
    """
    import asyncpg  # local import: keep optional for non-DB callers
    try:
        conn = await asyncpg.connect(database_url)
    except Exception as e:
        logger.warning(f"feature-store load: connect failed: {e}")
        return None

    try:
        rows = await conn.fetch(
            """
            SELECT feature_vector, outcome_label
            FROM ai_feature_store
            WHERE outcome_label IS NOT NULL
              AND feature_vector ? $1
            ORDER BY timestamp DESC
            LIMIT $2
            """,
            feature_key,
            int(limit),
        )
    except Exception as e:
        logger.warning(f"feature-store load: query failed: {e}")
        return None
    finally:
        await conn.close()

    if len(rows) < 10:
        return None

    X_list, pnl_list, won_list = [], [], []
    for r in rows:
        try:
            fv = json.loads(r["feature_vector"]) if isinstance(r["feature_vector"], str) else r["feature_vector"]
            ol = json.loads(r["outcome_label"]) if isinstance(r["outcome_label"], str) else r["outcome_label"]
            vec = fv.get(feature_key) if isinstance(fv, dict) else None
            if not isinstance(vec, list) or not vec:
                continue
            X_list.append(vec)
            pnl_list.append(float(ol.get("pnl_pct", 0.0) or 0.0))
            won_list.append(1 if ol.get("won") else 0)
        except Exception:
            continue

    if len(X_list) < 10:
        return None
    return (
        np.asarray(X_list, dtype=float),
        np.asarray(pnl_list, dtype=float),
        np.asarray(won_list, dtype=int),
    )
