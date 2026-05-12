"""
Feature-store write-side helpers.

Persists feature vectors per signal-generation cycle into ai_feature_store
(migrations/014). Training scripts query this table for real-data fits
(follow-up; today they fit on _generate_synthetic_* blocks).

Best-effort by design: write failures NEVER block signal generation.
"""

import json
import logging
from typing import Any, Dict, Optional

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
