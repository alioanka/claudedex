"""
KAP store -- DB read/write helpers.

This module is the single point of DB access for the KAP ingestion pipeline.
Used by: kap_listener.py, kap_archive_crawler.py, forward_return_accumulator.py,
         and the sibling classifier agent (read/write classification hook).

All functions accept an asyncpg connection or pool object.
All functions are fail-soft: they return None / empty list on DB errors
and log a WARNING rather than raising.

Sibling classifier contract (migration 063)
-------------------------------------------
The classifier agent creates kap_classifications with a disclosure_id FK.
This module exposes:
  get_unclassified(pool)    -> list of kap_disclosures rows with no matching
                               kap_classifications row (LEFT JOIN IS NULL).
  store_classification(...) -> INSERT/UPSERT into kap_classifications.

If migration 063 has not been run yet (table missing), both functions degrade
gracefully: get_unclassified() returns all recent disclosures; store_classification()
logs a WARNING and returns False.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger("advisor.kap.store")

# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

async def upsert_disclosure(pool, row: Dict[str, Any]) -> Optional[int]:
    """
    Insert or update a kap_disclosures row.
    Returns the DB id on success, None on error.
    Dedup key: disclosure_id (unique index).

    row keys expected:
      disclosure_id, ticker, tickers (list), company_name, subject,
      disclosure_type, summary, full_text, url, disclosed_at (datetime),
      source ('pykap'|'scrape'), raw_payload (dict).
    """
    sql = """
        INSERT INTO kap_disclosures (
            disclosure_id, ticker, tickers, company_name, subject,
            disclosure_type, summary, full_text, url, disclosed_at,
            source, raw_payload, fetched_at, updated_at
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12::jsonb,NOW(),NOW())
        ON CONFLICT (disclosure_id) DO UPDATE SET
            company_name    = EXCLUDED.company_name,
            summary         = EXCLUDED.summary,
            full_text       = EXCLUDED.full_text,
            raw_payload     = EXCLUDED.raw_payload,
            updated_at      = NOW()
        RETURNING id
    """
    import json

    tickers = row.get("tickers") or []
    if row.get("ticker") and row["ticker"] not in tickers:
        tickers = [row["ticker"]] + list(tickers)

    try:
        record = await pool.fetchrow(
            sql,
            str(row["disclosure_id"]),
            row.get("ticker"),
            tickers or None,
            row.get("company_name", ""),
            row.get("subject", ""),
            row.get("disclosure_type", ""),
            row.get("summary", ""),
            row.get("full_text", ""),
            row.get("url", ""),
            row["disclosed_at"],
            row.get("source", "pykap"),
            json.dumps(row.get("raw_payload") or {}),
        )
        return record["id"] if record else None
    except Exception as exc:
        logger.warning("[kap.store] upsert_disclosure failed for %s: %s",
                       row.get("disclosure_id"), exc)
        return None


async def upsert_company_profile(pool, ticker: str, company_name: str,
                                  company_id: Optional[str] = None,
                                  sector: Optional[str] = None,
                                  extra: Optional[dict] = None) -> bool:
    """
    Insert or update kap_company_profiles. Idempotent by ticker.
    Returns True on success.
    """
    import json
    sql = """
        INSERT INTO kap_company_profiles (ticker, company_name, company_id, sector, extra)
        VALUES ($1, $2, $3, $4, $5::jsonb)
        ON CONFLICT (ticker) DO UPDATE SET
            company_name = EXCLUDED.company_name,
            company_id   = COALESCE(EXCLUDED.company_id, kap_company_profiles.company_id),
            sector       = COALESCE(EXCLUDED.sector,      kap_company_profiles.sector),
            extra        = kap_company_profiles.extra || EXCLUDED.extra,
            updated_at   = NOW()
    """
    try:
        await pool.execute(sql, ticker, company_name, company_id, sector,
                           json.dumps(extra or {}))
        return True
    except Exception as exc:
        logger.warning("[kap.store] upsert_company_profile failed for %s: %s",
                       ticker, exc)
        return False


async def upsert_return_row(pool, disclosure_id: str, ticker: str,
                             anchor_price: Optional[float] = None,
                             anchor_date=None,
                             returns: Optional[Dict[str, Optional[float]]] = None,
                             price_source: str = "borsapy") -> bool:
    """
    Insert or update a kap_returns row for (disclosure_id, ticker).
    returns dict: {'1d': float|None, '3d': float|None, ...}
    Returns True on success.
    """
    r = returns or {}
    sql = """
        INSERT INTO kap_returns (
            disclosure_id, ticker,
            anchor_price, anchor_date,
            return_1d, return_3d, return_5d, return_10d, return_30d,
            price_source, last_computed, windows_complete
        ) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,NOW(),
            ($5::numeric IS NOT NULL AND $6::numeric IS NOT NULL AND $7::numeric IS NOT NULL
             AND $8::numeric IS NOT NULL AND $9::numeric IS NOT NULL)
        )
        ON CONFLICT (disclosure_id, ticker) DO UPDATE SET
            anchor_price    = COALESCE(EXCLUDED.anchor_price, kap_returns.anchor_price),
            anchor_date     = COALESCE(EXCLUDED.anchor_date,  kap_returns.anchor_date),
            return_1d       = COALESCE(EXCLUDED.return_1d,    kap_returns.return_1d),
            return_3d       = COALESCE(EXCLUDED.return_3d,    kap_returns.return_3d),
            return_5d       = COALESCE(EXCLUDED.return_5d,    kap_returns.return_5d),
            return_10d      = COALESCE(EXCLUDED.return_10d,   kap_returns.return_10d),
            return_30d      = COALESCE(EXCLUDED.return_30d,   kap_returns.return_30d),
            price_source    = EXCLUDED.price_source,
            last_computed   = NOW(),
            windows_complete = (
                COALESCE(EXCLUDED.return_1d,  kap_returns.return_1d)  IS NOT NULL AND
                COALESCE(EXCLUDED.return_3d,  kap_returns.return_3d)  IS NOT NULL AND
                COALESCE(EXCLUDED.return_5d,  kap_returns.return_5d)  IS NOT NULL AND
                COALESCE(EXCLUDED.return_10d, kap_returns.return_10d) IS NOT NULL AND
                COALESCE(EXCLUDED.return_30d, kap_returns.return_30d) IS NOT NULL
            ),
            updated_at      = NOW()
    """
    try:
        await pool.execute(
            sql,
            disclosure_id, ticker,
            float(anchor_price) if anchor_price is not None else None,
            anchor_date,
            _f(r.get("1d")), _f(r.get("3d")), _f(r.get("5d")),
            _f(r.get("10d")), _f(r.get("30d")),
            price_source,
        )
        return True
    except Exception as exc:
        logger.warning("[kap.store] upsert_return_row failed for %s/%s: %s",
                       disclosure_id, ticker, exc)
        return False


def _f(v) -> Optional[float]:
    """Safe float cast; None passthrough."""
    return float(v) if v is not None else None


# ---------------------------------------------------------------------------
# Read helpers (used by classifier sibling + dashboard)
# ---------------------------------------------------------------------------

async def get_recent_disclosures(pool, limit: int = 50,
                                  ticker: Optional[str] = None,
                                  disclosure_type: Optional[str] = None) -> List[dict]:
    """
    Return the most recent KAP disclosures.
    Optionally filter by ticker or disclosure_type.
    Returns [] on DB error.
    """
    where_clauses = []
    params: list = []
    if ticker:
        params.append(ticker)
        where_clauses.append(f"ticker = ${len(params)}")
    if disclosure_type:
        params.append(disclosure_type)
        where_clauses.append(f"disclosure_type = ${len(params)}")

    where = ("WHERE " + " AND ".join(where_clauses)) if where_clauses else ""
    params.append(limit)
    sql = f"""
        SELECT id, disclosure_id, ticker, tickers, company_name, subject,
               disclosure_type, summary, url, disclosed_at, source, fetched_at
        FROM kap_disclosures
        {where}
        ORDER BY disclosed_at DESC
        LIMIT ${len(params)}
    """
    try:
        rows = await pool.fetch(sql, *params)
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("[kap.store] get_recent_disclosures failed: %s", exc)
        return []


async def get_recent_classified_for_ticker(pool, ticker: str,
                                            lookback_days: int = 7,
                                            limit: int = 5) -> List[dict]:
    """
    Return the most recent CLASSIFIED disclosures for a single ticker within
    the lookback window. Joins kap_disclosures to kap_classifications.

    Used by AdviceEngine._overlay_kap_context to surface a recent-disclosure
    context block on BIST advice. base_polarity is a documented PRIOR — callers
    must NOT treat it as a quantitative impact estimate.

    Returns [] on DB error or if the classifications table is absent.
    """
    if not ticker:
        return []
    bare = str(ticker).upper().replace(".IS", "")
    sql = """
        SELECT d.disclosure_id, d.ticker, d.company_name, d.subject,
               d.disclosed_at, d.url,
               c.event_type, c.base_polarity, c.classifier_stage,
               c.confidence, c.classified_at
        FROM kap_disclosures d
        JOIN kap_classifications c ON c.disclosure_id = d.id
        WHERE UPPER(d.ticker) = $1
          AND d.disclosed_at > NOW() - ($2 || ' days')::INTERVAL
        ORDER BY d.disclosed_at DESC
        LIMIT $3
    """
    try:
        rows = await pool.fetch(sql, bare, str(int(lookback_days)), int(limit))
        return [dict(r) for r in rows]
    except Exception as exc:
        err_str = str(exc).lower()
        if "kap_classifications" in err_str or "does not exist" in err_str:
            logger.debug(
                "[kap.store] get_recent_classified_for_ticker: classifications "
                "table absent (migration 063 pending). Returning []."
            )
            return []
        logger.warning(
            "[kap.store] get_recent_classified_for_ticker failed for %s: %s",
            ticker, exc,
        )
        return []


async def get_unclassified(pool, limit: int = 100) -> List[dict]:
    """
    Return disclosures that have no classification in kap_classifications.
    If migration 063 (kap_classifications table) has not been run, falls back
    to returning all recent disclosures (classifier table absence = degrade).
    """
    sql_with_join = """
        SELECT d.id, d.disclosure_id, d.ticker, d.tickers, d.company_name,
               d.subject, d.disclosure_type, d.summary, d.full_text,
               d.url, d.disclosed_at, d.source
        FROM kap_disclosures d
        LEFT JOIN kap_classifications c ON c.disclosure_id = d.id
        WHERE c.disclosure_id IS NULL
        ORDER BY d.disclosed_at DESC
        LIMIT $1
    """
    sql_fallback = """
        SELECT id, disclosure_id, ticker, tickers, company_name,
               subject, disclosure_type, summary, full_text,
               url, disclosed_at, source
        FROM kap_disclosures
        ORDER BY disclosed_at DESC
        LIMIT $1
    """
    try:
        rows = await pool.fetch(sql_with_join, limit)
        return [dict(r) for r in rows]
    except Exception as exc:
        err_str = str(exc).lower()
        if "kap_classifications" in err_str or "does not exist" in err_str:
            # Migration 063 not yet applied -- degrade
            logger.info(
                "[kap.store] kap_classifications table not found (migration 063 pending). "
                "get_unclassified() falling back to all recent disclosures."
            )
            try:
                rows = await pool.fetch(sql_fallback, limit)
                return [dict(r) for r in rows]
            except Exception as exc2:
                logger.warning("[kap.store] get_unclassified fallback failed: %s", exc2)
                return []
        logger.warning("[kap.store] get_unclassified failed: %s", exc)
        return []


async def store_classification(pool, disclosure_id: str, event_type: str,
                                sentiment: Optional[str] = None,
                                confidence: Optional[float] = None,
                                classifier_model: Optional[str] = None,
                                extra: Optional[dict] = None) -> bool:
    """
    Write a classification result to kap_classifications (migration 063 table).
    Returns True on success.
    Returns False with a WARNING log if the table does not exist (migration 063
    not yet applied -- sibling classifier must run its migration first).

    This hook is here so the classifier agent can import from one place:
      from modules.advisor.core.kap.kap_store import store_classification

    Column mapping (migration 063 kap_classifications schema):
      disclosure_id    BIGINT  <- kap_disclosures.id (numeric PK, NOT the KAP index)
      base_polarity            <- sentiment kwarg (taxonomy BasePolarity value)
      classifier_stage         <- classifier_model kwarg ('rule'|'llm'|'unclassified')
    """
    import json
    sql = """
        INSERT INTO kap_classifications (
            disclosure_id, event_type, base_polarity, params,
            classifier_stage, confidence, raw_subject, extra,
            classified_at, updated_at
        ) VALUES ($1, $2, $3, $4::jsonb, $5, $6, $7, $8::jsonb, NOW(), NOW())
        ON CONFLICT (disclosure_id) DO UPDATE SET
            event_type       = EXCLUDED.event_type,
            base_polarity    = EXCLUDED.base_polarity,
            params           = EXCLUDED.params,
            classifier_stage = EXCLUDED.classifier_stage,
            confidence       = EXCLUDED.confidence,
            raw_subject      = EXCLUDED.raw_subject,
            extra            = EXCLUDED.extra,
            updated_at       = NOW()
    """
    extra_d = dict(extra or {})
    params = extra_d.pop("params", {}) or {}
    raw_subject = str(extra_d.pop("raw_subject", "") or "")
    try:
        await pool.execute(
            sql,
            int(disclosure_id),
            str(event_type),
            str(sentiment) if sentiment else "NEUTRAL",
            json.dumps(params),
            str(classifier_model) if classifier_model else "unclassified",
            float(confidence) if confidence is not None else 0.0,
            raw_subject,
            json.dumps(extra_d),
        )
        return True
    except Exception as exc:
        # Precisely distinguish a MISSING table from other DB errors. The old
        # heuristic ("kap_classifications" in str(exc)) mis-reported FK
        # violations / column mismatches as "table not found", which sent the
        # operator chasing migration 063 when the real cause was different.
        sqlstate = getattr(exc, "sqlstate", None)
        cls = type(exc).__name__
        if sqlstate == "42P01" or cls == "UndefinedTableError":   # undefined_table
            logger.warning(
                "[kap.store] kap_classifications table genuinely missing "
                "(SQLSTATE 42P01) -- run migrations (063/073). "
                "store_classification(%s) skipped.", disclosure_id
            )
        elif sqlstate == "23503" or cls == "ForeignKeyViolationError":  # fk_violation
            logger.warning(
                "[kap.store] store_classification(%s) FK violation: event_type=%r "
                "is not present in kap_event_taxonomy (seed missing? run migration "
                "063/069). %s", disclosure_id, event_type, exc
            )
        else:
            logger.warning(
                "[kap.store] store_classification(%s) failed [%s/%s]: %s",
                disclosure_id, cls, sqlstate, exc
            )
        return False


async def get_disclosures_needing_returns(pool, max_age_days: int = 45) -> List[dict]:
    """
    Return disclosures where kap_returns is incomplete or missing,
    and the disclosure is old enough for at least the 1d window to mature.
    Used by forward_return_accumulator.py.
    """
    sql = """
        SELECT d.disclosure_id, d.ticker, d.tickers, d.disclosed_at
        FROM kap_disclosures d
        LEFT JOIN kap_returns r ON r.disclosure_id = d.disclosure_id
            AND r.ticker = COALESCE(d.ticker, d.ticker)
        WHERE
            d.ticker IS NOT NULL
            AND d.disclosed_at > NOW() - ($1 || ' days')::INTERVAL
            AND d.disclosed_at < NOW() - INTERVAL '1 day'
            AND (r.windows_complete IS NULL OR r.windows_complete = FALSE)
        ORDER BY d.disclosed_at ASC
        LIMIT 500
    """
    try:
        rows = await pool.fetch(sql, str(max_age_days))
        return [dict(r) for r in rows]
    except Exception as exc:
        logger.warning("[kap.store] get_disclosures_needing_returns failed: %s", exc)
        return []
