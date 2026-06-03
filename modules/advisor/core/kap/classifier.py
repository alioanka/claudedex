"""
classifier.py — Two-stage KAP disclosure event classifier.

Architecture
------------
STAGE 1 — RULE-BASED (deterministic)
  For every TaxonomyEntry in TAXONOMY, match Turkish trigger patterns
  (regex, case-insensitive) against the combined subject + disclosure text.
  The highest-priority matching entry wins.  If exactly one entry matches
  at any priority level, it is selected.  If multiple entries match at the
  same priority, we pick the one with the most trigger pattern hits (most
  specific match).  Confidence for a rule match is always RULE_CONFIDENCE
  (0.90) — high but not 1.0 because Turkish morphology can produce false
  positives in edge cases.

STAGE 2 — LLM FALLBACK (Anthropic / OpenAI, lazy, fail-soft)
  Only reached when Stage 1 produces no confident match.  The LLM receives:
    - Original subject line (sanitized, max 256 chars)
    - First 1000 chars of disclosure text (sanitized)
    - The full list of valid KapEventType enum values + their Turkish triggers
      (to give the LLM a taxonomy map rather than letting it free-associate).
  LLM returns: event_type (one of the enum values or "UNCLASSIFIED") and
  optional structured params extracted from the text.
  LLM CLASSIFIES + EXTRACTS; it does NOT decide market impact.
  Fail-soft: any LLM error -> UNCLASSIFIED.

Output — ClassificationResult
  event_type     : KapEventType
  base_polarity  : BasePolarity  (from TAXONOMY; NEUTRAL for UNCLASSIFIED)
  params         : dict          (bonus_ratio, dividend_per_share, etc.)
  classifier_stage: "rule" | "llm" | "unclassified"
  confidence     : float in [0.0, 1.0]
  raw_subject    : str  (preserved for UNCLASSIFIED cases)

HONESTY NOTE
------------
ClassificationResult carries event_type + base_polarity ONLY.
There is NO numeric "impact score" and NO percentage confidence beyond
classifier certainty.  Base polarity is a documented prior (see taxonomy.py),
not a prediction.  Quantitative impact estimates require historical return
statistics which accumulate over months in kap_returns — they do not exist yet.

Usage
-----
    from modules.advisor.core.kap.classifier import classify, classify_batch

    result = classify(disclosure)              # single disclosure dict
    results = classify_batch(disclosures, config=config)   # batch + historical backfill

The `disclosure` dict must contain at minimum:
    subject  : str   (KAP disclosure subject line — Turkish)
    text     : str   (Full disclosure body — may be empty string)
    id       : int | str  (disclosure primary key — for logging only)

Optional keys used:
    disclosure_date : str  (ISO timestamp — copied to result.extra)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Optional

from modules.advisor.core.kap.taxonomy import (
    TAXONOMY,
    BasePolarity,
    KapEventType,
    TaxonomyEntry,
)

logger = logging.getLogger("advisor.kap.classifier")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

RULE_CONFIDENCE      = 0.90   # confidence assigned to a rule-based match
LLM_CONFIDENCE       = 0.65   # confidence assigned to an LLM match
UNCLASSIFIED_CONF    = 0.0    # confidence for UNCLASSIFIED

_MAX_SUBJECT_LEN     = 256
_MAX_TEXT_LEN        = 1000   # chars sent to LLM
_MAX_PARAM_STR       = 128    # max param value length (prevent prompt injection)

# Param extraction regexes (applied to combined subject+text after event type known)
_PARAM_PATTERNS: dict[str, re.Pattern] = {
    # bonus_ratio: e.g. "%50 bedelsiz", "bedelsiz artış oranı %25"
    "bonus_ratio": re.compile(
        r"%\s*([\d]+(?:[.,]\d+)?)\s*(?:bedelsiz|oran)",
        re.IGNORECASE | re.UNICODE,
    ),
    # dividend_per_share: "hisse başına X TL", "pay başına X TL kar payı"
    "dividend_per_share": re.compile(
        r"(?:hisse|pay)\s+ba[şs][ıi]na\s+([\d]+(?:[.,]\d+)?)\s*(?:TL|tl|lira)",
        re.IGNORECASE | re.UNICODE,
    ),
    # contract_value: "500 milyon TL", "2 milyar TL", "USD 50 million"
    "contract_value": re.compile(
        r"([\d]+(?:[.,]\d+)?)\s*(milyon|milyar|million|billion)?\s*(TL|USD|EUR|tl|usd|eur)?",
        re.IGNORECASE | re.UNICODE,
    ),
    # amount_tl: "toplam X TL"
    "amount_tl": re.compile(
        r"([\d]+(?:[.,]\d+)?)\s*(milyon|milyar)?\s*TL\b",
        re.IGNORECASE | re.UNICODE,
    ),
    # amount_usd
    "amount_usd": re.compile(
        r"USD\s*([\d]+(?:[.,]\d+)?)\s*(milyon|milyar|million|billion)?",
        re.IGNORECASE | re.UNICODE,
    ),
    # pct_stake: "% 5,24", "5.24%", "yüzde 10"
    "pct_stake": re.compile(
        r"(?:%\s*|yüzde\s*)([\d]+(?:[.,]\d+)?)\s*(?:pay|hisse|oran|oranında)?",
        re.IGNORECASE | re.UNICODE,
    ),
    # rating_from / rating_to: "AA- notundan BB+ notuna"
    "rating_from": re.compile(
        r"(AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|D)\s*(?:notundan|den|dan)",
        re.IGNORECASE | re.UNICODE,
    ),
    "rating_to": re.compile(
        r"(?:notuna|'ya|'e|'e yükseltildi|'e düşürüldü)\s*(AA[+-]?|A[+-]?|BBB[+-]?|BB[+-]?|B[+-]?|CCC[+-]?|CC|C|D)",
        re.IGNORECASE | re.UNICODE,
    ),
    # exec_name: heuristic — proper-noun-like sequence after genel müdür / yönetici
    "exec_name": re.compile(
        r"(?:genel\s+müdür|yönetici|ceo|icra\s+ba[şs]kan[ıi])\s+([A-ZÇĞİÖŞÜ][a-zçğışöşü]+(?:\s+[A-ZÇĞİÖŞÜ][a-zçğışöşü]+)*)",
        re.UNICODE,
    ),
    "exec_role": re.compile(
        r"(genel\s+müdür|yönetim\s+kurulu\s+ba[şs]kan[ıi]|cfo|coo|cto|icra\s+ba[şs]kan[ıi])",
        re.IGNORECASE | re.UNICODE,
    ),
}

# LLM taxonomy hint: list of (event_type_name, sample_triggers) for prompt
_LLM_TAXONOMY_HINT = "\n".join(
    f"- {et.value}: {', '.join(entry.turkish_triggers[:2])}"
    for et, entry in TAXONOMY.items()
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class ClassificationResult:
    """
    Output of the two-stage classifier.

    Fields
    ------
    event_type       : KapEventType enum value.
    base_polarity    : BasePolarity prior from taxonomy.
                       NEUTRAL for UNCLASSIFIED.
    params           : Structured extracted params (e.g. bonus_ratio,
                       dividend_per_share).  Empty dict if none found.
    classifier_stage : "rule" | "llm" | "unclassified"
    confidence       : float [0.0, 1.0] — classifier certainty only.
                       NOT a market-impact score.
    raw_subject      : Original subject line (preserved for UNCLASSIFIED).
    disclosure_id    : Disclosure primary key (for DB write).
    extra            : Dict for any additional metadata.
    """
    event_type        : KapEventType
    base_polarity     : BasePolarity
    params            : Dict[str, Any]     = field(default_factory=dict)
    classifier_stage  : str                = "unclassified"
    confidence        : float              = 0.0
    raw_subject       : str                = ""
    disclosure_id     : Optional[Any]      = None
    extra             : Dict[str, Any]     = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "disclosure_id"    : self.disclosure_id,
            "event_type"       : self.event_type.value,
            "base_polarity"    : self.base_polarity.value,
            "params"           : self.params,
            "classifier_stage" : self.classifier_stage,
            "confidence"       : self.confidence,
            "raw_subject"      : self.raw_subject,
            "extra"            : self.extra,
        }


def _unclassified(disclosure: dict) -> ClassificationResult:
    return ClassificationResult(
        event_type=KapEventType.UNCLASSIFIED,
        base_polarity=BasePolarity.NEUTRAL,
        params={},
        classifier_stage="unclassified",
        confidence=UNCLASSIFIED_CONF,
        raw_subject=str(disclosure.get("subject", ""))[:_MAX_SUBJECT_LEN],
        disclosure_id=disclosure.get("id"),
    )


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------

def _sanitize(text: str, max_len: int) -> str:
    """Strip control / non-printable characters and truncate."""
    cleaned = "".join(c for c in text if c.isprintable())
    return cleaned[:max_len]


def _combine_text(disclosure: dict) -> str:
    """
    Combine subject + text into a single string for pattern matching.
    Subject is repeated at the start (higher salience for short subjects).
    """
    subject = _sanitize(str(disclosure.get("subject", "")), _MAX_SUBJECT_LEN)
    body    = _sanitize(str(disclosure.get("text", "")), _MAX_TEXT_LEN * 3)
    return f"{subject}\n{subject}\n{body}"


# ---------------------------------------------------------------------------
# Param extraction
# ---------------------------------------------------------------------------

def _extract_params(text: str, param_keys: List[str]) -> Dict[str, Any]:
    """
    Extract structured params from text for the specified keys.
    Returns a dict with only the keys that matched.
    Values are truncated strings (raw regex matches — callers normalise).
    """
    out: Dict[str, Any] = {}
    for key in param_keys:
        pattern = _PARAM_PATTERNS.get(key)
        if pattern is None:
            continue
        m = pattern.search(text)
        if m:
            raw = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
            out[key] = str(raw)[:_MAX_PARAM_STR]
    return out


# ---------------------------------------------------------------------------
# Stage 1: Rule-based classifier
# ---------------------------------------------------------------------------

def _rule_stage(text: str) -> Optional[tuple[TaxonomyEntry, int]]:
    """
    Match text against all taxonomy entries.

    Returns (best_entry, hit_count) or None if no entry matches.
    Selection logic:
      1. Collect all matching entries.
      2. Keep only entries at the maximum priority.
      3. Among those, pick the one with the most trigger pattern hits
         (specificity tiebreak).
    """
    matches: List[tuple[TaxonomyEntry, int]] = []

    for entry in TAXONOMY.values():
        if not entry.matches(text):
            continue
        # Count how many distinct trigger patterns matched (specificity)
        entry._compile()
        hit_count = sum(
            1 for p in entry._compiled_triggers if p.search(text)
        )
        matches.append((entry, hit_count))

    if not matches:
        return None

    max_priority = max(e.priority for e, _ in matches)
    top = [(e, h) for e, h in matches if e.priority == max_priority]
    # Best = most trigger hits
    best = max(top, key=lambda x: x[1])
    return best


# ---------------------------------------------------------------------------
# Stage 2: LLM fallback
# ---------------------------------------------------------------------------

def _build_llm_prompt(subject: str, body: str) -> str:
    """Build the classification prompt sent to the LLM."""
    return (
        "You are a Turkish capital-markets disclosure classifier for BIST (Borsa Istanbul).\n\n"
        "Given the KAP disclosure below, identify the single best matching event type "
        "from the taxonomy list.  Return a JSON object with exactly these fields:\n"
        '  "event_type": one of the taxonomy values below (or "UNCLASSIFIED" if none fits),\n'
        '  "params": a JSON object with any structured params you can extract '
        "(bonus_ratio, dividend_per_share, contract_value, amount_tl, amount_usd, "
        "pct_stake, rating_from, rating_to, exec_name, exec_role) — omit fields you cannot find.\n\n"
        "IMPORTANT: You CLASSIFY the event type and EXTRACT params only.  "
        "Do NOT assign a numeric impact score.  Do NOT make polarity judgements.\n\n"
        "TENDER RULE (critical — do not conflate):\n"
        "  KAP tender disclosures (İhale Süreci / Sonucu) cover the WHOLE lifecycle.\n"
        "  - TENDER_BID  = the company PARTICIPATED in / SUBMITTED A BID to a tender, "
        "outcome UNKNOWN. Cues: 'ihaleye katıldı/katılmıştır', 'teklif verilmesi/verme', "
        "'1. oturuma katıldı', 'İhaleye Teklif Verme Tarihi', and the İhale Sonucu / "
        "İhale Bedeli (result/award amount) fields are EMPTY or '-'.\n"
        "  - TENDER_WIN  = the company WON / WAS AWARDED the tender. ONLY use TENDER_WIN "
        "when there is EXPLICIT award evidence: 'ihale kazanıldı/kazanılmıştır', "
        "'ihaleyi kazandı', 'ihale şirketimiz üzerinde kalmıştır', 'uhdemizde kaldı', "
        "or 'sonuçlanmıştır' together with a populated İhale Bedeli / award amount.\n"
        "  If you only see participation/bid language and no award is stated, you MUST "
        "return TENDER_BID (NOT TENDER_WIN). When unsure, prefer TENDER_BID.\n\n"
        f"TAXONOMY:\n{_LLM_TAXONOMY_HINT}\n\n"
        f"SUBJECT: {subject}\n\n"
        f"DISCLOSURE (first 1000 chars):\n{body}\n\n"
        'Return ONLY a valid JSON object, no other text.  Example: {"event_type": "DIVIDEND", "params": {"dividend_per_share": "1.50"}}'
    )


def _call_llm_classify(
    subject: str,
    body: str,
    config: dict,
    log: logging.Logger,
    disclosure_id: Any,
) -> Optional[tuple[KapEventType, Dict[str, Any]]]:
    """
    Call Anthropic (primary) or OpenAI (fallback) to classify one disclosure.
    Returns (event_type, params) or None on any failure.

    Key resolution follows rationale_helper pattern:
      Anthropic: config["advisor_anthropic_api_key"] or os.environ["ADVISOR_ANTHROPIC_API_KEY"]
      OpenAI:    config["advisor_openai_api_key"]    or os.environ["ADVISOR_OPENAI_API_KEY"]
    Both keys optional — if absent the respective call is skipped.
    """
    import os

    prompt = _build_llm_prompt(subject, body)

    # ---- Anthropic ----
    anthropic_key = (
        config.get("advisor_anthropic_api_key") or
        os.getenv("ADVISOR_ANTHROPIC_API_KEY") or ""
    )
    if anthropic_key:
        model_id = config.get("advisor_anthropic_model", "claude-opus-4-8")
        try:
            import anthropic  # lazy import

            client = anthropic.Anthropic(api_key=anthropic_key)
            msg = client.messages.create(
                model=model_id,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            raw = msg.content[0].text if msg.content else ""
            return _parse_llm_response(raw, log, disclosure_id, "anthropic")
        except Exception as exc:
            exc_str = str(exc)
            log.warning(
                "[kap.classifier] Anthropic LLM classify failed for disclosure %s "
                "(model=%s): %s. Trying OpenAI fallback.",
                disclosure_id, model_id, exc_str[:200],
            )
            if "404" in exc_str or "not_found" in exc_str.lower():
                log.warning(
                    "[kap.classifier] ANTHROPIC MODEL NOT FOUND: %s. "
                    "Update advisor_anthropic_model in advisor_config DB table.",
                    model_id,
                )

    # ---- OpenAI fallback ----
    openai_key = (
        config.get("advisor_openai_api_key") or
        os.getenv("ADVISOR_OPENAI_API_KEY") or ""
    )
    if openai_key:
        model_id = config.get("advisor_openai_model", "gpt-4o")
        try:
            import urllib.request
            import urllib.error
            import json as _json

            payload = {
                "model": model_id,
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 512,
                "temperature": 0.0,
            }
            req = urllib.request.Request(
                "https://api.openai.com/v1/chat/completions",
                data=_json.dumps(payload).encode(),
                headers={
                    "Authorization": f"Bearer {openai_key}",
                    "Content-Type": "application/json",
                },
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = _json.loads(resp.read().decode())
            raw = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            return _parse_llm_response(raw, log, disclosure_id, "openai")
        except Exception as exc:
            log.warning(
                "[kap.classifier] OpenAI LLM classify failed for disclosure %s "
                "(model=%s): %s. Falling back to UNCLASSIFIED.",
                disclosure_id, model_id, str(exc)[:200],
            )

    return None


def _parse_llm_response(
    raw: str,
    log: logging.Logger,
    disclosure_id: Any,
    provider: str,
) -> Optional[tuple[KapEventType, Dict[str, Any]]]:
    """
    Parse LLM JSON response.  Returns (KapEventType, params) or None.
    Robust to JSON embedded in markdown code fences.
    """
    # Strip markdown fences if present
    clean = raw.strip()
    for fence in ("```json", "```"):
        if clean.startswith(fence):
            clean = clean[len(fence):]
    if clean.endswith("```"):
        clean = clean[:-3]
    clean = clean.strip()

    try:
        obj = json.loads(clean)
    except json.JSONDecodeError:
        # Try to extract first {...} block
        m = re.search(r"\{.*\}", clean, re.DOTALL)
        if not m:
            log.warning(
                "[kap.classifier] %s: Could not parse LLM response for %s: %r",
                provider, disclosure_id, clean[:200],
            )
            return None
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return None

    et_raw = str(obj.get("event_type", "UNCLASSIFIED")).upper().strip()
    try:
        et = KapEventType(et_raw)
    except ValueError:
        log.warning(
            "[kap.classifier] %s: Unknown event_type '%s' from LLM for disclosure %s. "
            "Using UNCLASSIFIED.",
            provider, et_raw, disclosure_id,
        )
        return None

    params = {}
    raw_params = obj.get("params", {})
    if isinstance(raw_params, dict):
        for k, v in raw_params.items():
            if isinstance(v, (str, int, float)):
                params[str(k)[:64]] = str(v)[:_MAX_PARAM_STR]

    return et, params


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def classify(
    disclosure: dict,
    config: Optional[dict] = None,
    caller_logger: Optional[logging.Logger] = None,
) -> ClassificationResult:
    """
    Classify a single KAP disclosure.

    Parameters
    ----------
    disclosure : dict with keys:
        id      : int | str   — disclosure primary key (for logging)
        subject : str         — KAP subject line (Turkish)
        text    : str         — full disclosure text (may be empty)
    config : advisor config dict.  Used for LLM key resolution in Stage 2.
             Pass None to disable LLM fallback (rule-only mode).
    caller_logger : optional Logger from the calling context.

    Returns
    -------
    ClassificationResult — never raises.
    """
    log = caller_logger or logger
    cfg = config or {}
    disc_id = disclosure.get("id", "?")

    combined = _combine_text(disclosure)
    subject_clean = _sanitize(str(disclosure.get("subject", "")), _MAX_SUBJECT_LEN)
    body_clean    = _sanitize(str(disclosure.get("text", "")), _MAX_TEXT_LEN)

    # ----------------------------------------------------------------
    # Stage 1 — rule-based
    # ----------------------------------------------------------------
    rule_result = _rule_stage(combined)
    if rule_result is not None:
        entry, hit_count = rule_result
        params = _extract_params(combined, entry.param_keys)
        log.debug(
            "[kap.classifier] Rule match: disclosure=%s -> %s (hits=%d, priority=%d)",
            disc_id, entry.event_type.value, hit_count, entry.priority,
        )
        return ClassificationResult(
            event_type=entry.event_type,
            base_polarity=entry.base_polarity,
            params=params,
            classifier_stage="rule",
            confidence=RULE_CONFIDENCE,
            raw_subject=subject_clean,
            disclosure_id=disc_id,
            extra={"trigger_hits": hit_count},
        )

    # ----------------------------------------------------------------
    # Stage 2 — LLM fallback
    # ----------------------------------------------------------------
    if not cfg:
        # No config supplied — can't reach LLM
        log.debug(
            "[kap.classifier] No rule match and no config for LLM fallback: "
            "disclosure=%s -> UNCLASSIFIED",
            disc_id,
        )
        return _unclassified(disclosure)

    log.debug(
        "[kap.classifier] No rule match for disclosure=%s; trying LLM fallback.",
        disc_id,
    )
    llm_result = _call_llm_classify(subject_clean, body_clean, cfg, log, disc_id)
    if llm_result is None:
        return _unclassified(disclosure)

    et, params = llm_result
    if et == KapEventType.UNCLASSIFIED:
        return _unclassified(disclosure)

    taxonomy_entry = TAXONOMY.get(et)
    polarity = taxonomy_entry.base_polarity if taxonomy_entry else BasePolarity.NEUTRAL
    log.debug(
        "[kap.classifier] LLM match: disclosure=%s -> %s",
        disc_id, et.value,
    )
    return ClassificationResult(
        event_type=et,
        base_polarity=polarity,
        params=params,
        classifier_stage="llm",
        confidence=LLM_CONFIDENCE,
        raw_subject=subject_clean,
        disclosure_id=disc_id,
    )


def classify_batch(
    disclosures: List[dict],
    config: Optional[dict] = None,
    caller_logger: Optional[logging.Logger] = None,
) -> Iterator[ClassificationResult]:
    """
    Classify a batch of disclosures (generator — yields one result per disclosure).

    Suitable for historical backfill (the sibling kap_listener produces a
    list of unclassified disclosures; pass them here).  For live ingest,
    call `classify()` once per incoming disclosure.

    Yields ClassificationResult objects in the same order as the input list.
    Never raises — individual failures yield UNCLASSIFIED.

    Parameters
    ----------
    disclosures : List of disclosure dicts (same schema as classify()).
    config      : Advisor config dict.  Pass None for rule-only mode.
    caller_logger : optional Logger.
    """
    log = caller_logger or logger
    for disc in disclosures:
        try:
            yield classify(disc, config=config, caller_logger=log)
        except Exception as exc:
            log.warning(
                "[kap.classifier] Unexpected error classifying disclosure %s: %s. "
                "Yielding UNCLASSIFIED.",
                disc.get("id", "?"), str(exc)[:200],
            )
            yield _unclassified(disc)
