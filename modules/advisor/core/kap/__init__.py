"""
KAP (Kamuyu Aydinlatma Platformu / Public Disclosure Platform) package.

ADVICE-ONLY data pipeline -- no order execution, no trading module imports.

Legal / ToS note
----------------
KAP is operated by MKK (Central Securities Depository of Turkey) and Borsa Istanbul.
Public disclosures are legally public information (SPK regulation requires publication).
This implementation uses PyKap (MIT) over KAP's public JSON API where available,
with a polite rate-limited HTTP fallback to kap.org.tr public endpoints. A licensed
commercial KAP/BIST feed (Matriks/MKK) is the recommended production upgrade.

Package structure
-----------------
  taxonomy.py                   -- Turkish KAP event taxonomy (24 types + guard asserts)
  classifier.py                 -- two-stage event classifier (rule + LLM fallback)
  kap_store.py                  -- DB read/write helpers (read API for classifier)
  kap_listener.py               -- periodic poller for new disclosures
  kap_archive_crawler.py        -- incremental historical backfill helper
  forward_return_accumulator.py -- nightly job: fill T+N returns for matured windows

Migrations: 062 (ingestion tables + config), 063 (classifications + taxonomy ref).
Operator enable flag: advisor_kap_enabled='true' in config_settings (default false).
"""

from modules.advisor.core.kap.taxonomy import KapEventType, BasePolarity, TAXONOMY
from modules.advisor.core.kap.classifier import ClassificationResult, classify, classify_batch

__all__ = [
    "KapEventType",
    "BasePolarity",
    "TAXONOMY",
    "ClassificationResult",
    "classify",
    "classify_batch",
]
