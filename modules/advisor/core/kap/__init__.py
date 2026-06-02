"""
kap — KAP (Kamuyu Aydınlatma Platformu) disclosure processing package.

Exports:
  classify(disclosure) -> ClassificationResult   (two-stage classifier)
  classify_batch(disclosures) -> Iterator[ClassificationResult]
  TAXONOMY                                        (event-type reference dict)
  KapEventType                                    (enum of all known event types)
  BasePolarity                                    (enum: STRONG_POSITIVE .. VERY_NEGATIVE)
  ClassificationResult                            (result dataclass)
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
