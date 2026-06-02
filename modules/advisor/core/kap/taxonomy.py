"""
taxonomy.py — KAP disclosure event taxonomy for BIST (Borsa Istanbul).

This is the core asset of the KAP Event Classifier.  It defines:

  KapEventType  — enum of all known event types (22 high-impact + UNCLASSIFIED).
  BasePolarity  — enum of market-impact polarity priors.
  TaxonomyEntry — dataclass for one event type's full definition.
  TAXONOMY      — dict[KapEventType, TaxonomyEntry] — the authoritative reference.

DESIGN INTENT
-------------
The taxonomy is the hard 80% of Turkish NLP here.  The same Turkish
phrase can represent OPPOSITE signals:

  "Bedelsiz sermaye artırımı" -> BONUS_ISSUE  -> market-cap-unchanged -> POSITIVE
  "Bedelli sermaye artırımı"  -> RIGHTS_ISSUE -> dilution cash-call    -> NEGATIVE

These distinctions are hard-coded and deterministic.  The LLM is only
used for ambiguous tail events that don't match any trigger phrase.

BASE POLARITY IS NOT A PREDICTION
----------------------------------
BasePolarity is a documented PRIOR drawn from academic event-study
literature on BIST and similar emerging markets.  It is NOT a
quantitative impact score.  No historical-return statistics exist yet —
they accumulate over months through the kap_returns table.  Consumers
that want a measured impact estimate must wait for adequate sample sizes.

EXTENSIBILITY
-------------
To add a new event type:
  1. Add an entry to KapEventType.
  2. Add a TaxonomyEntry to TAXONOMY with Turkish trigger phrases.
  3. The rule-based stage in classifier.py picks it up automatically.
  4. Migration 063 re-seeds the kap_event_taxonomy table (idempotent upsert).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class BasePolarity(str, Enum):
    """
    Qualitative market-impact prior for a KAP event type.

    STRONG_POSITIVE  — typically a clear near-term price catalyst.
    POSITIVE         — mild / context-dependent positive.
    NEUTRAL          — informational; market-cap-neutral or ambiguous.
    NEGATIVE         — dilutive or operationally adverse.
    VERY_NEGATIVE    — severe distress signal (default, bankruptcy, fraud).
    """
    STRONG_POSITIVE = "STRONG_POSITIVE"
    POSITIVE        = "POSITIVE"
    NEUTRAL         = "NEUTRAL"
    NEGATIVE        = "NEGATIVE"
    VERY_NEGATIVE   = "VERY_NEGATIVE"


class KapEventType(str, Enum):
    """
    Enumeration of recognised KAP disclosure event types.

    Naming convention: upper-case English identifiers that are stable
    across Turkish regulatory wording changes.  The Turkish trigger
    phrases live in TaxonomyEntry.triggers — not in the enum names.
    """
    # Capital structure / equity events
    BONUS_ISSUE       = "BONUS_ISSUE"        # bedelsiz sermaye artırımı
    RIGHTS_ISSUE      = "RIGHTS_ISSUE"       # bedelli sermaye artırımı (dilution)
    SHARE_BUYBACK     = "SHARE_BUYBACK"      # pay/hisse geri alım programı
    CAPITAL_REDUCTION = "CAPITAL_REDUCTION"  # sermaye azaltımı

    # Income events
    DIVIDEND          = "DIVIDEND"           # temettü / kar payı

    # Business wins / commercial
    NEW_CONTRACT      = "NEW_CONTRACT"       # yeni iş / sözleşme / sipariş
    TENDER_WIN        = "TENDER_WIN"         # ihale kazanımı
    EXPORT_AGREEMENT  = "EXPORT_AGREEMENT"   # ihracat anlaşması / uluslararası sözleşme

    # Capacity / operations
    CAPACITY_INCREASE = "CAPACITY_INCREASE"  # kapasite artışı / genişletme yatırımı
    FACTORY_OPENING   = "FACTORY_OPENING"    # fabrika / tesis açılışı
    PRODUCTION_HALT   = "PRODUCTION_HALT"    # üretim durdurma / tesis kapanma

    # Corporate actions
    ACQUISITION       = "ACQUISITION"        # devralma / satın alma / birleşme
    PARTNERSHIP       = "PARTNERSHIP"        # ortaklık / iş birliği anlaşması

    # Management / governance
    CEO_CHANGE        = "CEO_CHANGE"         # genel müdür / yönetici atama veya istifa
    BOARD_MEETING     = "BOARD_MEETING"      # yönetim kurulu toplantısı
    GENERAL_ASSEMBLY  = "GENERAL_ASSEMBLY"   # genel kurul toplantısı

    # Investor / insider events
    INSIDER_PURCHASE  = "INSIDER_PURCHASE"   # yönetici / ortak pay alımı
    INSIDER_SALE      = "INSIDER_SALE"       # yönetici / ortak pay satışı

    # Regulatory / credit events
    SPK_INVESTIGATION = "SPK_INVESTIGATION"  # SPK soruşturma / inceleme
    CREDIT_RATING     = "CREDIT_RATING"      # kredi derecelendirme / not değişikliği
    LAWSUIT           = "LAWSUIT"            # dava / hukuki süreç

    # Financial results
    FINANCIAL_RESULTS = "FINANCIAL_RESULTS"  # finansal tablo / bilanço açıklaması
    GUIDANCE          = "GUIDANCE"           # beklenti / öngörü revizyonu

    # Distress
    DEFAULT           = "DEFAULT"            # konkordato / icra / iflas

    # Catch-all
    UNCLASSIFIED      = "UNCLASSIFIED"       # no rule matched; LLM also uncertain


# ---------------------------------------------------------------------------
# TaxonomyEntry
# ---------------------------------------------------------------------------

@dataclass
class TaxonomyEntry:
    """
    Full definition of one KAP event type.

    Attributes
    ----------
    event_type      : KapEventType enum value.
    turkish_triggers: List of Turkish regex patterns (compiled at import time).
                      Patterns are matched case-insensitively against the
                      combined subject + disclosure_text field.
                      Order within the list does not affect priority.
    base_polarity   : Qualitative prior — NOT a quantitative prediction.
    notes           : Human-readable explanation of the prior and any
                      important context (bedelsiz/bedelli distinction, etc.).
    param_keys      : Structured fields to extract when this event is
                      detected. The classifier stores these in params JSONB.
                      Possible values: bonus_ratio, dividend_per_share,
                      contract_value, rating_from, rating_to, exec_name,
                      exec_role, amount_tl, amount_usd, pct_stake.
    exclude_patterns: Patterns whose presence VETOES this entry's match
                      (used to distinguish bedelsiz from bedelli).
    priority        : Higher number wins when multiple entries match.
                      Most entries are priority 1; the bedelsiz/bedelli
                      pair uses 2 to override a generic CAPITAL_INCREASE
                      match at priority 1.
    """
    event_type       : KapEventType
    turkish_triggers : List[str]
    base_polarity    : BasePolarity
    notes            : str
    param_keys       : List[str]   = field(default_factory=list)
    exclude_patterns : List[str]   = field(default_factory=list)
    priority         : int         = 1

    # Compiled regex cache — populated by _compile() on first access.
    _compiled_triggers  : Optional[List[re.Pattern]] = field(default=None, init=False, repr=False, compare=False)
    _compiled_excludes  : Optional[List[re.Pattern]] = field(default=None, init=False, repr=False, compare=False)

    def _compile(self) -> None:
        if self._compiled_triggers is None:
            self._compiled_triggers = [
                re.compile(p, re.IGNORECASE | re.UNICODE)
                for p in self.turkish_triggers
            ]
        if self._compiled_excludes is None:
            self._compiled_excludes = [
                re.compile(p, re.IGNORECASE | re.UNICODE)
                for p in self.exclude_patterns
            ]

    def matches(self, text: str) -> bool:
        """
        Return True if any trigger matches AND no exclude pattern matches.
        Called against the concatenated subject + disclosure text.
        """
        self._compile()
        if not any(p.search(text) for p in self._compiled_triggers):
            return False
        if any(p.search(text) for p in self._compiled_excludes):
            return False
        return True


# ---------------------------------------------------------------------------
# TAXONOMY — the authoritative reference dict
# ---------------------------------------------------------------------------
#
# Key design decisions encoded here:
#
# 1. BONUS_ISSUE (bedelsiz) vs RIGHTS_ISSUE (bedelli)
#    The single most important distinction in BIST disclosures.
#    BONUS_ISSUE: bedelsiz = "free of charge" = existing shareholders get
#      new shares funded from retained earnings / revaluation reserves.
#      Market cap is unchanged; share count rises; price adjusts down.
#      Net effect: NEUTRAL to weakly POSITIVE (retail momentum, signals
#      healthy reserves, no cash drain).
#    RIGHTS_ISSUE: bedelli = "for a fee" = company sells new shares to
#      raise cash. Existing shareholders are diluted unless they subscribe.
#      Net effect: NEGATIVE to MIXED (depends on use-of-proceeds quality).
#
# 2. INSIDER_PURCHASE vs INSIDER_SALE
#    Purchase = bullish signal (management conviction).
#    Sale = bearish signal (can be liquidity-driven; weaker signal).
#
# 3. PRODUCTION_HALT and DEFAULT are VERY_NEGATIVE.
#    SPK_INVESTIGATION is VERY_NEGATIVE (regulatory overhang).
#
# 4. Priority 2 entries for bedelsiz/bedelli override any priority-1
#    generic capital-increase match.
#
# 5. UNCLASSIFIED is not in TAXONOMY — it's the output when nothing matches.

TAXONOMY: dict[KapEventType, TaxonomyEntry] = {

    # ------------------------------------------------------------------
    # BONUS_ISSUE — bedelsiz (free) capital increase
    # Priority 2 to beat generic capital-increase matches.
    # ------------------------------------------------------------------
    KapEventType.BONUS_ISSUE: TaxonomyEntry(
        event_type=KapEventType.BONUS_ISSUE,
        turkish_triggers=[
            r"bedelsiz\s+sermaye\s+art[ıi]r[ıi]m[ıi]",
            r"bedelsiz\s+pay\s+da[gğ][ıi]t[ıi]m[ıi]",
            r"bedelsiz\s+hisse",
            r"iç\s+kaynaktan\s+sermaye\s+art[ıi]r[ıi]m[ıi]",
            r"kar\s+pay[ıi]\s+sermayeye\s+ekleme",
        ],
        exclude_patterns=[
            r"bedelli",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Bedelsiz (free) capital increase: existing shareholders receive "
            "new shares funded from retained earnings or revaluation reserves. "
            "Market cap is mathematically unchanged; price adjusts down by the "
            "dilution factor.  Net effect is neutral to mildly positive due to "
            "retail momentum and the signal that reserves are sufficient. "
            "Do NOT confuse with bedelli (rights issue / cash call)."
        ),
        param_keys=["bonus_ratio"],
        priority=2,
    ),

    # ------------------------------------------------------------------
    # RIGHTS_ISSUE — bedelli (paid) capital increase
    # Priority 2 to beat generic capital-increase matches.
    # ------------------------------------------------------------------
    KapEventType.RIGHTS_ISSUE: TaxonomyEntry(
        event_type=KapEventType.RIGHTS_ISSUE,
        turkish_triggers=[
            r"bedelli\s+sermaye\s+art[ıi]r[ıi]m[ıi]",
            r"bedelli\s+pay\s+ihrac[ıi]",
            r"nakdi\s+sermaye\s+art[ıi]r[ıi]m[ıi]",
            r"rüçhan\s+hakkı",                 # pre-emptive rights
            r"rüçhan\s+hak\s+kullan[ıi]m[ıi]",
            r"d[ıi][şs]\s+kaynaktan\s+sermaye\s+art[ıi]r[ıi]m[ıi]",
        ],
        exclude_patterns=[
            r"bedelsiz",
        ],
        base_polarity=BasePolarity.NEGATIVE,
        notes=(
            "Bedelli (paid) capital increase: company issues new shares to raise "
            "cash.  Existing shareholders are diluted unless they exercise their "
            "pre-emptive rights (rüçhan hakkı). Effect depends on use-of-proceeds "
            "quality; the prior is NEGATIVE because dilution is certain and "
            "use-of-proceeds quality is uncertain at disclosure time. "
            "Do NOT confuse with bedelsiz (bonus issue / free shares)."
        ),
        param_keys=["amount_tl"],
        priority=2,
    ),

    # ------------------------------------------------------------------
    # DIVIDEND — cash distribution to shareholders
    # ------------------------------------------------------------------
    KapEventType.DIVIDEND: TaxonomyEntry(
        event_type=KapEventType.DIVIDEND,
        turkish_triggers=[
            r"temettü",
            r"kar\s+pay[ıi]\s+da[gğ][ıi]t[ıi]m[ıi]",
            r"kar\s+pay[ıi]\s+önerisi",
            r"kar\s+da[gğ][ıi]t[ıi]m\b",
            r"nakit\s+temettü",
            r"kar\s+pay[ıi]\s+oranı",
        ],
        exclude_patterns=[
            r"bedelsiz.*sermaye",
            r"kar\s+pay[ıi]\s+sermayeye",   # bonus issue via profit capitalisation
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Cash or in-kind dividend announcement.  Positive prior: signals "
            "profitability and management willingness to return capital. "
            "Size matters; a token dividend after a strong year has less impact "
            "than an unexpectedly large one."
        ),
        param_keys=["dividend_per_share"],
    ),

    # ------------------------------------------------------------------
    # NEW_CONTRACT — new significant commercial contract
    # ------------------------------------------------------------------
    KapEventType.NEW_CONTRACT: TaxonomyEntry(
        event_type=KapEventType.NEW_CONTRACT,
        turkish_triggers=[
            r"yeni\s+(önemli\s+)?sözle[şs]me",
            r"sipari[şs]\s+al[ıi]nd[ıi]",
            r"i[şs]\s+sözle[şs]mesi\s+imzaland[ıi]",
            r"çerçeve\s+sözle[şs]mesi\s+imzaland[ıi]",
            r"tedarik\s+sözle[şs]mesi",
            r"sat[ıi][şs]\s+sözle[şs]mesi\s+imzaland[ıi]",
            r"lisans\s+sözle[şs]mesi\s+imzaland[ıi]",
        ],
        exclude_patterns=[
            r"fesih",       # contract termination
            r"iptal",       # cancellation
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "New material commercial contract. Positive prior: adds revenue "
            "visibility. Size relative to market cap drives actual impact. "
            "Contract value param_key captures amount where disclosed."
        ),
        param_keys=["contract_value"],
    ),

    # ------------------------------------------------------------------
    # TENDER_WIN — public tender / government contract award
    # ------------------------------------------------------------------
    KapEventType.TENDER_WIN: TaxonomyEntry(
        event_type=KapEventType.TENDER_WIN,
        turkish_triggers=[
            r"ihale\s+kazan[ıi]ld[ıi]",
            r"ihale\s+sonucu",
            r"kamu\s+ihale",
            r"teklif\s+kabul",
        ],
        base_polarity=BasePolarity.STRONG_POSITIVE,
        notes=(
            "Government or public tender win. Strong positive prior: government "
            "contracts have lower counterparty risk and provide multi-year revenue "
            "visibility.  BIST construction, defence, and infrastructure sectors "
            "see especially strong reactions."
        ),
        param_keys=["contract_value"],
    ),

    # ------------------------------------------------------------------
    # EXPORT_AGREEMENT — international sales / export deal
    # ------------------------------------------------------------------
    KapEventType.EXPORT_AGREEMENT: TaxonomyEntry(
        event_type=KapEventType.EXPORT_AGREEMENT,
        turkish_triggers=[
            r"ihracat\s+sözle[şs]mesi",
            r"ihracat\s+anla[şs]mas[ıi]",
            r"uluslararas[ıi]\s+sözle[şs]me\s+imzaland[ıi]",
            r"yurt\s+d[ıi][şs][ıi]\s+sözle[şs]me",
            r"d[ıi][şs]\s+sat[ıi][şs]\s+anla[şs]mas[ıi]",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Export or international trade agreement. Positive: FX revenue "
            "diversification, especially valuable during TRY depreciation periods."
        ),
        param_keys=["contract_value"],
    ),

    # ------------------------------------------------------------------
    # CAPACITY_INCREASE — expansion investment / new production line
    # ------------------------------------------------------------------
    KapEventType.CAPACITY_INCREASE: TaxonomyEntry(
        event_type=KapEventType.CAPACITY_INCREASE,
        turkish_triggers=[
            r"kapasite\s+art[ıi][şs][ıi]",
            r"üretim\s+kapasitesi\s+art[ıi]r[ıi]ld[ıi]",
            r"yeni\s+üretim\s+hatt[ıi]",
            r"geni[şs]letme\s+yat[ıi]r[ıi]m[ıi]",
            r"yat[ıi]r[ıi]m\s+karar[ıi]\s+al[ıi]nd[ıi]",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Capacity expansion investment decision. Positive prior: signals "
            "growth confidence, but short-term capex impact can weigh on FCF. "
            "Duration of project matters for time-horizon alignment."
        ),
        param_keys=["amount_tl"],
    ),

    # ------------------------------------------------------------------
    # FACTORY_OPENING — new facility opening / inauguration
    # ------------------------------------------------------------------
    KapEventType.FACTORY_OPENING: TaxonomyEntry(
        event_type=KapEventType.FACTORY_OPENING,
        turkish_triggers=[
            r"fabrika\s+aç[ıi]l[ıi][şs][ıi]",
            r"tesis\s+aç[ıi]l[ıi][şs][ıi]",
            r"üretim\s+tesisi\s+devreye\s+al[ıi]nd[ıi]",
            r"yeni\s+fabrika\s+devreye",
            r"inauguration",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "New facility / factory officially opened or commissioned. "
            "Positive prior: marks transition from capex to revenue phase."
        ),
    ),

    # ------------------------------------------------------------------
    # PRODUCTION_HALT — operations stopped / facility shutdown
    # ------------------------------------------------------------------
    KapEventType.PRODUCTION_HALT: TaxonomyEntry(
        event_type=KapEventType.PRODUCTION_HALT,
        turkish_triggers=[
            r"üretim\s+durduruldu",
            r"üretim\s+durdu",
            r"tesis\s+kapat[ıi]ld[ıi]",
            r"i[şs]\s+durduruldu",
            r"faaliyetler\s+ask[ıi]ya\s+al[ıi]nd[ıi]",
            r"faaliyetler\s+durdu",
            r"grev",
        ],
        base_polarity=BasePolarity.VERY_NEGATIVE,
        notes=(
            "Production halt, facility shutdown, or strike. Very negative prior: "
            "direct revenue loss with no offsetting benefit announced."
        ),
    ),

    # ------------------------------------------------------------------
    # ACQUISITION — M&A: acquiring or merging with another entity
    # ------------------------------------------------------------------
    KapEventType.ACQUISITION: TaxonomyEntry(
        event_type=KapEventType.ACQUISITION,
        turkish_triggers=[
            r"devralma\s+karar[ıi]",
            r"sat[ıi]n\s+alma\s+karar[ıi]",
            r"sat[ıi]n\s+al[ıi]nd[ıi]",
            r"birle[şs]me\s+karar[ıi]",
            r"birle[şs]me\s+anla[şs]mas[ıi]",
            r"hisse\s+sat[ıi]n\s+al[ıi]m\s+karar[ıi]",   # share acquisition
            r"i[şs]tirak\s+sat[ıi]n\s+al[ıi]m[ıi]",       # subsidiary purchase
            r"tamamen\s+devral[ıi]nd[ıi]",
        ],
        exclude_patterns=[
            r"pay\s+geri\s+al[ıi]m",   # not a buyback
            r"geri\s+al[ıi]m\s+program[ıi]",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Acquisition / merger.  Positive prior: growth signal. "
            "Actual impact depends heavily on price paid and strategic fit. "
            "An acquirer often dips initially (integration risk premium); "
            "the target (if listed) typically rises sharply."
        ),
        param_keys=["amount_tl", "amount_usd", "pct_stake"],
    ),

    # ------------------------------------------------------------------
    # PARTNERSHIP — strategic partnership / JV (non-acquisition)
    # ------------------------------------------------------------------
    KapEventType.PARTNERSHIP: TaxonomyEntry(
        event_type=KapEventType.PARTNERSHIP,
        turkish_triggers=[
            r"ortakl[ıi]k\s+anla[şs]mas[ıi]",
            r"i[şs]\s+birli[gğ]i\s+anla[şs]mas[ıi]",
            r"ortak\s+giri[şs]im",
            r"joint\s+venture",
            r"konsorsiyum",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Strategic partnership or JV agreement (not full acquisition). "
            "Positive prior: signals market validation and shared risk."
        ),
        param_keys=["pct_stake"],
    ),

    # ------------------------------------------------------------------
    # SHARE_BUYBACK — company repurchases its own shares
    # ------------------------------------------------------------------
    KapEventType.SHARE_BUYBACK: TaxonomyEntry(
        event_type=KapEventType.SHARE_BUYBACK,
        turkish_triggers=[
            r"pay\s+geri\s+al[ıi]m\s+program[ıi]",
            r"hisse\s+geri\s+al[ıi]m\s+program[ıi]",
            r"geri\s+al[ıi]m\s+program[ıi]",
            r"öz\s+hisse\s+al[ıi]m[ıi]",
            r"kendi\s+pay[ıi]n[ıi]\s+sat[ıi]n\s+al",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Share buyback programme announcement.  Positive prior: signals "
            "management confidence and EPS accretion.  Actual execution rate "
            "may be lower than announced."
        ),
        param_keys=["amount_tl", "pct_stake"],
    ),

    # ------------------------------------------------------------------
    # INSIDER_PURCHASE — director/major shareholder buys shares
    # ------------------------------------------------------------------
    KapEventType.INSIDER_PURCHASE: TaxonomyEntry(
        event_type=KapEventType.INSIDER_PURCHASE,
        turkish_triggers=[
            r"yönetici\s+pay\s+al[ıi]m[ıi]",
            r"ortak\s+pay\s+al[ıi]m[ıi]",
            r"yönetim\s+kurulu\s+üyesi\s+pay\s+sat[ıi]n\s+al",
            r"genel\s+müdür\s+pay\s+sat[ıi]n\s+al",
            r"hakim\s+ortak\s+pay\s+al[ıi]m[ıi]",
            r"içeriden\s+al[ıi]m",
        ],
        exclude_patterns=[
            r"pay\s+sat[ıi][şs][ıi]",   # sale, not purchase
            r"sat[ıi]\s+gerçekle[şs]",
        ],
        base_polarity=BasePolarity.POSITIVE,
        notes=(
            "Insider (director or major shareholder) purchases shares on market. "
            "Positive prior: skin-in-the-game conviction signal."
        ),
        param_keys=["exec_name", "exec_role", "amount_tl", "pct_stake"],
    ),

    # ------------------------------------------------------------------
    # INSIDER_SALE — director/major shareholder sells shares
    # ------------------------------------------------------------------
    KapEventType.INSIDER_SALE: TaxonomyEntry(
        event_type=KapEventType.INSIDER_SALE,
        turkish_triggers=[
            r"yönetici\s+pay\s+sat[ıi][şs][ıi]",
            r"ortak\s+pay\s+sat[ıi][şs][ıi]",
            r"yönetim\s+kurulu\s+üyesi\s+pay\s+satt[ıi]",
            r"genel\s+müdür\s+pay\s+satt[ıi]",
            r"hakim\s+ortak\s+pay\s+sat[ıi][şs][ıi]",
            r"içeriden\s+sat[ıi][şs]",
        ],
        base_polarity=BasePolarity.NEGATIVE,
        notes=(
            "Insider sells shares. Negative prior but weaker than purchase signal: "
            "insiders may sell for personal liquidity reasons unrelated to outlook."
        ),
        param_keys=["exec_name", "exec_role", "amount_tl", "pct_stake"],
    ),

    # ------------------------------------------------------------------
    # CREDIT_RATING — credit rating change or affirmation
    # ------------------------------------------------------------------
    KapEventType.CREDIT_RATING: TaxonomyEntry(
        event_type=KapEventType.CREDIT_RATING,
        turkish_triggers=[
            r"kredi\s+derecelendirme",
            r"kredi\s+notu\s+(artır[ıi]ld[ıi]|yükseltildi|dü[şs]ürüldü|güncellendi)",
            r"notunu?\s+yükselt",
            r"notunu?\s+dü[şs]ür",
            r"görünüm[üu]\s+(olumlu|olumsuz|durağan|negatif|pozitif)",
            r"rating\s+karar[ıi]",
            r"fitch|moody|s&p|jcr",
        ],
        base_polarity=BasePolarity.NEUTRAL,
        notes=(
            "Credit rating action.  Base polarity is NEUTRAL because upgrades are "
            "POSITIVE and downgrades are NEGATIVE — the classifier alone cannot "
            "distinguish them without reading the content. The param_keys "
            "rating_from / rating_to store the direction for callers that extract them."
        ),
        param_keys=["rating_from", "rating_to"],
    ),

    # ------------------------------------------------------------------
    # FINANCIAL_RESULTS — periodic financial statement release
    # ------------------------------------------------------------------
    KapEventType.FINANCIAL_RESULTS: TaxonomyEntry(
        event_type=KapEventType.FINANCIAL_RESULTS,
        turkish_triggers=[
            r"finansal\s+tablo",
            r"bağımsız\s+denetim\s+raporu",
            r"bilanço\s+aç[ıi]kland[ıi]",
            r"kar\s+zarar\s+tablosu",
            r"[ıi]lk\s+çeyrek\s+sonuçlar[ıi]",
            r"ikinci\s+çeyrek\s+sonuçlar[ıi]",
            r"üçüncü\s+çeyrek\s+sonuçlar[ıi]",
            r"yıllık\s+sonuçlar\s+aç[ıi]kland[ıi]",
            r"dönem\s+sonu\s+finansal",
            r"faaliyet\s+sonuçlar[ıi]",
        ],
        base_polarity=BasePolarity.NEUTRAL,
        notes=(
            "Periodic financial statement publication.  Base polarity is NEUTRAL "
            "because results can beat or miss expectations. The actual signal "
            "direction requires comparison against consensus estimates which are "
            "not available in raw KAP text."
        ),
    ),

    # ------------------------------------------------------------------
    # GUIDANCE — forward guidance / earnings revision
    # ------------------------------------------------------------------
    KapEventType.GUIDANCE: TaxonomyEntry(
        event_type=KapEventType.GUIDANCE,
        turkish_triggers=[
            r"beklenti\s+(revizyonu|güncellemesi)",
            r"öngörü\s+revizyonu",
            r"hedef\s+(revize|güncelleme)",
            r"yıl\s+sonu\s+beklentisi\s+revize",
            r"kar\s+uyar[ıi]s[ıi]",
            r"ciroda\s+öngörü\s+revizyonu",
        ],
        base_polarity=BasePolarity.NEUTRAL,
        notes=(
            "Guidance revision (upward or downward).  NEUTRAL prior — direction "
            "of revision determines actual polarity; rule stage cannot distinguish "
            "upward from downward revision without reading the content."
        ),
    ),

    # ------------------------------------------------------------------
    # CEO_CHANGE — executive appointment or resignation
    # ------------------------------------------------------------------
    KapEventType.CEO_CHANGE: TaxonomyEntry(
        event_type=KapEventType.CEO_CHANGE,
        turkish_triggers=[
            r"genel\s+müdür\s+(atand[ıi]|istifa|de[gğ]i[şs]ikli[gğ]i)",
            r"genel\s+müdür\s+de[gğ]i[şs]ti",
            r"yönetim\s+kurulu\s+ba[şs]kan[ıi]\s+(atand[ıi]|istifa|de[gğ]i[şs]ti)",
            r"üst\s+yönetim\s+de[gğ]i[şs]ikli[gğ]i",
            r"icra\s+ba[şs]kan[ıi]\s+(atand[ıi]|istifa)",
            r"ceo\s+(de[gğ]i[şs]ikli[gğ]i|atand[ıi]|istifa)",
        ],
        base_polarity=BasePolarity.NEUTRAL,
        notes=(
            "Senior executive appointment or resignation.  NEUTRAL prior: "
            "unexpected departures can be negative; planned succession is neutral; "
            "high-profile external hires can be positive."
        ),
        param_keys=["exec_name", "exec_role"],
    ),

    # ------------------------------------------------------------------
    # BOARD_MEETING — board of directors meeting
    # ------------------------------------------------------------------
    KapEventType.BOARD_MEETING: TaxonomyEntry(
        event_type=KapEventType.BOARD_MEETING,
        turkish_triggers=[
            r"yönetim\s+kurulu\s+toplant[ıi]s[ıi]",
            r"yönetim\s+kurulu\s+karar[ıi]",
            r"yk\s+toplant[ıi]s[ıi]",
        ],
        exclude_patterns=[
            r"genel\s+kurul",   # that's GENERAL_ASSEMBLY
        ],
        base_polarity=BasePolarity.NEUTRAL,
        notes=(
            "Board of directors meeting notice or minutes.  NEUTRAL: meeting "
            "itself is informational; decisions made in the meeting are classified "
            "separately (dividend, rights issue, etc.)."
        ),
    ),

    # ------------------------------------------------------------------
    # GENERAL_ASSEMBLY — shareholder general assembly meeting
    # ------------------------------------------------------------------
    KapEventType.GENERAL_ASSEMBLY: TaxonomyEntry(
        event_type=KapEventType.GENERAL_ASSEMBLY,
        turkish_triggers=[
            r"olağan\s+genel\s+kurul",
            r"ola[gğ]anüstü\s+genel\s+kurul",
            r"genel\s+kurul\s+toplant[ıi]s[ıi]",
            r"genel\s+kurul\s+davet",
            r"genel\s+kurul\s+kararlar[ıi]",
        ],
        base_polarity=BasePolarity.NEUTRAL,
        notes=(
            "General assembly (AGM or EGM) notice, agenda, or minutes.  "
            "NEUTRAL: the meeting itself is informational; specific agenda items "
            "(dividend approval, capital increase vote) classified separately."
        ),
    ),

    # ------------------------------------------------------------------
    # SPK_INVESTIGATION — capital markets regulator investigation
    # ------------------------------------------------------------------
    KapEventType.SPK_INVESTIGATION: TaxonomyEntry(
        event_type=KapEventType.SPK_INVESTIGATION,
        turkish_triggers=[
            r"spk\s+soru[şs]turma",
            r"spk\s+inceleme",
            r"sermaye\s+piyasas[ıi]\s+kurulu\s+soru[şs]turma",
            r"sermaye\s+piyasas[ıi]\s+kurulu\s+inceleme",
            r"spk\s+yaptır[ıi]m",
            r"spk\s+idari\s+para\s+cezas[ıi]",
            r"manipülasyon\s+iddia",
        ],
        base_polarity=BasePolarity.VERY_NEGATIVE,
        notes=(
            "SPK (Sermaye Piyasası Kurulu — Turkish capital markets regulator) "
            "investigation or administrative action.  Very negative prior: "
            "regulatory uncertainty, possible trading halt, reputational damage."
        ),
    ),

    # ------------------------------------------------------------------
    # LAWSUIT — significant legal proceedings
    # ------------------------------------------------------------------
    KapEventType.LAWSUIT: TaxonomyEntry(
        event_type=KapEventType.LAWSUIT,
        turkish_triggers=[
            r"dava\s+aç[ıi]ld[ıi]",
            r"hukuki\s+süreç\s+ba[şs]lat[ıi]ld[ıi]",
            r"tahkim\s+süreci",
            r"icra\s+takibi",
            r"tazminat\s+davas[ıi]",
        ],
        exclude_patterns=[
            r"dava\s+sonuçland[ıi]",   # concluded lawsuit is different
            r"beraat",                  # acquittal
        ],
        base_polarity=BasePolarity.NEGATIVE,
        notes=(
            "Significant new lawsuit or legal proceeding.  Negative prior: "
            "uncertain liability, legal costs, management distraction."
        ),
        param_keys=["amount_tl"],
    ),

    # ------------------------------------------------------------------
    # DEFAULT — financial distress / bankruptcy
    # ------------------------------------------------------------------
    KapEventType.DEFAULT: TaxonomyEntry(
        event_type=KapEventType.DEFAULT,
        turkish_triggers=[
            r"konkordato",
            r"iflas\s+(karar[ıi]|talep|tescili)",
            r"icra\s+iflas",
            r"ödeme\s+güçlü[gğ]ü",
            r"kredi\s+temerrüt",
            r"bono\s+temerrüt",
            r"finansal\s+yükümlülük\s+yerine\s+getirilemiyor",
        ],
        base_polarity=BasePolarity.VERY_NEGATIVE,
        notes=(
            "Bankruptcy, concordat, or payment default event.  Very negative prior: "
            "existential threat to equity value.  Concordat (Turkish insolvency "
            "protection) often precedes significant haircuts."
        ),
    ),

    # ------------------------------------------------------------------
    # CAPITAL_REDUCTION — reducing registered capital
    # ------------------------------------------------------------------
    KapEventType.CAPITAL_REDUCTION: TaxonomyEntry(
        event_type=KapEventType.CAPITAL_REDUCTION,
        turkish_triggers=[
            r"sermaye\s+azalt[ıi]m[ıi]",
            r"sermaye\s+indirimi",
            r"ödenmi[şs]\s+sermayenin\s+azalt[ıi]lmas[ıi]",
        ],
        base_polarity=BasePolarity.NEGATIVE,
        notes=(
            "Registered capital reduction. Negative prior: may signal accumulated "
            "losses that must be absorbed (required by Turkish Commercial Code "
            "Article 376 when losses exceed half the capital)."
        ),
        param_keys=["amount_tl"],
    ),
}


# ---------------------------------------------------------------------------
# Guard assertions — bedelsiz/bedelli MUST map to opposite polarities
# ---------------------------------------------------------------------------
# These run at import time.  Any future taxonomy edit that accidentally
# makes bedelsiz and bedelli share the same polarity will raise AssertionError
# immediately, failing the module import and preventing silent misclassification.

assert TAXONOMY[KapEventType.BONUS_ISSUE].base_polarity != TAXONOMY[KapEventType.RIGHTS_ISSUE].base_polarity, (
    "CRITICAL TAXONOMY CORRUPTION: BONUS_ISSUE and RIGHTS_ISSUE must have "
    "DIFFERENT base polarities.  bedelsiz = free shares (positive prior); "
    "bedelli = paid shares / dilution (negative prior).  "
    "Misclassifying these gives BACKWARDS advice."
)

assert TAXONOMY[KapEventType.BONUS_ISSUE].base_polarity in (BasePolarity.POSITIVE, BasePolarity.STRONG_POSITIVE), (
    "BONUS_ISSUE (bedelsiz) must be POSITIVE or STRONG_POSITIVE."
)

assert TAXONOMY[KapEventType.RIGHTS_ISSUE].base_polarity in (BasePolarity.NEGATIVE, BasePolarity.VERY_NEGATIVE), (
    "RIGHTS_ISSUE (bedelli) must be NEGATIVE or VERY_NEGATIVE."
)

assert KapEventType.BONUS_ISSUE in TAXONOMY, "BONUS_ISSUE missing from TAXONOMY"
assert KapEventType.RIGHTS_ISSUE in TAXONOMY, "RIGHTS_ISSUE missing from TAXONOMY"
assert "bedelsiz" not in " ".join(TAXONOMY[KapEventType.RIGHTS_ISSUE].turkish_triggers), (
    "RIGHTS_ISSUE triggers must NOT contain 'bedelsiz'."
)
assert "bedelli" not in " ".join(TAXONOMY[KapEventType.BONUS_ISSUE].turkish_triggers), (
    "BONUS_ISSUE triggers must NOT contain 'bedelli'."
)
