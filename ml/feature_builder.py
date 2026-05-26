"""Shared ML feature-dict builder — the SINGLE source of truth for the nested
schema that ``EnsemblePredictor.extract_features`` consumes.

Why this module exists
----------------------
At inference, ``core/engine.py`` maps a live opportunity (DexScreener ``pair``
dict + a ``RiskScore`` object + a ``patterns`` dict) into the nested feature
dict the ensemble expects, then calls ``extract_features`` -> the canonical
82-element vector (``ENSEMBLE_FEATURE_NAMES``).

The offline trainer (``scripts/train_ensemble.py``) must produce the EXACT same
feature dict from each historical ``trades.metadata`` row so that the training
matrix is identical-by-construction to what the engine builds live (no
train/inference skew). Rather than copy the mapping into two places (which
would silently drift), both the engine and the trainer call this one function.

``core/engine.py::TradingBotEngine._build_ml_feature_dict`` is a thin delegating
wrapper around ``build_ml_feature_dict`` — moving the body here is a pure,
logic-preserving refactor (the engine still produces byte-identical dicts).

Contract of the inputs (unchanged from the engine's original method)
--------------------------------------------------------------------
- ``pair``: dict of DexScreener-style pair fields (``price_usd``,
  ``price_change_1h``, ``volume_24h``, ``liquidity_usd``, ``age_hours`` ...).
- ``risk_score``: an object exposing risk sub-scores as attributes
  (``liquidity_risk``, ``developer_risk`` ...). Accessed via ``getattr`` /
  ``hasattr`` so any object — the engine's ``RiskScore`` dataclass OR a
  ``SimpleNamespace`` the trainer builds from a JSONB dict — works identically.
  Pass ``None`` when no risk assessment is available.
- ``patterns``: dict of pattern flags (only ``trend_strength`` is forwarded
  today). Pass ``None``/``{}`` when absent.
"""
from typing import Dict


def build_ml_feature_dict(pair: Dict, risk_score, patterns) -> Dict:
    """Map gathered opportunity data into the nested-dict schema
    ``EnsemblePredictor.extract_features()`` consumes.

    This does NOT invent a feature pipeline: it only forwards values the caller
    already has (DexScreener pair fields + risk sub-scores + pattern flags).
    Fields we genuinely lack are left at ``extract_features``' own defaults
    (0 / 50 for RSI) rather than fabricated.
    """
    liq = pair.get('liquidity_usd') or pair.get('liquidity') or 0
    feat = {
        'price_data': {
            'current_price': pair.get('price_usd', 0) or 0,
            'price_change_1h': pair.get('price_change_1h', 0) or 0,
            'price_change_24h': pair.get('price_change_24h', 0) or 0,
            'volatility_24h': pair.get('volatility', 0) or 0,
        },
        'volume_data': {
            'volume_24h': pair.get('volume_24h', 0) or 0,
            'volume_change_24h': pair.get('volume_change_24h', 0) or 0,
            'buy_volume_ratio': pair.get('buy_sell_ratio', 0) or 0,
        },
        'liquidity_data': {
            'total_liquidity': liq,
            'liquidity_change_24h': pair.get('liquidity_change_24h', 0) or 0,
        },
        'time_data': {
            'days_since_launch': (pair.get('age_hours', 0) or 0) / 24.0,
        },
        'market_data': {
            'market_cap': pair.get('market_cap', 0) or 0,
            'fully_diluted_valuation': pair.get('fdv', 0) or 0,
        },
    }
    # Forward the real risk sub-scores when the assessment succeeded.
    if risk_score is not None and hasattr(risk_score, 'liquidity_risk'):
        feat['risk_data'] = {
            'liquidity_risk': getattr(risk_score, 'liquidity_risk', 0) or 0,
            'developer_risk': getattr(risk_score, 'developer_risk', 0) or 0,
            'contract_risk': getattr(risk_score, 'contract_risk', 0) or 0,
            'volume_risk': getattr(risk_score, 'volume_risk', 0) or 0,
            'holder_risk': getattr(risk_score, 'holder_risk', 0) or 0,
            'honeypot_probability': getattr(risk_score, 'honeypot_risk', 0) or 0,
        }
        feat['holder_data'] = {
            'top_10_holders_percent': getattr(risk_score, 'top_10_holders_percentage', 0) or 0,
            'whale_count': getattr(risk_score, 'whale_concentration', 0) or 0,
            'total_holders': getattr(risk_score, 'unique_holders', 0) or 0,
        }
    if isinstance(patterns, dict):
        feat['pattern_data'] = {
            'trend_strength': patterns.get('trend_strength', 0) or 0,
        }
    return feat
