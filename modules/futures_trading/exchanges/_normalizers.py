"""Canonical position/balance shapes for cross-exchange code paths.

Callers that mix Binance + Bybit (e.g. FuturesRiskManager, dashboard
reconcile) pipe raw executor dicts through normalize_position() /
normalize_balance() to get a single shape with American-English keys.

Additive: executor return shapes are untouched."""

from typing import Dict, Optional


# Canonical keys (American English; superset of both exchanges)
CANONICAL_POSITION_KEYS = {
    'symbol', 'side', 'size', 'entry_price', 'mark_price',
    'unrealized_pnl', 'leverage', 'notional_value', 'liquidation_price',
    'margin_type', 'source', 'raw',
}

CANONICAL_BALANCE_KEYS = {
    'asset', 'balance', 'available', 'unrealized_pnl',
    'margin_balance', 'source', 'raw',
}


def normalize_position(
    raw: Optional[Dict],
    source: str,
) -> Optional[Dict]:
    """Map a Binance or Bybit position dict to the canonical schema.

    Args:
        raw: dict from BinanceFuturesExecutor or BybitFuturesExecutor
        source: 'binance' or 'bybit' (case-insensitive)

    Returns:
        Canonical position dict, or None if raw is None/empty.
        Missing fields default to 0.0 / 0 / None; never raises.
    """
    if not raw:
        return None
    src = (source or '').lower()
    try:
        # Pull both spelling variants then prefer source-specific key
        size = raw.get('size', raw.get('position_amt', 0)) or 0
        unreal = raw.get('unrealized_pnl', raw.get('unrealised_pnl', 0)) or 0
        notional = raw.get('notional_value', raw.get('position_value', 0)) or 0
        liq = raw.get('liquidation_price')
        # Bybit V5 raw payload uses 'liqPrice' — try to recover from raw
        if liq is None and isinstance(raw.get('raw'), dict):
            liq_raw = raw['raw'].get('liqPrice', '')
            try:
                liq = float(liq_raw) if liq_raw else None
            except (TypeError, ValueError):
                liq = None
        margin_type = raw.get('margin_type')
        if margin_type is None and isinstance(raw.get('raw'), dict):
            tm = raw['raw'].get('tradeMode')
            # Bybit V5: tradeMode 1=ISOLATED 0=CROSS
            if tm == 1:
                margin_type = 'ISOLATED'
            elif tm == 0:
                margin_type = 'CROSS'

        return {
            'symbol': raw.get('symbol', ''),
            'side': raw.get('side', ''),
            'size': float(size),
            'entry_price': float(raw.get('entry_price', 0) or 0),
            'mark_price': float(raw.get('mark_price', 0) or 0),
            'unrealized_pnl': float(unreal),
            'leverage': float(raw.get('leverage', 0) or 0),
            'notional_value': float(notional),
            'liquidation_price': float(liq) if liq is not None else 0.0,
            'margin_type': margin_type,
            'source': src,
            'raw': raw,
        }
    except Exception:
        return None


def normalize_balance(
    raw: Optional[Dict],
    source: str,
) -> Optional[Dict]:
    """Map a Binance or Bybit balance dict to the canonical schema.
    Same contract as normalize_position; never raises."""
    if not raw:
        return None
    try:
        unreal = raw.get('unrealized_pnl', raw.get('unrealised_pnl', 0)) or 0
        return {
            'asset': raw.get('asset', 'USDT'),
            'balance': float(raw.get('balance', 0) or 0),
            'available': float(raw.get('available', 0) or 0),
            'unrealized_pnl': float(unreal),
            'margin_balance': float(raw.get('margin_balance', 0) or 0),
            'source': (source or '').lower(),
            'raw': raw,
        }
    except Exception:
        return None
