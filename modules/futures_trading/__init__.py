"""
Futures Trading Module - Professional CEX Futures Trading

Complete futures trading system with:
✅ Independent LONG & SHORT trading (not just hedging!)
✅ ML-powered decisions (ensemble models)
✅ Advanced technical analysis & chart patterns
✅ Sophisticated position management:
   - Trailing Stop Loss
   - Partial Take Profits (4 levels @ 25% each)
   - Dynamic position sizing
   - Auto-leverage adjustment
✅ Risk management for leverage trading
✅ Multi-timeframe analysis
✅ Professional strategies

Supported Exchanges:
- Binance Futures (USDT-M, COIN-M perpetuals)
- Bybit Derivatives
- OKX Futures (planned)

Trading Capabilities:
- Long positions in bullish markets
- Short positions in bearish markets
- Funding rate arbitrage
- Trend following
- Mean reversion
- Breakout trading

Position Management:
- Partial TPs: 25% @ TP1, 25% @ TP2, 25% @ TP3, 25% @ TP4
- Trailing SL: Activates after +3% profit, trails 2% below peak
- Dynamic sizing based on ML confidence & volatility
- Auto-leverage adjustment (1x-3x based on conditions)

This module operates INDEPENDENTLY from DEX module!
"""

from .futures_module import FuturesTradingModule

__all__ = ['FuturesTradingModule']

__version__ = "2.0.0"
