"""
DEX Trading Module - Spot trading on decentralized exchanges

This module handles:
- DEX spot trading (Raydium, Orca, Jupiter on Solana)
- Uniswap, SushiSwap on EVM chains
- Momentum, Scalping, and AI strategies
- Independent wallet management
- Risk management for DEX trades
"""

from .dex_module import DexTradingModule

__all__ = ['DexTradingModule']
