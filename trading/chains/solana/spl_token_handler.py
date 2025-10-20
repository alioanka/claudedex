# trading/chains/solana/spl_token_handler.py
"""
SPL Token Handler - Helper utilities for SPL token operations
This is OPTIONAL - not required for basic Jupiter trading
"""

import logging
from typing import Dict, Optional
from decimal import Decimal

logger = logging.getLogger(__name__)

class SPLTokenHandler:
    """
    Helper class for SPL token operations
    NOTE: Most token operations are handled by Jupiter API
    This is mainly for direct token queries
    """
    
    def __init__(self, solana_client):
        """
        Initialize SPL token handler
        
        Args:
            solana_client: SolanaClient instance
        """
        self.client = solana_client
        
    async def get_token_balance(
        self,
        wallet_address: str,
        token_mint: str
    ) -> Optional[Decimal]:
        """
        Get SPL token balance for a wallet
        
        Args:
            wallet_address: Wallet public key
            token_mint: Token mint address
            
        Returns:
            Token balance or None
        """
        try:
            balance = await self.client.get_token_balance(
                wallet_address,
                token_mint
            )
            return balance
            
        except Exception as e:
            logger.error(f"Error getting token balance: {e}")
            return None
            
    async def get_token_info(
        self,
        token_mint: str
    ) -> Optional[Dict]:
        """
        Get token metadata
        
        Args:
            token_mint: Token mint address
            
        Returns:
            Token info dict or None
        """
        try:
            supply_info = await self.client.get_token_supply(token_mint)
            
            if supply_info:
                return {
                    'mint': token_mint,
                    'supply': supply_info.get('value', {}).get('amount', 0),
                    'decimals': supply_info.get('value', {}).get('decimals', 9),
                    'ui_amount': supply_info.get('value', {}).get('uiAmount', 0)
                }
            
            return None
            
        except Exception as e:
            logger.error(f"Error getting token info: {e}")
            return None
            
    def format_token_amount(
        self,
        amount: int,
        decimals: int = 9
    ) -> Decimal:
        """
        Convert raw token amount to decimal
        
        Args:
            amount: Raw token amount (integer)
            decimals: Token decimals (default 9 for SOL)
            
        Returns:
            Decimal amount
        """
        return Decimal(amount) / Decimal(10 ** decimals)
        
    def to_raw_amount(
        self,
        amount: Decimal,
        decimals: int = 9
    ) -> int:
        """
        Convert decimal amount to raw integer
        
        Args:
            amount: Decimal amount
            decimals: Token decimals (default 9 for SOL)
            
        Returns:
            Raw integer amount
        """
        return int(amount * Decimal(10 ** decimals))
        
    async def get_token_accounts(
        self,
        wallet_address: str
    ) -> Optional[list]:
        """
        Get all token accounts for a wallet
        
        Args:
            wallet_address: Wallet public key
            
        Returns:
            List of token accounts or None
        """
        try:
            # This would need RPC method implementation
            # Not critical for trading functionality
            logger.warning("get_token_accounts not fully implemented")
            return None
            
        except Exception as e:
            logger.error(f"Error getting token accounts: {e}")
            return None


# Common SPL token decimals
TOKEN_DECIMALS = {
    'SOL': 9,
    'USDC': 6,
    'USDT': 6,
    'RAY': 6,
    'SRM': 6,
    'BONK': 5,
}


def get_token_decimals(token_mint: str) -> int:
    """
    Get decimals for a token (default 9)
    
    Args:
        token_mint: Token mint address
        
    Returns:
        Number of decimals
    """
    # Map of known tokens
    known_tokens = {
        'So11111111111111111111111111111111111111112': 9,  # SOL
        'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v': 6,  # USDC
        'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB': 6,  # USDT
    }
    
    return known_tokens.get(token_mint, 9)  # Default to 9 decimals