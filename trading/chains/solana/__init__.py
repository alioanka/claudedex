# trading/chains/solana/__init__.py
"""
Solana trading integration via Jupiter aggregator
"""

from .jupiter_executor import JupiterExecutor, JupiterQuote, JupiterRoute
from .solana_client import SolanaClient

__all__ = [
    'JupiterExecutor',
    'JupiterQuote',
    'JupiterRoute',
    'SolanaClient',
    'COMMON_TOKENS'
]

# Common Solana token mints for quick reference
COMMON_TOKENS = {
    'SOL': 'So11111111111111111111111111111111111111112',  # Wrapped SOL
    'USDC': 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
    'USDT': 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
    'RAY': '4k3Dyjzvzp8eMZWUXbBCjEvwSkkk59S5iCNLY3QrkX6R',
    'SRM': 'SRMuApVNdxXokk5GT7XD5cUUgXMBCoAz2LHeuAoKWRt',
    'BONK': 'DezXAZ8z7PnrnRJjz3wXBoRgixCa6xjnB7YaB1pPB263',
    'JUP': 'JUPyiwrYJFskUPiHa7hkeR8VUtAeFoSYbKedZNsDvCN',
    'ORCA': 'orcaEKTdK7LKz57vaAYr9QeNsVEPfiu6QeMU1kektZE',
}