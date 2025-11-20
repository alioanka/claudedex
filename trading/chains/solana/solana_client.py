# trading/chains/solana/solana_client.py
"""
Solana RPC Client with automatic failover and retry logic
"""

import asyncio
import aiohttp
import logging
from typing import Dict, List, Optional, Any
from decimal import Decimal

logger = logging.getLogger(__name__)

class SolanaClient:
    """
    Solana RPC client with automatic failover
    """
    
    def __init__(self, rpc_urls: List[str], timeout: int = 30):
        self.rpc_urls = rpc_urls
        self.current_rpc_index = 0
        self.timeout = aiohttp.ClientTimeout(total=timeout)
        self.session: Optional[aiohttp.ClientSession] = None
        
    async def initialize(self):
        """Initialize HTTP session"""
        self.session = aiohttp.ClientSession(timeout=self.timeout)
        
    async def _request(
        self,
        method: str,
        params: List[Any],
        rpc_url: Optional[str] = None
    ) -> Optional[Dict]:
        """Make RPC request with automatic failover"""
        
        urls_to_try = [rpc_url] if rpc_url else self.rpc_urls
        
        for url in urls_to_try:
            try:
                payload = {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": method,
                    "params": params
                }
                
                async with self.session.post(url, json=payload) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'result' in data:
                            return data['result']
                        elif 'error' in data:
                            logger.error(f"RPC error: {data['error']}")
                            continue
                            
            except Exception as e:
                logger.warning(f"RPC request failed for {url}: {e}")
                continue
                
        logger.error(f"All RPC endpoints failed for method: {method}")
        return None
        
    async def get_balance(self, address: str) -> Optional[Decimal]:
        """Get SOL balance for address"""
        result = await self._request("getBalance", [address])
        
        if result:
            return Decimal(str(result['value'])) / Decimal('1e9')
        return None
        
    async def get_token_balance(
        self,
        address: str,
        mint: str
    ) -> Optional[Decimal]:
        """Get SPL token balance"""
        result = await self._request(
            "getTokenAccountsByOwner",
            [
                address,
                {"mint": mint},
                {"encoding": "jsonParsed"}
            ]
        )
        
        if result and result.get('value'):
            accounts = result['value']
            if accounts:
                token_amount = accounts[0]['account']['data']['parsed']['info']['tokenAmount']
                return Decimal(token_amount['uiAmountString'])
                
        return Decimal('0')
        
    async def get_recent_blockhash(self) -> Optional[str]:
        """Get recent blockhash"""
        result = await self._request("getLatestBlockhash", [])
        
        if result:
            return result['blockhash']
        return None
        
    async def get_slot(self) -> Optional[int]:
        """Get current slot"""
        result = await self._request("getSlot", [])
        return result
        
    async def get_token_supply(self, mint: str) -> Optional[Dict]:
        """Get token supply info"""
        result = await self._request("getTokenSupply", [mint])
        return result
        
    async def get_transaction(self, signature: str) -> Optional[Dict]:
        """Get transaction details"""
        result = await self._request(
            "getTransaction",
            [signature, {"encoding": "jsonParsed"}]
        )
        return result
        
    async def cleanup(self):
        """Cleanup resources"""
        if self.session and not self.session.closed:
            await self.session.close()