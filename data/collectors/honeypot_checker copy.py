"""
Honeypot Checker - Multi-source honeypot and scam detection
"""

import aiohttp
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json
from enum import Enum

@dataclass
class HoneypotResult:
    """Honeypot check result"""
    is_honeypot: bool
    confidence: float  # 0-1
    buy_tax: float
    sell_tax: float
    is_mintable: bool
    is_proxy: bool
    is_pausable: bool
    can_blacklist: bool
    has_hidden_owner: bool
    owner_can_change_balance: bool
    cannot_sell_all: bool
    slippage_modifiable: bool
    transfer_pausable: bool
    trading_cooldown: bool
    personal_slippage_modifiable: bool
    anti_whale: bool
    anti_whale_modifiable: bool
    cannot_buy: bool
    can_take_back_ownership: bool
    honeypot_with_same_creator: int
    fake_token: bool
    sources_checked: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.now)
    
class RiskLevel(Enum):
    """Risk levels for tokens"""
    SAFE = "safe"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class HoneypotChecker:
    """Multi-source honeypot and scam detection"""
    
    def __init__(self, config: Dict):
        """
        Initialize honeypot checker
        
        Args:
            config: Configuration with API keys
        """
        self.config = config
        
        # API endpoints
        self.goplus_api = "https://api.gopluslabs.io/api/v1/token_security"
        self.tokensniffer_api = "https://tokensniffer.com/api/v2/tokens"
        self.honeypot_is_api = "https://api.honeypot.is/v2/IsHoneypot"
        
        # API keys
        self.goplus_key = config.get('goplus_api_key')
        self.tokensniffer_key = config.get('tokensniffer_api_key')
        
        # Cache for results
        self.cache = {}
        self.cache_ttl = 300  # 5 minutes
        
        # Risk thresholds
        self.tax_threshold = 0.1  # 10% tax considered high
        self.holder_threshold = 0.5  # 50% held by single address is risky
        
    async def check_honeypot(self, token_address: str, chain: str = 'eth',
                            comprehensive: bool = True) -> HoneypotResult:
        """
        Comprehensive honeypot check using multiple sources
        
        Args:
            token_address: Token contract address
            chain: Blockchain network
            comprehensive: If True, check all sources
            
        Returns:
            HoneypotResult with analysis
        """
        try:
            # Check cache
            cache_key = f"{chain}:{token_address}"
            if cache_key in self.cache:
                cached = self.cache[cache_key]
                if (datetime.now() - cached['timestamp']).seconds < self.cache_ttl:
                    return cached['result']
                    
            # Parallel checks from multiple sources
            tasks = []
            
            # Always check GoPlus
            tasks.append(self._check_goplus(token_address, chain))
            
            if comprehensive:
                tasks.append(self._check_tokensniffer(token_address, chain))
                tasks.append(self._check_honeypot_is(token_address, chain))
                
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Combine results
            combined = self._combine_results(results)
            
            # Cache result
            self.cache[cache_key] = {
                'result': combined,
                'timestamp': datetime.now()
            }
            
            return combined
            
        except Exception as e:
            print(f"Honeypot check error: {e}")
            return HoneypotResult(
                is_honeypot=False,
                confidence=0,
                buy_tax=0,
                sell_tax=0,
                is_mintable=False,
                is_proxy=False,
                is_pausable=False,
                can_blacklist=False,
                has_hidden_owner=False,
                owner_can_change_balance=False,
                cannot_sell_all=False,
                slippage_modifiable=False,
                transfer_pausable=False,
                trading_cooldown=False,
                personal_slippage_modifiable=False,
                anti_whale=False,
                anti_whale_modifiable=False,
                cannot_buy=False,
                can_take_back_ownership=False,
                honeypot_with_same_creator=0,
                fake_token=False,
                reasons=[f"Check failed: {e}"]
            )
            
    async def _check_goplus(self, token_address: str, chain: str) -> Dict:
        """Check token using GoPlus Security API"""
        try:
            chain_id = self._get_chain_id(chain)
            
            async with aiohttp.ClientSession() as session:
                url = f"{self.goplus_api}/{chain_id}"
                params = {'contract_addresses': token_address}
                
                headers = {}
                if self.goplus_key:
                    headers['Authorization'] = f"Bearer {self.goplus_key}"
                    
                async with session.get(url, params=params, headers=headers) as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        if 'result' in data and token_address.lower() in data['result']:
                            token_data = data['result'][token_address.lower()]
                            
                            return {
                                'source': 'goplus',
                                'is_anti_whale': token_data.get('is_anti_whale', '0') == '1',
                                'anti_whale_modifiable': token_data.get('anti_whale_modifiable', '0') == '1',
                                'trading_cooldown': token_data.get('trading_cooldown', '0') == '1',
                                'personal_slippage_modifiable': token_data.get('personal_slippage_modifiable', '0') == '1',
                                'transfer_pausable': token_data.get('transfer_pausable', '0') == '1',
                                'honeypot_with_same_creator': int(token_data.get('honeypot_with_same_creator', 0)),
                                'holder_count': int(token_data.get('holder_count', 0)),
                                'total_supply': float(token_data.get('total_supply', 0)),
                                'is_true_token': token_data.get('is_true_token', '0') == '1',
                                'is_airdrop_scam': token_data.get('is_airdrop_scam', '0') == '1',
                                'trust_list': token_data.get('trust_list', '0') == '1',
                                'other_potential_risks': token_data.get('other_potential_risks', '')
                            }
                            
            return {'source': 'goplus', 'error': 'Failed to fetch data'}
            
        except Exception as e:
            print(f"GoPlus check error: {e}")
            return {'source': 'goplus', 'error': str(e)}honeypot': token_data.get('is_honeypot', '0') == '1',
                                'buy_tax': float(token_data.get('buy_tax', 0)),
                                'sell_tax': float(token_data.get('sell_tax', 0)),
                                'is_mintable': token_data.get('is_mintable', '0') == '1',
                                'is_proxy': token_data.get('is_proxy', '0') == '1',
                                'can_take_back_ownership': token_data.get('can_take_back_ownership', '0') == '1',
                                'owner_change_balance': token_data.get('owner_change_balance', '0') == '1',
                                'hidden_owner': token_data.get('hidden_owner', '0') == '1',
                                'selfdestruct': token_data.get('selfdestruct', '0') == '1',
                                'external_call': token_data.get('external_call', '0') == '1',
                                'is_in_dex': token_data.get('is_in_dex', '0') == '1',
                                'is_open_source': token_data.get('is_open_source', '0') == '1',
                                'cannot_buy': token_data.get('cannot_buy', '0') == '1',
                                'cannot_sell_all': token_data.get('cannot_sell_all', '0') == '1',
                                'slippage_modifiable': token_data.get('slippage_modifiable', '0') == '1',
                                'is_blacklisted': token_data.get('is_blacklisted', '0') == '1',
                                'is_whitelisted': token_data.get('is_whitelisted', '0') == '1',
                                'is_

#missing some things