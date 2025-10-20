"""
DexScreener API Integration
Real-time DEX data collection and monitoring
"""

import asyncio
import aiohttp
from typing import Dict, List, Optional, Any, AsyncGenerator
from datetime import datetime, timedelta
import json
from dataclasses import dataclass, field
import time
from collections import deque
import numpy as np

@dataclass
class TokenPair:
    """Token pair data structure"""
    chain_id: str
    dex_id: str
    pair_address: str
    base_token: Dict
    quote_token: Dict
    price_native: float
    price_usd: float
    liquidity_usd: float
    liquidity_base: float
    liquidity_quote: float
    volume_24h: float
    volume_change_24h: float
    price_change_24h: float
    price_change_1h: float
    price_change_5m: float
    trades_24h: int
    buyers_24h: int
    sellers_24h: int
    created_at: int
    pair_created_at: datetime
    info: Dict = field(default_factory=dict)
    
    @property
    def age_hours(self) -> float:
        """Get pair age in hours"""
        return (datetime.now() - self.pair_created_at).total_seconds() / 3600
        
    @property
    def buy_sell_ratio(self) -> float:
        """Calculate buy/sell ratio"""
        total = self.buyers_24h + self.sellers_24h
        if total == 0:
            return 0.5
        return self.buyers_24h / total

class DexScreenerCollector:
    """DexScreener data collector"""
    
    def __init__(self, config: Dict):
        """
        Initialize DexScreener collector
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.api_key = config.get('api_key', '')
        self.base_url = "https://api.dexscreener.com"
        
        # Rate limiting
        self.rate_limit = config.get('rate_limit', 100)
        self.request_times = deque(maxlen=self.rate_limit)
        
        # Caching
        self.cache = {}
        self.cache_duration = config.get('cache_duration', 60)  # seconds
        
        # Monitoring
        self.monitored_pairs = set()
        self.new_pairs_queue = asyncio.Queue()
        
        # Filters
        self.min_liquidity = config.get('min_liquidity', 10000)
        self.min_volume = config.get('min_volume', 5000)
        self.max_age_hours = config.get('max_age_hours', 24)
        self.chains = config.get('chains', ['ethereum', 'bsc', 'polygon', 'arbitrum', 'base', 'solana'])
        
        # Session
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Statistics
        self.stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'pairs_found': 0,
            'pairs_filtered': 0
        }

    # Add this class-level constant after __init__:
    CHAIN_MAPPING = {
        # EVM chains
        'ethereum': 'ethereum',
        'eth': 'ethereum',
        'bsc': 'bsc',
        'bnb': 'bsc',
        'polygon': 'polygon',
        'matic': 'polygon',
        'arbitrum': 'arbitrum',
        'arb': 'arbitrum',
        'base': 'base',
        'optimism': 'optimism',
        'op': 'optimism',
        'avalanche': 'avalanche',
        'avax': 'avalanche',
        
        # Solana - IMPORTANT: DexScreener uses 'solana' (lowercase)
        'solana': 'solana',
        'sol': 'solana',
    }

    def _normalize_chain(self, chain: str) -> str:
        """
        Normalize chain name for DexScreener API
        
        Args:
            chain: Chain name (can be 'solana', 'SOL', 'Solana', etc.)
            
        Returns:
            Normalized chain name for API
        """
        chain_lower = chain.lower()
        return self.CHAIN_MAPPING.get(chain_lower, chain_lower)

    async def initialize(self):
        """Initialize the collector"""
        if not self.session:
            timeout = aiohttp.ClientTimeout(total=30)
            self.session = aiohttp.ClientSession(timeout=timeout)
            
    async def close(self):
        """Close the collector"""
        if self.session:
            await self.session.close()
            
    async def _rate_limit(self):
        """Implement rate limiting"""
        now = time.time()
        
        # Remove old requests outside the window
        while self.request_times and self.request_times[0] < now - 60:
            self.request_times.popleft()
            
        # Check if we need to wait
        if len(self.request_times) >= self.rate_limit:
            sleep_time = 60 - (now - self.request_times[0])
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
                
        self.request_times.append(now)
        
    async def _make_request(self, endpoint: str, params: Dict = None) -> Optional[Dict]:
        """
        Make API request with error handling
        
        Args:
            endpoint: API endpoint (without base URL)
            params: Query parameters
            
        Returns:
            Response data or None
        """
        await self._rate_limit()
        
        # Remove any leading slashes from endpoint
        endpoint = endpoint.lstrip('/')
        
        url = f"{self.base_url}/{endpoint}"
        headers = {}
        
        if self.api_key:
            headers['X-API-KEY'] = self.api_key
            
        try:
            self.stats['total_requests'] += 1
            
            async with self.session.get(url, params=params, headers=headers) as response:
                if response.status == 200:
                    self.stats['successful_requests'] += 1
                    data = await response.json()
                    return data
                else:
                    self.stats['failed_requests'] += 1
                    print(f"API request failed: {response.status} - URL: {url}")
                    return None
                    
        except asyncio.TimeoutError:
            self.stats['failed_requests'] += 1
            print("Request timeout")
            return None
        except Exception as e:
            self.stats['failed_requests'] += 1
            print(f"Request error: {e}")
            return None
                    
        except asyncio.TimeoutError:
            self.stats['failed_requests'] += 1
            print("Request timeout")
            return None
        except Exception as e:
            self.stats['failed_requests'] += 1
            print(f"Request error: {e}")
            return None
            
    # 1. Fix get_new_pairs signature (line ~165)
    # REPLACE the existing method signature and update the implementation:

    # ============================================
    # PATCH 2: Update get_new_pairs to normalize chain (line ~170)
    # Replace the beginning of get_new_pairs method:
    # ============================================

    async def get_new_pairs(self, chain: str = 'ethereum', limit: int = 100) -> List[Dict]:
        """
        Get newly created pairs using CORRECT DexScreener API v1 endpoints
        
        Args:
            chain: Blockchain network (ethereum, bsc, solana, etc.)
            limit: Maximum number of pairs to return
            
        Returns:
            List of new pair data
        """
        # ✅ CRITICAL: Normalize chain name for DexScreener
        chain = self._normalize_chain(chain)
        
        new_pairs = []
        seen_addresses = set()
        
        # Check cache
        cache_key = f"new_pairs_{chain}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                print(f"✅ Using cached data for {chain}")
                return cached_data
        
        try:
            print(f"🔍 Fetching new pairs for {chain}...")
            
            # ✅ STRATEGY 1: Latest boosted tokens
            try:
                boosted_data = await self._make_request("/token-boosts/latest/v1")
                
                if boosted_data:
                    print(f"  Found {len(boosted_data) if isinstance(boosted_data, list) else 1} boosted tokens")
                    
                    tokens_list = boosted_data if isinstance(boosted_data, list) else [boosted_data]
                    
                    for boost in tokens_list[:10]:
                        try:
                            # ✅ Normalize token chain too
                            token_chain = self._normalize_chain(boost.get('chainId', ''))
                            if token_chain != chain:
                                continue
                            
                            token_address = boost.get('tokenAddress')
                            if not token_address:
                                continue
                            
                            # ✅ Get pairs for this token
                            pairs_data = await self._make_request(
                                f"/token-pairs/v1/{chain}/{token_address}"
                            )
                            
                            if pairs_data and isinstance(pairs_data, list):
                                for pair_data in pairs_data[:2]:
                                    pair = self._parse_pair(pair_data)
                                    if pair and self._filter_pair(pair):
                                        pair_dict = self._pair_to_dict(pair)
                                        if pair_dict['pair_address'] not in seen_addresses:
                                            new_pairs.append(pair_dict)
                                            seen_addresses.add(pair_dict['pair_address'])
                                            self.stats['pairs_found'] += 1
                                            
                                            if len(new_pairs) >= limit:
                                                break
                        except Exception as e:
                            print(f"  Error processing boosted token: {e}")
                            continue
                        
                        if len(new_pairs) >= limit:
                            break
            except Exception as e:
                print(f"  Boosted tokens strategy failed: {e}")
            
            print(f"  Strategy 1 (Boosted): {len(new_pairs)} pairs")
            
            # ✅ STRATEGY 2: Latest token profiles (CORRECT endpoint)
            if len(new_pairs) < limit:
                try:
                    profiles_data = await self._make_request("/token-profiles/latest/v1")
                    
                    if profiles_data:
                        profiles_list = profiles_data if isinstance(profiles_data, list) else [profiles_data]
                        
                        for profile in profiles_list[:15]:
                            try:
                                profile_chain = profile.get('chainId', '').lower()
                                if profile_chain != chain.lower():
                                    continue
                                
                                token_address = profile.get('tokenAddress')
                                if not token_address or token_address in seen_addresses:
                                    continue
                                
                                # Get pairs for this token
                                pairs_data = await self._make_request(
                                    f"/token-pairs/v1/{chain}/{token_address}"
                                )
                                
                                if pairs_data and isinstance(pairs_data, list):
                                    for pair_data in pairs_data[:1]:  # Main pair only
                                        pair = self._parse_pair(pair_data)
                                        if pair and self._filter_pair(pair):
                                            pair_dict = self._pair_to_dict(pair)
                                            if pair_dict['pair_address'] not in seen_addresses:
                                                new_pairs.append(pair_dict)
                                                seen_addresses.add(pair_dict['pair_address'])
                                                self.stats['pairs_found'] += 1
                                                
                                                if len(new_pairs) >= limit:
                                                    break
                            except Exception as e:
                                continue
                            
                            if len(new_pairs) >= limit:
                                break
                except Exception as e:
                    print(f"  Token profiles strategy failed: {e}")
            
            print(f"  Strategy 2 (Profiles): {len(new_pairs)} pairs")
            
            # ✅ STRATEGY 3: Search common quote tokens (CORRECT endpoint)
            # ============================================
            # PATCH 3: Update search_pairs quote tokens for Solana (line ~310)
            # In the STRATEGY 3 section of get_new_pairs, update quote_tokens dict:
            # ============================================

            # ✅ STRATEGY 3: Search common quote tokens
            if len(new_pairs) < 10:
                quote_tokens = {
                    'ethereum': ['WETH', 'USDC', 'USDT'],
                    'bsc': ['WBNB', 'BUSD', 'USDT'],
                    'polygon': ['WMATIC', 'USDC'],
                    'arbitrum': ['WETH', 'USDC'],
                    'base': ['WETH', 'USDC'],
                    # ✅ ADD SOLANA:
                    'solana': ['SOL', 'USDC', 'USDT', 'RAY', 'BONK']
                }
                
                for quote in quote_tokens.get(chain, ['USDC'])[:2]:
                    if len(new_pairs) >= limit:
                        break
                    
                    try:
                        search_data = await self._make_request(
                            "/latest/dex/search",
                            params={'q': quote}
                        )
                        
                        if search_data and 'pairs' in search_data:
                            for pair_data in search_data['pairs'][:20]:
                                # ✅ Normalize chain for comparison
                                pair_chain = self._normalize_chain(pair_data.get('chainId', ''))
                                if pair_chain != chain:
                                    continue
                                
                                pair = self._parse_pair(pair_data)
                                if pair and pair.volume_24h > 10000 and self._filter_pair(pair):
                                    pair_dict = self._pair_to_dict(pair)
                                    if pair_dict['pair_address'] not in seen_addresses:
                                        new_pairs.append(pair_dict)
                                        seen_addresses.add(pair_dict['pair_address'])
                                        self.stats['pairs_found'] += 1
                                        
                                        if len(new_pairs) >= limit:
                                            break
                    except Exception as e:
                        print(f"  Search for {quote} failed: {e}")
                        continue
            
            print(f"  Strategy 3 (Search): {len(new_pairs)} pairs")
            
            # Sort by volume (highest first)
            new_pairs.sort(key=lambda p: p.get('volume_24h', 0), reverse=True)
            
            # Cache results
            final_pairs = new_pairs[:limit]
            self.cache[cache_key] = (final_pairs, time.time())
            
            # Add to monitoring
            for pair in final_pairs:
                self.monitored_pairs.add(pair['pair_address'])
            
            print(f"✅ Total pairs found for {chain}: {len(final_pairs)}")
            
            return final_pairs
            
        except Exception as e:
            print(f"❌ Error in get_new_pairs for {chain}: {e}")
            import traceback
            traceback.print_exc()
            return []
        
    # ============================================
    # PATCH 4: Update get_token_price to normalize chain (line ~370)
    # ============================================

    async def get_token_price(self, token_address: str, chain: str = 'ethereum') -> Optional[float]:
        """
        Get current token price
        
        Args:
            token_address: Token contract address
            chain: Blockchain network (ethereum, bsc, polygon, solana, etc.)
            
        Returns:
            Current price in USD or None
        """
        # ✅ Normalize chain name
        chain = self._normalize_chain(chain)
        
        # Check cache
        cache_key = f"price_{chain}_{token_address}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < 10:  # 10 second cache for prices
                return cached_data
        
        try:
            # ✅ Use CORRECT DexScreener endpoint with chain
            endpoint = f"latest/dex/tokens/{token_address}"
            data = await self._make_request(endpoint)
            
            if data and 'pairs' in data and len(data['pairs']) > 0:
                # Filter pairs for the correct chain
                chain_pairs = [
                    p for p in data['pairs'] 
                    if self._normalize_chain(p.get('chainId', '')) == chain
                ]
                
                if not chain_pairs:
                    # If no exact chain match, use any pair
                    chain_pairs = data['pairs']
                
                # Get price from most liquid pair
                pairs = sorted(
                    chain_pairs, 
                    key=lambda x: x.get('liquidity', {}).get('usd', 0), 
                    reverse=True
                )
                
                if pairs:
                    price_str = pairs[0].get('priceUsd')
                    
                    if price_str:
                        price = float(price_str)
                        self.cache[cache_key] = (price, time.time())
                        return price
            
            return None
            
        except Exception as e:
            print(f"Error getting token price for {token_address} on {chain}: {e}")
            return None
        
    async def get_pair_data(self, pair_address: str) -> Optional[Dict]:
        """
        Get detailed pair data
        
        Args:
            pair_address: Pair contract address
            
        Returns:
            Pair data dictionary or None
        """
        # Check cache
        cache_key = f"pair_{pair_address}"
        if cache_key in self.cache:
            cached_data, cached_time = self.cache[cache_key]
            if time.time() - cached_time < self.cache_duration:
                return cached_data
                
        # Fetch pair data
        endpoint = f"pairs/{pair_address}"
        data = await self._make_request(endpoint)
        
        if data and 'pair' in data:
            pair = self._parse_pair(data['pair'])
            if pair:
                pair_dict = self._pair_to_dict(pair)
                self.cache[cache_key] = (pair_dict, time.time())
                return pair_dict
                
        return None
        
    async def get_token_pairs(self, token_address: str) -> List[Dict]:
        """
        Get all pairs for a token
        
        Args:
            token_address: Token contract address
            
        Returns:
            List of pair data
        """
        endpoint = f"tokens/{token_address}"
        data = await self._make_request(endpoint)
        
        pairs = []
        if data and 'pairs' in data:
            for pair_data in data['pairs']:
                pair = self._parse_pair(pair_data)
                if pair:
                    pairs.append(self._pair_to_dict(pair))
                    
        return pairs
        
    async def search_pairs(self, query: str) -> List[Dict]:
        """
        Search for pairs by token name or symbol
        
        Args:
            query: Search query
            
        Returns:
            List of matching pairs
        """
        endpoint = "search"
        params = {'q': query}
        data = await self._make_request(endpoint, params)
        
        pairs = []
        if data and 'pairs' in data:
            for pair_data in data['pairs']:
                pair = self._parse_pair(pair_data)
                if pair and self._filter_pair(pair):
                    pairs.append(self._pair_to_dict(pair))
                    
        return pairs
        
    async def get_trending_pairs(self) -> List[Dict]:
        """
        Get trending pairs across all chains
        
        Returns:
            List of trending pairs
        """
        trending = []
        
        for chain in self.chains:
            endpoint = f"gainers/{chain}"
            data = await self._make_request(endpoint)
            
            if data and 'pairs' in data:
                for pair_data in data['pairs']:
                    pair = self._parse_pair(pair_data)
                    if pair and self._filter_pair(pair):
                        trending.append(self._pair_to_dict(pair))
                        
        # Sort by volume
        trending.sort(key=lambda x: x['volume_24h'], reverse=True)
        
        return trending[:100]  # Top 100
        
    async def get_boosts(self) -> List[Dict]:
        """
        Get boosted pairs (paid promotions)
        
        Returns:
            List of boosted pairs
        """
        endpoint = "boosts/active"
        data = await self._make_request(endpoint)
        
        boosted = []
        if data:
            for boost_data in data:
                if 'pair' in boost_data:
                    pair = self._parse_pair(boost_data['pair'])
                    if pair:
                        pair_dict = self._pair_to_dict(pair)
                        pair_dict['boost_amount'] = boost_data.get('amount', 0)
                        boosted.append(pair_dict)
                        
        return boosted

    # 3. Fix monitor_pair to accept address and chain (line ~371)
    # ADD this new method that matches the API signature:

    async def monitor_pair(self, address: str, chain: str = 'ethereum') -> AsyncGenerator:
        """
        Monitor a token pair for changes (AsyncGenerator version)
        
        Args:
            address: Token or pair address to monitor
            chain: Blockchain network
            
        Yields:
            Update events for the monitored pair
        """
        # First, find the pair address for this token
        pairs = await self.get_token_pairs(address)
        
        if not pairs:
            # If no pairs found, treat address as pair address
            pair_address = address
        else:
            # Use the most liquid pair
            pair_address = pairs[0].get('pair_address', address)
        
        self.monitored_pairs.add(pair_address)
        previous_data = None
        
        try:
            while pair_address in self.monitored_pairs:
                try:
                    current_data = await self.get_pair_data(pair_address)
                    
                    if current_data:
                        if previous_data:
                            # Check for significant changes
                            changes = self._detect_changes(previous_data, current_data)
                            
                            if changes:
                                yield {
                                    'type': 'update',
                                    'pair_address': pair_address,
                                    'chain': chain,
                                    'data': current_data,
                                    'changes': changes,
                                    'timestamp': time.time()
                                }
                        else:
                            # First data point
                            yield {
                                'type': 'initial',
                                'pair_address': pair_address,
                                'chain': chain,
                                'data': current_data,
                                'timestamp': time.time()
                            }
                        
                        previous_data = current_data
                        
                    await asyncio.sleep(5)  # Check every 5 seconds
                    
                except Exception as e:
                    yield {
                        'type': 'error',
                        'pair_address': pair_address,
                        'chain': chain,
                        'error': str(e),
                        'timestamp': time.time()
                    }
                    await asyncio.sleep(10)
                    
        finally:
            self.monitored_pairs.discard(pair_address)


    # Keep the original monitor_pair_with_callback for backward compatibility:
    async def monitor_pair_with_callback(self, pair_address: str, callback: callable = None):
        """
        Monitor a pair for changes with callback (original implementation)
        
        Args:
            pair_address: Pair to monitor
            callback: Function to call on updates
        """
        # Original implementation remains as is...
        # (Keep the existing code from the original monitor_pair method)
        self.monitored_pairs.add(pair_address)
        
        previous_data = None
        
        while pair_address in self.monitored_pairs:
            try:
                current_data = await self.get_pair_data(pair_address)
                
                if current_data and previous_data:
                    # Check for significant changes
                    changes = self._detect_changes(previous_data, current_data)
                    
                    if changes and callback:
                        await callback(pair_address, changes)
                        
                previous_data = current_data
                await asyncio.sleep(5)  # Check every 5 seconds
                
            except Exception as e:
                print(f"Error monitoring pair {pair_address}: {e}")
                await asyncio.sleep(10)
                
    def stop_monitoring(self, pair_address: str):
        """Stop monitoring a pair"""
        self.monitored_pairs.discard(pair_address)
        
    def _parse_pair(self, data: Dict) -> Optional[TokenPair]:
        """
        Parse raw pair data into TokenPair object
        
        Args:
            data: Raw pair data from API
            
        Returns:
            TokenPair object or None
        """
        try:
            # Extract base and quote tokens
            base_token = data.get('baseToken', {})
            quote_token = data.get('quoteToken', {})
            
            # Extract liquidity
            liquidity = data.get('liquidity', {})
            liquidity_usd = float(liquidity.get('usd', 0))
            liquidity_base = float(liquidity.get('base', 0))
            liquidity_quote = float(liquidity.get('quote', 0))
            
            # Extract volume
            volume = data.get('volume', {})
            volume_24h = float(volume.get('h24', 0))
            
            # Extract price changes
            price_change = data.get('priceChange', {})
            
            # Extract transaction counts
            txns = data.get('txns', {})
            h24_txns = txns.get('h24', {})
            
            # Create TokenPair object
            pair = TokenPair(
                chain_id=data.get('chainId', ''),
                dex_id=data.get('dexId', ''),
                pair_address=data.get('pairAddress', ''),
                base_token={
                    'address': base_token.get('address', ''),
                    'name': base_token.get('name', ''),
                    'symbol': base_token.get('symbol', '')
                },
                quote_token={
                    'address': quote_token.get('address', ''),
                    'name': quote_token.get('name', ''),
                    'symbol': quote_token.get('symbol', '')
                },
                price_native=float(data.get('priceNative', 0)),
                price_usd=float(data.get('priceUsd', 0)),
                liquidity_usd=liquidity_usd,
                liquidity_base=liquidity_base,
                liquidity_quote=liquidity_quote,
                volume_24h=volume_24h,
                volume_change_24h=float(volume.get('h24Change', 0)),
                price_change_24h=float(price_change.get('h24', 0)),
                price_change_1h=float(price_change.get('h1', 0)),
                price_change_5m=float(price_change.get('m5', 0)),
                trades_24h=h24_txns.get('buys', 0) + h24_txns.get('sells', 0),
                buyers_24h=h24_txns.get('buys', 0),
                sellers_24h=h24_txns.get('sells', 0),
                created_at=data.get('pairCreatedAt', 0),
                pair_created_at=datetime.fromtimestamp(data.get('pairCreatedAt', 0) / 1000) if data.get('pairCreatedAt') else datetime.now(),
                info=data.get('info', {})
            )
            
            return pair
            
        except Exception as e:
            print(f"Error parsing pair data: {e}")
            return None
            
    def _pair_to_dict(self, pair: TokenPair) -> Dict:
        """Convert TokenPair to dictionary"""
        return {
            'chain': pair.chain_id,
            'dex': pair.dex_id,
            'pair_address': pair.pair_address,
            'token_address': pair.base_token['address'],
            'token_name': pair.base_token['name'],
            'token_symbol': pair.base_token['symbol'],
            'price': pair.price_usd,
            'price_usd': pair.price_usd,          # ✅ ADD THIS
            'price_native': pair.price_native,
            'liquidity': pair.liquidity_usd,
            'liquidity_usd': pair.liquidity_usd,  # ✅ ADD THIS
            'volume_24h': pair.volume_24h,
            'volume_change_24h': pair.volume_change_24h,
            'price_change_24h': pair.price_change_24h,
            'price_change_1h': pair.price_change_1h,
            'price_change_5m': pair.price_change_5m,
            'trades_24h': pair.trades_24h,
            'buyers_24h': pair.buyers_24h,
            'sellers_24h': pair.sellers_24h,
            'buy_sell_ratio': pair.buy_sell_ratio,
            'age_hours': pair.age_hours,
            'created_at': pair.created_at,
            'creator_address': pair.info.get('creator', ''),
            'metadata': pair.info
        }
        
    # ============================================
    # PATCH 5: Update _filter_pair for Solana-specific thresholds (line ~680)
    # ============================================

    def _filter_pair(self, pair: TokenPair) -> bool:
        """
        Filter pairs based on criteria (chain-aware)
        
        Args:
            pair: TokenPair to filter
            
        Returns:
            True if pair passes filters
        """
        # ✅ Get chain-specific thresholds
        chain = self._normalize_chain(pair.chain_id)
        
        # Chain-specific minimum liquidity
        chain_min_liquidity = {
            'ethereum': 10000,
            'bsc': 1000,
            'polygon': 1000,
            'arbitrum': 10000,
            'base': 5000,
            'solana': 5000,  # ✅ Solana minimum liquidity
        }
        
        min_liquidity = chain_min_liquidity.get(chain, self.min_liquidity)
        
        # Check liquidity
        if pair.liquidity_usd < min_liquidity:
            return False
        
        # Chain-specific minimum volume
        chain_min_volume = {
            'ethereum': 5000,
            'bsc': 1000,
            'polygon': 1000,
            'arbitrum': 5000,
            'base': 2000,
            'solana': 2000,  # ✅ Solana minimum volume
        }
        
        min_volume = chain_min_volume.get(chain, self.min_volume)
        
        # Check volume
        if pair.volume_24h < min_volume:
            return False
        
        # Check age (Solana tokens can be newer)
        max_age = 48 if chain == 'solana' else self.max_age_hours  # ✅ Solana can be 48h
        if pair.age_hours > max_age:
            return False
        
        # Check for honeypot indicators
        if pair.buyers_24h == 0 and pair.sellers_24h > 0:
            return False  # Only sells, likely honeypot
        
        # Check price
        if pair.price_usd <= 0:
            return False
        
        return True
        
    def _detect_changes(self, previous: Dict, current: Dict) -> Optional[Dict]:
        """
        Detect significant changes in pair data
        
        Args:
            previous: Previous pair data
            current: Current pair data
            
        Returns:
            Dictionary of changes or None
        """
        changes = {}
        
        # Price changes
        prev_price = previous.get('price', 0)
        curr_price = current.get('price', 0)
        
        if prev_price > 0:
            price_change = (curr_price - prev_price) / prev_price
            
            if abs(price_change) > 0.05:  # 5% change
                changes['price_change'] = price_change
                changes['new_price'] = curr_price
                
        # Volume changes
        prev_volume = previous.get('volume_24h', 0)
        curr_volume = current.get('volume_24h', 0)
        
        if prev_volume > 0:
            volume_change = (curr_volume - prev_volume) / prev_volume
            
            if abs(volume_change) > 0.5:  # 50% change
                changes['volume_change'] = volume_change
                changes['new_volume'] = curr_volume
                
        # Liquidity changes
        prev_liquidity = previous.get('liquidity', 0)
        curr_liquidity = current.get('liquidity', 0)
        
        if prev_liquidity > 0:
            liquidity_change = (curr_liquidity - prev_liquidity) / prev_liquidity
            
            if abs(liquidity_change) > 0.2:  # 20% change
                changes['liquidity_change'] = liquidity_change
                changes['new_liquidity'] = curr_liquidity
                
                # Alert on significant liquidity removal
                if liquidity_change < -0.5:  # 50% liquidity removed
                    changes['alert'] = 'LIQUIDITY_REMOVAL'
                    
        # Buy/Sell ratio changes
        prev_ratio = previous.get('buy_sell_ratio', 0.5)
        curr_ratio = current.get('buy_sell_ratio', 0.5)
        
        if abs(curr_ratio - prev_ratio) > 0.2:
            changes['buy_sell_ratio_change'] = curr_ratio - prev_ratio
            changes['new_buy_sell_ratio'] = curr_ratio
            
        return changes if changes else None
        
    # 2. Fix get_price_history signature (line ~595)
    # REPLACE the existing method:

    async def get_price_history(self, address: str, chain: str = 'ethereum', interval: str = '5m') -> List[Dict]:
        """
        Get price history for a token
        
        Args:
            address: Token contract address
            chain: Blockchain network
            interval: Time interval (5m, 15m, 30m, 1h, 4h, 1d)
            
        Returns:
            List of OHLCV data
        """
        # Note: DexScreener doesn't provide historical data directly
        # This would need integration with another service or manual tracking
        
        # For now, we can get current price and build from recent data
        current_price = await self.get_token_price(address)
        
        if current_price:
            # Return simplified current data point
            return [{
                'timestamp': int(time.time() * 1000),
                'open': current_price,
                'high': current_price,
                'low': current_price,
                'close': current_price,
                'volume': 0,
                'chain': chain,
                'interval': interval
            }]
        
        return []
        
    async def calculate_metrics(self, pair_address: str) -> Dict:
        """
        Calculate advanced metrics for a pair
        
        Args:
            pair_address: Pair contract address
            
        Returns:
            Dictionary of calculated metrics
        """
        pair_data = await self.get_pair_data(pair_address)
        
        if not pair_data:
            return {}
            
        metrics = {
            'liquidity_to_mcap_ratio': 0,
            'volume_to_liquidity_ratio': 0,
            'price_impact_1_eth': 0,
            'holder_balance_score': 0,
            'momentum_score': 0,
            'risk_score': 0
        }
        
        # Liquidity to market cap ratio
        liquidity = pair_data.get('liquidity', 0)
        price = pair_data.get('price', 0)
        
        if liquidity > 0 and price > 0:
            # Estimate market cap (this is simplified)
            estimated_mcap = price * 1000000000  # Assume 1B supply
            metrics['liquidity_to_mcap_ratio'] = liquidity / estimated_mcap
            
        # Volume to liquidity ratio
        volume = pair_data.get('volume_24h', 0)
        if liquidity > 0:
            metrics['volume_to_liquidity_ratio'] = volume / liquidity
            
        # Price momentum score
        price_change_1h = pair_data.get('price_change_1h', 0)
        price_change_24h = pair_data.get('price_change_24h', 0)
        
        metrics['momentum_score'] = (price_change_1h * 0.7 + price_change_24h * 0.3) / 100
        
        # Risk score (simplified)
        age_hours = pair_data.get('age_hours', 0)
        buy_sell_ratio = pair_data.get('buy_sell_ratio', 0.5)
        
        risk_factors = []
        
        if liquidity < 50000:
            risk_factors.append(0.3)
        if age_hours < 24:
            risk_factors.append(0.2)
        if buy_sell_ratio < 0.3 or buy_sell_ratio > 0.7:
            risk_factors.append(0.2)
        if volume < 10000:
            risk_factors.append(0.3)
            
        metrics['risk_score'] = min(sum(risk_factors), 1.0)
        
        return metrics
        
    def get_stats(self) -> Dict:
        """Get collector statistics"""
        return self.stats.copy()

    # ============================================================================
    # PATCH FOR: dexscreener.py
    # Add these methods to the DexScreenerCollector class
    # ============================================================================

    # ============================================
    # PATCH 6: Update get_token_info to normalize chain (line ~880)
    # ============================================

    async def get_token_info(self, address: str, chain: str = 'ethereum') -> Dict:
        """
        Get token information
        
        Args:
            address: Token contract address
            chain: Blockchain network
            
        Returns:
            Token information dictionary
        """
        # ✅ Normalize chain
        chain = self._normalize_chain(chain)
        
        # Use existing get_token_price and extend it
        price = await self.get_token_price(address, chain)
        pairs = await self.get_token_pairs(address)
        
        if pairs and len(pairs) > 0:
            # Get most liquid pair info
            main_pair = pairs[0]
            
            return {
                'address': address,
                'chain': chain,
                'name': main_pair.get('token_name', 'Unknown'),
                'symbol': main_pair.get('token_symbol', '???'),
                'price': price or 0,
                'liquidity': main_pair.get('liquidity', 0),
                'volume_24h': main_pair.get('volume_24h', 0),
                'price_change_24h': main_pair.get('price_change_24h', 0),
                'market_cap': 0,
                'pairs_count': len(pairs),
                'main_pair': main_pair.get('pair_address', ''),
                'created_at': main_pair.get('created_at', 0)
            }
        
        return {
            'address': address,
            'chain': chain,
            'name': 'Unknown',
            'symbol': '???',
            'price': price or 0,
            'liquidity': 0,
            'volume_24h': 0,
            'price_change_24h': 0,
            'market_cap': 0,
            'pairs_count': 0,
            'main_pair': '',
            'created_at': 0
        }

    async def get_trending_tokens(self, chain: str = 'ethereum') -> List[Dict]:
        """
        Get trending tokens for a chain
        
        Args:
            chain: Blockchain network
            
        Returns:
            List of trending token data
        """
        # Use get_trending_pairs and extract unique tokens
        trending_pairs = await self.get_trending_pairs()
        
        # Extract unique tokens from pairs
        seen_tokens = set()
        trending_tokens = []
        
        for pair in trending_pairs:
            token_address = pair.get('token_address', '')
            
            if token_address and token_address not in seen_tokens:
                seen_tokens.add(token_address)
                
                trending_tokens.append({
                    'address': token_address,
                    'chain': pair.get('chain', chain),
                    'name': pair.get('token_name', 'Unknown'),
                    'symbol': pair.get('token_symbol', '???'),
                    'price': pair.get('price', 0),
                    'price_change_24h': pair.get('price_change_24h', 0),
                    'volume_24h': pair.get('volume_24h', 0),
                    'liquidity': pair.get('liquidity', 0),
                    'trending_score': pair.get('volume_24h', 0) * (1 + pair.get('price_change_24h', 0) / 100)
                })
        
        # Filter for requested chain
        if chain:
            trending_tokens = [t for t in trending_tokens if t.get('chain') == chain]
        
        # Sort by trending score
        trending_tokens.sort(key=lambda x: x.get('trending_score', 0), reverse=True)
        
        return trending_tokens

    async def get_gainers_losers(self, chain: str = 'ethereum', period: str = '24h') -> Dict:
        """
        Get top gainers and losers
        
        Args:
            chain: Blockchain network
            period: Time period (1h, 24h, 7d)
            
        Returns:
            Dictionary with gainers and losers
        """
        # Fetch gainers
        gainers_endpoint = f"gainers/{chain}"
        gainers_data = await self._make_request(gainers_endpoint)
        
        gainers = []
        losers = []
        
        if gainers_data and 'pairs' in gainers_data:
            for pair_data in gainers_data['pairs']:
                pair = self._parse_pair(pair_data)
                if pair:
                    pair_dict = self._pair_to_dict(pair)
                    
                    # Determine price change based on period
                    price_change = 0
                    if period == '1h':
                        price_change = pair_dict.get('price_change_1h', 0)
                    elif period == '24h':
                        price_change = pair_dict.get('price_change_24h', 0)
                    else:
                        price_change = pair_dict.get('price_change_24h', 0)
                    
                    token_info = {
                        'address': pair_dict.get('token_address', ''),
                        'name': pair_dict.get('token_name', ''),
                        'symbol': pair_dict.get('token_symbol', ''),
                        'price': pair_dict.get('price', 0),
                        'price_change': price_change,
                        'volume': pair_dict.get('volume_24h', 0),
                        'liquidity': pair_dict.get('liquidity', 0)
                    }
                    
                    if price_change > 0:
                        gainers.append(token_info)
                    else:
                        losers.append(token_info)
        
        # Sort by price change
        gainers.sort(key=lambda x: x['price_change'], reverse=True)
        losers.sort(key=lambda x: x['price_change'])
        
        return {
            'period': period,
            'chain': chain,
            'gainers': gainers[:20],  # Top 20 gainers
            'losers': losers[:20],    # Top 20 losers
            'timestamp': time.time()
        }

        
async def test_api_connection():
    """Test DexScreener API connectivity"""
    collector = DexScreenerCollector({'api_key': '', 'chains': ['ethereum']})
    await collector.initialize()
    
    try:
        # Try to fetch some data
        pairs = await collector.get_new_pairs(limit=1)
        if pairs:
            print("DexScreener API connection successful")
            return True
        else:
            print("DexScreener API returned no data")
            return False
            
    except Exception as e:
        print(f"DexScreener API connection failed: {e}")
        return False
        
    finally:
        await collector.close()