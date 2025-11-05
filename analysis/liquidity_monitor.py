# analysis/liquidity_monitor.py

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
import aiohttp

from core.event_bus import EventBus
from data.storage.database import DatabaseManager
from data.storage.cache import CacheManager
from monitoring.alerts import AlertsSystem
from config.config_manager import ConfigManager

logger = logging.getLogger(__name__)

@dataclass
class LiquidityEvent:
    """Represents a liquidity change event"""
    token: str
    chain: str
    pool_address: str
    event_type: str  # 'add', 'remove', 'lock', 'unlock'
    amount_usd: Decimal
    percentage_change: float
    timestamp: datetime
    transaction_hash: str
    wallet_address: str
    is_suspicious: bool = False
    risk_score: float = 0.0
    metadata: Dict = field(default_factory=dict)

@dataclass
class SlippageEstimate:
    """Slippage calculation result"""
    token: str
    amount_in: Decimal
    expected_slippage: float
    max_slippage: float
    price_impact: float
    effective_price: Decimal
    liquidity_depth: Decimal
    recommendation: str
    risk_level: str  # 'low', 'medium', 'high', 'extreme'

class LiquidityMonitor:
    """Advanced liquidity monitoring and analysis"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        cache_manager: CacheManager,
        event_bus: EventBus,
        alerts: AlertsSystem,
        config: Dict[str, Any]
    ):
        self.db = db_manager
        self.cache = cache_manager
        self.event_bus = event_bus
        self.alerts = alerts
        self.config = config
        
        # Monitoring parameters
        self.min_liquidity_usd = Decimal(str(config.get('min_liquidity_usd', 50000)))
        self.removal_threshold = config.get('removal_threshold', 0.2)  # 20% removal triggers alert
        self.lock_check_interval = config.get('lock_check_interval', 300)  # 5 minutes
        self.slippage_warning_threshold = config.get('slippage_warning', 0.05)  # 5%
        
        # Tracking data
        self.monitored_pools: Dict[str, Dict] = {}
        self.liquidity_history: Dict[str, List] = {}
        self.removal_patterns: Dict[str, List] = {}
        
        # API endpoints
        self.dexscreener_api = "https://api.dexscreener.com"
        self.defined_api = "https://api.defined.fi"
        
        # Running tasks
        self._monitoring_tasks: List[asyncio.Task] = []
        
    async def start_monitoring(self, tokens: List[str]) -> None:
        """Start monitoring liquidity for multiple tokens"""
        try:
            logger.info(f"Starting liquidity monitoring for {len(tokens)} tokens")
            
            for token in tokens:
                if token not in self.monitored_pools:
                    task = asyncio.create_task(self._monitor_token_liquidity(token))
                    self._monitoring_tasks.append(task)
                    
            # Start periodic liquidity health checks
            asyncio.create_task(self._periodic_health_check())
            
        except Exception as e:
            logger.error(f"Error starting liquidity monitoring: {e}")
            
    async def stop_monitoring(self, token: Optional[str] = None) -> None:
        """Stop monitoring specific token or all tokens"""
        if token:
            self.monitored_pools.pop(token, None)
        else:
            for task in self._monitoring_tasks:
                task.cancel()
            self._monitoring_tasks.clear()
            self.monitored_pools.clear()
            
    async def monitor_liquidity_changes(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """Monitor real-time liquidity changes for a token"""
        try:
            # Get current liquidity data
            current_liquidity = await self._get_current_liquidity(token, chain)
            
            # Get historical data from cache or database
            cache_key = f"liquidity:history:{chain}:{token}"
            history = await self.cache.get(cache_key, [])
            
            if not history:
                # Fetch from database if not in cache
                history = await self.db.get_liquidity_history(token, chain, hours=24)
                
            # Calculate changes
            changes = await self._calculate_liquidity_changes(
                current_liquidity,
                history
            )
            
            # Detect anomalies
            anomalies = await self._detect_liquidity_anomalies(changes, history)
            
            # Update tracking
            self.liquidity_history[token] = history[-100:] + [current_liquidity]
            await self.cache.set(cache_key, self.liquidity_history[token], ttl=3600)
            
            # Emit events for significant changes
            if changes['percentage_change'] > 0.1:  # 10% change
                await self._emit_liquidity_event(token, chain, changes)
                
            return {
                'token': token,
                'chain': chain,
                'current_liquidity': current_liquidity,
                'changes': changes,
                'anomalies': anomalies,
                'health_score': await self._calculate_liquidity_health(current_liquidity),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring liquidity changes for {token}: {e}")
            return {}
            
    async def detect_liquidity_removal(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """Detect potential liquidity removal or rug pull patterns"""
        try:
            # Get recent liquidity transactions
            recent_txs = await self._get_liquidity_transactions(token, chain)
            
            # Analyze removal patterns
            removal_analysis = {
                'total_removed_24h': Decimal('0'),
                'removal_count': 0,
                'large_removals': [],
                'suspicious_wallets': [],
                'removal_velocity': 0.0,
                'risk_score': 0.0
            }
            
            current_time = datetime.utcnow()
            time_24h_ago = current_time - timedelta(hours=24)
            
            for tx in recent_txs:
                if tx['type'] == 'remove' and tx['timestamp'] > time_24h_ago:
                    removal_analysis['total_removed_24h'] += Decimal(str(tx['amount_usd']))
                    removal_analysis['removal_count'] += 1
                    
                    # Check for large removals
                    if tx['amount_usd'] > float(self.min_liquidity_usd) * 0.1:
                        removal_analysis['large_removals'].append({
                            'amount': tx['amount_usd'],
                            'wallet': tx['wallet'],
                            'timestamp': tx['timestamp'],
                            'percentage': tx.get('pool_percentage', 0)
                        })
                        
                    # Track suspicious wallets
                    if await self._is_suspicious_wallet(tx['wallet']):
                        removal_analysis['suspicious_wallets'].append(tx['wallet'])
                        
            # Calculate removal velocity (removals per hour)
            if removal_analysis['removal_count'] > 0:
                hours_elapsed = min(24, (current_time - recent_txs[-1]['timestamp']).total_seconds() / 3600)
                removal_analysis['removal_velocity'] = removal_analysis['removal_count'] / max(hours_elapsed, 1)
                
            # Calculate risk score
            removal_analysis['risk_score'] = await self._calculate_removal_risk_score(removal_analysis)
            
            # Check for rug pull patterns
            rug_patterns = await self._detect_rug_patterns(token, chain, removal_analysis)
            
            # Alert if high risk
            if removal_analysis['risk_score'] > 0.7:
                await self._send_liquidity_alert(token, chain, removal_analysis, 'high')
                
            return {
                'token': token,
                'chain': chain,
                'removal_analysis': removal_analysis,
                'rug_patterns': rug_patterns,
                'recommendation': self._get_removal_recommendation(removal_analysis),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error detecting liquidity removal for {token}: {e}")
            return {}
            
    # Fix 2: calculate_slippage - Change to synchronous with correct parameters
    def calculate_slippage(self, amount: Decimal, liquidity_data: Dict) -> Decimal:
        """
        Calculate expected slippage for a trade
        
        Args:
            amount: Trade amount
            liquidity_data: Liquidity depth data containing reserves
            
        Returns:
            Expected slippage as Decimal
        """
        try:
            # Extract reserves from liquidity data
            reserve_in = Decimal(str(liquidity_data.get('reserve_token0', 0)))
            reserve_out = Decimal(str(liquidity_data.get('reserve_token1', 0)))
            
            if reserve_in == 0 or reserve_out == 0:
                max_slippage_bps = self.config.get('trading.max_slippage_bps', 100)  # Default 1%
                return Decimal(str(max_slippage_bps / 10_000))  # Convert bps to decimal
            
            # AMM formula with fee (constant product)
            #amount_with_fee = amount * Decimal('0.997')  # 0.3% fee
            dex_fee_bps = self.config.get('trading.dex_fee_bps', 30)  # Default 0.3%
            fee_multiplier = Decimal(str(1 - (dex_fee_bps / 10_000)))
            amount_with_fee = amount * fee_multiplier
            amount_out = (amount_with_fee * reserve_out) / (reserve_in + amount_with_fee)
            
            # Calculate slippage
            spot_price = reserve_out / reserve_in
            effective_price = amount / amount_out if amount_out > 0 else Decimal('0')
            
            slippage = abs(effective_price - spot_price) / spot_price if spot_price > 0 else Decimal('0')
            
            return slippage
            
        except Exception as e:
            logger.error(f"Error calculating slippage: {e}")
            max_slippage_bps = self.config.get('trading.max_slippage_bps', 100)  # Default 1%
            return Decimal(str(max_slippage_bps / 10_000))  # Convert bps to decimal


    # Keep the async version with different name for internal use:
    async def calculate_slippage_async(
        self,
        token: str,
        chain: str,
        amount: Decimal,
        is_buy: bool = True
    ) -> SlippageEstimate:
        """
        Calculate expected slippage for a trade (async extended version)
        
        Args:
            token: Token address
            chain: Blockchain network
            amount: Trade amount
            is_buy: Whether this is a buy order
            
        Returns:
            Complete SlippageEstimate object
        """
        try:
            # Get liquidity depth
            liquidity_data = await self._get_liquidity_depth(token, chain)
            
            # Get order book or AMM curve data
            if liquidity_data.get('type') == 'orderbook':
                slippage_data = await self._calculate_orderbook_slippage(
                    liquidity_data, amount, is_buy
                )
            else:  # AMM
                slippage_data = await self._calculate_amm_slippage(
                    liquidity_data, amount, is_buy
                )
                
            # Calculate price impact
            price_impact = await self._calculate_price_impact(
                liquidity_data, amount, is_buy
            )
            
            # Determine risk level
            risk_level = self._determine_slippage_risk(slippage_data['expected_slippage'])
            
            # Get recommendation
            recommendation = self._get_slippage_recommendation(
                slippage_data['expected_slippage'],
                price_impact,
                liquidity_data['depth']
            )
            

            max_slippage_bps = self.config.get('trading.max_slippage_bps', 100)
            max_slippage_decimal = max_slippage_bps / 10_000
            return SlippageEstimate(
                token=token,
                amount_in=amount,
                expected_slippage=max_slippage_decimal,
                max_slippage=slippage_data['max_slippage'],
                price_impact=price_impact,
                effective_price=slippage_data['effective_price'],
                liquidity_depth=liquidity_data['depth'],
                recommendation=recommendation,
                risk_level=risk_level
            )
            
        except Exception as e:
            logger.error(f"Error calculating slippage for {token}: {e}")
            max_slippage_bps = self.config.get('trading.max_slippage_bps', 100)
            max_slippage_decimal = max_slippage_bps / 10_000
            return SlippageEstimate(
                token=token,
                amount_in=amount,
                expected_slippage=max_slippage_decimal,
                max_slippage=max_slippage_decimal,
                price_impact=max_slippage_decimal,
                effective_price=Decimal('0'),
                liquidity_depth=Decimal('0'),
                recommendation="Unable to calculate - avoid trade",
                risk_level='extreme'
            )


            
    async def analyze_liquidity_locks(
        self,
        token: str,
        chain: str
    ) -> Dict[str, Any]:
        """Analyze liquidity lock status and duration"""
        try:
            lock_data = {
                'is_locked': False,
                'lock_percentage': 0.0,
                'unlock_date': None,
                'lock_platform': None,
                'lock_transaction': None,
                'remaining_days': 0,
                'trust_score': 0.0
            }
            
            # Check multiple lock platforms
            for platform in ['unicrypt', 'team.finance', 'pinksale', 'dxsale']:
                platform_lock = await self._check_lock_platform(token, chain, platform)
                if platform_lock and platform_lock['amount'] > 0:
                    lock_data['is_locked'] = True
                    lock_data['lock_percentage'] = platform_lock['percentage']
                    lock_data['unlock_date'] = platform_lock['unlock_date']
                    lock_data['lock_platform'] = platform
                    lock_data['lock_transaction'] = platform_lock.get('tx_hash')
                    
                    if platform_lock['unlock_date']:
                        remaining = (platform_lock['unlock_date'] - datetime.utcnow()).days
                        lock_data['remaining_days'] = max(0, remaining)
                        
                    break
                    
            # Calculate trust score based on lock
            if lock_data['is_locked']:
                lock_data['trust_score'] = min(1.0, (
                    lock_data['lock_percentage'] / 100 * 0.5 +
                    min(lock_data['remaining_days'] / 365, 1.0) * 0.5
                ))
                
            return lock_data
            
        except Exception as e:
            logger.error(f"Error analyzing liquidity locks for {token}: {e}")
            return {}
            
    async def get_liquidity_providers(
        self,
        token: str,
        chain: str
    ) -> List[Dict]:
        """Get list of liquidity providers and their shares"""
        try:
            providers = []
            
            # Get pool data
            pool_data = await self._get_pool_data(token, chain)
            if not pool_data:
                return providers
                
            # Get LP token holders
            lp_holders = await self._get_lp_token_holders(
                pool_data['lp_token_address'],
                chain
            )
            
            total_supply = sum(h['balance'] for h in lp_holders)
            
            for holder in lp_holders[:50]:  # Top 50 LPs
                share = (holder['balance'] / total_supply) * 100 if total_supply > 0 else 0
                
                providers.append({
                    'address': holder['address'],
                    'balance': holder['balance'],
                    'share_percentage': share,
                    'value_usd': holder['balance'] * pool_data['lp_price_usd'],
                    'is_locked': await self._is_lp_locked(holder['address']),
                    'first_provided': holder.get('first_tx_date'),
                    'last_activity': holder.get('last_tx_date')
                })
                
            return sorted(providers, key=lambda x: x['share_percentage'], reverse=True)
            
        except Exception as e:
            logger.error(f"Error getting liquidity providers for {token}: {e}")
            return []
            
    # Private helper methods
    
    async def _monitor_token_liquidity(self, token: str) -> None:
        """Continuous monitoring task for a token"""
        while token in self.monitored_pools or token in self.monitored_pools.keys():
            try:
                # Monitor changes
                changes = await self.monitor_liquidity_changes(token)
                
                # Check for removals
                if changes.get('changes', {}).get('percentage_change', 0) < -self.removal_threshold:
                    removal_data = await self.detect_liquidity_removal(token)
                    
                # Store monitoring data
                await self.db.save_liquidity_snapshot({
                    'token': token,
                    'liquidity_usd': changes.get('current_liquidity', {}).get('usd_value'),
                    'change_24h': changes.get('changes', {}).get('change_24h'),
                    'health_score': changes.get('health_score'),
                    'timestamp': datetime.utcnow()
                })
                
                await asyncio.sleep(self.lock_check_interval)
                
            except Exception as e:
                logger.error(f"Error in liquidity monitoring task for {token}: {e}")
                await asyncio.sleep(60)
                
    async def _calculate_amm_slippage(
        self,
        liquidity_data: Dict,
        amount: Decimal,
        is_buy: bool
    ) -> Dict[str, Any]:
        """Calculate slippage for AMM pools using x*y=k formula"""
        try:
            reserve_in = Decimal(str(liquidity_data['reserve_token0' if is_buy else 'reserve_token1']))
            reserve_out = Decimal(str(liquidity_data['reserve_token1' if is_buy else 'reserve_token0']))
            
            # Calculate output amount
            #amount_with_fee = amount * Decimal('0.997')  # 0.3% fee
            dex_fee_bps = self.config.get('trading.dex_fee_bps', 30)  # Default 0.3%
            fee_multiplier = Decimal(str(1 - (dex_fee_bps / 10_000)))
            amount_with_fee = amount * fee_multiplier
            amount_out = (amount_with_fee * reserve_out) / (reserve_in + amount_with_fee)
            
            # Calculate price impact
            spot_price = reserve_out / reserve_in
            effective_price = amount / amount_out if amount_out > 0 else Decimal('0')
            
            slippage = float(abs(effective_price - spot_price) / spot_price) if spot_price > 0 else 0
            
            # Calculate max slippage (for 2x the amount)
            amount_2x = amount * 2
            #amount_with_fee_2x = amount_2x * Decimal('0.997')
            dex_fee_bps = self.config.get('trading.dex_fee_bps', 30)  # Default 0.3%
            fee_multiplier = Decimal(str(1 - (dex_fee_bps / 10_000)))
            amount_with_fee_2x = amount_2x * fee_multiplier
            amount_out_2x = (amount_with_fee_2x * reserve_out) / (reserve_in + amount_with_fee_2x)
            effective_price_2x = amount_2x / amount_out_2x if amount_out_2x > 0 else Decimal('0')
            max_slippage = float(abs(effective_price_2x - spot_price) / spot_price) if spot_price > 0 else 0
            
            return {
                'expected_slippage': slippage,
                'max_slippage': max_slippage,
                'effective_price': effective_price,
                'amount_out': amount_out
            }
            
        except Exception as e:
            logger.error(f"Error calculating AMM slippage: {e}")
            max_slippage_bps = self.config.get('trading.max_slippage_bps', 100)
            max_slippage_decimal = max_slippage_bps / 10_000
            return {
                'expected_slippage': max_slippage_decimal,
                'max_slippage': max_slippage_decimal,
                'effective_price': Decimal('0'),
                'amount_out': Decimal('0')
            }
            
    def _determine_slippage_risk(self, slippage: float) -> str:
        """Determine risk level based on slippage"""
        if slippage < 0.01:
            return 'low'
        elif slippage < 0.03:
            return 'medium'
        elif slippage < 0.1:
            return 'high'
        else:
            return 'extreme'
            
    def _get_slippage_recommendation(
        self,
        slippage: float,
        price_impact: float,
        liquidity_depth: Decimal
    ) -> str:
        """Get trading recommendation based on slippage analysis"""
        if slippage > 0.1:
            return "AVOID: Extreme slippage detected. Trade not recommended."
        elif slippage > 0.05:
            return "CAUTION: High slippage. Consider smaller trade size or multiple orders."
        elif slippage > 0.02:
            return "MODERATE: Acceptable slippage. Use limit orders for better execution."
        else:
            return "GOOD: Low slippage expected. Safe to execute."
            
    async def _send_liquidity_alert(
        self,
        token: str,
        chain: str,
        analysis: Dict,
        severity: str
    ) -> None:
        """Send alert for liquidity issues"""
        message = f"""
            🚨 **Liquidity Alert - {severity.upper()}**

            Token: {token}
            Chain: {chain}
            Risk Score: {analysis.get('risk_score', 0):.2f}
            Total Removed (24h): ${analysis.get('total_removed_24h', 0):,.2f}
            Large Removals: {len(analysis.get('large_removals', []))}
            Removal Velocity: {analysis.get('removal_velocity', 0):.1f}/hour

            Recommendation: {analysis.get('recommendation', 'Monitor closely')}
            """
                    
        await self.alerts.send_alert(
                message=message,
                level=severity,
                channels=['telegram', 'discord']
            )

    async def monitor_liquidity(self, token: str, chain: str = 'ethereum') -> Dict:
        """
        Monitor liquidity for a token (wrapper for monitor_liquidity_changes)
        
        Args:
            token: Token to monitor
            chain: Blockchain network
            
        Returns:
            Liquidity monitoring data
        """
        # Use existing monitor_liquidity_changes method
        return await self.monitor_liquidity_changes(token, chain)

    # ============================================================================
    # FIXES FOR: liquidity_monitor.py
    # ============================================================================

    # Fix 1: get_liquidity_depth - Remove chain parameter
    async def get_liquidity_depth(self, pair_address: str) -> Dict:
        """
        Get liquidity depth for a pair
        
        Args:
            pair_address: DEX pair address
            
        Returns:
            Liquidity depth data
        """
        try:
            # Default to ethereum for backward compatibility
            chain = 'ethereum'
            
            # Fetch pair data
            pair_data = await self._get_pool_data(pair_address, chain)
            
            if not pair_data:
                return {
                    'depth': Decimal('0'),
                    'type': 'unknown',
                    'reserve_token0': 0,
                    'reserve_token1': 0,
                    'total_liquidity': 0
                }
            
            # Calculate depth based on reserves
            reserve0 = pair_data.get('reserve0', 0)
            reserve1 = pair_data.get('reserve1', 0)
            price0 = pair_data.get('token0_price', 1)
            price1 = pair_data.get('token1_price', 1)
            
            total_liquidity = (reserve0 * price0) + (reserve1 * price1)
            
            return {
                'depth': Decimal(str(total_liquidity)),
                'type': 'amm',  # or 'orderbook'
                'reserve_token0': reserve0,
                'reserve_token1': reserve1,
                'total_liquidity': total_liquidity,
                'pair_address': pair_address,
                'lp_price_usd': pair_data.get('lp_price_usd', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting liquidity depth: {e}")
            return {
                'depth': Decimal('0'),
                'type': 'unknown',
                'error': str(e)
            }


    # Keep the extended version for internal use:
    async def get_liquidity_depth_extended(self, pair_address: str, chain: str = 'ethereum') -> Dict:
        """
        Get liquidity depth for a pair with chain specification (internal use)
        
        Args:
            pair_address: DEX pair address
            chain: Blockchain network
            
        Returns:
            Liquidity depth data
        """
        # Original implementation with chain parameter
        try:
            pair_data = await self._get_pool_data(pair_address, chain)
            
            if not pair_data:
                return {
                    'depth': Decimal('0'),
                    'type': 'unknown',
                    'reserve_token0': 0,
                    'reserve_token1': 0,
                    'total_liquidity': 0
                }
            
            reserve0 = pair_data.get('reserve0', 0)
            reserve1 = pair_data.get('reserve1', 0)
            price0 = pair_data.get('token0_price', 1)
            price1 = pair_data.get('token1_price', 1)
            
            total_liquidity = (reserve0 * price0) + (reserve1 * price1)
            
            return {
                'depth': Decimal(str(total_liquidity)),
                'type': 'amm',
                'reserve_token0': reserve0,
                'reserve_token1': reserve1,
                'total_liquidity': total_liquidity,
                'pair_address': pair_address,
                'lp_price_usd': pair_data.get('lp_price_usd', 0)
            }
            
        except Exception as e:
            logger.error(f"Error getting liquidity depth: {e}")
            return {'depth': Decimal('0'), 'type': 'unknown', 'error': str(e)}


    # Fix 3: track_liquidity_changes - Remove chain parameter
    async def track_liquidity_changes(self, token: str):
        """
        Track liquidity changes as AsyncGenerator
        
        Args:
            token: Token to track
            
        Yields:
            Liquidity change events
        """
        # Default to ethereum for API compatibility
        chain = 'ethereum'
        
        self.monitored_pools[token] = {'chain': chain, 'started': datetime.utcnow()}
        
        try:
            while token in self.monitored_pools:
                # Monitor changes
                changes = await self.monitor_liquidity_changes(token, chain)
                
                # Yield if significant changes detected
                if changes and changes.get('changes', {}).get('percentage_change'):
                    yield {
                        'token': token,
                        'chain': chain,
                        'event_type': 'liquidity_change',
                        'changes': changes['changes'],
                        'current_liquidity': changes['current_liquidity'],
                        'anomalies': changes.get('anomalies', []),
                        'timestamp': datetime.utcnow()
                    }
                
                # Check for removal events
                removal_data = await self.detect_liquidity_removal(token, chain)
                if removal_data and removal_data.get('removal_analysis', {}).get('risk_score', 0) > 0.5:
                    yield {
                        'token': token,
                        'chain': chain,
                        'event_type': 'liquidity_removal',
                        'removal_data': removal_data['removal_analysis'],
                        'risk_score': removal_data['removal_analysis']['risk_score'],
                        'timestamp': datetime.utcnow()
                    }
                
                await asyncio.sleep(self.lock_check_interval)
                
        except GeneratorExit:
            # Clean up when generator is closed
            self.monitored_pools.pop(token, None)
        except Exception as e:
            logger.error(f"Error tracking liquidity changes: {e}")
            self.monitored_pools.pop(token, None)


    # Keep the extended version for internal use:
    async def track_liquidity_changes_extended(self, token: str, chain: str = 'ethereum'):
        """
        Track liquidity changes with chain specification (internal use)
        
        Args:
            token: Token to track
            chain: Blockchain network
            
        Yields:
            Liquidity change events
        """
        self.monitored_pools[token] = {'chain': chain, 'started': datetime.utcnow()}
        
        try:
            while token in self.monitored_pools:
                changes = await self.monitor_liquidity_changes(token, chain)
                
                if changes and changes.get('changes', {}).get('percentage_change'):
                    yield {
                        'token': token,
                        'chain': chain,
                        'event_type': 'liquidity_change',
                        'changes': changes['changes'],
                        'current_liquidity': changes['current_liquidity'],
                        'anomalies': changes.get('anomalies', []),
                        'timestamp': datetime.utcnow()
                    }
                
                removal_data = await self.detect_liquidity_removal(token, chain)
                if removal_data and removal_data.get('removal_analysis', {}).get('risk_score', 0) > 0.5:
                    yield {
                        'token': token,
                        'chain': chain,
                        'event_type': 'liquidity_removal',
                        'removal_data': removal_data['removal_analysis'],
                        'risk_score': removal_data['removal_analysis']['risk_score'],
                        'timestamp': datetime.utcnow()
                    }
                
                await asyncio.sleep(self.lock_check_interval)
                
        except GeneratorExit:
            self.monitored_pools.pop(token, None)
        except Exception as e:
            logger.error(f"Error tracking liquidity changes: {e}")
            self.monitored_pools.pop(token, None)
            
    # Sync wrapper for calculate_slippage (if needed for API compatibility)
    def calculate_slippage_sync(self, amount: Decimal, liquidity_data: Dict) -> Decimal:
        """
        Synchronous wrapper for calculate_slippage
        
        Args:
            amount: Trade amount
            liquidity_data: Liquidity depth data
            
        Returns:
            Expected slippage as Decimal
        """
        # Run async function synchronously
        import asyncio
        loop = asyncio.get_event_loop()
        
        # Create a simplified call to async calculate_slippage
        result = loop.run_until_complete(
            self.calculate_slippage(
                token='',  # Not needed for calculation
                chain='ethereum',
                amount=amount,
                is_buy=True
            )
        )
        
        return Decimal(str(result.expected_slippage))

    # ============================================================================
    # ADDITIONAL HELPER METHODS FOR LIQUIDITY MONITOR
    # ============================================================================

    # Add these stub methods that are referenced but might be missing:

    async def _get_current_liquidity(self, token: str, chain: str) -> Dict:
        """Get current liquidity data for token"""
        # This would fetch from DEX APIs
        return {
            'usd_value': 0,
            'token_amount': 0,
            'pair_address': '',
            'timestamp': datetime.utcnow()
        }

    async def _calculate_liquidity_changes(self, current: Dict, history: List) -> Dict:
        """Calculate liquidity changes"""
        if not history:
            return {'percentage_change': 0, 'absolute_change': 0}
        
        prev = history[-1] if history else current
        prev_value = prev.get('usd_value', 0)
        curr_value = current.get('usd_value', 0)
        
        change = curr_value - prev_value
        pct_change = (change / prev_value) if prev_value > 0 else 0
        
        return {
            'percentage_change': pct_change,
            'absolute_change': change,
            'change_24h': change
        }

    async def _detect_liquidity_anomalies(self, changes: Dict, history: List) -> List:
        """Detect anomalies in liquidity"""
        anomalies = []
        
        if abs(changes.get('percentage_change', 0)) > 0.5:
            anomalies.append('Large sudden change detected')
        
        return anomalies

    async def _calculate_liquidity_health(self, liquidity: Dict) -> float:
        """Calculate liquidity health score"""
        score = 0.5  # Base score
        
        usd_value = liquidity.get('usd_value', 0)
        if usd_value > 1000000:
            score += 0.3
        elif usd_value > 100000:
            score += 0.2
        elif usd_value > 10000:
            score += 0.1
        
        return min(1.0, score)

    async def _emit_liquidity_event(self, token: str, chain: str, changes: Dict) -> None:
        """Emit liquidity event"""
        await self.event_bus.emit('liquidity.change', {
            'token': token,
            'chain': chain,
            'changes': changes,
            'timestamp': datetime.utcnow()
        })

    async def _get_liquidity_transactions(self, token: str, chain: str) -> List:
        """Get recent liquidity transactions"""
        # This would fetch from blockchain or indexer
        return []

    async def _is_suspicious_wallet(self, wallet: str) -> bool:
        """Check if wallet is suspicious"""
        # Check against known bad actors
        return False

    async def _calculate_removal_risk_score(self, analysis: Dict) -> float:
        """Calculate removal risk score"""
        score = 0.0
        
        if analysis['removal_count'] > 5:
            score += 0.3
        if analysis['total_removed_24h'] > 100000:
            score += 0.3
        if len(analysis['suspicious_wallets']) > 0:
            score += 0.2
        if analysis['removal_velocity'] > 2:
            score += 0.2
        
        return min(1.0, score)

    async def _detect_rug_patterns(self, token: str, chain: str, analysis: Dict) -> Dict:
        """Detect rug pull patterns"""
        return {
            'pattern_detected': False,
            'pattern_type': 'none',
            'confidence': 0.0
        }

    def _get_removal_recommendation(self, analysis: Dict) -> str:
        """Get recommendation based on removal analysis"""
        if analysis['risk_score'] > 0.7:
            return "EXIT IMMEDIATELY - High rug risk detected"
        elif analysis['risk_score'] > 0.5:
            return "Consider exiting position - Moderate risk"
        else:
            return "Monitor closely"

    async def _get_liquidity_depth(self, token: str, chain: str) -> Dict:
        """Internal method to get liquidity depth"""
        return await self.get_liquidity_depth(token, chain)

    async def _get_pool_data(self, token: str, chain: str) -> Dict:
        """Get pool data for token"""
        # This would fetch from DEX APIs
        return {
            'lp_token_address': '',
            'lp_price_usd': 0,
            'reserve0': 0,
            'reserve1': 0,
            'token0_price': 1,
            'token1_price': 1
        }

    async def _calculate_orderbook_slippage(self, liquidity_data: Dict, amount: Decimal, is_buy: bool) -> Dict:
        """Calculate slippage for orderbook"""
        # Get configured slippage values
        expected_slippage_bps = self.config.get('trading.expected_slippage_bps', 10)  # 0.1%
        max_slippage_bps = self.config.get('trading.max_slippage_bps', 50)  # 0.5%

        expected_slippage = expected_slippage_bps / 10_000
        max_slippage = max_slippage_bps / 10_000
        # Simplified calculation
        return {
            'expected_slippage': expected_slippage,
            'max_slippage': max_slippage,
            'effective_price': Decimal('1'),
            'amount_out': amount * Decimal(str(1 - max_slippage))
        }

    async def _calculate_price_impact(self, liquidity_data: Dict, amount: Decimal, is_buy: bool) -> float:
        """Calculate price impact"""
        depth = liquidity_data.get('depth', Decimal('1'))

        # Get max price impact from config
        max_price_impact_bps = self.config.get('trading.max_price_impact_bps', 100)  # 1%
        max_price_impact = max_price_impact_bps / 10_000

        if depth == 0:
            return max_price_impact
        
        impact = float(amount / depth)
        return min(max_price_impact, impact)

    async def _check_lock_platform(self, token: str, chain: str, platform: str) -> Dict:
        """Check specific lock platform"""
        # This would check with lock platform APIs
        return {
            'amount': 0,
            'percentage': 0,
            'unlock_date': None,
            'tx_hash': None
        }

    async def _get_lp_token_holders(self, lp_token: str, chain: str) -> List:
        """Get LP token holders"""
        # This would fetch from blockchain
        return []

    async def _is_lp_locked(self, address: str) -> bool:
        """Check if LP tokens are locked"""
        return False

    async def _get_recent_price_volume(self, token: str, chain: str, minutes: int) -> List:
        """Get recent price and volume data"""
        # This would fetch from data provider
        return []

    async def _get_realtime_data(self, token: str, chain: str) -> Dict:
        """Get real-time data for token"""
        return {'price': 0, 'volume': 0}

    async def _calculate_momentum_indicators(self, data: List) -> Dict:
        """Calculate momentum indicators"""
        return {'rsi': 50, 'macd': 0}

    async def _calculate_pump_metrics(self, data: List, price_spike: Dict, volume_spike: Dict) -> Dict:
        """Calculate pump metrics"""
        return {
            'stage': 'initial',
            'duration_minutes': 0,
            'confidence': 0.5
        }

    async def _estimate_remaining_potential(self, data: List, metrics: Dict) -> float:
        """Estimate remaining pump potential"""
        return 0.5

    def _get_exit_recommendation(self, metrics: Dict, potential: float) -> str:
        """Get exit recommendation"""
        if potential < 0.2:
            return "Consider taking profits"
        return "Hold for now"
