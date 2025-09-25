"""
Mempool Monitor - Monitor pending transactions for MEV and sandwich attacks
"""

import asyncio
from web3 import Web3
from typing import Dict, List, Optional, Set, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
import json

@dataclass
class PendingTransaction:
    """Pending transaction data"""
    hash: str
    from_address: str
    to_address: str
    value: float
    gas_price: int
    gas_limit: int
    nonce: int
    input_data: str
    method: Optional[str] = None
    token_address: Optional[str] = None
    amount_in: Optional[float] = None
    amount_out_min: Optional[float] = None
    timestamp: datetime = field(default_factory=datetime.now)
    
@dataclass
class SandwichRisk:
    """Sandwich attack risk assessment"""
    at_risk: bool
    risk_score: float  # 0-1
    potential_loss: float
    front_runners: List[str]
    back_runners: List[str]
    detection_reason: str
    recommendations: List[str]
    
@dataclass
class MEVOpportunity:
    """MEV opportunity detection"""
    opportunity_type: str  # 'arbitrage', 'liquidation', 'sandwich'
    profit_estimate: float
    gas_cost: float
    net_profit: float
    target_tx: str
    execution_window: int  # seconds
    confidence: float

class MempoolMonitor:
    """Monitor mempool for MEV risks and opportunities"""
    
    def __init__(self, config: Dict):
        """
        Initialize mempool monitor
        
        Args:
            config: Configuration with RPC endpoints
        """
        self.config = config
        
        # Web3 connections
        self.w3_connections = {}
        self._setup_connections()
        
        # Monitoring state
        self.pending_txs = {}
        self.monitoring_tokens = set()
        self.detected_sandwiches = []
        
        # MEV detection parameters
        self.min_profit_threshold = config.get('min_profit_threshold', 0.01)  # 0.01 ETH
        self.sandwich_time_window = config.get('sandwich_time_window', 12)  # seconds
        
        # Known MEV bots
        self.known_mev_bots = self._load_mev_bots()
        
        # Router addresses for decoding
        self.routers = {
            '0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D': 'Uniswap V2',
            '0xE592427A0AEce92De3Edee1F18E0157C05861564': 'Uniswap V3',
            '0xd9e1cE17f2641f24aE83637ab66a2cca9C378B9F': 'SushiSwap',
            '0x10ED43C718714eb63d5aA57B78B54704E256024E': 'PancakeSwap V2'
        }
        
        # Method signatures
        self.method_signatures = {
            '0x7ff36ab5': 'swapExactETHForTokens',
            '0xfb3bdb41': 'swapETHForExactTokens',
            '0x18cbafe5': 'swapExactTokensForETH',
            '0x791ac947': 'swapExactTokensForETHSupportingFeeOnTransferTokens',
            '0x38ed1739': 'swapExactTokensForTokens',
            '0x8803dbee': 'swapTokensForExactTokens'
        }
        
        # Subscription handles
        self.subscriptions = {}
        
    def _setup_connections(self):
        """Setup Web3 WebSocket connections for mempool access"""
        try:
            # Ethereum WebSocket
            if 'ethereum_ws' in self.config:
                w3 = Web3(Web3.WebsocketProvider(self.config['ethereum_ws']))
                if w3.is_connected():
                    self.w3_connections['ethereum'] = w3
                    
            # BSC WebSocket  
            if 'bsc_ws' in self.config:
                w3 = Web3(Web3.WebsocketProvider(self.config['bsc_ws']))
                if w3.is_connected():
                    self.w3_connections['bsc'] = w3
                    
        except Exception as e:
            print(f"WebSocket connection error: {e}")
            
    def _load_mev_bots(self) -> Set[str]:
        """Load known MEV bot addresses"""
        # These would be loaded from a database
        return {
            '0x00000000000006B7e42F8b5C3Fd2e5BDC0d7a0AC',  # Example MEV bot
            '0xa69babEF1cA67A37Ffaf7a485DfFF3382056e78C',  # Example sandwich bot
            # Add more known bots
        }
        
    async def start_monitoring(self, chain: str = 'ethereum'):
        """Start monitoring mempool"""
        try:
            w3 = self.w3_connections.get(chain)
            if not w3:
                print(f"No connection for {chain}")
                return
                
            # Subscribe to pending transactions
            subscription = await w3.eth.subscribe('pending_transactions')
            self.subscriptions[chain] = subscription
            
            # Process pending transactions
            async for tx_hash in subscription:
                await self._process_pending_tx(tx_hash, chain)
                
        except Exception as e:
            print(f"Monitoring error: {e}")
            
    async def stop_monitoring(self, chain: str = 'ethereum'):
        """Stop monitoring mempool"""
        try:
            if chain in self.subscriptions:
                await self.subscriptions[chain].unsubscribe()
                del self.subscriptions[chain]
                
        except Exception as e:
            print(f"Stop monitoring error: {e}")
            
    async def _process_pending_tx(self, tx_hash: str, chain: str):
        """Process a pending transaction"""
        try:
            w3 = self.w3_connections[chain]
            
            # Get transaction details
            tx = w3.eth.get_transaction(tx_hash)
            
            if not tx:
                return
                
            # Parse transaction
            pending_tx = PendingTransaction(
                hash=tx_hash.hex(),
                from_address=tx['from'],
                to_address=tx['to'] if tx['to'] else '',
                value=tx['value'],
                gas_price=tx['gasPrice'],
                gas_limit=tx['gas'],
                nonce=tx['nonce'],
                input_data=tx['input']
            )
            
            # Decode if it's a swap
            if tx['to'] in self.routers:
                decoded = self._decode_swap(tx['input'], tx['to'])
                if decoded:
                    pending_tx.method = decoded['method']
                    pending_tx.token_address = decoded.get('token')
                    pending_tx.amount_in = decoded.get('amount_in')
                    pending_tx.amount_out_min = decoded.get('amount_out_min')
                    
            # Store pending transaction
            self.pending_txs[tx_hash.hex()] = pending_tx
            
            # Check for sandwich attack risk
            if pending_tx.method and 'swap' in pending_tx.method.lower():
                risk = await self.check_sandwich_risk(tx_hash.hex())
                if risk.at_risk:
                    self.detected_sandwiches.append({
                        'tx': tx_hash.hex(),
                        'risk': risk,
                        'timestamp': datetime.now()
                    })
                    
            # Clean old pending transactions
            self._cleanup_old_txs()
            
        except Exception as e:
            print(f"Transaction processing error: {e}")
            
    async def check_sandwich_risk(self, tx_hash: str) -> SandwichRisk:
        """Check if transaction is at risk of sandwich attack"""
        try:
            if tx_hash not in self.pending_txs:
                return SandwichRisk(
                    at_risk=False,
                    risk_score=0,
                    potential_loss=0,
                    front_runners=[],
                    back_runners=[],
                    detection_reason='Transaction not found',
                    recommendations=[]
                )
                
            target_tx = self.pending_txs[tx_hash]
            
            # Look for potential sandwich transactions
            front_runners = []
            back_runners = []
            risk_score = 0
            
            # Check for transactions from known MEV bots
            for pending_hash, pending_tx in self.pending_txs.items():
                if pending_hash == tx_hash:
                    continue
                    
                # Check if from MEV bot
                if pending_tx.from_address in self.known_mev_bots:
                    risk_score += 0.3
                    
                    # Check if targeting same token
                    if pending_tx.token_address == target_tx.token_address:
                        risk_score += 0.2
                        
                        # Check gas price (front-runners use higher gas)
                        if pending_tx.gas_price > target_tx.gas_price:
                            front_runners.append(pending_hash)
                        else:
                            back_runners.append(pending_hash)
                            
                # Check for similar transactions (potential sandwich)
                elif pending_tx.token_address == target_tx.token_address:
                    time_diff = abs((pending_tx.timestamp - target_tx.timestamp).seconds)
                    
                    if time_diff < self.sandwich_time_window:
                        # Same token, close timing
                        risk_score += 0.1
                        
                        if pending_tx.gas_price > target_tx.gas_price * 1.1:
                            front_runners.append(pending_hash)
                            risk_score += 0.1
                            
            # Calculate potential loss
            potential_loss = 0
            if risk_score > 0.5 and target_tx.amount_in:
                # Estimate 3-5% loss from sandwich
                potential_loss = target_tx.amount_in * 0.04
                
            # Generate recommendations
            recommendations = []
            
            if risk_score > 0.7:
                recommendations.append("Cancel transaction immediately")
                recommendations.append("Use private mempool (Flashbots)")
            elif risk_score > 0.5:
                recommendations.append("Consider using MEV protection")
                recommendations.append("Split into smaller transactions")
            elif risk_score > 0.3:
                recommendations.append("Monitor transaction carefully")
                recommendations.append("Consider increasing slippage tolerance")
                
            # Determine detection reason
            reasons = []
            if front_runners:
                reasons.append(f"{len(front_runners)} potential front-runners detected")
            if back_runners:
                reasons.append(f"{len(back_runners)} potential back-runners detected")
            if any(tx.from_address in self.known_mev_bots for tx in self.pending_txs.values()):
                reasons.append("Known MEV bots active")
                
            return SandwichRisk(
                at_risk=risk_score > 0.5,
                risk_score=min(risk_score, 1.0),
                potential_loss=potential_loss,
                front_runners=front_runners,
                back_runners=back_runners,
                detection_reason=', '.join(reasons) if reasons else 'No specific threats',
                recommendations=recommendations
            )
            
        except Exception as e:
            print(f"Sandwich risk check error: {e}")
            return SandwichRisk(
                at_risk=False,
                risk_score=0,
                potential_loss=0,
                front_runners=[],
                back_runners=[],
                detection_reason=f'Error: {e}',
                recommendations=[]
            )
            
    async def detect_mev_opportunities(self) -> List[MEVOpportunity]:
        """Detect MEV opportunities in mempool"""
        opportunities = []
        
        try:
            # Check for arbitrage opportunities
            arb_ops = await self._detect_arbitrage()
            opportunities.extend(arb_ops)
            
            # Check for liquidation opportunities
            liq_ops = await self._detect_liquidations()
            opportunities.extend(liq_ops)
            
            # Sort by profit
            opportunities.sort(key=lambda x: x.net_profit, reverse=True)
            
            return opportunities
            
        except Exception as e:
            print(f"MEV detection error: {e}")
            return opportunities
            
    async def _detect_arbitrage(self) -> List[MEVOpportunity]:
        """Detect arbitrage opportunities"""
        opportunities = []
        
        try:
            # Group pending swaps by token
            token_swaps = defaultdict(list)
            
            for tx_hash, tx in self.pending_txs.items():
                if tx.token_address:
                    token_swaps[tx.token_address].append((tx_hash, tx))
                    
            # Look for price discrepancies
            for token, swaps in token_swaps.items():
                if len(swaps) >= 2:
                    # Check for different prices
                    prices = []
                    
                    for _, tx in swaps:
                        if tx.amount_in and tx.amount_out_min:
                            price = tx.amount_out_min / tx.amount_in
                            prices.append(price)
                            
                    if prices and max(prices) > min(prices) * 1.02:  # 2% difference
                        # Potential arbitrage
                        profit = (max(prices) - min(prices)) * 0.1  # Simplified calculation
                        gas_cost = 200000 * 50 * 10**-9  # Estimated gas cost
                        
                        if profit - gas_cost > self.min_profit_threshold:
                            opportunities.append(MEVOpportunity(
                                opportunity_type='arbitrage',
                                profit_estimate=profit,
                                gas_cost=gas_cost,
                                net_profit=profit - gas_cost,
                                target_tx=swaps[0][0],
                                execution_window=12,
                                confidence=0.7
                            ))
                            
            return opportunities
            
        except Exception as e:
            print(f"Arbitrage detection error: {e}")
            return opportunities
            
    async def _detect_liquidations(self) -> List[MEVOpportunity]:
        """Detect liquidation opportunities"""
        # This would integrate with lending protocols
        # Placeholder for now
        return []
        
    def _decode_swap(self, input_data: str, router_address: str) -> Optional[Dict]:
        """Decode swap transaction input"""
        try:
            if len(input_data) < 10:
                return None
                
            method_id = input_data[:10]
            
            if method_id not in self.method_signatures:
                return None
                
            method_name = self.method_signatures[method_id]
            
            # Simplified decoding - would use eth-abi for proper decoding
            result = {
                'method': method_name,
                'router': self.routers.get(router_address, 'Unknown')
            }
            
            # Extract token address and amounts based on method
            # This is simplified - actual implementation would properly decode ABI
            if 'ETHForTokens' in method_name:
                # Token address is typically in the path
                result['token'] = '0x' + input_data[34:74] if len(input_data) > 74 else None
            elif 'TokensForETH' in method_name:
                result['token'] = '0x' + input_data[34:74] if len(input_data) > 74 else None
                
            return result
            
        except Exception as e:
            print(f"Decode error: {e}")
            return None
            
    def _cleanup_old_txs(self):
        """Remove old pending transactions"""
        try:
            cutoff = datetime.now() - timedelta(seconds=30)
            
            to_remove = [
                tx_hash for tx_hash, tx in self.pending_txs.items()
                if tx.timestamp < cutoff
            ]
            
            for tx_hash in to_remove:
                del self.pending_txs[tx_hash]
                
        except Exception as e:
            print(f"Cleanup error: {e}")
            
    async def get_gas_prices(self, chain: str = 'ethereum') -> Dict:
        """Get current gas prices with MEV context"""
        try:
            w3 = self.w3_connections.get(chain)
            if not w3:
                return {}
                
            # Get base gas price
            base_gas = w3.eth.gas_price
            
            # Analyze mempool for gas price distribution
            gas_prices = [tx.gas_price for tx in self.pending_txs.values()]
            
            if gas_prices:
                gas_prices.sort()
                
                return {
                    'base': base_gas,
                    'slow': gas_prices[int(len(gas_prices) * 0.25)],
                    'standard': gas_prices[int(len(gas_prices) * 0.5)],
                    'fast': gas_prices[int(len(gas_prices) * 0.75)],
                    'instant': gas_prices[int(len(gas_prices) * 0.95)],
                    'mev_protection': gas_prices[int(len(gas_prices) * 0.99)]  # Top 1%
                }
            else:
                return {
                    'base': base_gas,
                    'slow': int(base_gas * 0.9),
                    'standard': base_gas,
                    'fast': int(base_gas * 1.2),
                    'instant': int(base_gas * 1.5),
                    'mev_protection': int(base_gas * 2)
                }
                
        except Exception as e:
            print(f"Gas price error: {e}")
            return {}
            
    async def monitor_token_mempool(self, token_address: str, chain: str = 'ethereum') -> Dict:
        """Monitor mempool activity for specific token"""
        try:
            self.monitoring_tokens.add(token_address)
            
            # Count pending transactions for token
            token_txs = [
                tx for tx in self.pending_txs.values()
                if tx.token_address == token_address
            ]
            
            # Analyze activity
            buy_count = sum(1 for tx in token_txs if tx.method and 'buy' in tx.method.lower())
            sell_count = sum(1 for tx in token_txs if tx.method and 'sell' in tx.method.lower())
            
            # Calculate metrics
            total_buy_volume = sum(tx.amount_in for tx in token_txs if tx.amount_in and 'buy' in str(tx.method).lower())
            total_sell_volume = sum(tx.amount_in for tx in token_txs if tx.amount_in and 'sell' in str(tx.method).lower())
            
            # Detect unusual activity
            mev_bots_active = any(tx.from_address in self.known_mev_bots for tx in token_txs)
            
            return {
                'pending_count': len(token_txs),
                'buy_count': buy_count,
                'sell_count': sell_count,
                'buy_pressure': buy_count / (buy_count + sell_count) if (buy_count + sell_count) > 0 else 0.5,
                'total_buy_volume': total_buy_volume,
                'total_sell_volume': total_sell_volume,
                'mev_bots_active': mev_bots_active,
                'sandwich_risk': len([s for s in self.detected_sandwiches if any(tx.token_address == token_address for tx in self.pending_txs.values())]),
                'timestamp': datetime.now()
            }
            
        except Exception as e:
            print(f"Token monitoring error: {e}")
            return {}
            
    def add_mev_bot(self, address: str):
        """Add address to known MEV bots"""
        self.known_mev_bots.add(address.lower())
        
    def get_mempool_stats(self) -> Dict:
        """Get mempool statistics"""
        try:
            return {
                'total_pending': len(self.pending_txs),
                'monitored_tokens': len(self.monitoring_tokens),
                'detected_sandwiches': len(self.detected_sandwiches),
                'known_mev_bots': len(self.known_mev_bots),
                'oldest_tx': min([tx.timestamp for tx in self.pending_txs.values()]) if self.pending_txs else None,
                'newest_tx': max([tx.timestamp for tx in self.pending_txs.values()]) if self.pending_txs else None
            }
        except Exception as e:
            print(f"Stats error: {e}")
            return {}

    # ============================================================================
    # PATCH FOR: mempool_monitor.py
    # Add these wrapper methods to the MempoolMonitor class
    # ============================================================================

    async def monitor_mempool(self, chain: str = 'ethereum'):
        """
        Monitor mempool as AsyncGenerator (wrapper for start_monitoring)
        
        Args:
            chain: Blockchain network
            
        Yields:
            Pending transaction data
        """
        # Start monitoring in background
        monitoring_task = asyncio.create_task(self.start_monitoring(chain))
        
        try:
            while True:
                # Yield pending transactions from the queue
                if self.pending_txs:
                    for tx_hash, tx_data in list(self.pending_txs.items()):
                        yield {
                            'hash': tx_data.hash,
                            'from': tx_data.from_address,
                            'to': tx_data.to_address,
                            'value': tx_data.value,
                            'gas_price': tx_data.gas_price,
                            'gas_limit': tx_data.gas_limit,
                            'method': tx_data.method,
                            'token_address': tx_data.token_address,
                            'timestamp': tx_data.timestamp
                        }
                        
                        # Remove after yielding to avoid duplicates
                        del self.pending_txs[tx_hash]
                        
                await asyncio.sleep(0.1)  # Small delay
                
        except GeneratorExit:
            # Stop monitoring when generator is closed
            monitoring_task.cancel()
            await self.stop_monitoring(chain)

    async def detect_frontrun_risk(self, transaction: Dict) -> bool:
        """
        Detect if a transaction is at risk of frontrunning
        
        Args:
            transaction: Transaction details
            
        Returns:
            True if at risk of frontrunning
        """
        # Check if transaction is in our pending transactions
        tx_hash = transaction.get('hash', '')
        
        # Use existing sandwich risk check
        risk = await self.check_sandwich_risk(tx_hash)
        
        # Also check for general frontrun patterns
        if risk.at_risk:
            return True
            
        # Check gas price - high gas might indicate frontrun attempt
        gas_price = transaction.get('gas_price', 0)
        gas_prices = await self.get_gas_prices()
        
        if gas_price > gas_prices.get('instant', 0) * 1.5:
            return True
            
        # Check if from known MEV bot
        from_address = transaction.get('from', '')
        if from_address in self.known_mev_bots:
            return True
            
        return False

    async def analyze_pending_tx(self, tx_hash: str) -> Dict:
        """
        Analyze a pending transaction (public wrapper for _process_pending_tx)
        
        Args:
            tx_hash: Transaction hash
            
        Returns:
            Transaction analysis
        """
        # Check if already in pending_txs
        if tx_hash in self.pending_txs:
            tx_data = self.pending_txs[tx_hash]
            
            # Check for risks
            sandwich_risk = await self.check_sandwich_risk(tx_hash)
            frontrun_risk = await self.detect_frontrun_risk({
                'hash': tx_hash,
                'from': tx_data.from_address,
                'gas_price': tx_data.gas_price
            })
            
            return {
                'hash': tx_hash,
                'from': tx_data.from_address,
                'to': tx_data.to_address,
                'value': tx_data.value,
                'gas_price': tx_data.gas_price,
                'gas_limit': tx_data.gas_limit,
                'method': tx_data.method,
                'token_address': tx_data.token_address,
                'timestamp': tx_data.timestamp,
                'risks': {
                    'sandwich': sandwich_risk.at_risk,
                    'sandwich_score': sandwich_risk.risk_score,
                    'frontrun': frontrun_risk,
                    'mev_bot': tx_data.from_address in self.known_mev_bots
                },
                'recommendations': sandwich_risk.recommendations
            }
        
        # If not in pending_txs, try to fetch and analyze
        # For all chains in w3_connections
        for chain, w3 in self.w3_connections.items():
            try:
                tx = w3.eth.get_transaction(tx_hash)
                if tx:
                    # Process it
                    await self._process_pending_tx(tx_hash, chain)
                    
                    # Now analyze
                    if tx_hash in self.pending_txs:
                        return await self.analyze_pending_tx(tx_hash)
                        
            except Exception:
                continue
                
        return {
            'error': 'Transaction not found',
            'hash': tx_hash
        }