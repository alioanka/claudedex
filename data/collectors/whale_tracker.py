"""
Whale Tracker - Monitor Large Wallet Movements
Track and analyze whale wallets and their impact on token prices
"""

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Set
from decimal import Decimal
from datetime import datetime, timedelta
import aiohttp
from web3 import Web3
import numpy as np

from utils.helpers import (
    retry_async, rate_limit, is_valid_address, wei_to_ether,
    format_token_amount, calculate_percentage_change, TTLCache
)
from utils.constants import (
    Chain, BLOCK_EXPLORERS,
    WRAPPED_NATIVE_TOKENS, STABLECOINS
)

logger = logging.getLogger(__name__)

class WhaleTracker:
    """Advanced whale movement tracking system"""
    
    def __init__(self, config: Dict = None):
        """Initialize whale tracker with configuration"""
        self.config = config or {}
        self.session = None
        self._web3_cache = {}
        self._web3_locks = {}
        self.cache = TTLCache(ttl=300)  # 5 minute cache
        
        # Whale thresholds by chain (in USD)
        self.whale_thresholds = {
            Chain.ETHEREUM: 1000000,  # $1M
            Chain.BSC: 500000,  # $500k
            Chain.POLYGON: 250000,  # $250k
            Chain.ARBITRUM: 500000,  # $500k
            Chain.BASE: 500000,  # $500k
        }
        
        # Known whale addresses to track
        self.known_whales = set()
        self.whale_labels = {}  # Address -> label mapping
        
        # Tracking data
        self.whale_movements = {}  # token -> movements
        self.whale_balances = {}  # whale -> token -> balance
        self.impact_scores = {}  # token -> impact score
        
    async def initialize(self):
        """Initialize connections and resources"""
        self.session = aiohttp.ClientSession()
        # ✅ Web3 connections lazy-loaded on demand
        self._web3_cache = {}
        self._web3_locks = {}
        await self._load_known_whales()
        logger.info("WhaleTracker initialized successfully")
        
    def _chain_id_to_name(self, chain_id: int) -> str:
        """Convert chain ID to name"""
        chain_map = {
            1: 'ethereum',
            56: 'bsc',
            137: 'polygon',
            42161: 'arbitrum',
            8453: 'base',
        }
        return chain_map.get(chain_id, 'ethereum')

    async def _get_web3_for_chain(self, chain: str):
        """Lazy-load Web3 connection for chain"""
        if chain in self._web3_cache:
            w3 = self._web3_cache[chain]
            if w3.is_connected():
                return w3
            else:
                del self._web3_cache[chain]
        
        if chain not in self._web3_locks:
            self._web3_locks[chain] = asyncio.Lock()
        
        async with self._web3_locks[chain]:
            if chain in self._web3_cache:
                return self._web3_cache[chain]
            
            if hasattr(self.config, 'get_rpc_urls'):
                rpc_urls = self.config.get_rpc_urls(chain)
                if rpc_urls:
                    for rpc_url in rpc_urls:
                        try:
                            w3 = Web3(Web3.HTTPProvider(rpc_url, request_kwargs={'timeout': 10}))
                            if w3.is_connected():
                                self._web3_cache[chain] = w3
                                logger.info(f"✅ WhaleTracker lazy-loaded {chain}")
                                return w3
                        except Exception as e:
                            logger.warning(f"Failed to connect to {rpc_url}: {e}")
            
            return None

    async def _load_known_whales(self):
        """Load known whale addresses from various sources"""
        # Add some known whale addresses
        self.known_whales.update([
            # Example addresses (would be loaded from database/config)
            "0x0000000000000000000000000000000000000000",  # Placeholder
        ])
        
        # Load whale labels
        self.whale_labels = {
            # Example labels
            "0x0000000000000000000000000000000000000000": "Major Fund #1",
        }
        
    async def close(self):
        """Clean up resources"""
        if self.session:
            await self.session.close()
            
    async def track_whale_movements(self, token: str, chain: str) -> Dict:
        """
        Track whale movements for a specific token
        Returns comprehensive whale activity analysis
        """
        if not is_valid_address(token):
            return {"error": "Invalid token address"}
            
        chain_id = self._get_chain_id(chain)
        
        # Get whale wallets holding this token
        whale_wallets = await self.identify_whale_wallets(token, chain)
        
        # Track recent movements
        movements = await self._track_recent_transfers(token, whale_wallets, chain_id)
        
        # Analyze whale behavior patterns
        behavior_analysis = await self.analyze_whale_behavior(whale_wallets[0], chain) if whale_wallets else {}
        
        # Calculate impact score
        impact_score = self.get_whale_impact_score(movements)
        
        # Store tracking data
        self.whale_movements[token] = movements
        self.impact_scores[token] = impact_score
        
        return {
            "token": token,
            "chain": chain,
            "whale_count": len(whale_wallets),
            "whale_wallets": whale_wallets[:10],  # Top 10 whales
            "recent_movements": movements[:20],  # Last 20 movements
            "behavior_analysis": behavior_analysis,
            "impact_score": impact_score,
            "risk_level": self._calculate_risk_level(impact_score, movements),
            "recommendation": self._generate_recommendation(impact_score, movements)
        }
        
    async def identify_whale_wallets(self, token: str, chain: str) -> List[str]:
        """Identify whale wallets holding significant amounts of token"""
        try:
            chain_id = self._get_chain_id(chain)
            
            # Get top holders from blockchain
            holders = await self._get_top_holders(token, chain_id)
            
            # Filter for whale-sized holdings
            whale_threshold = self.whale_thresholds.get(chain_id, 500000)
            
            whale_wallets = []
            for holder in holders:
                if holder.get("value_usd", 0) >= whale_threshold:
                    whale_wallets.append(holder["address"])
                    
                    # Add to known whales if new
                    if holder["address"] not in self.known_whales:
                        self.known_whales.add(holder["address"])
                        
            return whale_wallets
            
        except Exception as e:
            logger.error(f"Failed to identify whale wallets: {e}")
            return []
            
    async def _get_top_holders(self, token: str, chain_id: int) -> List[Dict]:
        """Get top token holders from blockchain"""
        try:
            # This would integrate with blockchain explorers or indexing services
            # For now, using a simplified approach
            
            chain_name = self._chain_id_to_name(chain_id)
            w3 = await self._get_web3_for_chain(chain_name)
            if not w3:
                return []
            
            # Get token contract
            erc20_abi = self._get_erc20_abi()
            contract = w3.eth.contract(address=token, abi=erc20_abi)
            
            # Get Transfer events to identify holders
            # In production, this would use an indexing service
            latest_block = w3.eth.get_block_number()
            from_block = max(0, latest_block - 10000)  # Last 10k blocks
            
            events = contract.events.Transfer.get_logs(
                fromBlock=from_block,
                toBlock=latest_block
            )
            
            # Track balances
            balances = {}
            for event in events:
                to_address = event['args']['to']
                from_address = event['args']['from']
                value = event['args']['value']
                
                if to_address != "0x0000000000000000000000000000000000000000":
                    balances[to_address] = balances.get(to_address, 0) + value
                if from_address != "0x0000000000000000000000000000000000000000":
                    balances[from_address] = balances.get(from_address, 0) - value
                    
            # Get current balances and calculate USD values
            holders = []
            token_price = await self._get_token_price(token, chain_id)
            decimals = await self._get_token_decimals(contract)
            
            for address, balance in balances.items():
                if balance > 0:
                    token_amount = format_token_amount(balance, decimals)
                    value_usd = float(token_amount) * token_price
                    
                    holders.append({
                        "address": address,
                        "balance": balance,
                        "token_amount": float(token_amount),
                        "value_usd": value_usd,
                        "percentage": 0  # Would calculate based on total supply
                    })
                    
            # Sort by value
            holders.sort(key=lambda x: x["value_usd"], reverse=True)
            
            return holders[:100]  # Top 100 holders
            
        except Exception as e:
            logger.error(f"Failed to get top holders: {e}")
            return []
            
    async def _track_recent_transfers(self, token: str, whale_wallets: List[str], 
                                    chain_id: int) -> List[Dict]:
        """Track recent transfers from whale wallets"""
        try:
            chain_name = self._chain_id_to_name(chain_id)
            w3 = await self._get_web3_for_chain(chain_name)
            if not w3:
                return []
            
            # Get token contract
            erc20_abi = self._get_erc20_abi()
            contract = w3.eth.contract(address=token, abi=erc20_abi)
            
            movements = []
            latest_block = w3.eth.get_block_number()
            from_block = max(0, latest_block - 1000)  # Last 1000 blocks
            
            for whale in whale_wallets[:20]:  # Check top 20 whales
                # Get transfer events for this whale
                events_from = contract.events.Transfer.get_logs(
                    fromBlock=from_block,
                    toBlock=latest_block,
                    argument_filters={'from': whale}
                )
                
                events_to = contract.events.Transfer.get_logs(
                    fromBlock=from_block,
                    toBlock=latest_block,
                    argument_filters={'to': whale}
                )
                
                # Process events
                for event in events_from:
                    block = w3.eth.get_block(event['blockNumber'])
                    movements.append({
                        "type": "sell",
                        "whale": whale,
                        "whale_label": self.whale_labels.get(whale, "Unknown Whale"),
                        "to": event['args']['to'],
                        "value": event['args']['value'],
                        "block": event['blockNumber'],
                        "timestamp": block['timestamp'],
                        "tx_hash": event['transactionHash'].hex()
                    })
                    
                for event in events_to:
                    block = w3.eth.get_block(event['blockNumber'])
                    movements.append({
                        "type": "buy",
                        "whale": whale,
                        "whale_label": self.whale_labels.get(whale, "Unknown Whale"),
                        "from": event['args']['from'],
                        "value": event['args']['value'],
                        "block": event['blockNumber'],
                        "timestamp": block['timestamp'],
                        "tx_hash": event['transactionHash'].hex()
                    })
                    
            # Sort by timestamp
            movements.sort(key=lambda x: x["timestamp"], reverse=True)
            
            return movements
            
        except Exception as e:
            logger.error(f"Failed to track transfers: {e}")
            return []
            
    async def analyze_whale_behavior(self, wallet: str, chain: str) -> Dict:
        """Analyze historical behavior patterns of a whale wallet"""
        try:
            chain_id = self._get_chain_id(chain)
            
            # Get wallet transaction history
            history = await self._get_wallet_history(wallet, chain_id)
            
            # Analyze patterns
            analysis = {
                "wallet": wallet,
                "label": self.whale_labels.get(wallet, "Unknown"),
                "total_transactions": len(history),
                "active_tokens": set(),
                "trading_frequency": 0,
                "avg_hold_time": 0,
                "profit_rate": 0,
                "behavior_type": "unknown",
                "risk_score": 0
            }
            
            if not history:
                return analysis
                
            # Calculate metrics
            buy_count = sum(1 for tx in history if tx.get("type") == "buy")
            sell_count = sum(1 for tx in history if tx.get("type") == "sell")
            
            analysis["buy_sell_ratio"] = buy_count / max(sell_count, 1)
            
            # Determine behavior type
            if buy_count > sell_count * 2:
                analysis["behavior_type"] = "accumulator"
                analysis["risk_score"] = 30
            elif sell_count > buy_count * 2:
                analysis["behavior_type"] = "distributor"
                analysis["risk_score"] = 70
            elif len(history) > 100:
                analysis["behavior_type"] = "active_trader"
                analysis["risk_score"] = 50
            else:
                analysis["behavior_type"] = "holder"
                analysis["risk_score"] = 20
                
            # Calculate trading frequency (trades per day)
            if history:
                time_span = history[0]["timestamp"] - history[-1]["timestamp"]
                days = max(time_span / 86400, 1)
                analysis["trading_frequency"] = len(history) / days
                
            return analysis
            
        except Exception as e:
            logger.error(f"Failed to analyze whale behavior: {e}")
            return {"error": str(e)}
            
    async def _get_wallet_history(self, wallet: str, chain_id: int) -> List[Dict]:
        """Get transaction history for a wallet"""
        try:
            # This would integrate with blockchain explorers or indexing services
            # For now, returning mock data structure
            return []
            
        except Exception as e:
            logger.error(f"Failed to get wallet history: {e}")
            return []
            
    def get_whale_impact_score(self, movements: List[Dict]) -> float:
        """Calculate whale impact score based on recent movements"""
        if not movements:
            return 0.0
            
        score = 0.0
        
        # Analyze last 24 hours of movements
        current_time = datetime.now().timestamp()
        recent_movements = [
            m for m in movements 
            if current_time - m.get("timestamp", 0) < 86400
        ]
        
        if not recent_movements:
            return 0.0
            
        # Calculate metrics
        total_volume = sum(m.get("value", 0) for m in recent_movements)
        sell_volume = sum(
            m.get("value", 0) for m in recent_movements 
            if m.get("type") == "sell"
        )
        buy_volume = sum(
            m.get("value", 0) for m in recent_movements 
            if m.get("type") == "buy"
        )
        
        # Calculate impact factors
        if total_volume > 0:
            # Selling pressure
            if sell_volume > buy_volume:
                sell_ratio = sell_volume / total_volume
                score += sell_ratio * 50  # Max 50 points for selling
                
            # Buying pressure (positive but less impactful)
            elif buy_volume > sell_volume:
                buy_ratio = buy_volume / total_volume
                score -= buy_ratio * 20  # Negative score is good
                
        # Movement frequency impact
        movement_count = len(recent_movements)
        if movement_count > 10:
            score += 20  # High frequency movements
        elif movement_count > 5:
            score += 10
            
        # Unique whales involved
        unique_whales = len(set(m.get("whale") for m in recent_movements))
        if unique_whales > 5:
            score += 20  # Multiple whales moving
        elif unique_whales > 2:
            score += 10
            
        # Normalize score to 0-100
        score = max(0, min(100, score))
        
        return score
        
    def _calculate_risk_level(self, impact_score: float, movements: List[Dict]) -> str:
        """Calculate risk level based on whale activity"""
        if impact_score >= 70:
            return "high"
        elif impact_score >= 50:
            return "medium"
        elif impact_score >= 30:
            return "low"
        else:
            return "minimal"
            
    def _generate_recommendation(self, impact_score: float, movements: List[Dict]) -> str:
        """Generate trading recommendation based on whale activity"""
        if impact_score >= 70:
            return "AVOID - High whale selling pressure detected"
        elif impact_score >= 50:
            return "CAUTION - Moderate whale activity, monitor closely"
        elif impact_score <= 20:
            return "POSITIVE - Whale accumulation detected"
        else:
            return "NEUTRAL - Normal whale activity"
            
    def _get_chain_id(self, chain: str) -> int:
        """Convert chain string to chain ID"""
        chain_map = {
            "ethereum": Chain.ETHEREUM,
            "eth": Chain.ETHEREUM,
            "bsc": Chain.BSC,
            "binance": Chain.BSC,
            "polygon": Chain.POLYGON,
            "matic": Chain.POLYGON,
            "arbitrum": Chain.ARBITRUM,
            "base": Chain.BASE
        }
        
        return chain_map.get(chain.lower(), Chain.ETHEREUM)
        
    def _get_erc20_abi(self) -> List:
        """Get minimal ERC20 ABI"""
        return [
            {
                "anonymous": False,
                "inputs": [
                    {"indexed": True, "name": "from", "type": "address"},
                    {"indexed": True, "name": "to", "type": "address"},
                    {"indexed": False, "name": "value", "type": "uint256"}
                ],
                "name": "Transfer",
                "type": "event"
            },
            {
                "inputs": [],
                "name": "totalSupply",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [],
                "name": "decimals",
                "outputs": [{"name": "", "type": "uint8"}],
                "stateMutability": "view",
                "type": "function"
            },
            {
                "inputs": [{"name": "account", "type": "address"}],
                "name": "balanceOf",
                "outputs": [{"name": "", "type": "uint256"}],
                "stateMutability": "view",
                "type": "function"
            }
        ]
        
    async def _get_token_decimals(self, contract) -> int:
        """Get token decimals"""
        try:
            return await contract.functions.decimals().call()
        except:
            return 18  # Default to 18 decimals
            
    async def _get_token_price(self, token: str, chain_id: int) -> float:
        """Get current token price in USD"""
        try:
            # This would integrate with price feeds
            # For now, returning mock price
            return 1.0
        except:
            return 0.0
            
    async def monitor_whale_alerts(self, token: str, chain: str, callback):
        """Monitor whale movements and trigger alerts"""
        while True:
            try:
                movements = await self.track_whale_movements(token, chain)
                
                if movements.get("impact_score", 0) > 50:
                    await callback({
                        "type": "whale_alert",
                        "token": token,
                        "chain": chain,
                        "movements": movements,
                        "timestamp": datetime.now()
                    })
                    
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Whale monitoring error: {e}")
                await asyncio.sleep(60)
                
    def get_statistics(self) -> Dict:
        """Get whale tracker statistics"""
        return {
            "known_whales": len(self.known_whales),
            "labeled_whales": len(self.whale_labels),
            "tracked_tokens": len(self.whale_movements),
            "total_movements": sum(len(m) for m in self.whale_movements.values()),
            "high_risk_tokens": sum(1 for s in self.impact_scores.values() if s > 70)
        }