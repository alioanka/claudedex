"""
Volume Analyzer - Advanced volume pattern analysis for ClaudeDex Trading Bot
Detects wash trading, fake volume, and organic volume patterns
"""

import asyncio
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from decimal import Decimal
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
from collections import defaultdict, deque
import statistics
import aiohttp
from loguru import logger

from utils.helpers import retry_async, measure_time
from utils.constants import CHAIN_RPC_URLS, DEX_ROUTERS


class VolumePattern(Enum):
    """Types of volume patterns detected"""
    ORGANIC = "organic"
    WASH_TRADING = "wash_trading"
    PUMP_SCHEME = "pump_scheme"
    BOT_TRADING = "bot_trading"
    ACCUMULATION = "accumulation"
    DISTRIBUTION = "distribution"
    BREAKOUT = "breakout"
    EXHAUSTION = "exhaustion"


@dataclass
class VolumeProfile:
    """Volume analysis profile for a token"""
    token_address: str
    chain: str
    total_volume_24h: Decimal
    organic_volume: Decimal
    suspicious_volume: Decimal
    wash_trading_score: float  # 0-1 probability
    unique_traders: int
    average_trade_size: Decimal
    volume_velocity: float  # Rate of volume change
    volume_patterns: List[VolumePattern]
    dex_distribution: Dict[str, Decimal]  # Volume per DEX
    time_distribution: Dict[int, Decimal]  # Volume per hour
    whale_volume_percent: float
    retail_volume_percent: float
    smart_money_flow: str  # "in", "out", "neutral"
    confidence_score: float
    risk_flags: List[str]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class TradeCluster:
    """Cluster of related trades (potential wash trading)"""
    trades: List[Dict]
    wallets: Set[str]
    total_volume: Decimal
    time_span: timedelta
    pattern_type: VolumePattern
    similarity_score: float


class VolumeAnalyzer:
    """
    Advanced volume analysis to detect wash trading, fake volume,
    and identify genuine trading patterns
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Analysis thresholds
        self.wash_trading_threshold = config.get("wash_trading_threshold", 0.7)
        self.min_unique_traders = config.get("min_unique_traders", 50)
        self.volume_spike_multiplier = config.get("volume_spike_multiplier", 5.0)
        self.time_window_minutes = config.get("time_window_minutes", 60)
        
        # Pattern detection parameters
        self.min_cluster_size = config.get("min_cluster_size", 10)
        self.trade_similarity_threshold = config.get("trade_similarity_threshold", 0.85)
        self.bot_pattern_threshold = config.get("bot_pattern_threshold", 0.8)
        
        # Caches
        self.volume_cache: Dict[str, VolumeProfile] = {}
        self.trade_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=10000))
        self.pattern_history: Dict[str, List[VolumePattern]] = defaultdict(list)
        
        # Known wash trading addresses
        self.wash_trading_addresses: Set[str] = set()
        self.bot_addresses: Set[str] = set()
        
        logger.info("Volume Analyzer initialized")
    
    async def initialize(self) -> None:
        """Initialize volume analyzer components"""
        await self._load_known_wash_traders()
        await self._load_bot_addresses()
        logger.info("Volume Analyzer ready")
    
    @retry_async(max_retries=3, delay=1.0)
    @measure_time
    async def analyze_volume(
        self,
        token_address: str,
        chain: str,
        time_window: int = 24  # hours
    ) -> VolumeProfile:
        """
        Perform comprehensive volume analysis
        
        Args:
            token_address: Token to analyze
            chain: Blockchain network
            time_window: Analysis window in hours
            
        Returns:
            Complete volume analysis profile
        """
        try:
            # Check cache
            cache_key = f"{chain}:{token_address}"
            if cache_key in self.volume_cache:
                cached = self.volume_cache[cache_key]
                if (datetime.now() - cached.timestamp).seconds < 300:  # 5 min cache
                    return cached
            
            # Gather trade data
            trades = await self._fetch_trades(token_address, chain, time_window)
            if not trades:
                return self._empty_profile(token_address, chain)
            
            # Analyze volume patterns
            wash_trading_score = await self._detect_wash_trading(trades)
            volume_patterns = await self._identify_patterns(trades)
            trade_clusters = await self._find_trade_clusters(trades)
            
            # Calculate metrics
            organic_volume, suspicious_volume = await self._separate_volume(trades, trade_clusters)
            unique_traders = len(set(t["from"] for t in trades) | set(t["to"] for t in trades))
            dex_distribution = await self._analyze_dex_distribution(trades)
            time_distribution = self._analyze_time_distribution(trades)
            whale_percent, retail_percent = await self._analyze_trader_types(trades)
            smart_money_flow = await self._analyze_smart_money(trades)
            
            # Generate risk flags
            risk_flags = []
            if wash_trading_score > self.wash_trading_threshold:
                risk_flags.append("HIGH_WASH_TRADING_RISK")
            if unique_traders < self.min_unique_traders:
                risk_flags.append("LOW_UNIQUE_TRADERS")
            if VolumePattern.PUMP_SCHEME in volume_patterns:
                risk_flags.append("PUMP_SCHEME_DETECTED")
            if len(dex_distribution) == 1:
                risk_flags.append("SINGLE_DEX_CONCENTRATION")
            
            # Calculate confidence score
            confidence = self._calculate_confidence(
                trades_count=len(trades),
                unique_traders=unique_traders,
                time_coverage=self._calculate_time_coverage(trades, time_window)
            )
            
            profile = VolumeProfile(
                token_address=token_address,
                chain=chain,
                total_volume_24h=sum(Decimal(str(t["amount_usd"])) for t in trades),
                organic_volume=organic_volume,
                suspicious_volume=suspicious_volume,
                wash_trading_score=wash_trading_score,
                unique_traders=unique_traders,
                average_trade_size=self._calculate_avg_trade_size(trades),
                volume_velocity=self._calculate_velocity(trades),
                volume_patterns=volume_patterns,
                dex_distribution=dex_distribution,
                time_distribution=time_distribution,
                whale_volume_percent=whale_percent,
                retail_volume_percent=retail_percent,
                smart_money_flow=smart_money_flow,
                confidence_score=confidence,
                risk_flags=risk_flags
            )
            
            # Update cache
            self.volume_cache[cache_key] = profile
            
            # Log findings
            if risk_flags:
                logger.warning(f"Volume risks for {token_address}: {risk_flags}")
            
            return profile
            
        except Exception as e:
            logger.error(f"Volume analysis failed for {token_address}: {e}")
            return self._empty_profile(token_address, chain)
    
    async def _detect_wash_trading(self, trades: List[Dict]) -> float:
        """
        Detect wash trading patterns
        
        Returns:
            Probability score 0-1
        """
        if len(trades) < 10:
            return 0.0
        
        indicators = []
        
        # 1. Circular trading detection
        circular_score = await self._detect_circular_trades(trades)
        indicators.append(circular_score * 0.3)
        
        # 2. Trade timing patterns (regular intervals)
        timing_score = self._analyze_trade_timing(trades)
        indicators.append(timing_score * 0.2)
        
        # 3. Trade size patterns (similar amounts)
        size_score = self._analyze_trade_sizes(trades)
        indicators.append(size_score * 0.2)
        
        # 4. Address reuse patterns
        address_score = self._analyze_address_patterns(trades)
        indicators.append(address_score * 0.2)
        
        # 5. Price impact analysis (no impact = suspicious)
        impact_score = await self._analyze_price_impact(trades)
        indicators.append(impact_score * 0.1)
        
        return min(sum(indicators), 1.0)
    
    async def _detect_circular_trades(self, trades: List[Dict]) -> float:
        """Detect circular trading patterns between addresses"""
        address_graph = defaultdict(lambda: defaultdict(int))
        
        for trade in trades:
            address_graph[trade["from"]][trade["to"]] += 1
        
        # Find cycles
        cycles = 0
        total_paths = 0
        
        for start_addr in address_graph:
            for end_addr in address_graph[start_addr]:
                total_paths += 1
                # Check if there's a return path
                if end_addr in address_graph and start_addr in address_graph[end_addr]:
                    cycles += 1
        
        if total_paths == 0:
            return 0.0
            
        return cycles / total_paths
    
    def _analyze_trade_timing(self, trades: List[Dict]) -> float:
        """Analyze if trades occur at regular intervals (bot behavior)"""
        if len(trades) < 3:
            return 0.0
        
        timestamps = sorted([t["timestamp"] for t in trades])
        intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
        
        if not intervals:
            return 0.0
        
        # Calculate coefficient of variation
        mean_interval = statistics.mean(intervals)
        if mean_interval == 0:
            return 0.0
            
        std_interval = statistics.stdev(intervals) if len(intervals) > 1 else 0
        cv = std_interval / mean_interval if mean_interval > 0 else 0
        
        # Lower CV means more regular intervals (suspicious)
        if cv < 0.1:  # Very regular
            return 1.0
        elif cv < 0.3:  # Somewhat regular
            return 0.7
        elif cv < 0.5:  # Moderately regular
            return 0.4
        else:  # Irregular (organic)
            return 0.0
    
    def _analyze_trade_sizes(self, trades: List[Dict]) -> float:
        """Analyze if trade sizes are suspiciously similar"""
        amounts = [Decimal(str(t["amount_usd"])) for t in trades]
        if len(amounts) < 2:
            return 0.0
        
        # Group similar amounts (within 5%)
        clusters = []
        for amount in amounts:
            added = False
            for cluster in clusters:
                if abs(amount - cluster[0]) / cluster[0] < Decimal("0.05"):
                    cluster.append(amount)
                    added = True
                    break
            if not added:
                clusters.append([amount])
        
        # Calculate clustering ratio
        max_cluster_size = max(len(c) for c in clusters)
        clustering_ratio = max_cluster_size / len(amounts)
        
        return min(clustering_ratio * 1.5, 1.0)
    
    def _analyze_address_patterns(self, trades: List[Dict]) -> float:
        """Analyze address reuse and patterns"""
        addresses = []
        for trade in trades:
            addresses.append(trade["from"])
            addresses.append(trade["to"])
        
        total_addresses = len(addresses)
        unique_addresses = len(set(addresses))
        
        if unique_addresses == 0:
            return 1.0
        
        reuse_ratio = 1 - (unique_addresses / total_addresses)
        
        # Check for known wash trading addresses
        known_wash = sum(1 for addr in addresses if addr in self.wash_trading_addresses)
        wash_ratio = known_wash / total_addresses if total_addresses > 0 else 0
        
        return min(reuse_ratio + wash_ratio, 1.0)
    
    async def _analyze_price_impact(self, trades: List[Dict]) -> float:
        """Analyze if trades have realistic price impact"""
        # Large volume with no price impact is suspicious
        if not trades:
            return 0.0
        
        price_changes = []
        for i in range(1, len(trades)):
            if trades[i]["price"] and trades[i-1]["price"]:
                change = abs(trades[i]["price"] - trades[i-1]["price"]) / trades[i-1]["price"]
                price_changes.append(change)
        
        if not price_changes:
            return 0.0
        
        avg_impact = statistics.mean(price_changes)
        total_volume = sum(Decimal(str(t["amount_usd"])) for t in trades)
        
        # Expected impact based on volume
        expected_impact = float(total_volume) * 0.00001  # Rough estimate
        
        if expected_impact > 0 and avg_impact < expected_impact * 0.1:
            return 0.8  # Very low impact for volume
        elif avg_impact < expected_impact * 0.3:
            return 0.5
        else:
            return 0.0  # Normal impact
    
    async def _identify_patterns(self, trades: List[Dict]) -> List[VolumePattern]:
        """Identify volume patterns in trades"""
        patterns = []
        
        # Wash trading pattern
        wash_score = await self._detect_wash_trading(trades)
        if wash_score > self.wash_trading_threshold:
            patterns.append(VolumePattern.WASH_TRADING)
        
        # Bot trading pattern
        if self._detect_bot_trading(trades):
            patterns.append(VolumePattern.BOT_TRADING)
        
        # Pump scheme pattern
        if self._detect_pump_pattern(trades):
            patterns.append(VolumePattern.PUMP_SCHEME)
        
        # Accumulation/Distribution
        trend = self._detect_accumulation_distribution(trades)
        if trend:
            patterns.append(trend)
        
        # Volume breakout
        if self._detect_breakout(trades):
            patterns.append(VolumePattern.BREAKOUT)
        
        # If no suspicious patterns, mark as organic
        if not patterns:
            patterns.append(VolumePattern.ORGANIC)
        
        return patterns
    
    def _detect_bot_trading(self, trades: List[Dict]) -> bool:
        """Detect bot trading patterns"""
        bot_indicators = 0
        
        # Check for known bot addresses
        bot_trades = sum(1 for t in trades 
                         if t["from"] in self.bot_addresses or t["to"] in self.bot_addresses)
        if bot_trades / len(trades) > 0.3:
            bot_indicators += 1
        
        # Check for rapid trades
        rapid_trades = []
        for i in range(1, len(trades)):
            if trades[i]["timestamp"] - trades[i-1]["timestamp"] < 5:  # 5 seconds
                rapid_trades.append(trades[i])
        
        if len(rapid_trades) / len(trades) > 0.4:
            bot_indicators += 1
        
        # Check for exact amount patterns
        amounts = [str(t["amount"]) for t in trades]
        exact_amounts = sum(1 for a in amounts if a.endswith("00000"))
        if exact_amounts / len(amounts) > 0.5:
            bot_indicators += 1
        
        return bot_indicators >= 2
    
    def _detect_pump_pattern(self, trades: List[Dict]) -> bool:
        """Detect pump and dump patterns"""
        if len(trades) < 20:
            return False
        
        # Sort trades by timestamp
        sorted_trades = sorted(trades, key=lambda x: x["timestamp"])
        
        # Calculate volume acceleration
        first_quarter = sorted_trades[:len(sorted_trades)//4]
        last_quarter = sorted_trades[3*len(sorted_trades)//4:]
        
        early_volume = sum(Decimal(str(t["amount_usd"])) for t in first_quarter)
        late_volume = sum(Decimal(str(t["amount_usd"])) for t in last_quarter)
        
        if early_volume == 0:
            return False
        
        volume_acceleration = late_volume / early_volume
        
        # Check for sudden spike followed by decline
        return volume_acceleration > self.volume_spike_multiplier
    
    def _detect_accumulation_distribution(self, trades: List[Dict]) -> Optional[VolumePattern]:
        """Detect accumulation or distribution patterns"""
        if len(trades) < 10:
            return None
        
        buy_volume = sum(Decimal(str(t["amount_usd"])) for t in trades if t.get("side") == "buy")
        sell_volume = sum(Decimal(str(t["amount_usd"])) for t in trades if t.get("side") == "sell")
        
        total_volume = buy_volume + sell_volume
        if total_volume == 0:
            return None
        
        buy_ratio = buy_volume / total_volume
        
        if buy_ratio > 0.65:
            return VolumePattern.ACCUMULATION
        elif buy_ratio < 0.35:
            return VolumePattern.DISTRIBUTION
        
        return None
    
    def _detect_breakout(self, trades: List[Dict]) -> bool:
        """Detect volume breakout patterns"""
        if len(trades) < 10:
            return False
        
        volumes = [Decimal(str(t["amount_usd"])) for t in trades]
        avg_volume = statistics.mean(volumes)
        recent_volume = statistics.mean(volumes[-5:])
        
        return recent_volume > avg_volume * 3
    
    async def _find_trade_clusters(self, trades: List[Dict]) -> List[TradeCluster]:
        """Find clusters of related trades"""
        clusters = []
        used_trades = set()
        
        for i, trade in enumerate(trades):
            if i in used_trades:
                continue
            
            cluster_trades = [trade]
            cluster_wallets = {trade["from"], trade["to"]}
            used_trades.add(i)
            
            # Find similar trades
            for j, other_trade in enumerate(trades[i+1:], i+1):
                if j in used_trades:
                    continue
                
                similarity = self._calculate_trade_similarity(trade, other_trade)
                if similarity > self.trade_similarity_threshold:
                    cluster_trades.append(other_trade)
                    cluster_wallets.update({other_trade["from"], other_trade["to"]})
                    used_trades.add(j)
            
            if len(cluster_trades) >= self.min_cluster_size:
                cluster = TradeCluster(
                    trades=cluster_trades,
                    wallets=cluster_wallets,
                    total_volume=sum(Decimal(str(t["amount_usd"])) for t in cluster_trades),
                    time_span=timedelta(seconds=max(t["timestamp"] for t in cluster_trades) - 
                                              min(t["timestamp"] for t in cluster_trades)),
                    pattern_type=VolumePattern.WASH_TRADING,
                    similarity_score=similarity
                )
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_trade_similarity(self, trade1: Dict, trade2: Dict) -> float:
        """Calculate similarity between two trades"""
        scores = []
        
        # Time similarity (within 5 minutes)
        time_diff = abs(trade1["timestamp"] - trade2["timestamp"])
        time_score = max(0, 1 - time_diff / 300)
        scores.append(time_score * 0.3)
        
        # Amount similarity (within 10%)
        amount1 = Decimal(str(trade1["amount_usd"]))
        amount2 = Decimal(str(trade2["amount_usd"]))
        amount_diff = abs(amount1 - amount2) / max(amount1, amount2)
        amount_score = max(0, 1 - amount_diff / Decimal("0.1"))
        scores.append(float(amount_score) * 0.3)
        
        # Address overlap
        addresses1 = {trade1["from"], trade1["to"]}
        addresses2 = {trade2["from"], trade2["to"]}
        overlap = len(addresses1 & addresses2)
        address_score = overlap / 2
        scores.append(address_score * 0.4)
        
        return sum(scores)
    
    async def _separate_volume(
        self,
        trades: List[Dict],
        clusters: List[TradeCluster]
    ) -> Tuple[Decimal, Decimal]:
        """Separate organic and suspicious volume"""
        suspicious_trades = set()
        
        for cluster in clusters:
            for trade in cluster.trades:
                # Find trade index
                for i, t in enumerate(trades):
                    if (t["timestamp"] == trade["timestamp"] and 
                        t["from"] == trade["from"] and 
                        t["to"] == trade["to"]):
                        suspicious_trades.add(i)
                        break
        
        organic_volume = Decimal("0")
        suspicious_volume = Decimal("0")
        
        for i, trade in enumerate(trades):
            volume = Decimal(str(trade["amount_usd"]))
            if i in suspicious_trades:
                suspicious_volume += volume
            else:
                organic_volume += volume
        
        return organic_volume, suspicious_volume
    
    async def _analyze_dex_distribution(self, trades: List[Dict]) -> Dict[str, Decimal]:
        """Analyze volume distribution across DEXes"""
        dex_volumes = defaultdict(Decimal)
        
        for trade in trades:
            dex = trade.get("dex", "unknown")
            volume = Decimal(str(trade["amount_usd"]))
            dex_volumes[dex] += volume
        
        return dict(dex_volumes)
    
    def _analyze_time_distribution(self, trades: List[Dict]) -> Dict[int, Decimal]:
        """Analyze volume distribution over time (hourly)"""
        hourly_volumes = defaultdict(Decimal)
        
        for trade in trades:
            hour = datetime.fromtimestamp(trade["timestamp"]).hour
            volume = Decimal(str(trade["amount_usd"]))
            hourly_volumes[hour] += volume
        
        return dict(hourly_volumes)
    
    async def _analyze_trader_types(self, trades: List[Dict]) -> Tuple[float, float]:
        """Analyze whale vs retail volume percentages"""
        whale_threshold = Decimal("10000")  # $10k+ trades
        
        whale_volume = Decimal("0")
        retail_volume = Decimal("0")
        
        for trade in trades:
            volume = Decimal(str(trade["amount_usd"]))
            if volume >= whale_threshold:
                whale_volume += volume
            else:
                retail_volume += volume
        
        total_volume = whale_volume + retail_volume
        if total_volume == 0:
            return 0.0, 0.0
        
        whale_percent = float(whale_volume / total_volume * 100)
        retail_percent = float(retail_volume / total_volume * 100)
        
        return whale_percent, retail_percent
    
    async def _analyze_smart_money(self, trades: List[Dict]) -> str:
        """Analyze smart money flow direction"""
        # This would integrate with smart money wallet tracking
        # For now, return based on large trade analysis
        
        large_trades = [t for t in trades if Decimal(str(t["amount_usd"])) > Decimal("5000")]
        if not large_trades:
            return "neutral"
        
        buy_volume = sum(Decimal(str(t["amount_usd"])) for t in large_trades if t.get("side") == "buy")
        sell_volume = sum(Decimal(str(t["amount_usd"])) for t in large_trades if t.get("side") == "sell")
        
        net_flow = buy_volume - sell_volume
        total_volume = buy_volume + sell_volume
        
        if total_volume == 0:
            return "neutral"
        
        flow_ratio = net_flow / total_volume
        
        if flow_ratio > Decimal("0.2"):
            return "in"
        elif flow_ratio < Decimal("-0.2"):
            return "out"
        else:
            return "neutral"
    
    def _calculate_avg_trade_size(self, trades: List[Dict]) -> Decimal:
        """Calculate average trade size"""
        if not trades:
            return Decimal("0")
        
        total_volume = sum(Decimal(str(t["amount_usd"])) for t in trades)
        return total_volume / len(trades)
    
    def _calculate_velocity(self, trades: List[Dict]) -> float:
        """Calculate volume velocity (rate of change)"""
        if len(trades) < 2:
            return 0.0
        
        # Group trades by hour
        hourly_volumes = defaultdict(Decimal)
        for trade in trades:
            hour_key = trade["timestamp"] // 3600
            hourly_volumes[hour_key] += Decimal(str(trade["amount_usd"]))
        
        if len(hourly_volumes) < 2:
            return 0.0
        
        # Calculate hour-over-hour changes
        sorted_hours = sorted(hourly_volumes.keys())
        changes = []
        
        for i in range(1, len(sorted_hours)):
            prev_volume = hourly_volumes[sorted_hours[i-1]]
            curr_volume = hourly_volumes[sorted_hours[i]]
            if prev_volume > 0:
                change = float((curr_volume - prev_volume) / prev_volume)
                changes.append(change)
        
        return statistics.mean(changes) if changes else 0.0
    
    def _calculate_time_coverage(self, trades: List[Dict], expected_hours: int) -> float:
        """Calculate how much of the expected time window has trades"""
        if not trades:
            return 0.0
        
        # Find unique hours with trades
        hours_with_trades = set()
        for trade in trades:
            hour_key = trade["timestamp"] // 3600
            hours_with_trades.add(hour_key)
        
        coverage = len(hours_with_trades) / expected_hours
        return min(coverage, 1.0)
    
    def _calculate_confidence(
        self,
        trades_count: int,
        unique_traders: int,
        time_coverage: float
    ) -> float:
        """Calculate confidence score for the analysis"""
        scores = []
        
        # Trade count score
        if trades_count >= 1000:
            scores.append(1.0)
        elif trades_count >= 100:
            scores.append(0.7)
        elif trades_count >= 10:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        # Unique traders score
        if unique_traders >= 100:
            scores.append(1.0)
        elif unique_traders >= 50:
            scores.append(0.7)
        elif unique_traders >= 10:
            scores.append(0.4)
        else:
            scores.append(0.1)
        
        # Time coverage score
        scores.append(time_coverage)
        
        return statistics.mean(scores)
    
    async def _fetch_trades(
        self,
        token_address: str,
        chain: str,
        hours: int
    ) -> List[Dict]:
        """Fetch trade data from multiple sources"""
        # This would integrate with DexScreener and chain data collectors
        # Placeholder implementation
        return []
    
    async def _load_known_wash_traders(self) -> None:
        """Load known wash trading addresses"""
        # This would load from database or external source
        pass
    
    async def _load_bot_addresses(self) -> None:
        """Load known bot addresses"""
        # This would load from database or external source
        pass
    
    def _empty_profile(self, token_address: str, chain: str) -> VolumeProfile:
        """Return empty profile when analysis fails"""
        return VolumeProfile(
            token_address=token_address,
            chain=chain,
            total_volume_24h=Decimal("0"),
            organic_volume=Decimal("0"),
            suspicious_volume=Decimal("0"),
            wash_trading_score=0.0,
            unique_traders=0,
            average_trade_size=Decimal("0"),
            volume_velocity=0.0,
            volume_patterns=[],
            dex_distribution={},
            time_distribution={},
            whale_volume_percent=0.0,
            retail_volume_percent=0.0,
            smart_money_flow="neutral",
            confidence_score=0.0,
            risk_flags=["NO_DATA"]
        )
    
    async def monitor_volume_health(
        self,
        token_address: str,
        chain: str,
        callback: Optional[Any] = None
    ) -> None:
        """Continuously monitor volume health"""
        while True:
            try:
                profile = await self.analyze_volume(token_address, chain)
                
                if callback and profile.risk_flags:
                    await callback(profile)
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Volume monitoring error: {e}")
                await asyncio.sleep(60)