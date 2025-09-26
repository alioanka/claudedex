# analysis/pump_predictor.py

import asyncio
import logging
from typing import Dict, List, Optional, Tuple, Any
from decimal import Decimal
from datetime import datetime, timedelta
from dataclasses import dataclass, field
import numpy as np
import pandas as pd
from scipy import signal, stats
import warnings
warnings.filterwarnings('ignore')

from ..core.event_bus import EventBus
from ..data.storage.database import DatabaseManager
from ..data.storage.cache import CacheManager
from ..ml.models.pump_predictor import PumpPredictor as MLPumpPredictor
from .market_analyzer import MarketAnalyzer

logger = logging.getLogger(__name__)

@dataclass
class PumpSignal:
    """Pump detection signal"""
    token: str
    chain: str
    probability: float  # 0-1
    confidence: float  # 0-1
    signal_strength: str  # 'weak', 'moderate', 'strong', 'very_strong'
    type: str  # 'organic', 'coordinated', 'whale', 'social', 'technical'
    expected_magnitude: float  # Expected % increase
    expected_duration: int  # Expected duration in minutes
    risk_level: str  # 'low', 'medium', 'high', 'extreme'
    entry_price: Decimal
    target_prices: List[Decimal]
    stop_loss: Decimal
    indicators: Dict[str, Any]
    timestamp: datetime

@dataclass
class PumpPattern:
    """Historical pump pattern"""
    pattern_type: str
    occurrences: int
    avg_magnitude: float
    avg_duration: float
    success_rate: float
    last_seen: datetime

class PumpPredictorAnalysis:
    """Advanced pump prediction and analysis"""
    
    def __init__(
        self,
        db_manager: DatabaseManager,
        cache_manager: CacheManager,
        event_bus: EventBus,
        ml_predictor: MLPumpPredictor,
        market_analyzer: MarketAnalyzer,
        config: Dict[str, Any]
    ):
        self.db = db_manager
        self.cache = cache_manager
        self.event_bus = event_bus
        self.ml_predictor = ml_predictor
        self.market_analyzer = market_analyzer
        self.config = config
        
        # Prediction parameters
        self.min_probability_threshold = config.get('min_pump_probability', 0.65)
        self.volume_spike_threshold = config.get('volume_spike_threshold', 3.0)
        self.price_spike_threshold = config.get('price_spike_threshold', 0.05)
        
        # Pattern recognition parameters
        self.patterns = {
            'accumulation': {
                'min_duration': 60,  # minutes
                'max_volatility': 0.02,
                'volume_pattern': 'increasing'
            },
            'breakout': {
                'resistance_tests': 3,
                'volume_confirmation': 2.0,
                'momentum_threshold': 0.7
            },
            'whale_pump': {
                'min_buy_size': 50000,  # USD
                'wallet_reputation': 'known',
                'follow_through_rate': 0.6
            },
            'social_pump': {
                'mention_spike': 5.0,
                'sentiment_threshold': 0.8,
                'influencer_involvement': True
            }
        }
        
        # Tracking data
        self._active_predictions: Dict[str, PumpSignal] = {}
        self._pattern_history: Dict[str, List[PumpPattern]] = {}
        
    async def predict_pump(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> PumpSignal:
        """Predict pump probability for a token"""
        try:
            # Get comprehensive data
            data = await self._gather_prediction_data(token, chain)
            
            # Use ML model for prediction
            ml_prediction = await self.ml_predictor.predict_pump_probability(
                pd.DataFrame(data['price_data'])
            )
            
            # Analyze patterns
            pattern_signals = await self._analyze_pump_patterns(data)
            
            # Check for accumulation phase
            accumulation = await self._detect_accumulation_phase(data)
            
            # Analyze whale activity
            whale_signals = await self._analyze_whale_activity(data)
            
            # Analyze social signals
            social_signals = await self._analyze_social_momentum(data)
            
            # Combine all signals
            combined_probability = await self._combine_signals(
                ml_prediction[0],  # ML probability
                pattern_signals,
                accumulation,
                whale_signals,
                social_signals
            )
            
            # Determine pump type
            pump_type = self._identify_pump_type(
                pattern_signals,
                whale_signals,
                social_signals
            )
            
            # Calculate expected metrics
            expected_metrics = await self._calculate_expected_metrics(
                data,
                combined_probability,
                pump_type
            )
            
            # Calculate entry/exit points
            entry_exit = await self._calculate_entry_exit_points(
                data,
                expected_metrics
            )
            
            # Determine risk level
            risk_level = self._assess_pump_risk(
                combined_probability,
                pump_type,
                data
            )
            
            # Create signal
            signal = PumpSignal(
                token=token,
                chain=chain,
                probability=combined_probability,
                confidence=ml_prediction[1].get('confidence', 0.5),
                signal_strength=self._get_signal_strength(combined_probability),
                type=pump_type,
                expected_magnitude=expected_metrics['magnitude'],
                expected_duration=expected_metrics['duration'],
                risk_level=risk_level,
                entry_price=entry_exit['entry'],
                target_prices=entry_exit['targets'],
                stop_loss=entry_exit['stop_loss'],
                indicators=ml_prediction[1],
                timestamp=datetime.utcnow()
            )
            
            # Cache and emit if significant
            if combined_probability >= self.min_probability_threshold:
                await self._process_significant_signal(signal)
                
            return signal
            
        except Exception as e:
            logger.error(f"Error predicting pump for {token}: {e}")
            return self._get_default_signal(token, chain)
            
    async def detect_ongoing_pump(
        self,
        token: str,
        chain: str = 'ethereum'
    ) -> Dict[str, Any]:
        """Detect if a pump is currently happening"""
        try:
            # Get real-time data
            current_data = await self._get_realtime_data(token, chain)
            recent_data = await self._get_recent_price_volume(token, chain, minutes=30)
            
            # Check for price spike
            price_spike = await self._detect_price_spike(recent_data)
            
            # Check for volume spike
            volume_spike = await self._detect_volume_spike(recent_data)
            
            # Check momentum indicators
            momentum = await self._calculate_momentum_indicators(recent_data)
            
            # Determine pump status
            is_pumping = (
                price_spike['detected'] and 
                volume_spike['detected'] and
                momentum['rsi'] > 60
            )
            
            # Calculate pump metrics if detected
            pump_metrics = {}
            if is_pumping:
                pump_metrics = await self._calculate_pump_metrics(
                    recent_data,
                    price_spike,
                    volume_spike
                )
                
            # Estimate remaining potential
            remaining_potential = await self._estimate_remaining_potential(
                recent_data,
                pump_metrics
            ) if is_pumping else 0
            
            # Get exit recommendation
            exit_recommendation = self._get_exit_recommendation(
                pump_metrics,
                remaining_potential
            ) if is_pumping else None
            
            return {
                'token': token,
                'chain': chain,
                'is_pumping': is_pumping,
                'pump_stage': pump_metrics.get('stage', 'none'),
                'price_increase': price_spike.get('magnitude', 0),
                'volume_increase': volume_spike.get('magnitude', 0),
                'momentum': momentum,
                'pump_duration': pump_metrics.get('duration_minutes', 0),
                'remaining_potential': remaining_potential,
                'exit_recommendation': exit_recommendation,
                'confidence': pump_metrics.get('confidence', 0),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error detecting ongoing pump for {token}: {e}")
            return {'is_pumping': False, 'error': str(e)}
            
    async def analyze_pump_history(
        self,
        token: str,
        chain: str = 'ethereum',
        days: int = 30
    ) -> Dict[str, Any]:
        """Analyze historical pump patterns for a token"""
        try:
            # Get historical data
            history = await self.db.get_pump_history(token, chain, days)
            
            if not history:
                return {'patterns': [], 'statistics': {}}
                
            # Identify pump events
            pump_events = await self._identify_historical_pumps(history)
            
            # Analyze patterns
            patterns = await self._analyze_pump_patterns_historical(pump_events)
            
            # Calculate statistics
            statistics = {
                'total_pumps': len(pump_events),
                'avg_magnitude': np.mean([p['magnitude'] for p in pump_events]) if pump_events else 0,
                'avg_duration': np.mean([p['duration'] for p in pump_events]) if pump_events else 0,
                'pump_frequency': len(pump_events) / max(days, 1),
                'success_rate': sum(1 for p in pump_events if p['successful']) / max(len(pump_events), 1)
            }
            
            # Identify most common patterns
            common_patterns = self._identify_common_patterns(patterns)
            
            # Predict next pump timing
            next_pump_prediction = await self._predict_next_pump_timing(
                pump_events,
                patterns
            )
            
            return {
                'token': token,
                'chain': chain,
                'pump_events': pump_events,
                'patterns': patterns,
                'statistics': statistics,
                'common_patterns': common_patterns,
                'next_pump_prediction': next_pump_prediction,
                'analysis_period_days': days,
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pump history for {token}: {e}")
            return {'patterns': [], 'statistics': {}}
            
    async def monitor_pump_completion(
        self,
        signal: PumpSignal
    ) -> Dict[str, Any]:
        """Monitor and analyze pump completion"""
        try:
            # Get current price
            current_data = await self._get_realtime_data(signal.token, signal.chain)
            current_price = Decimal(str(current_data['price']))
            
            # Calculate performance
            price_change = float((current_price - signal.entry_price) / signal.entry_price)
            
            # Check target achievement
            targets_hit = [
                i for i, target in enumerate(signal.target_prices)
                if current_price >= target
            ]
            
            # Calculate duration
            duration = (datetime.utcnow() - signal.timestamp).total_seconds() / 60
            
            # Determine completion status
            status = self._determine_pump_status(
                price_change,
                signal.expected_magnitude,
                duration,
                signal.expected_duration,
                targets_hit
            )
            
            # Update prediction accuracy
            accuracy_data = {
                'predicted_magnitude': signal.expected_magnitude,
                'actual_magnitude': price_change,
                'predicted_duration': signal.expected_duration,
                'actual_duration': duration,
                'prediction_accuracy': 1 - abs(price_change - signal.expected_magnitude) / max(signal.expected_magnitude, 0.01)
            }
            
            # Store result for model improvement
            await self.db.save_pump_result({
                'signal': signal.__dict__,
                'result': accuracy_data,
                'status': status,
                'timestamp': datetime.utcnow()
            })
            
            return {
                'signal_id': f"{signal.token}_{signal.timestamp}",
                'status': status,
                'current_price': current_price,
                'price_change': price_change,
                'targets_hit': targets_hit,
                'duration_minutes': duration,
                'accuracy': accuracy_data,
                'recommendation': self._get_completion_recommendation(status, targets_hit),
                'timestamp': datetime.utcnow()
            }
            
        except Exception as e:
            logger.error(f"Error monitoring pump completion: {e}")
            return {}
            
    # Private helper methods
    
    async def _gather_prediction_data(self, token: str, chain: str) -> Dict[str, Any]:
        """Gather all data needed for pump prediction"""
        data = {}
        
        # Price and volume data
        data['price_data'] = await self.db.get_price_data(token, chain, hours=24)
        data['volume_data'] = await self.db.get_volume_data(token, chain, hours=24)
        
        # Order book data
        data['orderbook'] = await self._get_orderbook_data(token, chain)
        
        # Whale wallets
        data['whale_activity'] = await self.db.get_whale_activity(token, chain, hours=6)
        
        # Social data
        data['social'] = await self._get_social_data(token)
        
        # Market context
        data['market'] = await self.market_analyzer.analyze_market_conditions([token], chain)
        
        return data
        
    async def _analyze_pump_patterns(self, data: Dict) -> Dict[str, Any]:
        """Analyze data for pump patterns"""
        patterns = {}
        
        # Check accumulation pattern
        if await self._check_accumulation_pattern(data):
            patterns['accumulation'] = {
                'detected': True,
                'strength': await self._calculate_accumulation_strength(data)
            }
            
        # Check breakout pattern
        if await self._check_breakout_pattern(data):
            patterns['breakout'] = {
                'detected': True,
                'strength': await self._calculate_breakout_strength(data)
            }
            
        # Check volume pattern
        volume_pattern = await self._analyze_volume_pattern(data)
        if volume_pattern['bullish']:
            patterns['volume'] = volume_pattern
            
        return patterns
        
    async def _detect_accumulation_phase(self, data: Dict) -> Dict[str, Any]:
        """Detect if token is in accumulation phase"""
        try:
            price_data = pd.DataFrame(data['price_data'])
            if len(price_data) < 20:
                return {'detected': False}
                
            # Calculate price range
            price_range = price_data['price'].max() - price_data['price'].min()
            avg_price = price_data['price'].mean()
            range_percentage = (price_range / avg_price) * 100
            
            # Check for low volatility
            volatility = price_data['price'].pct_change().std()
            
            # Check for increasing volume on dips
            dips = price_data[price_data['price'] < avg_price]
            volume_on_dips = dips['volume'].mean() if len(dips) > 0 else 0
            avg_volume = price_data['volume'].mean()
            
            # Accumulation detected if:
            # 1. Low volatility (range < 5%)
            # 2. Higher volume on dips
            # 3. Price holding support
            is_accumulating = (
                range_percentage < 5 and
                volatility < 0.02 and
                volume_on_dips > avg_volume
            )
            
            return {
                'detected': is_accumulating,
                'range_percentage': range_percentage,
                'volatility': volatility,
                'volume_ratio': volume_on_dips / avg_volume if avg_volume > 0 else 0,
                'duration_hours': len(price_data) / 60 if is_accumulating else 0
            }
            
        except Exception as e:
            logger.error(f"Error detecting accumulation phase: {e}")
            return {'detected': False}
            
    async def _combine_signals(
        self,
        ml_probability: float,
        pattern_signals: Dict,
        accumulation: Dict,
        whale_signals: Dict,
        social_signals: Dict
    ) -> float:
        """Combine multiple signals into final probability"""
        weights = {
            'ml': 0.35,
            'patterns': 0.25,
            'accumulation': 0.15,
            'whale': 0.15,
            'social': 0.10
        }
        
        probabilities = {
            'ml': ml_probability,
            'patterns': self._calculate_pattern_probability(pattern_signals),
            'accumulation': 0.8 if accumulation.get('detected') else 0.2,
            'whale': whale_signals.get('pump_probability', 0.5),
            'social': social_signals.get('pump_probability', 0.5)
        }
        
        # Weighted average
        combined = sum(
            probabilities[key] * weights[key]
            for key in weights.keys()
        )
        
        # Boost if multiple strong signals
        strong_signals = sum(1 for p in probabilities.values() if p > 0.7)
        if strong_signals >= 3:
            combined = min(1.0, combined * 1.2)
            
        return combined
        
    def _identify_pump_type(
        self,
        pattern_signals: Dict,
        whale_signals: Dict,
        social_signals: Dict
    ) -> str:
        """Identify the type of pump"""
        if whale_signals.get('large_buys_detected'):
            return 'whale'
        elif social_signals.get('viral_detected'):
            return 'social'
        elif pattern_signals.get('breakout', {}).get('detected'):
            return 'technical'
        elif pattern_signals.get('accumulation', {}).get('detected'):
            return 'coordinated'
        else:
            return 'organic'
            
    async def _calculate_expected_metrics(
        self,
        data: Dict,
        probability: float,
        pump_type: str
    ) -> Dict[str, Any]:
        """Calculate expected pump metrics"""
        # Base expectations by pump type
        type_expectations = {
            'whale': {'magnitude': 0.5, 'duration': 30},
            'social': {'magnitude': 0.8, 'duration': 60},
            'technical': {'magnitude': 0.3, 'duration': 120},
            'coordinated': {'magnitude': 0.4, 'duration': 45},
            'organic': {'magnitude': 0.2, 'duration': 180}
        }
        
        base = type_expectations.get(pump_type, {'magnitude': 0.2, 'duration': 60})
        
        # Adjust based on probability
        magnitude = base['magnitude'] * (0.5 + probability * 0.5)
        duration = base['duration'] * (0.7 + probability * 0.3)
        
        # Adjust based on market conditions
        market = data.get('market', {})
        if market.get('trend') == 'bullish':
            magnitude *= 1.2
            duration *= 1.1
            
        return {
            'magnitude': magnitude,
            'duration': int(duration)
        }
        
    async def _calculate_entry_exit_points(
        self,
        data: Dict,
        expected_metrics: Dict
    ) -> Dict[str, Any]:
        """Calculate optimal entry and exit points"""
        current_price = Decimal(str(data['price_data'][-1]['price']))
        
        # Entry point (current price or slight pullback)
        entry = current_price * Decimal('0.995')  # 0.5% below current
        
        # Target prices based on expected magnitude
        magnitude = Decimal(str(1 + expected_metrics['magnitude']))
        targets = [
            entry * Decimal('1.1'),  # First target: 10%
            entry * Decimal('1.2'),  # Second target: 20%
            entry * magnitude,  # Final target: expected magnitude
        ]
        
        # Stop loss
        stop_loss = entry * Decimal('0.95')  # 5% stop loss
        
        return {
            'entry': entry,
            'targets': targets,
            'stop_loss': stop_loss
        }
        
    def _assess_pump_risk(
        self,
        probability: float,
        pump_type: str,
        data: Dict
    ) -> str:
        """Assess risk level of pump trade"""
        risk_score = 0
        
        # Probability factor
        if probability < 0.6:
            risk_score += 30
        elif probability < 0.7:
            risk_score += 15
            
        # Pump type factor
        type_risks = {
            'whale': 40,
            'coordinated': 35,
            'social': 30,
            'technical': 20,
            'organic': 10
        }
        risk_score += type_risks.get(pump_type, 25)
        
        # Market condition factor
        market = data.get('market', {})
        if market.get('volatility') == 'high':
            risk_score += 20
        elif market.get('volatility') == 'extreme':
            risk_score += 30
            
        # Determine risk level
        if risk_score >= 70:
            return 'extreme'
        elif risk_score >= 50:
            return 'high'
        elif risk_score >= 30:
            return 'medium'
        else:
            return 'low'
            
    def _get_signal_strength(self, probability: float) -> str:
        """Determine signal strength"""
        if probability >= 0.85:
            return 'very_strong'
        elif probability >= 0.75:
            return 'strong'
        elif probability >= 0.65:
            return 'moderate'
        else:
            return 'weak'
            
    async def _process_significant_signal(self, signal: PumpSignal) -> None:
        """Process and alert for significant pump signals"""
        # Store in database
        await self.db.save_pump_signal(signal.__dict__)
        
        # Cache for monitoring
        cache_key = f"pump_signal:{signal.chain}:{signal.token}"
        await self.cache.set(cache_key, signal.__dict__, ttl=3600)
        
        # Track active prediction
        self._active_predictions[signal.token] = signal
        
        # Emit event
        await self.event_bus.emit('pump.signal', signal.__dict__)
        
        # Send alert if strong signal
        if signal.signal_strength in ['strong', 'very_strong']:
            message = f"""
    🚀 **PUMP SIGNAL DETECTED**

    Token: {signal.token}
    Probability: {signal.probability:.1%}
    Type: {signal.type.upper()}
    Expected Gain: {signal.expected_magnitude:.1%}
    Duration: {signal.expected_duration} mins
    Risk: {signal.risk_level.upper()}

    Entry: ${signal.entry_price:.6f}
    Targets: {', '.join([f'${t:.6f}' for t in signal.target_prices])}
    Stop Loss: ${signal.stop_loss:.6f}

    Signal Strength: {signal.signal_strength.upper()}
    """
            await self.alerts.send_alert(
                message=message,
                level='high',
                channels=['telegram', 'discord']
            )
            
    async def _detect_price_spike(self, data: List[Dict]) -> Dict[str, Any]:
        """Detect price spike in recent data"""
        if len(data) < 2:
            return {'detected': False}
            
        prices = [d['price'] for d in data]
        initial_price = prices[0]
        current_price = prices[-1]
        
        change = (current_price - initial_price) / initial_price
        
        return {
            'detected': change > self.price_spike_threshold,
            'magnitude': change,
            'initial_price': initial_price,
            'current_price': current_price
        }
        
    async def _detect_volume_spike(self, data: List[Dict]) -> Dict[str, Any]:
        """Detect volume spike in recent data"""
        if len(data) < 20:
            return {'detected': False}
            
        volumes = [d['volume'] for d in data]
        avg_volume = np.mean(volumes[:-5])  # Average excluding last 5 periods
        recent_volume = np.mean(volumes[-5:])  # Last 5 periods
        
        ratio = recent_volume / avg_volume if avg_volume > 0 else 0
        
        return {
            'detected': ratio > self.volume_spike_threshold,
            'magnitude': ratio,
            'average_volume': avg_volume,
            'recent_volume': recent_volume
        }
        
    def _determine_pump_status(
        self,
        price_change: float,
        expected_magnitude: float,
        duration: float,
        expected_duration: float,
        targets_hit: List[int]
    ) -> str:
        """Determine pump completion status"""
        if len(targets_hit) >= 3:
            return 'completed_successful'
        elif len(targets_hit) >= 1:
            return 'in_progress'
        elif price_change < -0.05:
            return 'failed'
        elif duration > expected_duration * 1.5:
            return 'stalled'
        else:
            return 'ongoing'
            
    def _get_default_signal(self, token: str, chain: str) -> PumpSignal:
        """Return default signal when prediction fails"""
        return PumpSignal(
            token=token,
            chain=chain,
            probability=0.0,
            confidence=0.0,
            signal_strength='weak',
            type='unknown',
            expected_magnitude=0.0,
            expected_duration=0,
            risk_level='extreme',
            entry_price=Decimal('0'),
            target_prices=[],
            stop_loss=Decimal('0'),
            indicators={},
            timestamp=datetime.utcnow()
        )

    # ============================================================================
    # PATCH FOR: pump_predictor.py (PumpPredictorAnalysis class)
    # Add these methods to the PumpPredictorAnalysis class
    # ============================================================================

    def analyze_volume_patterns(self, volume_data: List) -> Dict:
        """
        Analyze volume patterns for pump signals
        
        Args:
            volume_data: List of volume data points
            
        Returns:
            Volume pattern analysis
        """
        if not volume_data or len(volume_data) < 2:
            return {'patterns': [], 'signals': {}}
        
        patterns = {
            'increasing': False,
            'spike_detected': False,
            'accumulation': False,
            'distribution': False,
            'average_volume': 0,
            'current_volume': 0,
            'volume_trend': 'neutral'
        }
        
        # Convert to numpy array for analysis
        volumes = np.array([d.get('volume', 0) for d in volume_data if isinstance(d, dict)])
        
        if len(volumes) > 0:
            # Calculate metrics
            patterns['average_volume'] = float(np.mean(volumes))
            patterns['current_volume'] = float(volumes[-1]) if len(volumes) > 0 else 0
            
            # Check for increasing volume
            if len(volumes) >= 5:
                recent = volumes[-5:]
                older = volumes[-10:-5] if len(volumes) >= 10 else volumes[:len(volumes)-5]
                
                if len(older) > 0:
                    recent_avg = np.mean(recent)
                    older_avg = np.mean(older)
                    
                    if recent_avg > older_avg * 1.5:
                        patterns['increasing'] = True
                        patterns['volume_trend'] = 'increasing'
                    elif recent_avg < older_avg * 0.7:
                        patterns['volume_trend'] = 'decreasing'
            
            # Check for volume spike
            if patterns['current_volume'] > patterns['average_volume'] * self.volume_spike_threshold:
                patterns['spike_detected'] = True
            
            # Check for accumulation/distribution
            if len(volume_data) >= 10:
                prices = [d.get('price', 0) for d in volume_data[-10:] if isinstance(d, dict)]
                if prices:
                    price_change = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
                    vol_change = (volumes[-1] - volumes[-10]) / volumes[-10] if volumes[-10] > 0 else 0
                    
                    if vol_change > 0.3 and abs(price_change) < 0.1:
                        patterns['accumulation'] = True
                    elif vol_change > 0.3 and price_change < -0.1:
                        patterns['distribution'] = True
        
        return patterns

    def detect_accumulation_phase_sync(self, price_data: List) -> bool:
        """
        Synchronous wrapper for detect_accumulation_phase
        
        Args:
            price_data: List of price data points
            
        Returns:
            True if accumulation phase detected
        """
        # Convert list to proper format if needed
        data = {'price_data': price_data}
        
        # Run async function synchronously
        import asyncio
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(self._detect_accumulation_phase(data))
        
        return result.get('detected', False)

    # ============================================================================
    # FIXES FOR: pump_predictor.py
    # ============================================================================

    # Add missing synchronous method: detect_accumulation_phase
    def detect_accumulation_phase(self, price_data: List) -> bool:
        """
        Detect if token is in accumulation phase (synchronous version)
        
        Args:
            price_data: List of price data points
            
        Returns:
            True if accumulation phase detected
        """
        if not price_data or len(price_data) < 20:
            return False
        
        try:
            import numpy as np
            
            # Extract prices from data
            if isinstance(price_data[0], dict):
                prices = [d.get('price', 0) for d in price_data]
            else:
                prices = price_data
            
            prices = np.array(prices)
            
            # Calculate metrics
            price_range = prices.max() - prices.min()
            avg_price = prices.mean()
            range_percentage = (price_range / avg_price) * 100 if avg_price > 0 else 100
            
            # Calculate volatility
            returns = np.diff(prices) / prices[:-1]
            volatility = np.std(returns)
            
            # Check for accumulation characteristics:
            # 1. Low volatility (< 2%)
            # 2. Tight price range (< 5%)
            # 3. Price holding support (not trending down)
            price_trend = (prices[-1] - prices[0]) / prices[0] if prices[0] > 0 else 0
            
            is_accumulating = (
                volatility < 0.02 and
                range_percentage < 5 and
                price_trend >= -0.05  # Not declining more than 5%
            )
            
            return is_accumulating
            
        except Exception as e:
            logger.error(f"Error in detect_accumulation_phase: {e}")
            return False

    def calculate_pump_probability(self, indicators: Dict) -> float:
        """
        Calculate pump probability from indicators
        
        Args:
            indicators: Dictionary of pump indicators
            
        Returns:
            Pump probability (0-1)
        """
        probability = 0.5  # Base probability
        
        # Volume indicators
        if indicators.get('volume_increasing'):
            probability += 0.15
        if indicators.get('volume_spike'):
            probability += 0.1
        
        # Price indicators
        if indicators.get('price_breakout'):
            probability += 0.15
        if indicators.get('accumulation_detected'):
            probability += 0.1
        
        # Technical indicators
        if indicators.get('rsi', 50) > 60:
            probability += 0.05
        if indicators.get('macd_bullish'):
            probability += 0.05
        
        # Social indicators
        if indicators.get('social_momentum', 0) > 0.7:
            probability += 0.1
        
        # Whale activity
        if indicators.get('whale_accumulation'):
            probability += 0.15
        
        # Cap at 0-1 range
        return min(1.0, max(0.0, probability))
