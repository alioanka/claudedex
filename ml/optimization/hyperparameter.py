"""
Hyperparameter Optimization Module
Uses Bayesian optimization for strategy parameter tuning
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from loguru import logger
import json


@dataclass
class HyperparameterSpace:
    """Defines the search space for hyperparameters"""
    name: str
    min_value: float
    max_value: float
    param_type: str = "float"  # "float", "int", "categorical"
    log_scale: bool = False


class HyperparameterOptimizer:
    """
    Optimizes strategy hyperparameters using Bayesian optimization
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize hyperparameter optimizer
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        self.optimization_history = []
        self.best_params = None
        self.best_score = float('-inf')
        
        # Define default parameter spaces
        self.param_spaces = self._define_parameter_spaces()
        
        logger.info("HyperparameterOptimizer initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            "optimization_method": "bayesian",  # "bayesian", "grid", "random"
            "n_iterations": 50,
            "n_random_starts": 10,
            "acquisition_function": "expected_improvement",
            "exploration_weight": 0.1,
            "convergence_threshold": 0.001,
            "max_history_size": 1000
        }
    
    def _define_parameter_spaces(self) -> List[HyperparameterSpace]:
        """Define hyperparameter search spaces"""
        return [
            # Risk management parameters
            HyperparameterSpace("max_position_size", 0.01, 0.2, "float"),
            HyperparameterSpace("stop_loss_percentage", 0.05, 0.3, "float"),
            HyperparameterSpace("take_profit_percentage", 0.1, 2.0, "float"),
            
            # Entry parameters
            HyperparameterSpace("min_liquidity", 1000, 100000, "float", log_scale=True),
            HyperparameterSpace("min_volume_24h", 5000, 1000000, "float", log_scale=True),
            HyperparameterSpace("slippage_tolerance", 0.001, 0.05, "float"),
            
            # ML confidence thresholds
            HyperparameterSpace("min_ml_confidence", 0.5, 0.95, "float"),
            HyperparameterSpace("min_pump_probability", 0.3, 0.9, "float"),
            HyperparameterSpace("max_rug_probability", 0.05, 0.3, "float"),
            
            # Timing parameters
            HyperparameterSpace("entry_delay", 0, 30, "int"),  # seconds
            HyperparameterSpace("max_hold_time", 300, 7200, "int"),  # seconds
            
            # Momentum indicators
            HyperparameterSpace("rsi_oversold", 20, 40, "int"),
            HyperparameterSpace("rsi_overbought", 60, 80, "int"),
            
            # Position sizing
            HyperparameterSpace("kelly_fraction", 0.1, 0.5, "float"),
            HyperparameterSpace("risk_per_trade", 0.01, 0.05, "float"),
        ]
    
    async def optimize(
        self, 
        performance_data: Dict[str, Any], 
        current_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters based on performance data
        
        Args:
            performance_data: Recent performance metrics including:
                - trades: List of trade results
                - win_rate: Overall win rate
                - avg_profit: Average profit per trade
                - sharpe_ratio: Risk-adjusted returns
                - max_drawdown: Maximum drawdown experienced
            current_params: Current strategy parameters
            
        Returns:
            Optimized parameters dictionary
        """
        try:
            logger.info("Starting hyperparameter optimization")
            
            # Calculate objective score from performance
            objective_score = self._calculate_objective(performance_data)
            
            # Store in history
            self.optimization_history.append({
                'params': current_params.copy(),
                'performance': performance_data.copy(),
                'score': objective_score,
                'iteration': len(self.optimization_history)
            })
            
            # Update best if improved
            if objective_score > self.best_score:
                self.best_score = objective_score
                self.best_params = current_params.copy()
                logger.info(f"New best score: {self.best_score:.4f}")
            
            # Check if we have enough history for optimization
            if len(self.optimization_history) < self.config["n_random_starts"]:
                # Random exploration phase
                new_params = self._random_sample_params()
                logger.debug(f"Random exploration: iteration {len(self.optimization_history)}")
            else:
                # Bayesian optimization phase
                if self.config["optimization_method"] == "bayesian":
                    new_params = self._bayesian_suggest_params()
                elif self.config["optimization_method"] == "grid":
                    new_params = self._grid_search_next()
                elif self.config["optimization_method"] == "random":
                    new_params = self._random_sample_params()
                else:
                    logger.warning(f"Unknown optimization method, using current params")
                    new_params = current_params
            
            # Maintain history size limit
            if len(self.optimization_history) > self.config["max_history_size"]:
                self.optimization_history = self.optimization_history[-self.config["max_history_size"]:]
            
            logger.info(f"Optimization iteration {len(self.optimization_history)} complete")
            return new_params
            
        except Exception as e:
            logger.error(f"Hyperparameter optimization failed: {e}")
            return current_params
    
    def _calculate_objective(self, performance: Dict[str, Any]) -> float:
        """
        Calculate optimization objective score
        
        Combines multiple metrics into single score:
        - Profit factor
        - Win rate
        - Sharpe ratio
        - Max drawdown (penalty)
        """
        try:
            # Extract metrics with defaults
            win_rate = performance.get('win_rate', 0.0)
            avg_profit = performance.get('avg_profit', 0.0)
            sharpe = performance.get('sharpe_ratio', 0.0)
            max_dd = abs(performance.get('max_drawdown', 0.0))
            total_trades = performance.get('total_trades', 0)
            
            # Minimum trades requirement
            if total_trades < 10:
                return -1.0  # Not enough data
            
            # Weighted score calculation
            score = (
                win_rate * 0.3 +
                min(avg_profit / 100, 1.0) * 0.3 +  # Normalize profit
                min(sharpe / 3.0, 1.0) * 0.3 +  # Normalize Sharpe
                max(0, 1 - max_dd) * 0.1  # Drawdown penalty
            )
            
            return score
            
        except Exception as e:
            logger.error(f"Error calculating objective: {e}")
            return 0.0
    
    def _random_sample_params(self) -> Dict[str, Any]:
        """Sample random parameters from search space"""
        params = {}
        
        for space in self.param_spaces:
            if space.param_type == "float":
                if space.log_scale:
                    # Log-uniform sampling
                    log_min = np.log10(space.min_value)
                    log_max = np.log10(space.max_value)
                    value = 10 ** np.random.uniform(log_min, log_max)
                else:
                    value = np.random.uniform(space.min_value, space.max_value)
            elif space.param_type == "int":
                value = int(np.random.uniform(space.min_value, space.max_value + 1))
            else:
                # For categorical, would need to define categories
                value = space.min_value
            
            params[space.name] = value
        
        return params
    
    def _bayesian_suggest_params(self) -> Dict[str, Any]:
        """
        Suggest next parameters using Bayesian optimization
        Uses Gaussian Process surrogate model
        """
        try:
            # Extract X (parameters) and y (scores) from history
            X = []
            y = []
            
            for entry in self.optimization_history:
                param_vector = [
                    entry['params'].get(space.name, space.min_value)
                    for space in self.param_spaces
                ]
                X.append(param_vector)
                y.append(entry['score'])
            
            X = np.array(X)
            y = np.array(y)
            
            # Fit Gaussian Process (simplified)
            mean_pred, std_pred = self._fit_gp_surrogate(X, y)
            
            # Generate candidate points
            n_candidates = 100
            candidates = []
            for _ in range(n_candidates):
                candidates.append(self._random_sample_params())
            
            # Convert to array
            candidate_vectors = []
            for cand in candidates:
                vec = [cand.get(space.name, space.min_value) for space in self.param_spaces]
                candidate_vectors.append(vec)
            candidate_vectors = np.array(candidate_vectors)
            
            # Calculate acquisition function
            best_y = np.max(y)
            acq_values = self._expected_improvement(
                candidate_vectors, 
                mean_pred, 
                std_pred, 
                best_y
            )
            
            # Select best candidate
            best_idx = np.argmax(acq_values)
            return candidates[best_idx]
            
        except Exception as e:
            logger.error(f"Bayesian optimization failed: {e}")
            return self._random_sample_params()
    
    def _fit_gp_surrogate(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fit Gaussian Process surrogate (simplified)
        Returns mean and std predictions
        """
        # Simplified GP - just return basic statistics
        # In production, use sklearn.gaussian_process.GaussianProcessRegressor
        mean = np.mean(y)
        std = np.std(y)
        
        return np.full(len(X), mean), np.full(len(X), std)
    
    def _expected_improvement(
        self,
        X: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
        best_y: float
    ) -> np.ndarray:
        """Calculate Expected Improvement acquisition function"""
        improvement = mean - best_y - self.config["exploration_weight"]
        
        # Avoid division by zero
        std = np.maximum(std, 1e-10)
        
        # Simplified EI calculation
        ei = improvement * (improvement > 0)
        
        return ei
    
    def _grid_search_next(self) -> Dict[str, Any]:
        """Get next parameters from grid search"""
        # Simplified grid search - just sample uniformly
        # In production, would implement proper grid
        return self._random_sample_params()
    
    def get_best_parameters(self) -> Optional[Dict[str, Any]]:
        """Get the best parameters found so far"""
        return self.best_params
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization progress"""
        if not self.optimization_history:
            return {"status": "no_history"}
        
        scores = [entry['score'] for entry in self.optimization_history]
        
        return {
            "iterations": len(self.optimization_history),
            "best_score": self.best_score,
            "best_params": self.best_params,
            "current_score": scores[-1] if scores else None,
            "mean_score": np.mean(scores),
            "score_std": np.std(scores),
            "improvement": scores[-1] - scores[0] if len(scores) > 1 else 0
        }