"""
Reinforcement Learning Optimization Module
Uses RL for adaptive strategy optimization
"""

from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass
from collections import deque
from loguru import logger


@dataclass
class State:
    """RL State representation"""
    market_volatility: float
    portfolio_exposure: float
    recent_win_rate: float
    current_drawdown: float
    time_in_market: int


@dataclass
class Action:
    """RL Action representation"""
    position_size_multiplier: float  # 0.5 to 2.0
    risk_adjustment: float  # 0.5 to 1.5
    entry_aggressiveness: float  # 0.0 to 1.0


@dataclass
class Experience:
    """Experience tuple for replay buffer"""
    state: State
    action: Action
    reward: float
    next_state: State
    done: bool


class RLOptimizer:
    """
    Reinforcement learning-based strategy optimizer
    Uses simplified Q-learning approach
    """
    
    def __init__(self, config: Optional[Dict] = None):
        """
        Initialize RL optimizer
        
        Args:
            config: Optional configuration dictionary
        """
        self.config = config or self._default_config()
        
        # Policy and value networks (simplified)
        self.q_table = {}  # State-action value table
        self.policy = self._initialize_policy()
        
        # Training components
        self.replay_buffer = deque(maxlen=self.config["replay_buffer_size"])
        self.training_history = []
        self.episode_rewards = []
        
        # Exploration parameters
        self.epsilon = self.config["initial_epsilon"]
        self.epsilon_min = self.config["epsilon_min"]
        self.epsilon_decay = self.config["epsilon_decay"]
        
        logger.info("RLOptimizer initialized")
    
    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            # Learning parameters
            "learning_rate": 0.001,
            "discount_factor": 0.99,
            "initial_epsilon": 1.0,
            "epsilon_min": 0.1,
            "epsilon_decay": 0.995,
            
            # Experience replay
            "replay_buffer_size": 10000,
            "batch_size": 32,
            "min_replay_size": 100,
            
            # Training
            "update_frequency": 10,  # episodes
            "target_update_frequency": 50,  # episodes
            
            # Action space
            "position_size_actions": [0.5, 0.75, 1.0, 1.25, 1.5],
            "risk_actions": [0.7, 0.85, 1.0, 1.15, 1.3],
            "aggressiveness_actions": [0.0, 0.25, 0.5, 0.75, 1.0],
            
            # Reward shaping
            "profit_reward_weight": 1.0,
            "risk_penalty_weight": 0.3,
            "drawdown_penalty": 0.5,
        }
    
    def _initialize_policy(self) -> Dict:
        """Initialize default policy"""
        return {
            "position_size_multiplier": 1.0,
            "risk_adjustment": 1.0,
            "entry_aggressiveness": 0.5
        }
    
    async def update_policy(self, performance_data: Dict[str, Any]) -> None:
        """
        Update RL policy based on performance data
        
        Args:
            performance_data: Recent performance metrics including:
                - trades: List of recent trades
                - current_state: Current market/portfolio state
                - reward: Reward signal from last action
        """
        try:
            logger.debug("Updating RL policy")
            
            # Extract state and reward
            current_state = self._extract_state(performance_data)
            reward = self._calculate_reward(performance_data)
            
            # Check if we have previous state-action pair
            if self.training_history:
                last_entry = self.training_history[-1]
                last_state = last_entry['state']
                last_action = last_entry['action']
                
                # Create experience
                experience = Experience(
                    state=last_state,
                    action=last_action,
                    reward=reward,
                    next_state=current_state,
                    done=False
                )
                
                # Add to replay buffer
                self.replay_buffer.append(experience)
                
                # Train if we have enough experiences
                if len(self.replay_buffer) >= self.config["min_replay_size"]:
                    await self._train_step()
            
            # Select next action using epsilon-greedy
            next_action = self._select_action(current_state)
            
            # Update policy with new action
            self.policy = {
                "position_size_multiplier": next_action.position_size_multiplier,
                "risk_adjustment": next_action.risk_adjustment,
                "entry_aggressiveness": next_action.entry_aggressiveness
            }
            
            # Store in history
            self.training_history.append({
                'state': current_state,
                'action': next_action,
                'reward': reward,
                'episode': len(self.training_history)
            })
            
            # Decay epsilon
            self.epsilon = max(
                self.epsilon_min,
                self.epsilon * self.config["epsilon_decay"]
            )
            
            # Cleanup old history
            if len(self.training_history) > 1000:
                self.training_history = self.training_history[-1000:]
            
        except Exception as e:
            logger.error(f"RL policy update failed: {e}")
    
    def _extract_state(self, performance_data: Dict[str, Any]) -> State:
        """Extract state from performance data"""
        return State(
            market_volatility=performance_data.get('market_volatility', 0.5),
            portfolio_exposure=performance_data.get('portfolio_exposure', 0.0),
            recent_win_rate=performance_data.get('win_rate', 0.5),
            current_drawdown=abs(performance_data.get('current_drawdown', 0.0)),
            time_in_market=performance_data.get('time_in_market', 0)
        )
    
    def _calculate_reward(self, performance_data: Dict[str, Any]) -> float:
        """
        Calculate reward signal from performance
        
        Reward function considers:
        - Profit/loss
        - Risk-adjusted returns
        - Drawdown penalty
        """
        # Extract metrics
        pnl = performance_data.get('last_trade_pnl', 0.0)
        risk_taken = performance_data.get('risk_taken', 1.0)
        drawdown = abs(performance_data.get('current_drawdown', 0.0))
        
        # Calculate reward components
        profit_reward = pnl * self.config["profit_reward_weight"]
        risk_penalty = -abs(risk_taken - 1.0) * self.config["risk_penalty_weight"]
        drawdown_penalty = -drawdown * self.config["drawdown_penalty"]
        
        total_reward = profit_reward + risk_penalty + drawdown_penalty
        
        return total_reward
    
    def _select_action(self, state: State) -> Action:
        """
        Select action using epsilon-greedy policy
        
        Args:
            state: Current state
            
        Returns:
            Selected action
        """
        # Epsilon-greedy: explore vs exploit
        if np.random.random() < self.epsilon:
            # Explore: random action
            return self._random_action()
        else:
            # Exploit: best known action
            return self._greedy_action(state)
    
    def _random_action(self) -> Action:
        """Sample random action from action space"""
        return Action(
            position_size_multiplier=np.random.choice(
                self.config["position_size_actions"]
            ),
            risk_adjustment=np.random.choice(
                self.config["risk_actions"]
            ),
            entry_aggressiveness=np.random.choice(
                self.config["aggressiveness_actions"]
            )
        )
    
    def _greedy_action(self, state: State) -> Action:
        """Select best action based on Q-values"""
        # Discretize state for Q-table lookup
        state_key = self._discretize_state(state)
        
        # If state not seen, return default action
        if state_key not in self.q_table:
            return Action(1.0, 1.0, 0.5)
        
        # Find action with highest Q-value
        q_values = self.q_table[state_key]
        best_action_key = max(q_values, key=q_values.get)
        
        return self._action_from_key(best_action_key)
    
    def _discretize_state(self, state: State) -> str:
        """Convert continuous state to discrete key"""
        # Bin continuous values
        vol_bin = int(state.market_volatility * 10)
        exp_bin = int(state.portfolio_exposure * 10)
        win_bin = int(state.recent_win_rate * 10)
        dd_bin = int(state.current_drawdown * 10)
        
        return f"{vol_bin}_{exp_bin}_{win_bin}_{dd_bin}"
    
    def _action_to_key(self, action: Action) -> str:
        """Convert action to string key"""
        return f"{action.position_size_multiplier}_{action.risk_adjustment}_{action.entry_aggressiveness}"
    
    def _action_from_key(self, key: str) -> Action:
        """Convert string key to action"""
        parts = key.split('_')
        return Action(
            position_size_multiplier=float(parts[0]),
            risk_adjustment=float(parts[1]),
            entry_aggressiveness=float(parts[2])
        )
    
    async def _train_step(self):
        """Perform one training step on mini-batch"""
        # Sample mini-batch from replay buffer
        batch_size = min(self.config["batch_size"], len(self.replay_buffer))
        indices = np.random.choice(len(self.replay_buffer), batch_size, replace=False)
        batch = [self.replay_buffer[i] for i in indices]
        
        # Update Q-values using Bellman equation
        for experience in batch:
            state_key = self._discretize_state(experience.state)
            action_key = self._action_to_key(experience.action)
            next_state_key = self._discretize_state(experience.next_state)
            
            # Initialize Q-table entries if needed
            if state_key not in self.q_table:
                self.q_table[state_key] = {}
            if action_key not in self.q_table[state_key]:
                self.q_table[state_key][action_key] = 0.0
            
            # Get max Q-value for next state
            if next_state_key in self.q_table and self.q_table[next_state_key]:
                max_next_q = max(self.q_table[next_state_key].values())
            else:
                max_next_q = 0.0
            
            # Q-learning update
            current_q = self.q_table[state_key][action_key]
            target_q = experience.reward + self.config["discount_factor"] * max_next_q
            
            # Update with learning rate
            self.q_table[state_key][action_key] = (
                current_q + self.config["learning_rate"] * (target_q - current_q)
            )
    
    def get_current_policy(self) -> Dict[str, float]:
        """Get current policy parameters"""
        return self.policy
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics"""
        if not self.training_history:
            return {"status": "no_training"}
        
        recent_rewards = [entry['reward'] for entry in self.training_history[-100:]]
        
        return {
            "episodes": len(self.training_history),
            "epsilon": self.epsilon,
            "q_table_size": len(self.q_table),
            "replay_buffer_size": len(self.replay_buffer),
            "mean_recent_reward": np.mean(recent_rewards) if recent_rewards else 0,
            "current_policy": self.policy
        }