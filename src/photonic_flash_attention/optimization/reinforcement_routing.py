"""
Reinforcement Learning-based Adaptive Routing for Photonic-Electronic Systems.

This module implements advanced RL algorithms for optimal device selection
and load balancing in hybrid photonic-electronic computing systems.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import threading
from typing import Dict, List, Tuple, Any, Optional, NamedTuple
from collections import deque, defaultdict
from dataclasses import dataclass, field
import time
import logging

from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicTimeoutError

logger = get_logger(__name__)


@dataclass 
class RoutingState:
    """State representation for RL routing decisions."""
    batch_size: int
    seq_length: int
    embed_dim: int
    num_heads: int
    current_load_gpu: float
    current_load_photonic: float
    recent_gpu_latency: float
    recent_photonic_latency: float
    temperature_photonic: float
    energy_budget_remaining: float
    time_of_day: float  # Normalized 0-1
    
    def to_tensor(self) -> torch.Tensor:
        """Convert state to tensor for neural network input."""
        return torch.tensor([
            self.batch_size / 32.0,  # Normalized
            self.seq_length / 8192.0,
            self.embed_dim / 1024.0,
            self.num_heads / 32.0,
            self.current_load_gpu,
            self.current_load_photonic,
            self.recent_gpu_latency / 100.0,  # Normalize by expected max latency
            self.recent_photonic_latency / 100.0,
            self.temperature_photonic / 100.0,  # Normalize by max safe temp
            self.energy_budget_remaining,
            self.time_of_day
        ], dtype=torch.float32)


@dataclass
class RoutingAction:
    """Action representation for RL routing decisions."""
    device: str  # 'gpu' or 'photonic'
    load_balance_factor: float  # 0.0-1.0 for splitting workload
    priority: int  # 0-2 for scheduling priority
    
    @classmethod
    def from_index(cls, action_index: int) -> 'RoutingAction':
        """Convert action index to action object."""
        # 8 total actions: 2 devices × 2 load factors × 2 priorities
        device = 'photonic' if action_index >= 4 else 'gpu'
        load_factor = 0.5 if (action_index % 4) >= 2 else 1.0
        priority = 2 if (action_index % 2) == 1 else 1
        return cls(device, load_factor, priority)


class ExperienceReplay:
    """Experience replay buffer for RL training."""
    
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        
    def push(self, state: torch.Tensor, action: int, reward: float, 
             next_state: torch.Tensor, done: bool, priority: float = 1.0):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(priority)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample batch with prioritized sampling."""
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        
        # Prioritized sampling
        priorities = np.array(self.priorities)
        probabilities = priorities / priorities.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, 
                                 replace=False, p=probabilities)
        
        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            torch.stack(states),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.stack(next_states),
            torch.tensor(dones, dtype=torch.bool)
        )
    
    def __len__(self):
        return len(self.buffer)


class DQNNetwork(nn.Module):
    """Deep Q-Network for routing decisions."""
    
    def __init__(self, state_dim: int = 11, action_dim: int = 8, hidden_dim: int = 128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(0.1),
            
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # Dueling DQN architecture
        self.value_head = nn.Linear(hidden_dim // 2, 1)
        self.advantage_head = nn.Linear(hidden_dim // 2, action_dim)
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        features = self.network[:-1](state)  # All layers except final
        
        # Dueling DQN
        value = self.value_head(features)
        advantage = self.advantage_head(features)
        
        # Combine value and advantage
        q_values = value + advantage - advantage.mean(dim=-1, keepdim=True)
        
        return q_values


class ReinforcementRoutingAgent:
    """
    Advanced RL agent for optimal routing decisions in hybrid systems.
    
    Uses Double DQN with prioritized experience replay and learned
    reward shaping for efficient exploration and exploitation.
    """
    
    def __init__(
        self,
        state_dim: int = 11,
        action_dim: int = 8,
        learning_rate: float = 1e-4,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        gamma: float = 0.99,
        target_update_freq: int = 1000,
        batch_size: int = 32,
        memory_size: int = 10000
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.batch_size = batch_size
        
        # Neural networks
        self.q_network = DQNNetwork(state_dim, action_dim)
        self.target_network = DQNNetwork(state_dim, action_dim)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Experience replay
        self.memory = ExperienceReplay(memory_size)
        
        # Training tracking
        self.steps = 0
        self.episode_rewards = deque(maxlen=100)
        self.losses = deque(maxlen=1000)
        
        # Performance tracking
        self.device_performance_history = defaultdict(list)
        self.reward_shaping_params = {
            'latency_weight': 0.4,
            'energy_weight': 0.3,
            'throughput_weight': 0.2,
            'reliability_weight': 0.1
        }
        
        # Thread safety
        self._lock = threading.Lock()
        
        # Copy weights to target network
        self.update_target_network()
        
        logger.info("RL routing agent initialized")
    
    def get_action(self, state: RoutingState, training: bool = True) -> Tuple[RoutingAction, float]:
        """
        Select action using epsilon-greedy policy with learned Q-values.
        
        Returns:
            action: Selected routing action
            q_value: Q-value of selected action
        """
        with self._lock:
            state_tensor = state.to_tensor().unsqueeze(0)
            
            # Epsilon-greedy action selection
            if training and random.random() < self.epsilon:
                # Random exploration
                action_index = random.randint(0, self.action_dim - 1)
                q_value = 0.0
            else:
                # Exploitation using Q-network
                with torch.no_grad():
                    q_values = self.q_network(state_tensor)
                    action_index = q_values.argmax().item()
                    q_value = q_values.max().item()
            
            # Decay epsilon
            if training and self.epsilon > self.epsilon_end:
                self.epsilon *= self.epsilon_decay
            
            action = RoutingAction.from_index(action_index)
            
            return action, q_value
    
    def compute_reward(self, action: RoutingAction, performance_metrics: Dict[str, float],
                      system_state: Dict[str, float]) -> float:
        """
        Compute shaped reward based on multiple objectives.
        
        Reward considers:
        - Latency (lower is better)
        - Energy efficiency (lower consumption is better)  
        - Throughput (higher is better)
        - System reliability/stability
        """
        weights = self.reward_shaping_params
        
        # Latency component (normalized, inverted)
        latency_ms = performance_metrics.get('latency_ms', 100.0)
        latency_reward = -np.log(1 + latency_ms / 10.0) * weights['latency_weight']
        
        # Energy component (normalized, inverted)
        energy_mj = performance_metrics.get('energy_mj', 10.0)
        energy_reward = -np.log(1 + energy_mj / 5.0) * weights['energy_weight']
        
        # Throughput component (normalized)
        throughput = performance_metrics.get('throughput_tokens_per_sec', 1000.0)
        throughput_reward = np.log(1 + throughput / 1000.0) * weights['throughput_weight']
        
        # Reliability component
        success_rate = performance_metrics.get('success_rate', 1.0)
        temp_penalty = max(0, system_state.get('temperature_c', 25.0) - 70.0) / 30.0
        reliability_reward = (success_rate - temp_penalty) * weights['reliability_weight']
        
        # Load balancing bonus
        load_balance_bonus = 0.0
        if action.load_balance_factor < 1.0:
            gpu_load = system_state.get('gpu_load', 0.5)
            photonic_load = system_state.get('photonic_load', 0.5)
            load_imbalance = abs(gpu_load - photonic_load)
            load_balance_bonus = (1.0 - load_imbalance) * 0.1
        
        total_reward = (
            latency_reward + 
            energy_reward + 
            throughput_reward + 
            reliability_reward + 
            load_balance_bonus
        )
        
        return total_reward
    
    def store_experience(self, state: RoutingState, action: RoutingAction,
                        reward: float, next_state: RoutingState, done: bool):
        """Store experience in replay buffer."""
        state_tensor = state.to_tensor()
        next_state_tensor = next_state.to_tensor()
        action_index = self._action_to_index(action)
        
        # Compute TD error for prioritization
        with torch.no_grad():
            current_q = self.q_network(state_tensor.unsqueeze(0))[0, action_index]
            next_q = self.target_network(next_state_tensor.unsqueeze(0)).max()
            target_q = reward + self.gamma * next_q * (not done)
            td_error = abs(current_q - target_q).item()
        
        self.memory.push(state_tensor, action_index, reward, 
                        next_state_tensor, done, td_error + 1e-6)
    
    def train_step(self) -> Optional[float]:
        """Perform one training step."""
        if len(self.memory) < self.batch_size:
            return None
        
        with self._lock:
            # Sample batch
            states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
            
            # Current Q-values
            current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
            
            # Next Q-values (Double DQN)
            with torch.no_grad():
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
                target_q_values = rewards.unsqueeze(1) + self.gamma * next_q_values * (~dones).unsqueeze(1)
            
            # Compute loss
            loss = nn.MSELoss()(current_q_values, target_q_values)
            
            # Optimize
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), 1.0)
            self.optimizer.step()
            
            # Update target network
            self.steps += 1
            if self.steps % self.target_update_freq == 0:
                self.update_target_network()
            
            # Track loss
            self.losses.append(loss.item())
            
            return loss.item()
    
    def update_target_network(self):
        """Copy main network weights to target network."""
        self.target_network.load_state_dict(self.q_network.state_dict())
    
    def _action_to_index(self, action: RoutingAction) -> int:
        """Convert action object to index."""
        device_idx = 4 if action.device == 'photonic' else 0
        load_idx = 2 if action.load_balance_factor < 1.0 else 0
        priority_idx = 1 if action.priority == 2 else 0
        return device_idx + load_idx + priority_idx
    
    def get_stats(self) -> Dict[str, Any]:
        """Get training and performance statistics."""
        return {
            'epsilon': self.epsilon,
            'steps': self.steps,
            'memory_size': len(self.memory),
            'avg_loss': np.mean(self.losses) if self.losses else 0.0,
            'avg_episode_reward': np.mean(self.episode_rewards) if self.episode_rewards else 0.0,
            'q_network_params': sum(p.numel() for p in self.q_network.parameters()),
        }
    
    def save_model(self, path: str):
        """Save trained model."""
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps': self.steps,
            'reward_shaping_params': self.reward_shaping_params
        }, path)
        
        logger.info(f"RL model saved to {path}")
    
    def load_model(self, path: str):
        """Load trained model."""
        checkpoint = torch.load(path)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps = checkpoint['steps']
        self.reward_shaping_params = checkpoint['reward_shaping_params']
        
        logger.info(f"RL model loaded from {path}")


class AdaptiveRoutingEnvironment:
    """
    Environment for training RL routing agent.
    
    Simulates photonic-electronic system dynamics and provides
    rewards for routing decisions.
    """
    
    def __init__(self):
        self.reset()
        self.performance_history = defaultdict(list)
        
    def reset(self) -> RoutingState:
        """Reset environment to initial state."""
        self.current_state = RoutingState(
            batch_size=random.randint(1, 16),
            seq_length=random.choice([128, 256, 512, 1024, 2048]),
            embed_dim=random.choice([512, 768, 1024]),
            num_heads=random.choice([8, 12, 16]),
            current_load_gpu=random.uniform(0.0, 1.0),
            current_load_photonic=random.uniform(0.0, 1.0),
            recent_gpu_latency=random.uniform(10.0, 100.0),
            recent_photonic_latency=random.uniform(5.0, 50.0),
            temperature_photonic=random.uniform(25.0, 85.0),
            energy_budget_remaining=random.uniform(0.2, 1.0),
            time_of_day=random.uniform(0.0, 1.0)
        )
        return self.current_state
    
    def step(self, action: RoutingAction) -> Tuple[RoutingState, float, bool, Dict[str, Any]]:
        """
        Execute action and return next state, reward, done, info.
        
        Simulates system dynamics based on action choice.
        """
        # Simulate performance based on action
        performance_metrics = self._simulate_performance(action)
        
        # Compute reward
        system_state = {
            'temperature_c': self.current_state.temperature_photonic,
            'gpu_load': self.current_state.current_load_gpu,
            'photonic_load': self.current_state.current_load_photonic
        }
        
        # Update system state based on action
        next_state = self._update_state(action, performance_metrics)
        
        # Episode termination conditions
        done = (
            next_state.temperature_photonic > 90.0 or  # Thermal shutdown
            next_state.energy_budget_remaining <= 0.0 or  # Energy exhausted
            performance_metrics['success_rate'] < 0.5  # Too many failures
        )
        
        info = {
            'performance_metrics': performance_metrics,
            'system_state': system_state
        }
        
        self.current_state = next_state
        return next_state, 0.0, done, info  # Reward computed separately
    
    def _simulate_performance(self, action: RoutingAction) -> Dict[str, float]:
        """Simulate performance metrics for given action."""
        if action.device == 'photonic':
            # Photonic performance model
            base_latency = max(5.0, self.current_state.recent_photonic_latency * 0.9)
            base_energy = 2.0  # Lower energy for photonic
            base_throughput = 1500.0
            
            # Temperature affects photonic performance
            temp_factor = 1.0 + max(0, self.current_state.temperature_photonic - 70.0) / 100.0
            base_latency *= temp_factor
            
        else:
            # GPU performance model
            base_latency = max(10.0, self.current_state.recent_gpu_latency * 0.9)
            base_energy = 8.0  # Higher energy for GPU
            base_throughput = 1200.0
        
        # Load balancing affects performance
        if action.load_balance_factor < 1.0:
            base_latency *= 1.2  # Overhead for splitting
            base_throughput *= 0.9
        
        # Add stochasticity
        latency_noise = random.uniform(0.8, 1.2)
        energy_noise = random.uniform(0.9, 1.1)
        throughput_noise = random.uniform(0.9, 1.1)
        
        # Success rate depends on system stability
        success_rate = min(1.0, max(0.5, 
            1.0 - abs(self.current_state.current_load_gpu - self.current_state.current_load_photonic) * 0.3
        ))
        
        return {
            'latency_ms': base_latency * latency_noise,
            'energy_mj': base_energy * energy_noise,
            'throughput_tokens_per_sec': base_throughput * throughput_noise,
            'success_rate': success_rate
        }
    
    def _update_state(self, action: RoutingAction, 
                     performance_metrics: Dict[str, float]) -> RoutingState:
        """Update system state based on action and results."""
        # Update loads
        if action.device == 'photonic':
            new_photonic_load = min(1.0, self.current_state.current_load_photonic + 0.1)
            new_gpu_load = max(0.0, self.current_state.current_load_gpu - 0.05)
            new_photonic_latency = performance_metrics['latency_ms']
            new_gpu_latency = self.current_state.recent_gpu_latency
        else:
            new_gpu_load = min(1.0, self.current_state.current_load_gpu + 0.1)
            new_photonic_load = max(0.0, self.current_state.current_load_photonic - 0.05)
            new_gpu_latency = performance_metrics['latency_ms']
            new_photonic_latency = self.current_state.recent_photonic_latency
        
        # Update temperature (photonic devices heat up under load)
        temp_change = 0.0
        if action.device == 'photonic':
            temp_change = 2.0 * new_photonic_load
        temp_dissipation = -1.0  # Natural cooling
        new_temperature = max(25.0, min(100.0, 
            self.current_state.temperature_photonic + temp_change + temp_dissipation
        ))
        
        # Update energy budget
        energy_consumed = performance_metrics['energy_mj'] / 1000.0  # Convert to J
        new_energy_budget = max(0.0, 
            self.current_state.energy_budget_remaining - energy_consumed / 100.0
        )
        
        return RoutingState(
            batch_size=self.current_state.batch_size,
            seq_length=self.current_state.seq_length,
            embed_dim=self.current_state.embed_dim,
            num_heads=self.current_state.num_heads,
            current_load_gpu=new_gpu_load,
            current_load_photonic=new_photonic_load,
            recent_gpu_latency=new_gpu_latency,
            recent_photonic_latency=new_photonic_latency,
            temperature_photonic=new_temperature,
            energy_budget_remaining=new_energy_budget,
            time_of_day=(self.current_state.time_of_day + 0.01) % 1.0
        )


def train_rl_agent(episodes: int = 1000, save_path: str = "rl_routing_model.pt") -> ReinforcementRoutingAgent:
    """
    Train RL agent for adaptive routing.
    
    Args:
        episodes: Number of training episodes
        save_path: Path to save trained model
        
    Returns:
        Trained RL agent
    """
    agent = ReinforcementRoutingAgent()
    env = AdaptiveRoutingEnvironment()
    
    logger.info(f"Starting RL training for {episodes} episodes")
    
    episode_rewards = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0.0
        done = False
        step_count = 0
        max_steps = 100
        
        while not done and step_count < max_steps:
            # Get action
            action, q_value = agent.get_action(state, training=True)
            
            # Execute action
            next_state, _, done, info = env.step(action)
            
            # Compute reward
            reward = agent.compute_reward(action, 
                                        info['performance_metrics'],
                                        info['system_state'])
            
            # Store experience
            agent.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = agent.train_step()
            
            episode_reward += reward
            state = next_state
            step_count += 1
        
        episode_rewards.append(episode_reward)
        agent.episode_rewards.append(episode_reward)
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            stats = agent.get_stats()
            logger.info(f"Episode {episode}: avg_reward={avg_reward:.3f}, "
                       f"epsilon={stats['epsilon']:.3f}, loss={stats['avg_loss']:.4f}")
    
    # Save trained model
    agent.save_model(save_path)
    
    logger.info(f"RL training completed. Model saved to {save_path}")
    return agent


if __name__ == "__main__":
    # Train RL agent
    trained_agent = train_rl_agent(episodes=500)
    
    # Demo usage
    demo_state = RoutingState(
        batch_size=8, seq_length=1024, embed_dim=768, num_heads=12,
        current_load_gpu=0.7, current_load_photonic=0.3,
        recent_gpu_latency=45.0, recent_photonic_latency=25.0,
        temperature_photonic=65.0, energy_budget_remaining=0.8,
        time_of_day=0.5
    )
    
    action, q_value = trained_agent.get_action(demo_state, training=False)
    print(f"Recommended action: device={action.device}, "
          f"load_factor={action.load_balance_factor}, "
          f"priority={action.priority}, q_value={q_value:.3f}")