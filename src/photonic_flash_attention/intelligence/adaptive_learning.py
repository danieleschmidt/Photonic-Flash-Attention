"""Adaptive Learning and Intelligence System.

Implements machine learning-driven adaptation, pattern recognition,
and intelligent decision making for photonic flash attention systems.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import time
import json
import pickle
from pathlib import Path
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

from ..config import get_config
from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicComputationError


@dataclass
class LearningExample:
    """A single learning example with features and outcomes."""
    timestamp: float
    workload_features: np.ndarray
    system_state: Dict[str, Any]
    decision_made: str
    outcome_metrics: Dict[str, float]
    success: bool
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LearningPattern:
    """Discovered pattern from learning examples."""
    pattern_id: str
    feature_signature: np.ndarray
    confidence_score: float
    success_rate: float
    usage_count: int
    last_updated: float
    recommended_action: str
    pattern_description: str
    supporting_examples: List[int] = field(default_factory=list)


class WorkloadPatternAnalyzer:
    """
    Analyzes workload patterns and identifies recurring characteristics.
    
    Uses unsupervised learning to discover natural groupings in workload
    characteristics and performance outcomes.
    """
    
    def __init__(self, max_patterns: int = 50, min_examples_per_pattern: int = 10):
        self.max_patterns = max_patterns
        self.min_examples_per_pattern = min_examples_per_pattern
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Machine learning components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=0.95)  # Keep 95% of variance
        self.clusterer = KMeans(n_clusters=8, random_state=42)
        
        # Pattern storage
        self.discovered_patterns: Dict[str, LearningPattern] = {}
        self.learning_examples: List[LearningExample] = []
        
        # Feature extraction
        self.feature_extractors = {
            'workload_complexity': self._extract_complexity_features,
            'resource_usage': self._extract_resource_features,
            'temporal_patterns': self._extract_temporal_features,
            'performance_history': self._extract_performance_features,
        }
        
        # Model state
        self.model_trained = False
        self.last_training_time = 0
        self.training_interval = 300  # 5 minutes
        
        self._lock = threading.RLock()
    
    def add_example(self, workload_characteristics: Dict[str, Any], 
                   system_state: Dict[str, Any], decision: str, 
                   outcome: Dict[str, float], success: bool) -> None:
        """Add a new learning example."""
        with self._lock:
            # Extract features
            features = self._extract_all_features(workload_characteristics, system_state)
            
            # Create learning example
            example = LearningExample(
                timestamp=time.time(),
                workload_features=features,
                system_state=system_state.copy(),
                decision_made=decision,
                outcome_metrics=outcome.copy(),
                success=success,
                metadata={
                    'workload': workload_characteristics.copy(),
                    'example_id': len(self.learning_examples)
                }
            )
            
            self.learning_examples.append(example)
            
            # Limit memory usage
            if len(self.learning_examples) > 10000:
                self.learning_examples = self.learning_examples[-8000:]  # Keep recent examples
            
            # Trigger retraining if needed
            if (time.time() - self.last_training_time > self.training_interval and 
                len(self.learning_examples) >= self.min_examples_per_pattern * 2):
                self._retrain_models()
    
    def _extract_all_features(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> np.ndarray:
        """Extract all features for a workload and system state."""
        all_features = []
        
        for extractor_name, extractor_func in self.feature_extractors.items():
            try:
                features = extractor_func(workload, system_state)
                all_features.extend(features)
            except Exception as e:
                self.logger.warning(f"Feature extraction failed for {extractor_name}: {e}")
                # Add zeros as placeholder
                all_features.extend([0.0] * 5)  # Assume 5 features per extractor
        
        return np.array(all_features, dtype=np.float32)
    
    def _extract_complexity_features(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> List[float]:
        """Extract workload complexity features."""
        batch_size = workload.get('batch_size', 1)
        seq_length = workload.get('seq_length', 512)
        embed_dim = workload.get('embed_dim', 768)
        num_heads = workload.get('num_heads', 12)
        
        # Complexity metrics
        computational_complexity = batch_size * seq_length * seq_length * embed_dim
        memory_complexity = batch_size * seq_length * embed_dim
        attention_complexity = seq_length * seq_length * num_heads
        parameter_count = embed_dim * embed_dim * 3  # QKV projections
        
        # Normalized complexity features
        return [
            np.log10(max(computational_complexity, 1)),
            np.log10(max(memory_complexity, 1)),
            np.log10(max(attention_complexity, 1)),
            np.log10(max(parameter_count, 1)),
            min(seq_length / 2048, 10.0),  # Sequence length ratio
        ]
    
    def _extract_resource_features(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> List[float]:
        """Extract resource usage features."""
        # System resource features
        cpu_usage = system_state.get('cpu_usage', 0.5)
        memory_usage = system_state.get('memory_usage', 0.5)
        gpu_usage = system_state.get('gpu_usage', 0.5)
        temperature = system_state.get('temperature', 45.0)
        
        # Normalized resource features
        return [
            cpu_usage,
            memory_usage,
            gpu_usage,
            min(temperature / 100.0, 1.0),
            system_state.get('photonic_available', 0.0),
        ]
    
    def _extract_temporal_features(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> List[float]:
        """Extract temporal pattern features."""
        current_time = time.time()
        
        # Time-based features
        hour_of_day = (current_time % (24 * 3600)) / (24 * 3600)
        day_of_week = ((current_time // (24 * 3600)) % 7) / 7
        
        # Recent activity features
        recent_examples = [ex for ex in self.learning_examples 
                          if current_time - ex.timestamp < 3600]  # Last hour
        
        recent_success_rate = (sum(ex.success for ex in recent_examples) / 
                              max(len(recent_examples), 1))
        recent_activity = len(recent_examples) / 100.0  # Normalize by typical activity
        
        return [
            hour_of_day,
            day_of_week,
            recent_success_rate,
            min(recent_activity, 2.0),
            workload.get('is_training', 0.0),
        ]
    
    def _extract_performance_features(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> List[float]:
        """Extract performance history features."""
        # Historical performance features
        recent_latencies = []
        recent_throughputs = []
        recent_energy = []
        
        for example in self.learning_examples[-100:]:  # Last 100 examples
            metrics = example.outcome_metrics
            if 'latency_ms' in metrics:
                recent_latencies.append(metrics['latency_ms'])
            if 'throughput' in metrics:
                recent_throughputs.append(metrics['throughput'])
            if 'energy_mj' in metrics:
                recent_energy.append(metrics['energy_mj'])
        
        # Statistical features
        avg_latency = np.mean(recent_latencies) if recent_latencies else 100.0
        avg_throughput = np.mean(recent_throughputs) if recent_throughputs else 1000.0
        avg_energy = np.mean(recent_energy) if recent_energy else 50.0
        
        return [
            min(avg_latency / 1000.0, 5.0),  # Normalized latency
            min(avg_throughput / 10000.0, 2.0),  # Normalized throughput
            min(avg_energy / 1000.0, 2.0),  # Normalized energy
            system_state.get('load_factor', 0.5),
            len(recent_latencies) / 100.0,  # Sample density
        ]
    
    def _retrain_models(self) -> None:
        """Retrain pattern recognition models."""
        if len(self.learning_examples) < self.min_examples_per_pattern:
            return
        
        try:
            self.logger.info(f"Retraining pattern models with {len(self.learning_examples)} examples")
            start_time = time.time()
            
            # Prepare training data
            X = np.array([ex.workload_features for ex in self.learning_examples])
            
            # Handle NaN/inf values
            X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
            
            if X.shape[0] < 2:
                return
            
            # Fit preprocessing
            X_scaled = self.scaler.fit_transform(X)
            
            # Dimensionality reduction
            if X_scaled.shape[1] > 10:
                X_reduced = self.pca.fit_transform(X_scaled)
            else:
                X_reduced = X_scaled
            
            # Clustering
            n_clusters = min(self.max_patterns // 2, max(2, len(self.learning_examples) // 20))
            self.clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = self.clusterer.fit_predict(X_reduced)
            
            # Extract patterns from clusters
            self._extract_patterns_from_clusters(cluster_labels)
            
            self.model_trained = True
            self.last_training_time = time.time()
            training_time = self.last_training_time - start_time
            
            self.logger.info(f"Model retraining completed in {training_time:.2f}s, discovered {len(self.discovered_patterns)} patterns")
            
        except Exception as e:
            self.logger.error(f"Model retraining failed: {e}")
    
    def _extract_patterns_from_clusters(self, cluster_labels: np.ndarray) -> None:
        """Extract learning patterns from cluster assignments."""
        new_patterns = {}
        
        for cluster_id in np.unique(cluster_labels):
            cluster_mask = cluster_labels == cluster_id
            cluster_examples = [self.learning_examples[i] for i, mask in enumerate(cluster_mask) if mask]
            
            if len(cluster_examples) < self.min_examples_per_pattern:
                continue
            
            # Calculate cluster statistics
            success_rate = sum(ex.success for ex in cluster_examples) / len(cluster_examples)
            
            # Find most common decision
            decisions = [ex.decision_made for ex in cluster_examples]
            most_common_decision = max(set(decisions), key=decisions.count)
            
            # Calculate feature signature (centroid)
            features = np.array([ex.workload_features for ex in cluster_examples])
            feature_signature = np.mean(features, axis=0)
            
            # Create pattern
            pattern = LearningPattern(
                pattern_id=f"pattern_{cluster_id}_{int(time.time())}",
                feature_signature=feature_signature,
                confidence_score=min(success_rate * (len(cluster_examples) / 100), 1.0),
                success_rate=success_rate,
                usage_count=len(cluster_examples),
                last_updated=time.time(),
                recommended_action=most_common_decision,
                pattern_description=self._describe_pattern(cluster_examples),
                supporting_examples=[ex.metadata.get('example_id', i) for i, ex in enumerate(cluster_examples)]
            )
            
            new_patterns[pattern.pattern_id] = pattern
        
        # Update discovered patterns
        with self._lock:
            self.discovered_patterns = new_patterns
    
    def _describe_pattern(self, examples: List[LearningExample]) -> str:
        """Generate human-readable pattern description."""
        if not examples:
            return "Empty pattern"
        
        # Analyze common characteristics
        workloads = [ex.metadata.get('workload', {}) for ex in examples]
        
        # Sequence length analysis
        seq_lengths = [w.get('seq_length', 0) for w in workloads]
        avg_seq_len = np.mean(seq_lengths) if seq_lengths else 0
        
        # Batch size analysis
        batch_sizes = [w.get('batch_size', 0) for w in workloads]
        avg_batch_size = np.mean(batch_sizes) if batch_sizes else 0
        
        # Success rate
        success_rate = sum(ex.success for ex in examples) / len(examples)
        
        # Common decision
        decisions = [ex.decision_made for ex in examples]
        most_common_decision = max(set(decisions), key=decisions.count)
        
        # Generate description
        description_parts = []
        
        if avg_seq_len < 256:
            description_parts.append("short sequences")
        elif avg_seq_len < 1024:
            description_parts.append("medium sequences")
        else:
            description_parts.append("long sequences")
        
        if avg_batch_size < 2:
            description_parts.append("small batches")
        elif avg_batch_size < 8:
            description_parts.append("medium batches")
        else:
            description_parts.append("large batches")
        
        description_parts.append(f"→ {most_common_decision}")
        description_parts.append(f"({success_rate:.1%} success)")
        
        return ", ".join(description_parts)
    
    def predict_best_action(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> Tuple[str, float]:
        """Predict the best action for a given workload and system state."""
        if not self.model_trained or not self.discovered_patterns:
            return "gpu", 0.5  # Default fallback
        
        try:
            # Extract features
            features = self._extract_all_features(workload, system_state)
            
            # Find best matching pattern
            best_pattern = None
            best_similarity = -1.0
            
            for pattern in self.discovered_patterns.values():
                similarity = self._calculate_pattern_similarity(features, pattern.feature_signature)
                
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pattern = pattern
            
            if best_pattern and best_similarity > 0.3:  # Minimum similarity threshold
                confidence = best_pattern.confidence_score * best_similarity
                return best_pattern.recommended_action, confidence
            else:
                return "gpu", 0.3  # Low confidence fallback
                
        except Exception as e:
            self.logger.error(f"Prediction failed: {e}")
            return "gpu", 0.1  # Error fallback
    
    def _calculate_pattern_similarity(self, features: np.ndarray, pattern_signature: np.ndarray) -> float:
        """Calculate similarity between features and pattern signature."""
        try:
            # Handle different lengths
            if len(features) != len(pattern_signature):
                min_len = min(len(features), len(pattern_signature))
                features = features[:min_len]
                pattern_signature = pattern_signature[:min_len]
            
            # Normalize features
            features_norm = features / (np.linalg.norm(features) + 1e-8)
            pattern_norm = pattern_signature / (np.linalg.norm(pattern_signature) + 1e-8)
            
            # Cosine similarity
            similarity = np.dot(features_norm, pattern_norm)
            
            # Convert to [0, 1] range
            return max(0.0, (similarity + 1.0) / 2.0)
            
        except Exception:
            return 0.0
    
    def get_pattern_insights(self) -> Dict[str, Any]:
        """Get insights about discovered patterns."""
        with self._lock:
            if not self.discovered_patterns:
                return {'insights': 'No patterns discovered yet'}
            
            patterns_list = list(self.discovered_patterns.values())
            
            # Calculate statistics
            avg_confidence = np.mean([p.confidence_score for p in patterns_list])
            avg_success_rate = np.mean([p.success_rate for p in patterns_list])
            
            # Find most successful patterns
            top_patterns = sorted(patterns_list, key=lambda p: p.success_rate * p.confidence_score, reverse=True)[:5]
            
            # Action distribution
            action_counts = defaultdict(int)
            for pattern in patterns_list:
                action_counts[pattern.recommended_action] += pattern.usage_count
            
            return {
                'total_patterns': len(patterns_list),
                'avg_confidence': avg_confidence,
                'avg_success_rate': avg_success_rate,
                'total_examples': len(self.learning_examples),
                'model_trained': self.model_trained,
                'top_patterns': [{
                    'description': p.pattern_description,
                    'success_rate': p.success_rate,
                    'confidence': p.confidence_score,
                    'usage_count': p.usage_count,
                } for p in top_patterns],
                'action_distribution': dict(action_counts),
                'last_training': self.last_training_time,
            }


class AdaptiveDecisionEngine:
    """
    Intelligent decision engine that learns from outcomes and adapts strategies.
    
    Combines multiple decision-making approaches:
    - Pattern-based decisions from learned examples
    - Rule-based decisions for known scenarios
    - Exploration for unknown scenarios
    - Multi-armed bandit for action selection
    """
    
    def __init__(self, exploration_rate: float = 0.1):
        self.exploration_rate = exploration_rate
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Decision components
        self.pattern_analyzer = WorkloadPatternAnalyzer()
        
        # Multi-armed bandit for action selection
        self.action_rewards = defaultdict(list)  # Track rewards per action
        self.action_counts = defaultdict(int)    # Track selection counts
        
        # Available actions
        self.available_actions = ['gpu', 'photonic', 'hybrid', 'auto']
        
        # Rule-based decision rules
        self.decision_rules = {
            'large_sequence_photonic': (self._large_sequence_rule, 'photonic'),
            'small_sequence_gpu': (self._small_sequence_rule, 'gpu'),
            'memory_pressure_reduce': (self._memory_pressure_rule, 'gpu'),
            'thermal_throttling': (self._thermal_rule, 'gpu'),
        }
        
        # Decision history
        self.decision_history = deque(maxlen=1000)
        
        # Learning parameters
        self.learning_rate = 0.1
        self.confidence_threshold = 0.7
        
        self._lock = threading.RLock()
    
    def make_decision(self, workload: Dict[str, Any], system_state: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make intelligent decision based on workload and system state."""
        context = context or {}
        decision_start_time = time.time()
        
        with self._lock:
            # Try pattern-based decision first
            pattern_action, pattern_confidence = self.pattern_analyzer.predict_best_action(workload, system_state)
            
            # Try rule-based decisions
            rule_action, rule_confidence = self._apply_decision_rules(workload, system_state)
            
            # Multi-armed bandit selection
            bandit_action, bandit_confidence = self._select_bandit_action(workload, system_state)
            
            # Combine decision approaches
            decision_candidates = [
                ('pattern', pattern_action, pattern_confidence),
                ('rule', rule_action, rule_confidence),
                ('bandit', bandit_action, bandit_confidence),
            ]
            
            # Select best decision approach
            selected_approach, selected_action, selected_confidence = max(
                decision_candidates, key=lambda x: x[2]
            )
            
            # Apply exploration
            if np.random.random() < self.exploration_rate:
                selected_action = np.random.choice(self.available_actions)
                selected_approach = 'exploration'
                selected_confidence = 0.1
            
            # Create decision record
            decision_record = {
                'action': selected_action,
                'confidence': selected_confidence,
                'approach': selected_approach,
                'timestamp': time.time(),
                'decision_time_ms': (time.time() - decision_start_time) * 1000,
                'alternatives': {
                    'pattern': (pattern_action, pattern_confidence),
                    'rule': (rule_action, rule_confidence),
                    'bandit': (bandit_action, bandit_confidence),
                },
                'workload_summary': {
                    'seq_length': workload.get('seq_length', 0),
                    'batch_size': workload.get('batch_size', 0),
                    'embed_dim': workload.get('embed_dim', 0),
                },
                'system_summary': {
                    'memory_usage': system_state.get('memory_usage', 0),
                    'gpu_usage': system_state.get('gpu_usage', 0),
                    'temperature': system_state.get('temperature', 0),
                },
            }
            
            self.decision_history.append(decision_record)
            
            self.logger.debug(f"Decision made: {selected_action} ({selected_approach}, confidence={selected_confidence:.2f})")
            
            return decision_record
    
    def _apply_decision_rules(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> Tuple[str, float]:
        """Apply rule-based decision logic."""
        best_action = 'gpu'
        best_confidence = 0.2
        
        for rule_name, (rule_func, recommended_action) in self.decision_rules.items():
            try:
                applies, confidence = rule_func(workload, system_state)
                if applies and confidence > best_confidence:
                    best_action = recommended_action
                    best_confidence = confidence
            except Exception as e:
                self.logger.warning(f"Rule {rule_name} failed: {e}")
        
        return best_action, best_confidence
    
    def _large_sequence_rule(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> Tuple[bool, float]:
        """Rule for large sequence lengths."""
        seq_length = workload.get('seq_length', 0)
        photonic_available = system_state.get('photonic_available', False)
        
        if seq_length > 1024 and photonic_available:
            confidence = min(0.9, 0.5 + (seq_length - 1024) / 2048)
            return True, confidence
        
        return False, 0.0
    
    def _small_sequence_rule(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> Tuple[bool, float]:
        """Rule for small sequence lengths."""
        seq_length = workload.get('seq_length', 0)
        
        if seq_length < 256:
            confidence = 0.8
            return True, confidence
        
        return False, 0.0
    
    def _memory_pressure_rule(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> Tuple[bool, float]:
        """Rule for memory pressure situations."""
        memory_usage = system_state.get('memory_usage', 0)
        
        if memory_usage > 0.85:
            confidence = min(0.9, memory_usage)
            return True, confidence
        
        return False, 0.0
    
    def _thermal_rule(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> Tuple[bool, float]:
        """Rule for thermal throttling."""
        temperature = system_state.get('temperature', 25)
        
        if temperature > 70:
            confidence = min(0.9, (temperature - 70) / 20)
            return True, confidence
        
        return False, 0.0
    
    def _select_bandit_action(self, workload: Dict[str, Any], system_state: Dict[str, Any]) -> Tuple[str, float]:
        """Select action using multi-armed bandit (Upper Confidence Bound)."""
        if not self.action_rewards:
            return 'gpu', 0.3  # Default when no history
        
        total_counts = sum(self.action_counts.values())
        if total_counts == 0:
            return 'gpu', 0.3
        
        ucb_values = {}
        
        for action in self.available_actions:
            if self.action_counts[action] == 0:
                ucb_values[action] = float('inf')  # Explore unvisited actions
            else:
                avg_reward = np.mean(self.action_rewards[action]) if self.action_rewards[action] else 0
                exploration_bonus = np.sqrt(2 * np.log(total_counts) / self.action_counts[action])
                ucb_values[action] = avg_reward + exploration_bonus
        
        best_action = max(ucb_values, key=ucb_values.get)
        confidence = min(0.8, ucb_values[best_action] / max(ucb_values.values()))
        
        return best_action, confidence
    
    def record_outcome(self, decision_record: Dict[str, Any], outcome_metrics: Dict[str, float], success: bool) -> None:
        """Record the outcome of a decision for learning."""
        with self._lock:
            action = decision_record['action']
            
            # Calculate reward based on outcome
            reward = self._calculate_reward(outcome_metrics, success)
            
            # Update bandit statistics
            self.action_rewards[action].append(reward)
            self.action_counts[action] += 1
            
            # Limit memory usage
            if len(self.action_rewards[action]) > 1000:
                self.action_rewards[action] = self.action_rewards[action][-800:]
            
            # Add to pattern analyzer
            workload = decision_record['workload_summary']
            system_state = decision_record['system_summary']
            
            self.pattern_analyzer.add_example(
                workload, system_state, action, outcome_metrics, success
            )
            
            # Update decision record
            decision_record['outcome_recorded'] = True
            decision_record['outcome_metrics'] = outcome_metrics
            decision_record['success'] = success
            decision_record['reward'] = reward
    
    def _calculate_reward(self, outcome_metrics: Dict[str, float], success: bool) -> float:
        """Calculate reward signal from outcome metrics."""
        if not success:
            return 0.0
        
        # Reward components
        latency_reward = 0.0
        throughput_reward = 0.0
        energy_reward = 0.0
        
        # Latency reward (lower is better)
        if 'latency_ms' in outcome_metrics:
            latency_ms = outcome_metrics['latency_ms']
            latency_reward = max(0, 1.0 - latency_ms / 1000.0)  # Normalize to [0, 1]
        
        # Throughput reward (higher is better)
        if 'throughput' in outcome_metrics:
            throughput = outcome_metrics['throughput']
            throughput_reward = min(1.0, throughput / 10000.0)  # Normalize
        
        # Energy reward (lower is better)
        if 'energy_mj' in outcome_metrics:
            energy_mj = outcome_metrics['energy_mj']
            energy_reward = max(0, 1.0 - energy_mj / 100.0)  # Normalize
        
        # Combined reward
        total_reward = (latency_reward + throughput_reward + energy_reward) / 3
        
        return max(0.0, min(1.0, total_reward))
    
    def get_learning_statistics(self) -> Dict[str, Any]:
        """Get comprehensive learning statistics."""
        with self._lock:
            stats = {
                'total_decisions': len(self.decision_history),
                'exploration_rate': self.exploration_rate,
                'action_statistics': {},
                'pattern_insights': self.pattern_analyzer.get_pattern_insights(),
                'recent_decisions': list(self.decision_history)[-10:],
            }
            
            # Action statistics
            for action in self.available_actions:
                rewards = self.action_rewards[action]
                stats['action_statistics'][action] = {
                    'count': self.action_counts[action],
                    'avg_reward': np.mean(rewards) if rewards else 0.0,
                    'reward_std': np.std(rewards) if rewards else 0.0,
                    'success_rate': sum(1 for r in rewards if r > 0.5) / max(len(rewards), 1),
                }
            
            # Decision approach distribution
            approach_counts = defaultdict(int)
            for decision in self.decision_history:
                approach_counts[decision['approach']] += 1
            
            stats['approach_distribution'] = dict(approach_counts)
            
            return stats
    
    def adjust_exploration_rate(self, new_rate: float) -> None:
        """Adjust exploration rate for learning."""
        with self._lock:
            old_rate = self.exploration_rate
            self.exploration_rate = max(0.0, min(1.0, new_rate))
            self.logger.info(f"Exploration rate adjusted: {old_rate:.3f} -> {self.exploration_rate:.3f}")
    
    def reset_learning_state(self) -> None:
        """Reset learning state for fresh start."""
        with self._lock:
            self.action_rewards.clear()
            self.action_counts.clear()
            self.decision_history.clear()
            
            # Reset pattern analyzer
            self.pattern_analyzer = WorkloadPatternAnalyzer()
            
            self.logger.info("Learning state reset")


class IntelligentSystemOrchestrator:
    """
    High-level orchestrator that coordinates intelligent decision making,
    learning, and adaptation across the entire photonic attention system.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Core intelligence components
        self.decision_engine = AdaptiveDecisionEngine(
            exploration_rate=self.config.get('exploration_rate', 0.1)
        )
        
        # System state tracking
        self.current_system_state = {
            'cpu_usage': 0.5,
            'memory_usage': 0.5,
            'gpu_usage': 0.5,
            'temperature': 45.0,
            'photonic_available': True,
            'load_factor': 0.5,
        }
        
        # Performance tracking
        self.performance_history = deque(maxlen=1000)
        self.system_health_score = 1.0
        
        # Adaptation parameters
        self.adaptation_interval = self.config.get('adaptation_interval', 60)  # seconds
        self.last_adaptation_time = time.time()
        
        # Background adaptation
        self._adaptation_active = False
        self._adaptation_thread = None
        
        self._lock = threading.RLock()
    
    def start_intelligent_orchestration(self) -> None:
        """Start intelligent orchestration with background adaptation."""
        with self._lock:
            if self._adaptation_active:
                return
            
            self._adaptation_active = True
            self._adaptation_thread = threading.Thread(
                target=self._adaptation_loop,
                daemon=True
            )
            self._adaptation_thread.start()
            
            self.logger.info("Intelligent orchestration started")
    
    def stop_intelligent_orchestration(self) -> None:
        """Stop intelligent orchestration."""
        with self._lock:
            if not self._adaptation_active:
                return
            
            self._adaptation_active = False
            
            if self._adaptation_thread and self._adaptation_thread.is_alive():
                self._adaptation_thread.join(timeout=5.0)
            
            self.logger.info("Intelligent orchestration stopped")
    
    def _adaptation_loop(self) -> None:
        """Background adaptation loop."""
        while self._adaptation_active:
            try:
                time.sleep(self.adaptation_interval)
                
                if self._adaptation_active:
                    self._perform_system_adaptation()
                    
            except Exception as e:
                self.logger.error(f"Adaptation loop error: {e}", exc_info=True)
    
    def _perform_system_adaptation(self) -> None:
        """Perform intelligent system adaptation."""
        current_time = time.time()
        
        try:
            # Update system state
            self._update_system_state()
            
            # Analyze recent performance
            performance_analysis = self._analyze_recent_performance()
            
            # Adapt exploration rate based on performance
            self._adapt_exploration_rate(performance_analysis)
            
            # System health assessment
            self._update_system_health(performance_analysis)
            
            self.last_adaptation_time = current_time
            
            self.logger.debug(f"System adaptation completed: health={self.system_health_score:.2f}")
            
        except Exception as e:
            self.logger.error(f"System adaptation failed: {e}")
    
    def _update_system_state(self) -> None:
        """Update current system state with real-time metrics."""
        try:
            import psutil
            
            # System metrics
            self.current_system_state.update({
                'cpu_usage': psutil.cpu_percent(interval=1) / 100.0,
                'memory_usage': psutil.virtual_memory().percent / 100.0,
                'timestamp': time.time(),
            })
            
            # GPU metrics
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_stats()
                allocated = gpu_memory.get('allocated_bytes.all.current', 0)
                max_memory = torch.cuda.get_device_properties(0).total_memory
                self.current_system_state['gpu_usage'] = allocated / max_memory
            
            # Load factor (simplified)
            recent_decisions = len([d for d in self.decision_engine.decision_history 
                                  if time.time() - d['timestamp'] < 60])
            self.current_system_state['load_factor'] = min(1.0, recent_decisions / 60.0)
            
        except Exception as e:
            self.logger.warning(f"System state update failed: {e}")
    
    def _analyze_recent_performance(self) -> Dict[str, Any]:
        """Analyze recent system performance."""
        recent_performance = [p for p in self.performance_history 
                             if time.time() - p.get('timestamp', 0) < 300]  # Last 5 minutes
        
        if not recent_performance:
            return {'trend': 'no_data', 'avg_success_rate': 0.5}
        
        # Success rate analysis
        success_rates = [p.get('success_rate', 0.5) for p in recent_performance]
        avg_success_rate = np.mean(success_rates)
        
        # Latency analysis
        latencies = [p.get('avg_latency', 100) for p in recent_performance if 'avg_latency' in p]
        avg_latency = np.mean(latencies) if latencies else 100.0
        
        # Trend analysis
        if len(success_rates) >= 3:
            recent_trend = np.mean(success_rates[-3:]) - np.mean(success_rates[:3])
            trend = 'improving' if recent_trend > 0.05 else 'declining' if recent_trend < -0.05 else 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'avg_success_rate': avg_success_rate,
            'avg_latency': avg_latency,
            'sample_count': len(recent_performance),
            'recent_trend': recent_trend if len(success_rates) >= 3 else 0.0,
        }
    
    def _adapt_exploration_rate(self, performance_analysis: Dict[str, Any]) -> None:
        """Adapt exploration rate based on performance trends."""
        current_rate = self.decision_engine.exploration_rate
        
        # Adaptation logic
        if performance_analysis['trend'] == 'declining':
            # Increase exploration when performance is declining
            new_rate = min(0.3, current_rate * 1.2)
        elif performance_analysis['trend'] == 'improving':
            # Reduce exploration when performance is improving
            new_rate = max(0.05, current_rate * 0.9)
        else:
            # Gradual return to baseline
            baseline_rate = self.config.get('exploration_rate', 0.1)
            new_rate = current_rate * 0.95 + baseline_rate * 0.05
        
        if abs(new_rate - current_rate) > 0.01:
            self.decision_engine.adjust_exploration_rate(new_rate)
    
    def _update_system_health(self, performance_analysis: Dict[str, Any]) -> None:
        """Update overall system health score."""
        # Health components
        performance_health = performance_analysis.get('avg_success_rate', 0.5)
        resource_health = 1.0 - max(
            self.current_system_state.get('cpu_usage', 0),
            self.current_system_state.get('memory_usage', 0),
            self.current_system_state.get('gpu_usage', 0)
        )
        
        # Thermal health
        temperature = self.current_system_state.get('temperature', 45)
        thermal_health = max(0.0, 1.0 - max(0, temperature - 60) / 40)
        
        # Combined health score
        new_health = (performance_health + resource_health + thermal_health) / 3
        
        # Smooth health score changes
        self.system_health_score = self.system_health_score * 0.8 + new_health * 0.2
    
    def make_intelligent_decision(self, workload: Dict[str, Any], context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Make intelligent decision with full context awareness."""
        decision = self.decision_engine.make_decision(
            workload, self.current_system_state, context
        )
        
        # Enhance decision with system context
        decision['system_health_score'] = self.system_health_score
        decision['intelligent_orchestration'] = True
        
        return decision
    
    def record_execution_outcome(self, decision: Dict[str, Any], outcome_metrics: Dict[str, float], success: bool) -> None:
        """Record execution outcome for learning."""
        # Record in decision engine
        self.decision_engine.record_outcome(decision, outcome_metrics, success)
        
        # Update performance history
        performance_record = {
            'timestamp': time.time(),
            'decision': decision['action'],
            'success': success,
            'metrics': outcome_metrics.copy(),
            'system_state': self.current_system_state.copy(),
        }
        
        self.performance_history.append(performance_record)
    
    def get_intelligence_status(self) -> Dict[str, Any]:
        """Get comprehensive intelligence status."""
        with self._lock:
            status = {
                'orchestration_active': self._adaptation_active,
                'system_health_score': self.system_health_score,
                'system_state': self.current_system_state.copy(),
                'learning_statistics': self.decision_engine.get_learning_statistics(),
                'performance_summary': self._get_performance_summary(),
                'last_adaptation': self.last_adaptation_time,
            }
            
            return status
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from recent history."""
        recent_performance = [p for p in self.performance_history 
                             if time.time() - p.get('timestamp', 0) < 3600]  # Last hour
        
        if not recent_performance:
            return {'status': 'no_recent_data'}
        
        success_rate = sum(p.get('success', False) for p in recent_performance) / len(recent_performance)
        
        # Latency statistics
        latencies = [p['metrics'].get('latency_ms', 0) for p in recent_performance 
                    if 'latency_ms' in p.get('metrics', {})]
        
        return {
            'total_executions': len(recent_performance),
            'success_rate': success_rate,
            'avg_latency_ms': np.mean(latencies) if latencies else 0,
            'median_latency_ms': np.median(latencies) if latencies else 0,
            'p95_latency_ms': np.percentile(latencies, 95) if latencies else 0,
        }
    
    def shutdown(self) -> None:
        """Shutdown intelligent orchestrator."""
        self.stop_intelligent_orchestration()
        self.logger.info("Intelligent orchestrator shutdown complete")
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.shutdown()
        except Exception:
            pass
