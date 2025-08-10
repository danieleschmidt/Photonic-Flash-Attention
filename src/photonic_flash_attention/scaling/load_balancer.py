"""Load balancing and auto-scaling for distributed photonic attention."""

import time
import threading
import queue
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import deque
import logging
import random
import hashlib

from ..utils.logging import get_logger
from ..photonic.hardware.detection import get_photonic_devices


logger = get_logger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections" 
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    RESOURCE_BASED = "resource_based"
    PERFORMANCE_BASED = "performance_based"
    CONSISTENT_HASH = "consistent_hash"


class NodeStatus(Enum):
    """Node status for load balancing."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    OVERLOADED = "overloaded"
    UNHEALTHY = "unhealthy"
    OFFLINE = "offline"


@dataclass
class ComputeNode:
    """Represents a compute node in the cluster."""
    node_id: str
    device_type: str  # 'photonic', 'gpu', 'cpu'
    capacity: int  # Maximum concurrent operations
    current_load: int = 0
    weight: float = 1.0  # Load balancing weight
    status: NodeStatus = NodeStatus.HEALTHY
    last_health_check: float = field(default_factory=time.time)
    avg_latency_ms: float = 0.0
    success_rate: float = 1.0
    total_requests: int = 0
    failed_requests: int = 0
    endpoints: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def utilization(self) -> float:
        """Current utilization as fraction of capacity."""
        return self.current_load / max(self.capacity, 1)
    
    @property
    def is_available(self) -> bool:
        """Check if node is available for requests."""
        return (
            self.status in (NodeStatus.HEALTHY, NodeStatus.DEGRADED) and
            self.current_load < self.capacity
        )


@dataclass
class LoadBalancerConfig:
    """Configuration for load balancer."""
    strategy: LoadBalancingStrategy = LoadBalancingStrategy.PERFORMANCE_BASED
    health_check_interval: float = 30.0
    max_retries: int = 3
    retry_delay: float = 1.0
    request_timeout: float = 30.0
    enable_circuit_breaker: bool = True
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout: float = 60.0
    enable_sticky_sessions: bool = False
    session_timeout: float = 3600.0  # 1 hour
    enable_auto_scaling: bool = True
    scale_up_threshold: float = 0.8  # 80% utilization
    scale_down_threshold: float = 0.3  # 30% utilization
    min_nodes: int = 1
    max_nodes: int = 10


class ConsistentHashRing:
    """Consistent hash ring for distributed load balancing."""
    
    def __init__(self, replicas: int = 150):
        """Initialize consistent hash ring."""
        self.replicas = replicas
        self.ring: Dict[int, str] = {}
        self.sorted_keys: List[int] = []
        self._lock = threading.RLock()
    
    def add_node(self, node_id: str) -> None:
        """Add node to hash ring."""
        with self._lock:
            for i in range(self.replicas):
                key = self._hash(f"{node_id}:{i}")
                self.ring[key] = node_id
            
            self.sorted_keys = sorted(self.ring.keys())
            logger.debug(f"Added node {node_id} to hash ring")
    
    def remove_node(self, node_id: str) -> None:
        """Remove node from hash ring."""
        with self._lock:
            keys_to_remove = []
            for key, node in self.ring.items():
                if node == node_id:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.ring[key]
            
            self.sorted_keys = sorted(self.ring.keys())
            logger.debug(f"Removed node {node_id} from hash ring")
    
    def get_node(self, key: str) -> Optional[str]:
        """Get node for given key."""
        with self._lock:
            if not self.ring:
                return None
            
            hash_key = self._hash(key)
            
            # Find first node with hash >= hash_key
            for ring_key in self.sorted_keys:
                if ring_key >= hash_key:
                    return self.ring[ring_key]
            
            # Wrap around to first node
            return self.ring[self.sorted_keys[0]]
    
    def _hash(self, key: str) -> int:
        """Hash function for consistent hashing."""
        return int(hashlib.md5(key.encode()).hexdigest(), 16)


class SessionManager:
    """Manages sticky sessions for load balancing."""
    
    def __init__(self, session_timeout: float = 3600.0):
        """Initialize session manager."""
        self.session_timeout = session_timeout
        self.sessions: Dict[str, Tuple[str, float]] = {}  # session_id -> (node_id, timestamp)
        self._lock = threading.RLock()
        
        # Start cleanup thread
        self._cleanup_thread = threading.Thread(target=self._cleanup_sessions, daemon=True)
        self._cleanup_thread.start()
    
    def get_session_node(self, session_id: str) -> Optional[str]:
        """Get node for session."""
        with self._lock:
            if session_id in self.sessions:
                node_id, timestamp = self.sessions[session_id]
                if time.time() - timestamp < self.session_timeout:
                    # Update timestamp
                    self.sessions[session_id] = (node_id, time.time())
                    return node_id
                else:
                    # Session expired
                    del self.sessions[session_id]
        
        return None
    
    def create_session(self, session_id: str, node_id: str) -> None:
        """Create new session."""
        with self._lock:
            self.sessions[session_id] = (node_id, time.time())
            logger.debug(f"Created session {session_id} -> {node_id}")
    
    def _cleanup_sessions(self) -> None:
        """Cleanup expired sessions."""
        while True:
            try:
                time.sleep(300)  # Clean up every 5 minutes
                current_time = time.time()
                
                with self._lock:
                    expired_sessions = [
                        session_id for session_id, (_, timestamp) in self.sessions.items()
                        if current_time - timestamp >= self.session_timeout
                    ]
                    
                    for session_id in expired_sessions:
                        del self.sessions[session_id]
                    
                    if expired_sessions:
                        logger.debug(f"Cleaned up {len(expired_sessions)} expired sessions")
                        
            except Exception as e:
                logger.error(f"Error in session cleanup: {e}")


class LoadBalancer:
    """Advanced load balancer for photonic attention workloads."""
    
    def __init__(self, config: Optional[LoadBalancerConfig] = None):
        """Initialize load balancer."""
        self.config = config or LoadBalancerConfig()
        self.nodes: Dict[str, ComputeNode] = {}
        self.request_queue = queue.PriorityQueue()
        self.hash_ring = ConsistentHashRing()
        self.session_manager = SessionManager(self.config.session_timeout) if self.config.enable_sticky_sessions else None
        
        # Tracking
        self.round_robin_index = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0
        self._lock = threading.RLock()
        
        # Start health check thread
        self._start_health_checker()
        
        # Auto-discover nodes
        self._discover_nodes()
        
        logger.info(f"Load balancer initialized with {len(self.nodes)} nodes")
    
    def _discover_nodes(self) -> None:
        """Auto-discover available compute nodes."""
        try:
            # Discover photonic devices
            photonic_devices = get_photonic_devices()
            for i, device in enumerate(photonic_devices):
                node_id = f"photonic_{device.device_id}"
                node = ComputeNode(
                    node_id=node_id,
                    device_type="photonic",
                    capacity=device.wavelengths // 8,  # Estimate based on wavelengths
                    weight=2.0,  # Higher weight for photonic devices
                    metadata={
                        "device_id": device.device_id,
                        "wavelengths": device.wavelengths,
                        "max_power": device.max_optical_power
                    }
                )
                self.add_node(node)
            
            # Add default GPU node
            gpu_node = ComputeNode(
                node_id="gpu_0",
                device_type="gpu",
                capacity=16,
                weight=1.5,
                metadata={"device_type": "cuda"}
            )
            self.add_node(gpu_node)
            
            # Add CPU fallback node
            cpu_node = ComputeNode(
                node_id="cpu_0",
                device_type="cpu",
                capacity=4,
                weight=0.5,
                metadata={"device_type": "cpu"}
            )
            self.add_node(cpu_node)
            
        except Exception as e:
            logger.error(f"Node discovery failed: {e}")
    
    def add_node(self, node: ComputeNode) -> None:
        """Add compute node to load balancer."""
        with self._lock:
            self.nodes[node.node_id] = node
            self.hash_ring.add_node(node.node_id)
            logger.info(f"Added node: {node.node_id} ({node.device_type})")
    
    def remove_node(self, node_id: str) -> None:
        """Remove compute node from load balancer."""
        with self._lock:
            if node_id in self.nodes:
                del self.nodes[node_id]
                self.hash_ring.remove_node(node_id)
                logger.info(f"Removed node: {node_id}")
    
    def select_node(
        self, 
        request_key: Optional[str] = None,
        session_id: Optional[str] = None,
        required_capacity: int = 1
    ) -> Optional[ComputeNode]:
        """Select optimal node for request."""
        with self._lock:
            available_nodes = [
                node for node in self.nodes.values() 
                if node.is_available and node.current_load + required_capacity <= node.capacity
            ]
            
            if not available_nodes:
                logger.warning("No available nodes for request")
                return None
            
            # Check sticky session first
            if self.session_manager and session_id:
                session_node_id = self.session_manager.get_session_node(session_id)
                if session_node_id and session_node_id in self.nodes:
                    session_node = self.nodes[session_node_id]
                    if session_node.is_available:
                        return session_node
            
            # Apply load balancing strategy
            if self.config.strategy == LoadBalancingStrategy.ROUND_ROBIN:
                return self._round_robin_select(available_nodes)
            elif self.config.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
                return self._least_connections_select(available_nodes)
            elif self.config.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
                return self._weighted_round_robin_select(available_nodes)
            elif self.config.strategy == LoadBalancingStrategy.RESOURCE_BASED:
                return self._resource_based_select(available_nodes)
            elif self.config.strategy == LoadBalancingStrategy.PERFORMANCE_BASED:
                return self._performance_based_select(available_nodes)
            elif self.config.strategy == LoadBalancingStrategy.CONSISTENT_HASH:
                return self._consistent_hash_select(available_nodes, request_key or "")
            else:
                return self._performance_based_select(available_nodes)  # Default
    
    def _round_robin_select(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Round-robin selection."""
        if not nodes:
            return None
        
        node = nodes[self.round_robin_index % len(nodes)]
        self.round_robin_index += 1
        return node
    
    def _least_connections_select(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select node with least connections."""
        return min(nodes, key=lambda n: n.current_load)
    
    def _weighted_round_robin_select(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Weighted round-robin selection."""
        total_weight = sum(node.weight for node in nodes)
        if total_weight == 0:
            return self._round_robin_select(nodes)
        
        # Use random selection based on weights
        rand = random.uniform(0, total_weight)
        current = 0
        
        for node in nodes:
            current += node.weight
            if rand <= current:
                return node
        
        return nodes[-1]  # Fallback
    
    def _resource_based_select(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select based on resource utilization."""
        return min(nodes, key=lambda n: n.utilization)
    
    def _performance_based_select(self, nodes: List[ComputeNode]) -> ComputeNode:
        """Select based on performance metrics."""
        def performance_score(node: ComputeNode) -> float:
            # Lower is better
            latency_score = node.avg_latency_ms / 1000.0  # Convert to seconds
            utilization_score = node.utilization
            reliability_score = 1.0 - node.success_rate
            
            # Weighted combination
            return latency_score * 0.4 + utilization_score * 0.4 + reliability_score * 0.2
        
        return min(nodes, key=performance_score)
    
    def _consistent_hash_select(self, nodes: List[ComputeNode], key: str) -> Optional[ComputeNode]:
        """Consistent hash selection."""
        node_id = self.hash_ring.get_node(key)
        if node_id and node_id in self.nodes:
            node = self.nodes[node_id]
            if node in nodes:  # Check if node is available
                return node
        
        # Fallback to least connections
        return self._least_connections_select(nodes)
    
    def execute_request(
        self,
        request_func: Callable,
        request_args: Tuple = (),
        request_kwargs: Optional[Dict[str, Any]] = None,
        request_key: Optional[str] = None,
        session_id: Optional[str] = None,
        required_capacity: int = 1,
        priority: int = 1
    ) -> Any:
        """Execute request with load balancing."""
        if request_kwargs is None:
            request_kwargs = {}
        
        start_time = time.time()
        
        with self._lock:
            self.total_requests += 1
        
        for attempt in range(self.config.max_retries + 1):
            try:
                # Select node
                node = self.select_node(request_key, session_id, required_capacity)
                if not node:
                    raise RuntimeError("No available nodes")
                
                # Reserve capacity
                with self._lock:
                    node.current_load += required_capacity
                
                try:
                    # Execute request
                    logger.debug(f"Executing request on node {node.node_id} (attempt {attempt + 1})")
                    
                    # Add node info to request
                    request_kwargs['_node_info'] = {
                        'node_id': node.node_id,
                        'device_type': node.device_type
                    }
                    
                    result = request_func(*request_args, **request_kwargs)
                    
                    # Update node metrics
                    end_time = time.time()
                    latency_ms = (end_time - start_time) * 1000
                    
                    with self._lock:
                        node.total_requests += 1
                        alpha = 0.1  # Learning rate for moving average
                        node.avg_latency_ms = (
                            (1 - alpha) * node.avg_latency_ms + alpha * latency_ms
                        )
                        node.success_rate = (
                            (1 - alpha) * node.success_rate + alpha * 1.0
                        )
                        self.successful_requests += 1
                    
                    # Create sticky session
                    if self.session_manager and session_id:
                        self.session_manager.create_session(session_id, node.node_id)
                    
                    logger.debug(f"Request completed successfully on {node.node_id}: {latency_ms:.2f}ms")
                    return result
                    
                finally:
                    # Release capacity
                    with self._lock:
                        node.current_load = max(0, node.current_load - required_capacity)
                        
            except Exception as e:
                logger.warning(f"Request failed on attempt {attempt + 1}: {e}")
                
                if node:
                    with self._lock:
                        node.failed_requests += 1
                        node.success_rate = (
                            0.9 * node.success_rate + 0.1 * 0.0
                        )
                
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_delay * (2 ** attempt))  # Exponential backoff
                else:
                    with self._lock:
                        self.failed_requests += 1
                    raise
        
        raise RuntimeError(f"Request failed after {self.config.max_retries + 1} attempts")
    
    def _start_health_checker(self) -> None:
        """Start health check background thread."""
        def health_check_loop():
            while True:
                try:
                    time.sleep(self.config.health_check_interval)
                    self._perform_health_checks()
                except Exception as e:
                    logger.error(f"Health check error: {e}")
        
        health_thread = threading.Thread(target=health_check_loop, daemon=True)
        health_thread.start()
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all nodes."""
        with self._lock:
            for node in self.nodes.values():
                try:
                    # Update node status based on metrics
                    if node.success_rate < 0.5:
                        node.status = NodeStatus.UNHEALTHY
                    elif node.utilization > 0.9:
                        node.status = NodeStatus.OVERLOADED
                    elif node.utilization > 0.7 or node.success_rate < 0.8:
                        node.status = NodeStatus.DEGRADED
                    else:
                        node.status = NodeStatus.HEALTHY
                    
                    node.last_health_check = time.time()
                    
                except Exception as e:
                    logger.error(f"Health check failed for node {node.node_id}: {e}")
                    node.status = NodeStatus.UNHEALTHY
    
    def get_cluster_stats(self) -> Dict[str, Any]:
        """Get cluster statistics."""
        with self._lock:
            node_stats = {}
            for node_id, node in self.nodes.items():
                node_stats[node_id] = {
                    'device_type': node.device_type,
                    'status': node.status.value,
                    'current_load': node.current_load,
                    'capacity': node.capacity,
                    'utilization': node.utilization,
                    'avg_latency_ms': node.avg_latency_ms,
                    'success_rate': node.success_rate,
                    'total_requests': node.total_requests,
                    'failed_requests': node.failed_requests
                }
            
            return {
                'total_nodes': len(self.nodes),
                'healthy_nodes': sum(1 for n in self.nodes.values() if n.status == NodeStatus.HEALTHY),
                'total_requests': self.total_requests,
                'successful_requests': self.successful_requests,
                'failed_requests': self.failed_requests,
                'success_rate': self.successful_requests / max(self.total_requests, 1),
                'strategy': self.config.strategy.value,
                'nodes': node_stats
            }


# Global load balancer instance
_load_balancer: Optional[LoadBalancer] = None


def get_load_balancer() -> LoadBalancer:
    """Get global load balancer instance."""
    global _load_balancer
    if _load_balancer is None:
        _load_balancer = LoadBalancer()
    return _load_balancer


def execute_balanced_request(
    request_func: Callable,
    request_args: Tuple = (),
    request_kwargs: Optional[Dict[str, Any]] = None,
    **load_balancer_kwargs
) -> Any:
    """Execute request with load balancing (convenience function)."""
    lb = get_load_balancer()
    return lb.execute_request(
        request_func, request_args, request_kwargs, **load_balancer_kwargs
    )