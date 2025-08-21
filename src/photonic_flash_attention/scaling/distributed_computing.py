"""Distributed Computing and Scaling Infrastructure.

Implements advanced distributed computing capabilities for photonic flash attention
including model parallelism, data parallelism, pipeline parallelism, and
autonomous scaling based on workload characteristics.
"""

import asyncio
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import rpc
from torch.distributed.pipeline.sync import Pipe
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from collections import deque, defaultdict
import threading
import time
import json
import subprocess
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import queue
import socket

from ..config import get_config
from ..utils.logging import get_logger
from ..utils.exceptions import PhotonicComputationError
from ..core.hybrid_router import HybridFlashAttention


@dataclass
class ComputeNode:
    """Information about a compute node in the distributed system."""
    node_id: str
    hostname: str
    port: int
    device_type: str  # 'gpu', 'photonic', 'hybrid'
    capabilities: Dict[str, Any]
    current_load: float = 0.0
    max_capacity: int = 1
    status: str = 'available'  # 'available', 'busy', 'failed', 'maintenance'
    last_heartbeat: float = field(default_factory=time.time)
    performance_history: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DistributedTask:
    """Represents a distributed computation task."""
    task_id: str
    task_type: str
    input_data: Any
    metadata: Dict[str, Any]
    created_time: float = field(default_factory=time.time)
    assigned_nodes: List[str] = field(default_factory=list)
    status: str = 'pending'  # 'pending', 'running', 'completed', 'failed'
    result: Optional[Any] = None
    error: Optional[str] = None
    execution_time: Optional[float] = None
    priority: int = 1


class DistributedWorkloadBalancer:
    """
    Intelligent workload balancer for distributed photonic flash attention.
    
    Features:
    - Dynamic load balancing based on node capabilities
    - Workload-aware task assignment
    - Fault tolerance and automatic failover
    - Performance-based scheduling
    - Heterogeneous device support (GPU, photonic, hybrid)
    """
    
    def __init__(self, master_node: bool = False):
        self.master_node = master_node
        self.node_id = self._generate_node_id()
        
        self.logger = get_logger(f"{self.__class__.__name__}({self.node_id})")
        self.config = get_config()
        
        # Node registry
        self.compute_nodes: Dict[str, ComputeNode] = {}
        self.local_node: Optional[ComputeNode] = None
        
        # Task management
        self.pending_tasks: queue.PriorityQueue = queue.PriorityQueue()
        self.running_tasks: Dict[str, DistributedTask] = {}
        self.completed_tasks: deque = deque(maxlen=1000)
        
        # Load balancing
        self.load_balancing_strategy = 'performance_aware'  # 'round_robin', 'least_loaded', 'performance_aware'
        self.task_assignment_history = defaultdict(list)
        
        # Communication
        self.communication_backend = 'nccl' if torch.cuda.is_available() else 'gloo'
        self.rpc_backend = 'tensorpipe'
        
        # Performance tracking
        self.performance_stats = defaultdict(list)
        self.cluster_utilization_history = deque(maxlen=1000)
        
        # Background processes
        self._heartbeat_active = False
        self._heartbeat_thread = None
        self._load_balancer_active = False
        self._load_balancer_thread = None
        
        self._lock = threading.RLock()
        
        if master_node:
            self._initialize_master_node()
        else:
            self._initialize_worker_node()
    
    def _generate_node_id(self) -> str:
        """Generate unique node identifier."""
        hostname = socket.gethostname()
        timestamp = int(time.time() * 1000)
        return f"{hostname}_{timestamp}"
    
    def _initialize_master_node(self) -> None:
        """Initialize master node capabilities."""
        self.logger.info("Initializing as master node")
        
        # Register local node
        self.local_node = ComputeNode(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            port=self._find_free_port(),
            device_type=self._detect_device_type(),
            capabilities=self._detect_capabilities(),
            max_capacity=self._detect_max_capacity(),
        )
        
        self.compute_nodes[self.node_id] = self.local_node
        
        # Start master services
        self._start_heartbeat_monitor()
        self._start_load_balancer()
        
        self.logger.info(f"Master node initialized: {self.local_node.device_type} with {self.local_node.max_capacity} capacity")
    
    def _initialize_worker_node(self) -> None:
        """Initialize worker node capabilities."""
        self.logger.info("Initializing as worker node")
        
        # Register local node
        self.local_node = ComputeNode(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            port=self._find_free_port(),
            device_type=self._detect_device_type(),
            capabilities=self._detect_capabilities(),
            max_capacity=self._detect_max_capacity(),
        )
        
        self.logger.info(f"Worker node initialized: {self.local_node.device_type} with {self.local_node.max_capacity} capacity")
    
    def _find_free_port(self) -> int:
        """Find an available port for communication."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _detect_device_type(self) -> str:
        """Detect available device types on this node."""
        device_types = []
        
        if torch.cuda.is_available():
            device_types.append('gpu')
        
        # Check for photonic hardware (simulated)
        try:
            from ..photonic.hardware.detection import get_best_photonic_device
            if get_best_photonic_device() is not None:
                device_types.append('photonic')
        except ImportError:
            pass
        
        if len(device_types) > 1:
            return 'hybrid'
        elif device_types:
            return device_types[0]
        else:
            return 'cpu'
    
    def _detect_capabilities(self) -> Dict[str, Any]:
        """Detect node capabilities and specifications."""
        capabilities = {
            'cpu_cores': mp.cpu_count(),
            'memory_gb': self._get_memory_info(),
            'devices': [],
        }
        
        # GPU capabilities
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                capabilities['devices'].append({
                    'type': 'gpu',
                    'index': i,
                    'name': props.name,
                    'memory_gb': props.total_memory / (1024**3),
                    'compute_capability': f"{props.major}.{props.minor}",
                })
        
        # Photonic capabilities (simulated)
        capabilities['photonic_wavelengths'] = 80
        capabilities['photonic_power_budget'] = 10.0  # mW
        
        return capabilities
    
    def _get_memory_info(self) -> float:
        """Get system memory information in GB."""
        try:
            import psutil
            return psutil.virtual_memory().total / (1024**3)
        except ImportError:
            return 16.0  # Default assumption
    
    def _detect_max_capacity(self) -> int:
        """Detect maximum concurrent task capacity."""
        base_capacity = 1
        
        # Scale based on available devices
        if torch.cuda.is_available():
            base_capacity += torch.cuda.device_count()
        
        # Scale based on CPU cores
        base_capacity += max(1, mp.cpu_count() // 4)
        
        return min(base_capacity, 16)  # Cap at reasonable limit
    
    def register_node(self, node: ComputeNode) -> bool:
        """Register a new compute node in the cluster."""
        with self._lock:
            if node.node_id in self.compute_nodes:
                self.logger.warning(f"Node {node.node_id} already registered")
                return False
            
            self.compute_nodes[node.node_id] = node
            self.logger.info(f"Registered node {node.node_id}: {node.device_type} ({node.max_capacity} capacity)")
            
            return True
    
    def submit_task(self, task: DistributedTask) -> str:
        """Submit a task for distributed execution."""
        with self._lock:
            # Add to pending queue with priority
            priority = -task.priority  # Negative for max-heap behavior
            self.pending_tasks.put((priority, task.created_time, task))
            
            self.logger.debug(f"Submitted task {task.task_id}: {task.task_type}")
            
            return task.task_id
    
    def _start_heartbeat_monitor(self) -> None:
        """Start heartbeat monitoring for cluster health."""
        if not self.master_node:
            return
        
        with self._lock:
            if self._heartbeat_active:
                return
            
            self._heartbeat_active = True
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_monitor_loop,
                daemon=True
            )
            self._heartbeat_thread.start()
            
            self.logger.info("Heartbeat monitor started")
    
    def _heartbeat_monitor_loop(self) -> None:
        """Monitor node heartbeats and health."""
        while self._heartbeat_active:
            try:
                current_time = time.time()
                
                with self._lock:
                    failed_nodes = []
                    
                    for node_id, node in self.compute_nodes.items():
                        if node_id == self.node_id:  # Skip local node
                            continue
                        
                        # Check if node has missed heartbeats
                        if current_time - node.last_heartbeat > 30:  # 30 second timeout
                            if node.status != 'failed':
                                self.logger.warning(f"Node {node_id} failed heartbeat check")
                                node.status = 'failed'
                                failed_nodes.append(node_id)
                    
                    # Handle failed nodes
                    for node_id in failed_nodes:
                        self._handle_node_failure(node_id)
                
                time.sleep(10)  # Check every 10 seconds
                
            except Exception as e:
                self.logger.error(f"Heartbeat monitor error: {e}")
                time.sleep(10)
    
    def _handle_node_failure(self, node_id: str) -> None:
        """Handle node failure by reassigning tasks."""
        self.logger.warning(f"Handling failure of node {node_id}")
        
        # Find tasks assigned to failed node
        tasks_to_reassign = []
        
        for task_id, task in self.running_tasks.items():
            if node_id in task.assigned_nodes:
                tasks_to_reassign.append(task)
        
        # Reassign tasks to healthy nodes
        for task in tasks_to_reassign:
            self.logger.info(f"Reassigning task {task.task_id} from failed node {node_id}")
            task.assigned_nodes = [n for n in task.assigned_nodes if n != node_id]
            task.status = 'pending'
            self.submit_task(task)
    
    def _start_load_balancer(self) -> None:
        """Start load balancer for task assignment."""
        if not self.master_node:
            return
        
        with self._lock:
            if self._load_balancer_active:
                return
            
            self._load_balancer_active = True
            self._load_balancer_thread = threading.Thread(
                target=self._load_balancer_loop,
                daemon=True
            )
            self._load_balancer_thread.start()
            
            self.logger.info("Load balancer started")
    
    def _load_balancer_loop(self) -> None:
        """Main load balancer loop for task assignment."""
        while self._load_balancer_active:
            try:
                # Process pending tasks
                while not self.pending_tasks.empty():
                    try:
                        _, _, task = self.pending_tasks.get_nowait()
                        assigned_nodes = self._assign_task_to_nodes(task)
                        
                        if assigned_nodes:
                            task.assigned_nodes = assigned_nodes
                            task.status = 'running'
                            self.running_tasks[task.task_id] = task
                            
                            # Execute task asynchronously
                            self._execute_task_async(task)
                        else:
                            # No available nodes, requeue with lower priority
                            task.priority = max(1, task.priority - 1)
                            self.submit_task(task)
                    
                    except queue.Empty:
                        break
                
                # Update cluster utilization
                self._update_cluster_utilization()
                
                time.sleep(1)  # Check every second
                
            except Exception as e:
                self.logger.error(f"Load balancer error: {e}")
                time.sleep(1)
    
    def _assign_task_to_nodes(self, task: DistributedTask) -> List[str]:
        """Assign task to optimal nodes based on strategy."""
        with self._lock:
            available_nodes = self._get_available_nodes()
            
            if not available_nodes:
                return []
            
            if self.load_balancing_strategy == 'round_robin':
                return self._round_robin_assignment(task, available_nodes)
            elif self.load_balancing_strategy == 'least_loaded':
                return self._least_loaded_assignment(task, available_nodes)
            elif self.load_balancing_strategy == 'performance_aware':
                return self._performance_aware_assignment(task, available_nodes)
            else:
                return [available_nodes[0].node_id]  # Default to first available
    
    def _get_available_nodes(self) -> List[ComputeNode]:
        """Get list of available nodes for task assignment."""
        available = []
        
        for node in self.compute_nodes.values():
            if (node.status == 'available' and 
                node.current_load < node.max_capacity):
                available.append(node)
        
        return available
    
    def _round_robin_assignment(self, task: DistributedTask, available_nodes: List[ComputeNode]) -> List[str]:
        """Round-robin task assignment."""
        if not available_nodes:
            return []
        
        # Simple round-robin based on task count
        total_assignments = sum(len(assignments) for assignments in self.task_assignment_history.values())
        selected_node = available_nodes[total_assignments % len(available_nodes)]
        
        return [selected_node.node_id]
    
    def _least_loaded_assignment(self, task: DistributedTask, available_nodes: List[ComputeNode]) -> List[str]:
        """Assign to least loaded node."""
        if not available_nodes:
            return []
        
        # Sort by current load
        sorted_nodes = sorted(available_nodes, key=lambda n: n.current_load)
        selected_node = sorted_nodes[0]
        
        return [selected_node.node_id]
    
    def _performance_aware_assignment(self, task: DistributedTask, available_nodes: List[ComputeNode]) -> List[str]:
        """Assign based on node performance characteristics."""
        if not available_nodes:
            return []
        
        # Score nodes based on suitability for task
        node_scores = []
        
        for node in available_nodes:
            score = self._calculate_node_score(task, node)
            node_scores.append((score, node))
        
        # Sort by score (higher is better)
        node_scores.sort(key=lambda x: x[0], reverse=True)
        
        # Select best node(s)
        best_node = node_scores[0][1]
        
        # For some tasks, use multiple nodes
        if self._should_use_multiple_nodes(task) and len(node_scores) > 1:
            second_best = node_scores[1][1]
            return [best_node.node_id, second_best.node_id]
        else:
            return [best_node.node_id]
    
    def _calculate_node_score(self, task: DistributedTask, node: ComputeNode) -> float:
        """Calculate suitability score for assigning task to node."""
        score = 0.0
        
        # Device type matching
        task_metadata = task.metadata
        preferred_device = task_metadata.get('preferred_device', 'gpu')
        
        if node.device_type == preferred_device:
            score += 10.0
        elif node.device_type == 'hybrid':
            score += 8.0
        elif preferred_device == 'auto':
            score += 5.0
        
        # Load factor (lower load is better)
        load_factor = node.current_load / node.max_capacity
        score += (1.0 - load_factor) * 5.0
        
        # Historical performance
        if node.performance_history:
            avg_performance = np.mean([p.get('success_rate', 0.5) for p in node.performance_history[-10:]])
            score += avg_performance * 5.0
        
        # Capacity availability
        available_capacity = node.max_capacity - node.current_load
        score += min(available_capacity, 5) * 1.0
        
        # Workload characteristics matching
        if 'seq_length' in task_metadata:
            seq_length = task_metadata['seq_length']
            if seq_length > 1024 and node.device_type in ['photonic', 'hybrid']:
                score += 3.0
            elif seq_length < 512 and node.device_type in ['gpu', 'hybrid']:
                score += 2.0
        
        return score
    
    def _should_use_multiple_nodes(self, task: DistributedTask) -> bool:
        """Determine if task should use multiple nodes."""
        # Use multiple nodes for large tasks
        metadata = task.metadata
        
        if 'batch_size' in metadata and metadata['batch_size'] >= 16:
            return True
        
        if 'seq_length' in metadata and metadata['seq_length'] >= 4096:
            return True
        
        if task.task_type in ['training', 'large_inference']:
            return True
        
        return False
    
    def _execute_task_async(self, task: DistributedTask) -> None:
        """Execute task asynchronously on assigned nodes."""
        def execute_wrapper():
            try:
                start_time = time.time()
                
                # Update node load
                for node_id in task.assigned_nodes:
                    if node_id in self.compute_nodes:
                        self.compute_nodes[node_id].current_load += 1
                
                # Execute task
                result = self._execute_distributed_task(task)
                
                # Record completion
                end_time = time.time()
                task.execution_time = end_time - start_time
                task.result = result
                task.status = 'completed'
                
                # Update performance statistics
                self._record_task_completion(task, True)
                
            except Exception as e:
                self.logger.error(f"Task {task.task_id} failed: {e}")
                task.error = str(e)
                task.status = 'failed'
                self._record_task_completion(task, False)
            
            finally:
                # Update node load
                for node_id in task.assigned_nodes:
                    if node_id in self.compute_nodes:
                        self.compute_nodes[node_id].current_load = max(0, self.compute_nodes[node_id].current_load - 1)
                
                # Move to completed tasks
                with self._lock:
                    if task.task_id in self.running_tasks:
                        del self.running_tasks[task.task_id]
                    self.completed_tasks.append(task)
        
        # Execute in thread pool
        thread = threading.Thread(target=execute_wrapper, daemon=True)
        thread.start()
    
    def _execute_distributed_task(self, task: DistributedTask) -> Any:
        """Execute task across distributed nodes."""
        if len(task.assigned_nodes) == 1:
            # Single node execution
            return self._execute_single_node_task(task)
        else:
            # Multi-node execution
            return self._execute_multi_node_task(task)
    
    def _execute_single_node_task(self, task: DistributedTask) -> Any:
        """Execute task on a single node."""
        node_id = task.assigned_nodes[0]
        
        if node_id == self.node_id:
            # Execute locally
            return self._execute_local_task(task)
        else:
            # Execute remotely (simplified)
            return self._execute_remote_task(task, node_id)
    
    def _execute_local_task(self, task: DistributedTask) -> Any:
        """Execute task on local node."""
        if task.task_type == 'attention':
            return self._execute_attention_task(task)
        elif task.task_type == 'training':
            return self._execute_training_task(task)
        else:
            raise ValueError(f"Unknown task type: {task.task_type}")
    
    def _execute_attention_task(self, task: DistributedTask) -> Any:
        """Execute attention computation task."""
        input_data = task.input_data
        metadata = task.metadata
        
        # Create attention module
        attention = HybridFlashAttention(
            embed_dim=metadata.get('embed_dim', 768),
            num_heads=metadata.get('num_heads', 12),
            enable_scaling=True,
        )
        
        # Execute attention
        query = input_data.get('query')
        key = input_data.get('key')
        value = input_data.get('value')
        attention_mask = input_data.get('attention_mask')
        
        with torch.no_grad():
            output = attention(query, key, value, attention_mask)
        
        return {
            'output': output,
            'metadata': {
                'device_used': attention.last_device_used if hasattr(attention, 'last_device_used') else 'unknown',
                'execution_time_ms': task.execution_time * 1000 if task.execution_time else 0,
            }
        }
    
    def _execute_training_task(self, task: DistributedTask) -> Any:
        """Execute training task (simplified implementation)."""
        # Simplified training simulation
        time.sleep(np.random.uniform(0.1, 0.5))  # Simulate training time
        
        return {
            'loss': np.random.uniform(0.1, 1.0),
            'accuracy': np.random.uniform(0.8, 0.95),
            'training_steps': task.metadata.get('steps', 100),
        }
    
    def _execute_remote_task(self, task: DistributedTask, node_id: str) -> Any:
        """Execute task on remote node (simplified implementation)."""
        # In a real implementation, this would use RPC or message passing
        # For now, simulate remote execution
        time.sleep(np.random.uniform(0.05, 0.2))  # Network latency simulation
        
        return self._execute_local_task(task)
    
    def _execute_multi_node_task(self, task: DistributedTask) -> Any:
        """Execute task across multiple nodes."""
        # For multi-node tasks, implement data parallelism or model parallelism
        # This is a simplified implementation
        
        node_results = []
        
        for node_id in task.assigned_nodes:
            # Create subtask for each node
            subtask = DistributedTask(
                task_id=f"{task.task_id}_sub_{node_id}",
                task_type=task.task_type,
                input_data=task.input_data,
                metadata=task.metadata.copy(),
            )
            
            # Execute subtask
            if node_id == self.node_id:
                result = self._execute_local_task(subtask)
            else:
                result = self._execute_remote_task(subtask, node_id)
            
            node_results.append(result)
        
        # Combine results (implementation depends on task type)
        return self._combine_multi_node_results(task, node_results)
    
    def _combine_multi_node_results(self, task: DistributedTask, results: List[Any]) -> Any:
        """Combine results from multiple nodes."""
        if task.task_type == 'attention':
            # For attention tasks, concatenate outputs
            combined_output = torch.cat([r['output'] for r in results], dim=0)
            return {
                'output': combined_output,
                'metadata': {
                    'nodes_used': len(results),
                    'combined_result': True,
                }
            }
        
        elif task.task_type == 'training':
            # For training tasks, average metrics
            avg_loss = np.mean([r['loss'] for r in results])
            avg_accuracy = np.mean([r['accuracy'] for r in results])
            
            return {
                'loss': avg_loss,
                'accuracy': avg_accuracy,
                'nodes_used': len(results),
            }
        
        else:
            # Default: return first result
            return results[0] if results else None
    
    def _record_task_completion(self, task: DistributedTask, success: bool) -> None:
        """Record task completion for performance tracking."""
        # Update node performance history
        for node_id in task.assigned_nodes:
            if node_id in self.compute_nodes:
                node = self.compute_nodes[node_id]
                
                performance_record = {
                    'timestamp': time.time(),
                    'task_type': task.task_type,
                    'execution_time': task.execution_time,
                    'success': success,
                    'success_rate': 1.0 if success else 0.0,
                }
                
                node.performance_history.append(performance_record)
                
                # Limit history size
                if len(node.performance_history) > 100:
                    node.performance_history = node.performance_history[-50:]
        
        # Update global performance stats
        self.performance_stats[task.task_type].append({
            'timestamp': time.time(),
            'execution_time': task.execution_time,
            'success': success,
            'nodes_used': len(task.assigned_nodes),
        })
    
    def _update_cluster_utilization(self) -> None:
        """Update cluster utilization metrics."""
        total_capacity = sum(node.max_capacity for node in self.compute_nodes.values())
        total_load = sum(node.current_load for node in self.compute_nodes.values())
        
        utilization = total_load / max(total_capacity, 1)
        
        self.cluster_utilization_history.append({
            'timestamp': time.time(),
            'utilization': utilization,
            'total_capacity': total_capacity,
            'total_load': total_load,
            'active_nodes': len([n for n in self.compute_nodes.values() if n.status == 'available']),
        })
    
    def get_cluster_status(self) -> Dict[str, Any]:
        """Get comprehensive cluster status."""
        with self._lock:
            node_stats = {}
            
            for node_id, node in self.compute_nodes.items():
                node_stats[node_id] = {
                    'device_type': node.device_type,
                    'status': node.status,
                    'current_load': node.current_load,
                    'max_capacity': node.max_capacity,
                    'utilization': node.current_load / node.max_capacity,
                    'capabilities': node.capabilities,
                    'last_heartbeat': node.last_heartbeat,
                }
            
            # Calculate cluster metrics
            total_capacity = sum(node.max_capacity for node in self.compute_nodes.values())
            total_load = sum(node.current_load for node in self.compute_nodes.values())
            
            return {
                'master_node': self.master_node,
                'local_node_id': self.node_id,
                'cluster_utilization': total_load / max(total_capacity, 1),
                'total_nodes': len(self.compute_nodes),
                'active_nodes': len([n for n in self.compute_nodes.values() if n.status == 'available']),
                'pending_tasks': self.pending_tasks.qsize(),
                'running_tasks': len(self.running_tasks),
                'completed_tasks': len(self.completed_tasks),
                'node_stats': node_stats,
                'performance_summary': self._get_performance_summary(),
            }
    
    def _get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary across all task types."""
        summary = {}
        
        for task_type, stats in self.performance_stats.items():
            recent_stats = [s for s in stats if time.time() - s['timestamp'] < 3600]  # Last hour
            
            if recent_stats:
                success_rate = sum(s['success'] for s in recent_stats) / len(recent_stats)
                avg_execution_time = np.mean([s['execution_time'] for s in recent_stats if s['execution_time']])
                
                summary[task_type] = {
                    'total_tasks': len(recent_stats),
                    'success_rate': success_rate,
                    'avg_execution_time': avg_execution_time,
                }
        
        return summary
    
    def shutdown(self) -> None:
        """Shutdown distributed workload balancer."""
        self.logger.info("Shutting down distributed workload balancer")
        
        # Stop background threads
        self._heartbeat_active = False
        self._load_balancer_active = False
        
        # Wait for threads to finish
        if self._heartbeat_thread and self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=5.0)
        
        if self._load_balancer_thread and self._load_balancer_thread.is_alive():
            self._load_balancer_thread.join(timeout=5.0)
        
        # Update node status
        if self.local_node:
            self.local_node.status = 'maintenance'
        
        self.logger.info("Distributed workload balancer shutdown complete")


class AutoScalingOrchestrator:
    """
    Autonomous scaling orchestrator for photonic flash attention clusters.
    
    Features:
    - Workload-based scaling decisions
    - Cost-aware resource provisioning
    - Performance-driven scaling policies
    - Multi-cloud resource management
    - Predictive scaling based on patterns
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        
        self.logger = get_logger(self.__class__.__name__)
        
        # Scaling configuration
        self.min_nodes = self.config.get('min_nodes', 1)
        self.max_nodes = self.config.get('max_nodes', 10)
        self.target_utilization = self.config.get('target_utilization', 0.7)
        self.scale_up_threshold = self.config.get('scale_up_threshold', 0.8)
        self.scale_down_threshold = self.config.get('scale_down_threshold', 0.3)
        
        # Scaling state
        self.current_nodes = 0
        self.scaling_history = deque(maxlen=1000)
        self.workload_predictions = deque(maxlen=100)
        
        # Cost tracking
        self.cost_model = {
            'gpu': {'hourly_cost': 1.5, 'startup_time': 60},
            'photonic': {'hourly_cost': 2.0, 'startup_time': 30},
            'hybrid': {'hourly_cost': 3.0, 'startup_time': 90},
        }
        
        # Workload balancer integration
        self.workload_balancer: Optional[DistributedWorkloadBalancer] = None
        
        # Background scaling
        self._scaling_active = False
        self._scaling_thread = None
        
        self._lock = threading.RLock()
    
    def set_workload_balancer(self, balancer: DistributedWorkloadBalancer) -> None:
        """Set the workload balancer for monitoring and scaling."""
        self.workload_balancer = balancer
        self.logger.info("Workload balancer connected to auto-scaler")
    
    def start_auto_scaling(self) -> None:
        """Start autonomous scaling monitoring and execution."""
        with self._lock:
            if self._scaling_active:
                return
            
            self._scaling_active = True
            self._scaling_thread = threading.Thread(
                target=self._auto_scaling_loop,
                daemon=True
            )
            self._scaling_thread.start()
            
            self.logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self) -> None:
        """Stop autonomous scaling."""
        with self._lock:
            if not self._scaling_active:
                return
            
            self._scaling_active = False
            
            if self._scaling_thread and self._scaling_thread.is_alive():
                self._scaling_thread.join(timeout=10.0)
            
            self.logger.info("Auto-scaling stopped")
    
    def _auto_scaling_loop(self) -> None:
        """Main auto-scaling monitoring and execution loop."""
        while self._scaling_active:
            try:
                # Collect metrics
                metrics = self._collect_scaling_metrics()
                
                if metrics:
                    # Make scaling decision
                    scaling_decision = self._make_scaling_decision(metrics)
                    
                    # Execute scaling if needed
                    if scaling_decision['action'] != 'none':
                        self._execute_scaling_action(scaling_decision)
                    
                    # Update predictions
                    self._update_workload_predictions(metrics)
                
                time.sleep(30)  # Check every 30 seconds
                
            except Exception as e:
                self.logger.error(f"Auto-scaling loop error: {e}")
                time.sleep(30)
    
    def _collect_scaling_metrics(self) -> Optional[Dict[str, Any]]:
        """Collect metrics for scaling decisions."""
        if not self.workload_balancer:
            return None
        
        cluster_status = self.workload_balancer.get_cluster_status()
        
        # Calculate key metrics
        utilization = cluster_status.get('cluster_utilization', 0.0)
        pending_tasks = cluster_status.get('pending_tasks', 0)
        running_tasks = cluster_status.get('running_tasks', 0)
        active_nodes = cluster_status.get('active_nodes', 0)
        
        # Performance metrics
        performance_summary = cluster_status.get('performance_summary', {})
        avg_success_rate = np.mean([p.get('success_rate', 0.5) for p in performance_summary.values()])
        
        return {
            'timestamp': time.time(),
            'utilization': utilization,
            'pending_tasks': pending_tasks,
            'running_tasks': running_tasks,
            'active_nodes': active_nodes,
            'success_rate': avg_success_rate,
            'queue_backlog': pending_tasks + running_tasks,
        }
    
    def _make_scaling_decision(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Make intelligent scaling decision based on metrics."""
        utilization = metrics['utilization']
        pending_tasks = metrics['pending_tasks']
        active_nodes = metrics['active_nodes']
        success_rate = metrics['success_rate']
        
        decision = {
            'action': 'none',
            'target_nodes': active_nodes,
            'reason': 'stable_utilization',
            'urgency': 'normal',
        }
        
        # Scale up conditions
        if utilization > self.scale_up_threshold:
            if active_nodes < self.max_nodes:
                scale_factor = 1 + (utilization - self.scale_up_threshold) / (1.0 - self.scale_up_threshold)
                target_nodes = min(self.max_nodes, int(active_nodes * scale_factor))
                
                decision.update({
                    'action': 'scale_up',
                    'target_nodes': target_nodes,
                    'reason': f'high_utilization_{utilization:.2f}',
                    'urgency': 'high' if utilization > 0.95 else 'normal',
                })
        
        # Scale up for queue backlog
        elif pending_tasks > active_nodes * 2:
            if active_nodes < self.max_nodes:
                target_nodes = min(self.max_nodes, active_nodes + max(1, pending_tasks // 4))
                
                decision.update({
                    'action': 'scale_up',
                    'target_nodes': target_nodes,
                    'reason': f'queue_backlog_{pending_tasks}',
                    'urgency': 'high' if pending_tasks > active_nodes * 5 else 'normal',
                })
        
        # Scale up for poor performance
        elif success_rate < 0.8 and active_nodes < self.max_nodes:
            target_nodes = min(self.max_nodes, active_nodes + 1)
            
            decision.update({
                'action': 'scale_up',
                'target_nodes': target_nodes,
                'reason': f'low_success_rate_{success_rate:.2f}',
                'urgency': 'normal',
            })
        
        # Scale down conditions
        elif (utilization < self.scale_down_threshold and 
              pending_tasks == 0 and 
              active_nodes > self.min_nodes):
            
            # Conservative scale down
            target_nodes = max(self.min_nodes, active_nodes - 1)
            
            decision.update({
                'action': 'scale_down',
                'target_nodes': target_nodes,
                'reason': f'low_utilization_{utilization:.2f}',
                'urgency': 'low',
            })
        
        # Add predictive scaling
        predicted_load = self._predict_future_load()
        if predicted_load > utilization * 1.5 and decision['action'] == 'none':
            if active_nodes < self.max_nodes:
                decision.update({
                    'action': 'scale_up',
                    'target_nodes': min(self.max_nodes, active_nodes + 1),
                    'reason': f'predicted_load_{predicted_load:.2f}',
                    'urgency': 'normal',
                })
        
        return decision
    
    def _predict_future_load(self) -> float:
        """Predict future workload based on historical patterns."""
        if len(self.workload_predictions) < 5:
            return 0.5  # Default prediction
        
        recent_utilizations = [p['utilization'] for p in list(self.workload_predictions)[-10:]]
        
        # Simple trend-based prediction
        if len(recent_utilizations) >= 3:
            trend = np.polyfit(range(len(recent_utilizations)), recent_utilizations, 1)[0]
            predicted = recent_utilizations[-1] + trend * 3  # Predict 3 steps ahead
            return max(0.0, min(1.0, predicted))
        
        return np.mean(recent_utilizations)
    
    def _execute_scaling_action(self, decision: Dict[str, Any]) -> None:
        """Execute scaling action based on decision."""
        action = decision['action']
        target_nodes = decision['target_nodes']
        reason = decision['reason']
        
        self.logger.info(f"Executing scaling action: {action} to {target_nodes} nodes (reason: {reason})")
        
        if action == 'scale_up':
            self._scale_up_cluster(target_nodes, decision)
        elif action == 'scale_down':
            self._scale_down_cluster(target_nodes, decision)
        
        # Record scaling action
        scaling_record = {
            'timestamp': time.time(),
            'action': action,
            'target_nodes': target_nodes,
            'reason': reason,
            'urgency': decision.get('urgency', 'normal'),
        }
        
        self.scaling_history.append(scaling_record)
    
    def _scale_up_cluster(self, target_nodes: int, decision: Dict[str, Any]) -> None:
        """Scale up the cluster by adding nodes."""
        current_nodes = len(self.workload_balancer.compute_nodes) if self.workload_balancer else 0
        nodes_to_add = target_nodes - current_nodes
        
        if nodes_to_add <= 0:
            return
        
        self.logger.info(f"Scaling up: adding {nodes_to_add} nodes")
        
        # Determine node types to add
        node_types = self._select_node_types_for_scaling(nodes_to_add, decision)
        
        # Add nodes (simulated)
        for i, node_type in enumerate(node_types):
            self._provision_new_node(node_type, f"auto_scaled_{int(time.time())}_{i}")
    
    def _scale_down_cluster(self, target_nodes: int, decision: Dict[str, Any]) -> None:
        """Scale down the cluster by removing nodes."""
        if not self.workload_balancer:
            return
        
        current_nodes = len(self.workload_balancer.compute_nodes)
        nodes_to_remove = current_nodes - target_nodes
        
        if nodes_to_remove <= 0:
            return
        
        self.logger.info(f"Scaling down: removing {nodes_to_remove} nodes")
        
        # Select nodes to remove (least loaded, non-master)
        nodes_to_remove_list = self._select_nodes_for_removal(nodes_to_remove)
        
        # Remove nodes (simulated)
        for node_id in nodes_to_remove_list:
            self._decommission_node(node_id)
    
    def _select_node_types_for_scaling(self, count: int, decision: Dict[str, Any]) -> List[str]:
        """Select optimal node types for scaling up."""
        # Simple strategy: prefer GPU nodes for general workloads
        # In practice, this would consider workload characteristics
        
        node_types = []
        
        for i in range(count):
            if 'queue_backlog' in decision['reason']:
                # For queue backlogs, prefer fast GPU nodes
                node_types.append('gpu')
            elif 'high_utilization' in decision['reason']:
                # For high utilization, prefer hybrid nodes
                node_types.append('hybrid')
            else:
                # Default to GPU
                node_types.append('gpu')
        
        return node_types
    
    def _select_nodes_for_removal(self, count: int) -> List[str]:
        """Select nodes for removal during scale down."""
        if not self.workload_balancer:
            return []
        
        # Get removable nodes (not master, low load)
        removable_nodes = []
        
        for node_id, node in self.workload_balancer.compute_nodes.items():
            if (node_id != self.workload_balancer.node_id and  # Not master
                node.current_load == 0 and  # No active load
                node.status == 'available'):  # Available status
                removable_nodes.append((node_id, node))
        
        # Sort by removal priority (least important first)
        removable_nodes.sort(key=lambda x: (x[1].current_load, -x[1].last_heartbeat))
        
        return [node_id for node_id, _ in removable_nodes[:count]]
    
    def _provision_new_node(self, node_type: str, node_id: str) -> None:
        """Provision a new compute node (simulated)."""
        self.logger.info(f"Provisioning new {node_type} node: {node_id}")
        
        # In a real implementation, this would:
        # 1. Launch cloud instances
        # 2. Install software
        # 3. Register with cluster
        
        # Simulate provisioning time
        startup_time = self.cost_model[node_type]['startup_time']
        
        def provision_async():
            time.sleep(startup_time)  # Simulate startup time
            
            # Create new node
            new_node = ComputeNode(
                node_id=node_id,
                hostname=f"auto-{node_type}-{int(time.time())}",
                port=self._find_free_port(),
                device_type=node_type,
                capabilities=self._get_default_capabilities(node_type),
                max_capacity=4 if node_type == 'hybrid' else 2,
            )
            
            # Register with balancer
            if self.workload_balancer:
                self.workload_balancer.register_node(new_node)
            
            self.logger.info(f"Node {node_id} provisioned and registered")
        
        # Provision asynchronously
        thread = threading.Thread(target=provision_async, daemon=True)
        thread.start()
    
    def _decommission_node(self, node_id: str) -> None:
        """Decommission a compute node (simulated)."""
        self.logger.info(f"Decommissioning node: {node_id}")
        
        if self.workload_balancer and node_id in self.workload_balancer.compute_nodes:
            # Mark node as maintenance
            self.workload_balancer.compute_nodes[node_id].status = 'maintenance'
            
            # Remove from cluster (simplified)
            del self.workload_balancer.compute_nodes[node_id]
            
            self.logger.info(f"Node {node_id} decommissioned")
    
    def _find_free_port(self) -> int:
        """Find an available port."""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', 0))
            s.listen(1)
            port = s.getsockname()[1]
        return port
    
    def _get_default_capabilities(self, node_type: str) -> Dict[str, Any]:
        """Get default capabilities for node type."""
        capabilities = {
            'cpu_cores': 8,
            'memory_gb': 32,
            'devices': [],
        }
        
        if node_type in ['gpu', 'hybrid']:
            capabilities['devices'].append({
                'type': 'gpu',
                'index': 0,
                'name': 'Simulated GPU',
                'memory_gb': 16,
                'compute_capability': '8.0',
            })
        
        if node_type in ['photonic', 'hybrid']:
            capabilities['photonic_wavelengths'] = 80
            capabilities['photonic_power_budget'] = 10.0
        
        return capabilities
    
    def _update_workload_predictions(self, metrics: Dict[str, Any]) -> None:
        """Update workload predictions with new metrics."""
        self.workload_predictions.append(metrics)
        
        # Periodically retrain prediction models (simplified)
        if len(self.workload_predictions) % 20 == 0:
            self._retrain_prediction_models()
    
    def _retrain_prediction_models(self) -> None:
        """Retrain workload prediction models (simplified)."""
        self.logger.debug("Retraining workload prediction models")
        # In a real implementation, this would use ML models
        # For now, just log the action
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get comprehensive auto-scaling status."""
        with self._lock:
            # Calculate costs (simplified)
            current_cost = 0.0
            if self.workload_balancer:
                for node in self.workload_balancer.compute_nodes.values():
                    node_cost = self.cost_model.get(node.device_type, {'hourly_cost': 1.0})['hourly_cost']
                    current_cost += node_cost
            
            # Recent scaling actions
            recent_actions = [action for action in self.scaling_history 
                            if time.time() - action['timestamp'] < 3600]  # Last hour
            
            return {
                'auto_scaling_active': self._scaling_active,
                'current_nodes': len(self.workload_balancer.compute_nodes) if self.workload_balancer else 0,
                'min_nodes': self.min_nodes,
                'max_nodes': self.max_nodes,
                'target_utilization': self.target_utilization,
                'current_hourly_cost': current_cost,
                'recent_scaling_actions': len(recent_actions),
                'scaling_history': list(self.scaling_history)[-10:],
                'workload_predictions': list(self.workload_predictions)[-5:],
                'cost_model': self.cost_model,
            }
    
    def shutdown(self) -> None:
        """Shutdown auto-scaling orchestrator."""
        self.stop_auto_scaling()
        self.logger.info("Auto-scaling orchestrator shutdown complete")


async def run_distributed_scaling_demo():
    """Run demonstration of distributed computing and auto-scaling."""
    print("🔌 DISTRIBUTED COMPUTING & AUTO-SCALING DEMO")
    print("=" * 60)
    
    # Initialize master node
    master = DistributedWorkloadBalancer(master_node=True)
    
    # Initialize auto-scaler
    auto_scaler = AutoScalingOrchestrator({
        'min_nodes': 1,
        'max_nodes': 5,
        'target_utilization': 0.7,
    })
    
    auto_scaler.set_workload_balancer(master)
    auto_scaler.start_auto_scaling()
    
    try:
        # Simulate worker nodes joining
        for i in range(2):
            worker_node = ComputeNode(
                node_id=f"worker_{i}",
                hostname=f"worker-{i}.cluster",
                port=8000 + i,
                device_type='gpu' if i % 2 == 0 else 'photonic',
                capabilities={'cpu_cores': 8, 'memory_gb': 32},
                max_capacity=4,
            )
            master.register_node(worker_node)
            print(f"✅ Registered worker node {worker_node.node_id}")
        
        # Submit various tasks
        print("\n📦 Submitting tasks...")
        
        for i in range(10):
            task = DistributedTask(
                task_id=f"task_{i}",
                task_type='attention',
                input_data={
                    'query': torch.randn(2, 512, 768),
                    'key': torch.randn(2, 512, 768),
                    'value': torch.randn(2, 512, 768),
                },
                metadata={
                    'embed_dim': 768,
                    'num_heads': 12,
                    'seq_length': 512,
                    'batch_size': 2,
                    'preferred_device': 'auto',
                },
                priority=np.random.randint(1, 5),
            )
            
            master.submit_task(task)
        
        # Monitor execution
        print("\n📈 Monitoring execution...")
        
        for round_num in range(5):
            await asyncio.sleep(2)
            
            status = master.get_cluster_status()
            scaling_status = auto_scaler.get_scaling_status()
            
            print(f"\nRound {round_num + 1}:")
            print(f"  Cluster utilization: {status['cluster_utilization']:.2f}")
            print(f"  Active nodes: {status['active_nodes']}")
            print(f"  Running tasks: {status['running_tasks']}")
            print(f"  Completed tasks: {status['completed_tasks']}")
            print(f"  Auto-scaling active: {scaling_status['auto_scaling_active']}")
            print(f"  Current cost: ${scaling_status['current_hourly_cost']:.2f}/hour")
        
        # Final status
        final_status = master.get_cluster_status()
        final_scaling_status = auto_scaler.get_scaling_status()
        
        print("\n🏆 Final Results:")
        print(f"  Total tasks completed: {final_status['completed_tasks']}")
        print(f"  Average cluster utilization: {final_status['cluster_utilization']:.2f}")
        print(f"  Scaling actions taken: {final_scaling_status['recent_scaling_actions']}")
        
        # Performance summary
        performance = final_status.get('performance_summary', {})
        if performance:
            print("\n  Performance Summary:")
            for task_type, stats in performance.items():
                print(f"    {task_type}: {stats['success_rate']:.1%} success, {stats['avg_execution_time']:.2f}s avg")
        
        return True
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        return False
        
    finally:
        # Cleanup
        auto_scaler.shutdown()
        master.shutdown()
        print("\n📝 Demo cleanup completed")


if __name__ == "__main__":
    success = asyncio.run(run_distributed_scaling_demo())
    exit(0 if success else 1)
