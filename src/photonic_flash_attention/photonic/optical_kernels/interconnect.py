"""
Photonic interconnect management for optical computing systems.

This module implements network-on-chip (NoC) functionality using silicon photonics,
including optical routing, switching, and communication protocols for distributed
photonic computing systems.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import threading
import time
import logging
from collections import defaultdict, deque

from ...utils.exceptions import PhotonicComputeError, HardwareNotAvailableError
from ...utils.validation import validate_optical_tensor
from ...config import get_config

logger = logging.getLogger(__name__)


class RoutingProtocol(Enum):
    """Photonic routing protocols."""
    CIRCUIT_SWITCHED = "circuit_switched"
    PACKET_SWITCHED = "packet_switched"
    WAVELENGTH_ROUTED = "wavelength_routed"
    HYBRID = "hybrid"


class TopologyType(Enum):
    """Network topology types."""
    MESH = "mesh"
    TORUS = "torus"
    FAT_TREE = "fat_tree"
    CROSSBAR = "crossbar"
    RING = "ring"
    HYPERCUBE = "hypercube"


@dataclass
class InterconnectConfig:
    """Configuration for photonic interconnect."""
    topology: TopologyType = TopologyType.MESH
    routing_protocol: RoutingProtocol = RoutingProtocol.WAVELENGTH_ROUTED
    n_nodes: int = 16
    n_wavelengths: int = 80
    switching_time: float = 1e-9  # nanosecond switching
    propagation_delay: float = 1e-12  # picosecond per meter
    link_bandwidth: float = 100e9  # 100 Gbps per wavelength
    buffer_size: int = 1024  # packets
    max_path_length: int = 8  # hops
    enable_adaptive_routing: bool = True
    congestion_threshold: float = 0.8
    wavelength_conversion: bool = False


@dataclass
class PhotonicPacket:
    """Photonic data packet for network transmission."""
    packet_id: int
    source_node: int
    dest_node: int
    wavelength: int
    data: torch.Tensor
    timestamp: float
    priority: int = 0
    path: List[int] = field(default_factory=list)
    hop_count: int = 0
    max_hops: int = 8


@dataclass
class OpticalLink:
    """Optical link between nodes."""
    src_node: int
    dst_node: int
    wavelengths: Set[int]
    bandwidth_per_wavelength: float
    propagation_delay: float
    insertion_loss: float = 0.1  # dB
    crosstalk_level: float = -40.0  # dB
    is_active: bool = True
    utilization: Dict[int, float] = field(default_factory=dict)  # per wavelength


class OpticalSwitch:
    """Optical switching element for routing."""
    
    def __init__(self, node_id: int, n_ports: int = 4, n_wavelengths: int = 80):
        self.node_id = node_id
        self.n_ports = n_ports
        self.n_wavelengths = n_wavelengths
        self.switching_matrix = torch.zeros(n_ports, n_ports, n_wavelengths)
        self.switching_time = 1e-9  # nanoseconds
        self.last_switch_time = 0.0
        self.crosstalk_matrix = self._generate_crosstalk_matrix()
        
    def _generate_crosstalk_matrix(self) -> torch.Tensor:
        """Generate crosstalk matrix for optical switching."""
        crosstalk = torch.full((self.n_ports, self.n_ports, self.n_wavelengths), -40.0)
        # No crosstalk on diagonal (self-connections)
        for i in range(self.n_ports):
            crosstalk[i, i, :] = 0.0
        return crosstalk
    
    def configure_path(self, input_port: int, output_port: int, 
                      wavelength: int) -> bool:
        """Configure optical path through switch."""
        current_time = time.time()
        
        # Check switching time constraint
        if current_time - self.last_switch_time < self.switching_time:
            return False
        
        # Check port and wavelength validity
        if (input_port >= self.n_ports or output_port >= self.n_ports or
            wavelength >= self.n_wavelengths):
            return False
        
        # Configure switching matrix
        self.switching_matrix[input_port, output_port, wavelength] = 1.0
        self.last_switch_time = current_time
        
        return True
    
    def clear_path(self, input_port: int, output_port: int, wavelength: int) -> None:
        """Clear optical path through switch."""
        if (input_port < self.n_ports and output_port < self.n_ports and
            wavelength < self.n_wavelengths):
            self.switching_matrix[input_port, output_port, wavelength] = 0.0
    
    def get_transmission_matrix(self) -> torch.Tensor:
        """Get current transmission matrix with crosstalk."""
        # Apply crosstalk effects
        transmission = self.switching_matrix.clone()
        
        for i in range(self.n_ports):
            for j in range(self.n_ports):
                for k in range(self.n_wavelengths):
                    if transmission[i, j, k] > 0:
                        # Add crosstalk from other channels
                        crosstalk_power = 0.0
                        for m in range(self.n_ports):
                            for n in range(self.n_ports):
                                if m != i or n != j:
                                    crosstalk_power += (
                                        transmission[m, n, k] * 
                                        10 ** (self.crosstalk_matrix[m, n, k] / 10)
                                    )
                        
                        # Apply crosstalk degradation
                        transmission[i, j, k] *= (1.0 - crosstalk_power * 0.01)
        
        return transmission


class RoutingTable:
    """Routing table for photonic network."""
    
    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.routes = {}  # (src, dst) -> [(next_hop, wavelength, cost)]
        self.link_states = {}  # link_id -> state info
        self._lock = threading.Lock()
        
    def add_route(self, src: int, dst: int, next_hop: int, 
                 wavelength: int, cost: float) -> None:
        """Add route to routing table."""
        with self._lock:
            key = (src, dst)
            if key not in self.routes:
                self.routes[key] = []
            
            route_entry = (next_hop, wavelength, cost)
            
            # Insert route sorted by cost
            routes_list = self.routes[key]
            inserted = False
            for i, (_, _, existing_cost) in enumerate(routes_list):
                if cost < existing_cost:
                    routes_list.insert(i, route_entry)
                    inserted = True
                    break
            
            if not inserted:
                routes_list.append(route_entry)
    
    def get_best_route(self, src: int, dst: int) -> Optional[Tuple[int, int, float]]:
        """Get best route for source-destination pair."""
        with self._lock:
            key = (src, dst)
            if key in self.routes and self.routes[key]:
                return self.routes[key][0]  # Return best route
            return None
    
    def update_link_state(self, link_id: str, utilization: float, 
                         latency: float) -> None:
        """Update link state information."""
        with self._lock:
            self.link_states[link_id] = {
                'utilization': utilization,
                'latency': latency,
                'timestamp': time.time()
            }


class PhotonicRouter:
    """Photonic router for network-on-chip."""
    
    def __init__(self, node_id: int, config: InterconnectConfig):
        self.node_id = node_id
        self.config = config
        self.routing_table = RoutingTable(config.n_nodes)
        self.optical_switch = OpticalSwitch(node_id, 
                                          n_wavelengths=config.n_wavelengths)
        self.packet_buffers = defaultdict(deque)  # per output port
        self.wavelength_allocator = set(range(config.n_wavelengths))
        self.active_paths = {}  # (src, dst) -> wavelength
        self._lock = threading.Lock()
        
    def route_packet(self, packet: PhotonicPacket) -> Optional[int]:
        """Route packet to next hop."""
        if packet.dest_node == self.node_id:
            return None  # Packet reached destination
        
        # Get best route from routing table
        route = self.routing_table.get_best_route(self.node_id, packet.dest_node)
        if not route:
            logger.warning(f"No route found from {self.node_id} to {packet.dest_node}")
            return None
        
        next_hop, wavelength, cost = route
        
        # Check if wavelength is available
        if wavelength not in self.wavelength_allocator:
            # Try wavelength conversion if enabled
            if self.config.wavelength_conversion:
                available_wavelengths = list(self.wavelength_allocator)
                if available_wavelengths:
                    wavelength = available_wavelengths[0]
                else:
                    # Buffer packet if no wavelength available
                    self._buffer_packet(packet, next_hop)
                    return next_hop
            else:
                self._buffer_packet(packet, next_hop)
                return next_hop
        
        # Allocate wavelength and configure switch
        packet.wavelength = wavelength
        packet.path.append(self.node_id)
        packet.hop_count += 1
        
        # Configure optical switch
        output_port = self._get_output_port(next_hop)
        if output_port is not None:
            success = self.optical_switch.configure_path(0, output_port, wavelength)
            if success:
                with self._lock:
                    self.wavelength_allocator.discard(wavelength)
                return next_hop
        
        # If routing failed, buffer the packet
        self._buffer_packet(packet, next_hop)
        return next_hop
    
    def _buffer_packet(self, packet: PhotonicPacket, output_port: int) -> None:
        """Buffer packet when routing is blocked."""
        with self._lock:
            if len(self.packet_buffers[output_port]) < self.config.buffer_size:
                self.packet_buffers[output_port].append(packet)
            else:
                logger.warning(f"Buffer overflow at node {self.node_id}, port {output_port}")
    
    def _get_output_port(self, next_hop: int) -> Optional[int]:
        """Get output port for next hop node."""
        # Simple mapping - in practice would be topology-dependent
        if next_hop > self.node_id:
            return (next_hop - self.node_id) % self.optical_switch.n_ports
        else:
            return (self.node_id - next_hop) % self.optical_switch.n_ports
    
    def process_buffered_packets(self) -> None:
        """Process packets in buffers."""
        with self._lock:
            for output_port, buffer in self.packet_buffers.items():
                if buffer and self.wavelength_allocator:
                    packet = buffer.popleft()
                    wavelength = list(self.wavelength_allocator)[0]
                    
                    # Try to route buffered packet
                    success = self.optical_switch.configure_path(0, output_port, wavelength)
                    if success:
                        packet.wavelength = wavelength
                        self.wavelength_allocator.discard(wavelength)
                    else:
                        # Put packet back in buffer
                        buffer.appendleft(packet)


class PhotonicInterconnect:
    """Main photonic interconnect management system."""
    
    def __init__(self, config: Optional[InterconnectConfig] = None):
        self.config = config or InterconnectConfig()
        self.routers = {}
        self.links = {}
        self.topology_graph = self._build_topology()
        self.performance_stats = {
            "packets_sent": 0,
            "packets_received": 0,
            "total_latency": 0.0,
            "blocked_packets": 0,
            "average_path_length": 0.0,
        }
        
        # Initialize routers
        for node_id in range(self.config.n_nodes):
            self.routers[node_id] = PhotonicRouter(node_id, self.config)
        
        # Build routing tables
        self._build_routing_tables()
        
        logger.info(f"Initialized PhotonicInterconnect with {self.config.n_nodes} nodes")
    
    def _build_topology(self) -> Dict[int, List[int]]:
        """Build network topology graph."""
        graph = defaultdict(list)
        
        if self.config.topology == TopologyType.MESH:
            # 2D mesh topology
            side = int(np.sqrt(self.config.n_nodes))
            for i in range(side):
                for j in range(side):
                    node = i * side + j
                    
                    # Connect to neighbors
                    neighbors = []
                    if i > 0: neighbors.append((i-1) * side + j)  # Up
                    if i < side-1: neighbors.append((i+1) * side + j)  # Down
                    if j > 0: neighbors.append(i * side + (j-1))  # Left
                    if j < side-1: neighbors.append(i * side + (j+1))  # Right
                    
                    graph[node] = neighbors
                    
                    # Create optical links
                    for neighbor in neighbors:
                        link_id = f"{min(node, neighbor)}_{max(node, neighbor)}"
                        if link_id not in self.links:
                            self.links[link_id] = OpticalLink(
                                src_node=node,
                                dst_node=neighbor,
                                wavelengths=set(range(self.config.n_wavelengths)),
                                bandwidth_per_wavelength=self.config.link_bandwidth,
                                propagation_delay=self.config.propagation_delay
                            )
        
        elif self.config.topology == TopologyType.CROSSBAR:
            # Full crossbar - all nodes connected to all others
            for i in range(self.config.n_nodes):
                for j in range(self.config.n_nodes):
                    if i != j:
                        graph[i].append(j)
                        
                        link_id = f"{min(i, j)}_{max(i, j)}"
                        if link_id not in self.links:
                            self.links[link_id] = OpticalLink(
                                src_node=i,
                                dst_node=j,
                                wavelengths=set(range(self.config.n_wavelengths)),
                                bandwidth_per_wavelength=self.config.link_bandwidth,
                                propagation_delay=self.config.propagation_delay
                            )
        
        return graph
    
    def _build_routing_tables(self) -> None:
        """Build routing tables using shortest path algorithm."""
        # Use Floyd-Warshall algorithm for all-pairs shortest paths
        n_nodes = self.config.n_nodes
        distances = np.full((n_nodes, n_nodes), np.inf)
        next_hops = np.full((n_nodes, n_nodes), -1, dtype=int)
        
        # Initialize distances
        for i in range(n_nodes):
            distances[i, i] = 0
            for neighbor in self.topology_graph[i]:
                distances[i, neighbor] = 1  # Unit cost for now
                next_hops[i, neighbor] = neighbor
        
        # Floyd-Warshall
        for k in range(n_nodes):
            for i in range(n_nodes):
                for j in range(n_nodes):
                    if distances[i, k] + distances[k, j] < distances[i, j]:
                        distances[i, j] = distances[i, k] + distances[k, j]
                        next_hops[i, j] = next_hops[i, k]
        
        # Populate routing tables
        for src in range(n_nodes):
            for dst in range(n_nodes):
                if src != dst and next_hops[src, dst] != -1:
                    # Assign wavelengths (simple round-robin for now)
                    wavelength = (src * n_nodes + dst) % self.config.n_wavelengths
                    cost = distances[src, dst]
                    
                    self.routers[src].routing_table.add_route(
                        src, dst, next_hops[src, dst], wavelength, cost
                    )
    
    def send_data(self, src_node: int, dst_node: int, 
                 data: torch.Tensor, priority: int = 0) -> int:
        """Send data from source to destination node."""
        if src_node >= self.config.n_nodes or dst_node >= self.config.n_nodes:
            raise PhotonicComputeError(f"Invalid node IDs: {src_node}, {dst_node}")
        
        validate_optical_tensor(data)
        
        # Create photonic packet
        packet_id = self.performance_stats["packets_sent"]
        packet = PhotonicPacket(
            packet_id=packet_id,
            source_node=src_node,
            dest_node=dst_node,
            wavelength=-1,  # Will be assigned by router
            data=data,
            timestamp=time.time(),
            priority=priority,
            max_hops=self.config.max_path_length
        )
        
        # Route packet through network
        current_node = src_node
        routing_successful = True
        
        while current_node != dst_node and packet.hop_count < packet.max_hops:
            next_hop = self.routers[current_node].route_packet(packet)
            
            if next_hop is None:
                # Packet reached destination
                break
            elif next_hop == current_node:
                # Routing blocked
                routing_successful = False
                self.performance_stats["blocked_packets"] += 1
                break
            
            current_node = next_hop
            
            # Add propagation delay
            link_id = f"{min(packet.path[-1], current_node)}_{max(packet.path[-1], current_node)}"
            if link_id in self.links:
                delay = self.links[link_id].propagation_delay
                time.sleep(delay)  # Simulate propagation delay
        
        # Update statistics
        self.performance_stats["packets_sent"] += 1
        if routing_successful and current_node == dst_node:
            self.performance_stats["packets_received"] += 1
            latency = time.time() - packet.timestamp
            self.performance_stats["total_latency"] += latency
            
            # Update average path length
            old_avg = self.performance_stats["average_path_length"]
            n_received = self.performance_stats["packets_received"]
            new_avg = (old_avg * (n_received - 1) + packet.hop_count) / n_received
            self.performance_stats["average_path_length"] = new_avg
        
        return packet_id
    
    def get_network_utilization(self) -> Dict[str, float]:
        """Get current network utilization statistics."""
        utilization = {}
        
        for link_id, link in self.links.items():
            total_capacity = len(link.wavelengths) * link.bandwidth_per_wavelength
            used_capacity = sum(link.utilization.values())
            utilization[link_id] = used_capacity / total_capacity if total_capacity > 0 else 0.0
        
        return utilization
    
    def adapt_routing(self) -> None:
        """Adapt routing based on network conditions."""
        if not self.config.enable_adaptive_routing:
            return
        
        utilization = self.get_network_utilization()
        
        # Identify congested links
        congested_links = {
            link_id: util for link_id, util in utilization.items()
            if util > self.config.congestion_threshold
        }
        
        if congested_links:
            logger.info(f"Congested links detected: {list(congested_links.keys())}")
            # In practice, would trigger routing table updates
            # For now, just log the congestion
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get network performance statistics."""
        stats = self.performance_stats.copy()
        
        if stats["packets_received"] > 0:
            stats["average_latency"] = stats["total_latency"] / stats["packets_received"]
            stats["delivery_ratio"] = stats["packets_received"] / stats["packets_sent"]
        else:
            stats["average_latency"] = 0.0
            stats["delivery_ratio"] = 0.0
        
        return stats
    
    def reset_stats(self) -> None:
        """Reset performance statistics."""
        self.performance_stats = {
            "packets_sent": 0,
            "packets_received": 0,
            "total_latency": 0.0,
            "blocked_packets": 0,
            "average_path_length": 0.0,
        }


# Convenience functions
def create_photonic_network(n_nodes: int = 16, 
                          topology: TopologyType = TopologyType.MESH,
                          n_wavelengths: int = 80) -> PhotonicInterconnect:
    """Create a photonic interconnect network."""
    config = InterconnectConfig(
        n_nodes=n_nodes,
        topology=topology,
        n_wavelengths=n_wavelengths
    )
    return PhotonicInterconnect(config)


def send_tensor(network: PhotonicInterconnect, src: int, dst: int,
               tensor: torch.Tensor) -> int:
    """Send tensor data across photonic network."""
    return network.send_data(src, dst, tensor)


# Global photonic interconnect instance
_global_photonic_network = None

def get_global_photonic_network() -> PhotonicInterconnect:
    """Get global photonic network instance."""
    global _global_photonic_network
    if _global_photonic_network is None:
        config = InterconnectConfig(
            n_wavelengths=get_config().photonic_wavelengths,
            n_nodes=min(16, get_config().photonic_wavelengths // 4)  # Reasonable default
        )
        _global_photonic_network = PhotonicInterconnect(config)
    return _global_photonic_network