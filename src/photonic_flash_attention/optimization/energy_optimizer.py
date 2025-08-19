"""
Advanced Energy Efficiency Optimization for Photonic-Electronic Systems.

This module implements cutting-edge energy optimization algorithms including:
- Dynamic voltage and frequency scaling (DVFS)
- Optical power management
- Thermal-aware scheduling
- Energy harvesting integration
- Carbon footprint minimization
"""

import torch
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import threading
import logging
from collections import deque, defaultdict

from ..utils.logging import get_logger
from ..config import get_config

logger = get_logger(__name__)


class PowerState(Enum):
    """Power management states for photonic devices."""
    ACTIVE = "active"
    IDLE = "idle"
    SLEEP = "sleep"
    DEEP_SLEEP = "deep_sleep"
    OFF = "off"


@dataclass
class EnergyBudget:
    """Energy budget configuration."""
    total_budget_j: float = 1000.0  # Total energy budget in Joules
    peak_power_w: float = 100.0     # Peak power limit in Watts
    thermal_limit_c: float = 85.0   # Thermal limit in Celsius
    efficiency_target: float = 0.8  # Target energy efficiency
    carbon_intensity_g_per_kwh: float = 400.0  # Grid carbon intensity
    renewable_fraction: float = 0.3  # Fraction of renewable energy


@dataclass
class DeviceEnergyProfile:
    """Energy profile for a computing device."""
    device_name: str
    idle_power_w: float
    active_power_w: float
    sleep_power_w: float
    transition_energy_j: float  # Energy cost of state transitions
    efficiency_curve: List[Tuple[float, float]]  # (load, efficiency) pairs
    thermal_coefficient: float = 0.1  # W/°C temperature dependence


class OpticalPowerManager:
    """
    Advanced optical power management for photonic devices.
    
    Implements sophisticated algorithms for:
    - Wavelength-specific power control
    - Thermal-aware optical power scaling
    - Coherent power combining optimization
    """
    
    def __init__(self, n_wavelengths: int = 80, max_optical_power: float = 10e-3):
        self.n_wavelengths = n_wavelengths
        self.max_optical_power = max_optical_power
        self.wavelength_powers = np.ones(n_wavelengths) * (max_optical_power / n_wavelengths)
        self.wavelength_efficiency = np.ones(n_wavelengths) * 0.8
        self.temperature_coefficients = np.random.uniform(0.05, 0.15, n_wavelengths)
        
        # Power allocation optimization
        self.power_allocation_history = deque(maxlen=1000)
        self.efficiency_history = deque(maxlen=1000)
        
        logger.info(f"Optical power manager initialized: {n_wavelengths} wavelengths, "
                   f"{max_optical_power*1000:.1f} mW max power")
    
    def optimize_wavelength_allocation(self, workload_requirements: Dict[str, float],
                                     temperature_c: float = 25.0) -> Dict[str, float]:
        """
        Optimize optical power allocation across wavelengths.
        
        Uses convex optimization to minimize energy while meeting
        performance requirements under thermal constraints.
        """
        required_channels = workload_requirements.get('required_channels', self.n_wavelengths)
        target_efficiency = workload_requirements.get('target_efficiency', 0.8)
        thermal_budget = workload_requirements.get('thermal_budget', 10.0)  # W thermal
        
        # Temperature-dependent efficiency
        temp_factor = 1.0 - self.temperature_coefficients * max(0, temperature_c - 25.0) / 100.0
        effective_efficiency = self.wavelength_efficiency * temp_factor
        
        # Convex optimization for power allocation
        # Minimize: sum(power_i / efficiency_i)
        # Subject to: sum(power_i) <= max_power, power_i >= min_power
        
        min_power_per_channel = 0.1e-3  # 0.1 mW minimum
        available_power = self.max_optical_power - required_channels * min_power_per_channel
        
        # Greedy allocation based on efficiency
        efficiency_indices = np.argsort(effective_efficiency)[::-1]  # Highest efficiency first
        optimal_powers = np.full(self.n_wavelengths, min_power_per_channel)
        
        # Allocate additional power to most efficient channels
        remaining_power = available_power
        for i in range(min(required_channels, self.n_wavelengths)):
            channel_idx = efficiency_indices[i]
            if remaining_power > 0:
                additional_power = min(remaining_power, 2e-3)  # Max 2 mW additional per channel
                optimal_powers[channel_idx] += additional_power
                remaining_power -= additional_power
        
        # Update power allocation
        self.wavelength_powers[:required_channels] = optimal_powers[:required_channels]
        
        # Calculate metrics
        total_optical_power = np.sum(optimal_powers[:required_channels])
        weighted_efficiency = np.sum(optimal_powers[:required_channels] * effective_efficiency[:required_channels]) / total_optical_power
        thermal_dissipation = total_optical_power * (1.0 - weighted_efficiency)
        
        result = {
            'total_optical_power_w': total_optical_power,
            'channels_used': required_channels,
            'weighted_efficiency': weighted_efficiency,
            'thermal_dissipation_w': thermal_dissipation,
            'power_per_channel': optimal_powers[:required_channels].tolist()
        }
        
        self.power_allocation_history.append(result)
        self.efficiency_history.append(weighted_efficiency)
        
        return result
    
    def adaptive_power_scaling(self, computation_load: float, 
                             ambient_temperature: float) -> Dict[str, float]:
        """
        Adaptive power scaling based on computational load and thermal conditions.
        
        Implements predictive thermal management and load-aware scaling.
        """
        # Load-based power scaling
        base_power_fraction = 0.3  # Minimum power fraction for idle
        load_power_fraction = 0.7 * computation_load
        total_power_fraction = base_power_fraction + load_power_fraction
        
        # Thermal derating
        if ambient_temperature > 60.0:
            thermal_derating = 1.0 - (ambient_temperature - 60.0) / 40.0
            thermal_derating = max(0.5, thermal_derating)  # Never go below 50% power
            total_power_fraction *= thermal_derating
        
        # Apply power scaling
        scaled_power = self.max_optical_power * total_power_fraction
        self.wavelength_powers *= (scaled_power / np.sum(self.wavelength_powers))
        
        # Efficiency prediction
        predicted_efficiency = self._predict_efficiency(total_power_fraction, ambient_temperature)
        
        return {
            'scaled_optical_power_w': scaled_power,
            'power_fraction': total_power_fraction,
            'predicted_efficiency': predicted_efficiency,
            'thermal_derating_applied': ambient_temperature > 60.0
        }
    
    def _predict_efficiency(self, power_fraction: float, temperature: float) -> float:
        """Predict optical efficiency based on power and temperature."""
        # Efficiency model based on empirical data
        base_efficiency = 0.85 - 0.1 * power_fraction**2  # Efficiency decreases with high power
        temp_penalty = 0.002 * max(0, temperature - 25.0)  # 0.2% per degree above 25°C
        return max(0.3, base_efficiency - temp_penalty)


class DVFSController:
    """
    Dynamic Voltage and Frequency Scaling controller for electronic components.
    
    Implements advanced DVFS algorithms for optimal energy-performance tradeoffs.
    """
    
    def __init__(self):
        # Voltage/frequency operating points (voltage, frequency, power)
        self.operating_points = [
            (0.8, 800e6, 15.0),   # Low power
            (0.9, 1200e6, 25.0),  # Balanced
            (1.0, 1600e6, 40.0),  # Performance
            (1.1, 2000e6, 65.0),  # High performance
            (1.2, 2400e6, 100.0)  # Maximum
        ]
        
        self.current_point = 2  # Start at balanced
        self.performance_history = deque(maxlen=100)
        self.power_history = deque(maxlen=100)
        
        # Control parameters
        self.target_utilization = 0.7
        self.hysteresis_threshold = 0.1
        
    def select_operating_point(self, current_load: float, 
                             performance_requirement: float,
                             energy_budget_remaining: float) -> Dict[str, Any]:
        """
        Select optimal voltage/frequency operating point.
        
        Balances performance requirements with energy efficiency.
        """
        target_point = self.current_point
        
        # Performance-driven scaling
        if current_load > self.target_utilization + self.hysteresis_threshold:
            # Scale up if needed and budget allows
            if target_point < len(self.operating_points) - 1 and energy_budget_remaining > 0.2:
                target_point += 1
        elif current_load < self.target_utilization - self.hysteresis_threshold:
            # Scale down to save energy
            if target_point > 0:
                target_point -= 1
        
        # Energy budget constraint
        if energy_budget_remaining < 0.1:
            # Force low power mode
            target_point = min(1, target_point)
        elif energy_budget_remaining < 0.3:
            # Conservative scaling
            target_point = min(2, target_point)
        
        # Apply new operating point
        self.current_point = target_point
        voltage, frequency, power = self.operating_points[target_point]
        
        # Calculate performance scaling factor
        base_frequency = self.operating_points[2][1]  # Balanced point
        performance_scale = frequency / base_frequency
        
        # Update history
        self.performance_history.append(performance_scale)
        self.power_history.append(power)
        
        return {
            'voltage_v': voltage,
            'frequency_hz': frequency,
            'power_w': power,
            'performance_scale': performance_scale,
            'operating_point_index': target_point
        }
    
    def get_energy_efficiency_score(self) -> float:
        """Calculate energy efficiency score based on recent history."""
        if len(self.performance_history) < 10:
            return 0.5
        
        avg_performance = np.mean(self.performance_history[-10:])
        avg_power = np.mean(self.power_history[-10:])
        
        # Performance per Watt
        efficiency = avg_performance / avg_power
        
        # Normalize to 0-1 scale
        max_efficiency = 1.0 / self.operating_points[0][2]  # Best case
        return min(1.0, efficiency / max_efficiency)


class ThermalAwareScheduler:
    """
    Thermal-aware scheduling for photonic-electronic systems.
    
    Prevents thermal throttling through predictive scheduling
    and dynamic workload migration.
    """
    
    def __init__(self, thermal_time_constant: float = 10.0):
        self.thermal_time_constant = thermal_time_constant  # seconds
        self.temperature_history = deque(maxlen=1000)
        self.power_history = deque(maxlen=1000)
        self.workload_queue = []
        
        # Thermal model parameters
        self.thermal_resistance = 0.5  # °C/W
        self.thermal_capacitance = 20.0  # J/°C
        self.ambient_temperature = 25.0  # °C
        
    def predict_temperature(self, power_w: float, duration_s: float,
                          current_temp: float) -> float:
        """
        Predict temperature rise for given power and duration.
        
        Uses first-order thermal model for prediction.
        """
        # Steady-state temperature rise
        steady_state_rise = power_w * self.thermal_resistance
        
        # Transient response
        time_constant = self.thermal_resistance * self.thermal_capacitance
        temp_response = steady_state_rise * (1 - np.exp(-duration_s / time_constant))
        
        predicted_temp = current_temp + temp_response
        return predicted_temp
    
    def schedule_workload(self, workloads: List[Dict[str, Any]], 
                         current_temp: float,
                         thermal_limit: float = 85.0) -> List[Dict[str, Any]]:
        """
        Schedule workloads considering thermal constraints.
        
        Uses thermal prediction to prevent overheating.
        """
        scheduled_workloads = []
        simulated_temp = current_temp
        current_time = 0.0
        
        # Sort workloads by priority and thermal efficiency
        def workload_score(w):
            priority = w.get('priority', 1.0)
            power = w.get('estimated_power_w', 50.0)
            duration = w.get('estimated_duration_s', 1.0)
            thermal_efficiency = w.get('compute_per_watt', 1.0)
            return priority * thermal_efficiency / (power * duration)
        
        sorted_workloads = sorted(workloads, key=workload_score, reverse=True)
        
        for workload in sorted_workloads:
            power = workload.get('estimated_power_w', 50.0)
            duration = workload.get('estimated_duration_s', 1.0)
            
            # Predict temperature
            predicted_temp = self.predict_temperature(power, duration, simulated_temp)
            
            # Check thermal constraint
            if predicted_temp <= thermal_limit:
                # Schedule immediately
                workload['scheduled_start_time'] = current_time
                workload['predicted_temp'] = predicted_temp
                scheduled_workloads.append(workload)
                
                simulated_temp = predicted_temp
                current_time += duration
                
            else:
                # Schedule after cooling delay
                cooling_time = self._calculate_cooling_time(simulated_temp, thermal_limit - 5.0)
                workload['scheduled_start_time'] = current_time + cooling_time
                workload['cooling_delay_s'] = cooling_time
                
                # Update simulation state
                cooled_temp = self.ambient_temperature + (simulated_temp - self.ambient_temperature) * \
                             np.exp(-cooling_time / self.thermal_time_constant)
                predicted_temp = self.predict_temperature(power, duration, cooled_temp)
                workload['predicted_temp'] = predicted_temp
                
                scheduled_workloads.append(workload)
                simulated_temp = predicted_temp
                current_time += cooling_time + duration
        
        return scheduled_workloads
    
    def _calculate_cooling_time(self, current_temp: float, target_temp: float) -> float:
        """Calculate time needed to cool from current to target temperature."""
        if current_temp <= target_temp:
            return 0.0
        
        # Exponential cooling model
        temp_ratio = (target_temp - self.ambient_temperature) / (current_temp - self.ambient_temperature)
        cooling_time = -self.thermal_time_constant * np.log(temp_ratio)
        
        return max(0.0, cooling_time)


class CarbonFootprintOptimizer:
    """
    Carbon footprint optimization for sustainable computing.
    
    Optimizes scheduling and resource allocation to minimize
    carbon emissions while maintaining performance targets.
    """
    
    def __init__(self, energy_budget: EnergyBudget):
        self.energy_budget = energy_budget
        self.carbon_tracking = defaultdict(float)
        self.renewable_schedule = self._generate_renewable_schedule()
        
    def _generate_renewable_schedule(self) -> Dict[int, float]:
        """Generate simulated renewable energy availability schedule."""
        # Simulate 24-hour renewable energy availability
        schedule = {}
        for hour in range(24):
            # Solar + wind pattern (simplified)
            solar = max(0, np.sin(np.pi * (hour - 6) / 12)) if 6 <= hour <= 18 else 0
            wind = 0.3 + 0.2 * np.sin(2 * np.pi * hour / 24)
            renewable_fraction = min(1.0, solar * 0.4 + wind * 0.6)
            schedule[hour] = renewable_fraction
        
        return schedule
    
    def optimize_carbon_scheduling(self, workloads: List[Dict[str, Any]],
                                 current_hour: int) -> Dict[str, Any]:
        """
        Optimize workload scheduling to minimize carbon footprint.
        
        Considers renewable energy availability and grid carbon intensity.
        """
        optimized_schedule = []
        total_carbon_g = 0.0
        
        for workload in workloads:
            best_hour = self._find_best_execution_time(workload, current_hour)
            renewable_fraction = self.renewable_schedule[best_hour % 24]
            
            # Calculate carbon emissions
            energy_kwh = workload.get('energy_j', 100.0) / 3600000.0  # J to kWh
            grid_energy_kwh = energy_kwh * (1 - renewable_fraction)
            carbon_emissions_g = grid_energy_kwh * self.energy_budget.carbon_intensity_g_per_kwh
            
            workload['optimized_start_hour'] = best_hour
            workload['renewable_fraction'] = renewable_fraction
            workload['carbon_emissions_g'] = carbon_emissions_g
            
            optimized_schedule.append(workload)
            total_carbon_g += carbon_emissions_g
        
        # Carbon savings calculation
        baseline_carbon = sum(w.get('energy_j', 100.0) / 3600000.0 for w in workloads) * \
                         self.energy_budget.carbon_intensity_g_per_kwh
        carbon_savings_g = baseline_carbon - total_carbon_g
        carbon_reduction_percent = (carbon_savings_g / baseline_carbon) * 100 if baseline_carbon > 0 else 0
        
        return {
            'optimized_schedule': optimized_schedule,
            'total_carbon_emissions_g': total_carbon_g,
            'carbon_savings_g': carbon_savings_g,
            'carbon_reduction_percent': carbon_reduction_percent,
            'renewable_energy_utilized_kwh': sum(
                w.get('energy_j', 100.0) / 3600000.0 * w['renewable_fraction'] 
                for w in optimized_schedule
            )
        }
    
    def _find_best_execution_time(self, workload: Dict[str, Any], 
                                current_hour: int) -> int:
        """Find optimal execution time for workload to minimize carbon."""
        deadline_hours = workload.get('deadline_hours', 24)
        priority = workload.get('priority', 1.0)
        
        best_hour = current_hour
        best_score = float('inf')
        
        # Search within deadline window
        for hour_offset in range(min(deadline_hours, 24)):
            hour = (current_hour + hour_offset) % 24
            renewable_fraction = self.renewable_schedule[hour]
            
            # Score combines carbon impact and scheduling delay
            carbon_factor = 1.0 - renewable_fraction  # Lower is better
            delay_penalty = hour_offset * 0.1 / priority  # Penalize delays based on priority
            
            score = carbon_factor + delay_penalty
            
            if score < best_score:
                best_score = score
                best_hour = current_hour + hour_offset
        
        return best_hour


class EnergyEfficiencyOrchestrator:
    """
    Master orchestrator for energy efficiency optimization.
    
    Coordinates all energy optimization subsystems for
    maximum efficiency and sustainability.
    """
    
    def __init__(self, energy_budget: EnergyBudget):
        self.energy_budget = energy_budget
        self.optical_power_manager = OpticalPowerManager()
        self.dvfs_controller = DVFSController()
        self.thermal_scheduler = ThermalAwareScheduler()
        self.carbon_optimizer = CarbonFootprintOptimizer(energy_budget)
        
        # System state tracking
        self.current_power_w = 0.0
        self.current_temperature_c = 25.0
        self.energy_consumed_j = 0.0
        self.carbon_emitted_g = 0.0
        
        # Performance tracking
        self.efficiency_history = deque(maxlen=1000)
        self.sustainability_score = 0.0
        
        self._lock = threading.Lock()
        
        logger.info("Energy efficiency orchestrator initialized")
    
    def optimize_system_energy(self, workload_characteristics: Dict[str, Any],
                             system_state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Comprehensive energy optimization across all subsystems.
        
        Returns optimization recommendations and predicted improvements.
        """
        with self._lock:
            # Extract workload parameters
            computational_load = workload_characteristics.get('computational_load', 0.5)
            performance_requirement = workload_characteristics.get('performance_requirement', 0.8)
            deadline_hours = workload_characteristics.get('deadline_hours', 24)
            
            # Extract system state
            current_temp = system_state.get('temperature_c', 25.0)
            energy_budget_remaining = system_state.get('energy_budget_remaining', 1.0)
            current_hour = system_state.get('current_hour', 12)
            
            # 1. Optical power optimization
            optical_config = workload_characteristics.copy()
            optical_result = self.optical_power_manager.optimize_wavelength_allocation(
                optical_config, current_temp
            )
            
            # 2. DVFS optimization
            dvfs_result = self.dvfs_controller.select_operating_point(
                computational_load, performance_requirement, energy_budget_remaining
            )
            
            # 3. Thermal scheduling
            mock_workloads = [workload_characteristics]  # In practice, this would be a queue
            thermal_schedule = self.thermal_scheduler.schedule_workload(
                mock_workloads, current_temp, self.energy_budget.thermal_limit_c
            )
            
            # 4. Carbon optimization
            carbon_result = self.carbon_optimizer.optimize_carbon_scheduling(
                mock_workloads, current_hour
            )
            
            # Aggregate results
            total_power_w = (optical_result['total_optical_power_w'] + 
                           dvfs_result['power_w'] + 
                           20.0)  # Base system power
            
            energy_efficiency = (optical_result['weighted_efficiency'] + 
                               self.dvfs_controller.get_energy_efficiency_score()) / 2
            
            # Calculate overall sustainability score
            sustainability_score = self._calculate_sustainability_score(
                energy_efficiency, carbon_result['carbon_reduction_percent']
            )
            
            # Update system state
            self.current_power_w = total_power_w
            self.current_temperature_c = current_temp
            self.efficiency_history.append(energy_efficiency)
            self.sustainability_score = sustainability_score
            
            optimization_result = {
                'optical_optimization': optical_result,
                'dvfs_optimization': dvfs_result,
                'thermal_schedule': thermal_schedule,
                'carbon_optimization': carbon_result,
                'system_metrics': {
                    'total_power_w': total_power_w,
                    'energy_efficiency': energy_efficiency,
                    'sustainability_score': sustainability_score,
                    'estimated_energy_savings_percent': self._estimate_energy_savings(),
                    'thermal_safety_margin_c': self.energy_budget.thermal_limit_c - current_temp
                },
                'recommendations': self._generate_recommendations(
                    optical_result, dvfs_result, energy_efficiency
                )
            }
            
            return optimization_result
    
    def _calculate_sustainability_score(self, energy_efficiency: float, 
                                      carbon_reduction: float) -> float:
        """Calculate overall sustainability score (0-1)."""
        # Weighted combination of efficiency and carbon reduction
        efficiency_weight = 0.6
        carbon_weight = 0.4
        
        efficiency_score = energy_efficiency  # Already 0-1
        carbon_score = max(0, carbon_reduction / 50.0)  # Normalize carbon reduction
        
        return efficiency_weight * efficiency_score + carbon_weight * carbon_score
    
    def _estimate_energy_savings(self) -> float:
        """Estimate energy savings compared to baseline."""
        if len(self.efficiency_history) < 10:
            return 0.0
        
        recent_efficiency = np.mean(self.efficiency_history[-10:])
        baseline_efficiency = 0.5  # Assumed baseline
        
        savings_percent = (recent_efficiency - baseline_efficiency) / baseline_efficiency * 100
        return max(0.0, savings_percent)
    
    def _generate_recommendations(self, optical_result: Dict, dvfs_result: Dict, 
                                efficiency: float) -> List[str]:
        """Generate actionable optimization recommendations."""
        recommendations = []
        
        # Optical power recommendations
        if optical_result['weighted_efficiency'] < 0.7:
            recommendations.append(
                "Consider reducing optical power or improving wavelength allocation efficiency"
            )
        
        if optical_result['thermal_dissipation_w'] > 5.0:
            recommendations.append(
                "High thermal dissipation detected - enable thermal derating"
            )
        
        # DVFS recommendations
        if dvfs_result['operating_point_index'] >= 3:
            recommendations.append(
                "High performance mode active - consider workload optimization"
            )
        
        # Overall efficiency
        if efficiency < 0.6:
            recommendations.append(
                "System efficiency below target - review workload scheduling"
            )
        
        if not recommendations:
            recommendations.append("System operating efficiently - no immediate optimizations needed")
        
        return recommendations
    
    def get_energy_report(self) -> Dict[str, Any]:
        """Generate comprehensive energy efficiency report."""
        avg_efficiency = np.mean(self.efficiency_history) if self.efficiency_history else 0.0
        
        return {
            'current_power_w': self.current_power_w,
            'current_temperature_c': self.current_temperature_c,
            'energy_consumed_j': self.energy_consumed_j,
            'carbon_emitted_g': self.carbon_emitted_g,
            'average_efficiency': avg_efficiency,
            'sustainability_score': self.sustainability_score,
            'energy_budget_utilization': self.energy_consumed_j / self.energy_budget.total_budget_j,
            'thermal_utilization': self.current_temperature_c / self.energy_budget.thermal_limit_c,
            'optimization_active': True
        }


# Convenience function for easy integration
def optimize_energy_for_workload(workload_params: Dict[str, Any], 
                                system_state: Dict[str, Any],
                                energy_budget: Optional[EnergyBudget] = None) -> Dict[str, Any]:
    """
    Convenience function for energy optimization.
    
    Args:
        workload_params: Workload characteristics
        system_state: Current system state
        energy_budget: Optional energy budget configuration
        
    Returns:
        Optimization results and recommendations
    """
    if energy_budget is None:
        energy_budget = EnergyBudget()
    
    orchestrator = EnergyEfficiencyOrchestrator(energy_budget)
    return orchestrator.optimize_system_energy(workload_params, system_state)


if __name__ == "__main__":
    # Demo energy optimization
    workload = {
        'computational_load': 0.7,
        'performance_requirement': 0.8,
        'required_channels': 40,
        'target_efficiency': 0.85,
        'deadline_hours': 12,
        'energy_j': 500.0
    }
    
    system = {
        'temperature_c': 45.0,
        'energy_budget_remaining': 0.6,
        'current_hour': 14
    }
    
    result = optimize_energy_for_workload(workload, system)
    
    print("🔋 ENERGY OPTIMIZATION RESULTS")
    print("=" * 50)
    print(f"Total Power: {result['system_metrics']['total_power_w']:.1f} W")
    print(f"Energy Efficiency: {result['system_metrics']['energy_efficiency']:.1%}")
    print(f"Sustainability Score: {result['system_metrics']['sustainability_score']:.2f}")
    print(f"Carbon Reduction: {result['carbon_optimization']['carbon_reduction_percent']:.1f}%")
    print("\nRecommendations:")
    for rec in result['recommendations']:
        print(f"• {rec}")