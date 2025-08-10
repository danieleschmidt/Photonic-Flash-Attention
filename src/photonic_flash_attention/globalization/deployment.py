"""Multi-region deployment and geographic distribution support."""

import time
import json
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import logging
import hashlib

from ..utils.logging import get_logger


logger = get_logger(__name__)


class Region(Enum):
    """Supported deployment regions."""
    US_EAST_1 = "us-east-1"
    US_WEST_2 = "us-west-2"
    EU_WEST_1 = "eu-west-1"
    EU_CENTRAL_1 = "eu-central-1"
    AP_SOUTHEAST_1 = "ap-southeast-1"
    AP_NORTHEAST_1 = "ap-northeast-1"
    AP_NORTHEAST_2 = "ap-northeast-2"
    CA_CENTRAL_1 = "ca-central-1"
    SA_EAST_1 = "sa-east-1"
    ME_SOUTH_1 = "me-south-1"
    AF_SOUTH_1 = "af-south-1"


class DeploymentTier(Enum):
    """Deployment tier for different service levels."""
    PRODUCTION = "production"
    STAGING = "staging"
    DEVELOPMENT = "development"
    DISASTER_RECOVERY = "disaster_recovery"


@dataclass
class RegionInfo:
    """Information about a deployment region."""
    region: Region
    name: str
    country_code: str
    continent: str
    timezone: str
    data_residency_rules: List[str]
    compliance_regimes: List[str]
    latency_zones: List[str] = field(default_factory=list)
    available_services: List[str] = field(default_factory=list)
    photonic_hardware_available: bool = False
    gpu_hardware_available: bool = True
    cpu_only: bool = False
    cost_multiplier: float = 1.0


@dataclass
class DeploymentConfig:
    """Configuration for multi-region deployment."""
    primary_region: Region
    secondary_regions: List[Region]
    tier: DeploymentTier
    auto_failover: bool = True
    cross_region_replication: bool = True
    data_encryption_at_rest: bool = True
    data_encryption_in_transit: bool = True
    monitoring_enabled: bool = True
    logging_enabled: bool = True
    backup_retention_days: int = 30
    health_check_interval: int = 30
    load_balancing_enabled: bool = True
    cdn_enabled: bool = False


class RegionManager:
    """Manages multi-region deployment and geographic distribution."""
    
    def __init__(self):
        """Initialize region manager."""
        self.regions: Dict[Region, RegionInfo] = {}
        self.deployments: Dict[str, DeploymentConfig] = {}
        self.region_health: Dict[Region, Dict[str, Any]] = {}
        self.traffic_routing: Dict[str, List[Tuple[Region, float]]] = {}
        
        # Initialize region information
        self._initialize_regions()
        
        logger.info("Region manager initialized")
    
    def _initialize_regions(self) -> None:
        """Initialize region information."""
        regions_config = {
            Region.US_EAST_1: RegionInfo(
                region=Region.US_EAST_1,
                name="US East (N. Virginia)",
                country_code="US",
                continent="North America",
                timezone="America/New_York",
                data_residency_rules=["US_CLOUD_ACT", "US_PATRIOT_ACT"],
                compliance_regimes=["CCPA", "HIPAA", "SOX"],
                latency_zones=["us-east", "ca-central"],
                available_services=["photonic", "gpu", "cpu"],
                photonic_hardware_available=True,
                cost_multiplier=1.0
            ),
            Region.US_WEST_2: RegionInfo(
                region=Region.US_WEST_2,
                name="US West (Oregon)",
                country_code="US",
                continent="North America", 
                timezone="America/Los_Angeles",
                data_residency_rules=["US_CLOUD_ACT", "US_PATRIOT_ACT"],
                compliance_regimes=["CCPA", "HIPAA", "SOX"],
                latency_zones=["us-west", "ca-central"],
                available_services=["photonic", "gpu", "cpu"],
                photonic_hardware_available=True,
                cost_multiplier=1.05
            ),
            Region.EU_WEST_1: RegionInfo(
                region=Region.EU_WEST_1,
                name="Europe (Ireland)",
                country_code="IE",
                continent="Europe",
                timezone="Europe/Dublin",
                data_residency_rules=["GDPR", "DATA_ACT"],
                compliance_regimes=["GDPR", "AI_ACT", "DATA_ACT"],
                latency_zones=["eu-west", "eu-central"],
                available_services=["photonic", "gpu", "cpu"],
                photonic_hardware_available=True,
                cost_multiplier=1.15
            ),
            Region.EU_CENTRAL_1: RegionInfo(
                region=Region.EU_CENTRAL_1,
                name="Europe (Frankfurt)",
                country_code="DE",
                continent="Europe",
                timezone="Europe/Berlin",
                data_residency_rules=["GDPR", "DATA_ACT", "GERMAN_BSI"],
                compliance_regimes=["GDPR", "AI_ACT", "DATA_ACT"],
                latency_zones=["eu-central", "eu-west"],
                available_services=["photonic", "gpu", "cpu"],
                photonic_hardware_available=True,
                cost_multiplier=1.20
            ),
            Region.AP_SOUTHEAST_1: RegionInfo(
                region=Region.AP_SOUTHEAST_1,
                name="Asia Pacific (Singapore)",
                country_code="SG",
                continent="Asia",
                timezone="Asia/Singapore",
                data_residency_rules=["PDPA", "BANKING_ACT"],
                compliance_regimes=["PDPA", "BANKING_ACT"],
                latency_zones=["ap-southeast", "ap-northeast"],
                available_services=["gpu", "cpu"],  # Limited photonic availability
                photonic_hardware_available=False,
                cost_multiplier=1.25
            ),
            Region.AP_NORTHEAST_1: RegionInfo(
                region=Region.AP_NORTHEAST_1,
                name="Asia Pacific (Tokyo)",
                country_code="JP",
                continent="Asia",
                timezone="Asia/Tokyo",
                data_residency_rules=["APPI", "BANKING_LAW"],
                compliance_regimes=["APPI", "FINANCIAL_INSTRUMENTS"],
                latency_zones=["ap-northeast", "ap-southeast"],
                available_services=["photonic", "gpu", "cpu"],
                photonic_hardware_available=True,
                cost_multiplier=1.30
            ),
        }
        
        self.regions.update(regions_config)
        
        # Initialize health monitoring for all regions
        for region in self.regions:
            self.region_health[region] = {
                'status': 'healthy',
                'last_check': time.time(),
                'latency_ms': 0.0,
                'availability': 1.0,
                'error_rate': 0.0,
                'capacity_utilization': 0.0,
            }
    
    def get_optimal_region(
        self,
        user_location: Optional[str] = None,
        compliance_requirements: Optional[List[str]] = None,
        service_requirements: Optional[List[str]] = None,
        cost_sensitive: bool = False
    ) -> Region:
        """
        Get optimal deployment region based on requirements.
        
        Args:
            user_location: User's geographic location (country code)
            compliance_requirements: Required compliance regimes
            service_requirements: Required services (photonic, gpu, cpu)
            cost_sensitive: Whether to optimize for cost
            
        Returns:
            Optimal region for deployment
        """
        candidates = list(self.regions.values())
        
        # Filter by compliance requirements
        if compliance_requirements:
            candidates = [
                r for r in candidates
                if any(req in r.compliance_regimes for req in compliance_requirements)
            ]
        
        # Filter by service requirements
        if service_requirements:
            candidates = [
                r for r in candidates
                if all(service in r.available_services for service in service_requirements)
            ]
        
        if not candidates:
            logger.warning("No regions meet all requirements, using default")
            return Region.US_EAST_1
        
        # Score each candidate region
        scores = []
        for region_info in candidates:
            score = 0.0
            
            # Geographic proximity (if user location provided)
            if user_location:
                if user_location.upper() == region_info.country_code:
                    score += 50  # Same country
                elif user_location.upper() in ['US', 'CA'] and region_info.continent == 'North America':
                    score += 30  # Same continent
                elif user_location.upper() in ['GB', 'DE', 'FR', 'IT', 'ES'] and region_info.continent == 'Europe':
                    score += 30
                elif region_info.continent == 'Asia' and user_location.upper() in ['JP', 'KR', 'SG', 'CN']:
                    score += 30
            
            # Health and performance
            health = self.region_health.get(region_info.region, {})
            score += health.get('availability', 0.0) * 20
            score -= health.get('error_rate', 0.0) * 10
            score -= health.get('latency_ms', 0.0) / 10
            
            # Cost consideration
            if cost_sensitive:
                score -= (region_info.cost_multiplier - 1.0) * 20
            
            # Photonic hardware availability bonus
            if region_info.photonic_hardware_available:
                score += 15
            
            scores.append((region_info.region, score))
        
        # Return region with highest score
        best_region = max(scores, key=lambda x: x[1])[0]
        
        logger.info(f"Selected optimal region: {best_region.value}")
        return best_region
    
    def create_deployment(
        self,
        deployment_id: str,
        config: DeploymentConfig
    ) -> bool:
        """
        Create a new multi-region deployment.
        
        Args:
            deployment_id: Unique identifier for deployment
            config: Deployment configuration
            
        Returns:
            True if deployment created successfully
        """
        try:
            # Validate regions
            all_regions = [config.primary_region] + config.secondary_regions
            for region in all_regions:
                if region not in self.regions:
                    raise ValueError(f"Unknown region: {region}")
            
            # Check region capabilities
            for region in all_regions:
                region_info = self.regions[region]
                if not region_info.available_services:
                    logger.warning(f"Region {region.value} has limited service availability")
            
            self.deployments[deployment_id] = config
            
            # Initialize traffic routing (start with primary region)
            self.traffic_routing[deployment_id] = [(config.primary_region, 1.0)]
            
            logger.info(f"Created deployment {deployment_id} in {len(all_regions)} regions")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create deployment {deployment_id}: {e}")
            return False
    
    def update_region_health(
        self,
        region: Region,
        status: str,
        latency_ms: float = 0.0,
        availability: float = 1.0,
        error_rate: float = 0.0,
        capacity_utilization: float = 0.0
    ) -> None:
        """Update health information for a region."""
        self.region_health[region] = {
            'status': status,
            'last_check': time.time(),
            'latency_ms': latency_ms,
            'availability': availability,
            'error_rate': error_rate,
            'capacity_utilization': capacity_utilization,
        }
        
        # Trigger auto-failover if needed
        if status == 'unhealthy':
            self._trigger_failover(region)
    
    def _trigger_failover(self, failed_region: Region) -> None:
        """Trigger automatic failover from failed region."""
        affected_deployments = [
            dep_id for dep_id, config in self.deployments.items()
            if config.primary_region == failed_region and config.auto_failover
        ]
        
        for deployment_id in affected_deployments:
            config = self.deployments[deployment_id]
            
            if config.secondary_regions:
                # Find healthy secondary region
                for secondary_region in config.secondary_regions:
                    health = self.region_health.get(secondary_region, {})
                    if health.get('status') == 'healthy':
                        # Update traffic routing
                        self.traffic_routing[deployment_id] = [(secondary_region, 1.0)]
                        logger.warning(f"Failed over deployment {deployment_id} to {secondary_region.value}")
                        break
    
    def get_deployment_status(self, deployment_id: str) -> Dict[str, Any]:
        """Get status of a deployment."""
        if deployment_id not in self.deployments:
            return {'error': 'Deployment not found'}
        
        config = self.deployments[deployment_id]
        all_regions = [config.primary_region] + config.secondary_regions
        
        region_statuses = {}
        for region in all_regions:
            health = self.region_health.get(region, {})
            region_info = self.regions.get(region, {})
            
            region_statuses[region.value] = {
                'health': health,
                'info': {
                    'name': region_info.name if hasattr(region_info, 'name') else 'Unknown',
                    'country_code': region_info.country_code if hasattr(region_info, 'country_code') else 'XX',
                    'photonic_available': region_info.photonic_hardware_available if hasattr(region_info, 'photonic_hardware_available') else False,
                }
            }
        
        # Current traffic routing
        current_routing = self.traffic_routing.get(deployment_id, [])
        
        return {
            'deployment_id': deployment_id,
            'config': {
                'primary_region': config.primary_region.value,
                'secondary_regions': [r.value for r in config.secondary_regions],
                'tier': config.tier.value,
                'auto_failover': config.auto_failover,
            },
            'regions': region_statuses,
            'traffic_routing': [
                {'region': region.value, 'weight': weight}
                for region, weight in current_routing
            ],
            'overall_health': self._calculate_overall_health(deployment_id),
        }
    
    def _calculate_overall_health(self, deployment_id: str) -> str:
        """Calculate overall health of a deployment."""
        if deployment_id not in self.deployments:
            return 'unknown'
        
        config = self.deployments[deployment_id]
        all_regions = [config.primary_region] + config.secondary_regions
        
        healthy_regions = 0
        total_regions = len(all_regions)
        
        for region in all_regions:
            health = self.region_health.get(region, {})
            if health.get('status') == 'healthy':
                healthy_regions += 1
        
        if healthy_regions == total_regions:
            return 'healthy'
        elif healthy_regions > total_regions // 2:
            return 'degraded'
        elif healthy_regions > 0:
            return 'critical'
        else:
            return 'down'
    
    def get_compliance_report(self, deployment_id: str) -> Dict[str, Any]:
        """Generate compliance report for deployment."""
        if deployment_id not in self.deployments:
            return {'error': 'Deployment not found'}
        
        config = self.deployments[deployment_id]
        all_regions = [config.primary_region] + config.secondary_regions
        
        compliance_regimes = set()
        data_residency_rules = set()
        countries = set()
        
        for region in all_regions:
            if region in self.regions:
                region_info = self.regions[region]
                compliance_regimes.update(region_info.compliance_regimes)
                data_residency_rules.update(region_info.data_residency_rules)
                countries.add(region_info.country_code)
        
        return {
            'deployment_id': deployment_id,
            'regions_count': len(all_regions),
            'countries': sorted(list(countries)),
            'compliance_regimes': sorted(list(compliance_regimes)),
            'data_residency_rules': sorted(list(data_residency_rules)),
            'encryption_at_rest': config.data_encryption_at_rest,
            'encryption_in_transit': config.data_encryption_in_transit,
            'backup_retention_days': config.backup_retention_days,
            'monitoring_enabled': config.monitoring_enabled,
            'recommendations': self._get_compliance_recommendations(all_regions),
        }
    
    def _get_compliance_recommendations(self, regions: List[Region]) -> List[str]:
        """Get compliance recommendations for deployment."""
        recommendations = []
        
        # Check for GDPR regions
        eu_regions = [r for r in regions if r in self.regions and self.regions[r].continent == 'Europe']
        if eu_regions:
            recommendations.append("GDPR compliance required for EU regions")
            recommendations.append("Consider data residency requirements for EU data")
        
        # Check for mixed jurisdictions
        if len(set(self.regions[r].continent for r in regions if r in self.regions)) > 1:
            recommendations.append("Multi-jurisdiction deployment - review cross-border data transfer rules")
        
        # Check for high-security regions
        high_security_regions = [
            r for r in regions if r in self.regions and 
            'BANKING_ACT' in self.regions[r].data_residency_rules
        ]
        if high_security_regions:
            recommendations.append("Enhanced security controls required for financial services regions")
        
        return recommendations


# Global region manager
_region_manager: Optional[RegionManager] = None


def get_region_manager() -> RegionManager:
    """Get global region manager."""
    global _region_manager
    if _region_manager is None:
        _region_manager = RegionManager()
    return _region_manager


def get_optimal_region(**kwargs) -> Region:
    """Get optimal region (convenience function)."""
    return get_region_manager().get_optimal_region(**kwargs)


def create_deployment(deployment_id: str, config: DeploymentConfig) -> bool:
    """Create deployment (convenience function)."""
    return get_region_manager().create_deployment(deployment_id, config)