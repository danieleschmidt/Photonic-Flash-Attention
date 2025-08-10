"""Global deployment, internationalization, and compliance support."""

from .i18n import (
    PhotonicI18n,
    get_i18n,
    set_language,
    translate,
    t,
    get_available_languages,
    init_i18n
)

from .compliance import (
    ComplianceManager,
    ComplianceRegime,
    DataCategory,
    ProcessingPurpose,
    DataRecord,
    get_compliance_manager,
    register_data_processing
)

from .deployment import (
    RegionManager,
    Region,
    DeploymentTier,
    RegionInfo,
    DeploymentConfig,
    get_region_manager,
    get_optimal_region,
    create_deployment
)

__all__ = [
    # I18n
    "PhotonicI18n",
    "get_i18n",
    "set_language", 
    "translate",
    "t",
    "get_available_languages",
    "init_i18n",
    
    # Compliance
    "ComplianceManager",
    "ComplianceRegime",
    "DataCategory", 
    "ProcessingPurpose",
    "DataRecord",
    "get_compliance_manager",
    "register_data_processing",
    
    # Deployment
    "RegionManager",
    "Region",
    "DeploymentTier",
    "RegionInfo",
    "DeploymentConfig", 
    "get_region_manager",
    "get_optimal_region",
    "create_deployment"
]