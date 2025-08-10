"""Compliance and regulatory support for global deployment."""

import hashlib
import json
import time
import uuid
from typing import Dict, Any, List, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import logging

from ..utils.logging import get_logger
from ..utils.security import secure_hash


logger = get_logger(__name__)


class ComplianceRegime(Enum):
    """Supported compliance regimes."""
    GDPR = "gdpr"  # European Union
    CCPA = "ccpa"  # California, USA
    PDPA = "pdpa"  # Singapore
    LGPD = "lgpd"  # Brazil
    PIPEDA = "pipeda"  # Canada
    DATA_ACT = "data_act"  # EU Data Act
    AI_ACT = "ai_act"  # EU AI Act


class DataCategory(Enum):
    """Categories of data for compliance."""
    PERSONAL = "personal"
    SENSITIVE = "sensitive"
    BIOMETRIC = "biometric"
    TECHNICAL = "technical"
    PERFORMANCE = "performance"
    ANONYMOUS = "anonymous"


class ProcessingPurpose(Enum):
    """Purposes for data processing."""
    PERFORMANCE_OPTIMIZATION = "performance_optimization"
    SYSTEM_MONITORING = "system_monitoring"
    ERROR_ANALYSIS = "error_analysis"
    RESEARCH = "research"
    SERVICE_PROVISION = "service_provision"


@dataclass
class DataRecord:
    """Record of data processing for compliance."""
    record_id: str
    timestamp: float
    data_category: DataCategory
    processing_purpose: ProcessingPurpose
    data_subject_id: Optional[str] = None
    retention_period: Optional[int] = None  # days
    geographic_location: Optional[str] = None
    compliance_regimes: List[ComplianceRegime] = field(default_factory=list)
    consent_given: bool = False
    anonymized: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


class ComplianceManager:
    """Manages compliance with global data protection regulations."""
    
    def __init__(self):
        """Initialize compliance manager."""
        self.data_records: List[DataRecord] = []
        self.consent_records: Dict[str, Dict[str, Any]] = {}
        self.retention_policies: Dict[DataCategory, int] = {}  # days
        self.geographic_restrictions: Dict[str, Set[str]] = {}
        self.compliance_config: Dict[str, Any] = {}
        
        # Load default configurations
        self._load_default_configs()
        
        logger.info("Compliance manager initialized")
    
    def _load_default_configs(self) -> None:
        """Load default compliance configurations."""
        # Default retention periods (in days)
        self.retention_policies = {
            DataCategory.PERSONAL: 365 * 2,  # 2 years
            DataCategory.SENSITIVE: 365,  # 1 year
            DataCategory.BIOMETRIC: 90,  # 3 months
            DataCategory.TECHNICAL: 365 * 5,  # 5 years
            DataCategory.PERFORMANCE: 365,  # 1 year
            DataCategory.ANONYMOUS: -1,  # No limit for truly anonymous data
        }
        
        # Geographic data restrictions
        self.geographic_restrictions = {
            'EU': {'personal', 'sensitive', 'biometric'},
            'US': {'personal', 'sensitive'},
            'CN': {'personal', 'sensitive', 'technical'},
            'RU': {'personal', 'technical'},
        }
        
        # Compliance regime requirements
        self.compliance_config = {
            ComplianceRegime.GDPR: {
                'requires_consent': True,
                'requires_data_protection_officer': True,
                'max_retention_days': 365 * 2,
                'requires_impact_assessment': True,
                'right_to_deletion': True,
                'right_to_portability': True,
                'breach_notification_hours': 72,
            },
            ComplianceRegime.CCPA: {
                'requires_consent': False,  # Opt-out model
                'requires_privacy_policy': True,
                'right_to_deletion': True,
                'right_to_know': True,
                'right_to_opt_out': True,
                'non_discrimination': True,
            },
            ComplianceRegime.PDPA: {
                'requires_consent': True,
                'requires_data_protection_officer': True,
                'breach_notification_hours': 72,
                'right_to_deletion': True,
            },
            ComplianceRegime.AI_ACT: {
                'risk_assessment_required': True,
                'transparency_required': True,
                'human_oversight_required': True,
                'accuracy_monitoring_required': True,
                'bias_monitoring_required': True,
            }
        }
    
    def register_data_processing(
        self,
        data_category: DataCategory,
        processing_purpose: ProcessingPurpose,
        data_subject_id: Optional[str] = None,
        geographic_location: Optional[str] = None,
        compliance_regimes: Optional[List[ComplianceRegime]] = None,
        consent_given: bool = False,
        anonymized: bool = False,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Register data processing activity for compliance.
        
        Args:
            data_category: Category of data being processed
            processing_purpose: Purpose for processing
            data_subject_id: Optional identifier for data subject
            geographic_location: Geographic location of processing
            compliance_regimes: Applicable compliance regimes
            consent_given: Whether consent was given
            anonymized: Whether data is anonymized
            metadata: Additional metadata
            
        Returns:
            Record ID for the processing activity
        """
        record_id = str(uuid.uuid4())
        
        # Determine retention period
        retention_period = self.retention_policies.get(data_category)
        
        # Auto-detect compliance regimes based on location
        if not compliance_regimes:
            compliance_regimes = self._detect_compliance_regimes(geographic_location)
        
        record = DataRecord(
            record_id=record_id,
            timestamp=time.time(),
            data_category=data_category,
            processing_purpose=processing_purpose,
            data_subject_id=data_subject_id,
            retention_period=retention_period,
            geographic_location=geographic_location,
            compliance_regimes=compliance_regimes or [],
            consent_given=consent_given,
            anonymized=anonymized,
            metadata=metadata or {}
        )
        
        self.data_records.append(record)
        
        logger.debug(f"Registered data processing: {record_id} ({data_category.value})")
        
        # Check compliance
        compliance_issues = self._check_record_compliance(record)
        if compliance_issues:
            logger.warning(f"Compliance issues for record {record_id}: {compliance_issues}")
        
        return record_id
    
    def _detect_compliance_regimes(
        self, 
        geographic_location: Optional[str]
    ) -> List[ComplianceRegime]:
        """Auto-detect applicable compliance regimes."""
        regimes = []
        
        if not geographic_location:
            return regimes
        
        location_upper = geographic_location.upper()
        
        # EU countries and GDPR
        eu_countries = {
            'AT', 'BE', 'BG', 'CY', 'CZ', 'DE', 'DK', 'EE', 'ES', 'FI',
            'FR', 'GR', 'HR', 'HU', 'IE', 'IT', 'LT', 'LU', 'LV', 'MT',
            'NL', 'PL', 'PT', 'RO', 'SE', 'SI', 'SK', 'EU'
        }
        
        if any(country in location_upper for country in eu_countries):
            regimes.append(ComplianceRegime.GDPR)
            regimes.append(ComplianceRegime.AI_ACT)
        
        # California CCPA
        if 'CA' in location_upper or 'CALIFORNIA' in location_upper:
            regimes.append(ComplianceRegime.CCPA)
        
        # Singapore PDPA
        if 'SG' in location_upper or 'SINGAPORE' in location_upper:
            regimes.append(ComplianceRegime.PDPA)
        
        # Brazil LGPD
        if 'BR' in location_upper or 'BRAZIL' in location_upper:
            regimes.append(ComplianceRegime.LGPD)
        
        # Canada PIPEDA
        if 'CA' in location_upper or 'CANADA' in location_upper:
            regimes.append(ComplianceRegime.PIPEDA)
        
        return regimes
    
    def _check_record_compliance(self, record: DataRecord) -> List[str]:
        """Check compliance issues for a data record."""
        issues = []
        
        for regime in record.compliance_regimes:
            config = self.compliance_config.get(regime, {})
            
            # Check consent requirements
            if config.get('requires_consent') and not record.consent_given and not record.anonymized:
                issues.append(f"{regime.value}: Consent required but not given")
            
            # Check retention period
            max_retention = config.get('max_retention_days')
            if max_retention and record.retention_period and record.retention_period > max_retention:
                issues.append(f"{regime.value}: Retention period exceeds limit")
            
            # Check data localization requirements
            if record.geographic_location:
                restrictions = self.geographic_restrictions.get(record.geographic_location, set())
                if record.data_category.value in restrictions:
                    issues.append(f"Data localization restriction for {record.data_category.value} in {record.geographic_location}")
        
        return issues
    
    def record_consent(
        self,
        data_subject_id: str,
        purposes: List[ProcessingPurpose],
        consent_given: bool = True,
        geographic_location: Optional[str] = None,
        expiry_timestamp: Optional[float] = None
    ) -> str:
        """
        Record consent from data subject.
        
        Args:
            data_subject_id: Identifier for data subject
            purposes: Processing purposes consented to
            consent_given: Whether consent was given
            geographic_location: Location where consent was given
            expiry_timestamp: When consent expires
            
        Returns:
            Consent record ID
        """
        consent_id = str(uuid.uuid4())
        
        consent_record = {
            'consent_id': consent_id,
            'data_subject_id': data_subject_id,
            'timestamp': time.time(),
            'purposes': [p.value for p in purposes],
            'consent_given': consent_given,
            'geographic_location': geographic_location,
            'expiry_timestamp': expiry_timestamp,
            'ip_address_hash': None,  # Could store hashed IP for verification
            'user_agent_hash': None,
        }
        
        self.consent_records[consent_id] = consent_record
        
        logger.info(f"Recorded consent: {consent_id} for subject {data_subject_id}")
        
        return consent_id
    
    def check_consent(
        self,
        data_subject_id: str,
        purpose: ProcessingPurpose
    ) -> bool:
        """
        Check if consent exists for data processing.
        
        Args:
            data_subject_id: Data subject identifier
            purpose: Processing purpose
            
        Returns:
            True if valid consent exists
        """
        for consent_record in self.consent_records.values():
            if (consent_record['data_subject_id'] == data_subject_id and
                consent_record['consent_given'] and
                purpose.value in consent_record['purposes']):
                
                # Check expiry
                expiry = consent_record.get('expiry_timestamp')
                if expiry and time.time() > expiry:
                    continue
                
                return True
        
        return False
    
    def anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Anonymize data to remove personal identifiers.
        
        Args:
            data: Data dictionary to anonymize
            
        Returns:
            Anonymized data dictionary
        """
        anonymized = {}
        
        # Fields that should be removed or hashed
        sensitive_fields = {
            'user_id', 'email', 'name', 'ip_address', 'device_id',
            'session_id', 'user_agent', 'location'
        }
        
        # Fields that should be aggregated/generalized
        generalizable_fields = {
            'timestamp': lambda t: int(t // 3600) * 3600,  # Round to hour
            'age': lambda a: (a // 10) * 10,  # Round to decade
            'location': lambda l: l.split(',')[0] if ',' in l else l  # Keep only country
        }
        
        for key, value in data.items():
            if key.lower() in sensitive_fields:
                # Hash sensitive data
                anonymized[f"{key}_hash"] = secure_hash(str(value))
            elif key.lower() in generalizable_fields:
                # Apply generalization
                anonymized[key] = generalizable_fields[key.lower()](value)
            else:
                # Keep as-is for non-sensitive data
                anonymized[key] = value
        
        # Add anonymization timestamp
        anonymized['_anonymized_at'] = time.time()
        
        return anonymized
    
    def generate_privacy_report(self, regime: ComplianceRegime) -> Dict[str, Any]:
        """
        Generate privacy compliance report.
        
        Args:
            regime: Compliance regime to report on
            
        Returns:
            Compliance report
        """
        # Filter records for this regime
        relevant_records = [
            r for r in self.data_records 
            if regime in r.compliance_regimes
        ]
        
        # Analyze data categories
        category_counts = {}
        purpose_counts = {}
        consent_stats = {'given': 0, 'not_given': 0}
        
        for record in relevant_records:
            # Category counts
            category = record.data_category.value
            category_counts[category] = category_counts.get(category, 0) + 1
            
            # Purpose counts
            purpose = record.processing_purpose.value
            purpose_counts[purpose] = purpose_counts.get(purpose, 0) + 1
            
            # Consent stats
            if record.consent_given:
                consent_stats['given'] += 1
            else:
                consent_stats['not_given'] += 1
        
        # Check compliance status
        compliance_issues = []
        for record in relevant_records:
            issues = self._check_record_compliance(record)
            compliance_issues.extend(issues)
        
        report = {
            'regime': regime.value,
            'generated_at': time.time(),
            'total_records': len(relevant_records),
            'data_categories': category_counts,
            'processing_purposes': purpose_counts,
            'consent_statistics': consent_stats,
            'compliance_issues': compliance_issues,
            'compliance_status': 'compliant' if not compliance_issues else 'issues_found',
            'retention_policies': {
                category.value: days for category, days in self.retention_policies.items()
            },
            'consent_records': len(self.consent_records),
        }
        
        logger.info(f"Generated privacy report for {regime.value}: {report['compliance_status']}")
        
        return report
    
    def cleanup_expired_data(self) -> int:
        """
        Clean up expired data based on retention policies.
        
        Returns:
            Number of records cleaned up
        """
        current_time = time.time()
        initial_count = len(self.data_records)
        
        # Remove expired records
        self.data_records = [
            record for record in self.data_records
            if (not record.retention_period or 
                record.retention_period < 0 or  # No limit
                (current_time - record.timestamp) < (record.retention_period * 86400))
        ]
        
        # Remove expired consents
        expired_consents = []
        for consent_id, consent_record in self.consent_records.items():
            expiry = consent_record.get('expiry_timestamp')
            if expiry and current_time > expiry:
                expired_consents.append(consent_id)
        
        for consent_id in expired_consents:
            del self.consent_records[consent_id]
        
        cleaned_count = initial_count - len(self.data_records)
        
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} expired data records and {len(expired_consents)} expired consents")
        
        return cleaned_count
    
    def export_user_data(self, data_subject_id: str) -> Dict[str, Any]:
        """
        Export all data for a user (GDPR Article 20 - Right to Data Portability).
        
        Args:
            data_subject_id: Data subject identifier
            
        Returns:
            All data associated with the user
        """
        user_records = [
            record for record in self.data_records
            if record.data_subject_id == data_subject_id
        ]
        
        user_consents = [
            consent for consent in self.consent_records.values()
            if consent['data_subject_id'] == data_subject_id
        ]
        
        export_data = {
            'data_subject_id': data_subject_id,
            'export_timestamp': time.time(),
            'processing_records': [
                {
                    'record_id': r.record_id,
                    'timestamp': r.timestamp,
                    'data_category': r.data_category.value,
                    'processing_purpose': r.processing_purpose.value,
                    'geographic_location': r.geographic_location,
                    'compliance_regimes': [cr.value for cr in r.compliance_regimes],
                    'consent_given': r.consent_given,
                    'anonymized': r.anonymized,
                    'metadata': r.metadata
                }
                for r in user_records
            ],
            'consent_records': user_consents,
            'total_processing_records': len(user_records),
            'total_consent_records': len(user_consents),
        }
        
        logger.info(f"Exported data for user {data_subject_id}: {len(user_records)} processing records")
        
        return export_data
    
    def delete_user_data(self, data_subject_id: str) -> int:
        """
        Delete all data for a user (Right to Erasure).
        
        Args:
            data_subject_id: Data subject identifier
            
        Returns:
            Number of records deleted
        """
        initial_count = len(self.data_records)
        
        # Remove data records
        self.data_records = [
            record for record in self.data_records
            if record.data_subject_id != data_subject_id
        ]
        
        # Remove consent records
        consent_keys_to_remove = [
            consent_id for consent_id, consent in self.consent_records.items()
            if consent['data_subject_id'] == data_subject_id
        ]
        
        for consent_id in consent_keys_to_remove:
            del self.consent_records[consent_id]
        
        deleted_count = initial_count - len(self.data_records)
        
        logger.info(f"Deleted {deleted_count} data records and {len(consent_keys_to_remove)} consent records for user {data_subject_id}")
        
        return deleted_count


# Global compliance manager
_compliance_manager: Optional[ComplianceManager] = None


def get_compliance_manager() -> ComplianceManager:
    """Get global compliance manager."""
    global _compliance_manager
    if _compliance_manager is None:
        _compliance_manager = ComplianceManager()
    return _compliance_manager


def register_data_processing(
    data_category: DataCategory,
    processing_purpose: ProcessingPurpose,
    **kwargs
) -> str:
    """Register data processing (convenience function)."""
    return get_compliance_manager().register_data_processing(
        data_category, processing_purpose, **kwargs
    )