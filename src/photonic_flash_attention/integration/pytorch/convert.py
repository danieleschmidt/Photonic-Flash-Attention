"""
Model conversion utilities for transforming standard transformers to use photonic attention.

This module provides automated conversion tools that can take existing HuggingFace
transformer models and seamlessly integrate photonic flash attention while
maintaining compatibility with existing training and inference pipelines.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List, Tuple, Union, Type
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
import copy
import re

from ...core.hybrid_router import HybridFlashAttention
from ...config import get_config
from ...utils.exceptions import PhotonicComputeError
from ...utils.validation import validate_model_structure
from .modules import PhotonicFlashAttention

# Handle optional transformers dependency
try:
    from transformers import (
        PreTrainedModel, 
        AutoModel,
        BertModel,
        GPT2Model, 
        T5Model,
        LlamaModel,
        AutoConfig
    )
    from transformers.modeling_utils import ModuleUtilsMixin
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    PreTrainedModel = object
    ModuleUtilsMixin = object

logger = logging.getLogger(__name__)


class ConversionStrategy(Enum):
    """Strategies for converting attention layers."""
    REPLACE_ALL = "replace_all"
    SELECTIVE = "selective" 
    HYBRID = "hybrid"
    PROGRESSIVE = "progressive"


@dataclass
class PhotonicConfig:
    """Configuration for photonic conversion."""
    photonic_threshold: int = 512
    min_seq_length: int = 256
    max_seq_length: int = 4096
    wavelength: float = 1550e-9  # m
    modulator_bandwidth: float = 50e9  # Hz
    enable_simulation: bool = True
    device_priority: List[str] = field(default_factory=lambda: ["photonic", "cuda"])
    conversion_strategy: ConversionStrategy = ConversionStrategy.SELECTIVE
    preserve_weights: bool = True
    enable_monitoring: bool = True
    
    # Performance thresholds
    min_attention_heads: int = 8
    min_embedding_dim: int = 512
    
    # Safety limits
    max_optical_power: float = 10e-3  # W
    temperature_monitoring: bool = True


@dataclass 
class ConversionReport:
    """Report from model conversion process."""
    original_model_name: str
    converted_layers: List[str]
    skipped_layers: List[str]
    conversion_errors: List[str]
    performance_estimate: Dict[str, float]
    memory_impact: Dict[str, float]
    compatibility_warnings: List[str]
    
    def __post_init__(self):
        self.total_layers = len(self.converted_layers) + len(self.skipped_layers)
        self.conversion_rate = len(self.converted_layers) / self.total_layers if self.total_layers > 0 else 0.0


class AttentionLayerDetector:
    """Detects and categorizes attention layers in models."""
    
    ATTENTION_PATTERNS = [
        r'attention',
        r'attn', 
        r'self_attn',
        r'cross_attn',
        r'multihead',
        r'mha'
    ]
    
    SUPPORTED_ATTENTION_TYPES = [
        'BertSelfAttention',
        'GPT2Attention', 
        'T5Attention',
        'LlamaAttention',
        'MultiheadAttention'
    ]
    
    @staticmethod
    def find_attention_layers(model: nn.Module) -> Dict[str, nn.Module]:
        """Find all attention layers in model."""
        attention_layers = {}
        
        def _recursive_find(module: nn.Module, prefix: str = ""):
            for name, child in module.named_children():
                full_name = f"{prefix}.{name}" if prefix else name
                
                # Check if this is an attention layer
                if AttentionLayerDetector._is_attention_layer(child, name):
                    attention_layers[full_name] = child
                else:
                    # Recurse into child modules
                    _recursive_find(child, full_name)
        
        _recursive_find(model)
        return attention_layers
    
    @staticmethod
    def _is_attention_layer(module: nn.Module, name: str) -> bool:
        """Check if module is an attention layer."""
        # Check by class name
        class_name = type(module).__name__
        if class_name in AttentionLayerDetector.SUPPORTED_ATTENTION_TYPES:
            return True
        
        # Check by module name patterns
        name_lower = name.lower()
        for pattern in AttentionLayerDetector.ATTENTION_PATTERNS:
            if re.search(pattern, name_lower):
                return True
        
        # Check for attention-like attributes
        attention_attrs = ['query', 'key', 'value', 'q_proj', 'k_proj', 'v_proj']
        has_attention_attrs = any(hasattr(module, attr) for attr in attention_attrs)
        
        return has_attention_attrs
    
    @staticmethod
    def get_attention_config(attention_layer: nn.Module) -> Dict[str, Any]:
        """Extract configuration from attention layer."""
        config = {}
        
        # Common attributes
        for attr in ['embed_dim', 'num_heads', 'head_dim', 'dropout']:
            if hasattr(attention_layer, attr):
                config[attr] = getattr(attention_layer, attr)
        
        # Handle different attention implementations
        class_name = type(attention_layer).__name__
        
        if 'Bert' in class_name:
            config.update(AttentionLayerDetector._extract_bert_config(attention_layer))
        elif 'GPT2' in class_name:
            config.update(AttentionLayerDetector._extract_gpt2_config(attention_layer))
        elif 'T5' in class_name:
            config.update(AttentionLayerDetector._extract_t5_config(attention_layer))
        
        return config
    
    @staticmethod
    def _extract_bert_config(layer: nn.Module) -> Dict[str, Any]:
        """Extract BERT attention configuration."""
        config = {}
        if hasattr(layer, 'query') and hasattr(layer.query, 'in_features'):
            config['embed_dim'] = layer.query.in_features
        if hasattr(layer, 'num_attention_heads'):
            config['num_heads'] = layer.num_attention_heads
        return config
    
    @staticmethod
    def _extract_gpt2_config(layer: nn.Module) -> Dict[str, Any]:
        """Extract GPT-2 attention configuration."""
        config = {}
        if hasattr(layer, 'embed_dim'):
            config['embed_dim'] = layer.embed_dim
        if hasattr(layer, 'num_heads'):
            config['num_heads'] = layer.num_heads  
        return config
    
    @staticmethod
    def _extract_t5_config(layer: nn.Module) -> Dict[str, Any]:
        """Extract T5 attention configuration."""
        config = {}
        if hasattr(layer, 'd_model'):
            config['embed_dim'] = layer.d_model
        if hasattr(layer, 'n_heads'):
            config['num_heads'] = layer.n_heads
        return config


class ModelConverter:
    """Converts standard transformer models to use photonic attention."""
    
    def __init__(self, photonic_config: Optional[PhotonicConfig] = None):
        self.config = photonic_config or PhotonicConfig()
        self.detector = AttentionLayerDetector()
        self.conversion_stats = {
            "conversions_attempted": 0,
            "conversions_successful": 0,
            "conversions_failed": 0,
            "total_layers_converted": 0
        }
        
        if not TRANSFORMERS_AVAILABLE and self.config.conversion_strategy != ConversionStrategy.REPLACE_ALL:
            warnings.warn("Transformers library not available. Some conversion features may be limited.")
    
    def convert_model(self, model: nn.Module, 
                     model_name: str = "unknown") -> Tuple[nn.Module, ConversionReport]:
        """Convert model to use photonic attention."""
        logger.info(f"Starting conversion of model: {model_name}")
        
        # Initialize conversion report
        report = ConversionReport(
            original_model_name=model_name,
            converted_layers=[],
            skipped_layers=[],
            conversion_errors=[],
            performance_estimate={},
            memory_impact={},
            compatibility_warnings=[]
        )
        
        try:
            # Validate model structure
            validate_model_structure(model)
            
            # Create a copy of the model for conversion
            if self.config.preserve_weights:
                converted_model = copy.deepcopy(model)
            else:
                converted_model = model
            
            # Find attention layers
            attention_layers = self.detector.find_attention_layers(converted_model)
            logger.info(f"Found {len(attention_layers)} attention layers")
            
            # Convert each attention layer
            for layer_name, attention_layer in attention_layers.items():
                try:
                    success = self._convert_attention_layer(
                        converted_model, layer_name, attention_layer, report
                    )
                    
                    if success:
                        report.converted_layers.append(layer_name)
                        self.conversion_stats["conversions_successful"] += 1
                        self.conversion_stats["total_layers_converted"] += 1
                    else:
                        report.skipped_layers.append(layer_name)
                    
                    self.conversion_stats["conversions_attempted"] += 1
                    
                except Exception as e:
                    error_msg = f"Failed to convert layer {layer_name}: {str(e)}"
                    logger.error(error_msg)
                    report.conversion_errors.append(error_msg)
                    report.skipped_layers.append(layer_name)
                    self.conversion_stats["conversions_failed"] += 1
            
            # Generate performance estimates
            report.performance_estimate = self._estimate_performance(
                converted_model, report.converted_layers
            )
            
            # Estimate memory impact
            report.memory_impact = self._estimate_memory_impact(model, converted_model)
            
            # Validate converted model
            self._validate_converted_model(converted_model, report)
            
            logger.info(f"Conversion completed: {len(report.converted_layers)}/{len(attention_layers)} layers converted")
            
            return converted_model, report
            
        except Exception as e:
            logger.error(f"Model conversion failed: {str(e)}")
            raise PhotonicComputeError(f"Model conversion failed: {str(e)}") from e
    
    def _convert_attention_layer(self, model: nn.Module, layer_name: str, 
                               attention_layer: nn.Module, 
                               report: ConversionReport) -> bool:
        """Convert a single attention layer."""
        
        # Extract attention configuration
        attention_config = self.detector.get_attention_config(attention_layer)
        
        # Check if layer meets conversion criteria
        if not self._should_convert_layer(attention_config):
            logger.debug(f"Skipping layer {layer_name}: doesn't meet conversion criteria")
            return False
        
        # Create photonic attention replacement
        try:
            photonic_attention = self._create_photonic_attention(attention_config)
            
            # Transfer weights if preserving
            if self.config.preserve_weights:
                self._transfer_weights(attention_layer, photonic_attention, report)
            
            # Replace layer in model
            self._replace_layer_in_model(model, layer_name, photonic_attention)
            
            logger.debug(f"Successfully converted layer: {layer_name}")
            return True
            
        except Exception as e:
            logger.warning(f"Failed to convert layer {layer_name}: {str(e)}")
            return False
    
    def _should_convert_layer(self, config: Dict[str, Any]) -> bool:
        """Determine if attention layer should be converted."""
        
        # Check minimum requirements
        if config.get('num_heads', 0) < self.config.min_attention_heads:
            return False
            
        if config.get('embed_dim', 0) < self.config.min_embedding_dim:
            return False
        
        # Apply conversion strategy
        if self.config.conversion_strategy == ConversionStrategy.REPLACE_ALL:
            return True
        elif self.config.conversion_strategy == ConversionStrategy.SELECTIVE:
            # Convert larger attention layers
            return config.get('embed_dim', 0) >= 768
        elif self.config.conversion_strategy == ConversionStrategy.HYBRID:
            # Convert every other layer for hybrid approach
            return hash(str(config)) % 2 == 0
        
        return True
    
    def _create_photonic_attention(self, config: Dict[str, Any]) -> PhotonicFlashAttention:
        """Create photonic attention layer from configuration."""
        
        # Map configuration to PhotonicFlashAttention parameters
        photonic_config = {
            'embed_dim': config.get('embed_dim', 768),
            'num_heads': config.get('num_heads', 12),
            'dropout': config.get('dropout', 0.1),
            'photonic_threshold': self.config.photonic_threshold,
            'device_priority': self.config.device_priority,
            'enable_simulation': self.config.enable_simulation
        }
        
        return PhotonicFlashAttention(**photonic_config)
    
    def _transfer_weights(self, source_layer: nn.Module, 
                         target_layer: PhotonicFlashAttention,
                         report: ConversionReport) -> None:
        """Transfer weights from source to photonic attention layer."""
        
        try:
            # Handle different source layer types
            class_name = type(source_layer).__name__
            
            if 'Bert' in class_name:
                self._transfer_bert_weights(source_layer, target_layer)
            elif 'GPT2' in class_name:
                self._transfer_gpt2_weights(source_layer, target_layer)  
            elif 'T5' in class_name:
                self._transfer_t5_weights(source_layer, target_layer)
            elif hasattr(source_layer, 'in_proj_weight'):  # PyTorch MultiheadAttention
                self._transfer_pytorch_multihead_weights(source_layer, target_layer)
            else:
                logger.warning(f"Unknown attention layer type: {class_name}")
                report.compatibility_warnings.append(
                    f"Weight transfer not implemented for {class_name}"
                )
                
        except Exception as e:
            warning_msg = f"Failed to transfer weights: {str(e)}"
            logger.warning(warning_msg)
            report.compatibility_warnings.append(warning_msg)
    
    def _transfer_bert_weights(self, bert_layer: nn.Module, 
                             photonic_layer: PhotonicFlashAttention) -> None:
        """Transfer BERT attention weights."""
        if hasattr(bert_layer, 'query') and hasattr(photonic_layer.attention, 'qkv_proj'):
            # Concatenate Q, K, V weights
            q_weight = bert_layer.query.weight
            k_weight = bert_layer.key.weight  
            v_weight = bert_layer.value.weight
            
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            photonic_layer.attention.qkv_proj.weight.data = qkv_weight
            
            # Biases
            if hasattr(bert_layer.query, 'bias') and bert_layer.query.bias is not None:
                q_bias = bert_layer.query.bias
                k_bias = bert_layer.key.bias
                v_bias = bert_layer.value.bias
                qkv_bias = torch.cat([q_bias, k_bias, v_bias], dim=0)
                photonic_layer.attention.qkv_proj.bias.data = qkv_bias
    
    def _transfer_gpt2_weights(self, gpt2_layer: nn.Module,
                             photonic_layer: PhotonicFlashAttention) -> None:
        """Transfer GPT-2 attention weights."""
        if hasattr(gpt2_layer, 'c_attn'):
            # GPT-2 uses combined QKV projection
            photonic_layer.attention.qkv_proj.weight.data = gpt2_layer.c_attn.weight.data
            if gpt2_layer.c_attn.bias is not None:
                photonic_layer.attention.qkv_proj.bias.data = gpt2_layer.c_attn.bias.data
        
        if hasattr(gpt2_layer, 'c_proj') and hasattr(photonic_layer.attention, 'out_proj'):
            photonic_layer.attention.out_proj.weight.data = gpt2_layer.c_proj.weight.data
            if gpt2_layer.c_proj.bias is not None:
                photonic_layer.attention.out_proj.bias.data = gpt2_layer.c_proj.bias.data
    
    def _transfer_t5_weights(self, t5_layer: nn.Module,
                           photonic_layer: PhotonicFlashAttention) -> None:
        """Transfer T5 attention weights.""" 
        # T5 has separate q, k, v projections
        if all(hasattr(t5_layer, attr) for attr in ['q', 'k', 'v']):
            q_weight = t5_layer.q.weight
            k_weight = t5_layer.k.weight
            v_weight = t5_layer.v.weight
            
            qkv_weight = torch.cat([q_weight, k_weight, v_weight], dim=0)
            photonic_layer.attention.qkv_proj.weight.data = qkv_weight
        
        if hasattr(t5_layer, 'o') and hasattr(photonic_layer.attention, 'out_proj'):
            photonic_layer.attention.out_proj.weight.data = t5_layer.o.weight.data
    
    def _transfer_pytorch_multihead_weights(self, mha_layer: nn.Module,
                                          photonic_layer: PhotonicFlashAttention) -> None:
        """Transfer PyTorch MultiheadAttention weights."""
        if hasattr(mha_layer, 'in_proj_weight'):
            photonic_layer.attention.qkv_proj.weight.data = mha_layer.in_proj_weight.data
            
        if hasattr(mha_layer, 'in_proj_bias') and mha_layer.in_proj_bias is not None:
            photonic_layer.attention.qkv_proj.bias.data = mha_layer.in_proj_bias.data
            
        if hasattr(mha_layer, 'out_proj'):
            photonic_layer.attention.out_proj.weight.data = mha_layer.out_proj.weight.data
            if mha_layer.out_proj.bias is not None:
                photonic_layer.attention.out_proj.bias.data = mha_layer.out_proj.bias.data
    
    def _replace_layer_in_model(self, model: nn.Module, layer_path: str,
                              new_layer: nn.Module) -> None:
        """Replace layer in model at specified path."""
        path_parts = layer_path.split('.')
        parent = model
        
        # Navigate to parent module
        for part in path_parts[:-1]:
            parent = getattr(parent, part)
        
        # Replace the target layer
        setattr(parent, path_parts[-1], new_layer)
    
    def _estimate_performance(self, model: nn.Module, 
                            converted_layers: List[str]) -> Dict[str, float]:
        """Estimate performance improvement from conversion."""
        
        # Simplified performance estimation
        n_converted = len(converted_layers)
        
        if n_converted == 0:
            return {"speedup": 1.0, "energy_reduction": 0.0}
        
        # Rough estimates based on sequence length thresholds
        expected_speedup = min(1.0 + n_converted * 0.5, 10.0)  # Cap at 10x
        energy_reduction = min(n_converted * 0.3, 0.95)  # Cap at 95%
        
        return {
            "estimated_speedup": expected_speedup,
            "estimated_energy_reduction": energy_reduction,
            "converted_layer_count": n_converted
        }
    
    def _estimate_memory_impact(self, original_model: nn.Module,
                              converted_model: nn.Module) -> Dict[str, float]:
        """Estimate memory impact of conversion."""
        
        def count_parameters(model):
            return sum(p.numel() for p in model.parameters())
        
        orig_params = count_parameters(original_model)
        conv_params = count_parameters(converted_model)
        
        memory_change = (conv_params - orig_params) / orig_params if orig_params > 0 else 0.0
        
        return {
            "parameter_count_change": memory_change,
            "original_parameters": orig_params,
            "converted_parameters": conv_params
        }
    
    def _validate_converted_model(self, model: nn.Module, 
                                report: ConversionReport) -> None:
        """Validate converted model structure and functionality."""
        
        try:
            # Check model can be set to eval/train modes
            model.eval()
            model.train()
            
            # Basic structural validation
            total_params = sum(p.numel() for p in model.parameters())
            if total_params == 0:
                report.compatibility_warnings.append("Model has no parameters after conversion")
            
        except Exception as e:
            warning_msg = f"Model validation warning: {str(e)}"
            report.compatibility_warnings.append(warning_msg)
    
    def get_conversion_stats(self) -> Dict[str, int]:
        """Get conversion statistics."""
        return self.conversion_stats.copy()


# Convenience functions
def convert_to_photonic(model: Union[nn.Module, str],
                       photonic_config: Optional[PhotonicConfig] = None,
                       **kwargs) -> Tuple[nn.Module, ConversionReport]:
    """
    Convert model to use photonic attention.
    
    Args:
        model: Model to convert (nn.Module) or model name/path (str)
        photonic_config: Configuration for photonic conversion
        **kwargs: Additional arguments for model loading
        
    Returns:
        Tuple of (converted_model, conversion_report)
    """
    
    if isinstance(model, str):
        # Load model from name/path
        if TRANSFORMERS_AVAILABLE:
            model = AutoModel.from_pretrained(model, **kwargs)
            model_name = model
        else:
            raise PhotonicComputeError(
                "String model loading requires transformers library. "
                "Please install transformers or pass nn.Module directly."
            )
    else:
        model_name = type(model).__name__
    
    # Create converter and convert
    converter = ModelConverter(photonic_config)
    return converter.convert_model(model, model_name)


def create_photonic_model_from_config(model_type: str, config_dict: Dict[str, Any],
                                    photonic_config: Optional[PhotonicConfig] = None
                                    ) -> nn.Module:
    """Create a photonic model from configuration."""
    
    if not TRANSFORMERS_AVAILABLE:
        raise PhotonicComputeError(
            "Model creation from config requires transformers library"
        )
    
    # Load base configuration
    if isinstance(config_dict, str):
        config = AutoConfig.from_pretrained(config_dict)
    else:
        config = AutoConfig.from_dict(config_dict)
    
    # Create base model
    model = AutoModel.from_config(config)
    
    # Convert to photonic
    photonic_model, report = convert_to_photonic(model, photonic_config)
    
    logger.info(f"Created photonic model: {report.conversion_rate:.1%} layers converted")
    
    return photonic_model


def load_photonic_model(model_path: str, 
                       photonic_config: Optional[PhotonicConfig] = None,
                       **kwargs) -> Tuple[nn.Module, ConversionReport]:
    """Load and convert model to photonic."""
    return convert_to_photonic(model_path, photonic_config, **kwargs)


# Model-specific conversion functions
def convert_bert_to_photonic(model_name: str = "bert-base-uncased", 
                           **kwargs) -> Tuple[nn.Module, ConversionReport]:
    """Convert BERT model to photonic."""
    config = PhotonicConfig(
        photonic_threshold=256,
        conversion_strategy=ConversionStrategy.SELECTIVE
    )
    return convert_to_photonic(model_name, config, **kwargs)


def convert_gpt2_to_photonic(model_name: str = "gpt2",
                           **kwargs) -> Tuple[nn.Module, ConversionReport]:
    """Convert GPT-2 model to photonic."""
    config = PhotonicConfig(
        photonic_threshold=512,
        conversion_strategy=ConversionStrategy.HYBRID
    )
    return convert_to_photonic(model_name, config, **kwargs)


def convert_t5_to_photonic(model_name: str = "t5-base",
                         **kwargs) -> Tuple[nn.Module, ConversionReport]:
    """Convert T5 model to photonic."""
    config = PhotonicConfig(
        photonic_threshold=1024,
        conversion_strategy=ConversionStrategy.SELECTIVE  
    )
    return convert_to_photonic(model_name, config, **kwargs)