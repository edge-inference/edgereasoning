"""
Configuration loader for LLM energy/power/latency models
"""

import yaml
import pathlib
from typing import Dict, List, Any, Optional

class ModelConfig:
    """Centralized configuration manager for LLM models"""
    
    def __init__(self, config_path: str = "config/analytic.yaml"):
        self.config_path = pathlib.Path(config_path)
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    # ===========================
    # Model Information
    # ===========================
    def get_supported_models(self) -> List[str]:
        """Get list of supported model names"""
        return self._config['models']['supported_models']
    
    def get_exponential_threshold(self) -> int:
        """Get threshold for switching to exponential decode model"""
        return self._config['thresholds']['exponential_model_switch']
    
    # ===========================
    # Token Ranges
    # ===========================
    def get_token_ranges(self, model_family: str) -> Dict[str, List[int]]:
        """Get input/output token ranges for model family (llama/qwen)"""
        family = model_family.lower()
        if family not in self._config['token_ranges']:
            raise ValueError(f"Unknown model family: {model_family}")
        return self._config['token_ranges'][family]
    
    def get_input_tokens(self, model_name: str) -> List[int]:
        """Get supported input token lengths for a model"""
        family = self._get_model_family(model_name)
        return self.get_token_ranges(family)['input_tokens']
    
    def get_output_tokens(self, model_name: str) -> List[int]:
        """Get supported output token lengths for a model"""
        family = self._get_model_family(model_name)
        return self.get_token_ranges(family)['output_tokens']
    
    def _get_model_family(self, model_name: str) -> str:
        """Extract model family (llama/qwen) from model name"""
        if 'llama' in model_name.lower():
            return 'llama'
        elif 'qwen' in model_name.lower():
            return 'qwen'
        else:
            raise ValueError(f"Cannot determine model family for: {model_name}. Add new case in _get_model_family()")
    
    # ===========================
    # File Paths
    # ===========================
    def get_data_file_path(self, file_type: str) -> pathlib.Path:
        """Get path to validation/parameter data files"""
        if file_type not in self._config['data_files']:
            raise ValueError(f"Unknown data file type: {file_type}")
        return pathlib.Path(self._config['data_files'][file_type])
    
    def get_prefill_validation_path(self) -> pathlib.Path:
        return self.get_data_file_path('prefill_validation')
    
    def get_decode_validation_path(self) -> pathlib.Path:
        return self.get_data_file_path('decode_validation')
    
    def get_exponential_params_path(self) -> pathlib.Path:
        return self.get_data_file_path('exponential_parameters')
    
    # ===========================
    # Name Mappings
    # ===========================
    def get_name_mapping(self, mapping_type: str, model_name: str) -> Optional[str]:
        """Get mapped name for model in specific context"""
        mappings = self._config['name_mappings'].get(mapping_type, {})
        return mappings.get(model_name, model_name)
    
    def get_prefill_validation_object(self, model_name: str) -> str:
        """Get the JSON key for this model in prefill.json"""
        return self.get_name_mapping('prefill_validation', model_name)
    
    def get_decode_validation_object(self, model_name: str) -> str:
        """Get the JSON key for this model in decode_parameters.json"""
        return self.get_name_mapping('exponential_parameters', model_name)
    
    # ===========================
    # Model Coefficients
    # ===========================
    def get_power_coefficients(self, phase: str, model_name: str) -> Dict[str, Any]:
        """Get power model coefficients (prefill/decode)"""
        if phase not in ['prefill', 'decode']:
            raise ValueError(f"Phase must be 'prefill' or 'decode', got: {phase}")
        
        coeffs = self._config['power_coefficients'][phase].get(model_name)
        if coeffs is None:
            raise ValueError(f"No {phase} power coefficients found for model: {model_name}")
        return coeffs
    
    def get_energy_coefficients(self, phase: str, model_name: str) -> Dict[str, Any]:
        """Get energy model coefficients (prefill/decode)"""  
        if phase not in ['prefill', 'decode']:
            raise ValueError(f"Phase must be 'prefill' or 'decode', got: {phase}")
        
        coeffs = self._config['energy_coefficients'][phase].get(model_name)
        if coeffs is None:
            raise ValueError(f"No {phase} energy coefficients found for model: {model_name}")
        return coeffs
    
    # ===========================
    # Convenience Methods
    # ===========================
    def get_all_power_coefficients(self, phase: str) -> Dict[str, Dict[str, Any]]:
        """Get all power coefficients for a phase"""
        return self._config['power_coefficients'][phase]
    
    def get_all_energy_coefficients(self, phase: str) -> Dict[str, Dict[str, Any]]:
        """Get all energy coefficients for a phase"""
        return self._config['energy_coefficients'][phase]

# Global configuration instance
_config_instance = None

def get_config() -> ModelConfig:
    """Get singleton configuration instance"""
    global _config_instance
    if _config_instance is None:
        _config_instance = ModelConfig()
    return _config_instance

def reload_config(config_path: str = "config/analytic.yaml"):
    """Reload configuration from file"""
    global _config_instance
    _config_instance = ModelConfig(config_path)
    return _config_instance
