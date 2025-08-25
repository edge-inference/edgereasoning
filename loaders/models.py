"""
Models Configuration Loader
Handles loading and accessing model configurations from files/models.yaml
"""

import yaml
import pathlib
from typing import Dict, List, Any, Optional


class ModelsConfig:
    """Centralized models configuration manager"""
    
    def __init__(self, config_path: str = "files/models.yaml"):
        self.config_path = config_path
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load models configuration from YAML file"""
        repo_root = pathlib.Path(__file__).parent.parent
        config_file = repo_root / self.config_path
        
        try:
            with open(config_file, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Models configuration file not found: {config_file}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def get_reasoning_models(self) -> Dict[str, str]:
        """Get all reasoning models"""
        return self._config.get('reasoning_models', {})
    
    def get_direct_models(self) -> Dict[str, str]:
        """Get all direct models"""
        return self._config.get('direct_models', {})
    
    def get_default_reasoning_model(self) -> str:
        """Get default reasoning model"""
        default_key = self._config.get('defaults', {}).get('reasoning')
        if not default_key:
            raise ValueError("No default reasoning model configured")
        
        reasoning_models = self.get_reasoning_models()
        if default_key not in reasoning_models:
            raise ValueError(f"Default reasoning model '{default_key}' not found in reasoning_models")
        
        return reasoning_models[default_key]
    
    def get_default_direct_model(self) -> str:
        """Get default direct model"""
        default_key = self._config.get('defaults', {}).get('direct')
        if not default_key:
            raise ValueError("No default direct model configured")
        
        direct_models = self.get_direct_models()
        if default_key not in direct_models:
            raise ValueError(f"Default direct model '{default_key}' not found in direct_models")
        
        return direct_models[default_key]
    
    def get_all_reasoning_model_paths(self) -> List[str]:
        """Get list of all reasoning model paths"""
        return list(self.get_reasoning_models().values())
    
    def get_all_direct_model_paths(self) -> List[str]:
        """Get list of all direct model paths"""
        return list(self.get_direct_models().values())
    
    def get_model_by_name(self, model_name: str) -> Optional[str]:
        """Get model path by name from either reasoning or direct models"""
        all_models = {**self.get_reasoning_models(), **self.get_direct_models()}
        return all_models.get(model_name)
    
    def is_reasoning_model(self, model_path: str) -> bool:
        """Check if a model path is a reasoning model"""
        return model_path in self.get_reasoning_models().values()
    
    def is_direct_model(self, model_path: str) -> bool:
        """Check if a model path is a direct model"""
        return model_path in self.get_direct_models().values()


# Global instance for easy access
_models_config_instance = None

def get_models_config() -> ModelsConfig:
    """Get the global models configuration instance"""
    global _models_config_instance
    if _models_config_instance is None:
        _models_config_instance = ModelsConfig()
    return _models_config_instance


# Convenience functions
def get_reasoning_models() -> Dict[str, str]:
    """Get all reasoning models"""
    return get_models_config().get_reasoning_models()


def get_direct_models() -> Dict[str, str]:
    """Get all direct models"""
    return get_models_config().get_direct_models()


def get_default_reasoning_model() -> str:
    """Get default reasoning model path"""
    return get_models_config().get_default_reasoning_model()


def get_default_direct_model() -> str:
    """Get default direct model path"""
    return get_models_config().get_default_direct_model()


def get_all_reasoning_model_paths() -> List[str]:
    """Get list of all reasoning model paths"""
    return get_models_config().get_all_reasoning_model_paths()


def get_all_direct_model_paths() -> List[str]:
    """Get list of all direct model paths"""
    return get_models_config().get_all_direct_model_paths()
