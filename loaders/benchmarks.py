"""
Benchmark Configuration Loader
Centralized management for benchmark locations and settings
"""

import yaml
import pathlib
from typing import Dict, List, Any, Optional

class BenchmarkConfig:
    """Centralized benchmark configuration manager"""
    
    def __init__(self, config_path: str = "files/benchmarks.yaml"):
        repo_root = pathlib.Path(__file__).parent.parent
        self.config_path = repo_root / config_path
        self._config = None
        self._load_config()
    
    def _load_config(self):
        """Load YAML configuration file"""
        try:
            with open(self.config_path, 'r') as f:
                self._config = yaml.safe_load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Benchmark configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    # ===========================
    # Directory Paths
    # ===========================
    def get_benchmark_dir(self, benchmark_type: str) -> pathlib.Path:
        """Get benchmark directory path"""
        dirs = self._config['benchmark_dirs']
        if benchmark_type not in dirs:
            raise ValueError(f"Unknown benchmark type: {benchmark_type}")
        return pathlib.Path(dirs[benchmark_type])
    
    def get_root_dir(self) -> pathlib.Path:
        """Get root benchmark directory"""
        return self.get_benchmark_dir('root')
    
    def get_synthetic_dir(self) -> pathlib.Path:
        """Get synthetic benchmark directory"""
        return self.get_benchmark_dir('synthetic')
    
    def get_mmlu_dir(self) -> pathlib.Path:
        """Get MMLU benchmark directory"""
        return self.get_benchmark_dir('mmlu')
    
    # ===========================
    # Dataset Paths
    # ===========================
    def get_dataset_path(self, benchmark: str, dataset: str) -> pathlib.Path:
        """Get path to specific dataset"""
        datasets = self._config['datasets']
        if benchmark not in datasets:
            raise ValueError(f"Unknown benchmark: {benchmark}")
        if dataset not in datasets[benchmark]:
            raise ValueError(f"Unknown dataset '{dataset}' in benchmark '{benchmark}'")
        return pathlib.Path(datasets[benchmark][dataset])
    
    def get_synthetic_dataset_path(self, model_family: str) -> pathlib.Path:
        """Get synthetic dataset path for model family"""
        return self.get_dataset_path('synthetic', model_family.lower())
    
    # ===========================
    # Model Family Detection
    # ===========================
    def detect_model_family(self, model_name: str) -> Optional[str]:
        """Detect model family from model name"""
        families = self._config['model_families']
        
        for family, config in families.items():
            indicators = config.get('indicators', [])
            if any(indicator in model_name for indicator in indicators):
                return family
        return None
    
    def get_model_family_config(self, family: str) -> Dict[str, Any]:
        """Get configuration for specific model family"""
        families = self._config['model_families']
        if family not in families:
            raise ValueError(f"Unknown model family: {family}")
        return families[family]
    
    def get_model_synthetic_dataset(self, model_name: str) -> pathlib.Path:
        """Get synthetic dataset path for model"""
        family = self.detect_model_family(model_name)
        if not family:
            # Default fallback
            family = 'llama'
        
        family_config = self.get_model_family_config(family)
        dataset_type = family_config['synthetic_dataset']
        return self.get_synthetic_dataset_path(dataset_type)
    
    # ===========================
    # Benchmark Settings
    # ===========================
    def get_synthetic_settings(self) -> Dict[str, Any]:
        """Get synthetic benchmark settings"""
        return self._config['synthetic']
    
    def get_mmlu_settings(self) -> Dict[str, Any]:
        """Get MMLU benchmark settings"""
        return self._config['mmlu']
    
    def get_output_settings(self) -> Dict[str, Any]:
        """Get output configuration"""
        return self._config['outputs']
    
    def get_evaluation_defaults(self) -> Dict[str, Any]:
        """Get default evaluation settings"""
        return self._config['evaluation']
    
    # ===========================
    # Convenience Methods
    # ===========================

    
    def get_available_synthetic_tokens(self) -> List[int]:
        """Get available input token lengths for synthetic datasets"""
        synthetic_settings = self.get_synthetic_settings()
        return synthetic_settings.get('available_input_tokens', [384])
    
    def get_default_config_path(self, model_name: str) -> str:
        """Get default config path for model"""
        family = self.detect_model_family(model_name)
        if family:
            family_config = self.get_model_family_config(family)
            return family_config.get('default_config', self.get_evaluation_defaults()['default_config'])
        return self.get_evaluation_defaults()['default_config']

# Global benchmark configuration instance
_benchmark_config_instance = None

def get_benchmark_config() -> BenchmarkConfig:
    """Get singleton benchmark configuration instance"""
    global _benchmark_config_instance
    if _benchmark_config_instance is None:
        _benchmark_config_instance = BenchmarkConfig()
    return _benchmark_config_instance

def reload_benchmark_config(config_path: str = "files/benchmarks.yaml"):
    """Reload benchmark configuration from file"""
    global _benchmark_config_instance
    _benchmark_config_instance = BenchmarkConfig(config_path)
    return _benchmark_config_instance
