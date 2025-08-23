"""
Results Configuration Loader
Used for result storage and post-processing paths
"""

import yaml
import pathlib
from typing import Dict, List, Any, Optional
from setup import HardwareDetector

class ResultsConfig:
    """Centralized results configuration manager"""
    
    def __init__(self, config_path: str = "files/results.yaml"):
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
            raise FileNotFoundError(f"Results configuration file not found: {self.config_path}")
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML configuration: {e}")
    
    def get_result_base_dir(self, benchmark_type: str, platform: str = None, model_name: str = None) -> pathlib.Path:
        """Get base directory for storing benchmark results"""
        if platform is None:
            platform = self._detect_platform()
        
        repo_root = pathlib.Path(__file__).parent.parent
        
        if benchmark_type in self._config['results']:
            benchmark_config = self._config['results'][benchmark_type]
            if platform and platform in benchmark_config:
                input_dir = benchmark_config[platform]['input_dir']
                base_path = repo_root / input_dir
                
                if benchmark_config[platform].get('model_specific', False) and model_name:
                    model_dir_name = self._get_model_dir_name(model_name)
                    return base_path / model_dir_name
                
                return base_path
        
        base_dir = repo_root / self._config['results']['base_dir']
        return base_dir / benchmark_type / (platform or "")
    
    def get_processed_output_dir(self, benchmark_type: str, platform: str = None) -> pathlib.Path:
        """Get directory for processed results"""
        if platform is None:
            platform = self._detect_platform()
        
        repo_root = pathlib.Path(__file__).parent.parent
            
        if benchmark_type in self._config['results']:
            benchmark_config = self._config['results'][benchmark_type]
            if platform and platform in benchmark_config:
                output_dir = benchmark_config[platform]['output_dir']
                return repo_root / output_dir
        
        # Fallback
        return self.get_result_base_dir(benchmark_type, platform) / "processed"

    def get_synthetic_input_dir(self, kind: str) -> pathlib.Path:
        """Get base directory for synthetic benchmark raw results (e.g., prefill/decode)."""
        repo_root = pathlib.Path(__file__).parent.parent
        try:
            path = self._config['results']['synthetic'][kind]['input_dir']
            return repo_root / path
        except Exception:
            return repo_root / 'data' / 'synthetic' / kind

    def get_synthetic_output_dir(self, kind: str) -> pathlib.Path:
        """Get directory for processed synthetic results (e.g., prefill/decode)."""
        repo_root = pathlib.Path(__file__).parent.parent
        try:
            path = self._config['results']['synthetic'][kind]['output_dir']
            return repo_root / path
        except Exception:
            return self.get_synthetic_input_dir(kind) / 'processed'
    
    def _detect_platform(self) -> str:
        """Detect current platform"""
        try:
            detector = HardwareDetector()
            return detector.detect_platform()
        except Exception:
            return "unknown"
    
    def _get_model_dir_name(self, model_name: str) -> str:
        """Convert model name to directory-safe name using mapping"""
        model_names = self._config.get('model_names', {})
        if model_name in model_names:
            return model_names[model_name]
        
        if '/' in model_name:
            clean_name = model_name.split('/')[-1]
        else:
            clean_name = model_name
        
        clean_name = clean_name.replace(':', '-').replace(' ', '_')
        return clean_name

_results_config_instance = None

def get_results_config() -> ResultsConfig:
    """Get singleton results configuration instance"""
    global _results_config_instance
    if _results_config_instance is None:
        _results_config_instance = ResultsConfig()
    return _results_config_instance
