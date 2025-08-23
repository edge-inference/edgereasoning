"""
Power consumption model for LLM inference

"""

import numpy as np
import math
import bisect
import json
import pathlib
from loaders.analytic import get_config
import latency_model

model_config = get_config()

# --------------------------------------------------
# Exponential decode power model (for output_tokens >= 65)
# --------------------------------------------------
_DECODE_POWER_PARAMS_PATH = model_config.get_exponential_params_path()
if _DECODE_POWER_PARAMS_PATH.exists():
    with _DECODE_POWER_PARAMS_PATH.open() as _f:
        _decode_power_data = json.load(_f)

    _EXPONENTIAL_DECODE_PARAMS = {}
    for model_key, configs in _decode_power_data.items():
        if model_key == "_metadata":
            continue
        _EXPONENTIAL_DECODE_PARAMS[model_key] = configs

    _EXPONENTIAL_MODEL_MAP = {
        model: model_config.get_decode_validation_object(model) 
        for model in model_config.get_supported_models()
    }

    def _find_closest_decode_config(model_name: str, input_length: int, output_length: int):
        """Find the closest exponential decode configuration."""
        exp_model_key = _EXPONENTIAL_MODEL_MAP.get(model_name)
        if exp_model_key is None or exp_model_key not in _EXPONENTIAL_DECODE_PARAMS:
            return None
        
        configs = _EXPONENTIAL_DECODE_PARAMS[exp_model_key]
        
        best_config = None
        min_distance = float('inf')
        
        for config in configs:
            inp_dist = abs(config['input_tokens'] - input_length)
            out_dist = abs(config['output_tokens'] - output_length)
            distance = inp_dist + out_dist
            
            if distance < min_distance:
                min_distance = distance
                best_config = config
        
        return best_config

    def _decode_energy_function(P_inf: float, delta_P: float, tau: float, total_time: float) -> float:
        """Calculate energy using exponential power model: E = P_∞×T - ΔP×τ×(1 - exp(-T/τ))"""
        if total_time <= 0 or tau <= 0:
            return 0.0
        
        exponential_term = 1 - math.exp(-total_time / tau)
        energy = P_inf * total_time - delta_P * tau * exponential_term
        return max(0.0, energy)

    def _lookup_parameters(model_name: str, input_length: int, output_length: int) -> float:
        """Get measured decode_energy from decode_power_parameters.json for validation."""
        config = _find_closest_decode_config(model_name, input_length, output_length)
        if config is not None:
            return config.get('decode_energy', None)
        return None

else:
    _EXPONENTIAL_DECODE_PARAMS = {}
    _EXPONENTIAL_MODEL_MAP = {}
    _find_closest_decode_config = lambda *args: None
    _decode_energy_function = lambda *args: 0.0
    _lookup_parameters = lambda *args: None

prefill_power_coeffs = model_config.get_all_power_coefficients('prefill')
decode_power_coeffs = model_config.get_all_power_coefficients('decode')

def prefill_power_model(model_name, input_length):
    """Return average prefill power (W) given input length."""
    coeffs = prefill_power_coeffs[model_name]
    model_type = coeffs.get('type', 'linear')

    if model_type == 'linear':
        power = coeffs['slope'] * input_length + coeffs['intercept']
        return max(0.0, power)
    
    elif model_type == 'piecewise_log':
        threshold = coeffs['threshold']
        if input_length <= threshold:
            power = coeffs['constant']
        else:
            power = coeffs['a'] * math.log(input_length) + coeffs['b']
        return max(0.0, power)
    
    else:
        raise ValueError(f"Unknown prefill power model type: {model_type}")

def _find_closest_token_length(target, supported_list):
    """Find the closest supported token length using bisect."""
    if target <= supported_list[0]:
        return supported_list[0]
    if target >= supported_list[-1]:
        return supported_list[-1]
    
    idx = bisect.bisect_left(supported_list, target)
    if idx == 0:
        return supported_list[0]
    if idx == len(supported_list):
        return supported_list[-1]
    
    left_val = supported_list[idx - 1]
    right_val = supported_list[idx]
    if abs(target - left_val) <= abs(target - right_val):
        return left_val
    else:
        return right_val

def decode_power_model(model_name, input_length, output_length):
    """
    Calculate decode power using hybrid approach:
    - Logarithmic model for output_tokens 1-64  
    - Exponential power model for output_tokens >= 65
    
    Returns average power for decode phase.
    """
    if output_length <= 0:
        return 0.0

    # Use logarithmic model for short sequences  
    exponential_threshold = model_config.get_exponential_threshold()
    if output_length < exponential_threshold:
        coeffs = decode_power_coeffs[model_name]
        power = coeffs['a'] * math.log(output_length) + coeffs['b']
        return max(0.0, power)
    
    # For longer sequences: Use exponential model if available
    config = _find_closest_decode_config(model_name, input_length, output_length)
    if config is not None:
        # Use exponential model parameters
        P_inf = config['P_inf']
        delta_P = config['delta_P']  
        tau = config['tau']
        
        # Calculate average power during decode using exponential model
        # For P(t) = P_∞ - ΔP × exp(-t/τ), average over [0,T] is:
        # P_avg = P_∞ - ΔP × τ/T × (1 - exp(-T/τ))
        
        import latency_model
        decode_time = latency_model.decode_latency_model(model_name, input_length, output_length)
        
        if decode_time > 0 and tau > 0:
            exponential_term = 1 - math.exp(-decode_time / tau)
            avg_power = P_inf - delta_P * (tau / decode_time) * exponential_term
            return max(0.0, avg_power)
    
    # Fallback to logarithmic model
    coeffs = decode_power_coeffs[model_name]
    power = coeffs['a'] * math.log(output_length) + coeffs['b']
    return max(0.0, power)

def calculate_decode_energy(model_name, input_length, output_length):
    """
    Calculate decode energy using exponential power model when available.
    
    For output_tokens >= 65: Uses E = P_∞×T - ΔP×τ×(1 - exp(-T/τ))
    For output_tokens 1-64: Falls back to Power × Time
    
    Returns: decode_energy (J), used_exponential_model (bool)
    """
    if output_length <= 0:
        return 0.0, False
    
    # For small outputs or models without exponential data: use P×T
    if output_length <= 64:
        decode_time = latency_model.decode_latency_model(model_name, input_length, output_length) 
        decode_power = decode_power_model(model_name, input_length, output_length)
        return decode_power * decode_time, False
    
    # For larger outputs: use exponential model
    config = _find_closest_decode_config(model_name, input_length, output_length)
    if config is not None:
        import latency_model
        decode_time = latency_model.decode_latency_model(model_name, input_length, output_length)
        
        P_inf = config['P_inf']
        delta_P = config['delta_P']
        tau = config['tau']
        
        energy = _decode_energy_function(P_inf, delta_P, tau, decode_time)
        return energy, True
    
    # Fallback to P×T approach
    import latency_model
    decode_time = latency_model.decode_latency_model(model_name, input_length, output_length)
    decode_power = decode_power_model(model_name, input_length, output_length)  
    return decode_power * decode_time, False

def total_power_model(model_name, input_length, output_length):
    """
    Power consumption for prefill and decode phases
    
    Returns:
        tuple: (prefill_power, decode_power) in watts
        
    """
    prefill_power = prefill_power_model(model_name, input_length)
    decode_power = decode_power_model(model_name, input_length, output_length)
    return prefill_power, decode_power

if __name__ == "__main__":
    input_length = 500
    output_length = 128

    print(f"{'Model':<20} {'Prefill Power':<15} {'Decode Power':<15}")
    print("-" * 50)
    
    for model_name in prefill_power_coeffs.keys():
        prefill_pow, decode_pow = total_power_model(model_name, input_length, output_length)
        print(f"{model_name:<20} {prefill_pow:<15.2f} {decode_pow:<15.2f}") 