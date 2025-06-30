"""
Power consumption model for LLM inference

"""

import numpy as np
import math

# Prefill power coefficients 
prefill_power_coeffs = {
    'DSR1-Llama-8B': {
        'threshold': 800,
        'constant': 3.25,
        'a': 11.93,
        'b': -73.37
    },
    'DSR1-Qwen-1.5B': {
        'threshold': 3000,
        'constant': 2.67,
        'a': 36.65,
        'b': -293.74
    },
    'DSR1-Qwen-14B': {
        'threshold': 800,
        'constant': 10.83,
        'a': 11.21,
        'b': -60.41
    }
}

# Decode power coefficients 
decode_power_coeffs = {
    'DSR1-Llama-8B': {'a': 1.44, 'b': 17.39},
    'DSR1-Qwen-1.5B': {'a': 2.69, 'b': 2.45},
    'DSR1-Qwen-14B': {'a': 0.92, 'b': 21.51}
}

def prefill_power_model(model_name, input_length):
    """
    
    Fitted power model:
    DSR1-Llama-8B:   P(x) = { 3.25,                    x ≤ 800
                            { 11.93*ln(x) - 73.37,      x > 800
    
    DSR1-Qwen-1.5B:  P(x) = { 2.67,                    x ≤ 3000  
                            { 36.65*ln(x) - 293.74,     x > 3000
    
    DSR1-Qwen-14B:   P(x) = { 10.83,                   x ≤ 800
                            { 11.21*ln(x) - 60.41,      x > 800
    """
    coeffs = prefill_power_coeffs[model_name]
    
    if input_length <= coeffs['threshold']:
        return coeffs['constant']
    else:
        return coeffs['a'] * math.log(input_length) + coeffs['b']

def decode_power_model(model_name, output_length):
    """
    Calculate decode power
    
    DSR1-Llama-8B:   P(tok) = 1.44*ln(tok) + 17.39
    DSR1-Qwen-1.5B:  P(tok) = 2.69*ln(tok) + 2.45  
    DSR1-Qwen-14B:   P(tok) = 0.92*ln(tok) + 21.51
    """
    coeffs = decode_power_coeffs[model_name]
    return coeffs['a'] * math.log(output_length) + coeffs['b']

def total_power_model(model_name, input_length, output_length):
    """
    power consumption for prefill and decode phases
    
    Returns:
        tuple: (prefill_power, decode_power) in watts
        
    """
    prefill_power = prefill_power_model(model_name, input_length)
    decode_power = decode_power_model(model_name, output_length)
    return prefill_power, decode_power

if __name__ == "__main__":
    input_length = 500
    output_length = 128

    print(f"{'Model':<20} {'Prefill Power':<15} {'Decode Power':<15}")
    print("-" * 50)
    
    for model_name in prefill_power_coeffs.keys():
        prefill_pow, decode_pow = total_power_model(model_name, input_length, output_length)
        print(f"{model_name:<20} {prefill_pow:<15.2f} {decode_pow:<15.2f}") 