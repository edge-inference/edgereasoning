"""
Power consumption model for LLM inference

"""

import numpy as np
import math

# Prefill power coefficients 
prefill_power_coeffs = {
    'DSR1-Llama-8B': {
        'threshold': 800,
        'constant': 3.2512499999999998,
        'a': 11.927798396907887,
        'b': -73.3710759267571
    },
    'DSR1-Qwen-1.5B': {
        'threshold': 3000,
        'constant': 2.6741428571428574,
        'a': 36.65125098685668,
        'b': -293.7413929014477
    },
    'DSR1-Qwen-14B': {
        'threshold': 800,
        'constant': 10.82625,
        'a': 11.205603899138284,
        'b': -60.40954041332665
    }
}

# Decode power coefficients 
decode_power_coeffs = {
    'DSR1-Llama-8B': {'a': 1.4370685303251562, 'b': 17.386399999445654},
    'DSR1-Qwen-1.5B': {'a': 2.6906630031738046, 'b': 2.4464267515923543},
    'DSR1-Qwen-14B': {'a': 0.9246232482387127, 'b': 21.506100020426842}
}

def prefill_power_model(model_name, input_length):
    """
    
    Fitted power model:
    DSR1-Llama-8B:   P(x) = { 3.2512499999999998,                    x ≤ 800
                            { 11.927798396907887*ln(x) - 73.3710759267571,      x > 800
    
    DSR1-Qwen-1.5B:  P(x) = { 2.6741428571428574,                    x ≤ 3000  
                            { 36.65125098685668*ln(x) - 293.7413929014477,     x > 3000
    
    DSR1-Qwen-14B:   P(x) = { 10.82625,                   x ≤ 800
                            { 11.205603899138284*ln(x) - 60.40954041332665,      x > 800
    
    """
    coeffs = prefill_power_coeffs[model_name]
    
    if input_length <= coeffs['threshold']:
        power = coeffs['constant']
    else:
        power = coeffs['a'] * math.log(input_length) + coeffs['b']
    
    return max(0.0, power)

def decode_power_model(model_name, output_length):
    """
    Calculate decode power
    
    DSR1-Llama-8B:   P(tok) = 1.4370685303251562*ln(tok) + 17.386399999445654
    DSR1-Qwen-1.5B:  P(tok) = 2.6906630031738046*ln(tok) + 2.4464267515923543  
    DSR1-Qwen-14B:   P(tok) = 0.9246232482387127*ln(tok) + 21.506100020426842
    """
    if output_length <= 0:
        return 0.0

    coeffs = decode_power_coeffs[model_name]
    power = coeffs['a'] * math.log(output_length) + coeffs['b']
    return max(0.0, power)

def total_power_model(model_name, input_length, output_length):
    """
    Power consumption for prefill and decode phases
    
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