import latency_model
import power_model
import math

#Original decode fitting (validation)
decode_energy_coeffs = {
    'DSR1-Llama-8B': {'a': 0.63, 'b': -2.51},
    'DSR1-Qwen-1.5B': {'a': 0.05, 'b': -0.03},
    'DSR1-Qwen-14B': {'a': 1.18, 'b': -4.69}
}

# Original prefill fitting (validation)
prefill_energy_coeffs = {
    'DSR1-Llama-8B': {'linear': 0.00, 'constant': 17.57},
    'DSR1-Qwen-1.5B': {'cubic': 0.00, 'quadratic': 0.00, 'linear': -0.01, 'constant': 15.50},
    'DSR1-Qwen-14B': {'exp_coeff': 3260.06, 'exp_rate': -0.01, 'constant': 61.67}
}

def calculate_energy_from_power_time(model_name, input_length, output_length):
    
    # Get power consumption (watts)
    prefill_power, decode_power = power_model.total_power_model(model_name, input_length, output_length)
    
    # Get latency (seconds)
    prefill_latency, decode_latency, total_latency = latency_model.total_latency_model(model_name, input_length, output_length)
    
    # Calculate energy (joules = watts × seconds)
    prefill_energy = prefill_power * prefill_latency
    decode_energy = decode_power * decode_latency
    total_energy = prefill_energy + decode_energy
    
    return prefill_energy, decode_energy, total_energy

def prefill_energy_formula(model_name, input_length):
    """Original energy per token formulas """
    coeffs = prefill_energy_coeffs[model_name]
    
    if model_name == 'DSR1-Llama-8B':
        return coeffs['linear'] * input_length + coeffs['constant']
    elif model_name == 'DSR1-Qwen-1.5B':
        x = input_length
        return coeffs['cubic'] * x**3 + coeffs['quadratic'] * x**2 + coeffs['linear'] * x + coeffs['constant']
    elif model_name == 'DSR1-Qwen-14B':
        return coeffs['exp_coeff'] * math.exp(coeffs['exp_rate'] * input_length) + coeffs['constant']

def decode_energy_formula(model_name, output_length):
    """Original energy per token formulas """
    coeffs = decode_energy_coeffs[model_name]
    return coeffs['a'] * math.log(output_length) + coeffs['b']

def total_energy_model(model_name, input_length, output_length, validate=False):
    # Primary method: Power × Time
    prefill_energy_pt, decode_energy_pt, total_energy_pt = calculate_energy_from_power_time(model_name, input_length, output_length)
    
    if validate:
        # Cross-validation
        prefill_energy_direct = prefill_energy_formula(model_name, input_length)
        decode_energy_direct = decode_energy_formula(model_name, output_length)
        total_energy_direct = prefill_energy_direct + decode_energy_direct
        
        # diff
        prefill_diff = abs(prefill_energy_pt - prefill_energy_direct)
        decode_diff = abs(decode_energy_pt - decode_energy_direct)
        total_diff = abs(total_energy_pt - total_energy_direct)
        
        print(f"\n🔍 CROSS-VALIDATION for {model_name}:")
        print(f"   Prefill Energy: P×T={prefill_energy_pt:.2f}J | Direct={prefill_energy_direct:.2f}J | Diff={prefill_diff:.2f}J")
        print(f"   Decode Energy:  P×T={decode_energy_pt:.2f}J | Direct={decode_energy_direct:.2f}J | Diff={decode_diff:.2f}J")
        print(f"   Total Energy:   P×T={total_energy_pt:.2f}J | Direct={total_energy_direct:.2f}J | Diff={total_diff:.2f}J")
        
        return {
            'power_time': (prefill_energy_pt, decode_energy_pt, total_energy_pt),
            'direct': (prefill_energy_direct, decode_energy_direct, total_energy_direct),
            'differences': (prefill_diff, decode_diff, total_diff)
        }
    
    return prefill_energy_pt, decode_energy_pt, total_energy_pt

if __name__ == "__main__":
    input_length = 116
    output_length = 82

    print("="*65)
    print("🔋 ENERGY MODEL RESULTS (Power × Time Method)")
    print("="*65)
    print(f"{'Model':<20} {'Prefill Energy':<15} {'Decode Energy':<15} {'Total Energy':<15}")
    print("-" * 65)
    
    for model_name in power_model.prefill_power_coeffs.keys():
        prefill_eng, decode_eng, total_eng = total_energy_model(model_name, input_length, output_length)
        print(f"{model_name:<20} {prefill_eng:<15.2f} {decode_eng:<15.2f} {total_eng:<15.2f}")
    
    print("\n" + "="*65)
    print("⚡ ENERGY BREAKDOWN (Joules)")
    print("="*65)
    
    for model_name in power_model.prefill_power_coeffs.keys():
        prefill_eng, decode_eng, total_eng = total_energy_model(model_name, input_length, output_length)
        prefill_pct = (prefill_eng / total_eng) * 100
        decode_pct = (decode_eng / total_eng) * 100
        print(f"{model_name}: Prefill {prefill_pct:.1f}% | Decode {decode_pct:.1f}%")
    
    print("\n" + "="*65)
    print("🔍 CROSS-VALIDATION: Power×Time vs Direct Formulas")
    print("="*65)
    
    for model_name in power_model.prefill_power_coeffs.keys():
        total_energy_model(model_name, input_length, output_length, validate=True) 