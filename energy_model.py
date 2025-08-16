import latency_model
import power_model
import math
import argparse

#Original decode fitting (validation)
decode_energy_coeffs = {
    'DSR1-Llama-8B': {'a': 0.63, 'b': -2.51},
    'DSR1-Qwen-1.5B': {'a': 0.05, 'b': -0.03},
    'DSR1-Qwen-14B': {'a': 1.18, 'b': -4.69}
}

# Original prefill fitting (validation)
prefill_energy_coeffs = {
    'DSR1-Llama-8B': {'linear': 4.131263088291703e-06, 'constant': 0.017569106343149676},
    'DSR1-Qwen-1.5B': {'quadratic': 2e-9, 'linear': -1e-5, 'constant': 0.0151},
    'DSR1-Qwen-14B': {'exp_coeff': 3.260064334788192, 'exp_rate': -0.012940691920715262, 'constant': 0.061665552717004045}
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
    """Return TOTAL prefill energy in joules using the *direct* curve.

    The coefficient tables now store energy *per token* directly in **joules**. Steps:
      1. Pad the input length exactly like `prefill_latency_model` (multiple of 128).
      2. Evaluate the per-token energy curve (already in J/token).
      3. Multiply by the padded token count to obtain total energy (J).
    """
    coeffs = prefill_energy_coeffs[model_name]

    # ➊ Align with latency model padding
    padded_len = ((input_length + 127) // 128) * 128

    # ➋ Energy per token (J) – formula depends on model
    if model_name == 'DSR1-Llama-8B':
        e_pt_j = coeffs['linear'] * padded_len + coeffs['constant']
    elif model_name == 'DSR1-Qwen-1.5B':
        x = padded_len
        e_pt_j = (
            coeffs['quadratic'] * x**2 +
            coeffs['linear'] * x +
            coeffs['constant']
        )
    elif model_name == 'DSR1-Qwen-14B':
        e_pt_j = coeffs['exp_coeff'] * math.exp(coeffs['exp_rate'] * padded_len) + coeffs['constant']
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    # ➌ Scale by tokens to get total energy (J)
    energy_total_j = max(0.0, e_pt_j * padded_len)
    return energy_total_j


def decode_energy_formula(model_name, output_length):
    """Return TOTAL decode energy in joules.

    The stored coefficients provide per-token energy in joules following a log curve. We
    multiply by the output-token count to obtain total energy.
    """
    coeffs = decode_energy_coeffs[model_name]

    # ➊ Per-token energy (J)
    e_pt_j = coeffs['a'] * math.log(max(output_length, 1)) + coeffs['b']

    # ➋ Total energy in joules
    energy_total_j = max(0.0, e_pt_j * output_length)
    return energy_total_j

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

def _cli():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Energy-model demo: calculates prefill / decode / total Joules for each model.")
    parser.add_argument("-i", "--input", type=int, default=116,
                        help="Input tokens (prefill length). Default: 116")
    parser.add_argument("-o", "--output", type=int, default=82,
                        help="Output tokens (decode length). Default: 82")
    parser.add_argument("-v", "--validate", action="store_true",
                        help="Also print direct-formula cross-validation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _cli()
    input_length = args.input
    output_length = args.output

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
        prefill_pct = (prefill_eng / total_eng) * 100 if total_eng else 0
        decode_pct = (decode_eng / total_eng) * 100 if total_eng else 0
        print(f"{model_name}: Prefill {prefill_pct:.1f}% | Decode {decode_pct:.1f}%")

    if args.validate:
        print("\n" + "="*65)
        print("🔍 CROSS-VALIDATION: Power×Time vs Direct Formulas")
        print("="*65)
        for model_name in power_model.prefill_power_coeffs.keys():
            total_energy_model(model_name, input_length, output_length, validate=True) 