import latency_model
import power_model
import math
import json, pathlib, bisect
import argparse
from loaders.analytic import get_config

# Load configuration
energy_config = get_config()

# --------------------------------------------------
# Prefill energy lookup table (J / token) for validation
# --------------------------------------------------
_LOOKUP_PATH = energy_config.get_prefill_validation_path()
if _LOOKUP_PATH.exists():
    with _LOOKUP_PATH.open() as _f:
        model_block = json.load(_f)["data"]  

    _LOOKUP = {}
    for mk, data in model_block.items():
        ept = data["energy_per_token"]          
        tok_list = data["input_tokens"]        
        mapping = {int(t): float(v) for t, v in zip(tok_list, ept)}
        keys = sorted(mapping)
        vals = [mapping[k] for k in keys]
        _LOOKUP[mk] = (keys, vals)

    _MODEL_KEY_MAP = {
        'DSR1-Llama-8B':  'DeepSeek_R1_Distill_Llama_8B',
        'DSR1-Qwen-1.5B': 'DeepSeek_R1_Distill_Qwen_1.5B',
        'DSR1-Qwen-14B':  'DeepSeek_R1_Distill_Qwen_14B'
    }

    def _lookup_prefill_energy(model_name: str, input_len: int) -> float:
        """Total prefill energy (J) via nearest-neighbour interpolation from JSON lookup."""
        model_key = _MODEL_KEY_MAP.get(model_name, model_name)
        if model_key not in _LOOKUP:
            return None

        padded = ((input_len + 127) // 128) * 128
        tokens, values = _LOOKUP[model_key]
        idx = bisect.bisect_left(tokens, padded)
        if idx == 0:
            e_pt = values[0]
        elif idx == len(tokens):
            e_pt = values[-1]
        else:
            lo_t, hi_t = tokens[idx-1], tokens[idx]
            lo_v, hi_v = values[idx-1], values[idx]
            t = (padded - lo_t) / (hi_t - lo_t)
            e_pt = lo_v * (1 - t) + hi_v * t
        return e_pt * padded
else:
    _lookup_prefill_energy = lambda *args, **kwargs: None  # fallback

# --------------------------------------------------
# Decode energy lookup table (J) for validation
# --------------------------------------------------
_DECODE_LOOKUP_PATH = energy_config.get_decode_validation_path()
if _DECODE_LOOKUP_PATH.exists():
    with _DECODE_LOOKUP_PATH.open() as _f:
        decode_data = json.load(_f)

    _DECODE_LOOKUP = {}
    for model_key, data in decode_data.items():
        input_tokens = data["input_tokens"]
        output_tokens = data["output_tokens"] 
        total_energy = data["total_energy_j"]
        
        # Organize into 2D lookup: input_len -> output_len -> energy
        lookup_dict = {}
        for i, (inp, out, energy) in enumerate(zip(input_tokens, output_tokens, total_energy)):
            if inp not in lookup_dict:
                lookup_dict[inp] = {}
            lookup_dict[inp][out] = energy
        
        _DECODE_LOOKUP[model_key] = lookup_dict

    def _lookup_decode_energy(model_name: str, input_len: int, output_len: int) -> float:
        """Decode-only energy (J) via 2D lookup from JSON validation data.
        
        The JSON contains total_energy_j (prefill + decode), so we subtract 
        the prefill component to get pure decode energy for validation.
        """
        model_key = _MODEL_KEY_MAP.get(model_name, model_name)
        if model_key not in _DECODE_LOOKUP:
            return None
            
        lookup_table = _DECODE_LOOKUP[model_key]
        
        # Find closest input length
        available_inputs = sorted(lookup_table.keys())
        closest_input = min(available_inputs, key=lambda x: abs(x - input_len))
        
        # Find closest output length for that input
        available_outputs = sorted(lookup_table[closest_input].keys())
        closest_output = min(available_outputs, key=lambda x: abs(x - output_len))
        
        total_energy = lookup_table[closest_input][closest_output]
        
        # Subtract prefill energy to get decode-only energy
        prefill_energy = _lookup_prefill_energy(model_name, closest_input)
        if prefill_energy is None:
            # Fallback to prefill formula if lookup not available
            prefill_energy = prefill_energy_formula(model_name, closest_input)
        
        decode_only_energy = total_energy - prefill_energy
        return max(0.0, decode_only_energy)  # Ensure non-negative
else:
    _lookup_decode_energy = lambda *args, **kwargs: None  # fallback

def _lookup_parameters(model_name: str, input_length: int, output_length: int) -> float:
    """Get measured decode_energy from decode_power_parameters.json for validation."""
    return power_model._lookup_parameters(model_name, input_length, output_length)

# Updated decode fitting based on new logarithmic parameters
decode_energy_coeffs = {
    'DSR1-Llama-8B': {'a': 0.63, 'b': -2.51},  # Keep existing values - not provided in new data
    'DSR1-Qwen-1.5B': {'a': 0.08295945857124509, 'b': -0.18147099698102612},
    'DSR1-Qwen-14B': {'a': 0.7046762343090117, 'b': -1.5301621897397977}
}

# Original prefill fitting (validation)
prefill_energy_coeffs = {
    'DSR1-Llama-8B': {
        'exp_coeff': 0.2037618596417348,
        'exp_rate': -0.14845175210587422,
        'constant': 0.01900406658131384
    },
    'DSR1-Qwen-1.5B': {
        'exp_coeff': 0.07308412448027113,
        'exp_rate': -0.03194527903289569,
        'constant': 0.0009231594757018509
    },
    'DSR1-Qwen-14B': {
        'exp_coeff': 0.37083300132252434,
        'exp_rate': -0.15331526827621197,
        'constant': 0.042684758270411835
    }
}

def calculate_energy_from_power_time(model_name, input_length, output_length, use_exponential_decode=True):
    
    # Get power consumption (watts) and latency (seconds)
    prefill_power, decode_power = power_model.total_power_model(model_name, input_length, output_length)
    prefill_latency, decode_latency, total_latency = latency_model.total_latency_model(model_name, input_length, output_length)
    
    # Calculate prefill energy (always P×T)
    prefill_energy = prefill_power * prefill_latency
    
    # Calculate decode energy (exponential model when available/requested)
    if use_exponential_decode:
        decode_energy, used_exponential = power_model.calculate_decode_energy(model_name, input_length, output_length)
        if not used_exponential:
            # Fallback to standard P×T
            decode_energy = decode_power * decode_latency
    else:
        # Standard P×T approach
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
    e_pt_j = coeffs['exp_coeff'] * math.exp(coeffs['exp_rate'] * padded_len) + coeffs['constant']

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
    # Primary method: Integrated Energy (∫P(t)dt)
    prefill_energy_pt, decode_energy_pt, total_energy_pt = calculate_energy_from_power_time(model_name, input_length, output_length)
    
    if validate:
        # Cross-validation
        prefill_lookup = _lookup_prefill_energy(model_name, input_length)
        if prefill_lookup is not None:
            prefill_energy_direct = prefill_lookup
        else:
            prefill_energy_direct = prefill_energy_formula(model_name, input_length)
        
        # For decode validation, use different sources based on output length
        if output_length >= 65:
            # Use measured decode_energy from decode_power_parameters.json for >=65 tokens
            decode_energy_empirical = _lookup_parameters(model_name, input_length, output_length)
            if decode_energy_empirical is None:
                # Fallback to decode_power.json lookup
                decode_energy_empirical = _lookup_decode_energy(model_name, input_length, output_length)
                if decode_energy_empirical is None:
                    # Final fallback to formula
                    decode_energy_empirical = decode_energy_formula(model_name, output_length)
        else:
            # For <=64 tokens, use legacy method (decode_power.json or formula)
            decode_lookup = _lookup_decode_energy(model_name, input_length, output_length)
            if decode_lookup is not None:
                decode_energy_empirical = decode_lookup
            else:
                decode_energy_empirical = decode_energy_formula(model_name, output_length)
        
        total_energy_empirical = prefill_energy_direct + decode_energy_empirical
        
        # diff
        prefill_diff = abs(prefill_energy_pt - prefill_energy_direct)
        decode_diff = abs(decode_energy_pt - decode_energy_empirical)
        total_diff = abs(total_energy_pt - total_energy_empirical)
        
        print(f"\n🔍 CROSS-VALIDATION for {model_name}:")
        print(f"   Prefill Energy: ∫P(t)dt={prefill_energy_pt:.2f}J | Empirical={prefill_energy_direct:.2f}J | Diff={prefill_diff:.2f}J")
        print(f"   Decode Energy:  ∫P(t)dt={decode_energy_pt:.2f}J | Empirical={decode_energy_empirical:.2f}J | Diff={decode_diff:.2f}J")
        print(f"   Total Energy:   ∫P(t)dt={total_energy_pt:.2f}J | Empirical={total_energy_empirical:.2f}J | Diff={total_diff:.2f}J")
        
        return {
            'power_time': (prefill_energy_pt, decode_energy_pt, total_energy_pt),
            'empirical': (prefill_energy_direct, decode_energy_empirical, total_energy_empirical),
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
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-P", "--prefill", action="store_true", help="Show only prefill metrics")
    group.add_argument("-D", "--decode", action="store_true", help="Show only decode metrics")
    parser.add_argument("-v", "--validate", action="store_true",
                        help="Also print direct-formula cross-validation.")
    return parser.parse_args()


if __name__ == "__main__":
    args = _cli()
    input_length = args.input
    output_length = args.output

    def _fmt_time(sec: float) -> str:
        if sec < 1e-3:
            return f"{sec*1e6:.0f}µs"
        if sec < 1:
            return f"{sec*1e3:.1f}ms"
        return f"{sec:.2f}s"

    show_prefill = not args.decode
    show_decode  = not args.prefill

    header_parts = [f"{'Model':<20}"]
    if show_prefill:
        header_parts += [f"{'P-Lat':<8}", f"{'P-Pwr':<8}", f"{'P-E':<10}"]
    if show_decode:
        header_parts += [f"{'D-Lat':<8}", f"{'D-Pwr':<8}", f"{'D-E':<10}"]
    header_parts += [f"{'Total':<10}"]
    header = " ".join(header_parts)
    print("=" * len(header))
    print("🔋 ENERGY / LATENCY / POWER (I={} , O={})".format(input_length, output_length))
    print(header)
    print("-" * len(header))

    # Sort models by size (1.5B → 8B → 14B)
    def _extract_model_size(model_name):
        if '1.5B' in model_name:
            return 1.5
        elif '8B' in model_name:
            return 8.0
        elif '14B' in model_name:
            return 14.0
        else:
            return 999.0  # Unknown models go last
    
    sorted_models = sorted(power_model.prefill_power_coeffs.keys(), key=_extract_model_size)
    for model_name in sorted_models:
        # Latency and power
        pre_lat, dec_lat, _ = latency_model.total_latency_model(model_name, input_length, output_length)
        pre_pwr, dec_pwr = power_model.total_power_model(model_name, input_length, output_length)

        # Energy (Integrated: ∫P(t)dt)
        pre_eng, dec_eng, total_eng = calculate_energy_from_power_time(model_name, input_length, output_length)

        row_parts = [f"{model_name:<20}"]
        if show_prefill:
            row_parts += [f"{_fmt_time(pre_lat):<8}", f"{pre_pwr:<8.2f}", f"{pre_eng:<10.2f}"]
        if show_decode:
            row_parts += [f"{_fmt_time(dec_lat):<8}", f"{dec_pwr:<8.2f}", f"{dec_eng:<10.2f}"]
        row_parts += [f"{total_eng:<10.2f}"]
        print(" ".join(row_parts))

    print("=" * len(header))

    # Breakdown percentages
    if show_prefill and show_decode:
        print("\n⚡ ENERGY BREAKDOWN (Prefill vs Decode)")
        for model_name in sorted_models:
            pre_eng, dec_eng, total_eng = calculate_energy_from_power_time(model_name, input_length, output_length)
            pre_pct = (pre_eng / total_eng) * 100 if total_eng else 0
            dec_pct = 100 - pre_pct
            print(f"{model_name}: Prefill {pre_pct:.1f}% | Decode {dec_pct:.1f}%")

    if args.validate:
        print("\n" + "="*65)
        print("🔍 CROSS-VALIDATION: Integrated Energy vs Empirical Data")
        print("="*65)

        for model_name in sorted_models:
            # Integrated energy values
            pre_eng_integrated, dec_eng_integrated, total_eng_integrated = calculate_energy_from_power_time(model_name, input_length, output_length)

            # Lookup direct values
            direct_prefill = _lookup_prefill_energy(model_name, input_length)
            if direct_prefill is None:
                direct_prefill = prefill_energy_formula(model_name, input_length)

            if output_length == 0:
                empirical_decode = 0.0
            elif output_length >= 65:
                empirical_decode = _lookup_parameters(model_name, input_length, output_length)
                if empirical_decode is None:
                    empirical_decode = _lookup_decode_energy(model_name, input_length, output_length)
                    if empirical_decode is None:
                        empirical_decode = decode_energy_formula(model_name, output_length)
            else:
                # Use avg_decode_energy from JSON for output lengths < 65
                config = power_model._find_closest_decode_config(model_name, input_length, output_length) if hasattr(power_model, '_find_closest_decode_config') else None
                if config:
                    avg_decode_energy = config.get('avg_decode_energy')
                    if avg_decode_energy is not None:
                        # Subtract empirical prefill energy to get pure decode energy
                        empirical_decode = max(0.0, avg_decode_energy - direct_prefill)
                    else:
                        # Fallback to legacy lookup method
                        empirical_decode = _lookup_decode_energy(model_name, input_length, output_length)
                        if empirical_decode is None:
                            empirical_decode = decode_energy_formula(model_name, output_length)
                else:
                    # No config found, use legacy lookup
                    empirical_decode = _lookup_decode_energy(model_name, input_length, output_length)
                    if empirical_decode is None:
                        empirical_decode = decode_energy_formula(model_name, output_length)
            
            total_empirical = direct_prefill + empirical_decode

            print(f"\n🔍 {model_name}:")
            if show_prefill:
                diff_pre = pre_eng_integrated - direct_prefill
                print(f"   Prefill: ∫P(t)dt={pre_eng_integrated:.2f}J | Empirical={direct_prefill:.2f}J | Δ={diff_pre:.2f}J")
            if show_decode:
                diff_dec = dec_eng_integrated - empirical_decode
                print(f"   Decode : ∫P(t)dt={dec_eng_integrated:.2f}J | Empirical={empirical_decode:.2f}J | Δ={diff_dec:.2f}J")
            diff_tot = total_eng_integrated - total_empirical
            print(f"   Total  : ∫P(t)dt={total_eng_integrated:.2f}J | Empirical={total_empirical:.2f}J | Δ={diff_tot:.2f}J") 