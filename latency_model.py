import numpy as np
import argparse

prefill_coeffs = {
    'DSR1-Qwen-1.5B': {'a': 1.5635548753756998e-07, 'b': 2.309554283403154e-06, 'c': 0.04598102600399947},
    'DSR1-Llama-8B': {'a': 6.645207285297158e-07, 'b': 0.0002903545130874798, 'c': 0.10380849913420899},
    'DSR1-Qwen-14B': {'a': 1.233003335291953e-06, 'b': 0.0005304540126753786, 'c': 0.18858145785250313}
}

decode_coeffs = {'DSR1-Qwen-1.5B': {'m': 1.4968086938621786e-07, 'n': 0.023633865754453625}, 
                 'DSR1-Llama-8B': {'m': 6.920219504432032e-07, 'n': 0.10038429572372025}, 
                 'DSR1-Qwen-14B': {'m': 1.1301658358203775e-06, 'n': 0.18655369142562853}}

def decode_func(m, n, I, O):
    """
    Fit function: latency = O * (m*I + n + m*(O-1)/2)
    """
    return O * (m*I + n + m*(O-1)/2)

def prefill_latency_model(model_name, input_length):
    coeffs = prefill_coeffs[model_name]
    # Pad input length to multiple of 128
    padded_input_length = ((input_length + 127) // 128) * 128
    prefill_func = np.poly1d([coeffs['a'], coeffs['b'], coeffs['c']])
    prefill_latency = prefill_func(padded_input_length)
    return prefill_latency


def decode_latency_model(model_name, input_length, output_length):
    coeffs = decode_coeffs[model_name]
    decode_latency = decode_func(coeffs['m'], coeffs['n'], input_length, output_length)
    return decode_latency


def total_latency_model(model_name, input_length, output_length):
    prefill_latency = prefill_latency_model(model_name, input_length)
    decode_latency = decode_latency_model(model_name, input_length, output_length)
    total_latency = prefill_latency + decode_latency
    return prefill_latency, decode_latency, total_latency


if __name__ == "__main__":
    def _cli():
        parser = argparse.ArgumentParser(description="Latency model – shows prefill, decode, total latency for each model.")
        parser.add_argument("-i", "--input", type=int, default=512, help="Input tokens (prefill length). Default: 512")
        parser.add_argument("-o", "--output", type=int, default=1, help="Output tokens (decode length). Default: 1")
        return parser.parse_args()

    def _format_time(seconds: float) -> str:
        if seconds < 1e-3:
            return f"{seconds*1e6:.0f}µs"
        if seconds < 1.0:
            return f"{seconds*1e3:.1f}ms"
        return f"{seconds:.3f}s"

    args = _cli()
    I = args.input
    O = args.output

    header = f"{'Model':<20} {'Prefill':<12} {'Decode':<12} {'Total':<12}"
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for model_name in prefill_coeffs.keys():
        prefill_lat, decode_lat, total_lat = total_latency_model(model_name, I, O)
        print(f"{model_name:<20} {_format_time(prefill_lat):<12} {_format_time(decode_lat):<12} {_format_time(total_lat):<12}")

    print("=" * len(header))
