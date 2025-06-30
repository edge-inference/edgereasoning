import numpy as np

prefill_coeffs = {
    'DSR1-Qwen-1.5B': {'a': 1.5635548753756998e-07, 'b': 2.309554283403154e-06, 'c': 0.04598102600399947},
    'DSR1-Llama-8B': {'a': 6.645207285297158e-07, 'b': 0.0002903545130874798, 'c': 0.10380849913420899},
    'DSR1-Qwen-14B': {'a': 1.233003335291953e-06, 'b': 0.0005304540126753786, 'c': 0.18858145785250313}
}

decode_coeffs = {'DSR1-Qwen-1.5B': {'m': -1.6973069262930655e-06, 'n': 0.028776851933768718}, 
                 'DSR1-Llama-8B': {'m': 2.838752634710301e-07, 'n': 0.09292941860587282}, 
                'DSR1-Qwen-14B': {'m': 6.460185375534166e-08, 'n': 0.18595344002532865}}

decode_coeffs = {'DSR1-Qwen-14B': {'m': 6.460185375534166e-08, 'n': 0.18595344002532865}, 
                 'DSR1-Qwen-1.5B': {'m': 1.496488595602221e-07, 'n': 0.023635947648587594}, 
                 'DSR1-Llama-8B': {'m': 6.570033090550273e-07, 'n': 0.10051350928523411}}


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
    input_length = 116
    output_length = 82

    print(f"{'Model':<20} {'Prefill':<10} {'Decode':<10} {'Total':<10}")
    print("-" * 50)
    
    for model_name in prefill_coeffs.keys():
        prefill_lat, decode_lat, total_lat = total_latency_model(model_name, input_length, output_length)
        print(f"{model_name:<20} {prefill_lat:<10.2f} {decode_lat:<10.2f} {total_lat:<10.2f}")
