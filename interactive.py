import latency_model

def format_time(seconds):
    if seconds < 0.001:
        return f"{seconds*1000000:.1f}μs"
    elif seconds < 1:
        return f"{seconds*1000:.1f}ms"
    else:
        return f"{seconds:.3f}s"

def print_model_comparison(input_length, output_length):
    results = {}
    for model_name in latency_model.prefill_coeffs.keys():
        prefill_lat, decode_lat, total_lat = latency_model.total_latency_model(model_name, input_length, output_length)
        results[model_name] = {
            'prefill': prefill_lat,
            'decode': decode_lat, 
            'total': total_lat
        }
    
    print("=" * 90)
    print(f"Latency Model Comparison")
    print(f"Input: {input_length:,} tokens | Output: {output_length:,} tokens")
    print("=" * 90)
    
    print(f"{'Model':<20} {'Prefill':<12} {'Decode':<12} {'Total':<12} {'Ratio':<8} {'Speed':<10}")
    print("-" * 90)
    
    sorted_models = sorted(results.items(), key=lambda x: x[1]['total'])
    fastest_total = sorted_models[0][1]['total']
    
    for model_name, latencies in sorted_models:
        prefill_str = format_time(latencies['prefill'])
        decode_str = format_time(latencies['decode'])
        total_str = format_time(latencies['total'])
        
        ratio = latencies['decode'] / latencies['prefill'] if latencies['prefill'] > 0 else 0
        speed_multiplier = latencies['total'] / fastest_total
        speed_str = f"{speed_multiplier:.1f}x" if speed_multiplier > 1 else "baseline"
        
        print(f"{model_name:<20} {prefill_str:<12} {decode_str:<12} {total_str:<12} {ratio:<8.1f} {speed_str:<10}")
    
    print("-" * 90)
    
    fastest_model = sorted_models[0][0]
    slowest_model = sorted_models[-1][0]
    
    print("\n Insights:")
    print(f"   Fastest: {fastest_model}")
    print(f"   Slowest: {slowest_model}")
    
    total_tokens = input_length + output_length
    print(f"\n📈 TOKEN PROCESSING RATES:")
    for model_name, latencies in sorted_models:
        tokens_per_sec = total_tokens / latencies['total']
        print(f"   {model_name}: {tokens_per_sec:,.0f} tokens/sec")
    
    print(f"\n⚡ LATENCY BREAKDOWN:")
    for model_name, latencies in results.items():
        prefill_pct = (latencies['prefill'] / latencies['total']) * 100
        decode_pct = (latencies['decode'] / latencies['total']) * 100
        print(f"   {model_name}: Prefill {prefill_pct:.1f}% | Decode {decode_pct:.1f}%")

def test_multiple_scenarios():
    scenarios = [
        {"name": "Short Query", "input": 100, "output": 50},
        {"name": "Medium Query", "input": 500, "output": 200}, 
        {"name": "Long Query", "input": 2000, "output": 500},
        {"name": "Very Long Query", "input": 8000, "output": 1000},
    ]
    
    print("\n" + "=" * 90)
    print("Sweep")
    print("=" * 90)
    
    for scenario in scenarios:
        print(f"\n📋 {scenario['name']} ({scenario['input']} → {scenario['output']} tokens)")
        print_model_comparison(scenario['input'], scenario['output'])
        print()

def run_interactive_mode():
    input_length = 116
    output_length = 82
    
    print_model_comparison(input_length, output_length)
    
    print("\n" + "="*50)
    response = input("Sweep? (y/n): ").lower().strip()
    if response in ['y', 'yes']:
        test_multiple_scenarios()
    
    print("\n" + "="*50) 
    print("INTERACTIVE MODE")
    print("="*50)
    
    while True:
        try:
            response = input("\n input,output tokens (e.g., '1000,300') or 'q' to quit: ").strip()
            if response.lower() in ['q', 'quit', 'exit']:
                break
                
            if ',' in response:
                input_tokens, output_tokens = map(int, response.split(','))
                print_model_comparison(input_tokens, output_tokens)
            else:
                print("❌ input,output (e.g., '1000,300')")
                
        except ValueError:
            print("❌ input,output (e.g., '1000,300')")
        except KeyboardInterrupt:
            break
    

if __name__ == "__main__":
    run_interactive_mode() 