#!/usr/bin/env python3
"""
Sweep I/O and and plot the prediceted results

"""

import numpy as np
import matplotlib.pyplot as plt
import latency_model
import power_model
import energy_model

# Plot configuration
plt.style.use('default')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['lines.linewidth'] = 2

def generate_sweep_data():
    """swep 128 to 4096"""
    
    prefill_tokens = np.arange(128, 4097, 256)
    decode_tokens = 200
    
    models = list(latency_model.prefill_coeffs.keys())
    
    data = {}
    
    for model in models:
        data[model] = {
            'prefill_tokens': prefill_tokens,
            'prefill_latency': [],
            'decode_latency': [],
            'total_latency': [],
            'prefill_power': [],
            'decode_power': [],
            'prefill_energy': [],
            'decode_energy': [],
            'total_energy': []
        }
        
        for prefill in prefill_tokens:
            prefill_lat, decode_lat, total_lat = latency_model.total_latency_model(model, prefill, decode_tokens)
            data[model]['prefill_latency'].append(prefill_lat)
            data[model]['decode_latency'].append(decode_lat)
            data[model]['total_latency'].append(total_lat)
            
            prefill_pow, decode_pow = power_model.total_power_model(model, prefill, decode_tokens)
            data[model]['prefill_power'].append(prefill_pow)
            data[model]['decode_power'].append(decode_pow)
            
            prefill_eng, decode_eng, total_eng = energy_model.total_energy_model(model, prefill, decode_tokens)
            data[model]['prefill_energy'].append(prefill_eng)
            data[model]['decode_energy'].append(decode_eng)
            data[model]['total_energy'].append(total_eng)
    
    return data

def plot_prefill_analysis(data):
    """prefill plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    models = list(data.keys())
    
    ax1 = axes[0, 0]
    for i, model in enumerate(models):
        ax1.plot(data[model]['prefill_tokens'], data[model]['prefill_latency'], 
                color=colors[i], marker='o', markersize=4, label=model)
    ax1.set_xlabel('Input Tokens')
    ax1.set_ylabel('Prefill Latency (seconds)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(100, 4200)
    
    ax2 = axes[0, 1]
    for i, model in enumerate(models):
        ax2.plot(data[model]['prefill_tokens'], data[model]['prefill_power'], 
                color=colors[i], marker='s', markersize=4, label=model)
    ax2.set_xlabel('Input Tokens')
    ax2.set_ylabel('Prefill Power (watts)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    ax2.set_xlim(100, 4200)
    
    ax3 = axes[1, 0]
    for i, model in enumerate(models):
        ax3.plot(data[model]['prefill_tokens'], data[model]['prefill_energy'], 
                color=colors[i], marker='^', markersize=4, label=model)
    ax3.set_xlabel('Input Tokens')
    ax3.set_ylabel('Prefill Energy (joules)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(100, 4200)
    
    ax4 = axes[1, 1]
    for i, model in enumerate(models):
        energy_per_token = np.array(data[model]['prefill_energy']) / np.array(data[model]['prefill_tokens'])
        ax4.plot(data[model]['prefill_tokens'], energy_per_token, 
                color=colors[i], marker='d', markersize=4, label=model)
    ax4.set_xlabel('Input Tokens')
    ax4.set_ylabel('Energy per Token (J/token)')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(100, 4200)
    
    plt.tight_layout()
    plt.savefig('outputs/prefill_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("📊 Prefill analysis saved as 'outputs/prefill_analysis.pdf'")

def plot_decode_analysis(data):
    """decode plots"""
    
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    models = list(data.keys())
    
    ax1 = axes[0, 0]
    for i, model in enumerate(models):
        ax1.plot(data[model]['prefill_tokens'], data[model]['total_latency'], 
                color=colors[i], marker='o', markersize=4, label=model)
    ax1.set_xlabel('Input Tokens')
    ax1.set_ylabel('Total Latency (seconds)')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(100, 4200)
    
    ax2 = axes[0, 1]
    decode_powers = {}
    for i, model in enumerate(models):
        decode_power = data[model]['decode_power'][0]
        decode_powers[model] = decode_power
        ax2.bar(model, decode_power, color=colors[i], alpha=0.7)
    ax2.set_ylabel('Decode Power (watts)')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    ax3 = axes[1, 0]
    for i, model in enumerate(models):
        ax3.plot(data[model]['prefill_tokens'], data[model]['total_energy'], 
                color=colors[i], marker='^', markersize=4, label=model)
    ax3.set_xlabel('Input Tokens')
    ax3.set_ylabel('Total Energy (joules)')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    ax3.set_xlim(100, 4200)
    
    ax4 = axes[1, 1]
    for i, model in enumerate(models):
        energy_ratio = np.array(data[model]['decode_energy']) / np.array(data[model]['prefill_energy'])
        ax4.plot(data[model]['prefill_tokens'], energy_ratio, 
                color=colors[i], marker='d', markersize=4, label=model)
    ax4.set_xlabel('Input Tokens')
    ax4.set_ylabel('Decode/Prefill Energy Ratio')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    ax4.set_xlim(100, 4200)
    
    plt.tight_layout()
    plt.savefig('outputs/decode_analysis.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    print("[*] Decode plots saved to 'outputs/decode_analysis.pdf'")


def print_summary_stats(data):

    print("\n" + "="*80)
    print("[*] Stats (Input: 128-4096 tokens, Output: 200 tokens)")
    print("="*80)
    
    models = list(data.keys())
    
    for model in models:
        print(f"\n- {model}:")
        
        # Latency stats
        min_latency = min(data[model]['total_latency'])
        max_latency = max(data[model]['total_latency'])
        print(f"   Latency Range: {min_latency:.2f}s - {max_latency:.2f}s")
        
        # Power stats
        min_power = min(data[model]['prefill_power'])
        max_power = max(data[model]['prefill_power'])
        decode_power = data[model]['decode_power'][0]
        print(f"   Power Range: {min_power:.2f}W - {max_power:.2f}W (prefill), {decode_power:.2f}W (decode)")
        
        # Energy stats
        min_energy = min(data[model]['total_energy'])
        max_energy = max(data[model]['total_energy'])
        print(f"   Energy Range: {min_energy:.2f}J - {max_energy:.2f}J")
        
        total_tokens = np.array(data[model]['prefill_tokens'])
        throughput = total_tokens / np.array(data[model]['prefill_latency'])
        min_throughput = np.min(throughput)
        max_throughput = np.max(throughput)
        print(f"   Input Speed Range: {min_throughput:.1f} - {max_throughput:.1f} tokens/s")

if __name__ == "__main__":
    import pathlib
    output_dir = pathlib.Path("outputs")
    if not output_dir.exists():
        output_dir.mkdir(parents=True)
        
    print("Generating data for prefill tokens: 128 to 4096 (step=128)")
    print("Fixed decode tokens: 200")
    
    data = generate_sweep_data()
    print("[v] Data generation complete")
    
    print("\n[*] Generating plots...")
    plot_prefill_analysis(data)
    plot_decode_analysis(data)
    
    print_summary_stats(data)
    
    print("\n[v] Analysis complete! Generated PDF files:")
    print("   - outputs/prefill_analysis.pdf")
    print("   - outputs/decode_analysis.pdf") 