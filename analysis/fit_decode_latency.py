import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import pandas as pd


def parse_xlsx(xlsx_path, max_entries=1000, start_entry=0):
    ref_data = {}
    target_sheet_names = ["DeepSeek-R1-Distill-Llama-8B", "DeepSeek-R1-Distill-Qwen-1_5B", "DeepSeek-R1-Distill-Qwen-14B"]
    model_name_map = {
        "DeepSeek-R1-Distill-Llama-8B": "DSR1-Llama-8B",
        "DeepSeek-R1-Distill-Qwen-1_5B": "DSR1-Qwen-1.5B",
        "DeepSeek-R1-Distill-Qwen-14B": "DSR1-Qwen-14B",
    }
    df = pd.read_excel(xlsx_path, sheet_name=None)
    for sheet_name, sheet in df.items():
        truncated_sheet = sheet.iloc[start_entry:start_entry + max_entries]

        print(sheet_name, sheet)
        if sheet_name in target_sheet_names:
            ref_data[model_name_map[sheet_name]] = {}
            for _, row in truncated_sheet.iterrows():
                key = (row['input_tokens'], row['output_tokens'])
                ref_data[model_name_map[sheet_name]][key] = [row['ttft']/1000, row['decode_time'] / 1000, row['total_time_ms'] / 1000]

    return ref_data


# JENNY: this is the function that we use to fit the latency data, moved it to latency model.py
def latency_function(m, n, I, O):
    """
    Fit function: latency = O * (m*I + n + m*(O-1)/2)
    """
    return O * (m*I + n + m*(O-1)/2)

def fit_model_data(model_data, model_name):
    """
    Fit the latency data for a specific model to find m and n parameters
    """
    # Extract data points
    I_values = []
    O_values = []
    latency_values = []
    
    for (I, O), latency in model_data.items():
        I_values.append(I)
        O_values.append(O)
        latency_values.append(latency)
    
    I_values = np.array(I_values)
    O_values = np.array(O_values)
    latency_values = np.array(latency_values)
    
    # Fit the curve
    try:
        # Create a wrapper function that takes independent variables first
        def fit_func(data, m, n):
            I, O = data
            return latency_function(m, n, I, O)
        
        popt, pcov = curve_fit(fit_func, (I_values, O_values), latency_values)
        m, n = popt
        
        # Calculate R-squared
        y_pred = latency_function(m, n, I_values, O_values)
        ss_res = np.sum((latency_values - y_pred) ** 2)
        ss_tot = np.sum((latency_values - np.mean(latency_values)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        
        return m, n, r_squared, I_values, O_values, latency_values, y_pred
        
    except Exception as e:
        print(f"Error fitting {model_name}: {e}")
        return None, None, None, I_values, O_values, latency_values, None


def plot_latency_varying_input_length():
    # "DSR1-Llama-8B"
    data = {(1, 128): 11.57247281074524, (256, 128): 11.577944040298462, (512, 256): 23.379115343093872, (1024, 128): 11.68675446510315, (2048, 128): 11.796239614486694, (4096, 128): 11.934277057647703}

    tbt_dict = {}

    for key, value in data.items():
        tbt_dict[key[0]] = value / key[1]
    print(tbt_dict)
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 'o', 'o']  # Circle, Square, Triangle

    fig, ax = plt.subplots(figsize=(4, 4))
    
    # Extract input lengths and time between tokens
    input_lengths = list(tbt_dict.keys())
    time_between_tokens = list(tbt_dict.values())
    
    # Convert from ms to seconds
    time_between_tokens_s = [t for t in time_between_tokens]
    
    # Plot the data
    ax.scatter(input_lengths, time_between_tokens_s, s=80, alpha=0.8, color=colors[0], 
              marker=markers[0], label='DSR1-Llama-8B')
    
    # Add a trend line
    z = np.polyfit(input_lengths, time_between_tokens_s, 1)
    p = np.poly1d(z)
    ax.plot(input_lengths, time_between_tokens_s, '--', color=colors[0], alpha=0.7, linewidth=2)
    
    ax.set_xlabel('Input Length', fontsize=12)
    ax.set_ylabel('Time Between Tokens (s)', fontsize=12)
    # ax.set_title('Time Between Tokens vs Input Length', fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('time_between_tokens_vs_input_length.pdf', bbox_inches='tight')
    plt.show()


def plot_latency_fixed_input_length():
    data = {
        "DSR1-Qwen-1.5B": {
            (512, 128): 3.79905271530151,
            (512, 256): 7.14176154136657,
            (512, 512): 13.920802116394,
            (512, 1024): 27.7180814743042,
            (86, 1): 0.04810476303100586, (128, 1): 0.04881095886230469, (256, 1): 0.05393338203430176, (381, 1): 0.07274389266967773, (508, 1): 0.0865316390991211, (632, 1): 0.10982346534729,

        },        
        "DSR1-Llama-8B": {
            (512, 128): 12.9130632877349,
            (512, 256): 26.114182472229,
            (512, 512): 52.1693513393402,
            (512, 1024): 104.540081024169,
        #     (1, 1024): 93.4564197063446, (1, 128): 11.57247281074524, (1, 256): 23.234781503677368, (1, 512): 46.588874101638794,
        #     (128, 1024): 93.7253770828247, (128, 128): 11.61313271522522, (128, 256): 23.32401752471924, (128, 512): 46.76404881477356,
        #     (256, 1024): 93.61614060401917, (256, 128): 11.577944040298462, (256, 256): 23.256435871124268, (256, 512): 46.6482572555542,
        #     (511, 1024): 93.99799036979675, (511, 256): 23.379115343093872, (511, 512): 46.856157541275024, 
        #     (1013, 1024): 94.41459035873412, (1013, 128): 11.68675446510315, (1013, 256): 23.465210914611816, (1013, 512): 47.05947732925415, 
        #     (2048, 1024): 95.23411154747009, (2048, 128): 11.796239614486694, (2048, 256): 23.6745924949646, (2048, 512): 47.488571882247925,
        #     (4096, 1024): 96.44486570358276, (4096, 128): 11.934277057647703, (4096, 256): 23.970098733901978, (4096, 512): 48.0773663520813,
        #     (86, 1): 0.135932207107544, (128, 1): 0.1476736068725586, (256, 1): 0.2247731685638428, (384, 1): 0.3132507801055908, (491, 1): 0.4047152996063232, (630, 1): 0.5512475967407227,            
        },
        "DSR1-Qwen-14B": {
            (512, 128): 23.7478957176208,
            (512, 256): 47.4743678569793,
            (512, 512): 95.3553988933563,
            (512, 1024): 190.463392972946,
            (86, 1): 0.2438251972198486, (128, 1): 0.2686965465545654, (256, 1): 0.4200937747955322, (381, 1): 0.5983071327209473, (508, 1): 0.7612628936767578, (632, 1): 1.023871660232544,            
        }
    }
    
    # xlsx_path = "validation/full_mmlu_by_model_tegra.xlsx"
    # ref_data = parse_xlsx(xlsx_path, max_entries=100, start_entry=100)

    # ref_latency = {}
    # for model_name, model_data in ref_data.items():
    #     ref_latency[model_name] = {}
    #     for key, value in model_data.items():
    #         ref_latency[model_name][key] = value[2]
    # print(ref_latency)

    # data = {
    #     "DSR1-Qwen-14B": {
    #         (512, 128): 23.7478957176208,
    #         (512, 256): 47.4743678569793,
    #         (512, 512): 95.3553988933563,
    #         (512, 1024): 190.463392972946,
    #         (86, 1): 0.2438251972198486, (128, 1): 0.2686965465545654, (256, 1): 0.4200937747955322, (381, 1): 0.5983071327209473, (508, 1): 0.7612628936767578, (632, 1): 1.023871660232544,            
    #     }
    # }
    # data["DSR1-Qwen-1.5B"] = ref_latency["DSR1-Qwen-1.5B"]
    # data["DSR1-Llama-8B"] = ref_latency["DSR1-Llama-8B"]    
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
    markers = ['o', 'o', 'o']  # Circle, Square, Triangle

    fig, ax = plt.subplots(figsize=(4, 4))
    for idx, (model_name, model_data) in enumerate(data.items()):
        print(f"\nFitting {model_name}:")
        m, n, r_squared, I_vals, O_vals, lat_vals, y_pred = fit_model_data(model_data, model_name)
        
        if m is not None:
            # Create a range of output lengths for plotting
            O_plot = np.linspace(1, 4096, 100)
            I_fixed = I_vals[0] if len(I_vals) > 0 else 512
            
            # Calculate predicted values for the plot
            y_plot = latency_function(m, n, I_fixed, O_plot)
            
            # Plot the fitted function
            ax.plot(O_plot, y_plot, '--', color=colors[idx], linewidth=2)
            
            # Plot the actual data points
            ax.scatter(O_vals, lat_vals, s=60, alpha=0.7, color=colors[idx], 
                      marker=markers[idx], label=f'{model_name}')
    
    ax.set_xlabel('Output Length', fontsize=12)
    ax.set_ylabel('Decode Latency (s)', fontsize=12)
    # ax.set_title('Latency vs Output Length for All Models', fontsize=14)
    ax.legend(fontsize=10)
    # ax.grid(True, alpha=0.3)
    
    plt.tight_layout() 
    plt.savefig('latency_fitting_results_fixed_input_length.pdf',bbox_inches='tight')
    plt.show()


def fit_orin_data():
    xlsx_path = "validation/full_mmlu_by_model_tegra.xlsx"
    ref_data = parse_xlsx(xlsx_path, max_entries=100, start_entry=0)

    ref_latency = {}
    for model_name, model_data in ref_data.items():
        ref_latency[model_name] = {}
        for key, value in model_data.items():
            ref_latency[model_name][key] = value[2]
    print(ref_latency)

    # data = {
    #     "DSR1-Qwen-14B": {
    #         (512, 128): 23.7478957176208,
    #         (512, 256): 47.4743678569793,
    #         (512, 512): 95.3553988933563,
    #         (512, 1024): 190.463392972946,
    #         (86, 1): 0.2438251972198486, (128, 1): 0.2686965465545654, (256, 1): 0.4200937747955322, (381, 1): 0.5983071327209473, (508, 1): 0.7612628936767578, (632, 1): 1.023871660232544,            
    #     }
    # }
    data = {}
    data["DSR1-Qwen-1.5B"] = ref_latency["DSR1-Qwen-1.5B"]
    data["DSR1-Llama-8B"] = ref_latency["DSR1-Llama-8B"]
    data["DSR1-Qwen-14B"] = ref_latency["DSR1-Qwen-14B"]

    model_coeffs_dict = {}
    # Fit each model
    results = {}
    for model_name, model_data in data.items():
        print(f"\nFitting {model_name}:")
        m, n, r_squared, I_vals, O_vals, lat_vals, y_pred = fit_model_data(model_data, model_name)
        
        if m is not None:
            results[model_name] = {
                'm': m,
                'n': n,
                'r_squared': r_squared,
                'I_values': I_vals,
                'O_values': O_vals,
                'latency_values': lat_vals,
                'predicted_values': y_pred
            }
            
            print(f"  m = {m:.6e}")
            print(f"  n = {n:.6f}")
            print(f"  R² = {r_squared:.6f}")
            model_coeffs_dict[model_name] = {'m': m, 'n': n}
            # Print the fitted function
            print(f"  Fitted function: latency = O * ({m:.6f}*I + {n:.6f} + {m:.6f}*(O-1)/2)")
            print(f"  Simplified: latency = O * ({m:.6f}*I + {n:.6f} + {m:.6f}*(O-1)/2)")
    

    print(f'decode_coeffs = {model_coeffs_dict}')
    
    # Create 3D plots to visualize the fit
    fig = plt.figure(figsize=(15, 5))
    
    for i, (model_name, result) in enumerate(results.items()):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Plot actual data points
        ax.scatter(result['I_values'], result['O_values'], result['latency_values'], 
                  color='blue', label='Actual', s=50)
        
        # Create a meshgrid for the fitted surface
        I_range = np.linspace(min(result['I_values']), max(result['I_values']), 20)
        O_range = np.linspace(min(result['O_values']), max(result['O_values']), 20)
        I_mesh, O_mesh = np.meshgrid(I_range, O_range)
        
        # Calculate predicted values on the meshgrid
        lat_mesh = latency_function(result['m'], result['n'], I_mesh.flatten(), O_mesh.flatten()).reshape(I_mesh.shape)
        
        # Plot the fitted surface
        ax.plot_surface(I_mesh, O_mesh, lat_mesh, alpha=0.3, color='red', label='Fitted')
        
        ax.set_xlabel('Input Length (I)')
        ax.set_ylabel('Output Length (O)')
        ax.set_zlabel('Latency (ms)')
        ax.set_title(f'{model_name}\nm={result["m"]:.6f}, n={result["n"]:.6f}\nR²={result["r_squared"]:.6f}')
    
    plt.tight_layout()
    plt.savefig('latency_fitting_results_3d.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Print summary table
    print("\n" + "="*80)
    print("SUMMARY OF FITTING RESULTS")
    print("="*80)
    print(f"{'Model':<20} {'m':<12} {'n':<12} {'R²':<10}")
    print("-"*80)
    for model_name, result in results.items():
        print(f"{model_name:<20} {result['m']:<12.6f} {result['n']:<12.6f} {result['r_squared']:<10.6f}")




if __name__ == "__main__":
    fit_orin_data()
    # plot fig 3 in the paper
    # plot_latency_fixed_input_length()
    # plot_latency_varying_input_length()
