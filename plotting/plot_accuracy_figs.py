"""
viz's with combined stats
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path

from utils import PathManager, load_dataframe, save_figure

DEFAULT_DPI = 300

NR_DATA = {
    'DSR1-Qwen-1.5B': {'sample_size': 3000, 'total_tokens': 1899985, 'accuracy': 41.0, 'avg_tokens': 633.33},
    'DSR1-LLama-8B': {'sample_size': 3000, 'total_tokens': 2906924, 'accuracy': 51.0, 'avg_tokens': 968.98},
    'DSR1-Qwen-14B': {'sample_size': 3000, 'total_tokens': 1224311, 'accuracy': 69.0, 'avg_tokens': 408.10}
}

LATENCY_DATA = {'DSR1-Qwen-14B-256t': {'input_tokens': 508738, 'output_tokens': 338554, 'input_latency': 1186.1469265841479, 'output_latency': 63268.564811691525, 'input_energy': 12896.065651894713, 'output_energy': 1663226.9361748435, 'total_energy': 1676123.0018267382, 'sample_size': 3000}, 
 'DSR1-Qwen-1.5B-128t': {'input_tokens': 508738, 'output_tokens': 274380, 'input_latency': 169.89662059828345, 'output_latency': 6494.048941426593, 'input_energy': 453.6239769974226, 'output_energy': 96811.31331104535, 'total_energy': 97264.93728804277, 'sample_size': 3000}, 
 'DSR1-LLama-8B-256t_NC': {'input_tokens': 508738, 'output_tokens': 2799047, 'input_latency': 649.0025321215049, 'output_latency': 293074.1166980842, 'input_energy': 2134.9648961287803, 'output_energy': 8651311.656132357, 'total_energy': 8653446.621028487, 'sample_size': 3000}, 
 'DSR1-Qwen-1.5B': {'input_tokens': 508738, 'output_tokens': 2220489, 'input_latency': 169.89662059828228, 'output_latency': 56601.828205056416, 'input_energy': 453.62397699742013, 'output_energy': 1555916.4306453997, 'total_energy': 1556370.054622397, 'sample_size': 3000}, 
 'DSR1-Qwen-1.5B-128t_NC': {'input_tokens': 508738, 'output_tokens': 4422049, 'input_latency': 169.89662059828368, 'output_latency': 113833.99432932264, 'input_energy': 453.62397699741723, 'output_energy': 3293138.7311753845, 'total_energy': 3293592.355152382, 'sample_size': 3000}, 
 'L1-Max': {'input_tokens': 508738, 'output_tokens': 937919, 'input_latency': 169.89662059828117, 'output_latency': 22334.60398451146, 'input_energy': 453.6239769974173, 'output_energy': 441036.40466951806, 'total_energy': 441490.02864651545, 'sample_size': 3000}, 
 'DSR1-Qwen-14B': {'input_tokens': 508738, 'output_tokens': 3953376, 'input_latency': 1186.1469265841404, 'output_latency': 775865.7086657475, 'input_energy': 12896.065651894743, 'output_energy': 22972161.101568345, 'total_energy': 22985057.16722024, 'sample_size': 3000}, 
 'DSR1-LLama-8B': {'input_tokens': 508738, 'output_tokens': 2433294, 'input_latency': 649.0025321215045, 'output_latency': 260842.87969647636, 'input_energy': 2134.964896128796, 'output_energy': 7870829.633870066, 'total_energy': 7872964.598766195, 'sample_size': 3000}, 
 'DSR1-LLama-8B-128t_NC': {'input_tokens': 508738, 'output_tokens': 1310938, 'input_latency': 649.002532121511, 'output_latency': 140168.10268649395, 'input_energy': 2134.964896128813, 'output_energy': 4176153.430750004, 'total_energy': 4178288.395646133, 'sample_size': 3000}, 
 'DSR1-Qwen-1.5B-256t_NC': {'input_tokens': 508738, 'output_tokens': 2204252, 'input_latency': 169.89662059828134, 'output_latency': 54354.38976048776, 'input_energy': 453.623976997417, 'output_energy': 1365568.5386616192, 'total_energy': 1366022.1626386165, 'sample_size': 3000}, 
 'DSR1-Qwen-14B-128t': {'input_tokens': 508738, 'output_tokens': 234730, 'input_latency': 1186.1469265841597, 'output_latency': 43852.83608641214, 'input_energy': 12896.06565189476, 'output_energy': 1130755.6437794769, 'total_energy': 1143651.7094313717, 'sample_size': 3000}, 
 'DSR1-Qwen-1.5B-256t': {'input_tokens': 508738, 'output_tokens': 432230, 'input_latency': 169.89662059828194, 'output_latency': 10233.063088859128, 'input_energy': 453.6239769974175, 'output_energy': 167987.02664425655, 'total_energy': 168440.65062125397, 'sample_size': 3000}, 
 'DSR1-Qwen-14B-256t_NC': {'input_tokens': 508738, 'output_tokens': 1122540, 'input_latency': 1186.146926584137, 'output_latency': 211564.5122790479, 'input_energy': 12896.065651894736, 'output_energy': 5914966.523376907, 'total_energy': 5927862.589028802, 'sample_size': 3000}, 
 'DSR1-LLama-8B-128t': {'input_tokens': 508738, 'output_tokens': 228891, 'input_latency': 649.0025321215038, 'output_latency': 23013.732969052337, 'input_energy': 2134.9648961287976, 'output_energy': 552623.5892547426, 'total_energy': 554758.5541508714, 'sample_size': 3000}, 
 'DSR1-Qwen-14B-128t_NC': {'input_tokens': 508738, 'output_tokens': 1797130, 'input_latency': 1186.1469265841456, 'output_latency': 353087.8978757501, 'input_energy': 12896.065651894736, 'output_energy': 10407554.005346267, 'total_energy': 10420450.070998162, 'sample_size': 3000}, 
 'DSR1-LLama-8B-256t': {'input_tokens': 508738, 'output_tokens': 430856, 'input_latency': 649.0025321215096, 'output_latency': 43334.31998343928, 'input_energy': 2134.964896128807, 'output_energy': 1083580.1558644683, 'total_energy': 1085715.120760597, 'sample_size': 3000}}

LATENCY_DATA_NR = {'DSR1-LLama-8B': {'input_tokens': 452993, 'output_tokens': 548820, 'input_latency': 593.4059474385691, 'output_latency': 55388.434326811235, 'input_energy': 1951.1188809184869, 'output_energy': 1466960.1542205072, 'total_energy': 1468911.2731014257, 'sample_size': 3000}, 
                   'DSR1-Qwen-1.5B': {'input_tokens': 458815, 'output_tokens': 704573, 'input_latency': 163.82145409367686, 'output_latency': 16767.364386237346, 'input_energy': 437.40328243011714, 'output_energy': 347104.92473877827, 'total_energy': 347542.3280212084, 'sample_size': 3000}, 
                   'DSR1-Qwen-14B': {'input_tokens': 458815, 'output_tokens': 542152, 'input_latency': 1091.3584808527155, 'output_latency': 101512.97538768438, 'input_energy': 11865.612386590057, 'output_energy': 2753903.030188635, 'total_energy': 2765768.642575225, 'sample_size': 3000}}

model_names = [
    'DSR1-Qwen-14B-128t',
    'DSR1-LLama-8B-128t_NC-no-cut',
    'DSR1-Qwen-1.5B-128t_NC-no-cut',
    'DSR1-Qwen-14B-128t_NC-no-cut',
    'DSR1-LLama-8B-256t',
    'DSR1-Qwen-1.5B-256t',
    'DSR1-Qwen-14B-256t',
    'DSR1-LLama-8B-256t_NC-no-cut',
    'DSR1-Qwen-1.5B-256t_NC-no-cut',
    'DSR1-Qwen-14B-256t_NC-no-cut'
]

def load_and_process_data(excel_path: str) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name='All_Stats')
    
    processed_data = []
    
    for _, row in df.iterrows():
        section = row['Section']
        model_name = row['Model']
        
        if section == 'Base Models':
            config = 'Base'
        elif section == '128T Budget':
            config = '128T'
        elif section == '128T Budget_No_Cut':
            config = '128T-NC'
        elif section == '256T Budget':
            config = '256T'
        elif section == '256T Budget_No_Cut':
            config = '256T-NC'
        elif section == 'Direct Models':
            config = 'Direct'
        elif section == 'L1':
            config = 'L1'
        else:
            continue
        

        if config == 'Direct':
            base_model_name = model_name
        if config == 'L1':
            base_model_name = 'DSR1-Qwen-1.5B'
        else:
            base_model_name = model_name.replace('-128t_NC', '').replace('-256t_NC', '').replace('-128t', '').replace('-256t', '').replace('-no-cut', '')
        
        if config == 'L1':
            latency_lookup_model_name = 'L1-Max'
        else:     
            latency_lookup_model_name = model_name.replace('-no-cut', '')
        latency_data = LATENCY_DATA[latency_lookup_model_name]
        assert latency_data['sample_size'] == 3000
        avg_latency = (latency_data['input_latency'] + latency_data['output_latency']) / latency_data['sample_size']
        avg_energy = (latency_data['input_energy'] + latency_data['output_energy']) / latency_data['sample_size']
        
        processed_data.append({
            'Model': base_model_name,
            'Config': config,
            'sample_size': int(row['sample_size']),
            'total_tokens': int(row['total_tokens']),
            'accuracy': float(row['accuracy']),
            'avg_tokens': float(row['avg_tokens']),
            'avg_latency': avg_latency,
            'avg_energy': avg_energy,
        })
    
    for model, data in NR_DATA.items():
        latency_energy_data = LATENCY_DATA_NR[model]
        avg_latency = (latency_energy_data['input_latency'] + latency_energy_data['output_latency']) / latency_energy_data['sample_size']
        avg_energy = (latency_energy_data['input_energy'] + latency_energy_data['output_energy']) / latency_energy_data['sample_size']
        processed_data.append({
            'Model': model,
            'Config': 'NR',
            'sample_size': data['sample_size'],
            'total_tokens': data['total_tokens'],
            'accuracy': data['accuracy'],
            'avg_tokens': data['avg_tokens'],
            'avg_latency': avg_latency,
            'avg_energy': avg_energy,
        })
    
    result_df = pd.DataFrame(processed_data)
    result_df = result_df.sort_values(['Model', 'total_tokens'])
    return result_df

def create_visualizations(df: pd.DataFrame, output_dir: str):
    """Create comprehensive visualizations."""
    PathManager.ensure_dirs()
    
    plt.style.use('default')
    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#592E83", "#7FB069", "#4D5382", "#000000"]
    config_colors = dict(zip(['128T', '128T-NC', '256T', '256T-NC', 'Base', 'NR', 'Direct', 'L1'], colors))

    models = df['Model'].unique()
    
    model_sizes = {}
    for model in models:
        size_str = model.split('-')[-1]
        try:
            if 'B' in size_str:
                size = float(size_str.replace('B', ''))
                model_sizes[model] = size
            else:
                model_sizes[model] = 999
        except ValueError:
            model_sizes[model] = 999
    
    models = sorted(models, key=lambda x: model_sizes[x])
    
    model_colors = dict(zip(models, colors[:len(models)]))
    
    df['efficiency'] = df['accuracy'] / (df['avg_tokens'] / 1000)
    
    display_names = {}
    for model in models:
        if 'Qwen' in model:
            if 'Qwen-1.5B' in model:
                display_names[model] = 'DSR1-Qwen-1.5B'
            elif '14B' in model:
                display_names[model] = 'DSR1-Qwen-14B'
            else:
                display_names[model] = f'DSR1-Qwen-{model.split("-")[-1]}'
        elif 'LLama' in model:
            display_names[model] = 'DSR1-Llama-8B'
        else:
            display_names[model] = model
    
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        ax = axes[i]
        
        for config in model_data['Config'].unique():
            config_data = model_data[model_data['Config'] == config]
            ax.scatter(config_data['total_tokens'], config_data['accuracy'],
                      label=config, color=config_colors[config], s=80, alpha=0.8)
            
            for _, row in config_data.iterrows():
                ax.annotate(config, (row['total_tokens'], row['accuracy']),
                           xytext=(8, 8), textcoords='offset points', fontsize=8, fontweight='bold')
        
        ax.set_xlabel('Total Tokens')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(display_names[model])
        ax.legend(title='Configuration')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # save_figure(plt.gcf(), 'total_tokens_vs_accuracy.pdf', chart_type="insight")
    plt.close()
    
    fig, axes = plt.subplots(1, len(models), figsize=(6*len(models), 6))
    if len(models) == 1:
        axes = [axes]
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        ax = axes[i]
        
        for config in model_data['Config'].unique():
            config_data = model_data[model_data['Config'] == config]
            ax.scatter(config_data['avg_tokens'], config_data['accuracy'],
                      label=config, color=config_colors[config], s=120, alpha=0.8)
            
            for _, row in config_data.iterrows():
                ax.annotate(config, (row['avg_tokens'], row['accuracy']),
                           xytext=(8, 8), textcoords='offset points', fontsize=10, fontweight='bold')
        
        ax.set_xlabel('Average Tokens per sequence')
        ax.set_ylabel('Accuracy (%)')
        ax.set_title(display_names[model])
        ax.legend(title='Configuration')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    # save_figure(plt.gcf(), 'avg_tokens_vs_accuracy.pdf', chart_type="insight")
    plt.close()
    
    config_markers = {
        'Base': 'o',
        'NR': '*',       
        '128T': 'X',
        '128T-NC': 'D',
        '256T': '^',
        '256T-NC': 'v',
        'L1': 'P',
        # 'Direct': 'P'
    }
    
    # Create marker legend
    marker_legend_elements = []
    for config, marker in config_markers.items():
        marker_legend_elements.append(plt.Line2D([0], [0], marker=marker, color='gray', 
                                               label=config, linestyle='', markersize=7))
    

    metric_name = 'avg_latency'
    x_label_name_dict = {
        'total_tokens': 'Total Tokens',
        'avg_tokens': 'Average Tokens / Questions',
        'avg_latency': 'Average Latency (s) / Questions',
        'avg_energy': 'Average Energy (J) / Questions'
    }
    xlim_dict = {
        'total_tokens': (0, None),
        'avg_tokens': (0, 1600),
        'avg_latency': (0, None),
        'avg_energy': (0, None)
    }

    for plot_log_scale in [False, True]:
        for metric_name in ['avg_tokens', 'avg_latency', 'avg_energy']:
            # plot avg tokens vs accuracy
            plt.figure(figsize=(6, 4))
            for i, model in enumerate(models):
                model_data = df[df['Model'] == model]
                model_color = model_colors[model]
                
                for config in model_data['Config'].unique():
                    config_data = model_data[model_data['Config'] == config]
                    marker = config_markers.get(config, 'o')
                    
                    plt.scatter(config_data[f'{metric_name}'], config_data['accuracy'],
                            label=display_names[model] if config == model_data['Config'].iloc[0] else "",
                            color=model_color, marker=marker, s=80, alpha=0.8)
                    
                    for _, row in config_data.iterrows():
                        plt.annotate(config, (row[f'{metric_name}'], row['accuracy']),
                                xytext=(4, 4), textcoords='offset points', fontsize=8)# fontweight='bold'
            
            plt.xlabel(x_label_name_dict[metric_name]) # , fontsize=16
            plt.ylabel('MMLU Accuracy (%)') # , fontsize=16
            
            if plot_log_scale:
                plt.xscale('log')
            plt.xlim(xlim_dict[metric_name])
            plt.ylim(0, 90)

            log_str = '_log' if plot_log_scale else ''
            # Combine model and marker legends
            from matplotlib.patches import Patch
            model_legend_elements = [Patch(facecolor=model_colors[model], label=display_names[model]) 
                                for model in models]
            model_legend = plt.legend(handles=model_legend_elements, title='Base Model', 
                                    bbox_to_anchor=(1.0, 0.0), loc='lower right', 
                                    fontsize=8, title_fontsize=8, handletextpad=0.3)
            plt.gca().add_artist(model_legend)
            plt.legend(handles=marker_legend_elements, fontsize=8,
                    loc='upper center', bbox_to_anchor=(0.5, 1.1), ncol=7, handletextpad=0.3) # , title='Token Control'
            plt.tight_layout()
            save_figure(plt.gcf(), f'{metric_name}{log_str}_vs_accuracy_comparison.pdf', chart_type="insight")
            plt.close()
            
    configs = ['128T', '128T-NC', '256T', '256T-NC', 'Base', 'NR', 'Direct']
    
    # fig = plt.figure(figsize=(6*len(models) + 2, 6))
    fig = plt.figure(figsize=(7, 5))

    gs = fig.add_gridspec(1, len(models) + 1, width_ratios=[1] * len(models) + [0.2])
    
    axes = []
    for i in range(len(models)):
        ax = fig.add_subplot(gs[0, i], projection='polar')
        axes.append(ax)
    
    legend_ax = fig.add_subplot(gs[0, -1])
    legend_ax.axis('off')
    
    accuracy_line = None
    efficiency_line = None
    
    for i, model in enumerate(models):
        model_data = df[df['Model'] == model]
        acc_norm = model_data.set_index('Config')['accuracy'] / 100
        eff_norm = model_data.set_index('Config')['efficiency'] / model_data['efficiency'].max()
        
        angles = [i * 2 * 3.14159 / len(configs) for i in range(len(configs))]
        angles += angles[:1]
        
        acc_values = [acc_norm.get(config, 0) for config in configs] + [acc_norm.get(configs[0], 0)]
        eff_values = [eff_norm.get(config, 0) for config in configs] + [eff_norm.get(configs[0], 0)]
        
        acc_line = axes[i].plot(angles, acc_values, 'o-', linewidth=2, label='Accuracy')
        axes[i].fill(angles, acc_values, alpha=0.25)
        eff_line = axes[i].plot(angles, eff_values, 's-', linewidth=2, label='Efficiency [Acc/1K Tokens]')
        axes[i].fill(angles, eff_values, alpha=0.25)
        
        if i == 0:
            accuracy_line = acc_line[0]
            efficiency_line = eff_line[0]
        
        axes[i].set_xticks(angles[:-1])
        axes[i].set_xticklabels(configs)
        axes[i].set_title(display_names[model], pad=20)
    
    legend_ax.legend([accuracy_line, efficiency_line], 
                 ['Accuracy', 'Efficiency [Acc/1K Tokens]'],
                 loc='upper center', bbox_to_anchor=(0.5, 1.0))
    
    plt.tight_layout()
    # save_figure(plt.gcf(), 'performance_radar.pdf', chart_type="insight")
    plt.close()

def main():
    file_path = 'combined_stats.xlsx'
    try:
        df = load_dataframe(file_path, sheet_name='All_Stats')
    except FileNotFoundError:
        print("Excel file combined_stats.xlsx not found. Please run combined.py first.")
        return
    
    df = load_and_process_data(PathManager.get_data_path(file_path))

    print(f"\nComplete Data ({len(df)} rows):")
    print(df)
    
    create_visualizations(df, PathManager.INSIGHT_CHARTS_DIR)
    print(f"\nInsight charts saved to {PathManager.INSIGHT_CHARTS_DIR}/")

if __name__ == '__main__':
    main()
