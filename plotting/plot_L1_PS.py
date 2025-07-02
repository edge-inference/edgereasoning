import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Sample data
data = {
    "model": [
        "L1-Qwen-1.5B-Exact", "L1-Qwen-1.5B-Exact",
        "L1-Qwen-1.5B-Exact", "L1-Qwen-1.5B-Exact"
    ],
    "token_budget": [128, 128, 256, 256],
    "scaling_factor": [8, 16, 8, 16],
    "accuracy": [0.390333333, 0.41, 0.447, 0.456],
    'latency_seconds': [0.654733149, 0.726727363, 1.23028788, 1.259383664]

}
df = pd.DataFrame(data)

# Define colorblind-friendly colors and patterns
colors = ['#3498db', '#ffa726']  # Pretty blue and lighter orange
patterns = ['+', '////']  # Solid and diagonal hatching

# Set up for group bar chart
token_budgets = sorted(df["token_budget"].unique())
scaling_factors = sorted(df["scaling_factor"].unique())
bar_width = 0.2
x = np.arange(len(scaling_factors))

# Plot
fig, ax = plt.subplots(figsize=(3, 3))
for i, budget in enumerate(token_budgets):
    subset = df[df["token_budget"] == budget]
    accuracies = [subset[subset["scaling_factor"] == sf]["accuracy"].values[0] for sf in scaling_factors]
    ax.bar(x + i * bar_width, accuracies, width=bar_width, 
           label=f'Token Budget {budget}', color=colors[i]) # , hatch=patterns[i]

# Add horizontal line
ax.axhline(y=0.43, color='black', linestyle='-', linewidth=1)
ax.text(0.6, 0.43, 'Sequential Scaling \n w/o Token Budget', va='bottom', ha='center', fontsize=10, color='black')

# Final touches
ax.set_xlabel('Parallel Scaling Factor')
ax.set_ylabel('Accuracy')
# ax.set_title('Accuracy vs. Parallel Scaling Factor by Token Budget')
ax.set_xticks(x + bar_width / 2)
ax.set_xticklabels(scaling_factors)
ax.set_ylim(0.35, 0.5)
ax.legend()

# plt.tight_layout()
# plt.show()
# Save the figure to PDF
plt.savefig('L1_accuracy.pdf', dpi=100, bbox_inches='tight')


# Pivot data for grouped bar chart
pivot_df = df.pivot(index='token_budget', columns='scaling_factor', values='latency_seconds')
pivot_df = pivot_df[sorted(pivot_df.columns)]  # Sort by scaling factor

# Plotting with custom colors and patterns
fig, ax = plt.subplots(figsize=(3, 3))
bars = pivot_df.plot(kind='bar', ax=ax, width=0.6, 
                     color=['#2ca02c', '#ff7f0e'],  # Classic green and orange
                     legend=True)

# Add hatching patterns to the bars
# bar_patches = ax.patches
# patterns_latency = ['\\', 'x']  # Different patterns for each scaling factor
# for i, patch in enumerate(bar_patches):
#     # Apply pattern based on the scaling factor (column), not the bar index
#     scaling_factor_index = i % len(patterns_latency)
#     patch.set_hatch(patterns_latency[scaling_factor_index])

# ax.set_title('Latency vs. Token Budget and Scaling Factor')
ax.set_xlabel('Token Budget')
ax.set_ylabel('Latency (s)')
ax.legend(title='Scaling Factor')
# ax.grid(axis='y', linestyle='--', linewidth=0.5)

plt.xticks(rotation=0)
# plt.tight_layout()
# plt.show()

plt.savefig('L1_latency.pdf', dpi=100, bbox_inches='tight')


