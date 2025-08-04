import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# Load your data
df = pd.read_csv("./tests/verify/verify_and_time_K.csv")

# Group by semifield and kernel_size, then aggregate mean and std
agg = df.groupby(["semifield", "kernel_size"])["train_time"].agg(["mean", "std"]).reset_index()

# Ensure sorting by semifield, then kernel size
agg = agg.sort_values(by=["semifield", "kernel_size"])

# Define colors per semifield
colors = {
    "Standard": "#1f77b4",    # blue
    "SmoothMax": "#d62728",   # red
    "MaxPlus": "#ff7f0e",     # orange
    "MinPlus": "#2ca02c"      # green
}

# Create labels and positions
labels = []
means = []
stds = []
bar_colors = []

# Mapping for custom bar positions
semifields = agg["semifield"].unique()
kernel_sizes = sorted(df["kernel_size"].unique())
bar_width = 0.8
gap = 1.0  # gap between semifield groups

x = []
pos = 0

for semifield in semifields:
    sf_data = agg[agg["semifield"] == semifield]
    for _, row in sf_data.iterrows():
        labels.append(f"{semifield}\nK={int(row['kernel_size'])}")
        means.append(row["mean"])
        stds.append(row["std"])
        bar_colors.append(colors.get(semifield, "gray"))
        x.append(pos)
        pos += 1
    pos += gap  # Add gap after each semifield group

# Plot
fig, ax = plt.subplots(figsize=(14, 6))
ax.bar(x, means, yerr=stds, capsize=5, color=bar_colors, edgecolor='black')
ax.set_ylim([60, 110])

# Custom ticks
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=45, ha='right')

# Labels and title
ax.set_ylabel("Training Time (s)")
ax.set_title("Training Time per Kernel Size, Split by Semifield")
ax.grid(True, axis='y', linestyle='--', alpha=0.5)

# Create legend manually from color mapping
legend_elements = [Patch(facecolor=clr, label=lbl) for lbl, clr in colors.items()]
ax.legend(handles=legend_elements, title="Max-pooling substitute", loc="upper left")
plt.title("Training Times for Max-pooling Hyperparameters")

plt.tight_layout()
plt.savefig("train_time_plot.png", dpi=300, bbox_inches='tight')
