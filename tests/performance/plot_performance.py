import pandas as pd
import matplotlib.pyplot as plt

# Load data
gpu = input()
df = pd.read_csv(f"./tests/performance/performance{gpu}.csv")

# Sort values
df = df.sort_values(by=["semifield", "kernel_size"])

# Colors
colors = {
    "Standard": "#1f77b4",
    "SmoothMax": "#d62728",
    "MaxPlus": "#ff7f0e",
    "MinPlus": "#2ca02c"
}

semifields = ["MaxPlus", "MinPlus", "SmoothMax", "Standard"]
kernel_sizes = [2, 3, 5]

fig, ax = plt.subplots(1, 3, figsize=(48, 15))
for idx, kernel_size in enumerate(kernel_sizes):
    for semifield in semifields:
        temp = df[(df["semifield"] == semifield) & (df["kernel_size"] == kernel_size)]
        # Convert to seconds
        time_s = temp["mean_time_ms"] / 1000
        if semifield != "Standard":
            ax[idx].plot(temp["mb"], time_s, 
                    label=f"{semifield} with kernel size {kernel_size}", 
                    linewidth=5, color=colors[semifield])
            ax[idx].scatter(temp["mb"], time_s, color=colors[semifield])
        else:
            ax[idx].plot(temp["mb"], time_s, 
                    label=f"MaxPool2d with kernel size {kernel_size}", 
                    linewidth=5, color=colors[semifield])
            ax[idx].scatter(temp["mb"], time_s, color=colors[semifield])
        ax[idx].set_yscale("log")
        ax[idx].set_ylim(10**-4.5, 1)
        ax[idx].set_xlim(-5, 128)
        # Axis formatting
        ax[idx].tick_params(axis='both', which='major', labelsize=40)
        for spine in ax[idx].spines.values():
            spine.set_linewidth(4)

        ax[idx].set_xlabel("MB", fontsize=50)
        ax[idx].set_ylabel("Mean Time (s)", fontsize=50)
        ax[idx].set_title(f"Performance for Kernel Size {kernel_size}", fontsize=30)
        ax[idx].legend(fontsize=20)

plt.tight_layout(pad=0)  # Remove padding between subplots and figure edges
plt.savefig(f"./tests/performance/total_size_{gpu}.png", bbox_inches='tight', pad_inches=0.05)