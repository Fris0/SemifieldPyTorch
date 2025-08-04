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

for idx, kernel_size in enumerate(kernel_sizes):
    fig, ax = plt.subplots(1, 1, figsize=(20, 15))
    for semifield in semifields:
        temp = df[(df["semifield"] == semifield) & (df["kernel_size"] == kernel_size)]
        # Convert to seconds
        time_s = temp["mean_time_ms"] / 1000
        if semifield != "Standard":
            ax.plot(temp["mb"], time_s, 
                    label=f"{semifield} with kernel size {kernel_size}", 
                    linewidth=5, color=colors[semifield])
            ax.scatter(temp["mb"], time_s, color=colors[semifield])
        else:
            ax.plot(temp["mb"], time_s, 
                    label=f"MaxPool2d with kernel size {kernel_size}", 
                    linewidth=5, color=colors[semifield])
            ax.scatter(temp["mb"], time_s, color=colors[semifield])

    # Log scale for time in seconds
    ax.set_yscale("log")
    ax.set_ylim(10**-3.3, 10**-0.5)
    ax.set_xlim(5, 72.5)

    # Axis formatting
    ax.tick_params(axis='both', which='major', labelsize=40)
    for spine in ax.spines.values():
        spine.set_linewidth(4)

    ax.set_xlabel("MB", fontsize=50)
    ax.set_ylabel("Mean Time (s)", fontsize=50)
    ax.set_title(f"Performance for Kernel Size {kernel_size}", fontsize=60)
    ax.legend(fontsize=30)

    plt.tight_layout()
    plt.savefig(f"./tests/performance/{kernel_size}_size_{gpu}.png")