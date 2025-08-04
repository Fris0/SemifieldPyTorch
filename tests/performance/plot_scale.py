import pandas as pd
import matplotlib.pyplot as plt

# Load data
print("Give gpu name:\n")
gpu = input()
df = pd.read_csv(f"./tests/performance/performance{gpu}.csv")

# Convert milliseconds to seconds
df["time_s"] = df["mean_time_ms"] / 1000

# Calculate average latency per (semifield, kernel_size)
avg_times = df.groupby(["kernel_size", "semifield"])["time_s"].mean().unstack()

# Compute slowdown relative to MaxPool2d (labeled "Standard")
slowdowns = avg_times.div(avg_times["Standard"], axis=0).drop(columns=["Standard"])

# Plot as bar chart
colors = {"MaxPlus": "#ff7f0e", "MinPlus": "#2ca02c", "SmoothMax": "#d62728"}

ax = slowdowns.plot(kind="bar", figsize=(12, 6), color=[colors[col] for col in slowdowns.columns])

# Beautify
ax.set_title("Average Slowdown Relative to MaxPool2d", fontsize=16)
ax.set_xlabel("Kernel Size", fontsize=14)
ax.set_ylabel("Slowdown Factor", fontsize=14)
ax.tick_params(axis='both', labelsize=12)
ax.grid(axis='y', linestyle='--', alpha=0.7)
ax.legend(title="Semifield", fontsize=12, title_fontsize=12)

plt.tight_layout()
plt.savefig(f"./tests/performance/bar_scale{gpu}.png")