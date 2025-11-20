# ncn_project/plot_log_enhanced.py

import re
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Configuration ---
# I've updated this to match the log file from our training script
log_file = "training.log" 
# The window size for the rolling average. A larger value means a smoother line.
# 50-100 is usually a good range to start with.
SMOOTHING_WINDOW = 500
# ---------------------

# Your original regex pattern is correct and preserved
pattern = re.compile(
    r"Step:\s*(\d+).*?PPL:\s*([0-9]+\.[0-9]+)",
    re.IGNORECASE
)

steps = []
ppls = []

print(f"Reading and parsing log file: '{log_file}'")
try:
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                ppls.append(float(match.group(2)))
except FileNotFoundError:
    print(f"\nERROR: Log file not found at '{log_file}'. Please ensure the file exists in the same directory.")
    exit()

if not steps:
    print("No data points were extracted. Please check the log file content and regex pattern.")
    exit()

print(f"Extracted {len(steps)} data points.")

# --- NEW: Use pandas for easy data handling and smoothing ---
# Convert the lists into a pandas DataFrame
df = pd.DataFrame({'Global Step': steps, 'PPL': ppls})

# Calculate the rolling average to smooth the curve
df['PPL_Smoothed'] = df['PPL'].rolling(window=SMOOTHING_WINDOW, min_periods=1).mean()


# --- ENHANCED: Plotting Section ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the raw, noisy data in the background (lightly for context)
ax.plot(df['Global Step'], df['PPL'], color='lightblue', alpha=0.6, label='Raw Perplexity (per log step)')

# Plot the smoothed data in the foreground (boldly to show the trend)
ax.plot(df['Global Step'], df['PPL_Smoothed'], color='blue', linewidth=2.5, label=f'Smoothed Perplexity ({SMOOTHING_WINDOW}-step rolling average)')

# --- KEY CHANGE 1: Set Y-axis to logarithmic scale ---
ax.set_yscale('log')

# --- KEY CHANGE 2 & Other Enhancements: Formatting ---
ax.set_title('Smoothed Training Perplexity vs Global Step (Log Scale)', fontsize=16)
ax.set_xlabel("Global Step", fontsize=12)
ax.set_ylabel("Perplexity (PPL) - Log Scale", fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, which="both", ls="--") # Grid for both major and minor ticks on log scale

# Improve tick formatting for log scale to show regular numbers
ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())
ax.tick_params(axis='both', which='major', labelsize=10)

# Set sensible axis limits to focus on the interesting part of the curve
ax.set_xlim(left=0) 
min_ppl_smoothed = df['PPL_Smoothed'].min()
if min_ppl_smoothed > 0:
    # Start the y-axis just below the lowest smoothed PPL
    ax.set_ylim(bottom=max(1, min_ppl_smoothed * 0.9))

plt.tight_layout()

# Save the plot with a new name and higher resolution
output_filename = "smoothed_perplexity_plot.png"
plt.savefig(output_filename, dpi=300)
print(f"Saved smoothed plot as '{output_filename}'")

# Also display the plot
plt.show()