import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# --- Configuration ---
log_file = "training.log" 
# ---------------------

# Regex to parse the log file
pattern = re.compile(r"Step:\s*(\d+).*?PPL:\s*([0-9]+\.[0-9]+)", re.IGNORECASE)

steps = []
ppls = []

print(f"Reading log file: '{log_file}'")
try:
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                ppls.append(float(match.group(2)))
except FileNotFoundError:
    print(f"ERROR: Log file '{log_file}' not found.")
    exit()

if not steps:
    print("No data extracted.")
    exit()

df = pd.DataFrame({'Global Step': steps, 'PPL': ppls})

# --- Step 1: Convert to Log-Loss Space ---
# Smoothing acts better on linear Loss than exponential PPL
df['Loss'] = np.log(df['PPL'])

# --- Step 2: Dual Smoothing ---
# Short Window: Captures local volatility (the "jaggies")
# Long Window: Captures the macro trend (the convergence)
short_span = max(5, int(len(df) * 0.02))  # 2% of history
long_span  = max(20, int(len(df) * 0.15)) # 15% of history

df['Loss_Short'] = df['Loss'].ewm(span=short_span, adjust=False).mean()
df['Loss_Long']  = df['Loss'].ewm(span=long_span, adjust=False).mean()

# Convert back to PPL for plotting
df['PPL_Short'] = np.exp(df['Loss_Short'])
df['PPL_Long']  = np.exp(df['Loss_Long'])

# --- Step 3: Mathematical Convergence Check ---
# We take the last 15% of the data points to calculate the slope
lookback_idx = int(len(df) * 0.85)
recent_df = df.iloc[lookback_idx:]

# Linear regression on the *Loss* (not PPL) vs Steps
# y = mx + b
if len(recent_df) > 1:
    slope, intercept = np.polyfit(recent_df['Global Step'], recent_df['Loss'], 1)
    
    # Normalize slope magnitude for display (change in loss per 1k steps)
    slope_per_1k = slope * 1000
    
    if slope_per_1k > 0:
        status = "DIVERGING (Overfitting?)"
        status_color = "red"
    elif slope_per_1k > -0.001:
        status = "CONVERGED (Flat)"
        status_color = "green"
    else:
        status = "STILL IMPROVING"
        status_color = "blue"
else:
    slope_per_1k = 0
    status = "INSUFFICIENT DATA"
    status_color = "gray"

print(f"Convergence Status: {status}")
print(f"Slope (Loss change per 1k steps): {slope_per_1k:.5f}")


# --- Plotting ---
plt.style.use('seaborn-v0_8-whitegrid')
fig, ax = plt.subplots(figsize=(12, 7))

# 1. Plot Raw Data (very faint)
ax.plot(df['Global Step'], df['PPL'], color='lightblue', alpha=0.3, linewidth=0.5, label='Raw PPL Noise')

# 2. Plot Short-Term Trend (Thin, shows volatility)
ax.plot(df['Global Step'], df['PPL_Short'], color='cornflowerblue', alpha=0.8, linewidth=1.5, 
        label=f'Local Trend (Span={short_span})')

# 3. Plot Long-Term Trend (Thick, shows convergence)
ax.plot(df['Global Step'], df['PPL_Long'], color='navy', linewidth=3, 
        label=f'Macro Trend (Span={long_span})')

# Add Convergence Info Box
info_text = (f"Status: {status}\n"
             f"Slope (last 15%): {slope_per_1k:.5f} log-loss/1k steps\n"
             f"Current PPL (Trend): {df['PPL_Long'].iloc[-1]:.2f}")

props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=status_color, linewidth=2)
ax.text(0.02, 0.05, info_text, transform=ax.transAxes, fontsize=12,
        verticalalignment='bottom', bbox=props)

ax.set_yscale('log')
ax.set_title(f'Training Convergence Analysis (N={len(df)})', fontsize=16)
ax.set_xlabel("Global Step", fontsize=12)
ax.set_ylabel("Perplexity (Log Scale)", fontsize=12)
ax.legend(loc='upper right', fontsize=11)
ax.grid(True, which="both", ls="--", alpha=0.4)

ax.get_yaxis().set_major_formatter(mticker.ScalarFormatter())

# Intelligent Y-Limits
recent_min = df['PPL_Long'].min()
ax.set_ylim(bottom=max(1, recent_min * 0.9), top=recent_min * 2.5)

plt.tight_layout()
output_filename = "convergence_analysis.png"
plt.savefig(output_filename, dpi=300)
print(f"Saved analysis to '{output_filename}'")
plt.show()