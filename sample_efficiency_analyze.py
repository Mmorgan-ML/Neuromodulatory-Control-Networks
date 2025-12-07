# ncn_project/sample_efficiency_analyze.py

"""
Interactive Loss and Sample Efficiency Analysis Dashboard for NCN Project.

Features:
1. Unified interface for Grammar (95%) vs Intellectual (99%) convergence analysis.
2. Context-aware grading scales for Sample Efficiency Index (SEI).
3. Interactive toggles using Matplotlib widgets.
4. Overfitting/Saturation detection on the training tail.
5. Auto-Save: Automatically saves clean screenshots (no UI) of both modes on launch.

Usage:
   Run this script in the same directory as 'training.log'.
   The dashboard allows real-time analysis switching without restarting the script.

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0 International License.
To view a copy of this license, visit http://creativecommons.org/licenses/by-sa/4.0/
or send a letter to Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

Original Author: Michael Morgan
Date: 2025-11-24
Github: https://github.com/Mmorgan-ML
Email: mmorgankorea@gmail.com
Twitter: @Mmorgan_ML
"""

import re
import json
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.widgets import RadioButtons

# --- Configuration ---
LOG_FILE = "training.log"
# ---------------------

class LossDashboard:
    def __init__(self, datafile):
        self.datafile = datafile
        self.df = None
        self.config = {
            "batch_size": 16,
            "gradient_accumulation_steps": 4,
            "block_size": 512,
            "world_size": 1
        }
        
        # Load data immediately
        self.load_data()
        
        # Setup Plot
        plt.style.use('seaborn-v0_8-whitegrid')
        self.fig, self.ax = plt.subplots(figsize=(14, 8))
        self.fig.canvas.manager.set_window_title('NCN Training Analysis Dashboard')
        
        # Adjust layout: maximize graph space, minimize margins
        plt.subplots_adjust(left=0.08, right=0.95, top=0.92, bottom=0.08)
        
        # Setup Radio Buttons (Smaller, top-left corner, unobtrusive)
        # [left, bottom, width, height]
        self.ax_radio = plt.axes([0.01, 0.82, 0.12, 0.12], facecolor='#f8f8f8', frameon=True)
        self.ax_radio.set_title("Analysis Mode", fontsize=10, fontweight='bold', pad=4)
        self.radio = RadioButtons(self.ax_radio, ('Grammar (95%)', 'Intellectual (99%)'), activecolor='crimson')
        
        # Reduce radio text size
        for label in self.radio.labels:
            label.set_fontsize(9)
        
        # --- AUTO-SAVE SEQUENCE ---
        # Temporarily hide UI elements for the screenshot
        self.save_snapshots()
        
        # Connect interactive callback
        self.radio.on_clicked(self.update_plot)
        
        # Set final view
        self.update_plot('Grammar (95%)')
        
    def save_snapshots(self):
        """Cycles through modes and saves clean static images (hiding UI)."""
        print("--- Generating Auto-Save Snapshots ---")
        
        # Hide Radio Buttons for the picture
        self.ax_radio.set_visible(False)
        
        # Save Grammar Mode
        self.update_plot('Grammar (95%)')
        filename_gram = "analysis_grammar_95.png"
        plt.savefig(filename_gram, dpi=300)
        print(f"Saved: {filename_gram}")
        
        # Save Intellectual Mode
        self.update_plot('Intellectual (99%)')
        filename_int = "analysis_intellectual_99.png"
        plt.savefig(filename_int, dpi=300)
        print(f"Saved: {filename_int}")
        
        # Restore Radio Buttons for the app
        self.ax_radio.set_visible(True)
        print("--------------------------------------")

    def load_data(self):
        """Parses the log file for config and loss data."""
        try:
            with open(self.datafile, "r", encoding="utf-8") as f:
                content = f.read()
                
                # Parse Config
                json_match = re.search(r'Command Line Arguments:\s*\n.*?({.*?})', content, re.DOTALL | re.MULTILINE)
                if json_match:
                    try:
                        raw_json = json_match.group(1)
                        clean_json = re.sub(r'^\d{4}-\d{2}-\d{2}.*?\[INFO\] ', '', raw_json, flags=re.MULTILINE)
                        self.config.update(json.loads(clean_json))
                        print("Config loaded successfully.")
                    except Exception as e:
                        print(f"Warning: Config parse failed ({e}). Using defaults.")

                # Parse Data
                pattern = re.compile(r"Step:\s*(\d+).*?Loss:\s*([0-9]+\.[0-9]+)", re.IGNORECASE)
                steps = []
                losses = []
                
                iter_lines = content.splitlines()
                for line in iter_lines:
                    match = pattern.search(line)
                    if match:
                        steps.append(int(match.group(1)))
                        losses.append(float(match.group(2)))
                
                if not steps:
                    print("ERROR: No data found in log file.")
                    sys.exit()
                    
                tokens_per_step = (self.config.get('batch_size', 16) * 
                                   self.config.get('gradient_accumulation_steps', 4) * 
                                   self.config.get('world_size', 1) * 
                                   self.config.get('block_size', 512))
                
                self.df = pd.DataFrame({'Global Step': steps, 'Loss': losses})
                self.df['Tokens'] = self.df['Global Step'] * tokens_per_step
                self.df['Loss_Smooth'] = self.df['Loss'].ewm(span=50, adjust=False).mean()
                
        except FileNotFoundError:
            print(f"ERROR: File '{self.datafile}' not found.")
            sys.exit()

    def get_rating(self, sei, mode_key):
        """Returns (Rating Text, Color, Star Count) based on context-aware scales."""
        if '95%' in mode_key:
            # Scale A: Fast initial drop expectations (Grammar)
            if sei > 0.25: return "EXCELLENT", "green", "★★★★★"
            if sei > 0.15: return "GOOD", "blue", "★★★★☆"
            if sei > 0.10: return "AVERAGE", "orange", "★★★☆☆"
            if sei > 0.05: return "BELOW AVG", "darkorange", "★★☆☆☆"
            return "POOR", "red", "★☆☆☆☆"
        else:
            # Scale B: Long-term convergence expectations (Intellectual)
            if sei > 0.10: return "EXCELLENT", "green", "★★★★★"
            if sei > 0.07: return "GOOD", "blue", "★★★★☆"
            if sei > 0.05: return "AVERAGE", "orange", "★★★☆☆"
            if sei > 0.03: return "BELOW AVG", "darkorange", "★★☆☆☆"
            return "POOR", "red", "★☆☆☆☆"

    def calculate_metrics(self, threshold):
        """Calculates SEI and splits data into Active vs Tail based on Learning Velocity."""
        # We use the First Derivative of the smoothed loss (Velocity) to detect convergence.
        # This ensures the convergence point remains static (historical) even as training continues.
        
        # 1. Calculate Velocity (Gradient of Smoothed Loss)
        velocity = np.gradient(self.df['Loss_Smooth'])
        vel_series = pd.Series(velocity, index=self.df.index)
        
        # 2. Smooth the velocity to remove noise and find the structural trend
        # Window=50 is standard for logging frequencies; centers the trend.
        vel_smooth = vel_series.rolling(window=50, center=True, min_periods=1).mean()
        
        # 3. Find Max Learning Speed (The steepest drop in the history of training)
        # Velocity is negative (loss going down), so we want the minimum value.
        min_vel = vel_smooth.min()
        min_vel_idx = vel_smooth.idxmin()
        
        # 4. Define Convergence Thresholds based on Velocity Decay
        # Grammar (95%) -> When speed slows to 5% of Max Speed
        # Intellectual (99%) -> When speed slows to 1% of Max Speed
        target_ratio = 1.0 - threshold
        target_vel = min_vel * target_ratio
        
        # 5. Find the Step where Velocity Flattens
        # We search for the first point AFTER the peak speed where the velocity 
        # rises above the target (becomes flatter/closer to zero).
        candidates = vel_smooth.index[(vel_smooth.index > min_vel_idx) & (vel_smooth > target_vel)]
        
        if not candidates.empty:
            settling_idx = candidates[0]
            is_converged = True
        else:
            settling_idx = len(self.df) - 1
            is_converged = False
            
        cutoff_idx = settling_idx if is_converged else len(self.df) - 1
        active_df = self.df.iloc[:cutoff_idx+1]
        
        sei = 0.0
        trend_line = np.full(len(self.df), np.nan)
        
        if len(active_df) > 10:
            fit_start = int(len(active_df) * 0.05)
            fit_data = active_df.iloc[fit_start:]
            
            if len(fit_data) > 2:
                log_tok = np.log(fit_data['Tokens'])
                log_loss = np.log(fit_data['Loss'])
                slope, intercept = np.polyfit(log_tok, log_loss, 1)
                sei = abs(slope)
                trend_line = np.exp(intercept + slope * np.log(self.df['Tokens']))

        return {
            "sei": sei,
            "is_converged": is_converged,
            "settling_idx": settling_idx,
            "trend_line": trend_line,
            "active_df": active_df
        }

    def update_plot(self, label):
        """Redraws the plot based on the selected mode."""
        self.ax.clear()
        
        threshold = 0.95 if '95%' in label else 0.99
        metrics = self.calculate_metrics(threshold)
        rating_text, rating_color, stars = self.get_rating(metrics['sei'], label)
        
        # --- PLOTTING ---
        self.ax.plot(self.df['Tokens'], self.df['Loss'], color='lightgray', alpha=0.4, linewidth=1, label='Raw Loss')
        
        self.ax.plot(metrics['active_df']['Tokens'], metrics['active_df']['Loss_Smooth'], 
                     color='crimson', linewidth=2.5, label='Active Phase')
        
        if metrics['is_converged']:
            tail_df = self.df.iloc[metrics['settling_idx']:]
            self.ax.plot(tail_df['Tokens'], tail_df['Loss_Smooth'], 
                         color='gray', linestyle='-', linewidth=2, alpha=0.6, label='Tail (Excluded)')
            
            settling_token = self.df.iloc[metrics['settling_idx']]['Tokens']
            self.ax.axvline(x=settling_token, color='black', linestyle=':', alpha=0.5)
            
            if len(tail_df) > 5:
                tail_idx = np.arange(len(tail_df))
                t_slope, _ = np.polyfit(tail_idx, tail_df['Loss_Smooth'], 1)
                if t_slope > 0:
                    self.ax.text(tail_df['Tokens'].iloc[-1], tail_df['Loss_Smooth'].iloc[-1], " ⚠️ Loss Rising", 
                                 color='red', fontsize=10, fontweight='bold', verticalalignment='bottom')
        
        if not np.isnan(metrics['trend_line']).all():
            self.ax.plot(self.df['Tokens'], metrics['trend_line'], color='navy', linestyle='--', linewidth=1.5, 
                         alpha=0.8, label=f'Power Law (SEI={metrics["sei"]:.2f})')

        # --- FORMATTING ---
        self.ax.set_xscale('log')
        self.ax.set_yscale('log')
        
        # Force Y-axis limits based on actual data to prevent trend line skewing.
        # This ensures the view (and ticks) remain identical between modes.
        data_min = self.df['Loss'].min()
        data_max = self.df['Loss'].max()
        self.ax.set_ylim(data_min * 0.9, data_max * 1.1)

        self.ax.set_title(f'Sample Efficiency Dashboard: {label}', fontsize=16)
        self.ax.set_xlabel('Total Tokens Processed (Log Scale)', fontsize=12)
        self.ax.set_ylabel('Loss (Log Scale)', fontsize=12)
        
        self.ax.grid(True, which="both", ls="-", alpha=0.2)
        self.ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, p: f'{x/1e6:.1f}M'))
        self.ax.yaxis.set_major_formatter(mticker.ScalarFormatter())
        
        # Legend (Inside, Upper Right - usually empty in loss curves)
        self.ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='white', framealpha=0.9, edgecolor='lightgray')
        
        # --- SCORECARD BOX ---
        settle_step = self.df.iloc[metrics['settling_idx']]['Global Step']
        status_str = f"Converged (Step {settle_step})" if metrics['is_converged'] else "Still Converging"
        
        info_text = (f"Mode: {label}\n"
                     f"-------------------\n"
                     f"Status: {status_str}\n"
                     f"SEI Score: {metrics['sei']:.4f}\n"
                     f"Rating: {rating_text}\n"
                     f"{stars}\n"
                     f"-------------------\n"
                     f"Current Loss: {self.df['Loss_Smooth'].iloc[-1]:.3f}")
        
        # Box color is fixed to Crimson Red as requested
        props = dict(boxstyle='round', facecolor='white', alpha=0.95, edgecolor='crimson', linewidth=2)
        self.ax.text(0.02, 0.05, info_text, transform=self.ax.transAxes, fontsize=12,
                     verticalalignment='bottom', bbox=props, fontfamily='monospace')

        if plt.get_fignums():
            self.fig.canvas.draw_idle()

if __name__ == "__main__":
    if not os.path.exists(LOG_FILE):
        print(f"File {LOG_FILE} not found.")
    else:
        print("Launching Dashboard...")
        print("Please wait for auto-save cycle to complete...")
        dashboard = LossDashboard(LOG_FILE)
        plt.show()