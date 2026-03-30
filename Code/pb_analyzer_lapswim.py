import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
import os

# Accept dynamic session_dir from command-line argument
if len(sys.argv) > 1:
    session_dir = sys.argv[1]
else:
    raise ValueError("Session directory path not provided.")

file_path = os.path.join(session_dir, "pacing_laps.xlsx")
output_dir = os.path.join(session_dir, "plot_laps")
os.makedirs(output_dir, exist_ok=True)
output_summary_file = os.path.join(output_dir, "lap_counts_by_hour.xlsx")
plot_path = os.path.join(output_dir, "pacing_laps_per_hour.png")

# ==== Check if pacing_laps.xlsx exists ====
if not os.path.exists(file_path):
    print(f"⚠️ File not found: {file_path} — skipping lap analysis.")
    sys.exit(0)

# ==== Load and preprocess safely ====
try:
    df = pd.read_excel(file_path)
except Exception as e:
    print(f"⚠️ Could not load pacing_laps.xlsx due to error: {e}")
    sys.exit(0)

# Validate required columns
if 'Datetime' not in df.columns or 'Lap Count' not in df.columns:
    print("⚠️ pacing_laps.xlsx missing required columns — skipping.")
    sys.exit(0)

if df.empty:
    print("⚠️ pacing_laps.xlsx is empty — skipping.")
    sys.exit(0)

# ==== Preprocess ====
df['Datetime'] = pd.to_datetime(df['Datetime'])
df['Hour'] = df['Datetime'].dt.hour
df['Date'] = df['Datetime'].dt.date

# Overall total laps (last non-zero, non-NaN value)
nz = df['Lap Count'].dropna()
nz = nz[nz != 0]
total_laps = float(nz.iloc[-1]) if len(nz) else 0.0

# ==== Compute lap bouts per hour from pacing_laps.xlsx ====
def get_hourly_lap_count(group):
    nonzero = group.loc[group['Lap Count'] > 0, 'Lap Count']
    if nonzero.empty:
        return 0.0
    return float(nonzero.max() - nonzero.min())

hourly_laps = (
    df.groupby(['Date', 'Hour'])
      .apply(get_hourly_lap_count)
      .reset_index(name='LapBouts')
)

# ==== Pivot-style summary -> lap_counts_by_hour.xlsx ====
pivot = hourly_laps.pivot(index='Hour', columns='Date', values='LapBouts').fillna(0.0)
pivot['MeanLaps'] = pivot.mean(axis=1)
pivot['StdLaps'] = pivot.std(axis=1)
pivot = pivot.reindex(range(24), fill_value=0.0)  # ensure all hours present
pivot.reset_index(names='Hour').to_excel(output_summary_file, index=False)
print(f"✅ Saved corrected lap summary to: {output_summary_file}")

# ==== READ the summary file for plotting ====
try:
    summary = pd.read_excel(output_summary_file)
except Exception as e:
    print(f"⚠️ Could not load {output_summary_file} due to error: {e}")
    sys.exit(0)

required_cols = {'Hour', 'MeanLaps', 'StdLaps'}
if not required_cols.issubset(summary.columns):
    print(f"⚠️ {output_summary_file} missing required columns {required_cols} — skipping plot.")
    sys.exit(0)

# Coerce numeric, reindex 0–23, fill NaNs
summary = (
    summary.set_index('Hour')[['MeanLaps', 'StdLaps']]
           .apply(pd.to_numeric, errors='coerce')
           .reindex(range(24), fill_value=0.0)
           .fillna(0.0)
)

# ==== Plot (bars + translucent overlay for error) ====
theta = np.linspace(0.0, 2 * np.pi, 24, endpoint=False)
width = 2 * np.pi / 24

radii  = summary['MeanLaps'].to_numpy(dtype=float)
errors = summary['StdLaps'].to_numpy(dtype=float)
radii_with_std = radii + np.nan_to_num(errors)

# radial max that includes overlay
max_val = np.nanmax(radii_with_std) if radii_with_std.size else 0.0
max_val = 5 if (not np.isfinite(max_val) or max_val == 0) else float(np.ceil(max_val))
step = max(1, int(max_val // 4))
yticks = np.arange(0, max_val + 1, step=step)

fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))

# Mean bars
ax.bar(theta, radii, width=width, color='tomato', edgecolor='black', label='Mean')

# Translucent overlay showing mean + SD
ax.bar(theta, radii_with_std, width=width, color='tomato', alpha=0.3, edgecolor='none', label='Mean + Standard Deviation')

# Polar formatting
ax.set_theta_zero_location('N')
ax.set_theta_direction(-1)
ax.set_xticks(theta)
ax.set_xticklabels([f"{h}:00" for h in range(24)], fontsize=12)

ax.set_yticks(yticks)
ax.set_yticklabels([str(int(y)) for y in yticks], fontsize=10, fontweight='bold')
ax.set_rlabel_position(225)

# Title + total laps from pacing_laps.xlsx (last non-zero)
ax.set_title("Loop Swimming Laps/Hr", fontsize=16, fontweight='bold')
fig.text(0.75, 0.95, f"Total # Loop Swimming Laps: {total_laps:.1f}",
         fontsize=14, fontweight='bold', ha='center', va='top', color='darkred')

ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1.1))

plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"✅ Saved polar chart to: {plot_path}")
