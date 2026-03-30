import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio.v2 as imageio
import cv2
import sys
import os

# Accept dynamic session_dir from command-line argument
if len(sys.argv) > 1:
    session_dir = sys.argv[1]
else:
    raise ValueError("Session directory path not provided.")

# Then replace fixed paths like:
# file_path = r"C:\Users\lchen\Desktop\pb_analyzer\pacing_laps.xlsx"

# With:
file_path = os.path.join(session_dir, "activity_log.xlsx")  
output_dir = os.path.join(session_dir, "plot_laps")
os.makedirs(output_dir, exist_ok=True)
output_summary_file = os.path.join(output_dir, "activity_counts_by_hour.xlsx")


# ==== Behavior list and color mapping ====
valid_behaviors = ["FORAGING", "HEAD SWINGING", "LOCOMOTION", "RESTING", "SWIMMING", "PACING"]
color_map = {
    "RESTING": "gold",
    "LOCOMOTION": "mediumspringgreen",
    "PACING": "tomato",
    "SWIMMING": "dodgerblue",
    "HEAD SWINGING": "orchid",
    "FORAGING": "darkorange",
    "STANDING STILL": "slategray",
    "No Detections": "lightgray"
}

# ==== Load and process ====
file_path = os.path.join(session_dir, "activity_log.csv")
df = pd.read_csv(file_path, encoding='latin1')  # Now matches actual file format
df['Datetime'] = pd.to_datetime(df['Timestamp'], format='%H:%M:%S')
df['Hour'] = df['Datetime'].dt.hour

# ==== Infer Day by detecting hour resets ====
days = []
current_day = 1
prev_hour = df['Hour'].iloc[0]

for hour in df['Hour']:
    if hour < prev_hour:
        current_day += 1
    days.append(current_day)
    prev_hour = hour

df['Day'] = days

# Filter only valid behaviors
df_filtered = df[df['Behavior'].isin(valid_behaviors)]

# Save Excel
output_excel = os.path.join(output_dir, "filtered_activity_log.xlsx")
with pd.ExcelWriter(output_excel, engine='xlsxwriter') as writer:
    df_filtered.to_excel(writer, sheet_name='With Hour and Day', index=False)
print(f"✅ Saved filtered Excel log to: {output_excel}")

# ==== Polar plot function ====
def plot_behavior_clock_period(df_period, day_num, label, hour_subset):
    hourly_behavior = df_period.groupby(['Hour', 'Behavior'])['Duration (seconds)'].sum().unstack(fill_value=0)
    hourly_behavior = hourly_behavior.reindex(columns=valid_behaviors, fill_value=0)
    hourly_behavior = hourly_behavior.reindex(pd.Index(hour_subset, name='Hour'), fill_value=0)
    hourly_behavior_props = hourly_behavior.div(hourly_behavior.sum(axis=1), axis=0).fillna(0)

    # Compute summary for legend
    total_durations_sec = df_period.groupby('Behavior')['Duration (seconds)'].sum().reindex(valid_behaviors, fill_value=0)
    total_minutes = total_durations_sec / 60
    total_sum_sec = total_durations_sec.sum()
    percentages = (total_durations_sec / total_sum_sec * 100).fillna(0)

    legend_labels = [
        f"{behavior} ({percentages[behavior]:.0f}%, {total_minutes[behavior]:.0f} min)"
        for behavior in valid_behaviors
    ]

    base_angles = np.linspace(0, 2 * np.pi, len(hour_subset), endpoint=False)
    width = (2 * np.pi / len(hour_subset)) / (len(valid_behaviors) + 1)

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)

    for i, behavior in enumerate(valid_behaviors):
        offset_angles = base_angles + (i - len(valid_behaviors) / 2) * width + width / 2
        values = hourly_behavior_props[behavior].values
        ax.bar(offset_angles, values, width=width, label=legend_labels[i],
               color=color_map.get(behavior, 'gray'), edgecolor='black')

    ax.set_xticks(base_angles)
    ax.set_xticklabels([f"{h}:00" for h in hour_subset], fontsize=11, fontweight='bold')
    ax.set_yticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(['20%', '40%', '60%', '80%'], fontsize=10, fontweight='bold')
    ax.set_rlabel_position(45)
    ax.set_title(f"{label} Behavior - Day {day_num}", fontsize=16, fontweight='bold')

    legend = ax.legend(
        title="Behaviours",
        title_fontsize=13,
        fontsize=12,
        loc='upper right',
        bbox_to_anchor=(1.4, 1.1)
    )
    legend.get_title().set_fontweight('bold')
    legend.get_title().set_fontstyle('italic')

    filename = f"day{day_num:02d}_{label.lower()}.png"
    path = os.path.join(output_dir, filename)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path

# ==== Generate plots ====
daytime_imgs = []
nighttime_imgs = []

for day in sorted(df_filtered['Day'].unique()):
    df_day = df_filtered[df_filtered['Day'] == day]
    df_daytime = df_day[df_day['Hour'].between(0, 11)]
    df_nighttime = df_day[df_day['Hour'].between(12, 23)]

    if not df_daytime.empty:
        img1 = plot_behavior_clock_period(df_daytime, day, "Daytime", list(range(0, 12)))
        daytime_imgs.append(img1)
    if not df_nighttime.empty:
        img2 = plot_behavior_clock_period(df_nighttime, day, "Nighttime", list(range(12, 24)))
        nighttime_imgs.append(img2)

# ==== Create GIFs ====
#imageio.mimsave(os.path.join(output_dir, "behavior_clock_daytime.gif"),
  #              [imageio.imread(p) for p in daytime_imgs], duration=100)
#imageio.mimsave(os.path.join(output_dir, "behavior_clock_nighttime.gif"),
   #             [imageio.imread(p) for p in nighttime_imgs], duration=100) #change duration / transition speed of the GIF

def create_video_from_images(image_paths, output_path, fps=0.3):
    # fps = 0.3 means ~3 seconds per frame
    img = cv2.imread(image_paths[0])
    height, width, _ = img.shape
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    for p in image_paths:
        img = cv2.imread(p)
        out.write(img)
    out.release()

# ==== Create Videos Only If Images Exist ====
if daytime_imgs:
    create_video_from_images(daytime_imgs, os.path.join(output_dir, "behavior_clock_daytime.mp4"), fps=0.66)
else:
    print("⚠️ No daytime behavior plots were generated — skipping daytime video.")

if nighttime_imgs:
    create_video_from_images(nighttime_imgs, os.path.join(output_dir, "behavior_clock_nighttime.mp4"), fps=0.66)
else:
    print("⚠️ No nighttime behavior plots were generated — skipping nighttime video.")

# ==== Final Output ====
print("📊 Unique behaviors in dataset:", df['Behavior'].unique())
print("📊 Days found:", df_filtered['Day'].unique())
print(f"✅ Clock-style plots saved to: {output_dir}")

