"""
Real-time Objective Animal Monitoring (ROAM) Behavioral Monitoring Framework

Implements four modular layers:
1. Detection Layer (Model load + YOLO inference)
2. Alerting Layer (Thresholds + email functions + alert calls)
3. Behaviour Inference Layer (motion correction, head swing window, lap counting, cooldowns, PacingDetector logic)
4. Post-Processing Layer (All exports, summaries, CSV, plots, email summary)

To adapt this framework 🛠:
    1. Replace detection model weights (i.e., the "model = " statement) 
    2. Update behaviour classes (i.e., "ALL_CLASSES" and "behaviors_of_interest" arguments) 
    3. Modify inference rules (e.g., for distinguishing swimming vs. stereotypical loop swimming)
    4. Specify if you are using ROAM to process: 1) real-time video streams or 2) pre-recorded videos. Do this by commenting (#) on or off "capture_and_infer" arguments at very bottom of document. For live video, you must modify the ".env" document and the "camera_ip_pb#" arguments below. For recorded videos, specify the  file path of the target video with "input_video_path = " statement. Also specify in the ".env" file the email sender credentials and the "email_receiver= " & "summary_email_receiver=" arguments below.
  

Core pipeline logic should not require modification.
"""

### Within Terminal, Anaconda Prompt (the working environment, first activate your working evironemnt with command "conda activate [NAME OF ENVIRONMENT]" and set your file directory to where you have placed the ROAM.py script using the command "cd [FILE LOCATION]"; on Windows devices, you can identify file location by right clicking on file or path location and select "Copy as Path" or "Copy Address".

### PYTHON PACKAGE IMPORTS
import subprocess
import cv2
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from ultralytics import YOLO
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import smtplib
import csv
from email.message import EmailMessage
import ssl
import time
import traceback
from collections import defaultdict
from collections import deque
from datetime import timedelta
from datetime import datetime
import requests
from AccessoryScripts.PacingDetector import PacingDetector
from AccessoryScripts import locationtracker
from AccessoryScripts.SessionManager import SessionManager

# ============================================================
# MONITORING SESSION SETTINGS
# ============================================================
SESSION_DURATION_HOURS = 24 #set to 999 if auto restart set to False
AUTO_RESTART_SESSIONS = True #case sensitive; set to true for long-term monitoring

session_manager = SessionManager(
    default_hours=SESSION_DURATION_HOURS,
    auto_restart=AUTO_RESTART_SESSIONS
)

from AccessoryScripts.SessionDiagnostics import SessionDiagnostics
diagnostics = SessionDiagnostics(session_manager)
from AccessoryScripts.DynamicBudget_military import generate_dynamic_activity_budget_military
from AccessoryScripts.DynamicBudget_military import generate_dynamic_activity_budget_military_heatmap
from AccessoryScripts.LoopSwimCriteriaVisualizer import (
    draw_loop_swim_criteria,
    configure as configure_loopswim_visualizer
)
from AccessoryScripts.LoopSwimCounter import (
    LoopSwimCounter
)

#--------------------#
import re
from datetime import datetime, timedelta

def extract_start_time_from_filename(path):
    fname = os.path.basename(path)

    pattern = r"(\d{1,2}_\d{1,2}_\d{4})\s+(\d{1,2}_\d{1,2}_\d{1,2})\s+(AM|PM)"
    match = re.search(pattern, fname)

    if not match:
        raise ValueError(f"Cannot parse datetime from filename: {fname}")

    date_part = match.group(1).replace("_", "-")
    time_part = match.group(2).replace("_", ":")
    ampm = match.group(3)

    dt_str = f"{date_part} {time_part} {ampm}"
    return datetime.strptime(dt_str, "%m-%d-%Y %I:%M:%S %p")

#---------------------------#
last_weather_update = None
cached_temp = "..."
head_swing_counter = 0
alert_sent = False
pacing_alert_sent = False  # New LOOP SWIMMING alert flag
last_pacing_alert_time = None
laps_since_cooldown = 0
cooldown_end_time = None

# ============================================
# HEADSWINGING ALERT PARAMETERS
# ============================================
HEADSWING_Trigger_mins = 14 # THIS NEEDS TO BE FIXED TO ADJUST PARAMETER (tune below by setting "window_duration_minutes" argument below
pacing_alert_interval_minutes = 60  # ✅ Set alert cooldown to 60 minutes; called pacing alert but should be called loopswaim_alert

# ============================================
# LOOP-SWIM ALERT PARAMETERS
# ============================================
# Camera dedicated to loop-swimming monitoring.
LOOPSWIM_CAMERA_IP = "##.###.##.##" # this is a dedicated IP address that was used with an accssory script "locationtracker.py", specifically for drawing dynamically discovered zone boundaries for loop swimming laps

LOOPSWIM_ALERT_LAP_THRESHOLD = 20      # number of laps required
LOOPSWIM_ALERT_WINDOW_MINUTES = 20   # 20 minute rolling window
RETURN_RADIUS = 65
MIN_NEW_ZONE_DISTANCE = 70
MAX_NEW_ZONE_DISTANCE = 180
LOOPSWIM_TIMEOUT_SECONDS = 25

SHOW_LOOPSWIM_CRITERIA = False

configure_loopswim_visualizer(
    RETURN_RADIUS,
    MIN_NEW_ZONE_DISTANCE,
    MAX_NEW_ZONE_DISTANCE
)

# ============================================================
# LOCATION DISPLAY MODULE Polar Bear Mat Habitat",
# ============================================================
#### only the first #'s matter, the names in quotes are customizable and informative to the user
camera_names = {
    0: "PB Mat Habitat",
    1: "PB Hill Habitat",
    2: "PB Main Habitat",
    3: "PB BOH Habitat"
} 

current_camera_location = "UNKNOWN"
last_known_camera_location = "UNKNOWN"

last_detection_time = None

camera_detection_start = {}

show_location_overlay = False

# =========================
# LOAD ROAM LOGO 
# =========================
logo_path = "ROAM_logo.png"
roam_logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

if roam_logo is None:
    print("❌ ERROR: ROAM logo failed to load. Check file path.")
else:
    # Resize logo (recommended)
    logo_width = 200
    aspect_ratio = roam_logo.shape[0] / roam_logo.shape[1]
    logo_height = int(logo_width * aspect_ratio)
    roam_logo = cv2.resize(roam_logo, (logo_width, logo_height))

### temperature widget
def get_temperature():
    url = (
        "https://api.open-meteo.com/v1/forecast"
        "?latitude=43.77&longitude=-79.25&current_weather=true"
    )
    try:
        response = requests.get(url, timeout=5)
        data = response.json()
        temp = data["current_weather"]["temperature"]
        return f"{temp:.1f} C"
    except:
        return "N/A"

# =========================
# LOAD TORONTO ZOO LOGO
# =========================
tz_logo_path = "TorontoZoo_logo.png"
tz_logo = cv2.imread(tz_logo_path, cv2.IMREAD_UNCHANGED)

if tz_logo is None:
    print("❌ ERROR: Toronto Zoo logo failed to load.")
else:
    target_h = 65  # slightly smaller than panel height for padding
    aspect = tz_logo.shape[1] / tz_logo.shape[0]
    target_w = int(target_h * aspect)
    tz_logo = cv2.resize(tz_logo, (target_w, target_h))

#### TEMPERATURE WIDGET
def draw_temp_widget(frame, temp, x, y):
    box_w, box_h = 220, 85
    radius = 12

    # =========================================
    # ROUNDED RECTANGLE FUNCTION
    # =========================================
    def draw_rounded_rect(img, x, y, w, h, r, color):
        # center
        cv2.rectangle(img, (x+r, y), (x+w-r, y+h), color, -1)
        cv2.rectangle(img, (x, y+r), (x+w, y+h-r), color, -1)

        # corners
        cv2.circle(img, (x+r, y+r), r, color, -1)
        cv2.circle(img, (x+w-r, y+r), r, color, -1)
        cv2.circle(img, (x+r, y+h-r), r, color, -1)
        cv2.circle(img, (x+w-r, y+h-r), r, color, -1)

    # =========================================
    # PANEL BACKGROUND
    # =========================================
    overlay = frame.copy()
    draw_rounded_rect(overlay, x, y, box_w, box_h, radius, (40, 40, 40)) # increasing values makes the overlay border lighter
    cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

    # =========================================
    # HEADER (bold effect)
    # =========================================
    cv2.putText(frame, "Toronto Zoo", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 2)

    cv2.putText(frame, "Toronto Zoo", (x + 10, y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (210, 210, 210), 1)

    # =========================================
    # DIVIDER
    # =========================================
    cv2.line(frame, (x + 8, y + 25), (x + box_w - 8, y + 25), (70, 70, 70), 1)

    # =========================================
    # LOCATION
    # =========================================
    cv2.putText(frame, "Scarborough, ON", (x + 10, y + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

    # =========================================
    # TEMP COLOR
    # =========================================
    color = (255, 255, 255)
    try:
        val = float(temp.replace("C", "").strip())
        if val <= 0:
            color = (255, 180, 100)
        elif val >= 25:
            color = (100, 180, 255)
    except:
        pass

    # =========================================
    # TEMPERATURE
    # =========================================
    cv2.putText(frame, temp, (x + 10, y + 72),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

    # =========================================
    # TIME
    # =========================================
    time_str = datetime.now().strftime("%H:%M")
    cv2.putText(frame, time_str, (x + box_w - 55, y + box_h - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)

    return frame

 ### Define email alerts triggered by various behaviours
def send_locomotion_pacing_alert():
    msg = EmailMessage()
    msg['Subject'] = "STEREOTYPY ALERT - POLAR BEAR PACING"
    msg['From'] = email_sender
    msg['To'] = summary_email_receiver
    msg.set_content(
        "Hello Wildlife Care Team,\n\n"
        "One of the polar bears has completed more than 20 pacing laps "
        "while engaged in sustained LOCOMOTION. This may indicate a "
        "stereotypical pacing pattern.\n\n"
        "Please adjust routine as needed and avoid reinforcing the behaviour.\n\n"
        "- - -\n"
        "Toronto Zoo Welfare Science"
    )

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(email_sender, email_password)
        server.send_message(msg)
 
def send_stereotypy_alert():
    msg = EmailMessage()
    msg['Subject'] = "STEREOTYPY ALERT - POLAR BEAR HEAD SWING"
    msg['From'] = email_sender
    msg['To'] = ", ".join(email_receiver)
    msg.set_content(
        "Hello Wildlife Care Team,\n\n"
        "Hudson has been engaged in HEAD SWINGING for the last 10 minutes. Please adjust your routine as needed to prevent accidentally reinforcing the behaviour - avoid feeding, provisioning with enrichment, giving access, or providing attention while they are engaged in the behaviour. Once you have confirmed the behaviour has ceased, you can resume any activities directed towards this individual or provide something extra to positively reinforce them for stopping! \n\n"
        "Depending on the duration of this bout and crowd size at the viewing window, guest messaging/interpretation may be necessary. \n\n"
        "- - -\n"
        "Toronto Zoo Welfare Science"
    )
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(email_sender, email_password)
        server.send_message(msg)
 
def send_pacing_alert():
    msg = EmailMessage()
    msg['Subject'] = "STEREOTYPY ALERT - POLAR BEAR LOOP SWIMMING"
    msg['From'] = email_sender
    msg['To'] =  ", ".join(email_receiver)    # otherwise>>>> ", ".join(email_receiver)
    msg.set_content(
        "Hello Wildlife Care Team,\n\n"
        "One of the polar bears has completed more than 20 laps while SWIMMING. This may indicate a stereotypical behaviour pattern. Please adjust your routine as needed to prevent accidentally reinforcing the behaviour - avoid feeding, provisioning with enrichment, giving access, or providing attention while they are engaged in the behaviour. Once you have confirmed the behaviour has ceased, you can resume any activities directed towards this individual or provide something extra to positively reinforce them for stopping! \n\n"
        "Depending on the duration of this bout and crowd size at the viewing window, guest messaging/interpretation may be necessary. \n\n"
        "- - -\n"
        "Toronto Zoo Welfare Science"
    )
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(email_sender, email_password)
        server.send_message(msg)

 
ENABLE_REALTIME_DISPLAY = True
matplotlib.use('Agg')
 
load_dotenv()
username = os.getenv("CAMERA_USERNAME") #login credentials for accessing RTSP cameras should be saved in a private .env file
password = os.getenv("CAMERA_PASSWORD") 
camera_ip_pb1 = os.getenv("CAMERA_IP_PB1", "____") #replace camera IP address to match your institution's livestreams
camera_ip_pb2 = os.getenv("CAMERA_IP_PB2","____")  #insert camera IP address
camera_ip_pb3 = os.getenv("CAMERA_IP_PB3", "____") #insert camera IP address
camera_ip_pb4 = os.getenv("CAMERA_IP_PB4", "____") #insert camera IP address
email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")
email_receiver = "email@email.com" #INSERT EMAIL ADDRESS OF RECIPIENT(S)
 
camera_urls = [
    f"rtsp://{username}:{password}@{camera_ip_pb1}/axis-media/media.amp",
    f"rtsp://{username}:{password}@{camera_ip_pb2}/axis-media/media.amp",
    f"rtsp://{username}:{password}@{camera_ip_pb3}/axis-media/media.amp",
    f"rtsp://{username}:{password}@{camera_ip_pb4}/axis-media/media.amp"
]
 
model = YOLO("best_640_F_100epoch_NoAug.pt")
 
ALL_CLASSES = [
    "FEEDING", "RESTING", "LOCOMOTION", "PACING", "SWIMMING", "HEAD SWINGING",
    "PLAY", "OBJECT MANIPULATION", "FORAGING", "STANDING STILL", "No Detections"
]
 
behaviors_of_interest = [
    "RESTING", "LOCOMOTION", "PACING", "SWIMMING", "HEAD SWINGING","FORAGING", "No Detections"
]
 
color_map = {
    "RESTING": "gold",
    "LOCOMOTION": "mediumspringgreen",
    "PACING": "tomato",
    "SWIMMING": "dodgerblue",
    "HEAD SWINGING": "orchid",
    "FORAGING": "darkorange",
    "No Detections": "lightgray"
}
 
output_base_dir = "MONITORING SESSION OUTPUTS"
 
 
def export_pacing_laps_to_excel(lap_timestamps, lap_counts, session_dir, start_time):
    if not lap_counts:
        return

    df = pd.DataFrame({
        "Date": [dt.date() for dt in lap_timestamps],
        "Time": [dt.time() for dt in lap_timestamps],
        "Lap Count": lap_counts,
        "Datetime": lap_timestamps
    })

    excel_path = os.path.join(session_dir, "pacing_laps.xlsx")

    if os.path.exists(excel_path):
        df_existing = pd.read_excel(excel_path)
        df = pd.concat([df_existing, df], ignore_index=True)

    df.to_excel(excel_path, index=False)
 
def pad_image_to_match_height(image, target_height, pad_color=(255, 255, 255)):
    h, _, _ = image.shape
    if h < target_height:
        padding = (target_height - h) // 2
        return cv2.copyMakeBorder(image, padding, target_height - h - padding, 0, 0, cv2.BORDER_CONSTANT, value=pad_color)
    return image
 
 
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

def update_ethogram(behavior_log):
    # Return a BGR image (np.ndarray) without touching disk
    spacing = 0.4
    pos_map = {b: i * spacing for i, b in enumerate(behaviors_of_interest)}

    if len(behavior_log) < 2:
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)
        ax.set_title("Polar Bear Activity Budget", fontsize=14)
        ax.text(0.5, 0.5, "Monitoring...", ha='center', va='center', fontsize=16)
        ax.axis('off')

        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.canvas.get_width_height()
        buf = np.asarray(canvas.buffer_rgba())
        rgb = buf[:, :, :3]
        plt.close(fig)

        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    start_time = behavior_log[0][0]
    current_time = behavior_log[-1][0]

    durations = [(behavior_log[i + 1][0] - behavior_log[i][0]).total_seconds()
                 for i in range(len(behavior_log) - 1)]
    x_start = [(behavior_log[i][0] - start_time).total_seconds()
               for i in range(len(behavior_log) - 1)]
    x_end = [x_start[i] + durations[i] for i in range(len(durations))]
    behaviors = [b for _, b in behavior_log[:-1]]

    # Build the figure
    fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=100)  # ~640x480
    for x1, x2, b in zip(x_start, x_end, behaviors):
        if b in color_map:
            ax.hlines(pos_map[b], x1, x2, colors=color_map[b], linewidth=10)

    max_time = x_end[-1] if x_end else 0
    if max_time <= 60:
        xticks = np.linspace(0, max(max_time, 1), 10)
        xlabels = [f"{int(t)}s" for t in xticks]
    elif max_time <= 3600:
        xticks = np.linspace(0, max_time, 10)
        xlabels = [f"{t/60:.1f}m" for t in xticks]
    else:
        xticks = np.linspace(0, max_time, 10)
        xlabels = [f"{t/3600:.2f}h" for t in xticks]

    ax.set_xlabel("Time", fontsize=12)
    ax.set_ylabel("Behaviours", fontsize=12)
    ax.set_title("Polar Bear Dynamic Activity Budget", fontsize=14)
    ax.set_yticks([pos_map[b] for b in behaviors_of_interest])
    ax.set_yticklabels(behaviors_of_interest, fontsize=10)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xlabels, fontsize=10, rotation=45)
    ax.grid(axis='x', linestyle='--', alpha=0.7)

    start_str = start_time.strftime("Session Start: %H:%M, %d %B %Y")
    now_str = current_time.strftime("Current Time: %H:%M, %d %B %Y")
    fig.text(0.01, 0.01, f"{start_str}\n{now_str}",
             ha='left', va='bottom', fontsize=9,
             bbox=dict(facecolor='white', alpha=0.6))

    fig.tight_layout()

    # Render to RGB buffer (no file I/O)
    canvas = FigureCanvas(fig)
    canvas.draw()
    w, h = fig.canvas.get_width_height()
    buf = np.asarray(canvas.buffer_rgba())
    rgb = buf[:, :, :3]
    plt.close(fig)

    # Return BGR for OpenCV
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) 
 
def plot_activity_budget(class_times, total_seconds, output_path):
    percent_budget = {
        cls: (class_times.get(cls, 0) / total_seconds) * 100 if total_seconds > 0 else 0
        for cls in ALL_CLASSES
    }
    behaviors = behaviors_of_interest
    percentages = [percent_budget[b] for b in behaviors]
    labels = [b.replace("_", " ").title() for b in behaviors]
    colors = [color_map[b] for b in behaviors]
 
    plt.figure(figsize=(11, 6))
    y_pos = np.arange(len(behaviors))
    plt.barh(y_pos, percentages, color=colors, edgecolor='black', height=0.7)
    plt.xlabel("Percentage of Time (%)", fontsize=16)
    plt.ylabel("Behaviour", fontsize=16)
    plt.title("Percentage of Time Spent in Each Behaviour", fontsize=17)
    plt.yticks(y_pos, labels, fontsize=16)
    plt.xticks(fontsize=16)
    plt.xlim(0, 100)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    for i, perc in enumerate(percentages):
        plt.text(perc + 1, y_pos[i], f"{perc:.1f}%", va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=600)
    plt.close()
 
def save_activity_budget_csv(class_times, total_seconds, output_csv, start_time, end_time):
    percent_budget = {
        cls: (class_times.get(cls, 0) / total_seconds) * 100 if total_seconds > 0 else 0
        for cls in ALL_CLASSES
    }

    with open(output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Class', 'Time (seconds)', 'Percent budget (%)'])
        for cls in ALL_CLASSES:
            time_sec = class_times.get(cls, 0)
            writer.writerow([cls, f"{time_sec:.2f}", f"{percent_budget[cls]:.2f}"])

        # Add metadata section
        writer.writerow([])
        writer.writerow(['Start Time', start_time.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['End Time', end_time.strftime('%Y-%m-%d %H:%M:%S')])
        writer.writerow(['Total Time (seconds)', f"{total_seconds:.2f}"])


def save_and_email_summary(start_time, behavior_log, laps, session_dir):
    """
    Save the session tables and plots, run post-processing, and email
    the summary. Plotting/email failures are reported without deleting
    outputs that were already created.
    """

    if len(behavior_log) < 2:
        print(
            "[ZooAI] Summary skipped because the session contains "
            "fewer than two timestamps."
        )
        return

    end_time = behavior_log[-1][0]
    total_seconds = (end_time - start_time).total_seconds()

    if total_seconds <= 0:
        print(
            "[ZooAI] Summary skipped because the session duration "
            "is not positive."
        )
        return

    durations = [
        max(
            0.0,
            (behavior_log[i + 1][0] - behavior_log[i][0]).total_seconds()
        )
        for i in range(len(behavior_log) - 1)
    ]

    class_times = defaultdict(float)
    for i, (_, behavior) in enumerate(behavior_log[:-1]):
        class_times[behavior] += durations[i]

    loop_swimming_laps = laps
    head_swinging_time = sum(
        durations[i]
        for i, (_, behavior) in enumerate(behavior_log[:-1])
        if behavior == "HEAD SWINGING"
    ) / 60

    dates = [
        timestamp.strftime("%Y-%m-%d")
        for timestamp, _ in behavior_log[:-1]
    ]
    timestamps = [
        timestamp.strftime("%H:%M:%S")
        for timestamp, _ in behavior_log[:-1]
    ]
    behaviors = [
        behavior
        for _, behavior in behavior_log[:-1]
    ]

    df = pd.DataFrame({
        "Date": dates,
        "Timestamp": timestamps,
        "Behavior": behaviors,
        "Duration (seconds)": durations
    })

    csv_path = os.path.join(
        session_dir,
        "activity_log.csv"
    )
    df.to_csv(csv_path, index=False)

    # These accessories previously raised
    # "arange: cannot compute length" for zero-duration sessions.
    # Finalization now prevents zero-duration input, and each accessory
    # is isolated so one plotting failure does not cancel all outputs.
    try:
        generate_dynamic_activity_budget_military(
            session_manager
        )
    except Exception as error:
        print(
            "⚠️ Dynamic activity-budget plot failed: "
            f"{error}"
        )
        traceback.print_exc()

    try:
        generate_dynamic_activity_budget_military_heatmap(
            session_manager
        )
    except Exception as error:
        print(
            "⚠️ Dynamic activity-budget heatmap failed: "
            f"{error}"
        )
        traceback.print_exc()

    budget_csv_path = os.path.join(
        session_dir,
        "activity_budget.csv"
    )
    save_activity_budget_csv(
        class_times,
        total_seconds,
        budget_csv_path,
        start_time,
        end_time
    )

    ethogram_path = os.path.join(
        session_dir,
        "ethogram.png"
    )
    final_ethogram = update_ethogram(behavior_log)
    cv2.imwrite(ethogram_path, final_ethogram)

    barplot_path = os.path.join(
        session_dir,
        "activity_budget_barplot.png"
    )
    plot_activity_budget(
        class_times,
        total_seconds,
        barplot_path
    )
    script_dir = os.path.dirname(os.path.abspath(__file__))
    combine_script = os.path.join(script_dir, "pb_analyzer_combine.py")
    try:
        print(
            "🧩 Running pb_analyzer_combine.py "
            "for post-processing..."
        )
        subprocess.run(
            ["python", combine_script, session_dir],
            check=True
        )
        print("✅ Post-processing complete.")
    except Exception as error:
        print(
            "❌ Failed to run post-processing script: "
            f"{error}"
        )

    plot_laps_dir = os.path.join(
        session_dir,
        "plot_laps"
    )
    behavior_clock_day_path = os.path.join(
        plot_laps_dir,
        "behaviour_clock_daytime.mp4"
    )
    behavior_clock_night_path = os.path.join(
        plot_laps_dir,
        "behaviour_clock_nighttime.mp4"
    )
    laps_plot_path = os.path.join(
        plot_laps_dir,
        "pacing_laps_per_hour.png"
    )
    behaviour_heatmap_relative_path = os.path.join(
        session_dir,
        "DynamicActivityBudget_military_relative.png"
    )

    summary_lines = [
        (
            f"• {loop_swimming_laps:.1f} loop swimming laps "
            "were detected throughout the monitoring session."
        ),
        (
            f"• {int(round(head_swinging_time))} minutes of "
            "head swinging were detected throughout the "
            "monitoring session."
        )
    ]
    summary_text = "\n".join(summary_lines)

    msg = EmailMessage()
    msg["Subject"] = (
        "Polar Bear Behaviour Monitoring Summary"
    )
    msg["From"] = email_sender
    msg["To"] = summary_email_receiver
    msg.set_content(
        "Hello Wildlife Care Team,\n\n"
        "Please see attached the activity summary files for "
        "the monitoring session.\n\n"
        "Stereotypical Summary:\n"
        f"{summary_text}\n\n"
        "- - -\n"
        "Toronto Zoo Welfare Science"
    )

    for file_path, mime_type, subtype in [
        (csv_path, "text", "csv"),
        (ethogram_path, "image", "png"),
        (barplot_path, "image", "png"),
        (behaviour_heatmap_relative_path, "image", "png"),
        (budget_csv_path, "text", "csv"),
        (behavior_clock_day_path, "video", "mp4"),
        (behavior_clock_night_path, "video", "mp4"),
        (laps_plot_path, "image", "png")
    ]:    
        if os.path.isfile(file_path):
            with open(file_path, "rb") as attachment:
                msg.add_attachment(
                    attachment.read(),
                    maintype=mime_type,
                    subtype=subtype,
                    filename=os.path.basename(file_path)
                )
        else:
            print(
                "⚠️ Warning: File not found and will not "
                f"be attached: {file_path}"
            )

    try:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(
            "smtp.gmail.com",
            465,
            context=context
        ) as server:
            server.login(
                email_sender,
                email_password
            )
            server.send_message(msg)

        print("✅ Monitoring summary email sent.")
    except Exception as error:
        print(
            "⚠️ Session outputs were saved, but the summary "
            f"email could not be sent: {error}"
        )
        traceback.print_exc()


#### ---- below controls session manager, the length or monitoring session, autorestart capability ---- ####

def finalize_monitoring_session(
    session_dir,
    start_time,
    behavior_log,
    laps,
    lap_timestamps,
    lap_counts,
    input_video_path=None,
    end_time=None
):
    """
    Close the final behavioural interval and save all session-level
    outputs. ``end_time`` should be the timestamp of the last frame
    successfully processed by the pipeline.
    """

    if not behavior_log:
        behavior_log.append(
            (start_time, "No Detections")
        )

    if end_time is None:
        end_time = (
            behavior_log[-1][0]
            if input_video_path
            else datetime.now()
        )

    # Never let downstream plotting receive a reversed interval.
    if behavior_log[-1][0] > end_time:
        end_time = behavior_log[-1][0]

    if end_time <= start_time:
        print(
            "[ZooAI] No positive monitoring duration was "
            "processed. Empty session plots were skipped."
        )
        return False

    final_behavior = behavior_log[-1][1]

    if behavior_log[-1][0] < end_time:
        behavior_log.append(
            (end_time, final_behavior)
        )

    session_manager.start_time = start_time
    session_manager.end_time = end_time

    plot_path = os.path.join(
        session_dir,
        "pacing_laps.png"
    )
    plot_pacing_laps(
        lap_timestamps,
        lap_counts,
        plot_path
    )

    export_pacing_laps_to_excel(
        lap_timestamps,
        lap_counts,
        session_dir,
        start_time
    )

    save_and_email_summary(
        start_time,
        behavior_log,
        laps,
        session_dir
    )

    print(
        "[ZooAI] Session outputs saved to: "
        f"{session_dir}"
    )

    return True

#### ---- Above controls session manager, the length or monitoring session, autorestart capability ---- ####

 # --- Pacing Visual ---
def plot_pacing_laps(timestamps, lap_counts, output_path):
    if not timestamps:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(timestamps, lap_counts, marker='o', linestyle='-', color='navy')
    plt.title('Pacing Laps Over Time')
    plt.xlabel('Time')
    plt.ylabel('Lap Count')
    plt.grid(True)
    plt.tight_layout()
  
  # Ensure the directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.savefig(output_path)
    plt.close()
 
# =========================
# OVERLAY LOGO - for ROAM and Toronto Zoo
# =========================
def overlay_logo(background, logo, x, y):
    h, w = logo.shape[:2]

    # Prevent out-of-bounds crash
    if y + h > background.shape[0] or x + w > background.shape[1]:
        return background

    if logo.shape[2] == 4:  # PNG with alpha channel
        alpha = logo[:, :, 3] / 255.0
        for c in range(3):
            background[y:y+h, x:x+w, c] = (
                alpha * logo[:, :, c] +
                (1 - alpha) * background[y:y+h, x:x+w, c]
            )
    else:
        background[y:y+h, x:x+w] = logo

    return background
# =========================
# TORONTO ZOO LOGO PANEL
# =========================
def draw_logo_panel(frame, logo, x, y):
    box_w, box_h = 220, 85

    # --- Center logo ONLY (no background) ---
    h, w = logo.shape[:2]
    logo_x = x + (box_w - w) // 2
    logo_y = y + (box_h - h) // 2

    frame = overlay_logo(frame, logo, logo_x, logo_y)

    return frame
 
def save_loop_swim_alert_snapshot(
    frames,
    session_dir,
    timestamp
):
    """
    Save the current multi-camera grid when a loop-swimming alert
    is issued.
    """

    if not frames:
        print(
            "⚠️ No frames available for loop-swim alert snapshot."
        )
        return

    snapshot_dir = os.path.join(
        session_dir,
        "plot_laps"
    )
    os.makedirs(snapshot_dir, exist_ok=True)

    snapshot_filename = (
        "loopswim_alert_"
        f"{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}"
        ".jpg"
    )
    snapshot_path = os.path.join(
        snapshot_dir,
        snapshot_filename
    )

    combined_snapshot = stack_frames_grid(frames)
    cv2.imwrite(snapshot_path, combined_snapshot)

    print(
        "📸 Loop-swim alert snapshot saved: "
        f"{snapshot_path}"
    )


def capture_and_infer(rtsp_urls=None, input_video_path=None):

    global last_weather_update, cached_temp, roam_logo, tz_logo
    global SHOW_LOOPSWIM_CRITERIA

    # =====================================
    # COMMON SESSION STATE
    # =====================================
    behavior_log = []
    current_behavior = "No Detections"
    previous_behavior = "No Detections"

    detected_behaviors = set()
    last_detected_behaviors = set()

    session_finalized = False

    # =====================================
    # INPUT MODE
    # =====================================
    if input_video_path:
        caps = [cv2.VideoCapture(input_video_path)]

        fps = float(
            caps[0].get(cv2.CAP_PROP_FPS)
        )
        if not np.isfinite(fps) or fps <= 0:
            print(
                "⚠️ Recorded-video FPS could not be read. "
                "Using 30 FPS."
            )
            fps = 30.0

        frame_count_total = float(
            caps[0].get(cv2.CAP_PROP_FRAME_COUNT)
        )

        video_start_time = extract_start_time_from_filename(
            input_video_path
        )
        start_time = video_start_time
        frame_index = 0

    else:
        if not rtsp_urls:
            raise ValueError(
                "rtsp_urls must be supplied in livestream mode."
            )

        caps = [
            cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            for url in rtsp_urls
        ]

        # YOLO inference is intentionally limited to 1 Hz for
        # livestream monitoring. Frames can still be read between
        # inference calls for the .76 MOSSE tracker.
        fps = 1.0
        frame_count_total = None
        frame_index = 0
        start_time = datetime.now()

    last_processed_timestamp = start_time

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            source_name = (
                "Input Video"
                if input_video_path
                else f"Camera {i + 1}"
            )
            print(
                f"Error: Cannot open video stream for {source_name}"
            )

        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0

    # =====================================
    # INFERENCE AND DISPLAY TIMING
    # =====================================
    LIVE_INFERENCE_INTERVAL = 1.0
    RECORDED_DISPLAY_INTERVAL = 1.0 / max(fps, 1.0)

    last_inference_time = 0.0
    last_display_time = 0.0

    last_bear_centroid = None

    # Head-swing samples are added only when YOLO actually runs.
    # Recorded video runs YOLO at its source FPS; livestream mode
    # contributes one sample per second.
    window_duration_minutes = 14
    min_percent_detection = 0.70
    head_swing_sample_rate = (
        fps
        if input_video_path
        else 1.0
    )
    window_size = max(
        1,
        int(
            round(
                window_duration_minutes
                * 60
                * head_swing_sample_rate
            )
        )
    )
    head_swing_window = deque(maxlen=window_size)
    alert_sent = False

    laps = 0.0
    lap_timestamps = []
    lap_counts = []

    laps_since_cooldown = 0
    last_pacing_alert_time = None
    cooldown_end_time = None

    # ==================================
    # LOCOMOTION PACING DETECTOR
    # ==================================
    pacing_detector = PacingDetector()
    last_motion_centroid = None

    session_dir = os.path.join(
        output_base_dir,
        f"session_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
    )
    os.makedirs(session_dir, exist_ok=True)

    session_manager.start(session_dir=session_dir)

    behavior_log.append(
        (start_time, current_behavior)
    )

    print("Monitoring started. Press Ctrl+C to stop.")

    # =====================================
    # LOOP-SWIMMING ACCESSORY
    # =====================================
    num_cameras = len(caps)

    if input_video_path:
        loop_swim_camera_index = 0
        loop_swim_source_name = "Recorded Video"
    else:
        loop_swim_camera_index = next(
            (
                i
                for i, url in enumerate(rtsp_urls)
                if LOOPSWIM_CAMERA_IP in url
            ),
            None
        )

        if loop_swim_camera_index is None:
            raise ValueError(
                "Loop-swimming camera "
                f"{LOOPSWIM_CAMERA_IP} "
                "was not found in rtsp_urls."
            )

        loop_swim_source_name = LOOPSWIM_CAMERA_IP

    loop_swim_counter = LoopSwimCounter(
        return_radius=RETURN_RADIUS,
        min_new_zone_distance=MIN_NEW_ZONE_DISTANCE,
        max_new_zone_distance=MAX_NEW_ZONE_DISTANCE,
        swimming_timeout_seconds=LOOPSWIM_TIMEOUT_SECONDS
    )
    loop_swim_state = loop_swim_counter.get_state()

    print(
        "[LoopSwim] Monitoring source: "
        f"{loop_swim_source_name} "
        f"(camera index {loop_swim_camera_index})"
    )
    print(
        "[LocationTracker] Multi-camera live/last-seen updates enabled."
    )

    try:
        while True:
            # =====================================
            # TRUE FRAME TIMESTAMP
            # =====================================
            if input_video_path:
                timestamp = (
                    video_start_time
                    + timedelta(
                        seconds=frame_index / fps
                    )
                )
                frame_index += 1
            else:
                timestamp = datetime.now()

            manager_session_ended = session_manager.update()
            diagnostics.update()

            if input_video_path:
                video_duration_seconds = (
                    frame_count_total / fps
                    if frame_count_total > 0
                    else None
                )

                if video_duration_seconds is not None:
                    true_end_time = (
                        video_start_time
                        + timedelta(
                            seconds=video_duration_seconds
                        )
                    )

                    if timestamp >= true_end_time:
                        print(
                            "Reached true end of recorded video."
                        )
                        break

                # Recorded sessions are divided using VIDEO time,
                # not the faster wall-clock processing time.
                session_ended = (
                    timestamp - start_time
                ).total_seconds() >= (
                    SESSION_DURATION_HOURS * 3600
                )
            else:
                session_ended = manager_session_ended

            # =====================================
            # SESSION END / OPTIONAL RESTART
            # =====================================
            if session_ended:
                print(
                    "[ZooAI] Session ended → saving session outputs."
                )

                finalized = finalize_monitoring_session(
                    session_dir=session_dir,
                    start_time=start_time,
                    behavior_log=behavior_log,
                    laps=laps,
                    lap_timestamps=lap_timestamps,
                    lap_counts=lap_counts,
                    input_video_path=input_video_path,
                    end_time=last_processed_timestamp
                )
                session_finalized = bool(finalized)

                if not session_manager.auto_restart:
                    print(
                        "[ZooAI] Automatic restart disabled "
                        "→ monitoring complete."
                    )
                    break

                start_time = (
                    timestamp
                    if input_video_path
                    else datetime.now()
                )

                session_dir = os.path.join(
                    output_base_dir,
                    f"session_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}"
                )
                os.makedirs(session_dir, exist_ok=True)

                current_behavior = "No Detections"
                previous_behavior = "No Detections"
                behavior_log = [
                    (start_time, current_behavior)
                ]

                head_swing_window.clear()
                alert_sent = False

                loop_swim_counter.reset()
                loop_swim_state = (
                    loop_swim_counter.get_state()
                )

                laps = 0.0
                lap_timestamps = []
                lap_counts = []

                laps_since_cooldown = 0
                last_pacing_alert_time = None
                cooldown_end_time = None

                pacing_detector = PacingDetector()
                last_motion_centroid = None
                last_bear_centroid = None
                last_detected_behaviors = set()

                frame_count = 0
                last_inference_time = 0.0
                last_display_time = 0.0
                last_processed_timestamp = start_time

                session_manager.start(
                    session_dir=session_dir
                )

                session_finalized = False

                print(
                    "[ZooAI] New monitoring session started: "
                    f"{session_dir}"
                )
                continue

            current_behavior = "No Detections"
            detected_behaviors = set()

            # =====================================
            # INFERENCE / DISPLAY FLAGS
            # =====================================
            current_time_sec = time.time()

            if input_video_path:
                # Accelerated recorded-video processing: infer on
                # every source frame, but cap GUI refresh near the
                # source FPS.
                run_inference = True
                run_display = (
                    current_time_sec - last_display_time
                    >= RECORDED_DISPLAY_INTERVAL
                )
                if run_display:
                    last_display_time = current_time_sec
            else:
                run_inference = (
                    current_time_sec - last_inference_time
                    >= LIVE_INFERENCE_INTERVAL
                )
                run_display = run_inference

                if run_inference:
                    last_inference_time = current_time_sec

            # =====================================
            # CAMERA LOOP
            # =====================================
            frames = []
            loop_swim_event = None
            loop_swim_state = loop_swim_counter.get_state()

            loop_camera_detected_behaviors = set()
            loop_camera_centroid = None
            loop_camera_best_confidence = -1.0

            any_frame_read = False

            for i, cap in enumerate(caps):
                ret, frame = cap.read()

                if not ret:
                    if input_video_path:
                        raise StopIteration

                    frame = np.zeros(
                        (480, 640, 3),
                        dtype=np.uint8
                    )
                else:
                    any_frame_read = True

                camera_results = []
                inference_ran_this_frame = bool(
                    ret and run_inference
                )

                if inference_ran_this_frame:
                    camera_results = model(
                        frame,
                        conf=0.69
                    )

                # Process results in BOTH recorded-video and
                # livestream modes.
                for result in camera_results:
                    for detection in result.boxes:
                        class_id = int(
                            detection.cls[0].item()
                        )
                        canonical_label = model.names[
                            class_id
                        ]
                        confidence = float(
                            detection.conf[0].item()
                        )

                        if (
                            canonical_label
                            in behaviors_of_interest
                        ):
                            detected_behaviors.add(
                                canonical_label
                            )

                            if i == loop_swim_camera_index:
                                loop_camera_detected_behaviors.add(
                                    canonical_label
                                )

                        xyxy = (
                            detection.xyxy[0]
                            .cpu()
                            .numpy()
                            .astype(int)
                        )
                        x1, y1, x2, y2 = xyxy
                        cx = int((x1 + x2) // 2)
                        cy = int((y1 + y2) // 2)

                        if (
                            i == loop_swim_camera_index
                            and confidence
                            > loop_camera_best_confidence
                        ):
                            loop_camera_best_confidence = confidence
                            loop_camera_centroid = (cx, cy)

                        cv2.rectangle(
                            frame,
                            (x1, y1),
                            (x2, y2),
                            (0, 255, 0),
                            2
                        )
                        cv2.putText(
                            frame,
                            canonical_label,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2
                        )

                # =================================
                # WHERE-ARE-POLAR-BEARS LOCATION TRACKER
                # =================================
                # Preserve the known-good locationtracker data path.
                # This call updates live/last-seen state independently
                # for every camera whenever YOLO inference runs.
                if inference_ran_this_frame:
                    locationtracker.update_from_yolo(
                        camera_index=i,
                        frame=frame,
                        results=camera_results
                    )

                # Loop-swimming centroid/tracker/checkpoint logic
                # runs only for the camera ending in .76.
                if (
                    i == loop_swim_camera_index
                    and ret
                ):
                    loop_swim_state = (
                        loop_swim_counter.update(
                            frame=frame,
                            results=camera_results,
                            model_names=model.names,
                            timestamp=timestamp,
                            run_inference=(
                                inference_ran_this_frame
                            )
                        )
                    )
                    loop_swim_event = (
                        loop_swim_state["event"]
                    )

                    if (
                        SHOW_LOOPSWIM_CRITERIA
                        and loop_swim_state["centroid"]
                        is not None
                    ):
                        frame = draw_loop_swim_criteria(
                            frame,
                            loop_swim_state["centroid"],
                            loop_swim_state["checkpoints"]
                        )

                display_camera_frame = cv2.resize(
                    frame,
                    (640, 480)
                )
                frames.append(display_camera_frame)

            if any_frame_read:
                last_processed_timestamp = timestamp

            # =====================================
            # DETECTION BUFFERING
            # =====================================
            if run_inference:
                last_detected_behaviors = (
                    detected_behaviors.copy()
                )
            else:
                detected_behaviors = (
                    last_detected_behaviors.copy()
                )

            laps = float(loop_swim_state["laps"])

            # =====================================
            # MOTION CORRECTION FOR .76 LOCOMOTION
            # =====================================
            if run_inference and loop_camera_centroid is not None:
                if (
                    last_motion_centroid is not None
                    and "LOCOMOTION"
                    in loop_camera_detected_behaviors
                    and frames
                ):
                    frame_h, frame_w = frames[
                        loop_swim_camera_index
                    ].shape[:2]

                    dx = abs(
                        loop_camera_centroid[0]
                        - last_motion_centroid[0]
                    )
                    dy = abs(
                        loop_camera_centroid[1]
                        - last_motion_centroid[1]
                    )

                    norm_dx = dx / max(frame_w, 1)
                    norm_dy = dy / max(frame_h, 1)

                    if norm_dx < 0.01 and norm_dy < 0.01:
                        detected_behaviors.discard(
                            "LOCOMOTION"
                        )
                        loop_camera_detected_behaviors.discard(
                            "LOCOMOTION"
                        )

                        if (
                            "OBJECT MANIPULATION"
                            not in detected_behaviors
                        ):
                            detected_behaviors.add(
                                "RESTING"
                            )

                last_motion_centroid = loop_camera_centroid
                last_bear_centroid = loop_camera_centroid

            # =====================================
            # SESSION BEHAVIOUR
            # =====================================
            behavior_priority = [
                "HEAD SWINGING",
                "PACING",
                "SWIMMING",
                "FORAGING",
                "LOCOMOTION",
                "RESTING"
            ]

            current_behavior = next(
                (
                    behavior
                    for behavior in behavior_priority
                    if behavior in detected_behaviors
                ),
                "No Detections"
            )

            if current_behavior != previous_behavior:
                # Replace the initial zero-duration entry when the
                # first frame has the same timestamp as start_time.
                if (
                    behavior_log
                    and timestamp
                    <= behavior_log[-1][0]
                ):
                    behavior_log[-1] = (
                        timestamp,
                        current_behavior
                    )
                else:
                    behavior_log.append(
                        (timestamp, current_behavior)
                    )

                previous_behavior = current_behavior

            # Add one head-swing sample only when YOLO ran.
            if run_inference:
                head_swing_window.append(
                    "HEAD SWINGING"
                    in detected_behaviors
                )

                if len(head_swing_window) == window_size:
                    percent_detected = (
                        sum(head_swing_window)
                        / len(head_swing_window)
                    )

                    if (
                        percent_detected
                        >= min_percent_detection
                        and not alert_sent
                    ):
                        send_stereotypy_alert()
                        alert_sent = True

            # ==================================
            # LOOP-SWIMMING EVENTS FROM .76
            # ==================================
            if loop_swim_event is not None:
                event_type = loop_swim_event["type"]

                if event_type == "checkpoint_created":
                    checkpoint_number = (
                        loop_swim_event[
                            "checkpoint_number"
                        ]
                    )
                    checkpoint_centroid = (
                        loop_swim_event["centroid"]
                    )

                    if checkpoint_number == 1:
                        print(
                            "[LoopSwim] Checkpoint 1 "
                            "established at "
                            f"{checkpoint_centroid}."
                        )
                    else:
                        checkpoint_distance = (
                            loop_swim_event["distance"]
                        )
                        print(
                            "[LoopSwim] Checkpoint 2 "
                            "established at "
                            f"{checkpoint_centroid}; "
                            "distance from checkpoint 1 "
                            f"= {checkpoint_distance:.1f}px."
                        )

                elif event_type == "checkpoint_entered":
                    print(
                        "[LoopSwim] Initial checkpoint "
                        "state established at checkpoint "
                        f"{loop_swim_event['checkpoint_number']}."
                    )

                elif event_type == "lap_increment":
                    laps = float(
                        loop_swim_event["laps"]
                    )

                    # Only log FULL laps (prevents half‑lap inflation)
                    if float(laps).is_integer():
                        lap_timestamps.append(timestamp)
                        lap_counts.append(laps)

                    previous_checkpoint = (
                        loop_swim_event[
                            "previous_checkpoint"
                        ]
                    )
                    current_checkpoint = (
                        loop_swim_event[
                            "current_checkpoint"
                        ]
                    )

                    print(
                        "[LoopSwim] Checkpoint "
                        f"{previous_checkpoint} → "
                        f"{current_checkpoint}; "
                        f"laps = {laps:.1f}"
                    )

                    complete_lap_reached = (
                        float(laps).is_integer()
                    )
                    if complete_lap_reached:
                        alert_window_open = (
                            last_pacing_alert_time is None
                            or (
                                cooldown_end_time is not None
                                and timestamp >= cooldown_end_time
                            )
                        )
                            #============================================
                            # 25‑MINUTE ROLLING WINDOW FOR LOOP‑SWIM ALERT
                            # ============================================
                        if alert_window_open:

                            cutoff = timestamp - timedelta(
                                minutes=LOOPSWIM_ALERT_WINDOW_MINUTES
                            )

                            # Trim BOTH timestamps and lap_counts together
                            filtered = [
                                (t, c)
                                for t, c in zip(
                                    lap_timestamps,
                                    lap_counts
                                )
                                if t >= cutoff
                            ]

                            lap_timestamps = [
                                t for t, c in filtered
                            ]
                            lap_counts = [
                                c for t, c in filtered
                            ]

                            laps_in_window = len(lap_timestamps)

                            if (
                                laps_in_window
                                >= LOOPSWIM_ALERT_LAP_THRESHOLD
                            ):
                                send_pacing_alert()

                                last_pacing_alert_time = timestamp
                                cooldown_end_time = (
                                    timestamp
                                    + timedelta(
                                        minutes=(
                                            pacing_alert_interval_minutes
                                        )
                                    )
                                )

                                # Reset window after alert
                                lap_timestamps = []
                                lap_counts = []
                                laps_since_cooldown = 0

                                save_loop_swim_alert_snapshot(
                                    frames=frames,
                                    session_dir=session_dir,
                                    timestamp=timestamp
                                )
            # ==================================
            # MODULAR PACING DETECTOR
            # ==================================
            if run_inference:
                pacing_alert = pacing_detector.update(
                    timestamp,
                    loop_camera_detected_behaviors,
                    loop_camera_centroid
                )

                if pacing_alert == "PACING_ALERT":
                    print("🚨 PACING ALERT TRIGGERED")
                    send_locomotion_pacing_alert()

            # ==================================
            # DISPLAY
            # ==================================
            if ENABLE_REALTIME_DISPLAY and run_display:
                now = datetime.now()
                if (
                    last_weather_update is None
                    or (
                        now - last_weather_update
                    ) > timedelta(minutes=10)
                ):
                    cached_temp = get_temperature()
                    last_weather_update = now

                ethogram_img = update_ethogram(
                    behavior_log
                )
                combined_feed = stack_frames_grid(frames)
                ethogram_padded = pad_image_to_match_height(
                    ethogram_img,
                    combined_feed.shape[0]
                )

                cv2.putText(
                    combined_feed,
                    f"Swimming Laps: {laps:.1f}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 144, 30),
                    2
                )
                cv2.putText(
                    combined_feed,
                    (
                        "Pacing Laps: "
                        f"{pacing_detector.total_laps}"
                    ),
                    (10, 65),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 255),
                    2
                )

                display_frame = np.hstack(
                    (ethogram_padded, combined_feed)
                )

                display_frame = draw_temp_widget(
                    display_frame,
                    cached_temp,
                    x=20,
                    y=20
                )
                display_frame = draw_logo_panel(
                    display_frame,
                    tz_logo,
                    x=250,
                    y=20
                )

                cv2.line(
                    display_frame,
                    (245, 20),
                    (245, 105),
                    (80, 80, 80),
                    1
                )

                if roam_logo is not None:
                    logo_x = 10
                    logo_y = (
                        display_frame.shape[0]
                        - roam_logo.shape[0]
                        - 10
                    )
                    display_frame = overlay_logo(
                        display_frame,
                        roam_logo,
                        logo_x,
                        logo_y
                    )

                if locationtracker.fullscreen_mode:
                    display_frame = (
                        locationtracker.draw_fullscreen_view(
                            display_frame
                        )
                    )
                    cv2.putText(
                        display_frame,
                        'Press "F" to exit Full Screen Mode',
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (255, 255, 255),
                        2
                    )

                elif locationtracker.show_location_overlay:
                    display_frame = (
                        locationtracker.draw_location_overlay(
                            display_frame
                        )
                    )
                    display_frame = (
                        locationtracker.draw_location_button(
                            display_frame
                        )
                    )
                    diagnostics.draw(display_frame)

                else:
                    display_frame = (
                        locationtracker.draw_location_button(
                            display_frame
                        )
                    )
                    diagnostics.draw(display_frame)

                cv2.imshow(
                    "Ethogram + Feed",
                    display_frame
                )
                cv2.setMouseCallback(
                    "Ethogram + Feed",
                    locationtracker.handle_mouse_click
                )
                key = cv2.waitKey(1) & 0xFF

                if key == ord("d"):
                    diagnostics.toggle()

                if key == ord("m"):
                    locationtracker.show_location_overlay = (
                        not locationtracker.show_location_overlay
                    )

                if key == ord("n"):
                    locationtracker.next_camera()

                if key in (ord("r"), ord("R")):
                    locationtracker.previous_camera()

                if key in (ord("f"), ord("F")):
                    locationtracker.fullscreen_mode = (
                        not locationtracker.fullscreen_mode
                    )

                if key == ord("c"):
                    SHOW_LOOPSWIM_CRITERIA = (
                        not SHOW_LOOPSWIM_CRITERIA
                    )

                if key == ord("q"):
                    break

            frame_count += 1

    except KeyboardInterrupt:
        print("Monitoring stopped by user.")

    except StopIteration:
        print("Reached end of available video frames.")

    finally:
        session_manager.stop()

        if not session_finalized:
            print(
                "[ZooAI] Finalizing the active session "
                "before shutdown."
            )

            try:
                finalize_monitoring_session(
                    session_dir=session_dir,
                    start_time=start_time,
                    behavior_log=behavior_log,
                    laps=laps,
                    lap_timestamps=lap_timestamps,
                    lap_counts=lap_counts,
                    input_video_path=input_video_path,
                    end_time=last_processed_timestamp
                )
            except Exception as save_error:
                print(
                    "❌ Error while finalizing session: "
                    f"{save_error}"
                )
                traceback.print_exc()
        else:
            print(
                "[ZooAI] Session was already finalized. "
                "Skipping duplicate save."
            )

        for cap in caps:
            cap.release()

        cv2.destroyAllWindows()


def stack_frames_grid(frames, tile_w=640, tile_h=480):
    """
    Stack frames into a 2x2 grid.
    Supports 1–4 frames.
    """
    black = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    # Normalize size
    frames = [cv2.resize(f, (tile_w, tile_h)) for f in frames]

    if len(frames) == 1:
        top = frames[0]
        bottom = black
    elif len(frames) == 2:
        top = np.hstack(frames)
        bottom = np.hstack([black, black])
    elif len(frames) == 3:
        top = np.hstack(frames[:2])
        bottom = np.hstack([frames[2], black])
    else:  # 4 or more
        top = np.hstack(frames[:2])
        bottom = np.hstack(frames[2:4])

    return np.vstack([top, bottom])
 
capture_and_infer(camera_urls) #comment on/off this line or next to choose whether want to conduct real-time inference or on a recorded video.
 
#capture_and_infer(input_video_path= "FILE NAME / FILE PATH TO VIDEO")
