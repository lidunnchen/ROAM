"""
StereotypyAI Behavioral Monitoring Framework

Implements four modular layers:
1. Detection Layer (Moel load + YOLO inference)
2. Alerting Layer (Thresholds + email functions + alert calls)
3. Behaviour Inference Layer (otion correction, head swing window, lap counting, cooldowns, PacingDetector logic)
4. Post-Processing Layer (All exports, summaries, CSV, plots, email summary)

To adapt this framework 🛠:
    1. Replace detection model weights (i.e., the "model = " statement)
    2. Update behavior classes (i.e., "ALL_CLASSES" and "behaviors_of_interest")
    3. Modify inference rules

Core pipeline logic should not require modification.
"""

### Within Terminal, Anaconda Prompt (the working environment, first activate your working evironemnt with command "conda activate [NAME OF ENVIRONMENT]" and set your file directory to where you have placed the BEHAVR_ANALYZER.py script using the command "cd [FILE LOCATION]"

### PYTHON PACKAGE IMPORTS
import subprocess
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
from ultralytics import YOLO
from dotenv import load_dotenv
from datetime import datetime
import pandas as pd
import smtplib
import csv
from email.message import EmailMessage
import ssl
import time
from collections import defaultdict
from collections import deque
from datetime import timedelta
from datetime import datetime
from PacingDetector import PacingDetector
import requests
last_weather_update = None
cached_temp = "..."

### CONFIGURATION behavioural triggers for real-time alerts:  
HEADSWING_Trigger_mins = 10 # ✅ Set # minutes of headswinging required to trigger alert
pacing_alert_interval_minutes = 60  # ✅ Set alert cooldown to 60 minutes
MIN_NEW_ZONE_DISTANCE = 30 # sets the minimum pixel distance to register a new pacing zone; prevents registering a new pacing zone unless it is far enough 
MAX_NEW_ZONE_DISTANCE =  90 # was 240; sets the maximum pixel distance to allow for a pacing zone; blocks faraway movements from being interpreted as pacing

### Define email alerts triggered by various behaviours
def send_locomotion_pacing_alert():
    msg = EmailMessage()
    msg['Subject'] = "STEREOTYPY ALERT - POLAR BEAR PACING"
    msg['From'] = email_sender
    msg['To'] = email_receiver
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
    msg['To'] = email_receiver
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
    msg['To'] = email_receiver
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

### 🛠 Input Camera IP Address & Login Credentials; select model weights for Object Detection Model, and input names of behaviours based off annotations/.yaml file (they need to match)
   
ENABLE_REALTIME_DISPLAY = True
matplotlib.use('Agg')
 
load_dotenv()
username = os.getenv("CAMERA_USERNAME") #login credentials for accessing RTSP cameras should be saved in a private .env file
password = os.getenv("CAMERA_PASSWORD") 
camera_ip_pb5 = os.getenv("CAMERA_IP_PB5", "10.254.16.73") #replace camera IP address to match your institution's livestreams
camera_ip_pb2 = os.getenv("CAMERA_IP_PB2", "10.254.16.76")
camera_ip_pb4 = os.getenv("CAMERA_IP_PB4", "10.254.16.186")
email_sender = os.getenv("EMAIL_SENDER")
email_password = os.getenv("EMAIL_PASSWORD")
email_receiver = "lchen@torontozoo.ca"
 
camera_urls = [
    f"rtsp://{username}:{password}@{camera_ip_pb5}/axis-media/media.amp",
    f"rtsp://{username}:{password}@{camera_ip_pb2}/axis-media/media.amp",
    f"rtsp://{username}:{password}@{camera_ip_pb4}/axis-media/media.amp"
]
 
model = YOLO(r"C:\Users\lchen\Desktop\PB_STEREOTYPY\best_640_E_100epoch_NoAug.pt")
 
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
 
output_base_dir = r"C:\\Users\\lchen\\Desktop\\PB_STEREOTYPY"

# =========================
# LOAD ROAM LOGO 
# =========================
logo_path = r"C:\Users\lchen\Desktop\PB_STEREOTYPY\ROAM_logo.png"
roam_logo = cv2.imread(logo_path, cv2.IMREAD_UNCHANGED)

if roam_logo is None:
    print("❌ ERROR: ROAM logo failed to load. Check file path.")
else:
    # Resize logo (recommended)
    logo_width = 350
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
tz_logo_path = r"C:\Users\lchen\Desktop\PB_STEREOTYPY\TorontoZoo_logo.png"
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

# =========================
# OVERLAY LOGO - for ROAM & Toronto Zoo Logo
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

### POST-PROCESSING LAYER 
### Pipeline to export behavioural data to excel spreadsheets and generate plots 
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
        ax.set_title("Polar Bear Dynamic Activity Budget", fontsize=14)
        ax.text(0.5, 0.5, "Monitoring...", ha='center', va='center', fontsize=16)
        ax.axis('off')

        canvas = FigureCanvas(fig)
        canvas.draw()
        w, h = fig.canvas.get_width_height()
        rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
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
    rgb = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
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


def save_and_email_summary(start_time, behavior_log, laps):
    end_time = behavior_log[-1][0]
    total_seconds = (end_time - start_time).total_seconds()

    durations = [(behavior_log[i + 1][0] - behavior_log[i][0]).total_seconds()
                 for i in range(len(behavior_log) - 1)]

    class_times = defaultdict(float)
    for i, (_, behavior) in enumerate(behavior_log[:-1]):
        class_times[behavior] += durations[i]

    # Calculate loop swimming laps and head swinging time
    loop_swimming_laps = laps
    head_swinging_time = sum(
        durations[i] for i, (_, behavior) in enumerate(behavior_log[:-1]) if behavior == "HEAD SWINGING"
    ) / 60  # in minutes

    # Create activity DataFrame
    dates = [t.strftime("%Y-%m-%d") for t, _ in behavior_log[:-1]]
    timestamps = [t.strftime("%H:%M:%S") for t, _ in behavior_log[:-1]]
    behaviors = [b for _, b in behavior_log[:-1]]
    df = pd.DataFrame({
        "Date": dates,
        "Timestamp": timestamps,
        "Behavior": behaviors,
        "Duration (seconds)": durations
    })

    # Setup output paths
    session_dir = os.path.join(output_base_dir, f"session_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(session_dir, exist_ok=True)

    csv_path = os.path.join(session_dir, "activity_log.csv")
    df.to_csv(csv_path, index=False)

    budget_csv_path = os.path.join(session_dir, "activity_budget.csv")
    save_activity_budget_csv(class_times, total_seconds, budget_csv_path, start_time, end_time)

    ethogram_path = os.path.join(session_dir, "ethogram.png")
    final_ethogram = update_ethogram(behavior_log)
    cv2.imwrite(ethogram_path, final_ethogram)

    barplot_path = os.path.join(session_dir, "activity_budget_barplot.png")
    plot_activity_budget(class_times, total_seconds, barplot_path)

    # === Run the post-processing script before sending email ===
    combine_script = r"C:\Users\lchen\Desktop\PB_STEREOTYPY\pb_analyzer_combine.py"
    try:
        print("🧩 Running pb_analyzer_combine.py for post-processing...")
        subprocess.run(["python", combine_script, session_dir], check=True)
        print("✅ Post-processing complete.")
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to run post-processing script: {e}")

    # Add new file paths from plot_laps
    plot_laps_dir = os.path.join(session_dir, "plot_laps")
    behavior_clock_day_path = os.path.join(plot_laps_dir, "behaviour_clock_daytime.mp4")
    behavior_clock_night_path = os.path.join(plot_laps_dir, "behaviour_clock_nighttime.mp4")
    laps_plot_path = os.path.join(plot_laps_dir, "pacing_laps_per_hour.png")

    # Create the summary lines
    summary_lines = [
        f"• {loop_swimming_laps} loop swimming laps were detected throughout the monitoring session.",
        f"• {int(round(head_swinging_time))} minutes of head swinging were detected throughout the monitoring session."
    ]
    summary_text = "\n".join(summary_lines)

    # Create and send the email
    msg = EmailMessage()
    msg['Subject'] = "Polar Bear Behaviour Monitoring Summary"
    msg['From'] = email_sender
    msg['To'] = email_receiver
    msg.set_content(
        "Hello Wildlife Care Team,\n\n"
        "Please see attached the activity summary files for the monitoring session.\n\n"
        "Stereotypical Summary:\n"
        f"{summary_text}\n\n"
        "- - -\n"
        "Toronto Zoo Welfare Science"
    )

    # Attach files only if they exist
    for file_path, mime_type, subtype in [
        (csv_path, 'text', 'csv'),
        (ethogram_path, 'image', 'png'),
        (barplot_path, 'image', 'png'),
        (budget_csv_path, 'text', 'csv'),
        (behavior_clock_day_path, 'video', 'mp4'),
        (behavior_clock_night_path, 'video', 'mp4'),
        (laps_plot_path, 'image', 'png')
    ]:
        if os.path.isfile(file_path):
            with open(file_path, 'rb') as f:
                msg.add_attachment(
                    f.read(),
                    maintype=mime_type,
                    subtype=subtype,
                    filename=os.path.basename(file_path)
                )
        else:
            print(f"⚠️ Warning: File not found and will not be attached: {file_path}")

    # Send the email
    context = ssl.create_default_context()
    with smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context) as server:
        server.login(email_sender, email_password)
        server.send_message(msg)
 
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

# ====================================================== 
# BEHAVIOUR DETECTION & INFERENCE LAYER 
# ======================================================
def capture_and_infer(rtsp_urls=None, input_video_path=None):

    global last_weather_update, cached_temp, roam_logo, tz_logo

    if input_video_path:
        caps = [cv2.VideoCapture(input_video_path)]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_fps = 30
        frame_width = int(caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
    else:
        caps = [cv2.VideoCapture(url, cv2.CAP_FFMPEG) for url in rtsp_urls]

    for i, cap in enumerate(caps):
        if not cap.isOpened():
            print(f"Error: Cannot open video stream for Camera {i+1 if not input_video_path else 'Input Video'}")

    for cap in caps:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    frame_count = 0
    behavior_log = []

    # ==============================
    # TRUE 1 Hz INFERENCE CONTROL
    # ==============================
    INFERENCE_INTERVAL = 1.0  # seconds
    DISPLAY_INTERVAL = 1.0    # seconds

    last_inference_time = 0
    last_display_time = 0

    last_detected_behaviors = set()
    last_bear_centroid = None

    window_duration_minutes = HEADSWING_Trigger_mins
    min_percent_detection = 0.6 #adjust to make trigger more/less stringent
    window_size = int(window_duration_minutes * 60)
    head_swing_window = deque(maxlen=window_size)
    alert_sent = False

    last_non_swim_frame = 0
    swim_active = False
    pacing_checkpoints = []
    last_checkpoint = None
    laps = 0
    lap_timestamps = []
    lap_counts = []
    laps_since_cooldown = 0
    pacing_timeout_start = None
    last_pacing_alert_time = None
    cooldown_end_time = None

    # ==================================
    # INITIALIZE PACING DETECTOR MODULE
    # ==================================
    pacing_detector = PacingDetector()
    last_motion_centroid = None

    start_time = datetime.now()
    session_dir = os.path.join(output_base_dir, f"session_{start_time.strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(session_dir, exist_ok=True)

    if input_video_path:
        annotated_video_path = os.path.join(session_dir, "annotated_output.mkv")
        out_writer = cv2.VideoWriter(annotated_video_path, fourcc, output_fps, (frame_width, frame_height))

    current_behavior = "No Detections"
    previous_behavior = current_behavior
    behavior_log.append((start_time, current_behavior))

    print("Monitoring started. Press Ctrl+C to stop.")

    try:
        while True:
            timestamp = datetime.now()
            current_behavior = "No Detections"
            detected_behaviors = set()
            bear_centroid = None
            frames = []

            # =====================================
            # 1 Hz TIMING CONTROL (MOVED OUTSIDE CAMERA LOOP)
            # =====================================
            current_time_sec = time.time()
            run_inference = False
            run_display = False

            if current_time_sec - last_inference_time >= INFERENCE_INTERVAL:
                run_inference = True
                run_display = True
                last_inference_time = current_time_sec

            # =====================================
            # CAMERA LOOP
            # =====================================
            for cap in caps:
                ret, frame = cap.read()
                if not ret:
                    if input_video_path:
                        raise StopIteration
                    frame = np.zeros((480, 640, 3), dtype=np.uint8)

                if run_inference:

                    results = model(frame, conf=0.69)
                    for result in results:
                        for detection in result.boxes:
                            cls = detection.cls[0].item()
                            label_class = model.names[int(cls)]

                            if label_class in behaviors_of_interest:
                                detected_behaviors.add(label_class)

                            xyxy = detection.xyxy[0].cpu().numpy().astype(int)
                            x1, y1, x2, y2 = xyxy
                            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                            if label_class in ["SWIMMING", "LOCOMOTION"]:
                                bear_centroid = (cx, cy)

                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, label_class, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 3)
                            cv2.putText(frame, label_class, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 1)

                else:
                    pass  # do not overwrite detected_behaviors here

                frames.append(cv2.resize(frame, (640, 480)))

            # =====================================
            # MOTION CORRECTION FOR LOCOMOTION, lap counting, rolling windows
	    # Converts detections --> higher-order behavioural states
            # =====================================

            if bear_centroid is not None:

                if last_motion_centroid is not None and "LOCOMOTION" in detected_behaviors:

                    frame_h, frame_w = frames[0].shape[:2]

                    dx = abs(bear_centroid[0] - last_motion_centroid[0])
                    dy = abs(bear_centroid[1] - last_motion_centroid[1])

                    norm_dx = dx / frame_w
                    norm_dy = dy / frame_h

                    # If movement less than 2% of frame in both axes
                    if norm_dx < 0.02 and norm_dy < 0.02:

                        detected_behaviors.discard("LOCOMOTION")

                        if "FORAGING" in detected_behaviors:
                            detected_behaviors.add("FORAGING")
                        else:
                            detected_behaviors.add("RESTING")

                # Update centroid for next iteration
                last_motion_centroid = bear_centroid

            if detected_behaviors:
                current_behavior = list(detected_behaviors)[0]

# ======================================================
# ALERTING LAYER (Real-time evaluation)
# Threshold checks + cooldown enforcement; email messages for triggered alerts customizable at top of script; set thresholds at top of script!
# ======================================================
            head_swing_window.append("HEAD SWINGING" in detected_behaviors)
            if len(head_swing_window) == window_size:
                percent_detected = sum(head_swing_window) / len(head_swing_window)
                if percent_detected >= min_percent_detection and not alert_sent:
                    send_stereotypy_alert()
                    alert_sent = True

            if current_behavior != previous_behavior:
                behavior_log.append((timestamp, current_behavior))
                previous_behavior = current_behavior

            if current_behavior == "SWIMMING" and bear_centroid:
                swim_active = True
                pacing_timeout_start = None

                if not pacing_checkpoints:
                    pacing_checkpoints = [bear_centroid]
                elif len(pacing_checkpoints) == 1:
                    distance_to_first = np.linalg.norm(np.array(bear_centroid) - np.array(pacing_checkpoints[0]))
                    if MIN_NEW_ZONE_DISTANCE < distance_to_first < MAX_NEW_ZONE_DISTANCE: # > 90: #was > 100; vvv 2 detect new loopswim zone; only accept new zone if pix distance is beteween 90-250 px apart
                        pacing_checkpoints.append(bear_centroid)
                else:
                    for i, cp in enumerate(pacing_checkpoints):
                        if np.linalg.norm(np.array(bear_centroid) - np.array(cp)) < 40: #was 50; sets deviance from returning location individual can be
                            if last_checkpoint is None:
                                lap_timestamps.append(timestamp)
                                lap_counts.append(0.0)
                                last_checkpoint = i
                            elif last_checkpoint != i:
                                laps += 0.5
                                lap_timestamps.append(timestamp)
                                lap_counts.append(laps)
                                last_checkpoint = i

                                # if no alert has ever been sent (this will be first ever alert)
                                if laps % 1 == 0:
                                    if not last_pacing_alert_time:
                                        laps_since_cooldown += 1
                                        if laps_since_cooldown >= 20:
                                            send_pacing_alert()
                                            last_pacing_alert_time = timestamp
                                            cooldown_end_time = last_pacing_alert_time + timedelta(
                                                minutes=pacing_alert_interval_minutes)
                                            laps_since_cooldown = 0

                                            # Save snapshot image from both cameras
                                            snapshot_dir = os.path.join(session_dir, "plot_laps")
                                            os.makedirs(snapshot_dir, exist_ok=True)
                                            snapshot_filename = f"loopswim_alert_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                                            snapshot_path = os.path.join(snapshot_dir, snapshot_filename)

                                            if len(frames) >= 1:
                                                combined_snapshot = np.hstack(frames)
                                            else:
                                                combined_snapshot = None

                                            if combined_snapshot is not None:
                                                cv2.imwrite(snapshot_path, combined_snapshot)
                                                print(f"📸 Snapshot saved for loop swim alert: {snapshot_path}")
                                            else:
                                                print("⚠️ No frames available to save snapshot.")

                                    # IF we're outside the cooldown period (cooldown expired)
                                    elif timestamp >= cooldown_end_time:
                                        laps_since_cooldown += 1
                                        if laps_since_cooldown >= 20:
                                            send_pacing_alert()
                                            last_pacing_alert_time = timestamp
                                            cooldown_end_time = last_pacing_alert_time + timedelta(
                                                minutes=pacing_alert_interval_minutes)
                                            laps_since_cooldown = 0

                                            # Save snapshot image from both cameras
                                            snapshot_dir = os.path.join(session_dir, "plot_laps")
                                            os.makedirs(snapshot_dir, exist_ok=True)
                                            snapshot_filename = f"loopswim_alert_{timestamp.strftime('%Y-%m-%d_%H-%M-%S')}.jpg"
                                            snapshot_path = os.path.join(snapshot_dir, snapshot_filename)

                                            if len(frames) == 2:
                                                combined_snapshot = np.hstack(frames)
                                            elif len(frames) == 1:
                                                combined_snapshot = frames[0]
                                            else:
                                                combined_snapshot = None

                                            if combined_snapshot is not None:
                                                cv2.imwrite(snapshot_path, combined_snapshot)
                                                print(f"📸 Snapshot saved for loop swim alert: {snapshot_path}")
                                            else:
                                                print("⚠️ No frames available to save snapshot.")
                                    else:
                                        pass  # Still in cooldown – ignore laps

            elif current_behavior != "SWIMMING" and current_behavior != "No Detections":
                last_non_swim_frame += 1
                if last_non_swim_frame > 30:
                    swim_active = False
                    pacing_checkpoints.clear()
                    last_checkpoint = None

            elif current_behavior == "No Detections":
                if not pacing_timeout_start:
                    pacing_timeout_start = datetime.now()
                elif (datetime.now() - pacing_timeout_start).total_seconds() > 10:
                    swim_active = False
                    pacing_checkpoints.clear()
                    last_checkpoint = None

            # ==================================
            # MODULAR PACING DETECTOR (LOCOMOTION)
            # ==================================

            pacing_alert = pacing_detector.update(
                timestamp,
                detected_behaviors,
                bear_centroid
            )

            if pacing_alert == "PACING_ALERT":
                print("🚨 PACING ALERT TRIGGERED")
                send_locomotion_pacing_alert()

            if ENABLE_REALTIME_DISPLAY and run_display:

                # =========================================
                # WEATHER UPDATE (every 10 minutes)
                # =========================================
                now = datetime.now()
                if last_weather_update is None or (now - last_weather_update) > timedelta(minutes=10):
                    cached_temp = get_temperature()
                    last_weather_update = now

                ethogram_img = update_ethogram(behavior_log)
                combined_feed = stack_frames_grid(frames)
                ethogram_padded = pad_image_to_match_height(ethogram_img, combined_feed.shape[0])

                cv2.putText(combined_feed, f"Swimming Laps: {int(laps)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 144, 30), 2)

                cv2.putText(combined_feed, f"Pacing Laps: {pacing_detector.total_laps}", (10, 70),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                # =========================================
                # COMBINE PANELS
                # =========================================
                display_frame = np.hstack((ethogram_padded, combined_feed))

                # =========================================
                # TEMPERATURE WIDGET (top-left)
                # =========================================
                display_frame = draw_temp_widget(
                    display_frame,
                    cached_temp,
                    x=20,
                    y=20
                )

                # =========================================
                # TORONTO ZOO LOGO PANEL (right of temp)
                # =========================================
                display_frame = draw_logo_panel(
                    display_frame,
                    tz_logo,
                    x=20 + 220 + 10,
                    y=20
                )

                # =========================================
                # OPTIONAL DIVIDER BETWEEN PANELS
                # =========================================
                cv2.line(display_frame,
                         (20 + 220 + 5, 20),
                         (20 + 220 + 5, 20 + 85),
                         (80, 80, 80), 1)

                # =========================================
                # ROAM LOGO (bottom-left, aligned with ethogram)
                # =========================================
                logo_x = 10
                logo_y = display_frame.shape[0] - roam_logo.shape[0] - 10
                display_frame = overlay_logo(display_frame, roam_logo, logo_x, logo_y)

                # =========================================
                # DISPLAY FINAL OUTPUT
                # =========================================
                cv2.imshow("Ethogram + Feed", display_frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if input_video_path:
                out_writer.write(frames[0])

            frame_count += 1

    except (KeyboardInterrupt, StopIteration):
        print("Monitoring stopped.")
    finally:
        end_time = datetime.now()
        if behavior_log[-1][0] != end_time:
            behavior_log.append((end_time, behavior_log[-1][1]))

        plot_path = os.path.join(session_dir, "pacing_laps.png")
        plot_pacing_laps(lap_timestamps, lap_counts, plot_path)
        export_pacing_laps_to_excel(lap_timestamps, lap_counts, session_dir, start_time)
        save_and_email_summary(start_time, behavior_log, laps)

        for cap in caps:
            cap.release()
        if input_video_path:
            out_writer.release()
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
 
#capture_and_infer(input_video_path=r"C:\Users\lchen\Desktop\PB_STEREOTYPY\VIDS4Testing\loop swim\loopTest1_6_6_2025 7_38_21 AM (UTC-04_00).mkv")