# You must specify the video path in the "video_path" argument below.
## You must also specify the path for the trained model you wish to deploy by adjusting the "model_path" argument below.


import cv2
import csv
import datetime
import os
import re
from ultralytics import YOLO
from collections import Counter
import numpy as np
from statistics import mode

# =========================================
# USER INPUTS
# =========================================

model_path = r"C:\Users\lchen\Desktop\PB_STEREOTYPY\best_640_E_100epoch_NoAug.pt"
video_path = r"C:\Users\lchen\Desktop\PolarBear_30minClips_ConcordanceTests\Media player format\C23_Polar Bear Mat N_ (10.254.16.73) - Camera 1\11_14_2025 12_00_53 PM (UTC-05_00).mkv"

INTERVAL_SECONDS = 120
SAMPLE_FRAMES = 45
CONF_THRESHOLD = 0.55
DISPLAY_FRAMES = 10
IOU_THRESHOLD = 0.6
SNIPPET_SECONDS = 4

# Locomotion override settings
PIXEL_MOVEMENT_THRESHOLD = 70      # locomotion must exceed this amount; have to set to >75 for small head movements of resting bear situated close to camera
VISIBILITY_REQUIRED = 0.50         # bear must appear in >= 50% frames for movement-based override

# Colors for annotation
color_map = {
    "RESTING": (0, 215, 255),
    "LOCOMOTION": (0, 250, 154),
    "PACING": (71, 99, 255),
    "SWIMMING": (255, 144, 30),
    "HEAD SWINGING": (204, 102, 255),
    "FORAGING": (0, 140, 255),
    "No Detections": (200, 200, 200)
}


# =========================================
# HELPERS — IOU + suppression
# =========================================

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    if interArea <= 0:
        return 0.0

    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea)


def non_max_suppression_iou(dets, iou_threshold=0.70):
    if len(dets) == 0:
        return []
    dets = sorted(dets, key=lambda x: x[4], reverse=True)
    keep = []
    while dets:
        best = dets.pop(0)
        keep.append(best)
        dets = [d for d in dets if compute_iou(best[:4], d[:4]) < iou_threshold]
    return keep


# =========================================
# PARSE DATETIME FROM FILENAME
# =========================================

filename = os.path.basename(video_path)
pattern = r"(\d{1,2})_(\d{1,2})_(\d{4}) (\d{1,2})_(\d{1,2})_(\d{1,2}) (AM|PM)"
match = re.search(pattern, filename)
if not match:
    raise ValueError("Could not parse datetime from filename")

month, day, year, hour, minute, second, ampm = match.groups()
month, day, year = int(month), int(day), int(year)
hour, minute, second = int(hour), int(minute), int(second)

if ampm.upper() == "PM" and hour != 12:
    hour += 12
if ampm.upper() == "AM" and hour == 12:
    hour = 0

video_start = datetime.datetime(year, month, day, hour, minute, second)
clean_name = os.path.splitext(filename)[0]

output_csv = clean_name + "_2min_predictions.csv"
output_video_annot = clean_name + "_annotated_intervals.mp4"
output_snippets_video = clean_name + "_observer_snippets.mp4"


# =========================================
# LOAD MODEL & VIDEO
# =========================================

model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

out_vid_annot = cv2.VideoWriter(output_video_annot, fourcc, fps, (width, height))
out_snippets = cv2.VideoWriter(output_snippets_video, fourcc, fps, (width, height))


# =========================================
# MAJORITY VOTE + PER-BEAR LOCOMOTION OVERRIDE
# =========================================

def get_majority_predictions(start_frame):

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    collected = []
    count_list = []
    centroid_tracks = [[], [], []]
    visibility_counts = [0, 0, 0]

    # -------------------------
    # SAMPLE FRAMES
    # -------------------------
    for _ in range(SAMPLE_FRAMES):
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, verbose=False)[0]
        boxes = results.boxes

        detections = []
        for xyxy, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
            conf = float(conf)
            if conf >= CONF_THRESHOLD:
                x1, y1, x2, y2 = map(float, xyxy)
                detections.append((x1, y1, x2, y2, conf, model.names[int(cls)]))

        detections = non_max_suppression_iou(detections, IOU_THRESHOLD)
        detections.sort(key=lambda d: d[0])  # left-to-right

        slot_labels = [""] * 3

        for i in range(min(3, len(detections))):
            x1, y1, x2, y2, conf, label = detections[i]
            slot_labels[i] = label

            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            centroid_tracks[i].append((cx, cy))
            visibility_counts[i] += 1

        collected.append(slot_labels)

    # -------------------------------------------
    # MAJORITY VOTE PER BEAR SLOT
    # -------------------------------------------

    final_behaviours = []
    for slot in range(3):
        preds = [labels[slot] for labels in collected if labels[slot] != ""]
        if len(preds) == 0:
            final_behaviours.append("")  # placeholder
        else:
            final_behaviours.append(Counter(preds).most_common(1)[0][0])

    # -------------------------------------------
    # LOCOMOTION OVERRIDE PER BEAR
    # -------------------------------------------

    for slot in range(3):
        behaviour = final_behaviours[slot]

        if behaviour != "LOCOMOTION":
            continue

        # must appear >= 50% of sampled frames
        if visibility_counts[slot] < SAMPLE_FRAMES * VISIBILITY_REQUIRED:
            final_behaviours[slot] = "RESTING"
            continue

        centroids = centroid_tracks[slot]
        if len(centroids) <= 1:
            final_behaviours[slot] = "RESTING"
            continue

        # compute movement
        movement_distance = 0
        for i in range(1, len(centroids)):
            x1, y1 = centroids[i - 1]
            x2, y2 = centroids[i]
            movement_distance += ((x2 - x1)**2 + (y2 - y1)**2)**0.5

        if movement_distance < PIXEL_MOVEMENT_THRESHOLD:
            final_behaviours[slot] = "RESTING"

    # ---------------------------------------------------------
    # FINAL "NO DETECTIONS" CLEANUP
    # ---------------------------------------------------------

    if all(b == "" for b in final_behaviours):
        return ["No Detections", "No Detections", "No Detections"]

    for i in range(3):
        if final_behaviours[i] == "":
            final_behaviours[i] = "No Detections"

    return final_behaviours


# =========================================
# ANNOTATE FRAME WITH CORRECTED BEHAVIOURS
# =========================================

def get_annotated_frame(frame_number, timestamp_str, minute_number, behaviours):

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = cap.read()
    if not ret:
        return None

    label = f"{timestamp_str}  |  Minute {minute_number}"
    cv2.putText(frame, label, (40, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                (255, 255, 255), 3)

    results = model(frame, verbose=False)[0]
    boxes = results.boxes

    dets = []
    raw_labels = []
    for xyxy, cls, conf in zip(boxes.xyxy, boxes.cls, boxes.conf):
        conf = float(conf)
        if conf >= CONF_THRESHOLD:
            x1, y1, x2, y2 = map(int, xyxy)
            dets.append((x1, y1, x2, y2, conf))
            raw_labels.append(model.names[int(cls)])

    if len(dets) == 0:
        cv2.putText(frame, "No Detections", (40, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    color_map["No Detections"], 3)
        return frame

    merged = [(d[0], d[1], d[2], d[3], d[4], raw_labels[i])
              for i, d in enumerate(dets)]

    merged = non_max_suppression_iou(merged, IOU_THRESHOLD)
    merged.sort(key=lambda d: d[0])

    for slot in range(min(3, len(merged))):
        x1, y1, x2, y2, conf, _ = merged[slot]
        beh = behaviours[slot]
        color = color_map.get(beh, (255, 255, 255))

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        cv2.putText(frame, beh, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    return frame


# =========================================
# SNIPPET VIDEO
# =========================================

def save_snippet(start_frame, timestamp_str, minute_number):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    total_snip_frames = int(SNIPPET_SECONDS * fps)
    label = f"{timestamp_str}  |  Minute {minute_number}"

    for _ in range(total_snip_frames):
        ret, frame = cap.read()
        if not ret:
            break

        cv2.putText(frame, label, (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2,
                    (255, 255, 255), 3)

        out_snippets.write(frame)


# =========================================
# MAIN LOOP
# =========================================

rows = []
current_frame = 0
current_datetime = video_start
elapsed_minutes = 0

while current_frame < total_frames:

    behaviours = get_majority_predictions(current_frame)
    bear1, bear2, bear3 = behaviours
    timestamp_str = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

    rows.append({
        "datetime": timestamp_str,
        "elapsed_minutes": elapsed_minutes,
        "bear1_behaviour": bear1,
        "bear1_behaviour_observer": "",
        "bear2_behaviour": bear2,
        "bear2_behaviour_observer": "",
        "bear3_behaviour": bear3,
        "bear3_behaviour_observer": ""
    })

    snapshot = get_annotated_frame(
        current_frame,
        timestamp_str,
        elapsed_minutes,
        behaviours
    )

    if snapshot is not None:
        for _ in range(DISPLAY_FRAMES):
            out_vid_annot.write(snapshot)

    save_snippet(current_frame, timestamp_str, elapsed_minutes)

    elapsed_minutes += 2
    current_datetime += datetime.timedelta(seconds=INTERVAL_SECONDS)
    current_frame += int(INTERVAL_SECONDS * fps)


cap.release()
out_vid_annot.release()
out_snippets.release()


# =========================================
# CSV OUTPUTS
# =========================================

with open(output_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "datetime", "elapsed_minutes",
        "bear1_behaviour", "bear1_behaviour_observer",
        "bear2_behaviour", "bear2_behaviour_observer",
        "bear3_behaviour", "bear3_behaviour_observer"
    ])
    writer.writeheader()
    writer.writerows(rows)

print("CSV saved:", output_csv)

manual_csv = clean_name + "_2min_predictions_manual_observer.csv"

manual_rows = [{
    "datetime": r["datetime"],
    "elapsed_minutes": r["elapsed_minutes"],
    "bear1_behaviour_observer": "",
    "bear2_behaviour_observer": "",
    "bear3_behaviour_observer": ""
} for r in rows]

with open(manual_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "datetime", "elapsed_minutes",
        "bear1_behaviour_observer",
        "bear2_behaviour_observer",
        "bear3_behaviour_observer"
    ])
    writer.writeheader()
    writer.writerows(manual_rows)

print("Manual observer CSV saved:", manual_csv)
print("Annotated MP4 saved:", output_video_annot)
print("Snippet MP4 saved:", output_snippets_video)
print("\nDone!")
