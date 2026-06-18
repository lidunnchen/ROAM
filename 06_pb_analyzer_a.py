import subprocess
import sys
import os

# Get session_dir from command-line argument
if len(sys.argv) > 1:
    session_dir = sys.argv[1]
else:
    raise ValueError("Session directory path not provided as an argument.")

# Optional: define paths just to check things work
file_path = os.path.join(session_dir, "pacing_laps.xlsx")
activity_log_path = os.path.join(session_dir, "activity_log.csv")
output_dir = os.path.join(session_dir, "plot_laps")
os.makedirs(output_dir, exist_ok=True)

# Full paths to the scripts you want to run
script1 = r"C:\Users\lchen\Desktop\pb_analyzer\pb_analyzer_c.py"
script2 = r"C:\Users\lchen\Desktop\pb_analyzer\pb_analyzer_lapswim.py"

print("🔁 Running pb_analyzer_c.py...")
subprocess.run(["python", script1, session_dir], check=True)

print("🔁 Running pb_analyzer_lapswim.py...")
subprocess.run(["python", script2, session_dir], check=True)

print("✅ Both scripts finished.")
