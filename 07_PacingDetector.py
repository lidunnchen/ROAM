import numpy as np
from datetime import timedelta
from collections import deque


class PacingDetector:
    def __init__(self,
                 min_zone_dist=10,     # for South mat camera
                 max_zone_dist=500, #was 200 
                 return_radius=75, #was 40
                 lap_threshold=20,
                 time_window_minutes=20,
                 cooldown_minutes=60):

        self.min_zone_dist = min_zone_dist
        self.max_zone_dist = max_zone_dist
        self.return_radius = return_radius
        self.lap_threshold = lap_threshold
        self.time_window = timedelta(minutes=time_window_minutes)
        self.cooldown_minutes = cooldown_minutes

        self.checkpoints = []
        self.last_checkpoint = None

        # Store timestamps of FULL laps
        self.lap_timestamps = deque()

        self.total_laps = 0
        self._half_counter = 0

        self.last_alert_time = None
        self.cooldown_end_time = None

    def update(self, timestamp, detected_behaviors, centroid):

        if "LOCOMOTION" not in detected_behaviors or centroid is None:
            return None

        if not self.checkpoints:
            self.checkpoints = [centroid]
            return None

        if len(self.checkpoints) == 1:
            dist = np.linalg.norm(np.array(centroid) - np.array(self.checkpoints[0]))
            if self.min_zone_dist < dist < self.max_zone_dist:
                self.checkpoints.append(centroid)
            return None

        for i, cp in enumerate(self.checkpoints):
            if np.linalg.norm(np.array(centroid) - np.array(cp)) < self.return_radius:

                if self.last_checkpoint is None:
                    self.last_checkpoint = i

                elif self.last_checkpoint != i:
                    self.last_checkpoint = i

                    # HALF lap
                    # Only count FULL lap on integer change
                    self._register_half_lap(timestamp)

                    return self._check_alert(timestamp)

        return None

    def _register_half_lap(self, timestamp):

        # Add half lap logic
        # Count full lap when two halves occur

        if not hasattr(self, "_half_counter"):
            self._half_counter = 0

        self._half_counter += 0.5

        if self._half_counter % 1 == 0:
            # FULL LAP COMPLETED
            self.total_laps +=1
            self.lap_timestamps.append(timestamp)

    def _check_alert(self, timestamp):

        # Remove laps outside time window
        while self.lap_timestamps and \
              timestamp - self.lap_timestamps[0] > self.time_window:
            self.lap_timestamps.popleft()

        # Check cooldown first
        if self.last_alert_time and timestamp < self.cooldown_end_time:
            return None

        # Trigger only if threshold met within window
        if len(self.lap_timestamps) >= self.lap_threshold:

            self.last_alert_time = timestamp
            self.cooldown_end_time = timestamp + timedelta(
                minutes=self.cooldown_minutes)

            return "PACING_ALERT"

        return None
