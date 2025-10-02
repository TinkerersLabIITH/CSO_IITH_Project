# import cv2
# import numpy as np
# import math
# from collections import defaultdict, Counter
# from .base_counter import BaseCounter
# from ultralytics import YOLO
# from deep_sort_realtime.deepsort_tracker import DeepSort
# # --- Helper Functions for Line Intersection Logic ---
# def on_segment(p, q, r):
#     """Given three collinear points p, q, r, check if point q lies on line segment 'pr'."""
#     return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
#             q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

# def orientation(p, q, r):
#     """Find orientation of ordered triplet (p, q, r)."""
#     val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
#     if val == 0: return 0  # Collinear
#     return 1 if val > 0 else 2  # Clockwise or Counterclockwise

# def intersects(seg1_p1, seg1_p2, seg2_p1, seg2_p2):
#     """Return true if line segment 'seg1' and 'seg2' intersect."""
#     o1 = orientation(seg1_p1, seg1_p2, seg2_p1)
#     o2 = orientation(seg1_p1, seg1_p2, seg2_p2)
#     o3 = orientation(seg2_p1, seg2_p2, seg1_p1)
#     o4 = orientation(seg2_p1, seg2_p2, seg1_p2)

#     if o1 != o2 and o3 != o4:
#         return True
#     if o1 == 0 and on_segment(seg1_p1, seg2_p1, seg1_p2): return True
#     if o2 == 0 and on_segment(seg1_p1, seg2_p2, seg1_p2): return True
#     if o3 == 0 and on_segment(seg2_p1, seg1_p1, seg2_p2): return True
#     if o4 == 0 and on_segment(seg2_p1, seg1_p2, seg2_p2): return True
#     return False

# # --- Main Counter Class ---
# class PeopleCounter:
#     def __init__(self, config):
#         self.config = config
#         self.detector = YOLO(self.config.YOLO_WEIGHTS)
#         self.tracker = DeepSort(max_age=self.config.DEEPSORT_MAX_AGE)
        
#         self.counts = Counter()
#         self.counted_track_ids = set()
#         self.track_history = defaultdict(list)

#     def run(self, video_source):
#         cap = cv2.VideoCapture(video_source)
#         if not cap.isOpened():
#             print(f"Error opening video: {video_source}")
#             return

#         polyline_np = np.array(self.config.COUNTING_POLYLINE, dtype=np.int32)

#         while True:
#             ret, frame = cap.read()
#             if not ret: 
#                 print("End of video stream.")
#                 break

#             # 1. Detection and Tracking
#             results = self.detector(frame, conf=self.config.CONF_THRESH, verbose=False)
#             detections_for_tracker = []
#             for r in results:
#                 for box in r.boxes:
#                     cls_id, conf = int(box.cls[0]), float(box.conf[0])
#                     label = self.detector.names.get(cls_id)
#                     if label in self.config.TARGET_CLASSES:
#                         x1, y1, x2, y2 = map(int, box.xyxy[0])
#                         w, h = x2 - x1, y2 - y1
#                         detections_for_tracker.append([[x1, y1, w, h], conf, label])
            
#             tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)

#             # 2. Process Tracks and Check for Crossing
#             for track in tracks:
#                 if not track.is_confirmed(): continue

#                 tid = int(track.track_id)
#                 l, t, r, b = map(int, track.to_ltrb())
                
#                 current_point = ((l + r) // 2, b)
#                 self.track_history[tid].append(current_point)
                
#                 if len(self.track_history[tid]) > 1:
#                     previous_point = self.track_history[tid][-2]
                    
#                     for i in range(len(polyline_np) - 1):
#                         line_seg_p1 = tuple(polyline_np[i])
#                         line_seg_p2 = tuple(polyline_np[i+1])

#                         if intersects(previous_point, current_point, line_seg_p1, line_seg_p2):
#                             if tid not in self.counted_track_ids:
#                                 self.counts["person"] += 1
#                                 self.counted_track_ids.add(tid)
#                             break 

#                 # 3. Drawing Bounding Boxes and Track Info
#                 # Draw GREEN bounding box
#                 cv2.rectangle(frame, (l, t), (r, b), (0, 200, 0), 2)
#                 # Draw WHITE text
#                 cv2.putText(frame, f"person-{tid}", (l, t-10), self.config.FONT, 0.5, (255,255,255), 2)
#                 # Draw YELLOW tracking dot
#                 cv2.circle(frame, current_point, 4, (0, 255, 255), -1)

#             # 4. Draw the BLACK Counting Line and Counter Text
#             # This is the only line drawn for counting.
#             cv2.polylines(frame, [polyline_np], isClosed=False, color=self.config.LINE_COLOR, thickness=self.config.LINE_THICKNESS)
            
#             total_count = self.counts["person"]
#             count_text = f"person: {total_count}"
            
#             # Draw YELLOW counter text
#             cv2.putText(frame, count_text, (20, 50), self.config.FONT, 1.5, (0, 255, 255), 3)

#             # 5. Display the Frame
#             cv2.imshow("People Counter", frame)
#             if cv2.waitKey(1) & 0xFF == ord("q"):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

# people_counter.py

import cv2
import numpy as np
from collections import defaultdict, Counter
from .base_counter import BaseCounter # Correctly imports the base class

# --- Helper Functions for Line Intersection Logic ---
def on_segment(p, q, r):
    return (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
            q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1]))

def orientation(p, q, r):
    val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
    if val == 0: return 0
    return 1 if val > 0 else 2

def intersects(seg1_p1, seg1_p2, seg2_p1, seg2_p2):
    o1 = orientation(seg1_p1, seg1_p2, seg2_p1)
    o2 = orientation(seg1_p1, seg1_p2, seg2_p2)
    o3 = orientation(seg2_p1, seg2_p2, seg1_p1)
    o4 = orientation(seg2_p1, seg2_p2, seg1_p2)
    if o1 != o2 and o3 != o4: return True
    if o1 == 0 and on_segment(seg1_p1, seg2_p1, seg1_p2): return True
    if o2 == 0 and on_segment(seg1_p1, seg2_p2, seg1_p2): return True
    if o3 == 0 and on_segment(seg2_p1, seg1_p1, seg2_p2): return True
    if o4 == 0 and on_segment(seg2_p1, seg1_p2, seg2_p2): return True
    return False

# --- Main Counter Class ---
class PeopleCounter(BaseCounter): # <-- CORRECTLY INHERITS FROM BaseCounter
    def __init__(self, config, detector, tracker):
        # Initialize the parent BaseCounter class
        super().__init__(config, detector, tracker)
        # Add specific history for this counter
        self.track_history = defaultdict(list)
        self.polyline_np = np.array(self.config.COUNTING_POLYLINE, dtype=np.int32)
        self.log_file = getattr(self.config, "LOG_FILE", "events.log")

    def log_event(self, event_type):
        import datetime, json
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        log_entry = {"ts": ts, "event": "entry", "type": event_type}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def process_frame(self, frame):
        """
        This method overrides the one in BaseCounter.
        It contains the main detection, tracking, and counting logic.
        """
        # 1. Detection and Tracking
        results = self.detector(frame, conf=self.config.CONF_THRESH, verbose=False)
        detections_for_tracker = []
        for r in results:
            for box in r.boxes:
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                if self.detector.names.get(cls_id) in self.config.TARGET_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    detections_for_tracker.append([[x1, y1, w, h], conf, "person"])
        
        tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)

        # 2. Process Tracks and Check for Crossing
        for track in tracks:
            if not track.is_confirmed(): continue

            tid = int(track.track_id)
            l, t, r, b = map(int, track.to_ltrb())
            current_point = ((l + r) // 2, b)
            self.track_history[tid].append(current_point)
            
            if len(self.track_history[tid]) > 1:
                previous_point = self.track_history[tid][-2]
                for i in range(len(self.polyline_np) - 1):
                    line_seg_p1 = tuple(self.polyline_np[i])
                    line_seg_p2 = tuple(self.polyline_np[i+1])
                    if intersects(previous_point, current_point, line_seg_p1, line_seg_p2):
                        if tid not in self.counted_ids:
                            self.counts["person"] += 1
                            self.counted_ids.add(tid)
                            self.log_event("person")
                        break
        return tracks # Return tracks to be used by draw_overlay

    def draw_overlay(self, frame, tracks):
        """
        This method overrides the one in BaseCounter.
        It draws the black line, bounding boxes, and counter text.
        """
        # 1. Draw the BLACK Counting Line
        cv2.polylines(frame, [self.polyline_np], isClosed=False, color=self.config.LINE_COLOR, thickness=self.config.LINE_THICKNESS)

        # 2. Draw Bounding Boxes and Track Info for current tracks
        for track in tracks:
             if not track.is_confirmed(): continue
             tid = int(track.track_id)
             l, t, r, b = map(int, track.to_ltrb())
             cv2.rectangle(frame, (l, t), (r, b), (0, 200, 0), 2)
             cv2.putText(frame, f"person-{tid}", (l, t-10), self.config.FONT, 0.5, (255,255,255), 2)

        # 3. Draw Counter Text
        total_count = self.counts["person"]
        count_text = f"person: {total_count}"
        cv2.putText(frame, count_text, (20, 50), self.config.FONT, 1.5, (0, 255, 255), 3)
        
        return frame