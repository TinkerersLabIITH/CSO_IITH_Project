# counters/car_counter.py
import cv2
import numpy as np
from collections import defaultdict
from .base_counter import BaseCounter

# Helper functions can be shared or defined here
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
    return False # Simplified for this case

class CarCounter(BaseCounter):
    def __init__(self, config, detector, tracker):
        super().__init__(config, detector, tracker)
        self.track_history = defaultdict(list)
        self.polyline_np = np.array(self.config.COUNTING_POLYLINE, dtype=np.int32)
        self.log_file = getattr(self.config, "LOG_FILE", "events.log")

    def log_event(self, event_type):
        import datetime, json
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        log_entry = {"ts": ts, "event": "entry", "type": event_type}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def process_frame(self, frame, return_events=False):
        """
        Overrides BaseCounter's method to process a frame for cars.
        If return_events=True, returns (tracks, [event_dict, ...]) for thread-safe logging.
        """
        import datetime
        results = self.detector(frame, conf=self.config.CONF_THRESH, verbose=False)
        detections_for_tracker = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.detector.names.get(cls_id)
                if label in self.config.TARGET_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])
                    w, h = x2 - x1, y2 - y1
                    detections_for_tracker.append([[x1, y1, w, h], conf, label])
        
        tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)

        events = []
        for track in tracks:
            if not track.is_confirmed() or track.det_class not in self.config.TARGET_CLASSES:
                continue

            tid = int(track.track_id)
            current_point = (track.to_ltrb()[0] + track.to_ltrb()[2]) // 2, track.to_ltrb()[3]
            self.track_history[tid].append(current_point)
            
            if len(self.track_history[tid]) > 1:
                previous_point = self.track_history[tid][-2]
                for i in range(len(self.polyline_np) - 1):
                    line_seg_p1 = tuple(self.polyline_np[i])
                    line_seg_p2 = tuple(self.polyline_np[i+1])
                    if intersects(previous_point, current_point, line_seg_p1, line_seg_p2):
                        if tid not in self.counted_ids:
                            self.counts[track.det_class] += 1
                            self.counted_ids.add(tid)
                            if return_events:
                                ts = datetime.datetime.utcnow().isoformat() + "Z"
                                events.append({"ts": ts, "event": "entry", "type": track.det_class})
                            else:
                                self.log_event(track.det_class)
                        break
        if return_events:
            return tracks, events
        return tracks

    def draw_overlay(self, frame, tracks):
        """
        Overrides BaseCounter's method to draw the overlay for cars.
        """
        cv2.polylines(frame, [self.polyline_np], isClosed=False, color=self.config.LINE_COLOR, thickness=self.config.LINE_THICKNESS)

        for track in tracks:
            if not track.is_confirmed() or track.det_class not in self.config.TARGET_CLASSES:
                continue
            tid = int(track.track_id)
            l, t, r, b = map(int, track.to_ltrb())
            cv2.rectangle(frame, (l, t), (r, b), (200, 0, 0), 2) # Blue for cars
            cv2.putText(frame, f"{track.det_class}-{tid}", (l, t-10), self.config.FONT, 0.5, (255,255,255), 2)

        y_offset = 30
        for cls, count in self.counts.items():
            count_text = f"{cls}: {count}"
            cv2.putText(frame, count_text, (10, y_offset), self.config.FONT, 0.9, (255, 255, 255), 2)
            y_offset += 30
            
        return frame