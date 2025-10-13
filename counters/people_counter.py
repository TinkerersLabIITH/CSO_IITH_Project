# people_counter.py

import cv2
import numpy as np
import math
from collections import defaultdict, Counter, deque
from .base_counter import BaseCounter # Correctly imports the base class

# ---------- HELPER FUNCTIONS (from people.py) ----------
# These are general utility functions that don't need to be part of the class.

def point_in_poly(poly_pts, x, y):
    """Checks if a point (x, y) is inside a polygon."""
    return cv2.pointPolygonTest(poly_pts, (float(x), float(y)), False) >= 0

def compute_hsv_hist(img, bbox, bins, hist_range):
    """Computes a normalized HSV histogram for a given bounding box."""
    l, t, r, b = bbox
    l = max(0, int(l)); t = max(0, int(t)); r = min(img.shape[1]-1, int(r)); b = min(img.shape[0]-1, int(b))
    w = r - l; h = b - t
    if w <= 2 or h <= 2: return None
    
    crop = img[t:b, l:r]
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return None
        
    hist = cv2.calcHist([hsv], [0, 1], None, [bins[0], bins[1]], hist_range)
    if hist is None: return None
    
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)

def signed_distance_to_line(p1x, p1y, px, py, line_angle_rad):
    """
    Calculates the signed perpendicular distance from a point (px, py) to a line.
    The line is defined by a point (p1x, p1y) on it and its angle.
    The sign indicates which side of the line the point is on.
    """
    # Vector from line point to the query point
    vec_x = px - p1x
    vec_y = py - p1y
    
    # Line normal vector
    nx = -math.sin(line_angle_rad)
    ny = math.cos(line_angle_rad)
    
    # Dot product gives the projected distance
    distance = vec_x * nx + vec_y * ny
    return distance


# ---------- MAIN COUNTER CLASS ----------

class PeopleCounter(BaseCounter):
    def __init__(self, config, detector, tracker):
        # Initialize the parent BaseCounter class
        super().__init__(config, detector, tracker)
        
        # --- State variables for the new band-crossing logic ---
        self.frame_idx = 0
        self.track_label_history = defaultdict(Counter)
        
        # Per-track state for crossing logic
        self.first_signed = {}          # {track_id: initial_signed_distance}
        self.prev_signed_dist = {}      # {track_id: last_known_signed_distance}
        self.track_frame_counts = {}    # {track_id: frames_seen}
        self.inband_consec_counts = {}  # {track_id: consecutive_frames_in_band}
        
        # Duplicate suppression state
        self.counted_events = deque()
        self.counted_signatures = []    # Stores {'hist', 'last_frame', 'pos'} for counted objects
        self.debug_reasons = {}         # Stores rejection reasons for drawing

        # --- Derived constants from config for efficiency ---
        self.line_angle_rad = math.radians(self.config.LINE_ANGLE_DEG_ANTICLOCKWISE)
        self.origin_side_needed = -1 if self.config.COUNT_ONLY_DIRECTION == 1 else 1
        self.target_side = 1 if self.config.COUNT_ONLY_DIRECTION == 1 else -1
        self.min_start_dist = self.config.BAND_HALF_WIDTH_PX * self.config.MIN_START_DIST_FRAC
        self.mid_threshold = self.config.BAND_HALF_WIDTH_PX * self.config.MID_THRESHOLD_FRAC

        # Load logging configuration
        self.log_file = getattr(self.config, "LOG_FILE", "events.log")

    def log_event(self, event_type):
        import datetime, json
        ts = datetime.datetime.utcnow().isoformat() + "Z"
        log_entry = {"ts": ts, "event": "entry", "type": event_type}
        with open(self.log_file, "a") as f:
            f.write(json.dumps(log_entry) + "\n")

    def process_frame(self, frame, return_events=False):
        """
        Overrides BaseCounter.process_frame with advanced directional counting logic.
        """
        self.frame_idx += 1
        fh, fw = frame.shape[:2]
        self.debug_reasons.clear() # Clear reasons from previous frame

        # --- 1. Define Counting Line and Band Geometry for this frame ---
        center_x, center_y = fw // 2, self.config.LINE_CENTER_Y
        
        # Calculate line endpoints for drawing (make it long enough to span the frame)
        line_length = max(fw, fh) * 1.6
        dx = math.cos(self.line_angle_rad) * (line_length / 2.0)
        dy = math.sin(self.line_angle_rad) * (line_length / 2.0)
        self.line_p1 = (int(center_x - dx), int(center_y - dy))
        self.line_p2 = (int(center_x + dx), int(center_y + dy))

        # Calculate the 4 corner points of the band polygon
        nx = -math.sin(self.line_angle_rad) # Normal vector's x component
        ny = math.cos(self.line_angle_rad)  # Normal vector's y component
        off_x = int(nx * self.config.BAND_HALF_WIDTH_PX)
        off_y = int(ny * self.config.BAND_HALF_WIDTH_PX)
        
        self.band_pts = np.array([
            [self.line_p1[0] - off_x, self.line_p1[1] - off_y], [self.line_p2[0] - off_x, self.line_p2[1] - off_y],
            [self.line_p2[0] + off_x, self.line_p2[1] + off_y], [self.line_p1[0] + off_x, self.line_p1[1] + off_y],
        ], dtype=np.int32)

        # Create a mask of the band area for overlap calculations
        band_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.fillPoly(band_mask, [self.band_pts], 255)

        # --- 2. Detection and Tracking ---
        results = self.detector(frame, conf=self.config.CONF_THRESH, verbose=False)
        detections_for_tracker = []
        for r in results:
            for box in r.boxes:
                cls_id, conf = int(box.cls[0]), float(box.conf[0])
                label = self.detector.names.get(cls_id)
                if label in self.config.TARGET_CLASSES:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    w, h = x2 - x1, y2 - y1
                    detections_for_tracker.append([[x1, y1, w, h], conf, label])
        
        tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)
        
        # --- 3. Process Each Track for Counting ---
        events = []
        for track in tracks:
            if not track.is_confirmed(): continue
            
            tid = int(track.track_id)
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2
            
            # Use detection class for label voting
            det_label = getattr(track, "det_class", "person")
            self.track_label_history[tid][det_label] += 1
            label = self.track_label_history[tid].most_common(1)[0][0]

            self.track_frame_counts[tid] = self.track_frame_counts.get(tid, 0) + 1
            frames_seen = self.track_frame_counts[tid]

            # --- Band Membership & Overlap ---
            bbox_area = max(1, r - l) * max(1, b - t)
            bbox_mask = np.zeros((fh, fw), dtype=np.uint8)
            cv2.rectangle(bbox_mask, (l, t), (r, b), 255, -1)
            overlap_pixels = int(cv2.countNonZero(cv2.bitwise_and(band_mask, bbox_mask)))
            overlap_ratio = overlap_pixels / (bbox_area + 1e-8)
            curr_in_band = (overlap_pixels >= self.config.MIN_OVERLAP_PIX) or (overlap_ratio >= self.config.MIN_OVERLAP_RATIO)
            
            if curr_in_band:
                self.inband_consec_counts[tid] = self.inband_consec_counts.get(tid, 0) + 1
            else:
                self.inband_consec_counts[tid] = 0

            # --- Positional Logic (Signed Distance) ---
            curr_signed = signed_distance_to_line(center_x, center_y, cx, cy, self.line_angle_rad)
            
            if tid not in self.first_signed:
                self.first_signed[tid] = curr_signed

            # --- Gating and Counting Decision ---
            should_count_now = False
            reject_reason = None

            if (self.inband_consec_counts.get(tid, 0) >= self.config.MIN_FRAMES_IN_BAND and
                frames_seen >= self.config.MIN_FRAMES_BEFORE_COUNT and
                label in self.config.TARGET_CLASSES and
                tid not in self.counted_ids):
                
                fs = self.first_signed.get(tid, 0.0)
                
                # Check if track started clearly on the required origin side
                origin_ok = (fs * self.origin_side_needed >= self.min_start_dist)

                if origin_ok and (curr_signed * self.target_side >= self.mid_threshold):
                    should_count_now = True
                else:
                    reject_reason = f"NotAcross fs={fs:.1f} curr={curr_signed:.1f}"
            else:
                if tid in self.counted_ids: reject_reason = "AlreadyCounted"
                else: reject_reason = "Gating"
            
            self.debug_reasons[tid] = reject_reason

            # --- Duplicate Checks and Final Count Registration ---
            if should_count_now:
                sig_hist = compute_hsv_hist(frame, (l, t, r, b), self.config.HIST_BINS, self.config.HIST_RANGE)
                is_dup = False
                
                # Check against recent counts via spatial proximity and color similarity
                if sig_hist is not None:
                    for sig in self.counted_signatures:
                        dist_sq = (sig['pos'][0] - cx)**2 + (sig['pos'][1] - cy)**2
                        if dist_sq < (self.config.DUP_RADIUS_PX**2):
                            sim = float(cv2.compareHist(sig['hist'], sig_hist, cv2.HISTCMP_CORREL))
                            if sim >= self.config.HIST_SIM_THRESH:
                                is_dup = True
                                self.debug_reasons[tid] = f"DupHist sim={sim:.2f}"
                                break
                
                if not is_dup:
                    self.counts[label] += 1
                    self.counted_ids.add(tid)
                    
                    # Log the event
                    if return_events:
                        import datetime
                        ts = datetime.datetime.utcnow().isoformat() + "Z"
                        events.append({"ts": ts, "event": "entry", "type": label})
                    else:
                        self.log_event(label)
                    
                    # Add to duplicate prevention signatures
                    if sig_hist is not None:
                        self.counted_signatures.append({'hist': sig_hist, 'last_frame': self.frame_idx, 'pos': (cx, cy)})
                    
                    self.debug_reasons[tid] = "COUNTED" # Mark as counted for visual feedback

            # Update state for next frame
            self.prev_signed_dist[tid] = curr_signed

        # --- Cleanup old signatures for performance ---
        self.counted_signatures = [s for s in self.counted_signatures if (self.frame_idx - s['last_frame']) <= self.config.DUP_FRAME_WINDOW]

        if return_events:
            return tracks, events
        return tracks

    def draw_overlay(self, frame, tracks):
        """
        Overrides BaseCounter.draw_overlay to draw the band, tracks, and debug info.
        """
        # --- 1. Draw the Counting Band and Center Line ---
        overlay = frame.copy()
        cv2.fillPoly(overlay, [self.band_pts], (0, 0, 128)) # Semi-transparent red for the band
        cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
        cv2.polylines(frame, [self.band_pts], isClosed=True, color=(0, 0, 255), thickness=2)
        cv2.line(frame, self.line_p1, self.line_p2, (0, 0, 180), 1)

        # --- 2. Draw Bounding Boxes and Track Info ---
        for track in tracks:
            if not track.is_confirmed(): continue
            tid = int(track.track_id)
            l, t, r, b = map(int, track.to_ltrb())
            
            # Set color based on status
            color = (0, 200, 0) # Default green
            if tid in self.counted_ids: color = (0, 255, 0) # Bright green if counted
            
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            label = self.track_label_history[tid].most_common(1)[0][0] if self.track_label_history[tid] else "person"
            cv2.putText(frame, f"{label}-{tid}", (l, t-10), self.config.FONT, 0.5, (255,255,255), 2)
            
            # Draw rejection reason if in debug mode
            if self.config.DEBUG and tid in self.debug_reasons:
                reason = self.debug_reasons[tid]
                if reason:
                    text_color = (0, 255, 0) if reason == "COUNTED" else (0, 128, 255)
                    cv2.putText(frame, reason, (l, b + 20), self.config.FONT, 0.5, text_color, 1)

        # --- 3. Draw Counter Text ---
        y_offset = 50
        for label, count in self.counts.items():
            count_text = f"{label}: {count}"
            cv2.putText(frame, count_text, (20, y_offset), self.config.FONT, 1.5, (0, 255, 255), 3)
            y_offset += 50
        
        return frame