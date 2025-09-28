import cv2
import numpy as np
from collections import defaultdict, Counter

class BaseCounter:
    def __init__(self, config, detector, tracker):
        self.config = config
        self.detector = detector
        self.tracker = tracker

        self.counts = Counter()
        self.counted_ids = set()
        self.histories = defaultdict(list)
        self.signatures = {}

    def process_frame(self, frame, frame_idx):
        detections = self.detector.detect(frame, self.config.TARGET_CLASSES)
        tracks = self.tracker.update(detections, frame)

        for track in tracks:
            if not track.is_confirmed() or track.det_conf < self.config.CONF_THRESH:
                continue

            tid = track.track_id
            x1, y1, x2, y2 = track.to_ltrb()
            cx, cy = int((x1 + x2) / 2), int((y1 + y2) / 2)

            # Store history for duplicate suppression
            self.histories[tid].append((frame_idx, (cx, cy)))

            # Band / overlap check
            if self._in_counting_band((cx, cy), frame.shape):
                if tid not in self.counted_ids:
                    self.counted_ids.add(tid)
                    self.counts[track.det_class] += 1

        return frame

    def _in_counting_band(self, point, frame_shape):
        # Simplified version, override for band geometry
        h, w = frame_shape[:2]
        x, y = point
        return abs(y - self.config.LINE_CENTER_Y) <= self.config.BAND_HALF_WIDTH_PX

    def draw_overlay(self, frame):
        # Draw line/band
        h, w = frame.shape[:2]
        y = self.config.LINE_CENTER_Y
        cv2.line(frame, (0, y), (w, y), (0, 255, 0), 2)

        # Draw counts
        y0 = 30
        for cls, cnt in self.counts.items():
            cv2.putText(frame, f"{cls}: {cnt}", (10, y0),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            y0 += 40
        return frame
