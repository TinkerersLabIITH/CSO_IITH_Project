import cv2
import numpy as np
import math
from collections import defaultdict, Counter
from .base_counter import BaseCounter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

class CarCounter(BaseCounter):
    def __init__(self, config):
        # Initialize YOLO detector and DeepSort tracker
        self.detector = YOLO(config.YOLO_WEIGHTS)
        self.tracker = DeepSort(max_age=config.DEEPSORT_MAX_AGE)
        super().__init__(config, self.detector, self.tracker)

        # Tracking state
        self.inband_counts = defaultdict(int)
        self.track_frame_counts = defaultdict(int)
        self.track_label_history = defaultdict(Counter)

    def run(self, video_source):
        cap = cv2.VideoCapture(video_source)
        if not cap.isOpened():
            print(f"Error opening video: {video_source}")
            return

        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            fh, fw = frame.shape[:2]

            # --- Compute counting line ---
            center_x = fw // 2
            center_y = self.config.LINE_CENTER_Y
            angle_rad = math.radians(self.config.LINE_ANGLE_DEG_ANTICLOCKWISE)
            line_length = int(max(fw, fh) * 1.6)
            dx = math.cos(angle_rad) * (line_length / 2.0)
            dy = math.sin(angle_rad) * (line_length / 2.0)
            x1_line = int(center_x - dx); y1_line = int(center_y - dy)
            x2_line = int(center_x + dx); y2_line = int(center_y + dy)

            # Normal vector for band
            lx = x2_line - x1_line; ly = y2_line - y1_line
            line_len = max(math.hypot(lx, ly), 1.0)
            nx = -ly / line_len; ny = lx / line_len
            off_x = int(nx * self.config.BAND_HALF_WIDTH_PX)
            off_y = int(ny * self.config.BAND_HALF_WIDTH_PX)

            band_pts = np.array([
                [x1_line - off_x, y1_line - off_y],
                [x2_line - off_x, y2_line - off_y],
                [x2_line + off_x, y2_line + off_y],
                [x1_line + off_x, y1_line + off_y],
            ], dtype=np.int32)

            # Band mask
            band_mask = np.zeros((fh, fw), dtype=np.uint8)
            cv2.fillPoly(band_mask, [band_pts], 255)

            # --- Detection ---
            results = self.detector(frame, conf=self.config.CONF_THRESH, verbose=False)
            all_boxes, all_scores, all_labels = [], [], []
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = self.detector.names.get(cls_id, None)
                    if label not in self.config.TARGET_CLASSES:
                        continue
                    x1, y1, x2, y2 = map(float, box.xyxy[0])
                    conf = float(box.conf[0])
                    all_boxes.append([x1, y1, x2, y2])
                    all_scores.append(conf)
                    all_labels.append(label)

            # Prepare for tracker ([x, y, w, h])
            detections_for_tracker = []
            if len(all_boxes) > 0:
                for i in range(len(all_boxes)):
                    x1, y1, x2, y2 = all_boxes[i]
                    w_box = x2 - x1; h_box = y2 - y1
                    detections_for_tracker.append([[int(x1), int(y1), int(w_box), int(h_box)],
                                                   float(all_scores[i]), str(all_labels[i])])

            # --- Tracking ---
            tracks = self.tracker.update_tracks(detections_for_tracker, frame=frame)

            # --- Process tracks ---
            for track in tracks:
                if not track.is_confirmed():
                    continue
                tid = int(track.track_id)
                l, t, r, b = map(int, track.to_ltrb())
                cx, cy = (l + r)//2, (t + b)//2
                px, py = cx, b

                # Label voting
                det_label = getattr(track, "det_class", None)
                if isinstance(det_label, int):
                    det_label = self.detector.names.get(det_label, None)
                if det_label:
                    self.track_label_history[tid][det_label] += 1
                    label = self.track_label_history[tid].most_common(1)[0][0]
                else:
                    label = self.track_label_history[tid].most_common(1)[0][0] if self.track_label_history[tid] else None

                # Update frames seen
                self.track_frame_counts[tid] += 1
                frames_seen = self.track_frame_counts[tid]

                # In-band / overlap check
                bottom_in = cv2.pointPolygonTest(band_pts, (px, py), False) >= 0
                center_in = cv2.pointPolygonTest(band_pts, (cx, cy), False) >= 0

                # bbox overlap
                bbox_mask = np.zeros((fh, fw), dtype=np.uint8)
                cv2.rectangle(bbox_mask, (l, t), (r, b), 255, -1)
                overlap_pixels = int(cv2.countNonZero(cv2.bitwise_and(band_mask, bbox_mask)))
                bbox_area = max(1, (r-l)*(b-t))
                overlap_ratio = overlap_pixels / (bbox_area + 1e-8)
                overlap_in = (overlap_pixels >= self.config.MIN_OVERLAP_PIX) or (overlap_ratio >= self.config.MIN_OVERLAP_RATIO)

                curr_in_band = bottom_in or center_in or overlap_in
                self.inband_counts[tid] = self.inband_counts.get(tid, 0) + (1 if curr_in_band else 0)

                # Decide to count
                should_count = (self.inband_counts[tid] >= self.config.MIN_FRAMES_IN_BAND
                                and frames_seen >= self.config.MIN_FRAMES_BEFORE_COUNT
                                and label not in self.counted_ids)

                if should_count:
                    self.counts[label] += 1
                    self.counted_ids.add(tid)

                # Draw bounding box + label
                txt = f"{label}-{tid}" if label else f"id-{tid}"
                cv2.rectangle(frame, (l, t), (r, b), (0, 0, 200), 2)
                cv2.putText(frame, txt, (l, max(12, t-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
                cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
                cv2.circle(frame, (px, py), 3, (0, 255, 255), -1)

            # --- Draw band + overlay ---
            overlay = frame.copy()
            cv2.fillPoly(overlay, [band_pts], (128,0,0))
            cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
            p1a = (x1_line - off_x, y1_line - off_y); p2a = (x2_line - off_x, y2_line - off_y)
            p1b = (x1_line + off_x, y1_line + off_y); p2b = (x2_line + off_x, y2_line + off_y)
            cv2.line(frame, p1a, p2a, (0,0,255), 2)
            cv2.line(frame, p1b, p2b, (0,0,255), 2)
            cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0,0,180), 1)

            # Draw counts
            y_offset = 30
            for cls, c in self.counts.items():
                cv2.putText(frame, f"{cls}: {c}", (10, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                y_offset += 30

            cv2.imshow("Car Counter", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cap.release()
        cv2.destroyAllWindows()
