import cv2
import numpy as np
import torch
import math
from collections import defaultdict, Counter
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Optional: use torchvision NMS (faster/robust). Fallback to python NMS if not available.
try:
    from torchvision.ops import nms as tv_nms
    _HAS_TORCHVISION_NMS = True
except Exception:
    _HAS_TORCHVISION_NMS = False

# ---------- CONFIG ----------
VIDEO_PATH = r"D:\pytorch_projects+tensorflow_projects_3.12\TL_FLAGSHIP\TL_MAIN_GATE\MAIN GATE_INGATE 1_20250902080000_20250902080459.mp4"
YOLO_WEIGHTS = "yolov8m.pt"   # swap to yolov8s.pt / yolov8m.pt for better accuracy
CONF_THRESH = 0.4
IOU_NMS = 0.5

# Define the (center) position of the counting line and the tilt angle
LINE_CENTER_Y = 600

# ---- CHANGE HERE: 30 degrees anticlockwise ----
LINE_ANGLE_DEG_ANTICLOCKWISE = 20
# For standard math coordinates positive angle = counter-clockwise, so:
LINE_ANGLE_RAD = math.radians(LINE_ANGLE_DEG_ANTICLOCKWISE)
# ------------------------------------------------

VEHICLE_CLASSES = {"car", "motorcycle", "bus", "truck"}  # "motorcycle" is YOLO label

# Duplicate protection & stability
MIN_FRAMES_BEFORE_COUNT = 2    # track must be seen at least this many frames
DUP_RADIUS_PX = 120            # radius to consider a counted event duplicate
DUP_FRAME_WINDOW = 100         # frames window to consider duplicates

# DeepSORT params
DEEPSORT_MAX_AGE = 30

# ---------- HELPERS ----------
def class_agnostic_nms(boxes, scores, iou_thresh=0.5):
    if len(boxes) == 0:
        return []
    if _HAS_TORCHVISION_NMS:
        with torch.no_grad():
            b = torch.from_numpy(boxes).float()
            s = torch.from_numpy(scores).float()
            keep = tv_nms(b, s, iou_thresh).cpu().numpy().tolist()
            return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

# ---------- INITIALIZE ----------
model = YOLO(YOLO_WEIGHTS)
tracker = DeepSort(max_age=DEEPSORT_MAX_AGE)

cap = cv2.VideoCapture(VIDEO_PATH)

vehicle_counts = {v: 0 for v in VEHICLE_CLASSES}
counted_ids = set()
prev_centers = {}
prev_sides = {}
track_frame_counts = {}
track_label_history = defaultdict(Counter)
counted_events = []
frame_idx = 0

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    h, w = frame.shape[:2]

    # compute angled line endpoints centered at (center_x, LINE_CENTER_Y)
    center_x = w // 2
    center_y = LINE_CENTER_Y
    # choose line length long enough to cross the frame diagonally
    line_length = int(max(w, h) * 1.6)
    dx = math.cos(LINE_ANGLE_RAD) * (line_length / 2.0)
    dy = math.sin(LINE_ANGLE_RAD) * (line_length / 2.0)
    x1_line = int(center_x - dx)
    y1_line = int(center_y - dy)
    x2_line = int(center_x + dx)
    y2_line = int(center_y + dy)

    # Run YOLO (suppress verbose output)
    results = model(frame, conf=CONF_THRESH, verbose=False)

    all_boxes = []
    all_scores = []
    all_labels = []
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names.get(cls_id, None)
            if label is None or label not in VEHICLE_CLASSES:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)
            all_labels.append(label)

    detections_for_tracker = []
    if len(all_boxes) > 0:
        boxes_np = np.array(all_boxes, dtype=np.float32)
        scores_np = np.array(all_scores, dtype=np.float32)
        labels_np = np.array(all_labels, dtype=object)

        keep_indices = class_agnostic_nms(boxes_np, scores_np, IOU_NMS)
        for i in keep_indices:
            x1, y1, x2, y2 = boxes_np[i]
            w_box = x2 - x1
            h_box = y2 - y1
            detections_for_tracker.append([[int(x1), int(y1), int(w_box), int(h_box)], float(scores_np[i]), str(labels_np[i])])

    tracks = tracker.update_tracks(detections_for_tracker, frame=frame)

    # draw the angled counting line
    cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 255), 2)
    # annotate the angle and direction
    cv2.putText(frame, f"{LINE_ANGLE_DEG_ANTICLOCKWISE}deg anticlockwise", (10, h - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)

    # helper: cross product sign for point (px,py) wrt line p1->p2
    def side_of_line(p1x, p1y, p2x, p2y, px, py):
        cross = (p2x - p1x) * (py - p1y) - (p2y - p1y) * (px - p1x)
        if cross > 0:
            return 1
        elif cross < 0:
            return -1
        else:
            return 0

    # Process confirmed tracks
    for track in tracks:
        try:
            confirmed = track.is_confirmed()
        except Exception:
            confirmed = getattr(track, "is_confirmed", True)
        if not confirmed:
            continue

        try:
            track_id = int(track.track_id)
        except Exception:
            track_id = int(getattr(track, "track_id", -1))
        if track_id == -1:
            continue

        # get bbox
        try:
            l, t, r, b = map(int, track.to_ltrb())
        except Exception:
            continue

        cx = int((l + r) / 2)
        cy = int((t + b) / 2)

        # get label from track (if available)
        det_label = None
        try:
            det_label = track.get_det_class()
        except Exception:
            det_label = getattr(track, "det_class", None)
        if isinstance(det_label, int):
            det_label = model.names.get(det_label, None)
        if det_label is not None:
            det_label = str(det_label)

        # update label voting
        if det_label:
            track_label_history[track_id][det_label] += 1
            label = track_label_history[track_id].most_common(1)[0][0]
        else:
            label = track_label_history[track_id].most_common(1)[0][0] if track_label_history[track_id] else None

        # draw bbox + id+label
        txt = f"{label}-{track_id}" if label else f"id-{track_id}"
        cv2.rectangle(frame, (l, t), (r, b), (0, 255, 0), 2)
        cv2.putText(frame, txt, (l, max(12, t - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)

        # update frames seen
        track_frame_counts[track_id] = track_frame_counts.get(track_id, 0) + 1
        frames_seen = track_frame_counts[track_id]

        # compute which side of the angled line the center is on
        curr_side = side_of_line(x1_line, y1_line, x2_line, y2_line, cx, cy)
        prev_side = prev_sides.get(track_id)

        crossed = False
        if prev_side is not None and prev_side != 0 and curr_side != 0 and prev_side != curr_side:
            crossed = True

        # counting logic
        if (frames_seen >= MIN_FRAMES_BEFORE_COUNT and label in vehicle_counts and crossed and track_id not in counted_ids):
            is_dup = False
            for ev in counted_events:
                ev_label, ev_cx, ev_cy, ev_frame = ev
                if ev_label == label and (frame_idx - ev_frame) <= DUP_FRAME_WINDOW:
                    dx_e = cx - ev_cx
                    dy_e = cy - ev_cy
                    if dx_e*dx_e + dy_e*dy_e <= DUP_RADIUS_PX * DUP_RADIUS_PX:
                        is_dup = True
                        break
            if not is_dup:
                vehicle_counts[label] += 1
                counted_ids.add(track_id)
                counted_events.append((label, cx, cy, frame_idx))
                cv2.putText(frame, f"COUNTED {label}", (cx - 20, cy - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

        # update previous side and center
        prev_sides[track_id] = curr_side
        prev_centers[track_id] = cy

    # prune counted_events
    while counted_events and (frame_idx - counted_events[0][3]) > DUP_FRAME_WINDOW:
        counted_events.pop(0)

    # display counts
    y_offset = 30
    for v, c in vehicle_counts.items():
        cv2.putText(frame, f"{v}: {c}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30

    cv2.imshow("Vehicle Counter - Angled Line (30° anticlockwise)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
