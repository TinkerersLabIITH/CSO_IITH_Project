# one_way_position_based_counter_fixed.py
import cv2
import numpy as np
import torch
import math
from collections import defaultdict, Counter, deque
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Optional: torchvision NMS fallback
try:
    from torchvision.ops import nms as tv_nms
    _HAS_TORCHVISION_NMS = True
except Exception:
    _HAS_TORCHVISION_NMS = False

# ---------- CONFIG ----------
VIDEO_PATH = r"D:\pytorch_projects+tensorflow_projects_3.12\TL_FLAGSHIP\TL_MAIN_GATE\cctv footage videos\Students footages\MAIN GATE_INGATE PEDESTRAIN_20250902075201_20250902075801.mp4"
YOLO_WEIGHTS = "yolov8m.pt"
CONF_THRESH = 0.35
IOU_NMS = 0.8

# Counting line/area
LINE_CENTER_Y = 600
LINE_ANGLE_DEG_ANTICLOCKWISE = 0
LINE_ANGLE_RAD = math.radians(LINE_ANGLE_DEG_ANTICLOCKWISE)
BAND_HALF_WIDTH_PX = 100  # thickness (perpendicular half-width)

# ONE-DIRECTION COUNTING
#  1 => count crossings from -1 -> +1 (entering)
# -1 => count crossings from +1 -> -1 (exiting)
COUNT_ONLY_DIRECTION = 1

# Position-based thresholds (we use signed distance to line)
MIN_START_DIST_FRAC = 0.35   # how far (fraction of band half-width) a track must be to be considered "clearly on origin side"
MID_THRESHOLD_FRAC = 0.0     # midpoint = crossing center line (0). Set >0 to require further movement.
FAR_THRESHOLD_FRAC = 0.5     # optional: require reaching at least this fraction of band half-width on target side

# HUMAN CLASS
HUMAN_CLASSES = {"person"}

# Stability & duplicate protection
MIN_FRAMES_BEFORE_COUNT = 1
MIN_FRAMES_IN_BAND = 1
DUP_RADIUS_PX = 80
DUP_FRAME_WINDOW = 100

# Overlap thresholds (tweak)
MIN_OVERLAP_PIX = 30
MIN_OVERLAP_RATIO = 0.005

# Appearance-based duplicate suppression (HSV histogram)
HIST_BINS = (16, 8)
HIST_RANGE = [0, 180, 0, 256]
HIST_SIM_THRESH = 0.78
COUNTED_SIGNATURE_MAX_AGE = DUP_FRAME_WINDOW

# DeepSORT params
DEEPSORT_MAX_AGE = 30

# Debugging
DEBUG = True   # set False to hide per-track reject reasons

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
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(int(i))
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1); h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(iou <= iou_thresh)[0]
        order = order[inds + 1]
    return keep

def point_in_poly(poly_pts, x, y):
    return cv2.pointPolygonTest(poly_pts, (float(x), float(y)), False) >= 0

def compute_hsv_hist(img, bbox, bins=HIST_BINS):
    l, t, r, b = bbox
    l = max(0, l); t = max(0, t); r = min(img.shape[1]-1, r); b = min(img.shape[0]-1, b)
    w = r - l; h = b - t
    if w <= 2 or h <= 2:
        return None
    crop = img[t:b, l:r]
    try:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    except Exception:
        return None
    hist = cv2.calcHist([hsv], [0, 1], None, [bins[0], bins[1]], HIST_RANGE)
    if hist is None:
        return None
    cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    return hist.flatten().astype(np.float32)

def signed_distance_to_line(p1x, p1y, p2x, p2y, px, py):
    lx = p2x - p1x
    ly = p2y - p1y
    line_len = math.hypot(lx, ly) if (lx != 0 or ly != 0) else 1.0
    cross = (p2x - p1x) * (py - p1y) - (p2y - p1y) * (px - p1x)
    return cross / line_len

# ---------- INITIALIZE ----------
model = YOLO(YOLO_WEIGHTS)
tracker = DeepSort(max_age=DEEPSORT_MAX_AGE)

cap = cv2.VideoCapture(VIDEO_PATH)

human_counts = {h: 0 for h in HUMAN_CLASSES}
counted_ids = set()
prev_centers = {}
prev_in_band = {}
inband_consec_counts = {}
track_frame_counts = {}
track_label_history = defaultdict(Counter)
counted_events = deque()  # keep as deque to prune efficiently
counted_signatures = []
frame_idx = 0

# Per-track crossing / first-signed state
first_signed = {}         # track_id -> signed distance when first seen
prev_signed_dist = {}     # track_id -> last signed distance
max_abs_pre_cross = {}    # track_id -> max abs signed dist before crossing

# Derived constants
origin_side_needed = -1 if COUNT_ONLY_DIRECTION == 1 else 1
target_side = 1 if COUNT_ONLY_DIRECTION == 1 else -1
min_start_dist = BAND_HALF_WIDTH_PX * MIN_START_DIST_FRAC
mid_threshold = BAND_HALF_WIDTH_PX * MID_THRESHOLD_FRAC
far_threshold = BAND_HALF_WIDTH_PX * FAR_THRESHOLD_FRAC
tolerance = BAND_HALF_WIDTH_PX * 1.5  # tolerance used for "near center" checks

# ---------- MAIN LOOP ----------
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    fh, fw = frame.shape[:2]

    # compute line endpoints & normal
    center_x = fw // 2
    center_y = LINE_CENTER_Y
    line_length = int(max(fw, fh) * 1.6)
    dx = math.cos(LINE_ANGLE_RAD) * (line_length / 2.0)
    dy = math.sin(LINE_ANGLE_RAD) * (line_length / 2.0)
    x1_line = int(center_x - dx); y1_line = int(center_y - dy)
    x2_line = int(center_x + dx); y2_line = int(center_y + dy)

    lx = x2_line - x1_line; ly = y2_line - y1_line
    line_len = math.hypot(lx, ly) if (lx != 0 or ly != 0) else 1.0
    nx = -ly / line_len; ny = lx / line_len
    off_x = int(nx * BAND_HALF_WIDTH_PX); off_y = int(ny * BAND_HALF_WIDTH_PX)
    band_pts = np.array([
        [x1_line - off_x, y1_line - off_y],
        [x2_line - off_x, y2_line - off_y],
        [x2_line + off_x, y2_line + off_y],
        [x1_line + off_x, y1_line + off_y],
    ], dtype=np.int32)

    # Build band mask once per frame
    band_mask = np.zeros((fh, fw), dtype=np.uint8)
    cv2.fillPoly(band_mask, [band_pts], 255)

    # Run YOLO
    try:
        results = model(frame, conf=CONF_THRESH, verbose=False)
    except Exception as e:
        print(f"YOLO error at frame {frame_idx}: {e}")
        continue

    all_boxes = []
    all_scores = []
    all_labels = []
    for r in results:
        for box in r.boxes:
            try:
                cls_id = int(box.cls[0])
                label = model.names.get(cls_id, None)
            except Exception:
                label = None
            if label is None or label not in HUMAN_CLASSES:
                continue
            x1, y1, x2, y2 = map(float, box.xyxy[0])
            conf = float(box.conf[0])
            all_boxes.append([x1, y1, x2, y2])
            all_scores.append(conf)
            all_labels.append(label)

    # prepare detections for tracker (DeepSort expects [x, y, w, h])
    detections_for_tracker = []
    if len(all_boxes) > 0:
        boxes_np = np.array(all_boxes, dtype=np.float32)
        scores_np = np.array(all_scores, dtype=np.float32)
        labels_np = np.array(all_labels, dtype=object)
        keep_indices = class_agnostic_nms(boxes_np, scores_np, IOU_NMS)
        for i in keep_indices:
            x1, y1, x2, y2 = boxes_np[i]
            w_box = x2 - x1; h_box = y2 - y1
            detections_for_tracker.append([[int(x1), int(y1), int(w_box), int(h_box)], float(scores_np[i]), str(labels_np[i])])

    try:
        tracks = tracker.update_tracks(detections_for_tracker, frame=frame)
    except Exception:
        tracks = []

    # Draw band overlay and info
    overlay = frame.copy()
    cv2.fillPoly(overlay, [band_pts], (0, 0, 128))
    cv2.addWeighted(overlay, 0.12, frame, 1 - 0.12, 0, frame)
    p1a = (x1_line - off_x, y1_line - off_y); p2a = (x2_line - off_x, y2_line - off_y)
    p1b = (x1_line + off_x, y1_line + off_y); p2b = (x2_line + off_x, y2_line + off_y)
    cv2.line(frame, p1a, p2a, (0, 0, 255), 2); cv2.line(frame, p1b, p2b, (0, 0, 255), 2)
    cv2.line(frame, (x1_line, y1_line), (x2_line, y2_line), (0, 0, 180), 1)
    cv2.putText(frame, f"{LINE_ANGLE_DEG_ANTICLOCKWISE}deg band_half={BAND_HALF_WIDTH_PX}px dir={'Enter' if COUNT_ONLY_DIRECTION==1 else 'Exit'}",
                (10, fh - 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 2)

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

        # bbox LTRB
        try:
            l, t, r, b = map(int, track.to_ltrb())
        except Exception:
            try:
                bbox = track.to_ltrb()
                l, t, r, b = map(int, bbox)
            except Exception:
                continue

        # center and bottom points
        cx = int((l + r) / 2); cy = int((t + b) / 2)
        bottom_x = cx; bottom_y = b

        # label voting
        det_label = None
        try:
            det_label = track.get_det_class()
        except Exception:
            det_label = getattr(track, "det_class", None)
        if isinstance(det_label, int):
            det_label = model.names.get(det_label, None)
        if det_label is not None:
            det_label = str(det_label)
        if det_label:
            track_label_history[track_id][det_label] += 1
            label = track_label_history[track_id].most_common(1)[0][0]
        else:
            label = track_label_history[track_id].most_common(1)[0][0] if track_label_history[track_id] else None

        # draw bbox + id + label
        cv2.rectangle(frame, (l, t), (r, b), (0,200,0), 2)
        txt = f"{label}-{track_id}" if label else f"id-{track_id}"
        cv2.putText(frame, txt, (l, max(12, t-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2)
        cv2.circle(frame, (cx, cy), 3, (255, 0, 0), -1)
        cv2.circle(frame, (bottom_x, bottom_y), 3, (0, 255, 255), -1)

        # update frames seen
        track_frame_counts[track_id] = track_frame_counts.get(track_id, 0) + 1
        frames_seen = track_frame_counts[track_id]

        # band membership tests (use bottom and center + overlap)
        bottom_in = point_in_poly(band_pts, bottom_x, bottom_y)
        center_in = point_in_poly(band_pts, cx, cy)
        bbox_w = max(1, r - l); bbox_h = max(1, b - t)
        bbox_area = bbox_w * bbox_h
        bbox_mask = np.zeros((fh, fw), dtype=np.uint8)
        cv2.rectangle(bbox_mask, (l, t), (r, b), 255, -1)
        overlap = cv2.bitwise_and(band_mask, bbox_mask)
        overlap_pixels = int(cv2.countNonZero(overlap))
        overlap_ratio = overlap_pixels / (bbox_area + 1e-8)
        overlap_in = (overlap_pixels >= MIN_OVERLAP_PIX) or (overlap_ratio >= MIN_OVERLAP_RATIO)
        curr_in_band = bottom_in or center_in or overlap_in

        # consecutive frames inside
        if curr_in_band:
            inband_consec_counts[track_id] = inband_consec_counts.get(track_id, 0) + 1
        else:
            inband_consec_counts[track_id] = 0

        # compute signed distance & side using center (more stable)
        curr_signed = signed_distance_to_line(x1_line, y1_line, x2_line, y2_line, cx, cy)
        curr_side = 1 if curr_signed > 0 else (-1 if curr_signed < 0 else 0)

        # initialize first_signed if needed
        if track_id not in first_signed:
            first_signed[track_id] = curr_signed
            max_abs_pre_cross[track_id] = abs(curr_signed)
        else:
            # update max absolute distance before crossing (useful for gating) when sign hasn't flipped
            prev_s = prev_signed_dist.get(track_id, None)
            if prev_s is None or (np.sign(curr_signed) == np.sign(prev_s)):
                max_abs_pre_cross[track_id] = max(max_abs_pre_cross.get(track_id, 0.0), abs(curr_signed))

        # gating (improved / corrected)
        should_count_now = False
        reject_reason = None

        if (inband_consec_counts.get(track_id, 0) >= MIN_FRAMES_IN_BAND
            and frames_seen >= MIN_FRAMES_BEFORE_COUNT
            and label in human_counts
            and track_id not in counted_ids):

            fs = first_signed.get(track_id, 0.0)
            prev_signed = prev_signed_dist.get(track_id, None)

            # determine whether track clearly started on the expected origin side
            if origin_side_needed == -1:
                origin_ok = (fs <= -min_start_dist)
            else:
                origin_ok = (fs >= min_start_dist)

            # Primary test: clearly started on origin side AND current center reached midpoint/far threshold on target side
            if origin_ok and (curr_signed * target_side >= mid_threshold):
                should_count_now = True
            else:
                # Secondary: sign flip detection with previous on origin and current at/over midpoint on target side
                if prev_signed is not None:
                    # consider prev_on_origin true if previous was on origin side (strong) or far enough on origin side
                    prev_on_origin = (prev_signed * origin_side_needed > 0) or (prev_signed * origin_side_needed <= -min_start_dist)
                    # strict sign flip: product < 0 -> opposite signs
                    if (prev_signed * curr_signed < 0) and prev_on_origin and (curr_signed * target_side >= mid_threshold):
                        should_count_now = True
                    else:
                        reject_reason = f"not_reached_mid fs={fs:.1f} prev={prev_signed:.1f} curr={curr_signed:.1f}"
                else:
                    reject_reason = f"started_inside fs={fs:.1f} curr={curr_signed:.1f}"

        else:
            if label not in human_counts:
                reject_reason = "label_not_human"
            elif track_id in counted_ids:
                reject_reason = "already_counted"
            else:
                reject_reason = "gating_frames_or_band"

        # duplicate checks and final register
        if should_count_now:
            sig_hist = compute_hsv_hist(frame, (l, t, r, b), bins=HIST_BINS)
            is_dup = False
            if sig_hist is not None and len(counted_signatures) > 0:
                for sig in counted_signatures:
                    if (frame_idx - sig['last_frame']) > COUNTED_SIGNATURE_MAX_AGE:
                        continue
                    sx, sy = sig['pos']
                    dx = sx - cx; dy = sy - cy
                    # allow a bit larger radius check (squared)
                    if dx*dx + dy*dy > (DUP_RADIUS_PX * DUP_RADIUS_PX * 9):
                        continue
                    try:
                        sim = float(cv2.compareHist(sig['hist'].astype(np.float32), sig_hist.astype(np.float32), cv2.HISTCMP_CORREL))
                    except Exception:
                        sim = -1.0
                    if sim >= HIST_SIM_THRESH:
                        is_dup = True
                        sig['last_frame'] = frame_idx
                        sig['pos'] = (cx, cy)
                        reject_reason = f"dup_hist_sim={sim:.2f}"
                        break

            if not is_dup:
                # spatial/time-based duplication against recent counted_events
                for ev in list(counted_events):
                    ev_label, ev_cx, ev_cy, ev_frame = ev
                    if ev_label == label and (frame_idx - ev_frame) <= DUP_FRAME_WINDOW:
                        dx_e = cx - ev_cx; dy_e = cy - ev_cy
                        if dx_e*dx_e + dy_e*dy_e <= DUP_RADIUS_PX * DUP_RADIUS_PX:
                            is_dup = True
                            reject_reason = "dup_spatial"
                            break

            if not is_dup:
                human_counts[label] += 1
                counted_ids.add(track_id)
                counted_events.append((label, cx, cy, frame_idx))
                # trim oldest events beyond window
                while counted_events and (frame_idx - counted_events[0][3]) > DUP_FRAME_WINDOW:
                    counted_events.popleft()
                cv2.putText(frame, f"COUNTED {label}", (cx-20, cy-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
                if sig_hist is not None:
                    counted_signatures.append({'hist': sig_hist, 'last_frame': frame_idx, 'pos': (cx, cy)})

            # clean counted_signatures by age
            counted_signatures = [s for s in counted_signatures if (frame_idx - s['last_frame']) <= COUNTED_SIGNATURE_MAX_AGE]

        # Debug overlay: why rejected (if DEBUG)
        if DEBUG and reject_reason:
            cv2.putText(frame, reject_reason, (max(0, l-4), min(fh-10, b+36)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 128, 255), 1)

        # Optional debug print for offline trace
        if DEBUG:
            prev_signed_str = f"{prev_signed_dist[track_id]:.1f}" if track_id in prev_signed_dist and prev_signed_dist[track_id] is not None else "None"
            print(f"F{frame_idx} id{track_id} label={label} fs={first_signed.get(track_id,0.0):.1f} prev={prev_signed_str} curr={curr_signed:.1f} origin_ok={('Y' if ('origin_ok' in locals() and origin_ok) else 'N')} inband={int(curr_in_band)} frames={frames_seen} should_count={should_count_now} rej={reject_reason}")

        # Update prev signed
        prev_signed_dist[track_id] = curr_signed

    # prune old counted_events (already done when appending, but keep safe)
    while counted_events and (frame_idx - counted_events[0][3]) > DUP_FRAME_WINDOW:
        counted_events.popleft()

    # display counts
    y_offset = 30
    for hlabel, c in human_counts.items():
        cv2.putText(frame, f"{hlabel}: {c}", (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += 30

    cv2.imshow("One-way Position-based Counter (fixed)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
