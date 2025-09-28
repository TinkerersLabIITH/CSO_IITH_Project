# # config/people_config.py
# import math
# import cv2
# class PeopleConfig:
#     YOLO_WEIGHTS = "yolov8m.pt"
#     TARGET_CLASSES = {"person"}
#     CONF_THRESH = 0.35
#     LINE_CENTER_Y = 600
#     BAND_HALF_WIDTH_PX = 100
#     DEEPSORT_MAX_AGE = 30
#     LINE_CENTER_Y = 600
#     LINE_ANGLE_DEG_ANTICLOCKWISE = 20
#     LINE_ANGLE_RAD = math.radians(LINE_ANGLE_DEG_ANTICLOCKWISE)
#     BAND_HALF_WIDTH_PX = 100  # thickness (perpendicular half-width)
#     # Stability & duplicate protection
#     MIN_FRAMES_BEFORE_COUNT = 2    # stable frames seen before counting logic
#     MIN_FRAMES_IN_BAND = 2        # consecutive frames inside band required to count
#     DUP_RADIUS_PX = 80
#     DUP_FRAME_WINDOW = 100
#     MIN_OVERLAP_PIX = 50          # minimal number of pixels overlap to consider "in-band"
#     MIN_OVERLAP_RATIO = 0.01      # or >1% of bbox area overlapping
#     # Appearance-based duplicate suppression (HSV histogram)
#     HIST_BINS = (16, 8)              # hue, sat bins
#     HIST_RANGE = [0, 180, 0, 256]    # ranges for H and S
#     HIST_SIM_THRESH = 0.62           # correlation threshold for "same person" (tune)
#     COUNTED_SIGNATURE_MAX_AGE = DUP_FRAME_WINDOW  # frames to keep signatures
#     # The polyline coordinates have been estimated from the image you provided.
#     # You can adjust these (x, y) pixel values if the line is not perfectly placed.
#     COUNTING_POLYLINE = [(20, 600), (350, 580), (600, 570), (850, 520), (1280, 450)]

#     # --- Model and Detection Settings ---
#     YOLO_WEIGHTS = "yolov8n.pt"  # Path to YOLO model. It will be downloaded if not present.
#     CONF_THRESH = 0.4            # Detection confidence threshold.
#     TARGET_CLASSES = ["person"]  # We only want to count people.

#     # --- Tracking Settings ---
#     DEEPSORT_MAX_AGE = 30        # How many frames to keep a track without a new detection.
    
#     # --- Aesthetics ---
#     LINE_COLOR = (0, 0, 0)       # Black color for the counting line.
#     LINE_THICKNESS = 3           # Make the line thicker and more visible.
#     FONT = cv2.FONT_HERSHEY_SIMPLEX
# config = PeopleConfig()

# config/people_config.py

import cv2

class PeopleConfig:
    # --- The custom line for counting ---
    # Coordinates estimated from your screenshot.
    # COUNTING_POLYLINE = [(20, 600), (350, 580), (600, 570), (850, 520), (1280, 450)]
    COUNTING_POLYLINE = [(10, 150), (1000, 150)]
    COUNTING_POLYLINE = [(3, 416), (514, 289)]

    # --- Model and Detection Settings ---
    YOLO_WEIGHTS = "yolov8n.pt"
    CONF_THRESH = 0.4
    TARGET_CLASSES = ["person"]

    # --- Tracking Settings ---
    DEEPSORT_MAX_AGE = 30
    
    # --- Aesthetics ---
    LINE_COLOR = (0, 0, 0)       # This is BLACK for the counting line.
    LINE_THICKNESS = 3
    FONT = cv2.FONT_HERSHEY_SIMPLEX

config = PeopleConfig()