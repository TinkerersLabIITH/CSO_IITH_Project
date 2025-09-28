# config/car_config.py
import math
class CarConfig:

    YOLO_WEIGHTS = "yolov8m.pt"
    TARGET_CLASSES = {"car", "bus", "truck"}
    CONF_THRESH = 0.4
    LINE_CENTER_Y = 500
    BAND_HALF_WIDTH_PX = 120
        # ---- CHANGE HERE: 30 degrees anticlockwise ----
    LINE_ANGLE_DEG_ANTICLOCKWISE = 20
    # For standard math coordinates positive angle = counter-clockwise, so:
    LINE_ANGLE_RAD = math.radians(LINE_ANGLE_DEG_ANTICLOCKWISE)
    # Stability & duplicate protection
    MIN_FRAMES_BEFORE_COUNT = 2    # stable frames seen before counting logic
    MIN_FRAMES_IN_BAND = 2        # consecutive frames inside band required to count
    DUP_RADIUS_PX = 80
    DUP_FRAME_WINDOW = 100
    MIN_OVERLAP_PIX = 50          # minimal number of pixels overlap to consider "in-band"
    MIN_OVERLAP_RATIO = 0.01      # or >1% of bbox area overlapping
    # Duplicate protection & stability
    MIN_FRAMES_BEFORE_COUNT = 2    # track must be seen at least this many frames
    DUP_RADIUS_PX = 120            # radius to consider a counted event duplicate
    DUP_FRAME_WINDOW = 100         # frames window to consider duplicates
    DEEPSORT_MAX_AGE = 30
config = CarConfig()