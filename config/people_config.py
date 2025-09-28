# config/people_config.py
import math
class PeopleConfig:
    YOLO_WEIGHTS = "yolov8m.pt"
    TARGET_CLASSES = {"person"}
    CONF_THRESH = 0.35
    LINE_CENTER_Y = 600
    BAND_HALF_WIDTH_PX = 100
    DEEPSORT_MAX_AGE = 30
    LINE_CENTER_Y = 600
    LINE_ANGLE_DEG_ANTICLOCKWISE = 20
    LINE_ANGLE_RAD = math.radians(LINE_ANGLE_DEG_ANTICLOCKWISE)
    BAND_HALF_WIDTH_PX = 100  # thickness (perpendicular half-width)
    # Stability & duplicate protection
    MIN_FRAMES_BEFORE_COUNT = 2    # stable frames seen before counting logic
    MIN_FRAMES_IN_BAND = 2        # consecutive frames inside band required to count
    DUP_RADIUS_PX = 80
    DUP_FRAME_WINDOW = 100
    MIN_OVERLAP_PIX = 50          # minimal number of pixels overlap to consider "in-band"
    MIN_OVERLAP_RATIO = 0.01      # or >1% of bbox area overlapping
    # Appearance-based duplicate suppression (HSV histogram)
    HIST_BINS = (16, 8)              # hue, sat bins
    HIST_RANGE = [0, 180, 0, 256]    # ranges for H and S
    HIST_SIM_THRESH = 0.62           # correlation threshold for "same person" (tune)
    COUNTED_SIGNATURE_MAX_AGE = DUP_FRAME_WINDOW  # frames to keep signatures
config = PeopleConfig()