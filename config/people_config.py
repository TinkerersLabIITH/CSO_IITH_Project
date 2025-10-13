# # config/people_config.py
import math
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
    # --- Configuration for Directional Band Counter ---

    # Line/Area for Counting
    LINE_CENTER_Y = 750                     # Vertical position (Y-coordinate) of the counting line's center.
    LINE_ANGLE_DEG_ANTICLOCKWISE = 0        # Angle of the line. 0 is horizontal, 90 is vertical.
    BAND_HALF_WIDTH_PX = 100                # The thickness of the counting zone. The total width will be 2x this value.

    # Counting Direction
    #  1 => Counts movement from the negative side to the positive side of the line (e.g., "entering").
    # -1 => Counts movement from the positive side to the negative side (e.g., "exiting").
    COUNT_ONLY_DIRECTION = 1

    # Gating Thresholds (how an object must behave to be counted)
    # These are fractions of BAND_HALF_WIDTH_PX.
    MIN_START_DIST_FRAC = 0.35              # An object must start at least this far into the origin side to be considered valid.
    MID_THRESHOLD_FRAC = 0.0                # An object must cross this point on the target side to be counted. 0.0 means crossing the center line is enough.

    # Stability and Duplicate Prevention
    MIN_FRAMES_BEFORE_COUNT = 1             # How many frames a track must exist before it can be counted.
    MIN_FRAMES_IN_BAND = 1                  # How many consecutive frames an object must be inside the band.
    DUP_RADIUS_PX = 80                      # Spatial radius (in pixels) to check for duplicates.
    DUP_FRAME_WINDOW = 100                  # Time window (in frames) to check for duplicates.

    # Appearance-Based Duplicate Suppression (using color histograms)
    HIST_BINS = (16, 8)                     # Bins for the HSV histogram.
    HIST_RANGE = [0, 180, 0, 256]           # Range for the HSV histogram.
    HIST_SIM_THRESH = 0.78                  # Similarity threshold (0 to 1). Higher is stricter.

    # Overlap Thresholds (for checking if a bbox is inside the band)
    MIN_OVERLAP_PIX = 30
    MIN_OVERLAP_RATIO = 0.005

    # Debugging
    DEBUG = True                            # Set to True to see visual debugging info, like why a track was rejected.

config = PeopleConfig()