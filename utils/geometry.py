import cv2
import numpy as np
import math

def make_band_polygon(frame_shape, center_y, angle_deg, half_width_px):
    """
    Construct a quadrilateral polygon for the counting band.

    Args:
        frame_shape: (H, W, C)
        center_y: y-coordinate of line center
        angle_deg: angle of line in degrees (anticlockwise)
        half_width_px: half thickness of band (perpendicular distance)
    Returns:
        np.ndarray polygon with shape (4,2)
    """
    h, w = frame_shape[:2]
    angle = math.radians(angle_deg)

    # Line equation: y = tan(angle) * (x - x0) + y0
    x0, y0 = w // 2, center_y
    dx, dy = math.cos(angle), math.sin(angle)

    # Perpendicular unit vector
    perp = np.array([-dy, dx])
    perp = perp / np.linalg.norm(perp)

    # Band edges (offset from line by ±half_width_px)
    offset = half_width_px * perp
    p1 = np.array([0, y0]) + offset
    p2 = np.array([w, y0 + dy * w]) + offset
    p3 = np.array([w, y0 + dy * w]) - offset
    p4 = np.array([0, y0]) - offset

    return np.array([p1, p2, p3, p4], dtype=np.int32)


def point_in_band(point, band_polygon):
    """
    Check if point is inside band polygon.
    """
    x, y = point
    return cv2.pointPolygonTest(band_polygon, (x, y), False) >= 0


def bbox_overlap_area(bbox, band_polygon):
    """
    Compute overlap area between bounding box and band.
    """
    x1, y1, x2, y2 = bbox
    box_poly = np.array([
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2]
    ], dtype=np.int32)

    inter = cv2.intersectConvexConvex(box_poly.astype(np.float32),
                                      band_polygon.astype(np.float32))
    if inter[0] > 0:
        return inter[0]
    return 0.0


def bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def iou_with_band(bbox, band_polygon):
    """
    Compute overlap ratio between bbox and band.
    """
    inter = bbox_overlap_area(bbox, band_polygon)
    area = bbox_area(bbox)
    if area == 0:
        return 0.0
    return inter / area


def euclidean_distance(p1, p2):
    """
    Simple 2D Euclidean distance.
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
