# yolo_detector.py
from ultralytics import YOLO

class YoloDetector:
    def __init__(self, weights, conf_thresh):
        self.model = YOLO(weights)
        self.conf_thresh = conf_thresh

    def detect(self, frame, target_classes):
        results = self.model(frame, conf=self.conf_thresh, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                label = self.model.names[cls_id]
                if label not in target_classes:
                    continue
                x1, y1, x2, y2 = map(float, box.xyxy[0])
                detections.append({
                    "label": label,
                    "conf": float(box.conf[0]),
                    "bbox": (x1, y1, x2, y2),
                })
        return detections
