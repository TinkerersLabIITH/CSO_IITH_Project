# deep_sort_tracker.py
from deep_sort_realtime.deepsort_tracker import DeepSort

class DeepSortWrapper:
    def __init__(self, max_age=30):
        self.tracker = DeepSort(max_age=max_age)

    def update(self, detections, frame):
        formatted = []
        for d in detections:
            x1, y1, x2, y2 = d["bbox"]
            w, h = x2-x1, y2-y1
            formatted.append([[int(x1), int(y1), int(w), int(h)], d["conf"], d["label"]])
        tracks = self.tracker.update_tracks(formatted, frame=frame)
        return tracks
