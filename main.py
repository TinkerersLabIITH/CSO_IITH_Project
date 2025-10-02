
import cv2
import threading
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from counters.people_counter import PeopleCounter
from counters.car_counter import CarCounter
import time

# Thread-safe log writing
log_lock = threading.Lock()
LOG_FILE = "events.log"

def thread_safe_log(event_dict):
    import json
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event_dict) + "\n")

def run_counter(mode, source, window_name):
    if mode == "people":
        from config.people_config import config as config_obj
        detector = YOLO(config_obj.YOLO_WEIGHTS)
        tracker = DeepSort(max_age=config_obj.DEEPSORT_MAX_AGE)
        counter = PeopleCounter(config_obj, detector, tracker)
    elif mode == "car":
        from config.car_config import config as config_obj
        detector = YOLO(config_obj.YOLO_WEIGHTS)
        tracker = DeepSort(max_age=config_obj.DEEPSORT_MAX_AGE)
        counter = CarCounter(config_obj, detector, tracker)
    else:
        print(f"Invalid mode: {mode}")
        return

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{source}' for {mode}")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"End of video or cannot read frame for {mode}.")
            break

        # Process frame and get tracks
        tracks, events = counter.process_frame(frame, return_events=True)
        # Log events (if any)
        for event in events:
            thread_safe_log(event)

        frame_with_overlay = counter.draw_overlay(frame, tracks)
        cv2.imshow(window_name, frame_with_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyWindow(window_name)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run people and car counting on two video feeds in parallel.")
    parser.add_argument("--people_source", required=True, help="Path to the video file for people counting.")
    parser.add_argument("--car_source", required=True, help="Path to the video file for car counting.")
    args = parser.parse_args()

    # Start both counters in separate threads
    people_thread = threading.Thread(target=run_counter, args=("people", args.people_source, "People Counter"))
    car_thread = threading.Thread(target=run_counter, args=("car", args.car_source, "Car Counter"))

    people_thread.start()
    car_thread.start()

    people_thread.join()
    car_thread.join()

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()