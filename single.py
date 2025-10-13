# File: run_single_counter.py

import cv2
import argparse
import json
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Import your counter classes
from counters.people_counter import PeopleCounter
from counters.car_counter import CarCounter  # Make sure this class exists

# --- Configuration ---
LOG_FILE = "events.log"

# A simple, non-threaded logger
def log_event(event_dict):
    """Appends a log entry (as a JSON line) to the log file."""
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(event_dict) + "\n")

def main():
    """
    Main function to parse arguments and run the selected counter.
    """
    # 1. Set up command-line argument parser
    parser = argparse.ArgumentParser(description="Run a single object counter on a video feed.")
    parser.add_argument("--mode", 
                        required=True, 
                        choices=["people", "car"], 
                        help="The type of counter to run: 'people' or 'car'.")
    parser.add_argument("--source", 
                        required=True, 
                        help="Path to the video file or camera source.")
    args = parser.parse_args()

    # 2. Initialize the correct counter based on the --mode argument
    if args.mode == "people":
        from config.people_config import config as config_obj
        detector = YOLO(config_obj.YOLO_WEIGHTS)
        tracker = DeepSort(max_age=config_obj.DEEPSORT_MAX_AGE)
        counter = PeopleCounter(config_obj, detector, tracker)
        window_name = "People Counter"
    elif args.mode == "car":
        from config.car_config import config as config_obj
        detector = YOLO(config_obj.YOLO_WEIGHTS)
        tracker = DeepSort(max_age=config_obj.DEEPSORT_MAX_AGE)
        counter = CarCounter(config_obj, detector, tracker)
        window_name = "Car Counter"
    else:
        # This case is already handled by argparse `choices`, but it's good practice
        print(f"Error: Invalid mode '{args.mode}' provided.")
        return

    # 3. Set up video capture
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.source}'")
        return

    print(f"✅ Running {args.mode} counter. Press 'q' in the video window to quit.")

    # 4. Main processing loop
    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video stream or error reading frame.")
            break

        # Let the counter process the frame and get any events for logging
        tracks, events = counter.process_frame(frame, return_events=True)
        
        # Log all events that occurred in this frame
        for event in events:
            log_event(event)

        # Draw the overlay and display the result
        frame_with_overlay = counter.draw_overlay(frame, tracks)
        cv2.imshow(window_name, frame_with_overlay)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 5. Cleanup
    print("Exiting...")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()