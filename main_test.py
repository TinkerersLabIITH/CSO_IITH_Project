import cv2
import threading
import json
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from counters.people_counter import PeopleCounter
from counters.car_counter import CarCounter

# Thread-safe log writing
log_lock = threading.Lock()
LOG_FILE = "events.log"

def thread_safe_log(event_dict):
    """Safely appends a JSON event record to the log file."""
    with log_lock:
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event_dict) + "\n")

def run_counter(mode, source, window_name):
    """
    Initializes and runs a counter on a given video source.
    Includes logic to automatically reconnect if the stream is lost.
    """
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
        print(f"Error: Invalid mode '{mode}'")
        return

    # --- Reconnection Loop ---
    while True:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"Error: Could not open video source '{source}' for {mode}. Retrying in 10 seconds...")
            time.sleep(10)
            continue  # Attempt to reconnect

        print(f"Successfully connected to source for {mode}: {source}")

        # --- Frame Reading Loop ---
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"Stream for '{mode}' ended or disconnected. Attempting to reconnect...")
                break  # Exit inner loop to trigger reconnection

            # Process frame, get tracks, and log events
            tracks, events = counter.process_frame(frame, return_events=True)
            for event in events:
                thread_safe_log(event)

            # Draw overlay and display the frame
            frame_with_overlay = counter.draw_overlay(frame, tracks)
            cv2.imshow(window_name, frame_with_overlay)

            # Check for 'q' to quit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cap.release()
                cv2.destroyWindow(window_name)
                print(f"Counter '{mode}' stopped by user.")
                return # Exit the function and thread completely

        # Clean up before next reconnection attempt
        cap.release()
        time.sleep(5) # Wait a bit before trying to reconnect

def main():
    """
    Loads configuration from config.json and starts the counting threads.
    """
    # Load configuration from the JSON file
    try:
        with open("config.json", "r") as f:
            config = json.load(f)
        people_cam_url = config["sources"]["people_camera"]
        car_cam_url = config["sources"]["car_camera"]
    except FileNotFoundError:
        print("Error: config.json not found. Please create it and add your camera URLs.")
        return
    except KeyError:
        print("Error: Invalid format in config.json. Ensure 'sources', 'people_camera', and 'car_camera' keys exist.")
        return

    # Start both counters in separate threads
    people_thread = threading.Thread(target=run_counter, args=("people", people_cam_url, "People Counter"))
    car_thread = threading.Thread(target=run_counter, args=("car", car_cam_url, "Car Counter"))

    print("Starting people and car counter threads...")
    people_thread.start()
    car_thread.start()

    # Wait for both threads to complete (they will run until 'q' is pressed in one window)
    people_thread.join()
    car_thread.join()

    print("All threads have finished. Exiting.")
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()