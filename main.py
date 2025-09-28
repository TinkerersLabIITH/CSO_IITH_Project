import cv2
import argparse
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Import your counter classes
from counters.people_counter import PeopleCounter
from counters.car_counter import CarCounter

def main():
    parser = argparse.ArgumentParser(description="Run object counting for people or cars.")
    parser.add_argument("--mode", choices=["people", "car"], required=True, help="Select 'people' or 'car' counting mode.")
    parser.add_argument("--source", required=True, help="Path to the video file or camera index (e.g., '0').")
    args = parser.parse_args()

    if args.mode == "people":
        from config.people_config import config as config_obj
        detector = YOLO(config_obj.YOLO_WEIGHTS)
        tracker = DeepSort(max_age=config_obj.DEEPSORT_MAX_AGE)
        counter = PeopleCounter(config_obj, detector, tracker)

    elif args.mode == "car":
        from config.car_config import config as config_obj
        detector = YOLO(config_obj.YOLO_WEIGHTS)
        tracker = DeepSort(max_age=config_obj.DEEPSORT_MAX_AGE)
        counter = CarCounter(config_obj, detector, tracker)

    else:
        print("Invalid mode selected. Exiting.")
        return

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source '{args.source}'")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read frame.")
            break

        tracks = counter.process_frame(frame)
        frame_with_overlay = counter.draw_overlay(frame, tracks)
        cv2.imshow("Counter", frame_with_overlay)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()