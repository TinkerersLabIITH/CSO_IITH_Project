from counters.people_counter import PeopleCounter
from counters.car_counter import CarCounter

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["people", "car"], required=True)
    parser.add_argument("--source", required=True, help="Path to video or camera index")
    args = parser.parse_args()

    # Import config dynamically based on mode
    if args.mode == "people":
        from config.people_config import config as config_obj
        counter = PeopleCounter(config_obj)
    elif args.mode == "car":
        from config.car_config import config as config_obj
        counter = CarCounter(config_obj)

    counter.run(video_source=args.source)

if __name__ == "__main__":
    main()
