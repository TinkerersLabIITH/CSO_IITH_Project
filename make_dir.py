import os

# Define project structure
structure = {
    "config": ["people_config.py", "car_config.py"],
    "detectors": ["yolo_detector.py"],
    "trackers": ["deep_sort_tracker.py"],
    "counters": ["base_counter.py", "human_counter.py", "car_counter.py"],
    "utils": ["geometry.py", "nms.py", "appearance.py"],
}

# Root project directory (you can change this)
root = "object_counter_project"

def make_structure(root, structure):
    os.makedirs(root, exist_ok=True)

    for folder, files in structure.items():
        folder_path = os.path.join(root, folder)
        os.makedirs(folder_path, exist_ok=True)

        for f in files:
            file_path = os.path.join(folder_path, f)
            if not os.path.exists(file_path):
                with open(file_path, "w") as fp:
                    fp.write("# " + f + "\n")

    # Create main.py at root
    main_path = os.path.join(root, "main.py")
    if not os.path.exists(main_path):
        with open(main_path, "w") as fp:
            fp.write("""\
# main.py
# Entry point for running counters

if __name__ == "__main__":
    print("Run with people_config or car_config")
""")

if __name__ == "__main__":
    make_structure(root, structure)
    print(f"Project structure created inside '{root}'")
