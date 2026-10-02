import cv2
import json
import os


IMAGE_PATH = "data/images/parking_clean.png"
SPACES_PATH = "data/parking_spaces.json"
OUTPUT_PATH = "data/videos/parking_video.mp4"

FPS = 8
DURATION = 30

os.makedirs("data/videos", exist_ok=True)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("ERROR: Could not load parking_clean.png")
    exit()

with open(SPACES_PATH, "r") as file:
    parking_spaces = json.load(file)

height, width = image.shape[:2]

print(f"Clean parking image loaded.")
print(f"Resolution: {width} x {height}")
print(f"Parking spaces loaded: {len(parking_spaces)}")


# ------------------------------------------------------------
# CAR COLORS
# ------------------------------------------------------------

CAR_COLORS = [
    (235, 235, 235),   # white
    (40, 55, 180),     # red
    (55, 65, 75),      # dark gray
    (45, 80, 140),     # blue
    (180, 190, 195),   # silver
]


# ------------------------------------------------------------
# DRAW CAR
# ------------------------------------------------------------

def draw_car(frame, space, color):

    x1, y1, x2, y2 = space

    w = x2 - x1
    h = y2 - y1

    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    car_w = int(w * 0.52)
    car_h = int(h * 0.72)

    left = cx - car_w // 2
    right = cx + car_w // 2
    top = cy - car_h // 2
    bottom = cy + car_h // 2

    # Shadow
    cv2.ellipse(
        frame,
        (cx, cy + 8),
        (car_w // 2, car_h // 2),
        0,
        0,
        360,
        (35, 35, 35),
        -1
    )

    # Main vehicle body
    cv2.ellipse(
        frame,
        (cx, top + car_h // 4),
        (car_w // 2, car_h // 4),
        0,
        180,
        360,
        color,
        -1
    )

    cv2.rectangle(
        frame,
        (left, top + car_h // 4),
        (right, bottom - car_h // 4),
        color,
        -1
    )

    cv2.ellipse(
        frame,
        (cx, bottom - car_h // 4),
        (car_w // 2, car_h // 4),
        0,
        0,
        180,
        color,
        -1
    )

    # Black outline
    cv2.ellipse(
        frame,
        (cx, cy),
        (car_w // 2, car_h // 2),
        0,
        0,
        360,
        (25, 25, 25),
        3
    )

    # Windows
    window_color = (55, 70, 78)

    cv2.ellipse(
        frame,
        (cx, top + int(car_h * 0.27)),
        (int(car_w * 0.32), int(car_h * 0.13)),
        0,
        180,
        360,
        window_color,
        -1
    )

    cv2.ellipse(
        frame,
        (cx, bottom - int(car_h * 0.27)),
        (int(car_w * 0.32), int(car_h * 0.13)),
        0,
        0,
        180,
        window_color,
        -1
    )

    # Window reflection
    cv2.line(
        frame,
        (cx - int(car_w * 0.20), top + int(car_h * 0.25)),
        (cx + int(car_w * 0.20), top + int(car_h * 0.25)),
        (150, 175, 185),
        2
    )

    # Wheels
    wheel_color = (15, 15, 15)

    for wx in [
        left + int(car_w * 0.10),
        right - int(car_w * 0.10)
    ]:

        cv2.ellipse(
            frame,
            (wx, top + int(car_h * 0.35)),
            (5, 12),
            0,
            0,
            360,
            wheel_color,
            -1
        )

        cv2.ellipse(
            frame,
            (wx, bottom - int(car_h * 0.35)),
            (5, 12),
            0,
            0,
            360,
            wheel_color,
            -1
        )

    # Headlights
    for lx in [
        cx - int(car_w * 0.23),
        cx + int(car_w * 0.23)
    ]:

        cv2.circle(
            frame,
            (lx, top + int(car_h * 0.10)),
            5,
            (220, 235, 245),
            -1
        )

    # Tail lights
    for lx in [
        cx - int(car_w * 0.23),
        cx + int(car_w * 0.23)
    ]:

        cv2.circle(
            frame,
            (lx, bottom - int(car_h * 0.10)),
            5,
            (40, 40, 200),
            -1
        )


# ------------------------------------------------------------
# PARKING SCENARIOS
# ------------------------------------------------------------

scenarios = [

    # 0 seconds
    [1, 2, 4, 5, 6, 8, 10],

    # 5 seconds - P3 enters
    [1, 2, 3, 4, 5, 6, 8, 10],

    # 10 seconds - P2 leaves
    [1, 3, 4, 5, 6, 8, 10],

    # 15 seconds - P7 enters
    [1, 3, 4, 5, 6, 7, 8, 10],

    # 20 seconds - P4 leaves
    [1, 3, 5, 6, 7, 8, 10],

    # 25 seconds - P9 enters
    [1, 3, 5, 6, 7, 8, 9, 10],
]


# ------------------------------------------------------------
# VIDEO WRITER
# ------------------------------------------------------------

fourcc = cv2.VideoWriter_fourcc(*"mp4v")

video = cv2.VideoWriter(
    OUTPUT_PATH,
    fourcc,
    FPS,
    (width, height)
)

if not video.isOpened():
    print("ERROR: Could not create video.")
    exit()


# ------------------------------------------------------------
# GENERATE VIDEO
# ------------------------------------------------------------

total_frames = FPS * DURATION

print()
print("Creating fast parking simulation...")
print()

for frame_number in range(total_frames):

    current_time = frame_number / FPS

    scenario_index = min(
        int(current_time // 5),
        len(scenarios) - 1
    )

    occupied_spaces = scenarios[scenario_index]

    frame = image.copy()

    # Draw cars
    for index, space_number in enumerate(occupied_spaces):

        color = CAR_COLORS[
            index % len(CAR_COLORS)
        ]

        draw_car(
            frame,
            parking_spaces[space_number - 1],
            color
        )

    # Detection boxes
    for i, space in enumerate(parking_spaces):

        x1, y1, x2, y2 = space

        number = i + 1

        if number in occupied_spaces:
            color = (0, 0, 255)
            status = "Occupied"
        else:
            color = (0, 255, 0)
            status = "Available"

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            frame,
            f"P{number}: {status}",
            (x1 + 8, y1 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            color,
            2
        )

    # Statistics
    total = len(parking_spaces)
    occupied = len(occupied_spaces)
    available = total - occupied
    occupancy = int((occupied / total) * 100)

    # Header
    cv2.rectangle(
        frame,
        (0, 0),
        (width, 65),
        (8, 20, 25),
        -1
    )

    cv2.putText(
        frame,
        "ParkVision AI",
        (18, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.70,
        (0, 255, 180),
        2
    )

    cv2.putText(
        frame,
        f"Occupied: {occupied}",
        (240, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 0, 255),
        2
    )

    cv2.putText(
        frame,
        f"Available: {available}",
        (400, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (0, 255, 0),
        2
    )

    cv2.putText(
        frame,
        f"Occupancy: {occupancy}%",
        (585, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (255, 255, 255),
        2
    )

    cv2.putText(
        frame,
        "LIVE SIMULATION",
        (width - 220, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (0, 255, 180),
        2
    )

    video.write(frame)

    # Progress
    if frame_number % (FPS * 5) == 0:
        print(
            f"Progress: {int((frame_number / total_frames) * 100)}%"
        )


video.release()

print()
print("================================")
print("VIDEO CREATED SUCCESSFULLY")
print("================================")
print()
print(f"Saved to: {OUTPUT_PATH}")