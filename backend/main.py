from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse, Response

import cv2
import json
import threading


app = FastAPI(title="ParkVision AI")

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ==================================================
# PATHS
# ==================================================

IMAGE_PATH = "data/images/parking_lot.png"
CLEAN_IMAGE_PATH = "data/images/parking_clean.png"
SPACES_PATH = "data/parking_spaces.json"
VIDEO_PATH = "data/videos/parking_video.mp4"


# ==================================================
# DETECTION SETTINGS
# ==================================================

DIFF_THRESHOLD = 25
OCCUPANCY_THRESHOLD = 0.08


# ==================================================
# PARKING SPACES
# ==================================================

with open(SPACES_PATH, "r") as file:
    PARKING_SPACES = json.load(file)


# ==================================================
# CLEAN BACKGROUND
# ==================================================

clean_background = cv2.imread(
    CLEAN_IMAGE_PATH
)


# ==================================================
# SHARED DATA
# ==================================================

latest_jpeg = None

latest_status = {
    "total": len(PARKING_SPACES),
    "occupied": 0,
    "available": len(PARKING_SPACES),
    "occupancy": 0,
    "spaces": []
}

data_lock = threading.Lock()


# ==================================================
# DETECTION
# ==================================================

def detect_frame(frame):

    occupied = 0
    available = 0

    spaces = []

    for i, space in enumerate(PARKING_SPACES):

        x1, y1, x2, y2 = space

        roi = frame[y1:y2, x1:x2]

        background_roi = clean_background[
            y1:y2,
            x1:x2
        ]

        current_gray = cv2.cvtColor(
            roi,
            cv2.COLOR_BGR2GRAY
        )

        background_gray = cv2.cvtColor(
            background_roi,
            cv2.COLOR_BGR2GRAY
        )

        difference = cv2.absdiff(
            current_gray,
            background_gray
        )

        _, mask = cv2.threshold(
            difference,
            DIFF_THRESHOLD,
            255,
            cv2.THRESH_BINARY
        )

        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5)
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel
        )

        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel
        )

        changed_ratio = (
            cv2.countNonZero(mask)
            / mask.size
        )

        if changed_ratio > OCCUPANCY_THRESHOLD:

            status = "Occupied"
            occupied += 1
            color = (0, 0, 255)

        else:

            status = "Available"
            available += 1
            color = (0, 255, 0)

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            3
        )

        cv2.putText(
            frame,
            f"P{i + 1}: {status}",
            (x1, y1 - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2
        )

        spaces.append({
            "id": f"P{i + 1}",
            "status": status
        })

    total = len(PARKING_SPACES)

    occupancy = round(
        (occupied / total) * 100
    )

    return (
        frame,
        spaces,
        occupied,
        available,
        occupancy
    )


# ==================================================
# VIDEO PROCESSOR
# ==================================================

def process_video():

    global latest_jpeg
    global latest_status

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print("ERROR: Could not open video.")

        return

    print("ParkVision AI video processor started.")

    while True:

        success, frame = cap.read()

        if not success:

            # Restart video
            cap.set(
                cv2.CAP_PROP_POS_FRAMES,
                0
            )

            continue

        # ------------------------------------------
        # DETECT CURRENT FRAME
        # ------------------------------------------

        (
            processed_frame,
            spaces,
            occupied,
            available,
            occupancy
        ) = detect_frame(frame)

        total = len(PARKING_SPACES)

        # ------------------------------------------
        # DRAW INFORMATION
        # ------------------------------------------

        cv2.rectangle(
            processed_frame,
            (10, 10),
            (275, 160),
            (10, 20, 25),
            -1
        )

        cv2.putText(
            processed_frame,
            f"Total: {total}",
            (25, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        cv2.putText(
            processed_frame,
            f"Occupied: {occupied}",
            (25, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )

        cv2.putText(
            processed_frame,
            f"Available: {available}",
            (25, 110),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.putText(
            processed_frame,
            f"Occupancy: {occupancy}%",
            (25, 145),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 0),
            2
        )

        # ------------------------------------------
        # ENCODE FRAME
        # ------------------------------------------

        success, encoded = cv2.imencode(
            ".jpg",
            processed_frame
        )

        if not success:
            continue

        jpeg_data = encoded.tobytes()

        # ------------------------------------------
        # SAVE LATEST DATA
        # ------------------------------------------

        with data_lock:

            latest_jpeg = jpeg_data

            latest_status = {
                "total": total,
                "occupied": occupied,
                "available": available,
                "occupancy": occupancy,
                "spaces": spaces
            }


# ==================================================
# START VIDEO PROCESSOR
# ==================================================

video_thread = threading.Thread(
    target=process_video,
    daemon=True
)

video_thread.start()


# ==================================================
# HOME
# ==================================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


# ==================================================
# IMAGE STATUS
# ==================================================

@app.get("/parking-status")
def parking_status():

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:

        return {
            "error": "Parking image not found."
        }

    (
        image,
        spaces,
        occupied,
        available,
        occupancy
    ) = detect_frame(image)

    return {
        "total": len(PARKING_SPACES),
        "occupied": occupied,
        "available": available,
        "occupancy": occupancy,
        "spaces": spaces
    }


# ==================================================
# PARKING IMAGE
# ==================================================
@app.get("/parking-image")
def parking_image():
    image = cv2.imread(IMAGE_PATH)

    if image is None:
        return {
            "error": "Parking image not found."
        }

    (
        image,
        spaces,
        occupied,
        available,
        occupancy
    ) = detect_frame(image)

    success, encoded = cv2.imencode(
        ".png",
        image
    )

    if not success:
        return {
            "error": "Could not process parking image."
        }

    return Response(
    content=encoded.tobytes(),
    media_type="image/png"
)
# ==================================================
# VIDEO STATUS
# ==================================================

@app.get("/video-parking-status")
def video_parking_status():

    with data_lock:

        return latest_status.copy()


# ==================================================
# VIDEO STREAM
# ==================================================

def generate_video_stream():

    while True:

        with data_lock:

            frame = latest_jpeg

        if frame is not None:

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n"
                + frame
                + b"\r\n"
            )


# ==================================================
# VIDEO ENDPOINT
# ==================================================

@app.get("/parking-video-stream")
def parking_video_stream():

    return StreamingResponse(
        generate_video_stream(),
        media_type=(
            "multipart/x-mixed-replace; "
            "boundary=frame"
        )
    )