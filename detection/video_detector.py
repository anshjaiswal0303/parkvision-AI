import cv2
import json

VIDEO_PATH = "data/videos/parking_video.mp4"
SPACES_PATH = "data/parking_spaces.json"

THRESHOLD = 0.08


def detect_frame(frame, parking_spaces):
    occupied = 0
    available = 0

    results = []

    for i, space in enumerate(parking_spaces):

        x1, y1, x2, y2 = space

        roi = frame[y1:y2, x1:x2]

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(
            gray,
            50,
            150
        )

        edge_density = (edges > 0).mean()

        if edge_density > THRESHOLD:

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

        results.append({
            "id": f"P{i + 1}",
            "status": status
        })

    return frame, results, occupied, available


def main():

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():

        print("Error: Could not open parking video.")

        return

    with open(SPACES_PATH, "r") as file:

        parking_spaces = json.load(file)

    print("ParkVision AI")
    print("-------------------------")
    print("Real-time OpenCV detection started.")
    print("Press Q to stop.")
    print("-------------------------")

    while True:

        success, frame = cap.read()

        if not success:

            print("Video ended.")

            break

        frame, spaces, occupied, available = detect_frame(
            frame,
            parking_spaces
        )

        total = len(parking_spaces)

        occupancy = round(
            (occupied / total) * 100
        )

        cv2.putText(
            frame,
            f"Total: {total}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"Occupied: {occupied}",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2
        )

        cv2.putText(
            frame,
            f"Available: {available}",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.putText(
            frame,
            f"Occupancy: {occupancy}%",
            (20, 140),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 0),
            2
        )

        cv2.imshow(
            "ParkVision AI - Real Detection",
            frame
        )

        key = cv2.waitKey(100)

        if key == ord("q"):

            break

    cap.release()

    cv2.destroyAllWindows()


if __name__ == "__main__":

    main()