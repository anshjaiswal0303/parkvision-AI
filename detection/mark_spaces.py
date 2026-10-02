import cv2
import json

IMAGE_PATH = "data/images/parking_lot.png"
OUTPUT_PATH = "data/parking_spaces.json"

image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Could not load parking image.")
    exit()

# Resize image so it fits on the screen
max_width = 1200
scale = min(1, max_width / image.shape[1])

display_image = cv2.resize(
    image,
    None,
    fx=scale,
    fy=scale
)

points = []
parking_spaces = []

window_name = "ParkVision AI - Mark Parking Spaces"

print("\nInstructions:")
print("1. Click the TOP-LEFT corner of a parking space.")
print("2. Click the BOTTOM-RIGHT corner.")
print("3. Repeat for every parking space.")
print("4. Press Q when finished.")
print("5. Press R to reset.\n")


def mouse_callback(event, x, y, flags, param):

    global points, display_image

    if event == cv2.EVENT_LBUTTONDOWN:

        points.append((x, y))

        # Draw clicked point
        cv2.circle(display_image, (x, y), 5, (0, 255, 255), -1)

        # Two points = one parking space
        if len(points) == 2:

            x1, y1 = points[0]
            x2, y2 = points[1]

            # Convert coordinates back to original image size
            original_x1 = int(x1 / scale)
            original_y1 = int(y1 / scale)
            original_x2 = int(x2 / scale)
            original_y2 = int(y2 / scale)

            parking_spaces.append([
                original_x1,
                original_y1,
                original_x2,
                original_y2
            ])

            # Draw rectangle
            cv2.rectangle(
                display_image,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )

            # Label parking space
            slot_number = len(parking_spaces)

            cv2.putText(
                display_image,
                f"P{slot_number}",
                (x1 + 10, y1 + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            print(
                f"Parking Space {slot_number}: "
                f"{parking_spaces[-1]}"
            )

            points = []


cv2.namedWindow(window_name)
cv2.setMouseCallback(window_name, mouse_callback)

while True:

    cv2.imshow(window_name, display_image)

    key = cv2.waitKey(1) & 0xFF

    # Reset
    if key == ord("r"):

        points = []
        parking_spaces = []

        display_image = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale
        )

        print("Reset complete.")

    # Finish
    elif key == ord("q"):
        break


cv2.destroyAllWindows()

# Save parking-space coordinates
with open(OUTPUT_PATH, "w") as file:
    json.dump(parking_spaces, file, indent=4)

print("\nParking spaces saved successfully!")
print(f"Total spaces marked: {len(parking_spaces)}")
print(f"Saved to: {OUTPUT_PATH}")