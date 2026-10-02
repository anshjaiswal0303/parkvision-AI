import cv2
import json

IMAGE_PATH = "data/images/parking_lot.png"
SPACES_PATH = "data/parking_spaces.json"

# Load image
image = cv2.imread(IMAGE_PATH)

if image is None:
    print("Error: Could not load parking image.")
    exit()

# Load parking spaces
with open(SPACES_PATH, "r") as file:
    parking_spaces = json.load(file)

occupied = 0
available = 0

# Edge-density threshold
THRESHOLD = 0.08

for i, space in enumerate(parking_spaces):

    x1, y1, x2, y2 = space

    # Crop parking space
    roi = image[y1:y2, x1:x2]

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edges = cv2.Canny(gray, 50, 150)

    # Calculate percentage of edge pixels
    edge_density = (edges > 0).mean()

    # Determine occupancy
    if edge_density > THRESHOLD:
        occupied += 1
        color = (0, 0, 255)
        status = "Occupied"
    else:
        available += 1
        color = (0, 255, 0)
        status = "Available"

    # Draw rectangle
    cv2.rectangle(
        image,
        (x1, y1),
        (x2, y2),
        color,
        2
    )

    # Display status
    cv2.putText(
        image,
        f"P{i + 1}: {status}",
        (x1, y1 - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        color,
        2
    )

    print(
        f"P{i + 1}: {status} | "
        f"Edge Density: {edge_density:.3f}"
    )


# Display totals
total = len(parking_spaces)

print("\n-------------------------")
print(f"Total Spaces : {total}")
print(f"Occupied     : {occupied}")
print(f"Available    : {available}")
print("-------------------------")
# Add dashboard header
cv2.rectangle(
    image,
    (0, 0),
    (image.shape[1], 80),
    (40, 40, 40),
    -1
)

cv2.putText(
    image,
    "PARKVISION AI",
    (30, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    1,
    (255, 255, 255),
    2
)

summary = f"Total: {total}   Occupied: {occupied}   Available: {available}"

cv2.putText(
    image,
    summary,
    (350, 35),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.8,
    (255, 255, 255),
    2
)
cv2.imshow("ParkVision AI - Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()