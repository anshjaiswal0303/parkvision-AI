import cv2

# Load the parking lot image
image = cv2.imread("data/images/parking_lot.png")

# Check if the image was loaded
if image is None:
    print("Error: Could not load the image.")
else:
    print("Image loaded successfully!")
    print("Image size:", image.shape)

    # Display the image
    cv2.imshow("ParkVision AI - Parking Lot", image)

    # Wait until a key is pressed
    cv2.waitKey(0)

    # Close the window
    cv2.destroyAllWindows()