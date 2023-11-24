# Import libraries
import cv2 as cv
from cv2.typing import MatLike
import numpy as np
import sys


"""
Define a function to display an image and wait for a key press or timer to expire
"""


def show_image(
    image: MatLike,
    window_name: str,
    default_return: bool | None = True,
    timer: int = 0,
    custom_suffix: bool = False,
) -> bool | None:
    # Check if the suffix should be custom
    if not custom_suffix:
        # Add a suffix to the window name
        window_name = window_name + " (Press the spacebar to continue or Esc to exit)"

    # Display the image
    cv.imshow(window_name, image)

    # Wait for a key press
    pressed_key = cv.waitKey(timer)

    # Check if the user pressed the Esc key
    if pressed_key == 27:
        # Return False
        return False

        # Check if the user pressed the spacebar
    elif pressed_key == 32:
        # Return True
        return True

    # Release function if no key is pressed
    else:
        # Return default return value
        return default_return


"""
Define a function to close a window with a specific name (and suffix)
"""


def close_window(
    window_name: str,
    custom_suffix: bool = False,
) -> None:
    # Check if the suffix should be custom
    if not custom_suffix:
        # Add a suffix to the window name
        window_name = window_name + " (Press the spacebar to continue or Esc to exit)"

    # Close the window
    cv.destroyWindow(window_name)


"""
Read an image from the camera on key press and display it
"""

while True:
    # Open the camera
    camera = cv.VideoCapture(0)

    # Check if camera is opened
    if camera is None or not camera.isOpened():
        # Exit the program
        sys.exit("Could not open the camera")

    colored_image: MatLike | None = None

    # Loop until the user presses the Esc key or spacebar
    while True:
        # Read a frame from the camera
        success, frame = camera.read()

        # Check if frame is not empty
        if not success:
            # Exit the program
            sys.exit("Could not read a frame from the camera")

        return_value = show_image(
            frame,
            "Camera (Press spacebar to take a photo or Esc to exit the program)",
            None,
            1,
            True,
        )

        # Display the frame
        if return_value:
            # Save the frame
            colored_image = frame

            # Close the window
            close_window(
                "Camera (Press spacebar to take a photo or Esc to exit the program)",
                True,
            )

            # Exit the loop
            break

        elif return_value is False:
            # Exit the program
            sys.exit("The program was closed")

    # Release the camera
    camera.release()

    # Load the image in color
    colored_image = cv.imread("image.jpg")

    # Check if image is loaded
    if colored_image is None:
        # Print an error message
        print("Could not read the image")

        # Continue to the next iteration
        continue

    # Display the image
    if not show_image(colored_image, "Processing image...", timer=2000):
        # Continue to the next iteration
        continue

    # Convert the image to HSV
    hsv_image = cv.cvtColor(colored_image, cv.COLOR_BGR2HSV)

    # Display the image
    if not show_image(hsv_image, "Processing image...", timer=2000):
        # Continue to the next iteration
        continue

    # Calculate the histogram of the hue channel
    histogram = cv.calcHist([hsv_image], [0], None, [180], [0, 180])

    # Find the hue with the highest count
    most_present_hue = np.argmax(histogram)

    # Define the lower and upper bounds for the color to be detected
    lower_bound = np.array([most_present_hue - 20, 100, 100])
    upper_bound = np.array([most_present_hue + 20, 255, 255])

    # Threshold the HSV image to get only the colors in the specified range
    hsv_mask = cv.inRange(hsv_image, lower_bound, upper_bound)

    # Display the image
    if not show_image(
        cv.bitwise_and(hsv_image, hsv_image, mask=hsv_mask),
        "Processing image...",
        timer=2000,
    ):
        # Continue to the next iteration
        continue

    # Find contours in the threshold image
    contours, _ = cv.findContours(hsv_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Create an empty mask to store the outline
    contour_mask = np.zeros_like(colored_image)

    # Draw the contours on the mask
    cv.drawContours(
        contour_mask, contours, -1, (255, 255, 255, 255), thickness=cv.FILLED
    )

    # Create a 4-channel image from the mask
    transparent_mask = cv.cvtColor(contour_mask, cv.COLOR_BGR2BGRA)

    # Set the alpha channel to 0 for the background and 255 for the object
    transparent_mask[:, :, 3] = np.where(transparent_mask[:, :, 0] <= 0, 0, 255).astype(
        np.uint8
    )

    # Invert the mask
    inverted_mask = cv.bitwise_not(transparent_mask)

    # Create a 4-channel image from the mask
    inverted_mask = cv.cvtColor(inverted_mask, cv.COLOR_BGR2BGRA)

    # Set the alpha channel to 0 for the background and 255 for the object
    inverted_mask[:, :, 3] = np.where(inverted_mask[:, :, 0] > 0, 0, 255).astype(
        np.uint8
    )

    # Apply the mask to the image
    inverted_mask_image = cv.bitwise_and(
        cv.cvtColor(colored_image, cv.COLOR_BGR2BGRA), inverted_mask
    )

    # Apply the mask to the image
    masked_image = cv.bitwise_and(
        cv.cvtColor(colored_image, cv.COLOR_BGR2BGRA), transparent_mask
    )

    # Calculate the area of the largest contour
    max_area = max(cv.contourArea(cnt) for cnt in contours)

    cx, cy = 0, 0

    # Loop over each pixel in the result image
    for y in range(masked_image.shape[0]):
        for x in range(masked_image.shape[1]):
            # Check if the pixel is not transparent
            if masked_image[y, x, 3] != 0:
                # Update the centroid coordinates
                cx += x
                cy += y

    center = (cx / max_area, cy / max_area)

    # Convert the center coordinates to integers
    rounded_cx = int(center[0])
    rounded_cy = int(center[1])

    # Calculate the average color of the masked image
    average_color_per_row = np.average(inverted_mask_image[..., :3], axis=0)
    average_color = np.average(average_color_per_row, axis=0)

    # Calculate the opposite color
    opposite_color = [255 - x for x in average_color]

    # Convert the opposite color to integers
    opposite_color = [int(x) for x in opposite_color]

    # Define the size of the cross
    cross_size = 5

    # Draw a cross at the center of gravity
    cv.line(
        masked_image,
        (rounded_cx - cross_size, rounded_cy + cross_size),
        (rounded_cx + cross_size, rounded_cy - cross_size),
        opposite_color,
        round(cross_size / 2),
    )
    cv.line(
        masked_image,
        (rounded_cx - cross_size, rounded_cy - cross_size),
        (rounded_cx + cross_size, rounded_cy + cross_size),
        opposite_color,
        round(cross_size / 2),
    )

    print(center)

    cv.imshow("Original image", colored_image)
    cv.imshow("HSV mask", hsv_mask)
    cv.imshow("Resulting image", masked_image)
    cv.waitKey(0)

    # Close all windows
    cv.destroyAllWindows()
