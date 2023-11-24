# Import libraries
import cv2 as cv
from cv2.typing import MatLike
import numpy as np
from screeninfo import get_monitors
import sys


# Get the resolution of the monitor
monitor = get_monitors()[0]
monitor_width = monitor.width
monitor_height = monitor.height


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

    # Calculate the scaling factors
    scale_width = monitor_width / image.shape[1]
    scale_height = monitor_height / image.shape[0]
    scale = min(scale_width, scale_height)

    # Calculate the new size of the image
    new_width = int(image.shape[1] * scale)
    new_height = int(image.shape[0] * scale)

    # Resize the image to match the resolution of the monitor
    resized_image = cv.resize(image, (round(new_width / 2), round(new_height / 2)))

    # Display the image
    cv.imshow(window_name, resized_image)

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


while True:
    # Create a variable to store the original image
    colored_image: MatLike | None = None

    # Prompt the user for input
    user_input = input(
        "Do you want to use the camera or use a saved image?\nEnter 'y' for using the camera, 'n' for importing an image, or 'Esc' to exit the program: "
    )

    # Print a newline
    print()

    # Check if the user pressed y
    if user_input.lower() == "y":
        # Open the camera
        camera = cv.VideoCapture(0)

        # Check if camera is opened
        if camera is None or not camera.isOpened():
            # Exit the program
            sys.exit("Unable to open the camera!")

        # Loop until the user presses the Esc key or spacebar
        while True:
            # Read a frame from the camera
            success, frame = camera.read()

            # Check if frame is not empty
            if not success:
                # Exit the program
                sys.exit("Unable to read a frame from the camera!")

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
                sys.exit("Program ended!")

        # Release the camera
        camera.release()

    elif user_input.lower() == "n":
        # Print a message for the user
        print(
            "Please insert the name of the image you want to use.\nMake sure to include the file extension (e.g. '.jpg')!"
        )

        # Wait for user input
        image_name = input()

        # Print a newline
        print()

        # Load the image in color
        # colored_image = cv.imread("Amoebe_1.jpg")
        # colored_image = cv.imread("Amoebe_2.jpg")
        try:
            colored_image = cv.imread(image_name)
        except:
            # Print an error message
            print(f"Unable to find the image '{image_name}'!\n")

            # Close all windows
            cv.destroyAllWindows()

            # Continue to the next iteration
            continue

    # Check if the user pressed the Esc key
    elif user_input.lower() == "esc" or user_input.lower() == "exit":
        # Exit the program
        sys.exit("Program ended!")

    else:
        # Print a warning message
        print("Please enter 'y', 'n' or 'Esc'!\n")

        # Close all windows
        cv.destroyAllWindows()

        # Continue to the next iteration
        continue

    # Check if image is loaded
    if colored_image is None:
        # Print an error message
        print("Unable to read the image\n")

        # Close all windows
        cv.destroyAllWindows()

        # Continue to the next iteration
        continue

    # Display the image
    if not show_image(colored_image, "Processing image...", timer=2000):
        # Close the window
        close_window("Processing image...")

        # Print a newline
        print()

        # Continue to the next iteration
        continue

    # Convert the image to HSV
    hsv_image = cv.cvtColor(colored_image, cv.COLOR_BGR2HSV)

    # Display the image
    if not show_image(hsv_image, "Processing image...", timer=2000):
        # Close the window
        close_window("Processing image...")

        # Print a newline
        print()

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
        # Close the window
        close_window("Processing image...")

        # Print a newline
        print()

        # Continue to the next iteration
        continue

    # Find contours in the threshold image
    contours, _ = cv.findContours(hsv_mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Calculate the area of the largest contour
    max_area = max(cv.contourArea(cnt) for cnt in contours)

    if max_area < colored_image.shape[0] * colored_image.shape[1] * 0.1:
        # Print a warning message
        print(
            "Unable to find the object!\nPlease provide a better image with more contrast!\n"
        )

        # Close all windows
        cv.destroyAllWindows()

        # Continue to the next iteration
        continue

    # Create an empty mask to store the contours
    contour_mask = np.zeros_like(colored_image)

    # Draw the contours on the mask
    cv.drawContours(
        contour_mask, contours, -1, (255, 255, 255, 255), thickness=cv.FILLED
    )

    # Display the image
    if not show_image(
        cv.bitwise_and(
            hsv_image,
            hsv_image,
            mask=cv.cvtColor(contour_mask, cv.COLOR_BGR2GRAY).astype("uint8"),
        ),
        "Processing image...",
        timer=2000,
    ):
        # Close the window
        close_window("Processing image...")

        # Print a newline
        print()

        # Continue to the next iteration
        continue

    # Create a 4-channel image from the mask
    transparent_mask = cv.cvtColor(contour_mask, cv.COLOR_BGR2BGRA)

    # Set the alpha channel to 0 for the background and 255 for the object
    transparent_mask[:, :, 3] = np.where(transparent_mask[:, :, 0] <= 0, 0, 255).astype(
        np.uint8
    )

    # Apply the mask to the image
    masked_image = cv.bitwise_and(
        cv.cvtColor(colored_image, cv.COLOR_BGR2BGRA), transparent_mask
    )

    # Display the image
    if not show_image(
        masked_image,
        "Processing image...",
        timer=2000,
    ):
        # Close the window
        close_window("Processing image...")

        # Print a newline
        print()

        # Continue to the next iteration
        continue

    # Define the center coordinates
    cx, cy = 0, 0

    # Loop over each pixel in the result image
    for y in range(masked_image.shape[0]):
        for x in range(masked_image.shape[1]):
            # Check if the pixel is not transparent
            if masked_image[y, x, 3] != 0:
                # Update the centroid coordinates
                cx += x
                cy += y

    # Convert the center coordinates to integers
    rounded_cx = int(cx / max_area)
    rounded_cy = int(cy / max_area)

    # Calculate the color of the cross to have a good contrast with the background
    cross_color = [
        255 - int(pixel.mean())
        for pixel in cv.split(masked_image[rounded_cy, rounded_cx])
    ]

    # Calculate the diagonal length of the image
    diagonal_length = np.sqrt(masked_image.shape[0] ** 2 + masked_image.shape[1] ** 2)

    # Define the size of the cross as a percentage of the diagonal length
    cross_size = int((diagonal_length * 0.02) / 2)

    # Draw a cross at the center of gravity
    cv.line(
        masked_image,
        (rounded_cx - cross_size, rounded_cy + cross_size),
        (rounded_cx + cross_size, rounded_cy - cross_size),
        cross_color,
        round(cross_size / 2),
    )
    cv.line(
        masked_image,
        (rounded_cx - cross_size, rounded_cy - cross_size),
        (rounded_cx + cross_size, rounded_cy + cross_size),
        cross_color,
        round(cross_size / 2),
    )

    # Display the image
    if not show_image(
        masked_image,
        "Processing image...",
    ):
        # Close the window
        close_window("Processing image...")

        # Print a newline
        print()

        # Continue to the next iteration
        continue

    # Print the coordinates of the centroid
    print(
        f"The coordinates of the center of gravity are: ({rounded_cx}, {rounded_cy})!\n"
    )

    # Ask the user if they want to save the image
    print(
        "Do you want to save the image?\nPress 'y' to save the image or enter anything else to not do so!"
    )

    # Wait for user input
    pressed_key = cv.waitKey(0)

    if pressed_key == ord("y"):
        # Prompt the user for input
        user_input = input("Please insert the name of the image you want to save: ")

        # Save the image
        cv.imwrite(f"{user_input}.png", masked_image)

        # Print a newline
        print()

    # Close all windows
    cv.destroyAllWindows()
