# Import libraries
import cv2 as cv
import numpy as np
import sys

# Load the image in color
colored_image = cv.imread("image.jpg")

# Check if image is loaded
if colored_image is None:
    sys.exit("Could not read the image")

# Create an image to be printed after every step
printed_image = colored_image

# Display the image
cv.imshow("Imported image", printed_image)

# Wait for a key press
cv.waitKey(0)

# Convert the image to grayscale
grayscale_image = cv.cvtColor(colored_image, cv.COLOR_BGR2GRAY)

# Apply a binary threshold to the grayscale image
_, grayscale_mask = cv.threshold(grayscale_image, 127, 255, cv.THRESH_BINARY)

# Update the image to be printed after every step
printed_image = grayscale_image

# Display the image
cv.imshow("Grayscale image", printed_image)

# Wait for a key press
cv.waitKey(0)

# Convert the image to HSV
hsv_image = cv.cvtColor(colored_image, cv.COLOR_BGR2HSV)

# Update the image to be printed after every step
printed_image = hsv_image

# Display the image
cv.imshow("HSV image", printed_image)

# Wait for a key press
cv.waitKey(0)

# Calculate the histogram of the hue channel
hist = cv.calcHist([hsv_image], [0], None, [180], [0, 180])

# Find the hue with the highest count
most_present_hue = np.argmax(hist)

# Define the lower and upper bounds for the color to be detected
lower_bound = np.array([most_present_hue - 20, 100, 100])
upper_bound = np.array([most_present_hue + 20, 255, 255])

# Threshold the HSV image to get only the colors in the specified range
hsv_mask = cv.inRange(hsv_image, lower_bound, upper_bound)

# Bitwise-AND the grayscale mask and the HSV mask
combined_mask = cv.bitwise_and(grayscale_mask, hsv_mask)

# Bitwise-AND the mask and the original image
resulting_image = cv.bitwise_and(colored_image, colored_image, mask=combined_mask)

# Update the image to be printed after every step
printed_image = resulting_image

# Display the image
cv.imshow("Masked image", printed_image)

# Wait for a key press
cv.waitKey(0)

# Close the window
cv.destroyAllWindows()
