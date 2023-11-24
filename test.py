import cv2
import numpy as np

# Load the image in color
image_color = cv2.imread("image.jpg")

# Convert the image to grayscale
image = cv2.cvtColor(image_color, cv2.COLOR_BGR2GRAY)

# Apply a binary threshold to the image
_, threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

"""
# Apply adaptive thresholding to the image
threshold = cv2.adaptiveThreshold(
    image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 5
)"""

# Define a kernel for the morphological operation
kernel = np.ones((10, 10), np.uint8)

# Perform the opening operation
opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN, kernel)

# Perform the closing operation
closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

# Find contours in the threshold image
contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Create an empty mask to store the outline
mask = np.zeros_like(image)

# Draw the contours on the mask
cv2.drawContours(mask, contours, -1, (255, 255, 255, 255), thickness=cv2.FILLED)

# Invert the mask
mask = cv2.bitwise_not(mask)

# Create a 4-channel image from the mask
mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGRA)

# Set the alpha channel to 0 for the background and 255 for the object
mask[:, :, 3] = np.where(mask[:, :, 0] > 0, 0, 255).astype(np.uint8)

# Apply the mask to the image
result = cv2.bitwise_and(cv2.cvtColor(image_color, cv2.COLOR_BGR2BGRA), mask)

# Find the bounding rectangle of the largest contour
x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))


# Find the largest contour and its moments
largest_contour = max(contours, key=cv2.contourArea)
M = cv2.moments(largest_contour)

# Calculate the centroid of the largest contour
cX = int(M["m10"] / M["m00"])
cY = int(M["m01"] / M["m00"])

# Draw a circle at the centroid
cv2.circle(result, (cX, cY), 5, (255, 255, 255, 255), -1)

# Display the coordinates of the centroid
print("Centroid coordinates: ({}, {})".format(cX, cY))

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Approximate the contour to smooth it
epsilon = 0.01 * cv2.arcLength(largest_contour, True)
smooth_contour = cv2.approxPolyDP(largest_contour, epsilon, True)

# Draw the smoothed contour on a blank image
mask_smooth = np.zeros_like(image)
cv2.drawContours(
    mask_smooth, [smooth_contour], -1, (255, 255, 255), thickness=cv2.FILLED
)

# Initialize a 3D list to store the distances
distances = []

# Loop over each pixel in the result image
for y in range(result.shape[0]):
    for x in range(result.shape[1]):
        # Check if the pixel is not transparent
        if result[y, x, 3] != 0:
            # Calculate the distance to the origin
            distance = np.sqrt(x**2 + y**2)
            # Append the coordinates and distance to the list
            distances.append([x, y, distance])

# Convert the list to a numpy array
distances = np.array(distances)

# Print the distances array
print(distances)

# Print the number of pixels in the outlined object
print("Number of pixels in the outlined object: {}".format(len(distances)))

# Find the largest contour
largest_contour = max(contours, key=cv2.contourArea)

# Calculate the area of the largest contour
area = cv2.contourArea(largest_contour)

# Print the area
print("Area of the outlined object: {}".format(area))

# Save the image with a transparent background
cv2.imwrite("result.png", result)

# Display the image
cv2.imshow("Image with Contours", result)

# Display the image with the smoothed contour
cv2.imshow("Image with Smoothed Contour", mask_smooth)

# Wait for a key press and close the window afterwards
cv2.waitKey(0)
cv2.destroyAllWindows()
