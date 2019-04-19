import cv2
import numpy as np


# Load the image
img = cv2.imread('W4.jpg', 1)

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Blur Image with different GaussianBlur
kernel_1 = (27, 27)                 # First Kernel
kernel_2 = (37, 37)                 # Second Kernel
kernel_3 = (41, 41)                 # Third Kernel
blur_1 = cv2.GaussianBlur(gray, kernel_1, 0)
blur_2 = cv2.GaussianBlur(gray, kernel_2, 0)
blur_3 = cv2.GaussianBlur(gray, kernel_3, 0)

# Edge Detection
low = 25
high = 25
edges_1 = cv2.Canny(blur_1, low, high)
edges_2 = cv2.Canny(blur_2, low, high)
edges_3 = cv2.Canny(blur_3, low, high)

# Find Contour
contours_1, hierarchy_1 = cv2.findContours(edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_2, hierarchy_2 = cv2.findContours(edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
contours_3, hierarchy_3 = cv2.findContours(edges_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Estimate the size of the largest contour
pts_1 = []                # Create empty array to store all the points detected for contour_1
pts_2 = []                # Create empty array to store all the points detected for contour_2
pts_3 = []                # Create empty array to store all the points detected for contour_3

for i in range(len(contours_1)):
    for j in range(len(contours_1[i])):
        pts_1.append(contours_1[i][j])

for i in range(len(contours_2)):
    for j in range(len(contours_2[i])):
        pts_2.append(contours_2[i][j])

for i in range(len(contours_3)):
    for j in range(len(contours_3[i])):
        pts_3.append(contours_3[i][j])

pts_1 = np.array(pts_1)                 # Convert to numpy array
pts_2 = np.array(pts_2)                 # Convert to numpy array
pts_3 = np.array(pts_3)                 # Convert to numpy array
results_1 = cv2.convexHull(pts_1)       # Use convexHull function to obtain the largest contour
results_2 = cv2.convexHull(pts_2)       # Use convexHull function to obtain the largest contour
results_3 = cv2.convexHull(pts_3)       # Use convexHull function to obtain the largest contour

L_1 = len(results_1)                    # Obtain the length of each result to fix the largest as the contour
L_2 = len(results_2)
L_3 = len(results_3)
result = []
if (L_1 > L_2) and (L_1 > L_3):
    result = results_1
elif (L_2 > L_1) and (L_2 > L_3):
    result = results_2
else:
    result = results_3

# Draw obtained contour
ori = img.copy()                    # Make a deep copy of the original image
index = -1                          # Indicate to draw all the points
thickness = 8                       # Thickness of the contour
color = (0, 255, 0)                 # Contour color is green

cv2.polylines(ori, [result], True, color, thickness)

# Save the image
cv2.imwrite('W4_R.jpg', ori)
