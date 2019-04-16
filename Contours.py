import cv2
import numpy as np


# Import image to test, covert to grayscale
color = cv2.imread('N6.jpg', 1)
img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

# Blur Image
blur = cv2.GaussianBlur(img, (19, 19), 0)
blur2 = cv2.medianBlur(img, 9)

# Edge Detection
edges = cv2.Canny(blur, 70, 70)
# blur2 = cv2.medianBlur(edges, 3)
# edges2 = cv2.Canny(blur2, 30, 30)
cv2.imwrite('Canny.jpg', edges)

# Find Contour
contours, hierarchy = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Find the convex hull object for all contours
pts = []
for i in range(len(contours)):
    for j in range(len(contours[i])):
        pts.append(contours[i][j])

pts = np.array(pts)
result = cv2.convexHull(pts)

# Draw convex hull contour, define needed parameters
color2 = color.copy()
index = -1
thickness = 3
color = (0, 255, 0)
cv2.polylines(color2, [result], True, color, thickness)
# cv2.drawContours(color2, result, index, color, thickness)

# Save Image
cv2.imwrite('ResultN6_192020.jpg', color2)

print(result)
result.sort()
print('ends here\n')
print(result)


cv2.waitKey(0)
cv2.destroyAllWindows()
