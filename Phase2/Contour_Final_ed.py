import cv2
import os
import numpy as np


def largestcontour(args):
    pts = []                                 # Create empty array to store all the points detected for contours

    for i in range(len(args)):
        for j in range(len(args[i])):
            pts.append(args[i][j])
    pts = np.array(pts)                      # Convert to numpy array
    return cv2.convexHull(pts)                # Use convexHull function to obtain the largest contour


# Obtain folder path
path = os.getcwd()
for filename in os.listdir(path):
    if filename.endswith('.JPG') or filename.endswith('.jpg'):
        print(filename, 'loaded')
        # Load image to process
        img = cv2.imread(f'{filename}')

        # Convert image to grayscale
        Bi_blur = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Blur image with different GaussianBlur
        kernel_1 = (27, 27)  # First Kernel
        kernel_2 = (37, 37)  # Second Kernel
        kernel_3 = (41, 41)  # Third Kernel

        blur_1 = cv2.GaussianBlur(Bi_blur, kernel_1, 0)
        blur_2 = cv2.GaussianBlur(Bi_blur, kernel_2, 0)
        blur_3 = cv2.GaussianBlur(Bi_blur, kernel_3, 0)

        # Different Sigma Value
        blur_4 = cv2.GaussianBlur(Bi_blur, kernel_1, sigmaX=5)
        blur_5 = cv2.GaussianBlur(Bi_blur, kernel_2, sigmaX=5)
        blur_6 = cv2.GaussianBlur(Bi_blur, kernel_3, sigmaX=5)

        blur_7 = cv2.GaussianBlur(Bi_blur, kernel_1, sigmaX=6)
        blur_8 = cv2.GaussianBlur(Bi_blur, kernel_2, sigmaX=6)
        blur_9 = cv2.GaussianBlur(Bi_blur, kernel_3, sigmaX=6)

        # Canny Edge Detection
        low = 19
        high = 24
        edges_1 = cv2.Canny(blur_1, low, high)
        edges_2 = cv2.Canny(blur_2, low, high)
        edges_3 = cv2.Canny(blur_3, low, high)
        edges_4 = cv2.Canny(blur_4, low, high)
        edges_5 = cv2.Canny(blur_5, low, high)
        edges_6 = cv2.Canny(blur_6, low, high)
        edges_7 = cv2.Canny(blur_7, low, high)
        edges_8 = cv2.Canny(blur_8, low, high)
        edges_9 = cv2.Canny(blur_9, low, high)

        # Find Contour
        contours_1, hierarchy_1 = cv2.findContours(edges_1, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_2, hierarchy_2 = cv2.findContours(edges_2, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_3, hierarchy_3 = cv2.findContours(edges_3, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_4, hierarchy_4 = cv2.findContours(edges_4, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_5, hierarchy_5 = cv2.findContours(edges_5, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_6, hierarchy_6 = cv2.findContours(edges_6, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_7, hierarchy_7 = cv2.findContours(edges_7, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_8, hierarchy_8 = cv2.findContours(edges_8, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        contours_9, hierarchy_9 = cv2.findContours(edges_9, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        # Estimate the size of the largest contour
        results_1 = largestcontour(contours_1)
        results_2 = largestcontour(contours_2)
        results_3 = largestcontour(contours_3)
        results_4 = largestcontour(contours_4)
        results_5 = largestcontour(contours_5)
        results_6 = largestcontour(contours_6)
        results_7 = largestcontour(contours_7)
        results_8 = largestcontour(contours_8)
        results_9 = largestcontour(contours_9)

        # Obtain the length of each result to fix the largest as the contour
        C = (results_1, results_2, results_3, results_4, results_5, results_6, results_7, results_8, results_9)
        max_con = 0

        for i in C:
            if len(i) >= max_con:
                max_con = len(i)
                result = i

        # Determine if the ratio of contour belongs to a card 1.4 < ratio <1.7
        rect = cv2.minAreaRect(result)
        if rect[1][0] > rect[1][1]:
            width = rect[1][0]
            height = rect[1][1]
        else:
            width = rect[1][1]
            height = rect[1][0]
        ratio = width/height                       # For Debugging purpose /rect[1][0]/width /rect[1][1]/height
        area_con = cv2.contourArea(result)
        area = width * height

        if (ratio > 1.4) and (ratio < 1.7) and (area_con > area * 0.9):

            # Draw obtained contour
            ori = img.copy()                # Make a deep copy of the original image
            index = -1                      # Indicate to draw all the points
            thickness = 8                   # Thickness of the contour
            color = (0, 255, 0)             # Contour color is green

            # Draw the final contour
            cv2.drawContours(ori, [result], index, color, thickness)
            print([len(i) for i in C])
            # Save the result as a new file
            cv2.imwrite(f'Result_{filename}', ori)
            print('output', f'Result_{filename}')

        else:
            # Try to find the contour by approximating polygonal curves
            epsilon = 0.115 * cv2.arcLength(result, True)
            approx = cv2.approxPolyDP(result, epsilon, True)
            rect = cv2.minAreaRect(approx)

            if rect[1][0] > rect[1][1]:
                width = rect[1][0]
                height = rect[1][1] if rect[1][1] != 0 else 1.0
            else:
                width = rect[1][1]
                height = rect[1][0] if rect[1][0] != 0 else 1.0
            ratio = width / height              # For Debugging purpose /rect[1][0]/width /rect[1][1]/height

            if (height != 0) and (ratio > 1.4) and (ratio < 1.7) and (area_con > area * 0.85):

                # Draw obtained contour
                ori = img.copy()                    # Make a deep copy of the original image
                index = -1                          # Indicate to draw all the points
                thickness = 8                       # Thickness of the contour
                color = (0, 255, 0)                 # Contour color is green

                # Draw the final contour
                cv2.drawContours(ori, [approx], index, color, thickness)

                print([len(i) for i in C])
                # Save the result as a new file
                cv2.imwrite(f'Result_{filename}', ori)
                print('output', f'Result_{filename}')

            else:
                print(f'Unable to detect the right contour for {filename}.')
                continue
