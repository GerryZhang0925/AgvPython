#!/usr/bin/env python

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import sys
import os

# Applying Canny Filter
def filterCanny(grayImage):
    # Define a kernel size for Gaussian Smoothing/Blurring
    szKernel = 3
    blurGrayImage = cv2.GaussianBlur(grayImage, (szKernel, szKernel), 0)
    # Define Canny Parameter
    lowThreshold = 180
    highThreshold = 200
    edgeImage = cv2.Canny(blurGrayImage, lowThreshold, highThreshold)

    return edgeImage

# Run Hough Transform to fit lane marks
def MaskEdges(edgeImage, topleft=0):
    maskImage = np.zeros_like(edgeImage)
    ignoreColor = 255

    # Define four sided polygon to mask
    shape = edgeImage.shape
    if (topleft == 0):
        vertices = np.array([[(200, shape[0]), (600, 450), (700, 450), (shape[1]-200, shape[0])]], dtype = np.int32)
    else:
        vertices = np.array([[(200, topleft), (600, 450), (700, 450), (shape[1]-200, topleft)]], dtype = np.int32)
    cv2.fillPoly(maskImage, vertices, ignoreColor)
    maskImage = cv2.bitwise_and(edgeImage, maskImage)

    # Define the Hough transform parameters
    rho = 1             # Distance resolution in pixels of the hough grid
    theta = np.pi/180   # Angular resolution in radians of the Hough grid
    threshold = 1       # Minimum number of votes
    min_line_length = 5 # Minimum number of pixels making up a line
    max_line_gap = 1    # Maximum gap in pixels between connectable line segments
    lineImage = np.copy(edgeImage)*0

    # Run Hough on edge detected image
    lines = cv2.HoughLinesP(maskImage, rho, theta, threshold, np.array([]),
                            min_line_length, max_line_gap)

    # Iterate over the lines
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(lineImage, (x1, y1), (x2, y2), (255, 0, 0), 10)
            
    return lineImage

# Processing one frame of video
def processFrame(Image):

    grayImage = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
    
    edgeImage = filterCanny(grayImage)
    maskImage = MaskEdges(edgeImage, 640)
    darkImage = np.copy(edgeImage)*0
    colorEdge = np.dstack((darkImage, darkImage, maskImage))
    linesEdge = cv2.addWeighted(Image, 0.8, colorEdge, 1, 0)
    
    return linesEdge


if __name__ == "__main__":
    args = sys.argv
    fname, fext = os.path.splitext(args[1])

    # Processing a mp4 video
    if (fext == ".mp4"):
        cap = cv2.VideoCapture(args[1])

        # if opened successfully
        if (cap.isOpened() == False):
            print("Error opening video stream or file")

        # Read until video is completed
        while (cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                frame = processFrame(frame)
                cv2.imshow('Frame', frame)

                # Press Q on keyboard to exit
                if cv2.waitKey(25) & 0xff == ord('q'):
                    break

            else:
                break

        cap.release()
        cv2.destroyAllWindows()
        
    # Processing a image file
    else:
        # Read GrayScaled Image
        Image = mpimg.imread(args[1])
        grayImage = cv2.cvtColor(Image, cv2.COLOR_RGB2GRAY)
    
        edgeImage = filterCanny(grayImage)
        maskImage = MaskEdges(edgeImage)
        darkImage = np.copy(edgeImage)*0
        colorEdge = np.dstack((maskImage, darkImage, darkImage))
        linesEdge = cv2.addWeighted(Image, 0.8, colorEdge, 1, 0)

        # Display the image
        plt.imshow(linesEdge)
        plt.show()

        cv2.imwrite("overrape.png", linesEdge)

