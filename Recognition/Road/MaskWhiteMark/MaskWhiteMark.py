#!/usr/bin/env python

import sys
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

#Define color selection criteria
red_threshold = 200
green_threshold = 200
blue_threshold = 200
    
def ColorRegionImage(image):
    # Make a copy of the image
    selectImage = np.copy(image)
    
    rgb_threshold = [red_threshold, green_threshold, blue_threshold]

    # Mask pixels below the threshold
    color_thresholds = (image[:,:,0] < rgb_threshold[0]) \
                       | (image[:,:,1] < rgb_threshold[1]) \
                       | (image[:,:,2] < rgb_threshold[2])

    selectImage[color_thresholds] = [0, 0, 0]

    return selectImage


def MaskImage(image):
    # Grab the x and y size and make a copy of the image
    (ysize, xsize, _) = image.shape
    selectImage = np.copy(image)
    
    # Define a triangle region of interest
    left_bottom = [0+200, ysize]
    right_bottom = [xsize-200, ysize]
    apex = [xsize/2, ysize/2 + 75]

    # Fit lines (y=Ax+B) to identify the 3 sided region of interest
    # np.polyfit() returns the coefficients [A,B] of the fit
    fit_left = np.polyfit((left_bottom[0], apex[0]), (left_bottom[1], apex[1]), 1)
    fit_right = np.polyfit((right_bottom[0], apex[0]), (right_bottom[1], apex[1]), 1)
    fit_bottom = np.polyfit((left_bottom[0], right_bottom[0]), (left_bottom[1], right_bottom[1]), 1)

    # Find the region inside the lines
    XX, YY = np.meshgrid(np.arange(0, xsize), np.arange(0, ysize))
    region_thresholds = (YY > (XX * fit_left[0] + fit_left[1])) & \
                        (YY > (XX * fit_right[0] + fit_right[1])) & \
                        (YY < (XX * fit_bottom[0] + fit_bottom[1]))

    selectImage[~region_thresholds] = [0, 0, 0]

    return selectImage


def OverlapImage(image, selectImage):
    baseImage = np.copy(image)

    color_thresholds = (selectImage[:,:,0] >= red_threshold) \
                       | (selectImage[:,:,1] >= green_threshold) \
                       | (selectImage[:,:,2] >= blue_threshold)

    baseImage[color_thresholds] = [255, 0, 0]

    return baseImage;


if __name__ == "__main__":
    args = sys.argv
    
    # Read in the image
    image = mpimg.imread(args[1])
    
    selectImage = ColorRegionImage(image)
    selectImage = MaskImage(selectImage)
    selectImage = OverlapImage(image, selectImage)
    # Display the image
    plt.imshow(selectImage)
    plt.show()

