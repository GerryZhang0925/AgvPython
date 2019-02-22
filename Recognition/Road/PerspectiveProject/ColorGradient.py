#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
import glob

# window settings
window_width = 50
window_height = 80 # Break image into 9 vertical layers since image height is 720
margin = 100 # How much to slide left and right for searching


def pipeline(img, s_thresh=(170, 255), sx_thresh=(20, 100)):
    img = np.copy(img)
    
    # Convert to HSV color space and separate the V channel
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    l_channel = hsv[:,:,1]
    s_channel = hsv[:,:,2]

    # Sobel x
    sobelx       = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0)  # Take the deriative in x
    abs_sobelx   = np.absolute(sobelx)                     # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))

    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Threashold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

    # Stack each channel
    # Note color_binary[:, :, 0]  is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    return color_binary

def applyPipeline(fname):
    image = mpimg.imread(fname)
    result = pipeline(image)

    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=40)

    ax2.imshow(result, cmap='gray')
    ax2.set_title('Pipeline Result', fontsize=40)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Define a function that applies Sobel x or y,
# then takes an absolute value and applies a threshold.
# Note; calling your function with orient='x', thresh_min=5, thresh_max=100
# should produce output like the example image shown above this quiz.
def abs_sobel_thresh(img, orient='x', thresh_min=0, thresh_max=255):
    '''
    Apply the following steps to img
   
    
    3) Take the absolute value of the derivative or gradient
    4) Scale to 8-bit (0-255) then convert to type = np.uint8
    5) Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    6) Return this mask as your binary_output image
    '''
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0))
    elif orient == 'y':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1))
    abssobel = np.uint8(255*sobel/np.max(sobel))
    mask = np.zeros_like(abssobel)
    mask[(abssobel >= thresh_min) & (abssobel <= thresh_max)] = 1
    return mask

def applySobel(fname):
    image = mpimg.imread(fname)
    grad_binary = abs_sobel_thresh(image, orient='x', thresh_min=20, thresh_max=100)
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(grad_binary, cmap='gray')
    ax2.set_title('Thresholded Gradient', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def window_mask(width, height, img_ref, center, level):
    output = np.zeros_like(img_ref)
    output[int(img_ref.shape[0] - (level+1)*height):int(img_ref.shape[0] - level*height), max(0, int(center-width/2)):min(int(center+width/2), img_ref.shape[1])]=1
    return output

def find_window_centroids(warped, window_width, window_height, margin):
    window_centroids = [] # Store the (left, right) window centroid positions per level
    window = np.ones(window_width) # Create out window template that we will use for convolutions

    # First find the two starting positions for the left and right lane by using np.sum to get the vertical image slice
    # and then np.convolve the vertical image slice with the window template

    # Sum quarter bottom of image to get slice, could use a different ratio
    l_sum = np.sum(warped[int(3*warped.shape[0]/4):,:int(warped.shape[1]/2)], axis=0)
    l_center = np.argmax(np.convolve(window, l_sum)) - window_width/2
    r_sum = np.sum(warped[int(3*warped.shape[0]/4):,int(warped.shape[1]/2):], axis=0)
    r_center = np.argmax(np.convolve(window, r_sum)) - window_width/2 + int(warped.shape[1]/2)

    # Add what we found for the first layer
    window_centroids.append((l_center, r_center))

    # Go through each layer looking for max pixel locations
    for level in range(1, (int)(warped.shape[0]/window_height)):
        # convolve the window into the vertical slice of the image
        image_layer = np.sum(warped[int(warped.shape[0]-(level+1)*window_height):int(warped.shape[0]-level*window_height), :], axis = 0)
        conv_signal = np.convolve(window, image_layer)
        # Find the best left centroid by using past left center as a reference
        # Use window_width/2 as offset because convollution signal reference is at right side of window, not center of window
        offset = window_width/2
        l_min_index = int(max(l_center+offset-margin,0))
        l_max_index = int(min(l_center+offset+margin,warped.shape[1]))
        l_center = np.argmax(conv_signal[l_min_index:l_max_index])+l_min_index-offset
        # Find the best right centroid by using past right center as a reference
        r_min_index = int(max(r_center+offset-margin,0))
        r_max_index = int(min(r_center+offset+margin, warped.shape[1]))
        r_center = np.argmax(conv_signal[r_min_index:r_max_index])+r_min_index-offset
        # Add what we found for that layer
        window_centroids.append((l_center, r_center))

    return window_centroids

def applyWindowCentroids(fname):
    warped = mpimg.imread(fname)
    window_centroids = find_window_centroids(warped, window_width, window_height, margin)

    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(warped)
        r_points = np.zeros_like(warped)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, warped, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, warped, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template)    # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((warped, warped, warped)), np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0)          # overlay the orignal road image with window results

    # If no window centers found, just display original road image
    else:
        output = np.array(cv2.merge((warped, warped, warped)), np.uint8)

    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()

def mag_thresh(img, sobel_kernel=3, mag_thresh=(0, 255)):
    # 1) Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take both Sobel x and y gradients
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the gradient magnitude
    gradmag = np.sqrt(sobelx**2 + sobely**2)
    # Rescale to 8 bit
    scale_factor = np.max(gradmag)/255
    gradmag = (gradmag/scale_factor).astype(np.uint8)
    # Create a binary image of ones where threshold is met, zeros otherwise
    binary_output = np.zeros_like(gradmag)
    binary_output[(gradmag >= mag_thresh[0]) & (gradmag <= mag_thresh[1])] = 1

    return binary_output

def applyMagThresh(fname):
    image = mpimg.imread(fname)
    mag_binary = mag_thresh(image, sobel_kernel=3, mag_thresh=(30, 100))
    # Plot the result
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(mag_binary, cmap='gray')
    ax2.set_title('Thresholded Magnitude', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()

def hls_select(img, thresh=(0, 255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def applyHlsSelect(fname):
    image = mpimg.imread(fname)
    hls_binary = hls_select(image, thresh=(90, 255))
    
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()
    ax1.imshow(image)
    ax1.set_title('Original Image', fontsize=50)
    ax2.imshow(hls_binary, cmap='gray')
    ax2.set_title('Thresholded S', fontsize=50)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


# Apply Threshold to each channel
def applyThreshold(image, ch0th, ch1th, ch2th):
    bin = np.zeros_like(image[:,:,0])
    bin[((image[:,:,0] > ch0th[0]) & (image[:,:,0] <= ch0th[1]) & \
         (image[:,:,1] > ch1th[0]) & (image[:,:,1] <= ch1th[1]) & \
         (image[:,:,2] > ch2th[0]) & (image[:,:,2] <= ch2th[1]))] = 1
    return bin
    
def tryThreshold(fname, colorspace, ch0th, ch1th, ch2th):
    image = mpimg.imread(fname)

    if (colorspace == 'HLS'):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    elif (colorspace == 'HSV'):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    elif (colorspace == 'LAB'):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    else:
        pass

    # for channel 0
    target_channel = image[:,:,0]
    start0 = ch0th[0]
    step0 = (ch0th[1]-ch0th[0])/3.0
    img0_0 = np.zeros_like(target_channel)
    img0_0[(target_channel > start0) & (target_channel <= start0 + step0)] = 1
    img0_1 = np.zeros_like(target_channel)
    img0_1[(target_channel > start0 + step0) & (target_channel <= start0 + 2*step0)] = 1
    img0_2 = np.zeros_like(target_channel)
    img0_2[(target_channel > start0 + 2*step0) & (target_channel <= start0 + 3*step0)] = 1

    # for channel 1
    target_channel = image[:,:,1]
    start1 = ch1th[0]
    step1 = (ch1th[1]-ch1th[0])/3
    img1_0 = np.zeros_like(target_channel)
    img1_0[(target_channel > start1) & (target_channel <= start1 + step1)] = 1
    img1_1 = np.zeros_like(target_channel)
    img1_1[(target_channel > start1 + step1) & (target_channel <= start1 + 2*step1)] = 1
    img1_2 = np.zeros_like(target_channel)
    img1_2[(target_channel > start1 + 2*step1) & (target_channel <= start1 + 3*step1)] = 1

    # for channel 2
    target_channel = image[:,:,2]
    start2 = ch2th[0]
    step2 = (ch2th[1]-ch2th[0])/3
    img2_0 = np.zeros_like(target_channel)
    img2_0[(target_channel > start2) & (target_channel <= start2 + step2)] = 1
    img2_1 = np.zeros_like(target_channel)
    img2_1[(target_channel > start2 + step2) & (target_channel <= start2 + 2*step2)] = 1
    img2_2 = np.zeros_like(target_channel)
    img2_2[(target_channel > start2 + 2*step2) & (target_channel <= start2 + 3*step2)] = 1
    

    # Plot Every Space
    f, ax = plt.subplots(3, 3, figsize=(24,9))
    f.tight_layout()

    ax[0, 0].imshow(img0_0, cmap='gray')
    ax[0, 0].set_title('({}-{})'.format(start0,start0+step0), fontsize=10)
    ax[0, 1].imshow(img0_1, cmap='gray')
    ax[0, 1].set_title('({}-{})'.format(start0+step0,start0+2*step0), fontsize=10)
    ax[0, 2].imshow(img0_2, cmap='gray')
    ax[0, 2].set_title('({}-{})'.format(start0+2*step0,start0+3*step0), fontsize=10)
    ax[1, 0].imshow(img1_0, cmap='gray')
    ax[1, 0].set_title('({}-{})'.format(start1,start1+step1), fontsize=10)
    ax[1, 1].imshow(img1_1, cmap='gray')
    ax[1, 1].set_title('({}-{})'.format(start1+step1,start1+2*step1), fontsize=10)
    ax[1, 2].imshow(img1_2, cmap='gray')
    ax[1, 2].set_title('({}-{})'.format(start1+2*step1,start1+3*step1), fontsize=10)
    ax[2, 0].imshow(img2_0, cmap='gray')
    ax[2, 0].set_title('({}-{})'.format(start2,start2+step2), fontsize=10)
    ax[2, 1].imshow(img2_1, cmap='gray')
    ax[2, 1].set_title('({}-{})'.format(start2+step2,start2+2*step2), fontsize=10)
    ax[2, 2].imshow(img2_2, cmap='gray')
    ax[2, 2].set_title('({}-{})'.format(start2+2*step2,start2+3*step2), fontsize=10)
    plt.show()
    

# Split Images into different channel
def tryColorSpaceSplit(fname):
    # Split at RGB Space
    image = mpimg.imread(fname)

    # Split at HLS Space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)

    # Split at HSV Space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    # Split at LAB Space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)

    # Plot Every Space
    f, ax = plt.subplots(4, 3, figsize=(24, 9))
    f.tight_layout()

    ax[0, 0].imshow(image[:,:,0], cmap='gray')
    ax[0, 0].set_title('Red Channel', fontsize=10)
    ax[0, 1].imshow(image[:,:,1], cmap='gray')
    ax[0, 1].set_title('Green Channel', fontsize=10)
    ax[0, 2].imshow(image[:,:,2], cmap='gray')
    ax[0, 2].set_title('Blue Channel', fontsize=10)
    ax[1, 0].imshow(hls[:,:,0], cmap='gray')
    ax[1, 0].set_title('Hue Channel', fontsize=10)
    ax[1, 1].imshow(hls[:,:,1], cmap='gray')
    ax[1, 1].set_title('Lightless Channel', fontsize=10)
    ax[1, 2].imshow(hls[:,:,2], cmap='gray')
    ax[1, 2].set_title('Saturation Channel', fontsize=10)
    ax[2, 0].imshow(hsv[:,:,0], cmap='gray')
    ax[2, 0].set_title('Hue Channel', fontsize=10)
    ax[2, 1].imshow(hsv[:,:,1], cmap='gray')
    ax[2, 1].set_title('Saturation Channel', fontsize=10)
    ax[2, 2].imshow(hsv[:,:,2], cmap='gray')
    ax[2, 2].set_title('Value Channel', fontsize=10)
    ax[3, 0].imshow(lab[:,:,0], cmap='gray')
    ax[3, 0].set_title('Lightness', fontsize=10)
    ax[3, 1].imshow(lab[:,:,1], cmap='gray')
    ax[3, 1].set_title('Green-Red (A) Channel', fontsize=10)
    ax[3, 2].imshow(lab[:,:,2], cmap='gray')
    ax[3, 2].set_title('Blue-Yellow (B) Channel', fontsize=10)
    plt.show()
    

if __name__ == "__main__":
    #applyPipeline('bridge_shadow.jpg')
    #applySobel('signs_vehicles_xygrad.png')
    #applyWindowCentroids('warped-example.jpg')
    #applyMagThresh('signs_vehicles_xygrad.png')
    #applyHlsSelect('bridge_shadow.jpg')
    #tryColorSpaceSplit('bridge_shadow.jpg')
    #tryThreshold('bridge_shadow.jpg', 'HLS', (0, 255), (0, 255), (0, 255))

    image = mpimg.imread('bridge_shadow.jpg')
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    image = applyThreshold(image, (0, 30), (105, 255), (170, 255))
    plt.imshow(image, cmap='gray')
    plt.show()