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

# Selection
color_type = {
    'HLS0': (cv2.COLOR_RGB2HLS, 0),
    'HLS1': (cv2.COLOR_RGB2HLS, 1),
    'HLS2': (cv2.COLOR_RGB2HLS, 2),
    'HSV0': (cv2.COLOR_RGB2HSV, 0),
    'HSV1': (cv2.COLOR_RGB2HSV, 1),
    'HSV2': (cv2.COLOR_RGB2HSV, 2),
    'LAB0': (cv2.COLOR_RGB2LAB, 0),
    'LAB1': (cv2.COLOR_RGB2LAB, 1),
    'LAB2': (cv2.COLOR_RGB2LAB, 2),
    'RGB0': (cv2.COLOR_RGB2GRAY, 4),
    'RGB1': (cv2.COLOR_RGB2GRAY, 5),
    'RGB2': (cv2.COLOR_RGB2GRAY, 6),
    'GRAY': (cv2.COLOR_RGB2GRAY, 3),
}

#############################################################################
# Functions to apply slidding window
#############################################################################
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


def applyWindowCentroids(img, window_width, window_height, margin):
    window_centroids = find_window_centroids(img, window_width, window_height, margin)
    print(img)

    # If we found any window centers
    if len(window_centroids) > 0:
        # Points used to draw all the left and right windows
        l_points = np.zeros_like(img)
        r_points = np.zeros_like(img)

        # Go through each level and draw the windows
        for level in range(0, len(window_centroids)):
            # Window_mask is a function to draw window areas
            l_mask = window_mask(window_width, window_height, img, window_centroids[level][0], level)
            r_mask = window_mask(window_width, window_height, img, window_centroids[level][1], level)
            # Add graphic points from window mask here to total pixels found
            l_points[(l_points == 255) | ((l_mask == 1) ) ] = 255
            r_points[(r_points == 255) | ((r_mask == 1) ) ] = 255

        # Draw the results
        template = np.array(r_points + l_points, np.uint8) # add both left and right window pixels together
        zero_channel = np.zeros_like(template)    # create a zero color channel
        template = np.array(cv2.merge((zero_channel, template, zero_channel)), np.uint8)  # make window pixels green
        warpage = np.array(cv2.merge((img, img, img)), np.uint8) # making the original road pixels 3 color channels
        output = cv2.addWeighted(warpage, 1, template, 0.5, 0.0) # overlay the orignal road image with window results

    # If no window centers found, just display original road image
    else:
        output = np.array(cv2.merge((img, img, img)), np.uint8)

    return output

def tryWindowCentroids(fname):
    warped = mpimg.imread(fname)

    print(warped.shape)
    output = applyWindowCentroids(warped, window_width, window_height, margin)
    
    # Display the final results
    plt.imshow(output)
    plt.title('window fitting results')
    plt.show()


#############################################################################
# Functions to find good result with different sobel filters
#############################################################################
# apply sobel filter in either x direction or y direction
def abs_sobel(gray, orient='x', kernel_size=3, thresh=(0, 255)):
    """
    Apply Sobel filter in x direction or y direction
    """
    # Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=kernel_size))
    elif orient == 'y':
        sobel = np.absolute(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=kernel_size))

    # Take the absolute value of the derivative or gradient
    # and scale to 8-bit (0-255) then convert to type = np.uint8
    abssobel = np.uint8(255*sobel/np.max(sobel))

    # Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(abssobel)
    binary_output[(abssobel >= thresh[0]) & (abssobel <= thresh[1])] = 1
    
    return binary_output

# apply sobel filter in both of x and y directions
def mag_sobel(gray, sobel_kernel=3, mag_thresh=(0, 255)):
    """
    Take both Sobel x and y gradients
    """
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

# Try to find best parameters for Sobel Filters
def trySobelFilter(fname, target_channel, filter_type, thres):
    image = mpimg.imread(fname)
    
    if (color_type[target_channel][1] <= 2): 
        image = cv2.cvtColor(image, color_type[target_channel][0])
        image = image[:,:,color_type[target_channel][1]]
    elif (color_type[target_channel][1] == 3): 
        image = cv2.cvtColor(image, color_type[target_channel][0])
    elif (color_type[target_channel][1] == 4):
        image = image[:,:,0]
    elif (color_type[target_channel][1] == 5):
        image = image[:,:,1]
    else:
        image = image[:,:,2]
        
    sob_3x3 = []
    sob_7x7 = []
    sob_11x11 = []
    sob_15x15 = []
    if (filter_type == 'ORX'):
        for threshold in thres:
            sob_3x3.append(abs_sobel(image, orient='x', kernel_size=3, thresh=threshold))
            sob_7x7.append(abs_sobel(image, orient='x', kernel_size=7, thresh=threshold))
            sob_11x11.append(abs_sobel(image, orient='x', kernel_size=11, thresh=threshold))
            sob_15x15.append(abs_sobel(image, orient='x', kernel_size=15, thresh=threshold))
    elif (filter_type == 'ORY'):
        for threshold in thres:
            sob_3x3.append(abs_sobel(image, orient='y', kernel_size=3, thresh=threshold))
            sob_7x7.append(abs_sobel(image, orient='y', kernel_size=7, thresh=threshold))
            sob_11x11.append(abs_sobel(image, orient='y', kernel_size=11, thresh=threshold))
            sob_15x15.append(abs_sobel(image, orient='y', kernel_size=15, thresh=threshold))
    else:
        for threshold in thres:
            sob_3x3.append(mag_sobel(image, sobel_kernel=3, mag_thresh=threshold))
            sob_7x7.append(mag_sobel(image, sobel_kernel=7, mag_thresh=threshold))
            sob_11x11.append(mag_sobel(image, sobel_kernel=11, mag_thresh=threshold))
            sob_15x15.append(mag_sobel(image, sobel_kernel=15, mag_thresh=threshold))
            
    # Plot Every Space
    f, ax = plt.subplots(4, len(thres), figsize=(24,9))
    f.tight_layout()

    for row in range(4):
        for col in range(len(thres)):
            if row == 0:
                ax[row, col].imshow(sob_3x3[col], cmap='gray')
                ax[row, col].set_title('3x3 ({}-{})'.format(thres[col][0], thres[col][1]), fontsize=10)
            elif row == 1:
                ax[row, col].imshow(sob_7x7[col], cmap='gray')
                ax[row, col].set_title('7x7 ({}-{})'.format(thres[col][0], thres[col][1]), fontsize=10)
            elif row == 2:
                ax[row, col].imshow(sob_11x11[col], cmap='gray')
                ax[row, col].set_title('11x11 ({}-{})'.format(thres[col][0], thres[col][1]), fontsize=10)
            else:
                ax[row, col].imshow(sob_15x15[col], cmap='gray')
                ax[row, col].set_title('15x15 ({}-{})'.format(thres[col][0], thres[col][1]), fontsize=10)

    plt.show()
    
#############################################################################
# Functions to find good result with different thresholds
#############################################################################
# Apply Threshold to each channel
def applyThreshold(image, ch0th, ch1th, ch2th):
    bin = np.zeros_like(image[:,:,0])
    bin[((image[:,:,0] > ch0th[0]) & (image[:,:,0] <= ch0th[1]) & \
         (image[:,:,1] > ch1th[0]) & (image[:,:,1] <= ch1th[1]) & \
         (image[:,:,2] > ch2th[0]) & (image[:,:,2] <= ch2th[1]))] = 1
    return bin

# Try to find optimized thresholds applied to each channel of specified color space
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
    ch0 = []
    target_channel = image[:,:,0]
    for th in ch0th:
        img = np.zeros_like(target_channel)
        img[(target_channel >= th[0]) & (target_channel <= th[1])] = 1
        ch0.append(img)

    # for channel 1
    ch1 = []
    target_channel = image[:,:,1]
    for th in ch1th:
        img = np.zeros_like(target_channel)
        img[(target_channel >= th[0]) & (target_channel <= th[1])] = 1
        ch1.append(img)

    # for channel 2
    ch2 = []
    target_channel = image[:,:,2]
    for th in ch2th:
        img = np.zeros_like(target_channel)
        img[(target_channel >= th[0]) & (target_channel <= th[1])] = 1
        ch2.append(img)
        
    # Plot Every Space
    f, ax = plt.subplots(3, 3, figsize=(24,9))
    f.tight_layout()

    for row in range(3):
        for col in range(3):
            if row == 0:   # for channel0
                ax[row, col].imshow(ch0[col], cmap='gray')
                ax[row, col].set_title('ch0:({}-{})'.format(ch0th[col][0], ch0th[col][1]), fontsize=10)
            elif row == 1: # for channel1
                ax[row, col].imshow(ch1[col], cmap='gray')
                ax[row, col].set_title('ch1:({}-{})'.format(ch1th[col][0], ch1th[col][1]), fontsize=10)
            else:          # for channel2
                ax[row, col].imshow(ch2[col], cmap='gray')
                ax[row, col].set_title('ch2:({}-{})'.format(ch2th[col][0], ch2th[col][1]), fontsize=10)
                
    plt.show()
    
#############################################################################
# Functions to find good result in the color spaces
# Try to split Images into channels to find best color space
#############################################################################
def tryColorSpaceSplit(fname):
    # Split at RGB Space
    image = mpimg.imread(fname)
    image_chname = ['Red Channel', 'Green Channel', 'Blue Channel']

    # Split at HLS Space
    hls = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    hls_chname = ['Hue Channel', 'Lightless Channel', 'Saturation Channel']

    # Split at HSV Space
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    hsv_chname = ['Hue Channel', 'Saturation Channel', 'Value Channel']

    # Split at LAB Space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    lab_chname = ['Lightness', 'Green-Red (A) Channel', 'Blue-Yellow (B) Channel']

    # Plot Every Space
    f, ax = plt.subplots(4, 3, figsize=(24, 9))
    f.tight_layout()

    for row in range(4):
        for col in range(3):
            if row == 0:
                ax[row, col].imshow(image[:,:,col], cmap='gray')
                ax[row, col].set_title(image_chname[col], fontsize=10)
            elif row == 1:
                ax[row, col].imshow(hls[:,:,col], cmap='gray')
                ax[row, col].set_title(hls_chname[col], fontsize=10)
            elif row == 2:
                ax[row, col].imshow(hsv[:,:,col], cmap='gray')
                ax[row, col].set_title(hsv_chname[col], fontsize=10)
            else:
                ax[row, col].imshow(lab[:,:,col], cmap='gray')
                ax[row, col].set_title(lab_chname[col], fontsize=10)
    plt.show()

#############################################################################
# Functions to apply pipeline operation
#############################################################################
def applyPipeline(origimage, target_channel, color_thresholds, filter_type, kernel_size, threshold=(0,255)):
    image = origimage

    if (color_type[target_channel][1] <= 2): 
        image = cv2.cvtColor(image, color_type[target_channel][0])
        gray = image[:,:,color_type[target_channel][1]]
    elif (color_type[target_channel][1] == 3): 
        gray = cv2.cvtColor(image, color_type[target_channel][0])
    elif (color_type[target_channel][1] == 4):
        gray = image[:,:,0]
    elif (color_type[target_channel][1] == 5):
        gray = image[:,:,1]
    else:
        gray = image[:,:,2]

    # Pipelined Operation
    result_with_color = applyThreshold(image, color_thresholds[0], color_thresholds[1], color_thresholds[2])
    
    if (filter_type == 'ORX'):
        result_with_filter = abs_sobel(gray, orient='x', kernel_size=kernel_size, thresh=threshold)
    elif (filter_type == 'ORY'):
        result_with_filter = abs_sobel(gray, orient='y', kernel_size=kernel_size, thresh=threshold)
    else:
        result_with_filter = mag_sobel(gray, sobel_kernel=kernel_size, mag_thresh=threshold)

    color_result = np.dstack((np.zeros_like(gray), result_with_filter*255, result_with_color*255))
    binary_result = np.zeros_like(gray)
    binary_result[(result_with_color == 1) | (result_with_filter == 1)] = 1

    return color_result, binary_result 
    
        
def tryPipeline(fname, target_channel, color_thresholds, filter_type, kernel_size, threshold=(0,255)):
    origimage = mpimg.imread(fname)

    color_result, binary_result = applyPipeline(origimage, target_channel, color_thresholds, filter_type, kernel_size, threshold)

# Plot the result
    f, ax = plt.subplots(1, 3, figsize=(24, 9))
    f.tight_layout()

    ax[0].imshow(origimage)
    ax[0].set_title('Original Image', fontsize=10)

    ax[1].imshow(color_result)
    ax[1].set_title('Pipelined Pseudo Result', fontsize=10)
    print(color_result.shape)
    print(color_result)

    ax[2].imshow(binary_result, cmap='gray')
    ax[2].set_title('Pipelined Binary Result', fontsize=10)
    plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    plt.show()


#############################################################################
# Functions to apply perspective transform
#############################################################################
def perspective_transform(img, src, dst):   
    """
    Applies a perspective 
    """
    M = cv2.getPerspectiveTransform(src, dst)
    img_size = (img.shape[1], img.shape[0])
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
    
    return warped

def applyPerspectiveTransform(image):
    print(image.shape)
    (bottom_px, right_px) = (image.shape[0] - 1, image.shape[1] - 1)
    pts = np.array([[50,bottom_px],[252,207],[330,207], [530, bottom_px]], np.int32)
    #cv2.polylines(image, [pts], True, (255,0,0), 10)
    #plt.axis('off')
    #plt.imshow(image)
    #plt.show()

    src_pts = pts.astype(np.float32)
    dst_pts = np.array([[0, bottom_px], [0, 0], [right_px, 0], [right_px, bottom_px]], np.float32)

    img = perspective_transform(image, src_pts, dst_pts)
    
    return img

def tryPerspectiveTransform(fname):
    image = mpimg.imread(fname)
    
    plt.imshow(applyPerspectiveTransform(image))
    plt.show()


if __name__ == "__main__":

    image = mpimg.imread('bridge_shadow.jpg')
    
    #tryWindowCentroids('warped-example.jpg')
    #tryColorSpaceSplit('bridge_shadow.jpg')
    #tryThreshold('bridge_shadow.jpg', 'HLS', ((0, 85), (86,  170), (171, 255)),  ((0, 85), (86,  170), (171, 255)),  ((0, 85), (86,  170), (171, 255)))
    #trySobelFilter('bridge_shadow.jpg', 'HLS2', 'ORX', ((20, 100), (50,  150), (80, 200)))
    #trySobelFilter('bridge_shadow.jpg', 'HLS2', 'AND', ((20, 120), (50,  150), (80, 200)))
    #tryPipeline('bridge_shadow.jpg', 'HLS2', ((0, 30), (105, 255), (170, 255)), 'ORX', 3, threshold=(20,100))
    #tryPerspectiveTransform('bridge_shadow.jpg')
    
    _, biimg = applyPipeline(image, 'HLS2', ((0, 30), (105, 255), (170, 255)), 'ORX', 3, threshold=(20,100))
    img = applyPerspectiveTransform(biimg)
    img = applyWindowCentroids(img, window_width, window_height, margin)
    img = img*255
    plt.imshow(img)
    plt.show()
    
    #image = mpimg.imread('bridge_shadow.jpg')
    #image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
    #image = applyThreshold(image, (0, 30), (105, 255), (170, 255))
    #plt.imshow(image, cmap='gray')
    #plt.show()
