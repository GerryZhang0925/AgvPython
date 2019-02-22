#!/usr/bin/env python

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')

templist = ['cutout1.jpg', 'cutout2.jpg', 'cutout3.jpg', 'cutout4.jpg', 'cutout5.jpg', 'cutout6.jpg']

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # make a copy of the image
    draw_img = np.copy(img)
    # draw each bounding box on your image copy using cv2.rectangle()
    # return the image copy with boxes drawn
    for bbox in bboxes:
        draw_img = cv2.rectangle(draw_img, bbox[0], bbox[1], color, thick)

    return draw_img

def find_matches(img, template_list):
    bbox_list = []
    method = cv2.TM_CCOEFF_NORMED
    for temp in template_list:
        tmp = mpimg.imread(temp)
        result = cv2.matchTemplate(img, tmp, method)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        w, h = (tmp.shape[1], tmp.shape[0])
        if method in [cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED]:
            top_left = min_loc
        else:
            top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        bbox_list.append((top_left, bottom_right))

    return bbox_list

# Add bounding boxes in this format, these are just example coordinates.

result = draw_boxes(image, find_matches(image, templist))
plt.imshow(result)
plt.show()
