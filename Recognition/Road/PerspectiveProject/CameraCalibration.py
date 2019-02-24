#!/usr/bin/env python

import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle

# prepare object points
nx = 8
ny = 6


def findCorners(fname):
    # read image for calibration
    img = cv2.imread(fname)

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    # If found, draw corners
    if ret == True:
        # Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        plt.imshow(img)
        plt.show()

def calibrateCamera(dirname):
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0)...,(7,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8, 0:6].T.reshape(-1,2)

    # arrays to store object points and image points from all the images
    objpoints = [] # 3d points in real world space
    imgpoints = [] # 2d points in image plane.

    # Make a list of calibration images
    fnames = dirname + '*.jpg'
    images = glob.glob(fnames)

    # Step through the list and search for chessboard corners
    for idx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (8,6), None)

        # If found, add object points, image points
        if ret == True:
            objpoints.append(objp)
            imgpoints.append(corners)

            # Draw and display the corners
            # cv2.drawChessboardCorners(img, (8,6), corners, ret)
            
            # cv2.imshow('img', img)
            # cv2.waitKey(500)

    # Test undistortion on an image
    fname = dirname + 'test_image.jpg'
    img_size = (img.shape[1], img.shape[0])

    # Do camera calibration given object points and image points
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    fname = dirname + 'test_undist.jpg'
    cv2.imwrite(fname, dst)

    # Save the camera calibration result for later use
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    fname = dirname + 'wide_dist_pickle.p'
    pickle.dump(dist_pickle, open(fname, 'wb'))

    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.imshow(img)
    ax1.set_title('Original Image', fontsize=30)
    ax2.imshow(dst)
    ax2.set_title('Undistorted Image', fontsize=30)
    plt.show()

if __name__ == '__main__':
    #findCorners('GOPR0032.jpg')
    calibrateCamera('../../data/calibration/')
