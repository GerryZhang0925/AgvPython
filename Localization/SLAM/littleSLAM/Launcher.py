#!/usr/bin/env python

import SensorDataReader
import LPoint2D
import Pose2D
import cv2
import numpy as np

MIN_X = -40.0
MIN_Y = -40.0
MAX_X = 40.0
MAX_Y = 40.0
RANG_X = MAX_X - MIN_X
RANG_Y = MAX_Y - MIN_Y
IMG_W  = 800
IMG_H  = 800

def showScans(scan, pose):
    img = np.zeros((IMG_W, IMG_H, 3))
    points = []
    for point in scan:
        points.append(pose.getGlobalPoint(point.x, point.y))
    for [x, y] in points:
        #print(x)
        x = int((x - MIN_X)*IMG_W/RANG_X)
        y = int((y - MIN_Y)*IMG_H/RANG_Y)
        #print(x)
        #print(y)
        cv2.rectangle(img, (x, y), (x+1, y+1), (255, 255, 0), thickness=1)
    cv2.imshow('Frame', img)
    cv2.waitKey(0)


def main():
    cnt = 0

    sensorReader = SensorDataReader.SensorDataReader()
    sensorReader.openScanFile('corridor.lsc')
    ret, scan, pose = sensorReader.loadScan(cnt)
    while (ret):
        showScans(scan, pose)
        ret, scan, pose = sensorReader.loadScan(++cnt)
    sensorReader.closeScanFile()
    cv2.destoryAllWindows()

if __name__ == '__main__':
    main()
