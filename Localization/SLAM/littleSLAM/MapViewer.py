#!/usr/bin/env python

import cv2
import LPoint2D
import Pose2D
import numpy as np

class MapViewer:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.min_x  = -40.0
        self.max_x  = 40.0
        self.min_y  = -40.0
        self.max_y  = 40.0
        self.ppi_x  = self.width/(self.max_x - self.min_x)
        self.ppi_y  = self.height/(self.max_y - self.min_y)
        self.img    = np.zeros((self.width, self.height, 3))
        self.poses  = []
        self.pnts   = []
        self.cnt    = 0

    def setXRange(self, xmin, xmax):
        self.min_x  = xmin
        self.max_x  = xmax
        self.ppi_x  = self.width/(xmax - xmin)

    def setYRange(self, ymin, ymax):
        self.min_y  = ymin
        self.max_y  = ymax
        self.ppi_y  = self.height/(ymax - ymin)

    def showScans(self, scan, pose):
        self.cnt += 1;

        if (self.cnt % 10 == 1):
            self.poses.append(pose)
            for point in scan:
                self.pnts.append(pose.getGlobalPoint(point.x, point.y))
        

        # plot
        self.img.fill(0)
        
        # show trajectory
        x2 = 0
        y2 = 0
        for i in np.arange(0, len(self.poses), 1):
            x1 = x2
            y1 = y2
            cx = self.poses[i].tx
            cy = self.poses[i].ty
            x2 = int((cx - self.min_x)*self.ppi_x)
            y2 = int((cy - self.min_y)*self.ppi_y)
            if i > 0:
                cv2.line(self.img, (x1, y1), (x2, y2), (0, 0, 255), 1) 

        for [x, y] in self.pnts:
             x = int((x - self.min_x)*self.ppi_x)
             y = int((y - self.min_y)*self.ppi_y)
                #print(f'({pose.tx}, {pose.ty}, {pose.th})')
             cv2.rectangle(self.img, (x, y), (x+1, y+1), (255, 255, 0), thickness=1)
            
        #pnts = []
        #elf.img.fill(0)
        #for point in scan:
        #    pnts.append(pose.getGlobalPoint(point.x, point.y))
        #    for [x, y] in pnts:
        #        x = int((x - self.min_x)*self.ppi_x)
        #        y = int((y - self.min_y)*self.ppi_y)
                #print(f'({pose.tx}, {pose.ty}, {pose.th})')
        #        cv2.rectangle(self.img, (x, y), (x+1, y+1), (255, 255, 0), thickness=1)
                
        cv2.imshow('Frame', self.img)
        
        if (cv2.waitKey(1) == 27):
            cv2.destroyAllWindows()
            return False
        else:
            return True
