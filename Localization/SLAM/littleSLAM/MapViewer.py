#!/usr/bin/env python

import cv2
import LPoint2D
import Pose2D
import Optimizer
import numpy as np

class MapViewer:
    def __init__(self, width, height):
        self.width  = width
        self.height = height
        self.min_x  = -40.0
        self.max_x  = 25.0
        self.min_y  = 0.0
        self.max_y  = 30.0
        self.ppi_x  = self.width/(self.max_x - self.min_x)
        self.ppi_y  = self.height/(self.max_y - self.min_y)
        self.img    = np.zeros((self.height, self.width, 3))
        self.poses   = []          # Positions of robot cars
        self.pnts    = []          # Scan points
        self.prescan = []
        self.scan    = []
        self.cnt     = 0
        self.pos     = []
        self.rot     = 0
        #######################
        # Paramters for dispaly
        #######################
        self.grid            = 5
        self.colorBackground = (255, 255, 255)  # Color of background
        self.colorScanpoints = (255, 0,   0)    # Color of scan points
        self.colorTrajectory = (0,   0,   255)  # Color of trajectory
        self.colorGridline   = (0,   0,   0)   # Color of grid lines
        self.colorForeground = (0,   0,   0)    # Color of foreground 
        
    def setXRange(self, xmin, xmax):
        self.min_x  = xmin
        self.max_x  = xmax
        self.ppi_x  = self.width/(xmax - xmin)

    def setYRange(self, ymin, ymax):
        self.min_y  = ymin
        self.max_y  = ymax
        self.ppi_y  = self.height/(ymax - ymin)

    ########################################
    # Show Frame
    ########################################
    def showFrame(self):
        # show background
        self.img[:] = self.colorBackground

        # show grid lines
        x = self.min_x + self.grid
        while x < self.max_x:
            dx = int((x - self.min_x)*self.ppi_x)
            cv2.line(self.img, (dx, 0), (dx, self.height - 1), self.colorGridline, thickness=1)
            x += self.grid
            
        y = self.min_y + self.grid
        while y < self.max_y:
            dy = int((y - self.min_y)*self.ppi_y)
            cv2.line(self.img, (0, dy), (self.width - 1, dy), self.colorGridline, thickness=1)
            y += self.grid

        # show Caption
        if (len(self.pos) > 0):
            strMsg = str('Position: ({0:.2f}, {1:.2f})'.format(self.pos[0], self.pos[1]))
            cv2.putText(self.img, strMsg, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colorForeground, 2, 4)
            strMsg = str('Rotation: {0:.2f} rad'.format(self.rot))
            cv2.putText(self.img, strMsg, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, self.colorForeground, 2, 4)
            
        # Show Cloud
        for [x, y] in self.pnts:
             x = int((x - self.min_x)*self.ppi_x)
             y = int((y - self.min_y)*self.ppi_y)
                #print(f'({pose.tx}, {pose.ty}, {pose.th})')
             cv2.rectangle(self.img, (x, y), (x+1, y+1), self.colorScanpoints, thickness=2)

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
                cv2.line(self.img, (x1, y1), (x2, y2), self.colorTrajectory, 2)
                
        cv2.imshow('littleSLAM', self.img)

        # Display and wait for keys
        if (cv2.waitKey(1) == 27):
            cv2.destroyAllWindows()
            return False
        else:
            return True

    ########################################
    # Add Scan Data
    ########################################
    def addScans(self, scan, pose):
        self.cnt += 1;
        opt       = Optimizer.Optimizer()

        if (self.cnt % 10 == 1):
            for point in scan:
                self.pnts.append(pose.getGlobalPoint(point.x, point.y))
            
            self.prescan = self.scan
            self.scan = scan.copy()
            
            if len(self.prescan) > 0:
                #print(f'old({pose.tx}, {pose.ty}, {pose.th})')
                initpose = Pose2D.Pose2D(0.0, 0.0, 0.0)
                estpose = opt.opt(initpose, self.scan, self.prescan)
                pose = Pose2D.Pose2D(pose.tx+estpose.tx, pose.ty+estpose.ty, pose.th+estpose.th)
                #print(f'new({newpose.tx}, {newpose.ty}, {newpose.th})')
                self.poses.append(pose)
            else:
                self.poses.append(pose)
                #print(f'orig({pose.tx}, {pose.ty}, {pose.th})')

            self.pos = [pose.tx, pose.ty]
            self.rot = pose.th
                
        return self.showFrame()
