#!/usr/bin/env python

import math
import numpy as np

class Pose2D:
    def __init__(self):
        self.tx = 0.0
        self.ty = 0.0
        self.th = 0.0
        self.rmat = np.array([[1.0, 0.0], [0.0, 1.0]])
                
    def __init__(self, tx, ty, th):
        self.tx = tx
        self.ty = ty
        self.th = th
        a = self.th * math.pi / 180.0
        self.rmat = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def calRmat(self):
        a = self.th * math.pi / 180.0
        self.rmat = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])

    def getGlobalPoint(self, x, y):
        g_x = self.rmat[0,0]*x + self.rmat[0,1]*y + self.tx
        g_y = self.rmat[1,0]*x + self.rmat[1,1]*y + self.ty

        return [g_x, g_y]



        
        
