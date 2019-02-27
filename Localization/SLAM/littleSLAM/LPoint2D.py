#!/usr/bin/env python

import math

class LPoint2D:
    def __init__(self):
        self.sid  = -1
        self.x    = 0.0
        self.y    = 0.0
        self.nx   = 0.0
        self.ny   = 0.0
        self.atd  = 0.0
        self.type = 'UNKNOWN'

    def setData(self, id, x, y):
        self.sid  = id
        self.x    = x
        self.y    = y

    def setXY(self, x, y):
        self.x = x
        self.y = y

    def calXY(self, dist, angle):
        a = angle * math.pi / 180.0
        self.x = dist * math.cos(a)
        self.y = dist * math.sin(a)

    def calXYi(self, dist, angle):
        a = angle * math.pi / 180.0
        self.x = dist * math.cos(a)
        self.y = -dist * math.sin(a)

    def setSid(self, id):
        self.sid = id

    def setAtd(self, atd):
        self.atd = atd

    def setType(self, type):
        self.type = type

    def setNormal(self, nx, ny):
        self.nx = nx
        self.ny = ny
        
