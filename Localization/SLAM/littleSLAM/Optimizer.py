#!/usr/bin/env python

import math
import Pose2D

class Optimizer:
    def __init__(self, threshold=0.000001, ddist=0.00001, dth=0.00001):
        self.thre     = threshold     # threshold to decide whether stop optimization or not
        self.dd       = ddist         # step of movement
        self.dt       = dth           # step of rotation
        self.coef     = 0.00001       # coefficient to decide update step
        self.hugeval  = 10000

    # cost function
    def costfunc(self, tx, ty, th, cur, ref):
        costlimit = 1
        pn = 0
        nn = 0
        error = 0.0
        for i in range(len(cur)):
            if (i < len(ref)):
                cx = cur[i].x
                cy = cur[i].y
                x  = math.cos(th)*cx - math.sin(th)*cy + tx
                y  = math.sin(th)*cx + math.cos(th)*cy + ty

                # Calculate Distance
                edis = (x - ref[i].x)*(x - ref[i].x) + (y - ref[i].y)*(y - ref[i].y)

                if (edis <= costlimit*costlimit):
                    pn += 1

                error += edis
                nn += 1
        error = error / nn if nn > 0 else self.hugeval
        return (error * 100)
    

    # optimization
    def opt(self, initPose, cur, ref):
        tx = initPose.tx
        ty = initPose.ty
        th = initPose.th
        
        txmin = tx
        tymin = ty
        thmin = th

        costmin = self.hugeval
        costold = costmin
        cost = self.costfunc(tx, ty, th, cur, ref)
        
        while abs(costold - cost) > self.thre:
            costold = cost            
            
            # partial differetial
            dEtx = (self.costfunc(tx+self.dd, ty, th, cur, ref) - cost)/self.dd
            dEty = (self.costfunc(tx, ty+self.dd, th, cur, ref) - cost)/self.dd
            dEth = (self.costfunc(tx, ty, th+self.dt, cur, ref) - cost)/self.dt

            # Update estimated position
            tx -= self.coef * dEtx
            ty -= self.coef * dEty
            th -= self.coef * dEth

            cost = self.costfunc(tx, ty, th, cur, ref)

            if cost < costmin:
                costmin = cost
                txmin = tx
                tymin = ty
                thmin = th

        pose = Pose2D.Pose2D(txmin, tymin, thmin)
        return pose
