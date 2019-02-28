#!/usr/bin/env python

import SensorDataReader
import MapViewer
import numpy as np

def main():
    cnt = 0
    mapviewer       = MapViewer.MapViewer(800, 800)
    sensorReader    = SensorDataReader.SensorDataReader('corridor.lsc')
    ret, scan, pose = sensorReader.loadScan(cnt)
    
    while (ret):
        if (mapviewer.showScans(scan, pose) == False):
            break

        ret, scan, pose = sensorReader.loadScan(++cnt)
        
    sensorReader.closeScanFile()

if __name__ == '__main__':
    main()
