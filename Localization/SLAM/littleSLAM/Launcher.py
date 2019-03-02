#!/usr/bin/env python

import SensorDataReader
import MapViewer
import numpy as np

def main():
    cnt = 0
    mapviewer       = MapViewer.MapViewer(800, 500)
    sensorReader    = SensorDataReader.SensorDataReader('../../../Localization/data/corridor.lsc')
    ret, scan, pose = sensorReader.loadScan(cnt)
    
    while (ret):
        if (mapviewer.addScans(scan, pose) == False):
            break

        ret, scan, pose = sensorReader.loadScan(++cnt)
        
    sensorReader.closeScanFile()

if __name__ == '__main__':
    main()
