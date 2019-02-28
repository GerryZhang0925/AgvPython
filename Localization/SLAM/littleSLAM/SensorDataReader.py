import LPoint2D
import Pose2D
import math

MIN_SCAN_RANGE = 0.1
MAX_SCAN_RANGE = 6

class SensorDataReader:
    def __init__(self, fname, offset=180):
        self.offset = offset
        self.file   = 0
        self.file   = open(fname, 'r')
        
    def setOffset(self, offset):
        self.offset = offset
        
    def openScanFile(self, fname):
        if self.file != 0:                                                                                                      
            self.file.close()
            self.file = 0
        self.file = open(fname, 'r')

    def closeScanFile(self):
        if self.file != 0:
            self.file.close()
            self.file = 0

    def loadScan(self, timeid):
        if self.file == 0:
            print('Scan file has not been opened.')
            return False, [], []
        else:
            line = self.file.readline()
            # Retrive one line as a list
            items = line.rstrip().split(' ')
                
            # Check if it is a avalible line
            if (items[0] != 'LASERSCAN'):
                return False, [], []
                
            # The number of scan points
            num = int(items[4])
            for i in range(5):
                items.pop(0)

            result = []            # Read Scan Data
            for i in range(num):
                
                if (len(items)):
                    angle = float(items.pop(0))
                else:
                    break
                
                if (len(items)):
                    distance = float(items.pop(0))
                else:
                    break

                angle += self.offset
                if (distance <= MIN_SCAN_RANGE or distance >= MAX_SCAN_RANGE):
                    continue

                lp = LPoint2D.LPoint2D()
                lp.setSid(timeid)
                lp.calXY(distance, angle)

                result.append(lp)

            # Read Odometry Data
            if (len(items)):
                tx = float(items.pop(0))
            else:
                return False, result, []

            if (len(items)):
                ty = float(items.pop(0))
            else:
                return False, result, []

            if (len(items)):
                th = float(items.pop(0))
            else:
                return False, result, []

            th = th * 180.0 / math.pi
            pose = Pose2D.Pose2D(tx, ty, th)
            pose.calRmat()
            
            return True, result, pose
