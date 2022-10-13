from operator import itemgetter
import math
import time

class Tracker:
    failureThreshold = 2.0
    failureValues = []
    errorMode = False
    id = None
    lastPos = None
    lastTime = None
    currentPos = None
    velocity = None

    def __init__(self,x,y,id):
        self.id = id
        self.currentPos = [x,y]
        self.lastTime = time.time()

    def update(self,x,y):
        global error
        if self.errorMode == False:
            now = time.time()
            self.lastPos = self.currentPos
            self.currentPos = [x,y]
            self.calcVelocity(now)
            self.lastTime = now
            error = False
        else:
            print(f"ERROR FROM OBJECT {self.id}! SHUT DOWN -> GPIO")
            
            error = True


    def printPos(self):
        print(self.currentPos)

    def calcVelocity(self, now):
        distance = math.sqrt(((int(self.lastPos[0])-int(self.currentPos[0]))**2)+((int(self.lastPos[1])-int(self.currentPos[1]))**2) )
        timedelta = (now-self.lastTime)
        velocity = distance / timedelta
        print(f"OBJECT {self.id}: velocity = {velocity:.4} px/s")
        self.velocity = velocity

        if velocity <= self.failureThreshold:
            self.failureValues.append(velocity)
        
        if len(self.failureValues) > 6:
            self.errorMode = True
        
        if velocity > self.failureThreshold and self.errorMode != True:
            self.failureValues = []


    def getVelocity(self):
        return self.velocity

class ObjectTracking:
    lastTime = None
    elapsedseconds = 0
    objectCount = None
    trackers = []
    

    def __init__(self):
        self.lastTime = time.time()
        self.objectCount = 0

    def update(self, midpoints):

        if len(midpoints) == 1 and len(self.trackers) == 1:
            point = midpoints[0]
            self.trackers[0].update(point[0],point[1])
            
        if len(midpoints) == 2 and len(self.trackers) == 2:
            left_side = min(midpoints, key=itemgetter(0))
            right_side = max(midpoints, key=itemgetter(0))
            self.trackers[0].update(left_side[0],left_side[1])
            self.trackers[1].update(right_side[0],right_side[1])

        if len(midpoints) > len(self.trackers):
            self.attachTracker(min(midpoints, key=itemgetter(0)))

        if len(midpoints) < len(self.trackers):
            self.detatchTracker()

    def attachTracker(self, newPoint):
        #objectcount acts as id
        self.objectCount +=1
        self.trackers.append(Tracker(newPoint[0],newPoint[1],self.objectCount))
        print(f"objectcounter: {self.objectCount}")

    def detatchTracker(self):
        self.trackers.pop(0)

    def getTrackers(self):
        return self.trackers