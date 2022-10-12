import cv2 as cv
import numpy as np
from operator import itemgetter
import math
import time
#from gpiozero import CPUTemperature

#cpu = CPUTemperature()
offset = 15
measurementInterval = 1
error = False
kernel = np.ones((9,9),np.uint8)

scale_percent = 50
width = 1280
height = 720

xVal = 100
yVal = 400
step = 1
roiY = 60
roiX = 900
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
        

Tracking = ObjectTracking()


def get_center_point(contour):
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY+offset]

def run(mode):
    global yVal
    global xVal
    global step
    global roiY
    global roiX
    if mode == "demo":
        cap = cv.VideoCapture("welle.mp4")
        cap.set(cv.CAP_PROP_POS_FRAMES, 1400)
        
    else:
        cap = cv.VideoCapture(0)
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv.CAP_PROP_FPS, 20)
    
    counter = 0


    while cap.isOpened():
        
        #print(counter)
        counter += 1
        ret, frame1 = cap.read()
        if ret:
            if mode == "live":
                #frame1 = cv.rotate(frame1, cv.ROTATE_180)
                pass

            
            now = time.time()

            
            
            preview = frame1
            preview = cv.rectangle(preview,(xVal-2,yVal-2),(xVal+roiX+2,yVal+roiY+2),(0,0,255),2)
            frame1 = frame1[yVal:yVal+roiY, xVal:xVal+roiX]
            #test = cv.resize(frame1, )
            gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (17, 17), 0)

            thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,81,8)

            crop = thresh[15:45, 0:1000]
            crop = cv.dilate(crop, kernel, iterations = 2)
            contours, _ = cv.findContours(crop, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            detected_contours = []
            
            for contour in contours:
                x,y,w,h = cv.boundingRect(contour)
                aspect_ratio = float(w)/h
                if cv.contourArea(contour) > 1000 and aspect_ratio < 8 and aspect_ratio > 5:
                    detected_contours.append(contour)

            midpoints = []

            try:
                for contour in detected_contours:
                    midpoints.append(get_center_point(contour))
            except ZeroDivisionError:
                print("zeroDivision")
            if len(midpoints) == 1:
                point = midpoints[0]
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(point[0]+xVal-20, point[1]+yVal-40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),3,cv.LINE_AA)
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(point[0]+xVal-20, point[1]+yVal-40),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
            if len(midpoints) == 2:
                left_side = min(midpoints, key=itemgetter(0))
                right_side = max(midpoints, key=itemgetter(0))
                
                cv.putText(preview,"ID:"+str(Tracking.objectCount-1),(left_side[0]+xVal-20, left_side[1]+yVal-40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),3,cv.LINE_AA)
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(right_side[0]+xVal-20, right_side[1]+yVal-40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),3,cv.LINE_AA)
                
                cv.putText(preview,"ID:"+str(Tracking.objectCount-1),(left_side[0]+xVal-20, left_side[1]+yVal-40),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(right_side[0]+xVal-20, right_side[1]+yVal-40),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
            
            
            if now - Tracking.lastTime > measurementInterval:
                Tracking.update(midpoints)
                Tracking.lastTime = now
                Tracking.elapsedseconds += measurementInterval

            activeTrackers = Tracking.getTrackers()
            if len(activeTrackers) > 0:
                for tracker in activeTrackers:
                    cv.drawMarker(frame1, (tracker.currentPos[0],tracker.currentPos[1]),(0, 0, 0),cv.MARKER_CROSS,20, thickness=2)
                    cv.drawMarker(frame1, (tracker.currentPos[0],tracker.currentPos[1]),(255, 255, 255),cv.MARKER_CROSS,20)

            if error == True:
                cv.putText(preview,"ALARM",(10,50),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv.LINE_AA)

            for contour in detected_contours:
                try:
                    (x, y, w, h) = cv.boundingRect(contour)
                    #cv.rectangle(frame1, (x, y), (x+w, y+h+offset), (0, 255, 0), 2)
                    cv.drawMarker(frame1, get_center_point(contour),(255, 0, 255),cv.MARKER_DIAMOND,35, thickness=2)
                except ZeroDivisionError:
                    print("zeroDivision")
                    
            #cv.putText(preview,"CPU:"+str(int(cpu.temperature))+"Celsius",(1000,50),cv.FONT_HERSHEY_PLAIN,2,(0,0,255),2,cv.LINE_AA)
            #cv.putText(frame1,str(len(Tracking.trackers)),(10,15),cv.FONT_HERSHEY_COMPLEX_SMALL,1,(0,0,255),1,cv.LINE_AA)
            #cv.namedWindow("preview", cv.WINDOW_NORMAL);
            #cv.setWindowProperty("preview", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN);
            cv.imshow("blur", blur)
            cv.imshow("preview", preview)
            cv.imshow("Threshold", thresh)
            cv.imshow("crop", crop)
            cv.imshow("Frame", frame1)

        else:
            print('no video')
            cap.set(cv.CAP_PROP_POS_FRAMES, 1400)
            continue
        k = cv.waitKey(20)
        if k==27:    # Esc key to stop
            break
        elif k==119:  # up
            if yVal > step:
                yVal = yVal - step
            continue
        elif k==115:  #down
            if yVal+roiY < height-step:
                yVal = yVal + step
            continue
        elif k==97:  # left
            if xVal > step:
                xVal = xVal - step
            continue
        elif k==100:  # right
            if xVal+roiX < width-step:
                xVal = xVal + step
            continue
        elif k==45:  # roiY smaller
            if roiY > 5:
                roiY -= 1
            continue
        elif k==43:  # roiY bigger
            if roiY < 60:
                roiY +=1
            continue
        elif k != -1:
            print(k)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run("demo")
    #run("live")


