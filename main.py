from multiprocessing import Event
from operator import itemgetter
import math
import time
import cv2 as cv
import json
import numpy as np
from threading import Thread
import platform

if platform.system() == "Linux" and platform.machine() == "armv7l":
    import RPi.GPIO as GPIO
    GPIO.setmode(GPIO.BCM)
    GPIO.setup(17, GPIO.OUT)
    GPIO.output(17, False)
    from gpiozero import CPUTemperature
    cpu = CPUTemperature()

measurementInterval = 1
error = False
kernel_d = np.ones((7, 21), np.uint8)
kernel_e = np.ones((15, 7), np.uint8)
crop_size = 5
step = 1
offset = 15

def get_center_point(contour):
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY+crop_size]


class StreamThread:

    def __init__(self, src=0):
        self.stream = cv.VideoCapture(src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False

    def start(self):
        Thread(target=self.update, args=()).start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            (self.grabbed, self.frame) = self.stream.read()

    def read(self):
        return self.frame

    def stop(self):
        self.stopped = True


class ObjectTracking:
    #out = cv.VideoWriter('output.mp4', -1, 20.0, (640,480))
    failureFrameSaved = False
    thresholdrate = 30
    useFullscreen = None
    stream = None
    Notifier = None
    Graph = None
    lastTime = None
    elapsedseconds = 0
    objectCount = None
    trackers = []
    mode = None
    graph_enabled = False
    tracking_enabled = False
    clustermidpoints = []
    file = ""
    framewidth = 100
    frameheight = 100
    roi_pos_x = 100
    roi_pos_y = 100
    roi_height = 100
    roi_width = 800

    def __init__(self, _useNotification=False,_useGraph=False, _mode="demo", _file="", _framewidth=640, _frameheight=480, _useFullscreen=False):
        if _useNotification:
            self.Notifier = CVNotifier()
            self.Notifier.newMessage("WASD to move the ROI", "Instruction")
            self.Notifier.newMessage("G to toggle graph", "Instruction")
            self.Notifier.newMessage("Q to toggle tracking", "Instruction")
            self.Notifier.newMessage("+ and - to change ROI height", "Instruction")
            self.Notifier.newMessage(", and . to change ROI width", "Instruction")
            self.Notifier.newMessage("o and l to change threshold value", "Instruction")
            self.Notifier.newMessage("X to save configuration", "Instruction")
            self.Notifier.newMessage("platform: "+platform.system(), "Warning")
            self.Notifier.newMessage("machine: "+platform.machine(), "Warning")


        self.Graph = CVGraph()
        self.graph_enabled = False

        self.useFullscreen = _useFullscreen
        self.framewidth = _framewidth
        self.frameheight = _frameheight
        self.file = _file
        self.mode = _mode
        self.load_config()
        self.lastTime = time.time()
        self.objectCount = 0

        if self.mode == "demo":
            self.cap = cv.VideoCapture(_file)
            self.cap.set(cv.CAP_PROP_POS_FRAMES, 1400)
        else:
            self.stream = StreamThread(src=1).start()

    def load_config(self):
        self.Notifier.newMessage("loading config.", "Info")
        f = open('config.json')
        data = json.load(f)
        positions = data['positions']
        self.roi_pos_x = positions["roi_pos_x"]
        self.roi_pos_y = positions["roi_pos_y"]
        self.roi_height = positions["roi_height"]
        self.roi_width = positions["roi_width"]
        self.thresholdrate = positions["thresholdrate"]
        f.close()
        self.Notifier.newMessage("success.", "Info")

    def save_config(self):

        with open("config.json", "r") as jsonFile:
            data = json.load(jsonFile)
        data["positions"]["roi_pos_x"] = self.roi_pos_x
        data["positions"]["roi_pos_y"] = self.roi_pos_y
        data["positions"]["roi_height"] = self.roi_height
        data["positions"]["roi_width"] = self.roi_width
        data["positions"]["thresholdrate"] = self.thresholdrate
        with open("config.json", "w") as jsonFile:
            json.dump(data, jsonFile, indent=4)
        string = f"config saved:"
        self.Notifier.newMessage(string, "Config")
        string = f"roi_pos_x: {self.roi_pos_x}"
        self.Notifier.newMessage(string, "Config")
        string = f"roi_pos_y: {self.roi_pos_y}"
        self.Notifier.newMessage(string, "Config")
        string = f"roi_height: {self.roi_height}"
        self.Notifier.newMessage(string, "Config")
        string = f"roi_width: {self.roi_width}"
        self.Notifier.newMessage(string, "Config")
        string = f"thresholdrate: {self.thresholdrate}"
        self.Notifier.newMessage(string, "Config")

    def toggle_graph(self):
        
        if self.graph_enabled == False:
            self.graph_enabled = True
            self.Notifier.newMessage("graph enabled", "Warning")
        else:
            self.graph_enabled = False
            self.Notifier.newMessage("graph disabled", "Warning")
    def toggle_tracking(self):
        
        if self.tracking_enabled == False:
            self.tracking_enabled = True
            self.Notifier.newMessage("tracking enabled", "Warning")
        else:
            self.tracking_enabled = False
            self.Notifier.newMessage("tracking disabled", "Warning")
            self.clustermidpoints = []
    def update(self, values):
        proximityValue = 20
        detectedValues = []

        found_clusters = []
        for index in range(len(values)-10):
            val = abs(values[index+4]-values[index])
            if val > self.thresholdrate:
                detectedValues.append((index))

        for value in detectedValues:
            cluster = [value]
            detectedValues.remove(value)
            for pos in detectedValues:
                if pos-value<proximityValue:
                    cluster.append(pos)
            
            for element in cluster[1:]:
                detectedValues.remove(element)
            found_clusters.append(cluster)
        self.clustermidpoints = []
        for cluster in found_clusters:
            self.clustermidpoints.append(sum(cluster)//len(cluster))

        if len(self.clustermidpoints) > len(self.trackers):
            self.attachTracker(max(self.clustermidpoints))
        while len(self.clustermidpoints) < len(self.trackers):
            self.detatchTracker()

    def attachTracker(self, x):
        self.objectCount += 1
        self.trackers.append(Tracker(x, self.objectCount))

    def detatchTracker(self):
        self.trackers.pop(0)

    def getTrackers(self):
        return self.trackers

    def loop(self):
        while True:

            frame1 = self.stream.read()

            if frame1 is not None:
                now = time.time()
                preview = frame1
                preview = cv.rectangle(preview, (self.roi_pos_x-2, self.roi_pos_y-2),(self.roi_pos_x+self.roi_width+2, self.roi_pos_y+self.roi_height+2), (0, 0, 255), 1)
                frame1 = frame1[self.roi_pos_y:self.roi_pos_y+self.roi_height,self.roi_pos_x:self.roi_pos_x+self.roi_width]
                gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                blur = cv.medianBlur(gray,21)

                resized = cv.resize(blur, (blur.shape[1], 1), interpolation=cv.INTER_AREA)
                values = resized[0].tolist()

                if platform.system() == "Linux" and platform.machine() == "armv7l":
                    cv.putText(preview, "CPU:"+str(int(cpu.temperature))+"Celsius",(1000, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
                if self.useFullscreen == True:
                    cv.namedWindow("preview", cv.WINDOW_NORMAL)
                    cv.setWindowProperty("preview", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

                self.Notifier.update(preview)

                if self.graph_enabled:
                    self.Graph.update(preview,values,self.thresholdrate,blur)
                for index,clustermidpoint in enumerate(self.clustermidpoints):
                    try:
                        cv.putText(preview,str(self.trackers[index].id),(self.roi_pos_x+clustermidpoint-5,self.roi_pos_y-10), cv.FONT_HERSHEY_PLAIN, 1, (0,0,0), 3, cv.LINE_AA)
                        cv.putText(preview,str(self.trackers[index].id),(self.roi_pos_x+clustermidpoint-5,self.roi_pos_y-10), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv.LINE_AA)
                    except:
                        print("empty")
                    cv.drawMarker(preview,(self.roi_pos_x+clustermidpoint,self.roi_pos_y+self.roi_height//2),(0,0,0),cv.MARKER_CROSS, 15, 3,cv.LINE_AA)
                    cv.drawMarker(preview,(self.roi_pos_x+clustermidpoint,self.roi_pos_y+self.roi_height//2),(0,255,0),cv.MARKER_CROSS, 15, 1,cv.LINE_AA)
                cv.imshow("preview", preview)
                if self.failureFrameSaved == False and error == True:
                    cv.imwrite("failure.jpg", preview)
                    self.failureFrameSaved = True
                if self.tracking_enabled:
                    self.update(values)
                    if now - self.lastTime > measurementInterval:
                        for count, tracker in enumerate(self.trackers):
                            tracker.update(self.clustermidpoints[count])
                        
                        self.lastTime = now
                        self.elapsedseconds += measurementInterval
                #self.out.write(preview)
            else:
                print('no video')
                self.cap.set(cv.CAP_PROP_POS_FRAMES, 1400)
                continue
            k = cv.waitKey(50)
            if k == 27:    # Esc key to stop
                break
            elif k == 119:  # up
                if self.roi_pos_y > step:
                    self.roi_pos_y = self.roi_pos_y - step
                    self.Notifier.newMessage(f"Y:{self.roi_pos_y}", "Warning")
                continue
            elif k == 115:  # down
                if self.roi_pos_y+self.roi_height < self.frameheight-step:
                    self.roi_pos_y = self.roi_pos_y + step
                    self.Notifier.newMessage(f"Y:{self.roi_pos_y}", "Warning")
                continue
            elif k == 97:  # left
                if self.roi_pos_x > step:
                    self.roi_pos_x = self.roi_pos_x - step
                    self.Notifier.newMessage(f"X:{self.roi_pos_x}", "Warning")
                continue
            elif k == 100:  # right
                if self.roi_pos_x+self.roi_width < self.framewidth-step:
                    self.roi_pos_x = self.roi_pos_x + step
                    self.Notifier.newMessage(f"X:{self.roi_pos_x}", "Warning")
                continue
            elif k == 45:  # roi_height
                if self.roi_height > crop_size+1:
                    self.roi_height -= 1
                    self.Notifier.newMessage(
                        f"HEIGHT:{self.roi_height}", "Warning")
                continue
            elif k == 43:  # roi_height
                if self.roi_height < self.frameheight:
                    self.roi_height += 1
                    self.Notifier.newMessage(
                        f"HEIGHT:{self.roi_height}", "Warning")
                continue
            elif k == 46:  # roi_width
                if self.roi_width > 100:
                    self.roi_width -= 1
                    self.Notifier.newMessage(
                        f"WIDTH:{self.roi_width}", "Warning")
                continue
            elif k == 44:  # roi_width
                if self.roi_width < self.framewidth:
                    self.roi_width += 1
                    self.Notifier.newMessage(
                        f"WIDTH:{self.roi_width}", "Warning")
                continue
            elif k == 120:  # save roi configuration to config.json
                self.save_config()
                continue
            elif k == 103:
                self.toggle_graph()
                continue
            elif k == 111:
                self.thresholdrate +=1
                continue
            elif k == 108:
                self.thresholdrate -=1
                continue
            elif k == 113:
                self.toggle_tracking()
                continue
            elif k != -1:
                print(k)
        self.stream.stop()
        cv.destroyAllWindows()

class CVMessage:

    messageType = None
    content = "test"
    colors = {"Error": (0, 0, 255), "Warning": (0, 255, 255), "Info": (
        0, 255, 0), "Instruction": (255, 0, 255), "Config": (255, 255, 0)}
    color = None

    def __init__(self, _content, _type,):
        self.content = _content
        self.messageType = _type
        self.color = self.colors[self.messageType]

class CVNotifier:

    textSize = 1
    font = cv.FONT_HERSHEY_PLAIN
    position_x = 2
    position_y = 0
    maxMessages = 10
    messageInstances = []
    cvLine = cv.LINE_AA

    def __init__(self):
        pass
        # for i in range(self.maxMessages):
        #    self.attachMessage(CVMessage(" ", "Info"))

    def setPosition(self, x, y):
        self.position_x = x
        self.position_y = y

    def setTextSize(self, size):
        self.textSize = size

    def newMessage(self, string, type):
        message = CVMessage(string, type)
        self.attachMessage(message)

    def attachMessage(self, cvmessage):
        self.messageInstances.append(cvmessage)

    def detatchMessage(self):
        self.messageInstances.pop(0)

    def update(self, screen):
        i = 1
        if len(self.messageInstances) >= self.maxMessages:
            self.detatchMessage()
        #cv.rectangle(screen,(self.position_x,self.position_y),(255, 145),(0,0,0),-1)
        for message in self.messageInstances:
            if message != "None":
                cv.putText(screen, message.content, (self.position_x, self.position_y+self.textSize *15*i), self.font, self.textSize, (0,0,0), self.textSize+2, self.cvLine)
                cv.putText(screen, message.content, (self.position_x, self.position_y+self.textSize *15*i), self.font, self.textSize, message.color, self.textSize, self.cvLine)
                i += 1

class CVGraph:

    position_x = 0
    position_y = 479
    cvLine = cv.LINE_AA
    threshhold = 30

    def __init__(self):
        pass

    def setPosition(self, x, y):
        self.position_x = x
        self.position_y = y

    def update(self, screen, values,threshold,blur):
        self.threshhold = threshold
        difarr = []
        difarrabs = []
        for index in range(len(values)-4):
            dif = values[index+4]-values[index]
            difarr.append(dif)
            difarrabs.append(abs(dif))
        
        
        #cv.rectangle(screen,(self.position_x-2,self.position_y-140),(self.position_x+len(values),self.position_y+5),(0,0,0),-1,cv.LINE_AA)
        test = cv.resize(blur, (blur.shape[1], 50), interpolation=cv.INTER_AREA)
        test = cv.cvtColor(test,cv.COLOR_GRAY2BGR)
        x_offset=self.position_x
        y_offset=self.position_y-50
        screen[y_offset:y_offset+test.shape[0], x_offset:x_offset+test.shape[1]] = test
        
        for index in range(len(values)-1):
            pty1 = values[index]
            pty2 = values[index+1]
            cv.line(screen,(index,self.position_y-pty1),(index+1,self.position_y-pty2),(255,255,0),1,cv.LINE_AA)
        cv.line(screen,(self.position_x,self.position_y-self.threshhold),(self.position_x+len(values),self.position_y-self.threshhold),(0,255,0),1,cv.LINE_AA)
        cv.putText(screen,"absolute",(self.position_x+len(values),self.position_y-50), cv.FONT_HERSHEY_PLAIN, 1, (255,255,0), 1, cv.LINE_AA)
        cv.putText(screen,"threshold",(self.position_x+len(values),self.position_y-self.threshhold+4), cv.FONT_HERSHEY_PLAIN, 1, (0,255,0), 1, cv.LINE_AA)
        cv.putText(screen,"rate",(self.position_x+len(values),self.position_y), cv.FONT_HERSHEY_PLAIN, 1, (255,0,255), 1, cv.LINE_AA)
        for index in range(len(difarrabs)-5):
            cv.line(screen,(2+self.position_x+index,self.position_y-difarrabs[index]),(2+self.position_x+index+1,self.position_y-difarrabs[index+1]),(255,0,255),1,cv.LINE_AA)
        
        cv.line(screen,(self.position_x,self.position_y-self.threshhold),(self.position_x+len(values),self.position_y-self.threshhold),(0,255,0),1,cv.LINE_AA)

class Tracker:

    failureThreshold = 2.0
    failureValues = []
    errorMode = False
    id = None
    lastPos = None
    lastTime = None
    currentPos = None
    velocity = None

    def __init__(self, x, id):
        self.id = id
        self.currentPos = x
        self.lastTime = time.time()

    def update(self, x):
        global error
        if self.errorMode == False:
            now = time.time()
            self.lastPos = self.currentPos
            self.currentPos = x
            self.calcVelocity(now)
            self.lastTime = now
            error = False
        else:
            if platform.system() == "Linux" and platform.machine() == "armv7l":
                GPIO.output(17, True)
            string = f"ERROR FROM ID{self.id}"
            Tracking.Notifier.newMessage(string, "Error")

            error = True

    def calcVelocity(self, now):
        distance = self.lastPos-self.currentPos
        
        timedelta = (now-self.lastTime)

        try:
            velocity = distance / timedelta
        except(ZeroDivisionError):
            velocity = 999
        if velocity is not None:
            string = f"ID{self.id}: {velocity:.1F}px/s"
            Tracking.Notifier.newMessage(string, "Info")

        self.velocity = velocity

        if velocity <= self.failureThreshold:
            self.failureValues.append(velocity)

        if len(self.failureValues) > 3:
            self.errorMode = True

        if velocity > self.failureThreshold and self.errorMode != True:
            self.failureValues = []

    def getVelocity(self):
        return self.velocity


if __name__ == "__main__":

    Tracking = ObjectTracking(_useNotification=True, _mode="live", _file="v4l2_example_crop.mp4", _useFullscreen=False,_useGraph=True)
    Tracking.loop()
