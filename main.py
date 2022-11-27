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
    useFullscreen = None
    stream = None
    Notifier = None
    lastTime = None
    elapsedseconds = 0
    objectCount = None
    trackers = []
    mode = None
    file = ""
    framewidth = 100
    frameheight = 100
    roi_pos_x = 100
    roi_pos_y = 100
    roi_height = 100
    roi_width = 800

    def __init__(self, _useNotification=False, _mode="demo", _file="", _framewidth=640, _frameheight=480, _useFullscreen=False):
        if _useNotification:
            self.Notifier = CVNotifier()
            self.Notifier.newMessage("using notification system", "Info")
            self.Notifier.newMessage("WASD to move the ROI", "Instruction")
            self.Notifier.newMessage(
                "+ and - to change ROI height", "Instruction")
            self.Notifier.newMessage(
                ", and . to change ROI width", "Instruction")
            self.Notifier.newMessage("X to save configuration", "Instruction")
            self.Notifier.newMessage(
                platform.system() + " " + platform.machine(), "Warning")

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
        f.close()
        self.Notifier.newMessage("success.", "Info")

    def save_config(self):

        with open("config.json", "r") as jsonFile:
            data = json.load(jsonFile)
        data["positions"]["roi_pos_x"] = self.roi_pos_x
        data["positions"]["roi_pos_y"] = self.roi_pos_y
        data["positions"]["roi_height"] = self.roi_height
        data["positions"]["roi_width"] = self.roi_width
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

    def update(self, midpoints):

        if len(midpoints) == 1 and len(self.trackers) == 1:
            point = midpoints[0]
            self.trackers[0].update(point[0], point[1])

        if len(midpoints) == 2 and len(self.trackers) == 2:
            left_side = min(midpoints, key=itemgetter(0))
            right_side = max(midpoints, key=itemgetter(0))
            self.trackers[0].update(left_side[0], left_side[1])
            self.trackers[1].update(right_side[0], right_side[1])

        if len(midpoints) > len(self.trackers):
            self.attachTracker(min(midpoints, key=itemgetter(0)))

        if len(midpoints) < len(self.trackers):
            self.detatchTracker()

    def attachTracker(self, newPoint):
        self.objectCount += 1
        self.trackers.append(
            Tracker(newPoint[0], newPoint[1], self.objectCount))
        print(f"objectcounter: {self.objectCount}")

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
                self.Notifier.update(preview)

                preview = cv.rectangle(preview, (self.roi_pos_x-2, self.roi_pos_y-2),(self.roi_pos_x+self.roi_width+2, self.roi_pos_y+self.roi_height+2), (0, 0, 255), 1)
                frame1 = frame1[self.roi_pos_y:self.roi_pos_y+self.roi_height,self.roi_pos_x:self.roi_pos_x+self.roi_width]
                gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
                blur = cv.medianBlur(gray,9)

                resized = cv.resize(
                    blur, (blur.shape[1], 1), interpolation=cv.INTER_AREA)
                values = resized[0].tolist()
                
                canvas = np.zeros((255, blur.shape[1], 3), np.uint8)

                #for index in range(len(values)-1):
                #    cv.line(preview,(index,255-values[index]),(index+1,255-values[index+1]),(255,255,255),1,cv.LINE_AA)
                
                difarr = []
                difarrabs = []
                for index in range(len(values)-3):
                    dif = values[index+3]-values[index]
                    difarr.append(dif)
                    difarrabs.append(abs(dif))

                #for index in range(len(difarr)-1):
                #    cv.line(preview,(index,127-difarr[index]),(index+1,127-difarr[index+1]),(255,255,0),1,cv.LINE_AA)

                for index in range(len(difarrabs)-1):
                    cv.line(preview,(index,127-difarrabs[index]),(index+1,127-difarrabs[index+1]),(0,0,255),1,cv.LINE_AA)

                canvas = cv.resize(canvas,(canvas.shape[1]*2,canvas.shape[0]*2), interpolation=cv.INTER_AREA)


                if platform.system() == "Linux" and platform.machine() == "armv7l":
                    cv.putText(preview, "CPU:"+str(int(cpu.temperature))+"Celsius",(1000, 50), cv.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2, cv.LINE_AA)
                if self.useFullscreen == True:
                    cv.namedWindow("preview", cv.WINDOW_NORMAL)
                    cv.setWindowProperty("preview", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

                cv.imshow("blur", blur)
                cv.imshow("preview", preview)
                cv.imshow("Frame", frame1)
                cv.imshow("resized", resized)
                cv.imshow("canvas", canvas)


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
            elif k != -1:
                print(k)
        self.stream.stop()
        cv.destroyAllWindows()


class CVMessage:

    messageType = None
    content = "test"
    colors = {"Error": (0, 0, 255), "Warning": (0, 255, 255), "Info": (
        0, 255, 0), "Instruction": (255, 0, 255), "Config": (255, 0, 0)}
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

        for message in self.messageInstances:
            if message != "None":
                cv.putText(screen, message.content, (self.position_x, self.position_y+self.textSize *
                           20*i), self.font, self.textSize, message.color, self.textSize, self.cvLine)
                i += 1

class CVGraph:

    position_x = 2
    position_y = 0
    cvLine = cv.LINE_AA

    def __init__(self):
            pass
            # for i in range(self.maxMessages):
            #    self.attachMessage(CVMessage(" ", "Info"))

    def setPosition(self, x, y):
        self.position_x = x
        self.position_y = y

    def update(self, screen):
        i = 1
        if len(self.messageInstances) >= self.maxMessages:
            self.detatchMessage()
class Tracker:

    failureThreshold = 2.0
    failureValues = []
    errorMode = False
    id = None
    lastPos = None
    lastTime = None
    currentPos = None
    velocity = None

    def __init__(self, x, y, id):
        self.id = id
        self.currentPos = [x, y]
        self.lastTime = time.time()

    def update(self, x, y):
        global error
        if self.errorMode == False:
            now = time.time()
            self.lastPos = self.currentPos
            self.currentPos = [x, y]
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
        distance = math.sqrt(((int(self.lastPos[0])-int(self.currentPos[0]))**2)+(
            (int(self.lastPos[1])-int(self.currentPos[1]))**2))
        timedelta = (now-self.lastTime)
        velocity = distance / timedelta

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

    Tracking = ObjectTracking(_useNotification=True, _mode="live",
                              _file="v4l2_example_crop.mp4", _useFullscreen=False)
    Tracking.loop()
