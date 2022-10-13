from importlib import import_module
import cv2 as cv
import numpy as np
from track import *
import json

f = open('settings.json')
data = json.load(f)
for i in data['positions']:
    print(i)
f.close()
#from gpiozero import CPUTemperature

#cpu = CPUTemperature()
offset = 15
measurementInterval = 1
error = False
kernel = np.ones((9,9),np.uint8)

scale_percent = 50
width = 1280
height = 720

roi_pos_x = 100
roi_pos_y = 400
roi_height = 60
roi_width = 900
crop_size = 15
step = 1
        

Tracking = ObjectTracking()


def get_center_point(contour):
    M = cv.moments(contour)
    cX = int(M["m10"] / M["m00"])
    cY = int(M["m01"] / M["m00"])
    return [cX, cY+offset]

def run(mode):
    global roi_pos_y
    global roi_pos_x
    global step
    global roi_height
    global roi_width
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
            preview = cv.rectangle(preview,(roi_pos_x-2,roi_pos_y-2),(roi_pos_x+roi_width+2,roi_pos_y+roi_height+2),(0,0,255),2)
            frame1 = frame1[roi_pos_y:roi_pos_y+roi_height, roi_pos_x:roi_pos_x+roi_width]
            #test = cv.resize(frame1, )
            gray = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
            blur = cv.GaussianBlur(gray, (17, 17), 0)

            thresh = cv.adaptiveThreshold(blur,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,91,8)

            crop = thresh[crop_size:roi_height-crop_size, 0:1000]
            crop = cv.dilate(crop, kernel, iterations = 2)
            contours, _ = cv.findContours(crop, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
            detected_contours = []
            
            for contour in contours:
                x,y,w,h = cv.boundingRect(contour)
                aspect_ratio = float(w)/h
                if cv.contourArea(contour) > 1000 and aspect_ratio < 15 and aspect_ratio > 5:
                    detected_contours.append(contour)

            midpoints = []

            try:
                for contour in detected_contours:
                    midpoints.append(get_center_point(contour))
            except ZeroDivisionError:
                print("zeroDivision")
            if len(midpoints) == 1:
                point = midpoints[0]
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(point[0]+roi_pos_x-20, point[1]+roi_pos_y-40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),3,cv.LINE_AA)
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(point[0]+roi_pos_x-20, point[1]+roi_pos_y-40),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
            if len(midpoints) == 2:
                left_side = min(midpoints, key=itemgetter(0))
                right_side = max(midpoints, key=itemgetter(0))
                
                cv.putText(preview,"ID:"+str(Tracking.objectCount-1),(left_side[0]+roi_pos_x-20, left_side[1]+roi_pos_y-40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),3,cv.LINE_AA)
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(right_side[0]+roi_pos_x-20, right_side[1]+roi_pos_y-40),cv.FONT_HERSHEY_PLAIN,2,(0,0,0),3,cv.LINE_AA)
                
                cv.putText(preview,"ID:"+str(Tracking.objectCount-1),(left_side[0]+roi_pos_x-20, left_side[1]+roi_pos_y-40),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
                cv.putText(preview,"ID:"+str(Tracking.objectCount),(right_side[0]+roi_pos_x-20, right_side[1]+roi_pos_y-40),cv.FONT_HERSHEY_PLAIN,2,(255,255,255),2,cv.LINE_AA)
            
            
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
            if roi_pos_y > step:
                roi_pos_y = roi_pos_y - step
            continue
        elif k==115:  #down
            if roi_pos_y+roi_height < height-step:
                roi_pos_y = roi_pos_y + step
            continue
        elif k==97:  # left
            if roi_pos_x > step:
                roi_pos_x = roi_pos_x - step
            continue
        elif k==100:  # right
            if roi_pos_x+roi_width < width-step:
                roi_pos_x = roi_pos_x + step
            continue
        elif k==45:  # roi_height smaller
            if roi_height > 5:
                roi_height -= 1
            continue
        elif k==43:  # roi_height bigger
            if roi_height < 100:
                roi_height +=1
            continue
        elif k != -1:
            print(k)

    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    run("demo")
    #run("live")


