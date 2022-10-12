import cv2 as cv
import  numpy as np



cap = cv.VideoCapture('welle.mp4')
cap.set(cv.CAP_PROP_POS_FRAMES, 1200)

# create a dictionary of all trackers in OpenCV that can be used for tracking
OPENCV_OBJECT_TRACKERS = {
	"csrt": cv.legacy.TrackerCSRT_create,
	"kcf": cv.legacy.TrackerKCF_create,
	"boosting": cv.legacy.TrackerBoosting_create,
	"mil": cv.legacy.TrackerMIL_create,
	"tld": cv.legacy.TrackerTLD_create,
	"medianflow": cv.legacy.TrackerMedianFlow_create,
	"mosse": cv.legacy.TrackerMOSSE_create
}


# Create MultiTracker object
trackers = cv.legacy.MultiTracker_create()

while True:
    frame = cap.read()[1]

    if frame is None:
        break
    frame = cv.resize(frame,(1280,720))

    (success, boxes) = trackers.update(frame)
    #print(success,boxes)
    # loop over the bounding boxes and draw then on the frame
    if success == False:
        bound_boxes = trackers.getObjects()
        idx = np.where(bound_boxes.sum(axis= 1) != 0)[0]
        bound_boxes = bound_boxes[idx]
        trackers = cv.legacy.MultiTracker_create()
        for bound_box in bound_boxes:
            trackers.add(tracker,frame,bound_box)

    for i,box in enumerate(boxes):
        (x, y, w, h) = [int(v) for v in box]
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(frame,'TRACKING',(x+10,y-3),cv.FONT_HERSHEY_PLAIN,1.5,(255,255,0),2)

    cv.imshow('Frame', frame)
    k = cv.waitKey(30)

    if k == ord("s"):
        # select the bounding box of the object we want to track (make
        # sure you press ENTER or SPACE after selecting the ROI)
        roi = cv.selectROI("Frame", frame, fromCenter=False,
                            showCrosshair=True)
        # create a new object tracker for the bounding box and add it
        # to our multi-object tracker
        tracker = OPENCV_OBJECT_TRACKERS['mosse']()
        trackers.add(tracker, frame, roi)


cap.release()
cv.destroyAllWindows()