from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import time
import cv2

OPENCV_OBJECT_TRACKERS = {
	"csrt": cv2.TrackerCSRT_create,
	"kcf": cv2.TrackerKCF_create,
	"boosting": cv2.TrackerBoosting_create,
	"mil": cv2.TrackerMIL_create,
	"tld": cv2.TrackerTLD_create,
	"medianflow": cv2.TrackerMedianFlow_create,
	"mosse": cv2.TrackerMOSSE_create
}

# Choose tracker
tracker_name = "boosting"
tracker = OPENCV_OBJECT_TRACKERS[tracker_name]()
#tracker = TrackerMOSSE_create()

# Initialize bounding box
initBB = None

# Choose video file
vs = cv2.VideoCapture("./JAAD_clips/video_0001.mp4")

# Init fps throughput estimator
fps = None

# Loop over frames
while True:
    frame = vs.read()[1]
    if frame is None:
        break
    # Resize frame and grab dimensions
    frame = imutils.resize(frame, width=500) # Potential hyperparam?
    (H, W) = frame.shape[:2]

    # If currently tracking an object
    if initBB is not None:
        # grab new BB coords
        (success, box) = tracker.update(frame)
        if success:
            (x, y, w, h) = [int(v) for v in box]
            cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)

        # Update FPS counter
        fps.update()
        fps.stop()

        # Init display information
        info = [
            ("Tracker", tracker_name),
            ("Success", "Yes" if success else "No"),
            ("FPS", "{:.2f}".format(fps.fps())),
        ]

        # loop over info tuples and draw onto frame
        for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # show output frame
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("s"):
            initBB = cv2.selectROI("Frame", frame, fromCenter=False, showCrosshair=True)

            tracker.init(frame,initBB)
            fps = FPS().start()
    elif key == ord("q"):
        break

vs.release()
cv2.destroyAllWindows()