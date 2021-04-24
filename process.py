from imutils.video import VideoStream
from imutils.video import FPS
from imutils.object_detection import non_max_suppression
import imutils
import time
import numpy as np
import cv2
import math

# Choose video file
vs = cv2.VideoCapture("./JAAD_clips/video_0002.mp4")

# Init fps throughput estimator
fps = None

# Init HOG descriptor/person detector
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Different approach...
net = cv2.dnn.readNetFromCaffe("MobileNetSSD_deploy.prototxt.txt", "MobileNetSSD_deploy.caffemodel")

fps = FPS().start()

threshold = 0.1

colorsCount = 0
COLORS = np.random.uniform(0, 255, size=(50, 3))

#(last known centroid for each person, # of frames since last appearance) delete after leaving the image?
PEOPLE = []

maxMissingFrames = 30

maxDist = 0.9 # adjusted for size of bounding box

f = 0

# Loop over frames
while True:
    frame = vs.read()[1]
    if frame is None:
        break

    #print(frame.shape)
    # Resize frame and grab dimensions
    frame = imutils.resize(frame, width=800) # Potential hyperparam?
    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)

    net.setInput(blob)
    detections = net.forward()

    # Delete people that leave, count number of frames missing
    PEOPLE = [p for p in PEOPLE if p[1] < maxMissingFrames]
    for p in PEOPLE:
            p[1] += 1
    print("frame " + str(f))
    f += 1
    # Loop over detectitons
    count = 0
    toAdd = []
    for i in np.arange(0, detections.shape[2]):
        idx = int(detections[0, 0, i, 1])
        if 15 != idx:
            continue
        confidence = detections[0, 0, i, 2]
        if confidence > threshold:
            print("count: " + str(count))
            count += 1
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
            (startX, startY, endX, endY) = box.astype("int")

            # Decide
            currCenter = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
            minC = 100000000
            minP = None
            # Search to find minimum distance person
            for p in range(len(PEOPLE)):
                pCenter = (int(PEOPLE[p][0][0] + (PEOPLE[p][0][2] - PEOPLE[p][0][0]) / 2), int(PEOPLE[p][0][1] + (PEOPLE[p][0][3] - PEOPLE[p][0][1]) / 2))
                dist = math.sqrt((pCenter[0] - currCenter[0])**2 + (pCenter[1] - currCenter[1])**2)
                if dist < minC:
                    minC = dist
                    minP = p
            # See if 'minimum' is too far (i.e. must be new Person)
            #print("minC: " + str(minC))
            #print("comp: " + str(maxDist * abs(endY - startY)))
            if minC > maxDist * abs(endY - startY): 
                minP = None
            # If 'new' Person
            if minP == None:
                toAdd.append([(startX, startY, endX, endY), 0, COLORS[colorsCount], colorsCount])
                #print("person " + str(colorsCount) + " added")
                colorsCount += 1
            # If 'existing' Person
            else:
                #pCenter = (int(PEOPLE[p][0][0] + (PEOPLE[p][0][2] - PEOPLE[p][0][0]) / 2), int(PEOPLE[p][0][1] + (PEOPLE[p][0][3] - PEOPLE[p][0][1]) / 2))
                #print("person " + str(PEOPLE[p][3]) + " moved from " + str(pCenter) + " to " + str(currCenter))
                PEOPLE[minP][0] = (startX, startY, endX, endY)
                PEOPLE[minP][1] = 0
    
    for p in toAdd:
        PEOPLE.append(p)
    

    # draw predictions on the frame
    for p in PEOPLE:
        label = "{} {}: {:.2f}%".format("Person", p[3],
            confidence * 100)
        (startX, startY, endX, endY) = p[0]
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            p[2], 2)
        y = startY - 15 if startY - 15 > 15 else startY + 15
        cv2.putText(frame, label, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, p[2], 2)

    """
    (rects, weights) = hog.detectMultiScale(frame, winStride=(4,4), padding=(8,8), scale=1.05) # Hyperparameters

    for (x, y, w, h) in rects:
        cv2.rectangle(frame, (x,y), (x + w, y + h), (0, 255, 0), 2)
    
    # apply non-maxima suppression to the bounding boxes using a
	# fairly large overlap threshold to try to maintain overlapping
	# boxes that are still people
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)

    # draw the final bounding boxes
    for (xA, yA, xB, yB) in pick:
        cv2.rectangle(frame, (xA, yA), (xB, yB), (0, 255, 0), 2)
    """


    fps.update()
    fps.stop()

    # Init display information

    info = [
        # ("Tracker", tracker_name),
        # ("Success", "Yes" if success else "No"),
        ("FPS", "{:.2f}".format(fps.fps())),
    ]

    # loop over info tuples and draw onto frame
    for (i, (k, v)) in enumerate(info):
            text = "{}: {}".format(k, v)
            cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
 
    # show output frame
    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
vs.release()
cv2.destroyAllWindows()