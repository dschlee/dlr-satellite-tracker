from collections import deque
import numpy as np
import cv2
import imutils
import time

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture("./data/DLR_Satellite_Tracking_1.mp4")

lower_threshold = 160
upper_threshold = 255
buffer = 64
pts = deque(maxlen=buffer)

# # allow the video file to warm up
# time.sleep(1.0)

# Check if video opened successfully
if cap.isOpened() == False:
    print("Error opening video file")

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None:
        break

    # resize the frame, blur it, and convert it to the Grayscale
    # color space
    frame = imutils.resize(frame, width=1000)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # construct a mask for the color "green", then perform
    # a series of dilations and erosions to remove any small
    # blobs left in the mask
    mask = cv2.inRange(gray, lower_threshold, upper_threshold)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 1:
            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(
                frame,
                center=(int(x), int(y)),
                radius=int(radius),
                color=(0, 255, 255),
                thickness=2,
            )
            cv2.circle(frame, center=center, radius=1, color=(0, 0, 255), thickness=-1)

    # update the points queue
    pts.appendleft(center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore
        # them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and
        # draw the connecting lines
        thickness = int(np.sqrt(buffer / float(i + 1)) * 1)
        cv2.line(
            frame, pt1=pts[i - 1], pt2=pts[i], color=(0, 0, 255), thickness=thickness
        )

    if ret == True:
        # Display the resulting frame
        cv2.imshow("Frame", frame)

        # Press Q on keyboard to exit
        if cv2.waitKey(25) & 0xFF == ord("q"):
            break

    # Break the loop
    else:
        break


# When everything done, release
# the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
