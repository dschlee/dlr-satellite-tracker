from collections import deque
import numpy as np
import cv2
import imutils
import time


# def calculate_distance(p1, p2):
#     """p1 and p2 in format (x1,y1) and (x2,y2) tuples"""
#     distance = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
#     return distance


# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture("./data/DLR_Satellite_Tracking_2.mp4")

lower_threshold = 160
upper_threshold = 255
buffer = 64
pts = deque(maxlen=buffer)

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

    # resize the frame, blur it, and convert it to the Grayscale color space
    frame = imutils.resize(frame, width=1000)
    # blurred = cv2.GaussianBlur(frame, (11, 11), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # construct a mask for the color "green", then perform a series of dilations
    # and erosions to remove any small blobs left in the mask
    mask = cv2.inRange(gray, lower_threshold, upper_threshold)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    # find contours in the mask and initialize the current
    # (x, y) center of the ball
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    satellite_center = None

    (frame_h, frame_w) = frame.shape[:2]
    frame_center = (frame_w // 2, frame_h // 2)
    cv2.circle(frame, (frame_center), 1, (255, 255, 0), -1)

    cv2.putText(
        img=frame,
        text="Distance to frame center:",
        org=(20, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text="Brightness satellite:",
        org=(20, 130),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text="Brightness avg:",
        org=(20, 230),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    # caluclate the average brightness of the frame (excluding the black borders)
    avg_brightness_per_row = np.mean(frame[:, 278:724], axis=0)
    avg_brightness = np.mean(avg_brightness_per_row, axis=0)
    avg_brightness, _, _ = tuple(avg_brightness.astype(int))

    cv2.putText(
        img=frame,
        text=f"{avg_brightness}",
        org=(20, 260),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use it to compute the minimum
        # enclosing circle and centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        satellite_center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
        # only proceed if the radius meets a minimum size
        if radius > 1:

            # get the brightness of the satellite center
            satellite_brightness, _, _ = frame[satellite_center[1]][satellite_center[0]]

            # draw the circle and centroid on the frame,
            # then update the list of tracked points
            cv2.circle(
                img=frame,
                center=(int(x), int(y)),
                radius=int(radius),
                color=(0, 255, 255),
                thickness=2,
            )
            cv2.circle(
                img=frame,
                center=satellite_center,
                radius=1,
                color=(0, 0, 255),
                thickness=-1,
            )

            # calculate the distance of the satellite center to the frame center
            # and display it the frame
            distance_satellite_to_frame_center = (
                (satellite_center[0] - frame_center[0]) ** 2
                + (satellite_center[1] - frame_center[1]) ** 2
            ) ** 0.5

            cv2.putText(
                img=frame,
                text=f"{int(distance_satellite_to_frame_center)} pts",
                org=(20, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            )

            cv2.putText(
                img=frame,
                text=f"{satellite_brightness}",
                org=(20, 160),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            )

    # update the points queue
    pts.appendleft(satellite_center)

    # loop over the set of tracked points
    for i in range(1, len(pts)):
        # if either of the tracked points are None, ignore them
        if pts[i - 1] is None or pts[i] is None:
            continue
        # otherwise, compute the thickness of the line and draw the connecting lines
        thickness = int(np.sqrt(buffer / float(i + 1)) * 1)
        cv2.line(
            img=frame,
            pt1=pts[i - 1],
            pt2=pts[i],
            color=(0, 0, 255),
            thickness=thickness,
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


# When everything done, release the video capture object
cap.release()

# Closes all the frames
cv2.destroyAllWindows()
