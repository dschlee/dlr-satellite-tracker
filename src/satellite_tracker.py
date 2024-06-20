from collections import deque
from pathlib import Path
import numpy as np
import cv2
import imutils

# Adjust this value to select the different video files
video_file_nr = 3

file_path = f"./data/DLR_Satellite_Tracking_{video_file_nr}.mp4"
file_name = Path(file_path).stem

print(file_path)
print(file_name)

# Create a VideoCapture object and read from input file
cap = cv2.VideoCapture(file_path)

lower_threshold = 160
upper_threshold = 255
buffer = 64
pts = deque(maxlen=buffer)
distance_satellite_to_center_deq = deque([])
satellite_brightness_deq = deque([])
avg_window_brightness_deq = deque([])

distance_satellite_to_center_avg = None
satellite_brightness_avg = None
avg_window_brightness_avg = None

fps = cap.get(cv2.CAP_PROP_FPS)
delay = int(1000 / fps)

# Check if video opened successfully
if cap.isOpened() == False:
    print("Error opening video file")

# size = (frame_width, frame_height)

out = cv2.VideoWriter(
    f"./data/{file_name}_output.mp4",
    cv2.VideoWriter_fourcc(*"MP4V"),
    20.0,
    (1000, 562),
)

# Read until video is completed
while cap.isOpened():

    # Capture frame-by-frame
    ret, frame = cap.read()

    # if we are viewing a video and we did not grab a frame,
    # then we have reached the end of the video
    if frame is None or not ret:
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
        text="Frame",
        org=(380, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text="Avg",
        org=(480, 30),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text="Distance to center [pixel]",
        org=(20, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text=f"{distance_satellite_to_center_avg}",
        org=(480, 60),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text="Satellite brightness [intensity]",
        org=(20, 90),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text=f"{satellite_brightness_avg}",
        org=(480, 90),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text="Avg window brightness [intensity]",
        org=(20, 120),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    # caluclate the average brightness of the frame (excluding the black borders)
    avg_window_brightness_per_row = np.mean(frame[:, 278:724], axis=0)
    avg_window_brightness_frame = np.mean(avg_window_brightness_per_row, axis=0)
    avg_window_brightness_frame, _, _ = tuple(avg_window_brightness_frame.astype(int))

    avg_window_brightness_deq.append(avg_window_brightness_frame)
    avg_window_brightness_avg = int(np.mean(avg_window_brightness_deq))

    cv2.putText(
        img=frame,
        text=f"{avg_window_brightness_frame}",
        org=(380, 120),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.6,
        color=(0, 255, 255),
        thickness=1,
        lineType=cv2.LINE_4,
    )

    cv2.putText(
        img=frame,
        text=f"{avg_window_brightness_avg}",
        org=(480, 120),
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
            satellite_brightness_frame, _, _ = frame[satellite_center[1]][
                satellite_center[0]
            ]

            satellite_brightness_deq.append(satellite_brightness_frame)
            satellite_brightness_avg = int(np.mean(satellite_brightness_deq))

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
            distance_satellite_to_center_frame = (
                (satellite_center[0] - frame_center[0]) ** 2
                + (satellite_center[1] - frame_center[1]) ** 2
            ) ** 0.5

            distance_satellite_to_center_deq.append(distance_satellite_to_center_frame)
            distance_satellite_to_center_avg = int(
                np.mean(distance_satellite_to_center_deq)
            )

            cv2.putText(
                img=frame,
                text=f"{int(distance_satellite_to_center_frame)}",
                org=(380, 60),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=(0, 255, 255),
                thickness=1,
                lineType=cv2.LINE_4,
            )

            cv2.putText(
                img=frame,
                text=f"{satellite_brightness_frame}",
                org=(380, 90),
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

    # Write the frame into the output file
    out.write(frame)

    # Display the resulting frame
    cv2.imshow("Video Playback", frame)

    # Press Q on keyboard to exit
    if cv2.waitKey(delay) & 0xFF == ord("q"):
        break

# Release the video capture object
# and close all OpenCV windows
cap.release()
out.release()
cv2.destroyAllWindows()

print(
    f"Avg satellite distance to frame center: {distance_satellite_to_center_avg} [pixel]"
)
print(f"Avg satellite brightness: {satellite_brightness_avg} [intensity]")
print(f"Avg window brightness: {avg_window_brightness_avg} [intensity]")
