import cv2
import numpy
import math
from copy import deepcopy
from tracker import Tracker

videoCapture = cv2.VideoCapture('videos/video1.avi')
frames_count, fps, height, width = videoCapture.get(cv2.CAP_PROP_FRAME_COUNT), videoCapture.get(
    cv2.CAP_PROP_FPS), int(videoCapture.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(
    videoCapture.get(cv2.CAP_PROP_FRAME_WIDTH))
frames = numpy.zeros((int(frames_count / 10), height, width, 3), numpy.uint8)
for i in range(int(frames_count / 10)):
    videoCapture.set(cv2.CAP_PROP_POS_FRAMES, i * 10)
    ret, frame = videoCapture.read()
    if not ret:
        break
    frames[i] = frame
background = numpy.median(frames, axis=0).astype(dtype=numpy.uint8)
grayBackground = cv2.cvtColor(background, cv2.COLOR_BGR2GRAY)

tracker = Tracker(250, 25, 100)
#tracker = Tracker(20, 5, 50)

videoCapture.set(cv2.CAP_PROP_POS_FRAMES, 0)
counter = 0
while True:
    ret, frame = videoCapture.read()
    if not ret:
        tracker.Finish()
        break
    Contours = []
    Centers = []
    frame = frame.astype(dtype=numpy.uint8)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    difference = cv2.absdiff(grayFrame, grayBackground)
    _, threshold = cv2.threshold(difference, 30, 255, cv2.THRESH_BINARY)
    kernel1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    dilation = cv2.dilate(threshold, kernel1)
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel2)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel2)
    dilation = cv2.erode(opening, kernel1)
    _, difference = cv2.threshold(dilation, 30, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(difference, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    minArea = 50
    maxArea = 50000
    for i in range(len(contours)):
        if hierarchy[0, i, 3] == -1:
            cnt = contours[i]
            area = cv2.contourArea(cnt)
            if minArea < area < maxArea:
                Contours.append(cnt)
                rect = cv2.boundingRect(cnt)
                M = cv2.moments(cnt)
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                Centers.append(numpy.round(numpy.array([[cx], [cy]])))
    if len(Centers) > 0:
        tracker.Update(frame, Contours, Centers, counter)
    counter = counter + 1
max_frames_count = 0
for i in range(len(tracker.finalTracks)):
    if len(tracker.finalTracks[i].frame) > max_frames_count:
        max_frames_count = len(tracker.finalTracks[i].frame)
random = numpy.random.random(len(tracker.finalTracks))
diff = []
for i in range(len(tracker.finalTracks)):
    diff.append(max_frames_count-len(tracker.finalTracks[i].frame))
start = numpy.floor(random * diff).astype(dtype=numpy.uint)
"""
magnitudes = []
angles = []
for i in range(len(tracker.finalTracks)):
    y = float(tracker.finalTracks[i].prediction[0][0] - tracker.finalTracks[i].first[0][0])
    x = float(tracker.finalTracks[i].prediction[1][0] - tracker.finalTracks[i].first[1][0])
    magnitude = numpy.sqrt(x ** 2 + y ** 2)
    if x > 0:
        angle = math.degrees(math.atan(y/x))
    elif x < 0:
        if y > 0:
            angle = math.degrees(math.atan(y / x)) + 180
        elif y < 0:
            angle = math.degrees(math.atan(y / x)) - 180
        else:
            angle = 0
    else:
        angle = 0
    magnitudes.append(int(magnitude))
    angles.append(int(angle))
filtered_finalTracks = []
filtered_magnitudes = []
filtered_angles = []
for i in range(len(tracker.finalTracks)):
    if magnitudes[i] > 100 and angles[i] != 0:
        filtered_finalTracks.append(tracker.finalTracks[i])
        filtered_magnitudes.append(magnitudes[i])
        filtered_angles.append(angles[i])
filtered_angles = numpy.array(filtered_angles).astype(dtype=numpy.float32)
_, labels, centers = cv2.kmeans(filtered_angles, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 1, 1.0), 10, cv2.KMEANS_RANDOM_CENTERS)
max_0_magnitude = 0
max_0_index = 0
count_0 = 0
max_1_magnitude = 0
max_1_index = 0
count_1 = 0
for i in range(len(filtered_finalTracks)):
    if labels[i][0] == 0:
        if filtered_magnitudes[i] > max_0_magnitude:
            max_0_magnitude = filtered_magnitudes[i]
            max_0_index = i
        count_0 = count_0 + 1
    if labels[i][0] == 1:
        if filtered_magnitudes[i] > max_1_magnitude:
            max_1_magnitude = filtered_magnitudes[i]
            max_1_index = i
        count_1 = count_1 + 1
track_0_first = filtered_finalTracks[max_0_index].first
track_0_last = numpy.round(numpy.array([[track_0_first[0][0] + filtered_magnitudes[max_0_index] * math.sin(math.radians(centers[0][0]))], [track_0_first[1][0] + filtered_magnitudes[max_0_index] * math.cos(math.radians(centers[0][0]))]]))
track_1_first = filtered_finalTracks[max_1_index].first
track_1_last = numpy.round(numpy.array([[track_1_first[0][0] + filtered_magnitudes[max_1_index] * math.sin(math.radians(centers[1][0]))], [track_1_first[1][0] + filtered_magnitudes[max_1_index] * math.cos(math.radians(centers[1][0]))]]))
bgCopy = deepcopy(background)
cv2.line(bgCopy, (int(track_0_first[0][0]), int(track_0_first[1][0])), (int(track_0_last[0][0]), int(track_0_last[1][0])), (255, 255, 255), 10)
cv2.putText(bgCopy, str(int(count_0 * 5)), (int(track_0_first[0][0]), int(track_0_first[1][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
cv2.line(bgCopy, (int(track_1_first[0][0]), int(track_1_first[1][0])), (int(track_1_last[0][0]), int(track_1_last[1][0])), (255, 255, 255), 10)
cv2.putText(bgCopy, str(int(count_1 * 5)), (int(track_1_first[0][0]), int(track_1_first[1][0])), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1)
cv2.imshow('background', bgCopy)
cv2.imwrite('tracks1.jpg', bgCopy)
cv2.waitKey(0)
"""
videoWriter = cv2.VideoWriter('videos/video1-Summarized.avi', cv2.VideoWriter_fourcc(*'MJPG'), 30.0, (background.shape[1], background.shape[0]))
for i in range(max_frames_count):
    bgCopy = deepcopy(background)
    for j in range(len(tracker.finalTracks)):
        if start[j] <= i <= start[j] + len(tracker.finalTracks[j].frame) - 1:
            grayTemp = cv2.cvtColor(tracker.finalTracks[j].frame[i - start[j]], cv2.COLOR_BGR2GRAY)
            frameTemp = deepcopy(tracker.finalTracks[j].frame[i - start[j]])
            for k in range(3):
                frameTemp[:, :, k] = grayTemp
            bgCopy = numpy.where(frameTemp != 0, tracker.finalTracks[j].frame[i - start[j]], bgCopy)
    videoWriter.write(bgCopy)
videoCapture.release()
videoWriter.release()
cv2.destroyAllWindows()
