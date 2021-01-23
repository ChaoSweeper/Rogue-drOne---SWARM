# Import libraries required for this project
import os
import re
from os.path import isfile, join

import cv2
import numpy as np
import matplotlib.pyplot as plt

model_frames = os.listdir("frames/")
model_frames.sort(key=lambda f: int(re.sub("\D", "", f)))

model_images = []
for i in model_frames:
    img = cv2.imread("frames/" + i)
    model_images.append(img)

i = 16
for frame in [i, i + 1]:
    plt.imshow(cv2.cvtColor(model_images[frame], cv2.COLOR_BGR2RGB))
    plt.title("Image Frame: " + str(frame))
    plt.show()

grayA = cv2.cvtColor(model_images[i], cv2.COLOR_BGR2GRAY)
grayB = cv2.cvtColor(model_images[i + 1], cv2.COLOR_BGR2GRAY)
diff_image = cv2.absdiff(grayB, grayA)
ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)
kernel = np.ones((3, 3), np.uint8)
dilated = cv2.dilate(thresh, kernel, iterations=1)

contours, hierarchy = cv2.findContours(
    thresh.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
)

valid_cntrs = []

for i, cntr in enumerate(contours):
    x, y, w, h = cv2.boundingRect(cntr)
    if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
        valid_cntrs.append(cntr)
len(valid_cntrs)

dmy = model_images[16].copy()

cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)
cv2.line(dmy, (0, 80), (256, 80), (100, 255, 255))

kernel = np.ones((4, 4), np.uint8)

font = cv2.FONT_HERSHEY_SIMPLEX

pathIn = "contour_frames/"

for i in range(len(model_images) - 1):

    grayA = cv2.cvtColor(model_images[i], cv2.COLOR_BGR2GRAY)
    grayB = cv2.cvtColor(model_images[i + 1], cv2.COLOR_BGR2GRAY)
    diff_image = cv2.absdiff(grayB, grayA)

    ret, thresh = cv2.threshold(diff_image, 30, 255, cv2.THRESH_BINARY)

    dilated = cv2.dilate(thresh, kernel, iterations=1)

    contours, hierarchy = cv2.findContours(
        dilated.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
    )

    valid_cntrs = []
    for cntr in contours:
        x, y, w, h = cv2.boundingRect(cntr)
        if (x <= 200) & (y >= 80) & (cv2.contourArea(cntr) >= 25):
            if (y >= 90) & (cv2.contourArea(cntr) < 40):
                break
            valid_cntrs.append(cntr)

    dmy = model_images[i].copy()
    cv2.drawContours(dmy, valid_cntrs, -1, (127, 200, 0), 2)

    cv2.putText(
        dmy,
        "Vehicles Detected: " + str(len(valid_cntrs)),
        (55, 15),
        font,
        0.6,
        (0, 180, 0),
        2,
    )
    cv2.line(dmy, (0, 80), (256, 80), (100, 255, 255))
    cv2.imwrite(pathIn + str(i) + ".png", dmy)

pathOut = "vehicle_detection_model.mp4"
fps = 14.0

frame_array = []
files = [f for f in os.listdir(pathIn) if isfile(join(pathIn, f))]

files.sort(key=lambda f: int(re.sub("\D", "", f)))

for i in range(len(files)):
    filename = pathIn + files[i]
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width, height)
    frame_array.append(img)

out = cv2.VideoWriter(pathOut, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)

for i in range(len(frame_array)):
    out.write(frame_array[i])
out.release()
