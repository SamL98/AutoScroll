from skimage.io import imread, imsave
import cv2 as cv

from detect_pupils import detect_pupils

import os

fnames = os.listdir('frames')
for fname in fnames:
    frame = imread('frames/'+fname)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    pupils = detect_pupils(gray)
    for pupil in pupils:
        cv.circle(frame, (int(pupil['x']), int(pupil['y'])), int(pupil['r']), (255, 0, 0))
    imsave('predicted/'+fname, frame)