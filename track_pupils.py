import numpy as np
import cv2 as cv

from haar_detector import *

def find_pupil(frame, y, x, s):
    print(y, x, s)
    roi = frame[y-s:y+s, x-s:x+s]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array([0., 60., 32.]), np.array([180., 255., 255.]))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    termination_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    dest = cv.calcBackProject([frame], [0], roi_hist, [0, 180], 1)

    _, loc = cv.meanShift(dest, (x, y, s, s), termination_criteria)
    diam = (loc[2] + loc[3])/2.
    rad = (diam - 1.)/2.
    return {'x': loc[1], 'y': loc[2], 'r': rad}


def find_pupils(frame, init_locs):
    faces = detect_faces(frame)
    if len(faces) != 1:
        return []

    (x, y, w, h) = faces[0]
    face = frame[y:y+h, x:x+w]
    eyes = detect_eyes(face)

    hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    locs = list(map(lambda loc: find_pupil(hsv_frame, int(loc['y']-loc['r']), int(loc['x']-loc['r']), int(loc['r']*2+1)), init_locs))
    return locs