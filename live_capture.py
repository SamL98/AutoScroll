import cv2 as cv
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

from detect_pupils import detect_pupils
from track_pupils import find_pupils

cap = cv.VideoCapture(0)

pupils = []

while True:
    _, frame = cap.read()

    if len(pupils) == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pupils = detect_pupils(gray)
    else:
        pupils = find_pupils(frame, pupils)

    for pup in pupils:
        cv.circle(frame, (int(pup['x']), int(pup['y'])), int(pup['r']), (0, 0, 255), 3)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()