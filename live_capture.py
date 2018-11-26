import cv2 as cv
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

from detect_pupils import detect_pupils
from fps_for_tracking import get_fps

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    pupils, eyes = detect_pupils(gray, ret_eyes=True)

    for pup in pupils:
        cv.circle(frame, (int(pup['x']), int(pup['y'])), int(pup['r']), (0, 0, 255), 2)

    #for ex,ey,ew,eh in eyes:
    #    cv.rectangle(frame, (ex, ey), (ew+ex, eh+ey), (0, 0, 255))

    cv.imshow('frame', frame)

    key = cv.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv.destroyAllWindows()

np.savetxt('locs.txt', np.array(pup_locs))