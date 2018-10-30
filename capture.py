import cv2 as cv
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

from haar_detector import detect_faces, detect_eyes

cap = cv.VideoCapture(0)

i = 1

while True:
    _, frame = cap.read()
    cv.imshow('frame', frame)

    key = cv.waitKey(1)
    if key & 0xFF == ord('e'):
        imsave('images/frame%d.png' % i, frame[:,:,::-1])
        i += 1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()