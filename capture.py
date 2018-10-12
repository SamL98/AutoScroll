import cv2 as cv
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        imsave('frame3.png', frame[:,:,::-1])
        break

cap.release()
cv.destroyAllWindows()