import cv2 as cv
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

from haar_detector import detect_faces, detect_eyes

cap = cv.VideoCapture(0)

i = 7

while True:
    _, frame = cap.read()
    cv.imshow('frame', frame)

    faces = detect_faces(frame)
    if len(faces) > 0:
        x, y, w, h = faces[0]
        eyes = detect_eyes(frame[y:y+h, x:x+w])

        cv.rectangle(frame, (x, y), (w, h), (255, 0, 0))
        for ex, ey, ew, eh in eyes:
            cv.rectangle(frame, (x+ex, y+ey), (ew, eh), (255, 0, 0))

    key = cv.waitKey(1)
    if key & 0xFF == ord('e'):
        imsave('images/frame%d.png' % i, frame[:,:,::-1])
        i += 1
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()