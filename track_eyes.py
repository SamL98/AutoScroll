import cv2 as cv
import numpy as np
from skimage.io import imsave
from skimage.transform import resize

from haar_detector import detect_faces, detect_eyes
from track_pupils import find_object

cap = cv.VideoCapture(0)

eyes = []

while True:
    _, frame = cap.read()

    if len(eyes) == 0:
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        if len(faces) == 0:
            continue
        x, y, w, h = faces[0]

        face = gray[y:y+h, x:x+w]
        eyes = detect_eyes(face)
    else:
        tmp_eyes = []
        for eye in eyes:
            tmp_eyes.append(find_object(frame, eye[0], eye[1], eye[2], eye[3]))
        eyes = tmp_eyes


    #for pup in pupils:
    #    cv.circle(frame, (int(pup['x']), int(pup['y'])), int(pup['r']), (0, 0, 255), 3)

    for eye in eyes:
     #   cv.polylines(frame, [])
        cv.rectangle(frame, (eye[0], eye[1]), (eye[2], eye[3]), (0, 0, 255), 3)

    cv.imshow('frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()