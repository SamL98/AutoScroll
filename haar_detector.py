import cv2 as cv
import numpy as np

''' Load the OpenCV face and eye classifiers '''
cv_path = '/library/frameworks/python.framework/versions/3.6/lib/python3.6/site-packages/cv2/data'
face_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_frontalface_default.xml')
face_cascade_alt = cv.CascadeClassifier(cv_path+'/haarcascade_frontalface_alt.xml')
face_cascade_alt2 = cv.CascadeClassifier(cv_path+'/haarcascade_frontalface_alt2.xml')

eye_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_eye.xml')
left_eye_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_lefteye_2splits.xml')
right_eye_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_righteye_2splits.xml')
glasses_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_eye_tree_eyeglasses.xml')

def percent_overlap(box1, box2):
    exw1, exw2 = box1[0]+box1[2], box2[0]+box2[2]
    exh1, exh2 = box1[1]+box1[3], box2[1]+box2[3]
    canvas = np.zeros((max(exh1, exh2), max(exw1, exw2)), dtype=np.uint8)

    ox, oy = min(box1[0], box2[0]), min(box1[1], box2[1])
    x1, y1, w1, h1 = box1[0]-ox, box1[1]-oy, box1[2], box1[3]
    x2, y2, w2, h2 = box2[0]-ox, box2[1]-oy, box2[2], box2[3]
    canvas[y1:y1+h1, x1:x1+w1] += 1
    canvas[y2:y2+h2, x2:x2+w2] += 1

    return np.sum(canvas==2) / np.sum(canvas>0)

def detect_faces(im):
    faces = face_cascade.detectMultiScale(im, 1.3, 5)
    if len(faces) == 0:
        faces = face_cascade_alt.detectMultiScale(im, 1.3, 5)
        if len(faces) == 0:
            faces = face_cascade_alt2.detectMultiScale(im, 1.3, 5)
    return faces

def detect_eyes(face):
    eyes = eye_cascade.detectMultiScale(face)
    if len(eyes) == 0:
        eyes = glasses_cascade.detectMultiScale(face)
        # if len(eyes) == 0:
        #     eyes = left_eye_cascade.detectMultiScale(face)
        #     eyes = np.concatenate((eyes, right_eye_cascade.detectMultiScale(face)), axis=0)
    return eyes