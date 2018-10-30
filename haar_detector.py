import cv2 as cv

''' Load the OpenCV face and eye classifiers '''
cv_path = '/library/frameworks/python.framework/versions/3.6/lib/python3.6/site-packages/cv2/data'
face_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_eye.xml')
glasses_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_eye_tree_eyeglasses.xml')

def detect_faces(im):
    return face_cascade.detectMultiScale(im, 1.3, 5)

def detect_eyes(face):
    eyes = eye_cascade.detectMultiScale(face)

    if len(eyes) == 0:
        eyes = glasses_cascade.detectMultiScale(face)

    return eyes