import numpy as np
import numpy.ma as ma

import skimage as sk
import skimage.feature as feat
import skimage.exposure as exposure
import skimage.filters.rank as rank
import skimage.morphology as morph

from scipy.ndimage.filters import gaussian_filter

import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

''' Load the OpenCV face and eye classifiers '''
cv_path = '/library/frameworks/python.framework/versions/3.6/lib/python3.6/site-packages/cv2/data'
face_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_frontalface_default.xml')
eye_cascade = cv.CascadeClassifier(cv_path+'/haarcascade_eye.xml')

''' Detect faces in the image. Hopefully there is only one face '''
im = cv.imread('frame2.png')
im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(im, 1.3, 5)

assert len(faces) == 0
x, y, w, h = faces[0]

''' Detect the eyes in the image '''
face = im[y:y+h, x:x+w]
eyes = eye_cascade.detectMultiScale(face)

for (ex, ey, ew, eh) in eyes:
    eye = face[ey:ey+eh, ex:ex+ew]

    ''' Equalize the histogram of the eye patch and threshold it '''
    eye = exposure.equalize_hist(eye)
    eye[eye<.15] = 0

    ''' Perform LoG spot detection on the eye patch '''
    blobs = feat.blob_log(1.-eye, min_sigma=1.25, max_sigma=8.0, num_sigma=15, threshold=0.2)

    _, ax = plt.subplots(1)
    ax.imshow(eye, 'gray')

    def shd_include(blob):
        bx, by = int(blob[1]), int(blob[0])
        rad = int(blob[2]*np.sqrt(2.))

        ''' Don't include a blob that extends out of the eye patch '''
        if bx < rad or by < rad or bx >= eye.shape[1]-rad or by >= eye.shape[0]-rad:
            return False

        patch = eye[by-rad : by+rad+1, bx-rad : bx+rad+1]
        roi = ma.masked_array(patch, 1-morph.disk(rad)).ravel()
        pct, rat = 25, 0.75

        ''' Only include blobs where at least 0.75 of the included pixels are
            below the 25th percentile of pixels in the blob '''
        if np.sum(roi <= np.percentile(roi, pct)) <= int(rat*np.count_nonzero(morph.disk(rad))):
            return False

        return True

    def size_for(blob):
        bx, by = int(blob[1]), int(blob[0])
        rad = int(blob[2]*np.sqrt(2.))

        patch = eye[by-rad : by+rad+1, bx-rad : bx+rad+1]
        roi = ma.masked_array(patch, 1-morph.disk(rad)).ravel()
        
        return np.sum(roi > 0)

    blobs = list(filter(lambda blob: shd_include(blob), blobs))
    sizes = np.array(list(map(lambda blob: size_for(blob), blobs)))
    
    blob = blobs[np.argmax(sizes)]
    ax.add_patch(
        Circle(
            (blob[1], blob[0]),
            np.sqrt(2.)*blob[2],
            edgecolor='r', facecolor='None'
        ))

    plt.show()