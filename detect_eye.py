import cv2 as cv
import numpy as np
import numpy.ma as ma

import sys

from skimage.io import imread
import skimage.feature as feat
import skimage.morphology as morph
import skimage.filters as filters
import skimage.exposure as expos

import sklearn.linear_model as lm

import matplotlib.pyplot as plt
from vis_utils import circle, ellipse, rect, retax

from detect_pupils import detect_pupils

def rotated_rect():
    pass

def look_for_sclera(eye, px, py, pr, theta, w, h):
    pass

def diamond_mask(eye):
    mask = np.zeros_like(eye)
    if eye.shape[0] % 2 == 0:
        i1, i2 = eye.shape[0]//2-1, eye.shape[0]//2
    else:
        mask[eye.shape[0]//2] = 1
        i1, i2 = eye.shape[0]//2-1, eye.shape[0]//2+1

def detect_eye_contour(im, px, py, pr, ex, ey, ew, eh):
    ''' Assume that pupil holds the relative coordinates within the eye patch '''

    eye = im[ey:ey+eh, ex:ex+ew]

    pad = 2
    t, b = max(0, py-pr-pad), min(eye.shape[0], py+pr+pad)
    eye_cropped = eye[t:b]
    py -= t

    eye_cropped = filters.median(eye_cropped)

    mask = morph.disk(pr)
    pmean = ma.masked_array(eye_cropped[pad:-pad+1, px-pr:px+pr+1], 1-mask).mean()

    eye_tmp = eye_cropped.copy()
    eye_tmp[eye_tmp>pmean*2] = 255

    _, ax = plt.subplots(1)
    ax.imshow(eye_cropped, 'gray')
    ax.add_patch(circle(px, py, pr))

    blobs = feat.blob_log(eye_tmp/255., min_sigma=0.75, max_sigma=5.0, num_sigma=12)

    pts = [[blob[0], blob[1]] for blob in blobs]
    pts.append([py, px])
    pts = np.array(pts)

    lr = lm.LinearRegression()
    lr.fit(np.expand_dims(pts[:,1], axis=1), np.expand_dims(pts[:,0], axis=1))
    
    m = lr.coef_[0,0]
    b = lr.intercept_[0]

    ax.plot([0, eye_cropped.shape[1]], [b, b+eye_cropped.shape[1]*m])
    ax.scatter(pts[:,1], pts[:,0], c='r')

    plt.show()


if __name__ == '__main__':
    fno = 2
    if len(sys.argv) > 1:
        fno = int(sys.argv[1])

    im = cv.imread('images/frame%d.png' % fno)
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)

    pupils, eyes = detect_pupils(im, ret_eyes=True, rel_coords=True)

    for pupil, eye in zip(pupils, eyes):
        detect_eye_contour(im, int(pupil['x']), int(pupil['y']), int(pupil['r']), *eye)