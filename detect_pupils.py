import numpy as np
import numpy.ma as ma

import skimage as sk
import skimage.feature as feat
import skimage.exposure as exposure
import skimage.filters.rank as rank
import skimage.filters as filters
import skimage.morphology as morph

from scipy.ndimage.filters import gaussian_filter

import cv2 as cv

import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle, Ellipse

from os.path import join
import sys

from haar_detector import *
from vis_utils import circle

def weighted_size_for(blob, mx, my, h):
    rad = int(blob[2]*np.sqrt(2.))
    size = np.count_nonzero(morph.disk(rad))
    
    ''' Use an Epanechnikov kernel to weight the size of the blob by how far away it is from the center of the eye '''
    dist = (np.sqrt((blob[0]-my)**2 + (blob[1]-mx)**2)/h)**2
    return size * max(1-dist, 0)

def detect_pupils(im, ret_eyes=False, rel_coords=False):
    ''' Detect faces in the image. Hopefully there is only one face '''
    faces = detect_faces(im)
    if len(faces) != 1:
        if ret_eyes: return [], []
        return []

    x, y, w, h = faces[0]

    ''' Detect the eyes in the image '''
    face = im[y:y+h, x:x+w]
    eyes = detect_eyes(face)
    if len(eyes) != 2:
        if ret_eyes: return [], []
        return []

    pupils = []

    for (ex, ey, ew, eh) in eyes:
        eye = face[ey:ey+eh, ex:ex+ew]

        ''' Preprocess the image '''
        eye = filters.median(eye, morph.disk(3)) # median filtering was found experimentally to improve results
        _, eye = cv.threshold(eye, np.percentile(eye.ravel(), 25), 255, cv.THRESH_BINARY_INV) # thresholding @ the 25th percentile results in a good blob canvas

        ''' Perform LoG spot detection on the eye patch '''
        blobs = feat.blob_log(eye/255., min_sigma=1.25, max_sigma=8.0, num_sigma=15, threshold=0.5)

        if len(blobs) == 0:
            print('No blobs found for eye')
            continue

        mx, my = ew/2., eh/2.
        h = (mx + my)/2.

        ''' Pick the blob that has the largest weighted size. This assumes that the pupil is likely to be both: '''
        '''             a) large-ish in size, and                                                               '''
        '''             b) near-ish to the center of the eye                                                    '''
        sizes = np.array(list(map(lambda blob: weighted_size_for(blob, mx, my, h), blobs)))
        blob = blobs[np.argmax(sizes)]

        bx, by = blob[1], blob[0]

        if not rel_coords:
            bx += x+ex
            by += y+ey

        pupils.append({
            'x': bx, 
            'y': by, 
            'r': blob[2]*np.sqrt(2.)
        })

    if ret_eyes:
        eyes = list(map(
            lambda eye: (x+eye[0], y+eye[1], eye[2], eye[3]),
            eyes
        ))
        return pupils, eyes

    return pupils

if __name__ == '__main__':
    fno = 2
    if len(sys.argv) > 1:
        fno = int(sys.argv[1])

    #start = time()

    im = cv.imread(join('frames', 'frame%d.png' % fno))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    pupils = detect_pupils(im)

    #end = time()
    #print('Detection took %.4f seconds' % (end-start))

    _, ax = plt.subplots(1)
    ax.imshow(im, 'gray')

    faces = detect_faces(im)
    x,y,w,h = faces[0]

    from vis_utils import rect
    ax.add_patch(rect(x, y, w, h))

    face = im[y:y+h, x:x+w]
    eyes = detect_eyes(face)

    for ex,ey,ew,eh in eyes:
        ax.add_patch(rect(ex+x, ey+y, ew, eh))

    for pup in pupils:
        ax.add_patch(
            Circle(
                (pup['x'], pup['y']), pup['r'],
                edgecolor='r', facecolor='None'
            ))

    plt.show()