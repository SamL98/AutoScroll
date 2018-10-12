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

def detect_pupils(im):
    ''' Detect faces in the image. Hopefully there is only one face '''
    faces = detect_faces(im)
    if len(faces) != 1:
        return []

    x, y, w, h = faces[0]

    ''' Detect the eyes in the image '''
    face = im[y:y+h, x:x+w]
    eyes = detect_eyes(face)
    if len(eyes) != 2:
        return []

    pupils = []

    for (ex, ey, ew, eh) in eyes:
        eye = face[ey:ey+eh, ex:ex+ew]

        ''' Equalize the histogram of the eye patch and threshold it '''
        eye = exposure.equalize_hist(eye)
        eye = filters.median(eye, morph.disk(3)) # median filtering was found experimentally to improve results

        ''' For each circle of radius 4 in the eye patch, check to see that it is mainly dark '''
        ''' Do deteremine this, check to see if no more than 35% percent of the pixels within the circle are greater than 255/6 '''
        mask = 1-morph.disk(3)
        eye_tmp = np.pad(eye.copy(), 3, 'constant', constant_values=255)

        for i in range(3, eye.shape[0]+1):
            for j in range(3, eye.shape[1]+1):
                patch = ma.masked_array(eye_tmp[i-3:i+3+1, j-3:j+3+1], mask)

                if np.sum(patch <= 255/6.) <= int(0.65*np.count_nonzero(mask)): eye[i-3, j-3] = 0
                else:                                                           eye[i-3, j-3] = 255


        ''' Perform LoG spot detection on the eye patch '''
        blobs = feat.blob_log(eye/255., min_sigma=1.25, max_sigma=8.0, num_sigma=15, threshold=0.5)

        if len(blobs) == 0:
            continue

        mx, my = ew/2., eh/2.
        h = (mx + my)/2.

        ''' Pick the blob that has the largest weighted size. This assumes that the pupil is likely to be both: '''
        '''             a) large-ish in size, and                                                               '''
        '''             b) near-ish to the center of the eye                                                    '''
        sizes = np.array(list(map(lambda blob: weighted_size_for(blob, mx, my, h), blobs)))
        blob = blobs[np.argmax(sizes)]

        pupils.append({
            'x': blob[1]+x+ex, 
            'y': blob[0]+y+ey, 
            'r': blob[2]*np.sqrt(2.)
        })

    return pupils

if __name__ == '__main__':
    fno = 2
    if len(sys.argv) > 1:
        fno = int(sys.argv[1])

    im = cv.imread(join('images', 'frame%d.png' % fno))
    im = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    pupils = detect_pupils(im)

    _, ax = plt.subplots(1)
    ax.imshow(im, 'gray')

    for pup in pupils:
        ax.add_patch(
            Circle(
                (pup['x'], pup['y']), pup['r'],
                edgecolor='r', facecolor='None'
            ))

    plt.show()