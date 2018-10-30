import numpy as np
import cv2 as cv

from haar_detector import *

def find_object(frame, y, x, h, w):
    if w == 0 or h == 1:
        return (0, 0, 0, 0)
        
    roi = frame[y:y+h, x:x+w]
    hsv_roi = cv.cvtColor(roi, cv.COLOR_BGR2HSV)
    mask = cv.inRange(hsv_roi, np.array([0., 60., 32.]), np.array([180., 255., 255.]))
    roi_hist = cv.calcHist([hsv_roi], [0], mask, [180], [0, 180])
    cv.normalize(roi_hist, roi_hist, 0, 255, cv.NORM_MINMAX)

    termination_criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
    dest = cv.calcBackProject([frame], [0], roi_hist, [0, 180], 1)

    ret, loc = cv.CamShift(dest, (x, y, w, h), termination_criteria)
    print(ret)
    print(loc)
    return (loc[0]+x, loc[1]+y, loc[2], loc[3])

import matplotlib.pyplot as plt
from vis_utils import *

def circularNeighbors(img, x, y, rad):
	neighbors = []
	for r, row in enumerate(img):
		for c, rgb in enumerate(row):
			if ((r-y)**2) + ((c-x)**2) > rad**2: continue
			neighbors.append([c, r, rgb[0], rgb[1], rgb[2]])
	return np.array(neighbors)

def colorHistogram(X, bins, x, y, h):
	hist = np.zeros((bins, bins, bins))
	rgb = X[:,2:]

	r = (np.sqrt((X[:,0]-x)**2 + (X[:,1]-y)**2)/h)**2
	k = 1-r
	k[r>=1] = 0

	binno = rgb//(256//bins)

	for i in range(bins**3):
		rx = i % bins
		gx = (i//bins) % bins
		bx = (i//(bins**2)) % bins
		hist[rx,gx,bx] = (k[np.argwhere(binno == [rx, gx, bx])]).sum()

	return hist/hist.sum()

def meanshiftWeights(X, q, p, bins):
	rgb = X[:,2:]
	b = rgb//(256//bins)
	w = np.sqrt(q[b[:,0], b[:,1], b[:,2]] / p[b[:,0], b[:,1], b[:,2]])
	return np.expand_dims(w, axis=1)

def find_pupils(frame, init_locs, init_eyes, init_hists):
    faces = detect_faces(frame)
    if len(faces) != 1:
        return []

    # (x, y, w, h) = faces[0]
    # face = frame[y:y+h, x:x+w]
    # eyes = detect_eyes(face)
    # eyes = [(ex+x, ey+y, ew, eh) for (ex,ey,ew,eh) in eyes]
    
    pupils, eyes = detect_pupils(cv.cvtColor(frame, cv.COLOR_BGR2GRAY), ret_eyes=True)
    centers = [(ex+ew/2., ey+eh/2.) for (ex,ey,ew,eh) in init_eyes]
    locs = []

    _, ax = plt.subplots(1)
    ax.imshow(frame, 'gray')

    #for ex,ey,ew,eh in eyes:
    for pup, (ex, ey, ew, eh) in zip(pupils, eyes):
        center = (ex+ew/2., ey+eh/2.)
        dists = np.array([
            np.sqrt((center[0]-c[0])**2 + (center[1]-c[1])**2)
            for c in centers
        ])
        match_idx = np.argmin(dists)
        p = init_locs[match_idx]
        model = init_hists[match_idx]
        
        loc = {'x': p['x'], 'y': p['y']}
        for i in range(5):
            neighbors = circularNeighbors(frame, loc['x'], loc['y'], p['r'])
            test = colorHistogram(neighbors, 16, loc['x'], loc['y'], p['r'])
            weights = meanshiftWeights(neighbors, model, test, 16)
            new_loc = weights.T.dot(neighbors[:,:2]) / weights.sum()
            loc = {'x': new_loc[0,0], 'y': new_loc[0,1]}

        ax.add_patch(rect(ex, ey, ew, eh))
        ax.add_patch(circle(loc['x'], loc['y'], p['r']))
        ax.add_patch(circle(pup['x'], pup['y'], pup['r']))

    plt.show()
    return locs

if __name__ == '__main__':
    from os.path import join
    from detect_pupils import detect_pupils

    i = 5
    im = cv.imread(join('frames', 'frame%d.png' % i))
    gray = cv.cvtColor(im, cv.COLOR_BGR2GRAY)
    init_locs, init_eyes = detect_pupils(gray, ret_eyes=True)
    init_neighbors = [circularNeighbors(im, p['x'], p['y'], p['r']) for p in init_locs]
    init_hists = [colorHistogram(n, 16, p['x'], p['y'], p['r']) for (n, p) in zip(init_neighbors, init_locs)]

    im = cv.imread(join('frames', 'frame%d.png' % (i+1)))
    find_pupils(im, init_locs, init_eyes, init_hists)