from skimage.io import imread
from skimage.color import rgb2gray
from skimage.exposure import equalize_hist
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Ellipse

from cv2 import circle
import numpy as np
from scipy.ndimage.filters import convolve
from scipy.ndimage import gaussian_laplace as LoG

def fspecial_gauss(sigma):
    size = 2*np.ceil(3*sigma) + 1
    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()

'''
def lap(sigma):
    m = int(2*np.ceil(3*sigma) + 1)
    hm = m//2
    xs, ys = range(-hm, hm+1), range(-hm, hm+1)
    
    Gxx, Gyy = np.zeros((m, m)), np.zeros((m, m))
    c = 1. / (2 * np.pi * sigma**4)

    for i, y in enumerate(ys):
        for j, x in enumerate(xs):
            exp_val = np.exp(-(x**2 + y**2) / (2 * sigma**2))
            Gxx[i,j] = c * (-1. + x**2/sigma**2) * exp_val
            Gyy[i,j] = c * (-1. + y**2/sigma**2) * exp_val

    return sigma**2*(Gxx + Gyy)
'''

fname = 'frame.png'
#fname = '/users/samlerner/downloads/gt_db/s01/01.jpg'
im = rgb2gray(imread(fname).astype(np.float64))
im = equalize_hist(im)
orig = im.copy()

while True:
    inpt = input().split(' ')
    sx, sy = float(inpt[0]), float(inpt[1])

    res = LoG(im, (sx, sy))
    mask = np.zeros_like(res)

    _, ax = plt.subplots(1)
    ax.imshow(orig, 'gray')

    for _ in range(10):
        pt = np.unravel_index(np.argmax(res), res.shape)

        rx = int(2*np.ceil(3*sx)+1)
        ry = int(2*np.ceil(3*sy)+1)

        ax.add_patch(Ellipse((pt[1], pt[0]), rx, ry, facecolor='None', edgecolor='r', linewidth=3.))

        res[pt[0]-ry//2:pt[0]+ry//2, pt[1]-rx//2:pt[1]+rx//2] = 0

    plt.title('Sigma=(%0.2f, %0.2f)' % (sx, sy))
    plt.show()
exit()

sigma = 0.5
min_sigma, d_sigma = sigma, 1.5
max_sigma = 16.0
cube = []

while sigma <= max_sigma:
    result = convolve(im, lap(sigma))
    cube.append(result)
    sigma += d_sigma

cube = -np.array(cube).transpose((1, 2, 0))
cube[cube<cube.max()*0.7] = 0
result = np.zeros((cube.shape[0], cube.shape[1]))

m = 5
for i in range(0, cube.shape[0]-m):
    for j in range(0, cube.shape[1]-m):
        volume = cube[i:i+m, j:j+m, :]
        maxima = np.unravel_index(np.argmax(volume), volume.shape)
        result[i+maxima[0], j+maxima[1]] = min_sigma + d_sigma*maxima[2]

result = np.ceil(3*result).astype(np.uint8)
mask = np.zeros_like(orig)

_, ax = plt.subplots(1)
for pt in np.argwhere(result>0):
    rad = result[pt[0], pt[1]]
    circle(mask, (pt[1], pt[0]), rad, 255, -1)

ax.imshow(orig, 'gray')
ax.imshow(result, cmap='OrRd', alpha=0.3)
plt.show()