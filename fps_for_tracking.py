import cv2
import numpy as np
import matplotlib.pyplot as plt
import time


def get_fps(img, box_coor, feature_opt):
    '''
    inputs:
    img: img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) , img = cv2.imread(filename)
    box_coor: (rt, ct, rb, cb) coordinates of top left and bottom right corners
    returns:
    fps_coor: (r_l, c_l, r_r, c_r)
    '''
    rt, ct, rb, cb = box_coor
    img_s = img[rt:rb, ct:cb,:].copy()
    gray = cv2.cvtColor(img_s, cv2.COLOR_RGB2GRAY)
    if feature_opt=='surf':
        surf = cv2.xfeatures2d.SURF_create()
        kps, _ = surf.detectAndCompute(gray, None)
    elif feature_opt=='sift':
        sift = cv2.xfeatures2d.SIFT_create()
        kps, _ = sift.detectAndCompute(gray, None)
    else: # fast features
        fast = cv2.FastFeatureDetector_create()
        kps = fast.detect(gray, None)


    # extract keypoints coordinates
    pts_c = np.zeros([len(kps), ])
    pts_r = np.zeros([len(kps), ])
    for i in range(len(kps)):
        pts_c[i] = kps[i].pt[0]
        pts_r[i] = kps[i].pt[1]

    idx_left = np.where(pts_c==min(pts_c))
    idx_right = np.where(pts_c==max(pts_c))

    # convert feature points coordinates back to the original full image
    c_l = pts_c[idx_left[0][0]] # left point
    r_l = pts_r[idx_left[0][0]]
    c_r = pts_c[idx_right[0][0]] # right point
    r_r = pts_r[idx_right[0][0]]

    # coordinates of feature points in the original image
    c_l += ct
    r_l += rt
    c_r += ct
    r_r += rt

    return (r_l, c_l, r_r, c_r)

# test script
if __name__=='__main__':

    img = cv2.imread('frame1.png')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    '''
    rt,ct = 313, 534
    rb, cb = 400, 630
    '''

    rt, ct = 337, 655
    rb, cb = 379, 732


    box_coor = (rt, ct, rb, cb)
    st = time.time()
    r_l, c_l, r_r, c_r = get_fps(img, box_coor,'surf')
    ed = time.time()
    print('\ttime elapsed for feature point extraction:',ed - st)
    # fps_coors = get_fps(img, box_coor)
    #print('fps_coors:', fps_coors)

    plt.imshow(img)
    plt.scatter([c_l, c_r],[r_l, r_r])
    plt.show()
