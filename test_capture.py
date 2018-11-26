import cv2 as cv
import numpy as np
from sklearn.covariance import EllipticEnvelope

from detect_pupils import detect_pupils
import matplotlib.pyplot as plt

_calibration_done = False   # flag to set to True once the calibration period has elapsed

def finish_calibration():
    ''' set the global flag that the calibration period has elapsed '''
    global _calibration_done
    _calibration_done = True

def perform_capture():
    ''' read frames from the webcam and perform pupil detection '''
    global _calibration_done

    f = open('test_data.txt', 'a')
    calib_pts1 = []
    calib_pts2 = []
    i1, i2 = [], []

    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pupils, eyes = detect_pupils(gray, ret_eyes=True)

        for p in pupils:
            cv.circle(frame, (int(p['x']), int(p['y'])), int(p['r']), (0, 0, 255), 2)

        if len(i1) > 0 and len(eyes) == 2:
            cv.circle(frame, (eyes[0][0]+int(i1[0]), eyes[0][1]+int(i1[1])), 2, (255, 0, 0), -1)
            cv.circle(frame, (eyes[1][0]+int(i2[0]), eyes[1][1]+int(i2[1])), 2, (0, 255, 0), -1)

        if len(pupils) == 1:
            pupils.append(pupils[0])
            eyes.append(eyes[0])

        cv.imshow('frame', frame)

        if len(pupils) == 2:
            px1, px2 = pupils[0]['x']-eyes[0][0], pupils[1]['x']-eyes[1][0]
            py1, py2 = pupils[0]['y']-eyes[0][1], pupils[1]['y']-eyes[1][1]

            if not _calibration_done:                       # calibration is still occurring, so recalculate the mean position
                calib_pts1.append([px1, py1])
                calib_pts2.append([px2, py2])

            else:                                           # calibration is over, we can now scroll
                d1, d2 = px1-i1[0], px2-i2[0]
                scroll_amt = -(d1+d2)/2.
                f.write('%f\n' % scroll_amt)

        key = cv.waitKey(1) & 0xFF
        if key == ord('f'):
            calib_pts1 = np.array(calib_pts1)
            calib_pts2 = np.array(calib_pts2)

            env = EllipticEnvelope(contamination=0.15)
            pts1_pred = env.fit(calib_pts1).predict(calib_pts1)
            pt1_inliers = calib_pts1[pts1_pred]
            i1 = pt1_inliers.mean(axis=0)

            env = EllipticEnvelope(contamination=0.15)
            pts2_pred = env.fit(calib_pts2).predict(calib_pts2)
            pt2_inliers = calib_pts2[pts2_pred]
            i2 = pt2_inliers.mean(axis=0)

            finish_calibration()
        elif key == ord('q'):
            break

    f.close()
    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    perform_capture()