import cv2 as cv
import numpy as np
import pyautogui as gui
import sys
import signal
from time import time
import json
from os.path import isfile

from detect_pupils import detect_pupils

_calibration_done = False
debug = 0
height, width = None, None

def finish_calibration(signum, frame):
    global _calibration_done
    _calibration_done = True

def read_dimensions(signum, frame):
    global height, width

    if not isfile('dimensions.txt'):
        print('No dimension file')
        return

    with open('dimensions.txt') as f:
        dimensions = json.loads(f.read())

    height = int(dimensions['height'])
    width = int(dimensions['width'])

def perform_capture():
    global _calibration_done, debug
    global height, width

    if height is None or width is None:
        read_dimensions(None, None)

    if debug == 1:
        f = open('displacement_data.txt', 'w')

    n_calib_frame = 1
    init_y = 0
    max_step = 10
    b = 1.5
    T = 2

    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pupils, eyes = detect_pupils(gray, ret_eyes=True)

        #or pup in pupils:
            #cv.circle(frame, (int(pup['x']), int(pup['y'])), int(pup['r']), (0, 0, 255), 2)

        #cv.imshow('frame', frame)

        if len(pupils) == 2:
            mean_y = (pupils[0]['y'] + pupils[1]['y'])/2.
            if mean_y < init_y:
                new_y = min(pupils[0]['y'], pupils[1]['y'])
            else:
                new_y = max(pupils[0]['y'], pupils[1]['y'])

            if not _calibration_done:
                n_calib_frame += 1
                init_y += (new_y-init_y+b)/n_calib_frame
            else:
                if debug == 0:
                    scroll_amt = -(new_y-init_y)/height*150
                    print('Scrolling by %f' % (scroll_amt))

                    scrolled = 0
                    step = scroll_amt/abs(scroll_amt)
                    while scrolled < abs(scroll_amt):
                        gui.vscroll(step)
                        scrolled += 1
                else:
                    f.write('%f,%d\n' % (new_y-init_y, time()))

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    if debug == 1:
        f.write('\n')
        f.close()
        
    cap.release()
    #cv.destroyAllWindows()

if __name__ == '__main__':
    debug = int(sys.argv[1])
    signal.signal(signal.SIGUSR1, finish_calibration)
    signal.signal(signal.SIGUSR2, read_dimensions)
    perform_capture()