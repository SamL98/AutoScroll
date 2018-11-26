# SYNAPSIS:
    # When this program is run, is reads frames from the webcam
    # and performs pupil detection on each frame.

    # The program works with the web interface to get the pupil's vertical distance moved.
    #
    # In the web interface, there is an initial calibration period (~5 seconds) in which
    # the user is asked to stare at a stationary red dot.
    #
    # During this period, the capture program (in this file) calculates a mean vertical
    # position for the pupils.
    #
    # Once the calibration period has finished, the pupil position found in all subsequent frames
    # is used to calculate how much to scroll.

import cv2 as cv
import numpy as np
import pyautogui as gui
import sys
import signal
from time import time
import json
from os.path import isfile

from detect_pupils import detect_pupils
from fps_for_tracking import get_fps
from tracking import *

calibration_done = False   # flag to set to True once the calibration period has elapsed
display = False
height, width = None, None  # the height and width of the window that the web interface is in 

def read_dimensions(signum, frame):
    ''' get the viewport dimensions that were saved to a file by the webserver '''
    global height, width

    if not isfile('dimensions.txt'):
        print('No dimension file')
        return

    # the dimensions.txt file should be a JSON string
    # formatted as: {height: <height>, width: <width>}
    with open('dimensions.txt') as f:
        dimensions = json.loads(f.read())

    height = int(dimensions['height'])
    width = int(dimensions['width'])

def dist(p1, p2):
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def perform_capture():
    ''' read frames from the webcam and perform pupil detection '''
    global calibration_done, debug
    global height, width

    if height is None or width is None:
        read_dimensions(None, None)

    height = 1

    prev_p1, prev_p2 = None, None
    prev_c1, prev_c2 = None, None
    init_y1, init_y2 = 0, 0

    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pupils, eyes, face = detect_pupils(gray, ret_eyes=True, ret_face=True, rel_coords=False)

        if not calibration_done:
            corners = []

            if display:
                for p in pupils:
                    cv.circle(frame, (int(p['x']), int(p['y'])), int(p['r']), (0, 0, 255), 2)

                for ex,ey,ew,eh in eyes:
                    cv.rectangle(frame, (int(ex), int(ey)), (int(ew+ex), int(eh+ey)), (0, 0, 255), 2)

            if len(pupils) == 1:
                pupils.append(pupils[0])
                eyes.append(eyes[0])

            for eye in eyes:
                bbox = (eye[1], eye[0], eye[1]+eye[3], eye[0]+eye[2])
                c = get_fps(gray, bbox, 'fast')

                if display:
                    cv.circle(frame, (int(c[1]), int(c[0])), 3, (0,255,0), -1)
                    cv.circle(frame, (int(c[3]), int(c[2])), 3, (0,255,0), -1)

                corners.append([(c[1], c[0]), (c[3], c[2])])

            if display:
                cv.imshow('frame', frame)

            if len(pupils) == 2:
                p1, p2 = pupils[0], pupils[1]
                c1, c2 = corners[0], corners[1]

                if prev_p1 is None or prev_p2 is None:
                    prev_p1 = (p1['x'], p1['y'])
                    prev_p2 = (p2['x'], p2['y'])
                    prev_c1 = c1
                    prev_c2 = c2
                else:
                    T = 3

                    curr_p1 = (p1['x'], p1['y'])
                    curr_p2 = (p2['x'], p2['y'])
                    
                    dist1 = dist(prev_p1, curr_p1)
                    dist2 = dist(prev_p2, curr_p2)

                    if dist1 < T and dist2 < T:

                        good_match = True
                        # for prev_c, curr_c in zip([prev_c1, prev_c2], [c1, c2]):
                        #     distl = dist(prev_c[0], curr_c[0])
                        #     distr = dist(prev_c[1], curr_c[1])
                        #     if distl >= T or distr >= T:
                        #         good_match = False
                        #         break

                        if good_match:
                            calibration_done = True

                            pad = 10
                            init_y1 = p1['r']+pad
                            init_y2 = p2['r']+pad
                            x, y, w, h = face[0], face[1], face[2], face[3]

                            ex, ey, ew, eh = eyes[0]
                            ex += pad
                            ew -= 2*pad
                            te1 = init_tracker((ex + ew//2, p1['y'], ew, 2*init_y1),
                                                'csrt', 
                                                frame, 
                                                'eye',
                                                (x, y))
                            tp1 = init_tracker((p1['x'], p1['y'], p1['r']), 
                                                'csrt', 
                                                frame, 
                                                'pupil',
                                                (ex, init_y1))

                            ex, ey, ew, eh = eyes[1]
                            ex += pad
                            ey -= 2*pad
                            te2 = init_tracker((ex + ew//2, p1['y'], ew, 2*init_y2),
                                                'csrt', 
                                                frame, 
                                                'eye',
                                                (x, y))
                            tp2 = init_tracker((p2['x'], p2['y'], p2['r']), 
                                                'csrt', 
                                                frame, 
                                                'pupil',
                                                (ex, init_y2))

                            init_y1 = p1['y']
                            init_y2 = p2['y']

                    prev_p1 = curr_p1
                    prev_p2 = curr_p2
                    prev_c1 = c1
                    prev_c2 = c2

        else:
            if len(eyes) > 0 and len(pupils) > 0:
                x, y, w, h = face[0], face[1], face[2], face[3]

                ex,ey,ew,eh = track(te1, frame, 'eye', (x, y))
                p1 = track(tp1, frame, 'pupil', (ex, ey))

                if display:
                    draw_ellipse(frame, (ex, ey, ew//2, eh//2))
                    draw_circle(frame, p1)

                new_y1 = (p1[1]+p1[3]/2)# - ey
                delta_y1 = new_y1-init_y1

                ex,ey,ew,eh = track(te2, frame, 'eye', (x, y))
                p2 = track(tp2, frame, 'pupil', (ex, ey))

                if display:
                    draw_ellipse(frame, (ex, ey, ew//2, eh//2))
                    draw_circle(frame, p2)

                new_y2 = (p2[1]+p2[3]/2)# - ey
                delta_y2 = new_y2-init_y2

                if display:
                    for p in pupils:
                        cv.circle(frame, (int(p['x']), int(p['y'])), int(p['r']), (0, 255, 0), 2)

            mean_delta = (delta_y1+delta_y2)/2.
            scroll_amt = -(mean_delta)/height*1000
            #print(scroll_amt, mean_delta)
            gui.vscroll(scroll_amt)

            if display:
                cv.imshow('frame', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    signal.signal(signal.SIGUSR1, read_dimensions)
    perform_capture()