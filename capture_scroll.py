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
from time import time, sleep
import json
from os.path import isfile

from detect_pupils import detect_pupils
from fps_for_tracking import get_fps
from tracking import *

calibration_done = None    # flag to set to True once the calibration period has elapsed
display = True               # flag to set to True if you want to see the detection/tracking, otherwise it is headlessly
height, width = None, None  # the height and width of the window that the web interface is in 

def start_calibration(signum, frame):
    global calibration_done
    calibration_done = False

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
    ''' Euclidean distance between two 2d points '''
    return np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def perform_capture():
    ''' read frames from the webcam and perform pupil detection '''
    global calibration_done, debug
    global height, width

    if height is None or width is None:
        read_dimensions(None, None)

        # if we still don't have dimensions, just make 'em up
        if height is None or width is None:
            height, width = 1, 1

    prev_p1, prev_p2 = None, None   # the previous positions of the two pupils
    init_y1, init_y2 = 0, 0         # the initial y position of both pupils
    T = 3                           # threshold distance that each pupil must have moved less
                                    # than between frames to be considered calibrated

    t = .5

    #conn = Client(('localhost', 6000), authkey=b'password')
    
    cap = cv.VideoCapture(0)

    while True:
        _, frame = cap.read()

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        pupils, eyes, face = detect_pupils(gray, ret_eyes=True, ret_face=True, rel_coords=False)

        if calibration_done is None:
            if display:
                cv.imshow('frame', frame)
        elif not calibration_done:
            # Calibration is stopped when there are two consecutive frames where
            # the L2 distance of the (x, y) positions that each pupil has moved between
            # frames is smaller than a threshold.
            #
            # Therefore, if we are here, then that condition has not yet been satisfied.
            # Usually, this happens pretty quickly so this is probably towards the beginning of capture.

            if display:
                for p in pupils:
                    cv.circle(frame, (int(p['x']), int(p['y'])), int(p['r']), (0, 0, 255), 2)

                for ex,ey,ew,eh in eyes:
                    cv.rectangle(frame, (int(ex), int(ey)), (int(ew+ex), int(eh+ey)), (0, 0, 255), 2)

                cv.imshow('frame', frame)

            if len(pupils) == 2:
                # Only determine if this frame is good enough to end
                # calibration if there are two pupils found. Crazy, right?

                p1, p2 = pupils[0], pupils[1]

                if prev_p1 is None or prev_p2 is None:
                    # If this is the first frame, then prev_p1
                    # and prev_p2 will be None, so set them.

                    prev_p1 = (p1['x'], p1['y'])
                    prev_p2 = (p2['x'], p2['y'])
                else:
                    # Otherwise, we are in not the first frame and should
                    # compare the found pupil positions to their previous position.

                    curr_p1 = (p1['x'], p1['y'])
                    curr_p2 = (p2['x'], p2['y'])
                    
                    dist1 = dist(prev_p1, curr_p1)
                    dist2 = dist(prev_p2, curr_p2)

                    if dist1 < T and dist2 < T:
                        # Only consider calibrated if BOTH the distances moved
                        # is less than the threshold.

                        calibration_done = True

                        pad = 10            # This padding is used for two things:
                                            #       1. Determining the upper boundary of the eye 
                        oy1 = p1['r']+pad   #          bounding box for tracking (giving the pupil some margin).
                                            #       2. Trimming the horizontal dimension of the eye
                        oy2 = p2['r']+pad   #          so that the corners and other junk is not included.
                                            # This is a very crude method but it seem to works OK.

                        ex, _, ew, _ = eyes[0]
                        ex += pad
                        ew -= 2*pad

                        # The eye tracker takes a tuple defining an ellipse as:
                        #       (x center, y center, x axis length * 2, y axis length * 2)
                        # te1 = init_tracker((ex + ew//2, p1['y'], ew, 2*oy1),
                        #                     'csrt', 
                        #                     frame, 
                        #                     'eye')

                        # The pupil tracker takes a tuple defining a circle as:
                        #       (x center, y center, radius)
                        tp1 = init_tracker((p1['x'], p1['y'], p1['r']), 
                                            'csrt', 
                                            frame, 
                                            'pupil')

                        ex, _, ew, _ = eyes[1]
                        ex += pad
                        ew -= 2*pad
                        # te2 = init_tracker((ex + ew//2, p1['y'], ew, 2*oy2),
                        #                     'csrt', 
                        #                     frame, 
                        #                     'eye')
                        tp2 = init_tracker((p2['x'], p2['y'], p2['r']), 
                                            'csrt', 
                                            frame, 
                                            'pupil')

                        init_y1 = p1['y']#-eyes[0][1]
                        init_y2 = p2['y']#-eyes[1][1]
                    else:
                        # Otherwise, update prev_p1 and
                        # prev_p2 and keep on chugging.

                        prev_p1 = curr_p1
                        prev_p2 = curr_p2

        else:
            # Once calibration has completed, we have four trackers:
            #       * One for the left eye (and one for the right eye)
            #       * One for the left pupil (and one for the right pupil)
            # These trackers are named:
            #       te1, tp1, te2, and tp2

            if len(eyes) > 0 and len(pupils) > 0:
                # This check is to make sure we do not try to track the pupil
                # when the user is blinking. 
                #
                # During blinking, the eye trackers are not affected, 
                # but the pupil trackers often jump to the eyebrow.
                #
                # The len(eyes) check is basically useless, but
                # len(pupils) ensures that the user is blinking (hopefully).

                #ex,ey,ew,eh = track(te1, frame, 'eye')
                p1 = track(tp1, frame, 'pupil')

                if display:
                    #draw_ellipse(frame, (ex, ey, ew//2, eh//2))
                    draw_circle(frame, p1)
                    #cv.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

                # The bounding box returned from track is of the format:
                #       (x origin, y origin, x width, y height)
                # So to determine the new y position, add half the height
                # to the origin.

                new_y1 = (p1[1]+p1[3]/2)#-ey
                delta_y1 = new_y1-init_y1

                #ex,ey,ew,eh = track(te2, frame, 'eye')
                p2 = track(tp2, frame, 'pupil')

                if display:
                    #draw_ellipse(frame, (ex, ey, ew//2, eh//2))
                    draw_circle(frame, p2)
                    #v.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 2)

                new_y2 = (p2[1]+p2[3]/2)#-ey
                delta_y2 = new_y2-init_y2

                if display:
                    #for p in pupils:
                    #    cv.circle(frame, (int(p['x']), int(p['y'])), int(p['r']), (0, 255, 0), 2)
                    cv.circle(frame, (int(p1[0]), int(init_y1)), int(p1[3]/2), (0, 255, 0), 2)
                    cv.circle(frame, (int(p2[0]), int(init_y2)), int(p2[3]/2), (0, 255, 0), 2)

                # To determine the amount to scroll, take the average
                # of the two y-deltas and scale it by the viewport height/1000.
                # 
                # The viewport height scaling factor is used because of the 
                # typical pixel height of my web browser. This would need
                # to be changed for different viewports.
                #
                # The scroll amount is also negated since positive y-delta's
                # correspond to downward movement but positive scroll amounts
                # correspond to upwards scrolling.

                mean_delta = (delta_y1+delta_y2)/2.
                scroll_amt = -(mean_delta)/height*1000

                # if abs(scroll_amt) > t:
                #     if scroll_amt > 0:
                #         scroll_amt *= 10
                #     conn.send('scroll:' + str(scroll_amt))

                gui.vscroll(scroll_amt)


            if display:
                cv.imshow('frame', frame)

        key = cv.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    # conn.send('close')
    # conn.close()

    cv.destroyAllWindows()
    cap.release()

if __name__ == '__main__':
    # _ = subprocess.Popen(['python', 'scroller.py'])
    # sleep(1.0)

    signal.signal(signal.SIGUSR1, read_dimensions)
    signal.signal(signal.SIGUSR2, start_calibration)

    perform_capture()