#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 21 21:22:58 2018

@author: zheweiqiu
"""

import cv2


def init_tracker(init_bb, tracker_name, frame, target, orig):
    tracker_dict = {
        "csrt": cv2.TrackerCSRT_create,
        "kcf": cv2.TrackerKCF_create,
        "boosting": cv2.TrackerBoosting_create,
        "mil": cv2.TrackerMIL_create,
        "tld": cv2.TrackerTLD_create,
        "medianflow": cv2.TrackerMedianFlow_create,
        "mosse": cv2.TrackerMOSSE_create
    }

    # Check tracker type
    if tracker_name not in tracker_dict:
        print('Invalid tracker name.')
        return
    
    if not init_bb:
        print('None bounding box')
        return
    
    tracker = tracker_dict[tracker_name]()
    if target == 'pupil':
        #bb = circle2bb(init_bb[0]-orig[0], init_bb[1]-orig[1], init_bb[2])
        bb = circle2bb(init_bb[0], init_bb[1], init_bb[2])
    else:
        #bb = ellipse2bb(init_bb[0]-orig[0], init_bb[1]-orig[1], init_bb[2], init_bb[3])
        bb = ellipse2bb(init_bb[0], init_bb[1], init_bb[2], init_bb[3])
    
    tracker.init(frame, bb)
    return tracker


def track(tracker, frame, target, orig):
    (success, bb) = tracker.update(frame)    
    #return (int(bb[0]+orig[0]), int(bb[1]+orig[1]), int(bb[2]), int(bb[3]))
    return (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))


# Convert circle to bounding box
def circle2bb(x, y, radius):
    new_x = int(x-radius)
    new_y = int(y-radius)
    w = int(2*radius)
    h = int(2*radius)
    return (new_x, new_y, w, h)


# Convert bounding box to circle
def bb2circle(bb):
    radius = int(bb[2]/2)
    x = int(bb[0]+radius)
    y = int(bb[1]+radius)
    return (x, y, radius)


def ellipse2bb(x, y, x_axis, y_axis):
    new_x = int(x - x_axis/2)
    new_y = int(y - y_axis/2)
    w = int(2*x_axis)
    h = int(2*y_axis)
    return (new_x, new_y, w, h)


def bb2ellipse(bb):
    x = int(bb[0] + bb[2]/2)
    y = int(bb[1] + bb[3]/2)  
    x_axis = int(bb[2]/2)
    y_axis = int(bb[3]/2)
    return (x, y, x_axis, y_axis)


# Convert a bounding box to a circle and draw the circle
def draw_circle(frame, bb):
    circle = bb2circle(bb)
    cv2.circle(frame, 
               center = (circle[0], circle[1]), 
               radius = circle[2], 
               color = (0, 0, 255), 
               thickness = 2)
    #cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1]+ bb[3]), color = (206, 0, 209))
    #return frame


# Convert a bounding box to a ellipse and draw the elllipse
def draw_ellipse(frame, bb):
    ellipse = bb2ellipse(bb)
    cv2.ellipse(frame,
                center = (ellipse[0], ellipse[1]),
                axes = (ellipse[2], ellipse[3]),
                angle = 0,
                startAngle = 0,
                endAngle = 360,
                #color = (206, 0, 209),
                color = (0, 0, 255),
                thickness = 2)
    #cv2.rectangle(frame, (bb[0], bb[1]), (bb[0] + bb[2], bb[1]+ bb[3]), color = (206, 0, 209))
    #return frame