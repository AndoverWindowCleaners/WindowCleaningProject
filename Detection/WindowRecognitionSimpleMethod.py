#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:48:06 2020

@author: michaelhyh
"""

import numpy as np
import math
import cv2
import time

num_frame = 600
frame_width = 128
frame_height = 128
target_frame_width = 28
target_frame_height = 28
input_frames = np.zeros(
    (2, frame_width, frame_height), dtype=np.float32)
derivative1 = np.zeros(
    (2, frame_width, frame_height), dtype=np.float32)
cur_derivative2_corrected = np.zeros(
    (frame_width, frame_height), dtype=np.float32)
differences = np.zeros(
    (num_frame, frame_width, frame_height), dtype=np.float32)
# note that these numpy arrays are used as cyclic arrays
start_frame = 0
rotation_frequency = 1  # enter in revolution per second

# say the input frames are periodic and can be described by a*sin(bx+c)+d
# its derivative is a*b*cos(bx+c)
# its second derivative is -a*b*b*sin(bx+c)
# so f(x)-f'(x-pi/(2*b))/b should be constant so should f(x)+f"(x)/b/b
# the second is much better because the first involves a phase shift which
# requires access to data taken pi/(2*b) ago and will thus lower the processing
# speed or accuracy.
# the key is to know b, which is equal to frequency*2pi

frequency_const = rotation_frequency*2*math.pi

difference_sum = sum(differences)
difference_square_sum = sum([i**2 for i in differences])


def get_new_frame():    # to be implemented in coordination with the camera
    return np.zeros((2040,2040), dtype=np.float32)


def image_pooling(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def find_block(image):  # to be implemented
    pass

start = 0
count = 1
while True:
    if count % 100 == 0:
        print(time.time()-start)
        start = time.time()

    input_frames[start_frame%2] = image_pooling(get_new_frame(),frame_width,frame_height)
    delta_time = 1.0  # get the delta time using api provided by raspberry PI or arduino
    derivative1[start_frame%2] = (
        input_frames[start_frame%2]-input_frames[(start_frame-1)%2])/delta_time

    cur_derivative2_corrected = (derivative1[start_frame%2]-derivative1[(start_frame-1)%2])/delta_time
    cur_derivative2_corrected /= frequency_const**2



    difference_sum -= differences[start_frame-1]
    difference_square_sum -= differences[start_frame-1]**2
    differences[start_frame-1] = cur_derivative2_corrected + \
        input_frames[(start_frame-1)%2]
    difference_sum += differences[start_frame-1]
    difference_square_sum += differences[start_frame-1]**2
    variances = (difference_square_sum/num_frame-(difference_sum/num_frame)**2)
    # note this is only an estimation of variance, not the actual variance, which may be difficult
    # to evaluate on a rolling basis
        
    start_frame += 1
    start_frame %= num_frame

    variances = image_pooling(variances, target_frame_width, target_frame_height)
    find_block(variances)
    count += 1
