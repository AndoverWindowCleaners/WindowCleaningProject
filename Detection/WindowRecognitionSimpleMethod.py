#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  2 13:48:06 2020

@author: michaelhyh
"""

import numpy as np
import math

num_frame = 5
frame_width=28
frame_height=28
input_frames = np.zeros((num_frame,frame_width,frame_height), dtype = 'float')
derivative1 = np.zeros((num_frame,frame_width,frame_height), dtype = 'float')
derivative2_corrected = np.zeros((num_frame,frame_width,frame_height), dtype = 'float')
differences = np.zeros((num_frame,frame_width,frame_height), dtype = 'float')
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
    pass

def image_pooling(image, half_box_size, stride):    
    # avery poor implementation, to be replaced
    # by prefix sum or rolling implementation
    newImage = np.zeros((int((len(image)-half_box_size*2+1)/stride),int((len(image[0])-half_box_size*2+1)/stride)),dtype='float')
    ri=0
    for row in range(half_box_size,len(image)-half_box_size,stride):
        ci=0
        for col in range(half_box_size,len(image[0])-half_box_size,stride):
            cutoutImage = image[(row-half_box_size):(row+half_box_size+1),(col-half_box_size):(col+half_box_size+1)]
            newImage[ri,ci] = sum(sum(cutoutImage))
            ci+=1
        ri+=1
            
def find_block(image):  # to be implemented
    pass

while(True):
    input_frames[start_frame] =  get_new_frame()
    delta_time = 1  # get the delta time using api provided by raspberry PI or arduino
    derivative1[start_frame] = (input_frames[start_frame]-input_frames[start_frame-1])/delta_time
    
    start_frame+=1
    if(start_frame>=num_frame):
        start_frame-=num_frame
    
    derivative2_corrected[start_frame-1] = (derivative1[start_frame]-derivative1[start_frame-1])/delta_time
    derivative2_corrected[start_frame-1]/=frequency_const**2
    difference_sum-=differences[start_frame-1]
    difference_square_sum-=differences[start_frame-1]**2
    differences[start_frame-1]=derivative2_corrected[start_frame-1]+input_frames[start_frame-1]
    variances = (difference_square_sum/num_frame-(difference_sum/num_frame)**2)
    # note this is only an estimation of variance, not the actual variance, which may be difficult
    # to evaluate on a rolling basis
    
    variances = image_pooling(variances,1,2)
    find_block(variances)
    