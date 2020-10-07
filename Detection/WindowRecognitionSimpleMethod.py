import numpy as np
import math
import cv2
import time
from queue import Queue

num_frame = 45
frame_width = 128
frame_height = 128
target_frame_width = 28
target_frame_height = 28
light_intensity_correction = 127.5


input_frames = np.zeros(
    (num_frame, frame_height, frame_width), dtype=np.float32)
derivative1 = np.zeros(
    (2, frame_height, frame_width), dtype=np.float32)
cur_derivative2_corrected = np.zeros(
    (frame_height, frame_width), dtype=np.float32)
differences = np.zeros(
    (num_frame, frame_height, frame_width), dtype=np.float32)
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

difference_sum = np.sum(differences,axis=0)
difference_square_sum = np.sum(np.square(differences),axis=0)
input_sum = np.sum(input_frames,axis=0)
input_square_sum = np.sum(np.square(input_frames),axis=0)


def get_new_frame():    # to be implemented in coordination with the camera
    return np.zeros((2040, 2040), dtype=np.float32)


def image_pooling(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def get_logarithmic(diff_variance, input_variance):
    # to be implemented, get log probability of being window
    return (0.5-diff_variance)*(input_variance-0.5)  # just a place holder


def find_block(diff_variance, input_variance):
    # to be modified; obviously the input variance has to be larger
    # than some certain value and the diff_variance has to be smaller
    # than some certain value for that pixel to be considered a window
    windowMarker = np.zeros(
        (target_frame_height, target_frame_width), dtype=np.int)
    windowLog = get_logarithmic(diff_variance, input_variance)
    windowPixels = windowLog > 0
    windowPositions = []
    windowCount = 0
    for r in range(target_frame_height):
        for c in range(target_frame_width):
            if not windowPixels[r, c]:
                continue
            windowCount += 1
            pixelCount = 0.0
            windowPositions.append(np.zeros((2), dtype=np.float32))
            frontier = Queue(target_frame_height*target_frame_width)
            frontier.put_nowait((r, c))
            while not frontier.empty:
                (thisR, thisC) = frontier.get_nowait()
                currentTotalPos = windowPositions[windowCount-1]*pixelCount
                pixelCount += windowLog[thisR, thisC]
                windowPositions[windowCount - 1] \
                    = (currentTotalPos+np.array((thisR, thisC))*windowLog[thisR, thisC])/pixelCount
                for hori in range(-1, 2):
                    for verti in range(-1, 2):
                        newR = thisR+verti
                        newC = thisC+hori
                        if newR < 0 or newR >= frame_height or newC < 0 or newC >= frame_width \
                                or windowMarker[newR, newC] > 0:
                            continue
                        windowMarker[newR, newC] = windowCount
                        frontier.put_nowait((newR, newC))

    return windowPositions
    # returns the average positions of windows as weighted by their probability of being a window

def computeRollingVariance(square_sum, sum, num_elements):
    return (square_sum/num_elements-(sum/num_elements)**2)

start = 0
count = 1
while True:
    delta_time = 1.0  # get the delta time using api provided by raspberry PI or arduino
    count += 1
    if count % 100 == 0:
        print('time ', time.time()-start)
        start = time.time()

    # dequeue variance
    input_sum -= input_frames[start_frame]
    input_square_sum -= input_frames[start_frame]**2
    difference_sum -= differences[start_frame-1]
    difference_square_sum -= differences[start_frame-1]**2

    # read in image
    input_frames[start_frame] = image_pooling(
        get_new_frame(), frame_width, frame_height)

    # compute first derivative
    derivative1[start_frame % 2] = (
        input_frames[start_frame]-input_frames[(start_frame-1)])/delta_time

    # compute second derivative and correct its coefficient
    cur_derivative2_corrected = (
        derivative1[start_frame % 2]-derivative1[(start_frame-1) % 2])/delta_time
    cur_derivative2_corrected /= frequency_const**2

    # compute difference between image and its second derivative. It's actually a +
    # because of the negative sign from differentiation
    differences[start_frame-1] = cur_derivative2_corrected + \
        input_frames[(start_frame-1)]

    # add in new variance of the newly read in image and newly computed difference
    input_sum += input_frames[start_frame]
    input_square_sum += input_frames[start_frame]**2
    difference_sum += differences[start_frame-1]
    difference_square_sum += differences[start_frame-1]**2

    # recompute variances
    input_variance = computeRollingVariance(input_square_sum,input_sum,num_frame)
    variances = computeRollingVariance(difference_square_sum,difference_sum,num_frame)
    # note this is only an estimation of variance, not the actual variance, which may be difficult
    # to evaluate on a rolling basis

    # scale down variance to ensure connectiveness
    variances = image_pooling(
        variances, target_frame_width, target_frame_height)
    input_variance = image_pooling(
        input_variance, target_frame_width, target_frame_height)

    # increment the modulo index counter
    start_frame = (start_frame+1) % num_frame

    # find windows
    windowPos = find_block(variances/(light_intensity_correction**2),
                           input_variance/(light_intensity_correction**2))
