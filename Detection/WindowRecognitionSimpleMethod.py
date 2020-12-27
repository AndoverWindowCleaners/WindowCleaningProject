from tensorflow.keras.models import load_model
import numpy as np
import cv2
import time
from queue import Queue
from picamera import PiCamera

model = load_model('simpleLogistic', compile=True)

num_frame = 45
frame_width = 128
frame_height = 96
target_frame_width = 28
target_frame_height = 21
light_intensity_correction = 127.5

camera = PiCamera(resolution=(frame_width, frame_height), framerate=15)

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
frequency_const = rotation_frequency*2*np.pi

difference_sum = np.sum(differences, axis=0)
difference_square_sum = np.sum(np.square(differences), axis=0)
input_sum = np.sum(input_frames, axis=0)
input_square_sum = np.sum(np.square(input_frames), axis=0)


def get_new_frame():
    # note that frame_height and frame_width are reversed
    frame = np.ones((frame_height, frame_width, 3), dtype='uint8')
    camera.capture(frame, format='rgb', use_video_port=True)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return np.transpose(frame)


def image_pooling(image, new_width, new_height):
    return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)


def get2D(diff_variances, input_variances):
    """
    diff_variance and input_variance are three dimensional numpy arrays, with third dimension being frame number
    Before inputting into this function, stack every frame from all images to diff_variance and input_variance

    Returns:
    2d numpy array with [variance, variance] as each row
    """
    return np.transpose(np.array((diff_variances.flatten(), input_variances.flatten())))


def get_probability(diff_variance, input_variance):
    features = get2D(diff_variance, input_variance)
    output = model.predict(features)
    output = output.reshape(diff_variance.shape)
    return output


def find_block(diff_variance, input_variance):
    # to be modified; obviously the input variance has to be larger
    # than some certain value and the diff_variance has to be smaller
    # than some certain value for that pixel to be considered a window
    windowMarker = np.zeros(
        (target_frame_height, target_frame_width), dtype=np.int)
    windowLog = get_probability(diff_variance, input_variance)
    windowPixels = windowLog > 0.5
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


def computeRollingVariance(square_sum, s, num_elements):
    return (square_sum/num_elements-np.square((s/num_elements)))


start = 0
while True:
    # get the delta time using api provided by raspberry PI or arduino
    delta_time = time.time()-start
    start = time.time()
    print('deltatime ', time.time()-start)

    # dequeue variance
    input_sum -= input_frames[start_frame]
    input_square_sum -= np.square(input_frames[start_frame])
    difference_sum -= differences[start_frame-1]
    difference_square_sum -= np.square(differences[start_frame-1])

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
    input_variance = computeRollingVariance(
        input_square_sum, input_sum, num_frame)
    variances = computeRollingVariance(
        difference_square_sum, difference_sum, num_frame)
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
    print(np.mean(variances), np.mean(input_variance))
    if len(windowPos) > 0:
        print(windowPos[0])
    else:
        print('no window found')
