from picamera import PiCamera
from queue import Queue
import cv2
import numpy as np
from tensorflow import lite
import time


# model = load_model('simpleLogistic', compile=True)

num_frame = 5
frame_width = 128
frame_height = 96
target_frame_width = 12
target_frame_height = 9
light_intensity_correction = 127.5


# Load the TFLite model and allocate tensors.
interpreter = lite.Interpreter(model_path="simpleLogistic.tflite")
interpreter.resize_tensor_input(
    0, [target_frame_width*target_frame_height, 2], strict=True)
interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

camera = PiCamera(resolution=(frame_width, frame_height), framerate=30)

input_frames = np.zeros(
    (num_frame, target_frame_width, target_frame_height), dtype=np.float32)
derivative1 = np.zeros(
    (2, target_frame_width, target_frame_height), dtype=np.float32)
cur_derivative2_corrected = np.zeros(
    (target_frame_width, target_frame_height), dtype=np.float32)
differences = np.zeros(
    (num_frame, target_frame_width, target_frame_height), dtype=np.float32)
# note that these numpy arrays are used as cyclic arrays
start_frame = 0
rotation_frequency = 150  # enter in revolution per second

# say the input frames are periodic and can be described by a*sin(bx+c)+d
# its derivative is a*b*cos(bx+c)
# its second derivative is -a*b*b*sin(bx+c)
# so f(x)-f'(x-pi/(2*b))/b should be constant so should f(x)+f"(x)/b/b
# the second is much better because the first involves a phase shift which
# requires access to data taken pi/(2*b) ago and will thus lower the processing
# speed or accuracy.
# the key is to know b, which is equal to frequency*2pi
frequency_const = rotation_frequency*2*np.pi

diff_mean = np.mean(differences, axis=0)
input_mean = np.mean(input_frames, axis=0)
input_varSum = np.var(input_frames, axis=0)
diff_varSum = np.var(differences, axis=0)

# def get_new_frame():
#     # note that frame_height and frame_width are reversed
#     frame = np.ones((frame_height, frame_width, 3), dtype='uint8')
#     camera.capture(frame, format='rgb', use_video_port=True)
#     frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
#     return np.transpose(frame)


def image_pooling(image, new_width, new_height):
    return cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_AREA)


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
    # output = model.predict(features)
    # output = output.reshape(diff_variance.shape)
    # return output
    # Test the model on random input data.
    input_shape = input_details[0]['shape']
    interpreter.set_tensor(
        input_details[0]['index'], features)

    interpreter.invoke()

    # The function `get_tensor()` returns a copy of the tensor data.
    # Use `tensor()` in order to get a pointer to the tensor.
    output_data = interpreter.get_tensor(output_details[0]['index'])
    output_data = output_data.reshape(diff_variance.shape)
    return output_data


def find_block(diff_variance, input_variance):
    # to be modified; obviously the input variance has to be larger
    # than some certain value and the diff_variance has to be smaller
    # than some certain value for that pixel to be considered a window
    windowLog = get_probability(diff_variance, input_variance)
    windowPixels = windowLog > 0.5
    windowPositions = []
    for x in range(target_frame_width):
        for y in range(target_frame_height):
            if not windowPixels[x, y]:
                continue
            pixelCount = 0.0
            windowPixels[x, y] = True
            windowPositions.append(np.zeros((2), dtype=np.float32))
            frontier = Queue(target_frame_height*target_frame_width)
            frontier.put_nowait((x, y))
            while not frontier.empty:
                (thisX, thisY) = frontier.get_nowait()
                currentTotalPos = windowPositions[-1]*pixelCount
                pixelCount += windowLog[thisX, thisY]
                windowPositions[-1] \
                    = (currentTotalPos+np.array((thisX, thisY))*windowLog[thisX, thisY])/pixelCount
                for hori in range(-1, 2):
                    for verti in range(-1, 2):
                        newX = thisX+hori
                        newY = thisY+verti
                        if newY < 0 or newY >= target_frame_height or newX < 0 or newX >= target_frame_width \
                                or not windowPixels[newX, newY]:
                            continue
                        windowPixels[newX, newY] = False
                        frontier.put_nowait((newX, newY))
    return windowPositions
    # returns the average positions of windows as weighted by their probability of being a window


def computeRollingVarianceSum(varSum, mean, prev_mean, cur, prev):
    return varSum + (cur-prev_mean)*(cur-mean)-(prev-prev_mean)*(prev-mean)


start = 0
capture = np.ones((frame_height, frame_width, 3), dtype='uint8')
for frame in camera.capture_continuous(capture, format="bgr", use_video_port=True):
    delta_time = time.time()-start
    start = time.time()

    # dequeue mean and sum
    prev_input_mean = input_mean.copy()
    prev_diff_mean = diff_mean.copy()
    input_mean -= input_frames[start_frame]/num_frame
    diff_mean -= differences[start_frame-1]/num_frame
    prev_input = input_frames[start_frame].copy()
    prev_diff = differences[start_frame-1].copy()

    # read in image
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    input_frames[start_frame] = image_pooling(
        np.transpose(frame), target_frame_width, target_frame_height)

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
    input_mean += input_frames[start_frame]/num_frame
    diff_mean += differences[start_frame-1]/num_frame

    # recompute variances
    input_varSum = computeRollingVarianceSum(input_varSum,
                                             input_mean, prev_input_mean, input_frames[start_frame], prev_input)
    diff_varSum = computeRollingVarianceSum(diff_varSum,
                                            diff_mean, prev_diff_mean, differences[start_frame-1], prev_diff)
    # note this is only an estimation of variance, not the actual variance, which may be difficult
    # to evaluate on a rolling basis

    diff_variance = diff_varSum/num_frame
    input_variance = input_varSum/num_frame

    # increment the modulo index counter
    start_frame = (start_frame+1) % num_frame
    # find windows
    windowPos = find_block(diff_variance/(light_intensity_correction**2),
                           input_variance/(light_intensity_correction**2))
    print(np.mean(input_variance), np.mean(diff_variance),
          np.mean(input_frames[start_frame]), np.mean(cur_derivative2_corrected), np.mean(differences[start_frame-1]))
    if len(windowPos) > 0:
        print(windowPos[0])
    else:
        print('no window found')
