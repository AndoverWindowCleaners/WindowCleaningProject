import numpy as np
import cv2
import matplotlib.pyplot as plt

base_dir = './Detection/training_videos/'
#,'20210710-195636.avi','20210710-200436.avi','20210710-195957.avi','20210710-194508.avi'
video_paths = [
    '20210710-195636.avi',
    '20210710-194933.avi',
    '20210710-195636.avi',
    '20210710-200436.avi',
    '20210710-195957.avi',
    '20210710-194508.avi',
    '20210710-202536.avi'
]
video_paths = [base_dir+video_path for video_path in video_paths]
captures = [cv2.VideoCapture(video_path) for video_path in video_paths]
lengths = [int(capture.get(7)) for capture in captures]
fpss = [capture.get(5) for capture in captures]
durations = [length/fps for length,fps in zip(lengths,fpss)]
print(lengths)
all_frames = [np.zeros((length,1,1),dtype=np.float16) for length in lengths]

def image_pooling(image, new_width, new_height):
    return cv2.resize(image, (new_height, new_width), interpolation=cv2.INTER_AREA)

for j,length,capture in zip(range(len(lengths)),lengths,captures):
    for i in range(length):
        ret, frame = capture.read()
        if ret:
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
            frame = image_pooling(frame, 1, 1)
            all_frames[j][i] = frame

all_frames = [(all_frame/255.0)**2 for all_frame in all_frames]

all_coeffs = [np.fft.rfft(all_frame, axis=0) for all_frame in all_frames]

all_amplitudes = [2*np.abs(coeffs)/length for coeffs in all_coeffs]
for i in range(len(all_amplitudes)):
    all_amplitudes[i][0]=0
for i in range(len(lengths)):
    if lengths[i]%2==0:
        all_amplitudes[i][lengths[i]//2]/=2

# amplitudes[0]=0

all_freqs = [np.arange(amplitudes.shape[0])/duration for amplitudes,duration in zip(all_amplitudes,durations)]
print(durations)
for i, freqs, amplitudes in zip(range(len(all_freqs)),all_freqs, all_amplitudes):
    plt.plot(freqs,np.mean(amplitudes,axis=(1,2)),label = f'graph {i}',linewidth=0.5)
plt.legend()
plt.show()
