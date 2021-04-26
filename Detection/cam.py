from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
camera = PiCamera(resolution=(128, 96), framerate=30)
camera.framerate = 30
rawCapture = PiRGBArray(camera, size=(128, 96))
print(camera.framerate_range.high)
time.sleep(0.1)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video_filename = 'indoor.avi'
out = cv2.VideoWriter(video_filename, fourcc, 30, (128, 96))

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("Stream", image)
    out.write(image)
    rawCapture.truncate(0)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
out.release()