from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
camera = PiCamera(resolution=(128, 96), framerate=15)
camera.framerate = 15
rawCapture = PiRGBArray(camera, size=(128, 96))
time.sleep(0.1)
for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    cv2.imshow("Stream", image)
    key = cv2.waitKey(1) & 0xFF
    rawCapture.truncate(0)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
