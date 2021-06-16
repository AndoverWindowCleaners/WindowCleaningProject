
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
camera = PiCamera(resolution=(128, 96), framerate=30)
camera.framerate = 30
camera.vflip = True
camera.hflip = True
rawCapture = PiRGBArray(camera, size=(128, 96))
time.sleep(0.1)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
video_filename = '/home/pi/Desktop/WindowCleaningProject/new_video.avi'
out = cv2.VideoWriter(video_filename, fourcc, 30, (128, 96))

time.sleep(3)

start = time.time()

for frame in camera.capture_continuous(rawCapture, format="bgr", use_video_port=True):
    image = frame.array
    #cv2.imshow("Stream", image)
    out.write(image)
    rawCapture.truncate(0)
    #key = cv2.waitKey(1) & 0xFF
    if time.time()-start>30:
        break

#cv2.destroyAllWindows()
out.release()
os.system('cd /home/pi/Desktop/WindowCleaningProject')
os.system('git add .')
os.system('git commit -m "new video"')
os.system('git push')