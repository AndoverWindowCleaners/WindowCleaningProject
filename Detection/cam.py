
from picamera.array import PiRGBArray
from picamera import PiCamera
import time
import cv2
import os
from datetime import datetime
from gpiozero import LED

rotor = LED(4)
camera = PiCamera(resolution=(128, 96), framerate=30)
camera.framerate = 30
camera.vflip = True
camera.hflip = True
rawCapture = PiRGBArray(camera, size=(128, 96))
time.sleep(0.1)

fourcc = cv2.VideoWriter_fourcc('M','J','P','G')
file_name = datetime.now().strftime("%Y%m%d-%H%M%S")
print(file_name)
video_filename = '/home/pi/Desktop/WindowCleaningProject/training_videos/'+file_name+'.avi'
print(video_filename)
out = cv2.VideoWriter('/home/pi/Desktop/WindowCleaningProject/test.avi', fourcc, 30, (128, 96))

time.sleep(3)

rotor.on()
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
rotor.off()
# time.sleep(1)
# os.system('cd /home/pi/Desktop/WindowCleaningProject')
# os.system('git add .')
# os.system('git commit -m "new video"')
# os.system('git push')
