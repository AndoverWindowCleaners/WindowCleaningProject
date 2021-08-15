import cv2
import os


EVERY_OTHER = 10

name = '20210710-195253.avi'

file_name = 'training_videos/' + name

new_folder = 'images/' + name

print(new_folder)

if not os.path.exists(new_folder):
    os.makedirs(new_folder)

cap = cv2.VideoCapture(file_name)

i = 0
num_images = 0

while cap.isOpened():
    ret, frame = cap.read()
    if i == 0:
        new_name = "%04d" % num_images
        cv2.imwrite(new_folder + '/' + new_name + '.jpg', frame)
        num_images += 1

    i = (i + 1) % EVERY_OTHER


cap.release()
cv2.destroyAllWindows()
