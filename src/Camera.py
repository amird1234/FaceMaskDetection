import numpy as np
import cv2 as cv
import time
cap = cv.VideoCapture('//Users/amirdahan/anaconda3/pkgs/torchvision-0.8.2-py38h83b45b8_1_cpu/info/test/test/assets/videos/RATRACE_wave_f_nm_np1_fr_goo_37.avi')
while cap.isOpened():
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    gray = frame #cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('frame', gray)
    time.sleep(0.1)
    if cv.waitKey(1) == ord('q'):
        break
cap.release()
cv.destroyAllWindows()
