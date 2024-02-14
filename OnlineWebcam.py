import numpy as np
import cv2 as cv
import glob
from constants import * 

cam = cv.VideoCapture(0)
def draw(img, corners, imgpts):
    # Extracting corner coordinates properly
    corner = tuple(corners[0].ravel())
    corner = tuple(map(int, corner))
    # Extracting imgpts coordinates properly
    imgpts = np.int32(imgpts).reshape(-1, 2)
    imgpts = imgpts.astype(int)  # Convert imgpts to Python integers
    
    # Drawing lines from corner to imgpts        
    img = cv.line(img, corner, tuple(imgpts[0]), (255, 0, 0), 2)
    img = cv.line(img, corner, tuple(imgpts[1]), (0, 255, 0), 2)
    img = cv.line(img, corner, tuple(imgpts[2]), (0, 0, 255), 2)

    return img

while True:
    check, frame = cam.read()

    gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    ret, corners = cv.findChessboardCorners(gray, (6,9),None)
    cv.imshow('video', frame)

    key = cv.waitKey(1)
    if key == 27:
        break

cam.release()
cv.destroyAllWindows()
