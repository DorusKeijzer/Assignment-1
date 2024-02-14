import numpy as np
import cv2 as cv
import glob
from constants import * 
import utils

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARDWIDTH*CHESSBOARDHEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARDHEIGHT,0:CHESSBOARDWIDTH].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

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

if __name__ == "__main__":

    print("Hold a chessboard up in front of the camera to callibrate")
    print("Press space to take a picture")
    pictures = []
    while True:
        check, frame = cam.read()
        cv.imshow('video', frame)
        
        if len(pictures) == 20:
            print("AAAAAAAAAa")
        else:
            photo = frame
            grayphoto = cv.cvtColor(photo,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(grayphoto, (CHESSBOARDWIDTH,CHESSBOARDHEIGHT),None)
            if ret:
                print("Chessboard detected")
                pictures.append(photo)
                print(len(pictures))

        key = cv.waitKey(1)
        if key == 27: # esc = exit webcam
            break


        
    cam.release()
    cv.destroyAllWindows()
