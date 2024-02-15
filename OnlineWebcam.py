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


if __name__ == "__main__":

    print("Hold a chessboard up in front of the camera to callibrate")
    print("Press esc to close")
    objp = np.zeros((CHESSBOARDWIDTH*CHESSBOARDHEIGHT,3), np.float32)
    objp[:,:2] = np.mgrid[0:CHESSBOARDWIDTH,0:CHESSBOARDHEIGHT].T.reshape(-1,2)

    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    calibrated = False
    aaaaaaaa = True
    while True:
        check, frame = cam.read()    

        # if there are not enough succesful frames to have calibrated with    
        if len(imgpoints) == NUMBERTOCALIBRATE and not calibrated:
            print("\nCalibrating...")
            # calibrate using built in function
            calibrateret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, grayphoto.shape[::-1], None, None)
            if calibrateret:
                print("Calibrated succesfully")
                calibrated = True
        
        elif calibrated: 
            # convert to greyscale
            gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(gray, (CHESSBOARDWIDTH,CHESSBOARDHEIGHT),None)
            
            if ret: # if a chessboard is detected
                # Find the rotation and translation vectors.
                ret,rvecs, tvecs = cv.solvePnP(objp, corners, mtx, dist)
                # project 3D points to image plane
                imgpts, jac = cv.projectPoints(AXIS, rvecs, tvecs, mtx, dist)
                # draw axes on the chessboard
                img = utils.drawaxes(frame,corners,imgpts[0:3])
                img = utils.drawcube(frame,corners,imgpts[3:])
 
        else:
            photo = frame
            # convert to greyscale and attempt to detect a chessboard
            grayphoto = cv.cvtColor(photo,cv.COLOR_BGR2GRAY)
            ret, corners = cv.findChessboardCorners(grayphoto, (CHESSBOARDWIDTH,CHESSBOARDHEIGHT),None)
            # save corners if a chessboard has been found
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)
                print(f"Collected {len(objpoints)}/{NUMBERTOCALIBRATE} succesful calibration pictures", end= "\r")

        key = cv.waitKey(1)
        if key == 27: # esc = exit loop, i.e. exit webcam
            break
        cv.imshow('video', frame)
    cam.release()
    cv.destroyAllWindows()
