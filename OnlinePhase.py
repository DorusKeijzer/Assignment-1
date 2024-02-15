import numpy as np
import cv2 as cv
import glob
from constants import *
import utils 
    
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARDWIDTH*CHESSBOARDHEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARDHEIGHT,0:CHESSBOARDWIDTH].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
cubeaxis = np.float32([[0,0,0], [0,3,0], [3,3,0], [3,0,0],
                   [0,0,-3],[0,3,-3],[3,3,-3],[3,0,-3] ])

def drawcube(img, corners, imgpts):
    imgpts = np.int32(imgpts).reshape(-1,2)
    # draw ground floor in green
    img = cv.drawContours(img, [imgpts[:4]],-1,(0,255,0),-3)
    # draw pillars in blue color
    for i,j in zip(range(4),range(4,8)):
        img = cv.line(img, tuple(imgpts[i]), tuple(imgpts[j]),(255),3)
    # draw top layer in red color
    img = cv.drawContours(img, [imgpts[4:]],-1,(0,0,255),3)
    return img


if __name__ == "__main__":

    for run in ["1","2", "3"]:
        print(f"Run {run}:")
        # images for every run are stored in seperate folders
        images = glob.glob(f'Images/run{run}/*.jpg')
        
        # Load the callibration matrix for this run
        with np.load(f'results/Calibration_run{run}.npz') as X:
            mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]

            for number, filename in enumerate(images):
                print(f"Image {number+1}: {filename}")

                img = cv.imread(filename)
                gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
                ret, corners = cv.findChessboardCorners(gray, (CHESSBOARDHEIGHT,CHESSBOARDWIDTH),None)

                # if the chessboard is detected
                if ret: 
                    corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
                    # Find the rotation and translation vectors.
                    ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
                    
                    # project 3D points to image plane
                    imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)

                    img = utils.drawaxes(img,corners2,imgpts)
                    cv.imshow('img',img)
                    k = cv.waitKey(0) & 0xFF
                    if k == ord('s'):
                        cv.imwrite(filename[:6]+'.png', img)

                cv.destroyAllWindows()
