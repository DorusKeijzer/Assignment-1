import numpy as np
import cv2 as cv
import os
import glob
from constants import *
import utils 
    
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARDWIDTH*CHESSBOARDHEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARDHEIGHT,0:CHESSBOARDWIDTH].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)

modified_folder = "results/images/"

if __name__ == "__main__":

    for run in ["1","2","3"]:
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
                    # imgpts contains both the points for the axes (index 0 to 3) 
                    # and for the cube (other indices)
                    imgpts, _ = cv.projectPoints(AXIS, rvecs, tvecs, mtx, dist )
                    # draw the cube and the axes
                    img = utils.drawaxes(img, corners2, imgpts[0:3])
                    img = utils.drawcube(img, corners2, imgpts[3:])
                    cv.imshow('img',img)
                    k = cv.waitKey(0) & 0xFF
                    # if the user presses s, save the image to the results folder
                    if k == ord('s'):
                        path = os.path.basename(filename)
                        path = os.path.splitext(path)[0]+'_axes.png'
                        modified_image_path = os.path.join(modified_folder, path)
                        print(f"Saved to {modified_image_path}")
                        cv.imwrite(modified_image_path, img)
    
                cv.destroyAllWindows()

