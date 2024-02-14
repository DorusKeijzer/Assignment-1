import numpy as np
import cv2 as cv
import glob

images = glob.glob('Images/*.jpg')
CHESSBOARDWIDTH = 6
CHESSBOARDHEIGHT = 9
RESIZEDWIDTH = 400

# Load the callibration matrix found in the offline phase
with np.load('Calibration.npz') as X:
    mtx, dist, rvecs, tvecs = [X[i] for i in ('mtx','dist','rvecs','tvecs')]
    
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
objp = np.zeros((CHESSBOARDWIDTH*CHESSBOARDHEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARDHEIGHT,0:CHESSBOARDWIDTH].T.reshape(-1,2)
axis = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)



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


def resizeImage(img, resize_width):
    """Resizes the image so that the image is of the specified width"""
    height, width = img.shape[:2]
    img = cv.resize(img, (resize_width, int(height*(resize_width/width))))
    return img


if __name__ == "__main__":
    for filename in images:
        img = cv.imread(filename)
        img = resizeImage(img, RESIZEDWIDTH)
        gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
        ret, corners = cv.findChessboardCorners(gray, (CHESSBOARDHEIGHT,CHESSBOARDWIDTH),None)

        if ret == True:
            corners2 = cv.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
            # Find the rotation and translation vectors.
            ret,rvecs, tvecs = cv.solvePnP(objp, corners2, mtx, dist)
            
            # project 3D points to image plane
            imgpts, jac = cv.projectPoints(axis, rvecs, tvecs, mtx, dist)
            img = draw(img,corners2,imgpts)
            cv.imshow('img',img)
            k = cv.waitKey(0) & 0xFF
            if k == ord('s'):
                cv.imwrite(filename[:6]+'.png', img)

        cv.destroyAllWindows()
