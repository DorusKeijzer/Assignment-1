import cv2 as cv
import numpy as np
import glob

# Constants
RESIZEDWIDTH = 400
SQUARESIZE = 22 # milimeters
CHESSBOARDWIDTH = 6
CHESSBOARDHEIGHT = 9

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Stores the corner points.
objp = np.zeros((CHESSBOARDWIDTH*CHESSBOARDHEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARDHEIGHT,0:CHESSBOARDWIDTH].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.



# Gets the filenames of the images in the Images directory
images = glob.glob('Images/*.jpg')

# temporarily stores the corners of one image resulting from the click event
clickcorners = []

# temporarily stores the corners of one image resulting from the click event
clickcorners = []

def click_event(event, x, y, flags, params): 
    if event == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), 3, 400, -1)
        clickcorners.append([x,y])
        cv.imshow('image', img) 

def manualCorners(img, chessboardwidth, chessboardheight) -> np.array:
    """ALlows the user to specify the corners. 
    These corners should be given in the same order as the program does.
    returns an array of the correct size with the corner points"""
    global clickcorners
    cv.imshow('image', img) 
    while(len(clickcorners)<chessboardwidth*chessboardheight):
        cv.waitKey(3)
    
    cv.destroyAllWindows()
    res = np.array(clickcorners, dtype=np.float32).reshape(-1, 1, 2)
    clickcorners = []
    return res

def resizeImage(img, resize_width):
    """Resizes the image so that the image is of the specified width"""
    height, width = img.shape[:2]
    img = cv.resize(img, (resize_width, int(height*(resize_width/width))))
    return img

def prepareImage(filename):
    """Reads the image and prepares it for being processed"""
    img = cv.imread(filename)
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = resizeImage(img, RESIZEDWIDTH)

    cv.namedWindow('image')
    return img



if __name__=="__main__":   
    # reading the image 

    for number, filename in enumerate(images):
        print(f"Image {number}: {filename}")
        img = prepareImage(filename)
        img = prepareImage(filename)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, (CHESSBOARDWIDTH,CHESSBOARDHEIGHT), None)
        # If the built in function cannot distinguish the corner points, the user must manually mark the corners
        if not ret:
            cv.setMouseCallback('image', click_event) 
            corners = manualCorners(img, CHESSBOARDWIDTH, CHESSBOARDHEIGHT)
            ret = True

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESSBOARDHEIGHT,CHESSBOARDWIDTH), corners2, ret)
        cv.imshow('Image', img)
        cv.waitKey(5000)
        cv.drawChessboardCorners(img, (CHESSBOARDHEIGHT,CHESSBOARDWIDTH), corners2, ret)
        cv.imshow('Image', img)
        cv.waitKey(5000)

        # close the window 
        cv.destroyAllWindows() 

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    print(ret, mtx, dist, rvecs, tvecs)
