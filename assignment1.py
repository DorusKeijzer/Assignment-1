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

def click_event(event, x, y, flags, params): 
    if event == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), 3, 400, -1)
        clickcorners.append([x,y])
        cv.imshow('Image', img) 

def manualCorners(img, chessboardwidth, chessboardheight) -> np.array:
    """ALlows the user to specify the corners. 
    These corners should be given in the same order as the program does.
    returns an array of the correct size with the corner points"""
    
    # The mouse click event writes the corners to this variable, hence it's used here.
    global clickcorners
    
    # Shows the image so the user can click the corners
    cv.imshow('Image', img) 

    # Waits for the user to click 4 corners
    while(len(clickcorners)<4):
        cv.waitKey(3)
    
    # stores these corners in an np array 
    res = np.array(clickcorners, dtype=np.float32)

    clickcorners = []
    cv.destroyAllWindows()
 
    interpolation = interpolate(res, chessboardwidth, chessboardheight)

    return interpolation

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
    return img

def interpolate(corners, chessboardwidth, chessboardheight):
    """Expects:
       the pixel coordinates of the 4 corners of a chessboard in an np array of size 4
       the proportions of the chessboard, 

       returns where the internal crossings are as a np array of (chessboardwidth x chessboard height) """

    # the coordinates of the outer corners of the chessboard in chessboard coordinates
    # these will be used to solve for the transformation matrix
    corner_chesscoordinates = np.float32([[0,0], 
                         [0,chessboardwidth-1], 
                         [chessboardheight-1, 0], 
                         [chessboardheight-1, chessboardwidth-1]])   
    
    # Finds the transformation matrix 
    TransformationMatrix = cv.getPerspectiveTransform(corner_chesscoordinates, corners)

    # The coordinates of the chessboard corners in chessboard coordinates (i.e. [[[0,0]],[[0,1]],[[0,2]],...
    #                                                                            [[1,1]],[[1,2]],[[1,3]],..)
    chessboard_coords = np.float32([[[x, y]] for x in range(chessboardheight) for y in range(chessboardwidth)])

    # applies the transformation
    inner_corners = cv.perspectiveTransform(chessboard_coords, TransformationMatrix)
    return inner_corners.reshape(-1, 1, 2)

if __name__=="__main__":   
    # reading the image 

    for number, filename in enumerate(images):
        print(f"Image {number}: {filename}")
        img = prepareImage(filename)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, (CHESSBOARDWIDTH,CHESSBOARDHEIGHT), None)
        # If the built in function cannot distinguish the corner points, the user must manually mark the corners
        if ret: 
            corners = cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        else:
            cv.imshow('Image',img)
            cv.setMouseCallback('Image', click_event) 
            corners = manualCorners(img, CHESSBOARDWIDTH, CHESSBOARDHEIGHT)
            ret = True

        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESSBOARDWIDTH, CHESSBOARDHEIGHT), corners, ret)
        cv.imshow('Image', img)
        cv.waitKey()

        # close the window 
        cv.destroyAllWindows() 

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    np.savez('Calibration.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
