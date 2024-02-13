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
        cv.imshow('image', img) 

def manualCorners(img, chessboardwidth, chessboardheight) -> np.array:
    """ALlows the user to specify the corners. 
    These corners should be given in the same order as the program does.
    returns an array of the correct size with the corner points"""
    
    # The mouse click event writes the corners to this variable, hence it's used here.
    global clickcorners
    
    # Shows the image so the user can click the corners
    cv.imshow('image', img) 

    # Waits for the user to click 4 corners
    while(len(clickcorners)<4):
        cv.waitKey(3)
    
    # stores these corners in an np array 
    res = np.array(clickcorners, dtype=np.float32)

    clickcorners = []
    cv.destroyAllWindows()
 
    interpolation = interpolateCorners(res, chessboardwidth, chessboardheight)

    return interpolation

def homogenize(vector):
    """Returns the homogeneous vector given an input vector"""
    return np.hstack((vector, np.ones((vector.shape[0], 1))))

def heterogenize(vector):
    return vector[:, :-1]


def fitAffine(source, destination):
    """ When given two sets of input vectors of corresponding size, 
        returns the affine transformation that moves source points to destination"""
    # makes homogeneous coordinates out of source coordinates by appending a 1 to each coordinate
    homo_source = homogenize(source)

    # Fits these coordinates
    T, _ = np.linalg.lstsq(homo_source, destination, rcond=None)[0:2]

    # this yields a 3 x 2 transformation matrix. 
    # We need to append [0,0,1].T to turn this into the full transformation matrix
    new_column = np.array([[0], [0], [1]])

    # Concatenate the new column to the original array
    Transformation = np.hstack((T, new_column))

    return Transformation.T

    
    #



def interpolateCorners(corners, chessboardwidth, chessboardheight) -> np.array:
    """Expects:
       the pixel coordinates of the 4 corners of a chessboard in an np array of size 4
       the proportions of the chessboard, 

       returns where the internal crossings are as a np array of (chessboardwidth x chessboard height) """
    
    # the coordinates of the outer corners of the chessboard in chessboard coordinates
    # these will be used to solve for the transformation matrix
    PreTransformCorners = np.array([[0,0], 
                                   [0,chessboardwidth-1], 
                                   [chessboardheight-1, 0], 
                                   [chessboardheight-1, chessboardwidth-1]])    
    
    # Finds the transformation matrix using least squares
    Transformation = fitAffine(PreTransformCorners, corners)

    # The coordinates of the chessboard corners in chessboard coordinates (i.e. [[[0,0],[1,0],[2,0],...
    #                                                                            [[1,1],[1,2],[1,3],..)
    # these will be transformed to yield the image coordinates of the corners
    chessboardCoords = np.mgrid[0:chessboardheight,0:chessboardwidth].T.reshape(-1,2)
    chessboardCoords = homogenize(chessboardCoords)
    

    # Performs the transformation on the chessboard points
    PostTransformationChessboard = np.dot(Transformation, chessboardCoords.T).T
    
    # returns to non homogenous coordinates
    PostTransformationChessboard = heterogenize(PostTransformationChessboard)
    return PostTransformationChessboard.reshape((54, 1, 2))
    
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

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, (CHESSBOARDWIDTH,CHESSBOARDHEIGHT), None)
        # If the built in function cannot distinguish the corner points, the user must manually mark the corners
        if not ret:
            cv.setMouseCallback('image', click_event) 
            corners = manualCorners(img, CHESSBOARDWIDTH, CHESSBOARDHEIGHT)
            ret = True

        objpoints.append(objp)
        try: 
            corners2 = cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        except:
            corners2 = corners
        print(corners2)
        imgpoints.append(corners2)

        # Draw and display the corners
        cv.drawChessboardCorners(img, (CHESSBOARDHEIGHT,CHESSBOARDWIDTH), corners2, ret)
        cv.imshow('Image', img)
        cv.waitKey(5000)

        # close the window 
        cv.destroyAllWindows() 

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, img.shape[::-1], None, None)
    print(ret, mtx, dist, rvecs, tvecs)
