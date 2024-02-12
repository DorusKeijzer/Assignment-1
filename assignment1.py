import cv2 as cv
import numpy as np
import glob

RESIZEDWIDTH = 400
SQUARESIZE = 22 # milimeters



# Gets the filenames of the images in the Images directory
images = glob.glob('Images/*.jpg')

def click_event(event, x, y, flags, params): 
    if event == cv.EVENT_FLAG_LBUTTON:
        clickcorners.append((x,y))
        print(clickcorners)
    

if __name__=="__main__":   
    # reading the image 
    # temporarily stores the corners of one image
    clickcorners = []

    # stores the corners of multiple images, i.e. allcorners[i] gives the corners for image number i
    allcorners = []

    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 
    objp = np.zeros((6*9,3), np.float32)
    objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.


    for number, filename in enumerate(images):
        print(f"Image {number}: {filename}")
        img = cv.imread(filename)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        cv.namedWindow('image')
        cv.setMouseCallback('image', click_event) 

        # resizes the images to the specified height so it fits on the screen
        height, width = img.shape[:2]
        img = cv.resize(img, (RESIZEDWIDTH, int(height*(RESIZEDWIDTH/width))))

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(img, (6,9), None)
        if not np.any(corners):
            while(True):
                cv.imshow('image', img) 
                cv.waitKey(0)
                cv.destroyAllWindows()
                if len(clickcorners) == 4:
                    corners = clickcorners
                    continue

        objpoints.append(objp)
        corners2 = cv.cornerSubPix(img, corners, (11,11), (-1,-1), criteria)
        imgpoints.append(corners2)
        print(corners2)
        # Draw and display the corners

        # close the window 
        cv.destroyAllWindows() 



clickcorners = []
while(True):
    # displaying the image 
    cv.imshow('image', img) 

    # setting mouse handler for the image 
    # and calling the click_event() function 
    if len(clickcorners) == 4:
        print(f"Corner points: {clickcorners}")
        cv.destroyAllWindows() 
