import cv2 as cv
import numpy as np
import glob

# Gets the filenames of the images in the Images directory
images = glob.glob('Images/*.jpg')

def click_event(event, x, y, flags, params): 
    if event == cv.EVENT_FLAG_LBUTTON:
        corners.append((x,y))

if __name__=="__main__":   
    # reading the image 
    RESIZEDWIDTH = 400
    # temporarily stores the corners of one image
    corners = []

    # stores the corners of multiple images, i.e. allcorners[i] gives the corners for image number i
    allcorners = []

    for number, filename in enumerate(images):
        print(f"Image {number}: {filename}")
        img = cv.imread(filename)
        cv.namedWindow('image')
        cv.setMouseCallback('image', click_event) 

        while(True):
            # displaying the image 
            cv.imshow('image', img) 

            height, width = img.shape[:2]

            img = cv.resize(img, (RESIZEDWIDTH, int(height*(RESIZEDWIDTH/width))))
            # setting mouse handler for the image 
            # and calling the click_event() function 
            if len(corners) == 4:
                print(f"Corner points: {corners}")
                allcorners.append(corners)
                corners = []
                break

            # wait for a key to be pressed to exit 
            k = cv.waitKey(1) & 0xFF
    
            if k == 27:
                break


        # close the window 
        cv.destroyAllWindows() 


#code door tijn voor automatisch tekenen van hoekpunten
# Import images from the image folder
images = glob.glob('images/IMG_4049.jpg')

# termination criteria
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


for fname in images:
    img = cv.imread(fname)
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    # Find the chess board corners
    ret, corners = cv.findChessboardCorners(gray, (6,9), None)
    # Draw and display the corners
    cv.drawChessboardCorners(img, (6,9), corners, ret)
    cv.imshow('img', img)
    cv.waitKey(500)
cv.destroyAllWindows()
