import cv2 as cv
import numpy as np
import glob

# Gets the filenames of the images in the Images directory
images = glob.glob('images/*.jpg')

for filename in images:
    img = cv.imread(filename)

corners = []

def click_event(event, x, y, flags, params): 
    if event == cv.EVENT_FLAG_LBUTTON:
        corners.append((x,y))

if __name__=="__main__":   
    # reading the image 
    img = np.zeros((500, 500, 3), np.uint8)
    cv.namedWindow('image')
    cv.setMouseCallback('image', click_event) 

    while(True):
        # displaying the image 
        cv.imshow('image', img) 
    
        # setting mouse handler for the image 
        # and calling the click_event() function 
    
        if len(corners) == 4:
            print(corners)
            break
        # wait for a key to be pressed to exit 
        k = cv.waitKey(1) & 0xFF
 
        if k == 27:
            break


    # close the window 
    cv.destroyAllWindows() 
