import cv2 as cv
import numpy as np
import glob

# Gets the filenames of the images in the Images directory
images = glob.glob('Images/*.jpg')


corners = []

def click_event(event, x, y, flags, params): 
    if event == cv.EVENT_FLAG_LBUTTON:
        corners.append((x,y))

if __name__=="__main__":   
    # reading the image 
    RESIZEDWIDTH = 400
    for filename in images:
        print(f"Image: {filename}")
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
        
            # wait for a key to be pressed to exit 
            k = cv.waitKey(1) & 0xFF
    
            if k == 27:
                break


        # close the window 
        cv.destroyAllWindows() 
