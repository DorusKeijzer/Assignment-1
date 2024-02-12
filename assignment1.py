import cv2 as cv
import numpy as np
import glob

# Gets the filenames of the images in the Images directory
images = glob.glob('images/*.jpg')

for filename in images:
    img = cv.imread(filename)

def click_event(event, x, y, flags, params): 
    if event == cv.EVENT_FLAG_LBUTTON:
        corner = x,y
        yield corner
    

