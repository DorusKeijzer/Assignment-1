import cv2 as cv 

def resizeImage(img, resize_width):
    """Resizes the image so that the image is of the specified width"""
    height, width = img.shape[:2]
    img = cv.resize(img, (resize_width, int(height*(resize_width/width))))
    return img
