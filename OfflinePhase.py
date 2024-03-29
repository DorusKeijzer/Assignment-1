import numpy as np
import cv2 as cv
import glob
from constants import * 
import utils
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Stores the corner points.
objp = np.zeros((CHESSBOARDWIDTH*CHESSBOARDHEIGHT,3), np.float32)
objp[:,:2] = np.mgrid[0:CHESSBOARDWIDTH,0:CHESSBOARDHEIGHT].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# temporarily stores the corners of one image resulting from the click event
clickcorners = []
rejectLowQuality = True
def click_event(event, x, y, flags, param): 
    if event == cv.EVENT_FLAG_LBUTTON:
        cv.circle(img, (x,y), 3, 400, -1)
        clickcorners.append([x,y])
        cv.imshow('Image', img) 

def manualCorners(img, chessboardwidth, chessboardheight) -> np.array:
    """ALlows the user to specify the corners. 
    These corners should be given in the same order as the program does.
    returns an array of the correct size with the corner points
    
    Order of clicks is important. Correct order for an image in portrait mode: 
    Top right, bottom right, top left, bottom left
    """
    
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

def prepareImage(filename):
    """Reads the image and prepares it for being processed"""
    img = cv.imread(filename)
    grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    return img, grey

def interpolate(corners, chessboardwidth, chessboardheight):
    """Expects:
       the pixel coordinates of the 4 corners of a chessboard in an np array of size 4
       the proportions of the chessboard, 

       returns where the internal crossings are as a np array of size (chessboardwidth x chessboard height) """

    # the coordinates of the outer corners of the chessboard in chessboard coordinates
    # these will be used to solve for the transformation matrix
    # manual coordinates need to be specified in the same order (Top right, bottom right, top left, bottom left)
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

def calculateError(output, objpoints, imgpoints):
    """"Calculates the error when reprojecting the object points using the matrix obtained by calibrateCamera()
    returns an array of the error for each image"""
    errors = []
    ret, mtx, dist, rvecs, tvecs = output
    for i in range(len(objpoints)):
        imgpoints2, _ = cv.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv.norm(imgpoints[i], imgpoints2, cv.NORM_L2)/len(imgpoints2)
        errors.append(error)
    return np.array(errors)

if __name__=="__main__":   
    # reading the image 
    # Gets the filenames of the images in the Images directory

    with open("results/results.txt", "w") as results:
        for run in [ "3", "2", "1"]:
            print(f"Run {run}:")
            
            # images for every run are stored in seperate folders
            images = glob.glob(f'Images/run{run}/*.jpg')

            for number, filename in enumerate(images):
                print(f"Image {number+1}: {filename}")
                image = cv.imread(filename)
                
                # enhancing the image by sharpening it
                # Create the sharpening kernel 
                kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]) 

                # Sharpen the image 
                img = cv.filter2D(image, -1, kernel) 

                
                grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

                # Find the chess board corners
                ret, corners = cv.findChessboardCorners(grey, (CHESSBOARDWIDTH,CHESSBOARDHEIGHT), None)

                
                if ret: # the built in function succeeds
                    corners = cv.cornerSubPix(grey, corners, (11,11), (-1,-1), criteria)
                else: # the built in function fails
                    print(f"Please manually provide the corners for {filename} in this order: ")
                    print("Bottom right, bottom left, top right, top left")
                    cv.imshow('Image', img)
                    cv.setMouseCallback('Image', click_event) 
                    corners = manualCorners(img, CHESSBOARDWIDTH, CHESSBOARDHEIGHT)
                    print(f"Thank you")
                    ret = True

                objpoints.append(objp)
                imgpoints.append(corners)

                # Draw and display the corners
                cv.drawChessboardCorners(img, (CHESSBOARDWIDTH, CHESSBOARDHEIGHT), corners, ret)
                cv.imshow('Image', img)
                cv.waitKey(250)

                # close the window 
                cv.destroyAllWindows() 
            #calculate camera matrix and extrinsics     
            calibrationoutput = cv.calibrateCamera(objpoints, imgpoints, grey.shape[::-1], None, None)
            ret, mtx, dist, rvecs, tvecs = calibrationoutput
            #calculate error
            errors = calculateError(calibrationoutput, objpoints, imgpoints)
            print(f"Mean error: {sum(errors)/len(errors)}")

            if rejectLowQuality:
                # Calculate the interquartile range (IQR)
                Q1 = np.percentile(errors, 25)
                Q3 = np.percentile(errors, 75)
                IQR = Q3 - Q1

                # using the 1.5 + IQR rule
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR

                # Reject outliers
                filtered_data = errors[(errors >= lower_bound) & (errors <= upper_bound)]
                print(f"Rejected {len(errors)-len(filtered_data)} images")
                print(f"Old error: {np.mean(errors)} ")
                print(f"New error: {np.mean(filtered_data)} ")

            results.write(f"Run {run}:\n")
            results.write(f"Camera matrix:\n{mtx}\nDistance coefficients:\n {dist}\n")
            results.write(f"total error: {errors/len(errors)}")
            results.write("\n")

            print(f"Saving matrix to Calibration_run{run}.npz")
            # save for future use in online phase
            np.savez(f'results/Calibration_run{run}.npz', mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)
            
            
            
            #plot extrinsics in 3d space
            # Define camera parameters
            camera_matrix = mtx
            dist_coeffs = np.zeros((4,1))  # Assuming no lens distortion

            # Create a matplotlib 3D plot
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')

            # Arrow Length
            arrow_length = 1  # Adjust this value to change the length of the arrows

            # Box Size
            size = 1  # Adjust this value to change the size of the box

            # Plot the camera positions
            for rvec, tvec in zip(rvecs, tvecs):
                # Project 3D axis points to image plane
                img_points, _ = cv.projectPoints(np.float32([[0,0,0], [1,0,0], [0,1,0], [0,0,1]]), rvec, tvec, camera_matrix, dist_coeffs)
                
                # Extract the image points for plotting
                img_points = np.int32(img_points).reshape(-1,2)
                
                # Plot the lines between the points to represent the camera axes
                for i in range(3):
                    ax.plot([img_points[0][0], img_points[i+1][0]], [img_points[0][1], img_points[i+1][1]], zs=0)

                # Plot the camera as a box
                ax.scatter(tvec[0], tvec[1], tvec[2], color='b', marker='o')  # Camera position
                R, _ = cv.Rodrigues(rvec)
                for i in range(3):
                    ax.quiver(tvec[0], tvec[1], tvec[2], R[0, i]*arrow_length, R[1, i]*arrow_length, R[2, i]*arrow_length, color='r', length=size)

            # Set labels and limits
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_xlim([0, 10])
            ax.set_ylim([10, 0])
            ax.set_zlim([0, 15])

            # Show the plot
            plt.show()
