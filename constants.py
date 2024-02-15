import numpy as np
CHESSBOARDWIDTH = 9
CHESSBOARDHEIGHT = 6
RESIZEDWIDTH = 400
SQUARESIZE = 22 # milimeters
NUMBERTOCALIBRATE = 30

AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3]]).reshape(-1,3)
