import numpy as np
CHESSBOARDWIDTH = 9
CHESSBOARDHEIGHT = 6
RESIZEDWIDTH = 400
SQUARESIZE = 22 # milimeters
NUMBERTOCALIBRATE = 30
mean_error = 0

AXIS = np.float32([[3,0,0], [0,3,0], [0,0,-3], [1,0,0],[0,1,0],[0,0,-1],[0,0,0 ],  [1,1,0], [0,0,0],[0,0,0],[1,1,0], [1,1,0],[0,0,-1],[1,1,-1], [1,1,-1],
                       [1,0,-1],[0,1,-1],[0,1,-1],[0,0,-1],[1,1,-1],[0,1,0],[1,0,0], [0,1,0], [1,0,0],[1,0,-1], [0,1,-1], [1,0,-1]])
