import numpy as np
import matplotlib.pyplot as plt
#import cv2
#import utils as utl

#------------------------- FUNCTIONS ----------------------#
def readData(path):
    "Reads data points from a file in specified path - returns numpy array"
    with open(path, 'r') as f:
        num_data = int(f.readline().split('\n')[0])
        data_points = []
        while 1:
            point = f.readline().split('\n')[0]
            if point:
                point = list(map(float, point.split(' ')))
                data_points.append(point)
            else:
                break
    if len(data_points) != num_data:
        raise ValueError("Some lines are missing, not able to read file!")
    else:
        return np.array(data_points, dtype=np.float64)

#------------------------- MAIN ---------------------------#
# reading data points
data = readData('./Points.txt')

# plotting data points
#plt.scatter(data[:, 0], data[:, 1])
#plt.show()

