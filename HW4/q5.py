import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl


img = cv2.imread('tasbih.jpg', cv2.IMREAD_COLOR)
M, N, _ = img.shape

# initial contours points
initial_points = np.array([[233, 220], [193, 312], [174, 392], [165, 486], [182, 560],
                           [210, 612], [236, 663], [260, 712], [282, 795], [290, 852],
                           [308, 912], [340, 936], [393, 910], [417, 849], [429, 780],
                           [422, 682], [478, 633], [515, 570], [528, 516], [512, 445],
                           [591, 424], [607, 388], [609, 330], [587, 283], [547, 260],
                           [506, 241], [460, 218], [411, 198], [370, 183], [316, 180],
                           [272, 194]])

initial_points_cv = np.stack([initial_points[:, 1], initial_points[:, 0]]).T

cv2.drawContours(img, [[10, 20], [30, 40]], -1, (0, 0, 255), 3)
#utl.showImg(img, 0.5)

