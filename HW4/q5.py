import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl


#---------------------------- FUNCTION ------------------------------#
def drawClosedContour(img, points, thic):
    "Drawes closed contour by connecting lines between two consecutive points"
    img_cop = np.copy(img)
    N = points.shape[0]
    for i in range(N):
        cv2.line(img_cop, points[i], points[(i+1) % N], (0, 255, 0), thic)
    return img_cop

#---------------------------- MAIN ----------------------------------#
#img = cv2.imread('1.jpeg', cv2.IMREAD_COLOR)
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

#initial_points_cv = np.stack([initial_points[:, 1], initial_points[:, 0]]).T
#img_con = drawClosedContour(img, initial_points_cv, 1)

#utl.showImg(img_con, 0.5)

#%%
def calAverageDistance(points):
    "Calculates average of d"
    N = points.shape[0]
    d_avg = 0
    for i in range(N):
        d = np.sum((points - points[i])**2, axis=1, dtype=np.float64)
        d_avg += np.sum(np.sqrt(d))
    return (d_avg / (N**2))

def calEnergy(img_grad, current_point, next_point, d_avg, alpha, gamma):
    "Calculates energy of E(v_i, v_i+1)"
    gradient = img_grad[current_point[0], current_point[1]]
    distance = np.sum((current_point - next_point)**2, dtype=np.float64)
    energy = (-gamma) * (gradient**2) + alpha * ((distance-d_avg)**2)
    return energy

# window size
K = 3
# number of states
M = K**2
# number of points on contour
N = 5

points = initial_points[0:N]
# viterbi implementation


# table for viterbi algorithm
table = np.zeros((K, N), dtype=np.float64)

alpha = 1
gamma = 5

img_grad = utl.calImageGradient(img, 5)

# average of length between points
d_avg = calAverageDistance(points)

# calculating energy
#energy = calEnergy(points, img_grad, current_point, next_point, d_avg, alpha, gamma)

# filling table:
for i in range(M):
    # i for considering each state for v0
    for j in range(1, N):
    # j for iterating over each coloumn
        for k in range(M):
        # k for iterating on every row in each coloumn
            if j == 1:
            # if we are in second coloumn we know the last coloumn which
            # is i
            #current_point = 
            #table[k, j] = calEnergy(img_grad, points[], next_point, d_avg, alpha, gamma)




