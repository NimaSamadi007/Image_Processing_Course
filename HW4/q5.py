import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl


#---------------------------- FUNCTION ------------------------------#
def drawClosedContour(img, points, thic):
    "Drawes closed contour by connecting lines between two consecutive points"
    img_cop = np.copy(img)
    N = points.shape[0]
    points_cv = np.stack([points[:, 1], points[:, 0]]).T
    for i in range(N):
        cv2.line(img_cop, points_cv[i], points_cv[(i+1) % N], (0, 255, 0), thic)
    return img_cop

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

#---------------------------- MAIN ----------------------------------#
#img = cv2.imread('1.jpeg', cv2.IMREAD_COLOR)
img = cv2.imread('tasbih.jpg', cv2.IMREAD_COLOR)

M_i, N_i, _ = img.shape

# initial contours points
initial_points = np.array([[281, 232], [362, 218], [388, 203], [425, 192], 
                           [451, 178], [489, 173], [521, 184], [534, 208],
                           [532, 231], [528, 254], [539, 280], [556, 300], 
                           [579, 305], [612, 310], [656, 306], [702, 301],
                           [736, 291], [746, 289], [791, 290], [830, 297],
                           [860, 305], [901, 308], [922, 328], [930, 369],
                           [906, 382], [868, 386], [836, 389], [812, 391],
                           [779, 390], [748, 391], [712, 396], [681, 404],
                           [651, 417], [629, 437], [612, 453], [592, 472],
                           [562, 490], [537, 507], [498, 513], [463, 504],
                           [430, 500], [431, 538], [422, 582], [385, 602],
                           [334, 606], [297, 578], [286, 544], [281, 502],
                           [282, 461], [295, 431], [310, 405], [304, 376],
                           [274, 381], [235, 374], [209, 351], [205, 316],
                           [220, 286], [242, 268], [265, 250]])

initial_points = np.stack([initial_points[:, 1], initial_points[:, 0]]).T

#img_con = drawClosedContour(img, initial_points, 1)
#utl.showImg(img_con, 0.5)

# window size
K = 3
# number of states
M = K**2
# number of points on contour
N = initial_points.shape[0]

points = initial_points[0:N]
# viterbi implementation


# table for viterbi algorithm
table = np.zeros((M, N), dtype=np.float64)
# shows predecessor of each state
path = np.zeros((M, N), dtype=np.int64)

alpha = 0.1
gamma = 0.3

img_grad = utl.calImageGradient(img, 3, 'Scharr')

#utl.showImg(utl.scaleIntensities(img_grad), 0.5)


all_energy = np.zeros(M, dtype=np.float64)

for step in range(50):
    # average of length between points
    d_avg = calAverageDistance(points)
    for j in range(1, N):
    # j for iterating over each coloumn
        for k in range(M):
        # k for iterating on every row in each coloumn
            state_vector = np.array([k//K, k%K], dtype=np.int64)        
            next_point = points[j] + state_vector - (K-1)//2

            # next_point[0] %= M_i
            # next_point[1] %= N_i

            all_energy[:] = 0
            # consider all M options of last coloumn
            for t in range(M):
                state_vector = np.array([t//K, t%K], dtype=np.int64)        
                current_point = points[j-1] + state_vector - (K-1)//2

                # current_point[0] %= M_i
                # current_point[1] %= N_i
                
                energy = calEnergy(img_grad, current_point, next_point, d_avg, alpha, gamma)
                all_energy[t] = energy + table[t, j-1]
            # assign best option
            best_state = np.argmin(all_energy)
            path[k, j] = best_state
            table[k, j] = all_energy[best_state] 

# finding path and updating points:
    
    best_state = np.argmin(table[:, -1])
    for i in range(N-1, -1, -1):
        state_vector = np.array([best_state//K, best_state % K], dtype=np.int64)  
        points[i] += state_vector - (K-1)//2
        best_state = path[best_state, i]

    img_con = drawClosedContour(img, points, 1)
    cv2.imwrite('./test/res-{}.jpg'.format(step), img_con)
    #utl.showImg(img_con, 0.5, 'new points')
    print("going to step {}".format(step))
print("Done!")
#%%
print(points[0])
for i in range(M):
    row = i // K
    col = i % K
    print(i)
    print(row-(K-1)//2, col-(K-1)//2)
    print("----------------")