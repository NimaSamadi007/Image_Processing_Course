import numpy as np
import cv2
import utils as utl


#---------------------------- FUNCTION ------------------------------#
def drawClosedContour(img, points, thic):
    "Drawes closed contour by connecting lines between two consecutive points"
    img_cop = np.copy(img)
    N = points.shape[0]
    points_cv = np.stack([points[:, 1], points[:, 0]]).T
    for i in range(N):
        cv2.line(img_cop, points_cv[i], points_cv[(i+1) % N], (0, 255, 0), thic)
        # cv2.circle(img_cop, (points_cv[i, 0], points_cv[i, 1]), 5, (0, 255, 0), -1)
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

def addMiddlePoints(points):
    "Adds points in the middle of give points"
    N = points.shape[0]
    new_points = np.zeros((N*2, 2), dtype=np.int64)
    for i in range(2*N):
        if i % 2 == 0:
            new_points[i] = points[i // 2]
        else:
            if i == 2*N - 1:
                new_points[i, :] = (points[(i-1)//2] + points[0]) // 2
            else:
                new_points[i, :] = (points[(i-1)//2] + points[(i+1)//2]) // 2
    return new_points
#---------------------------- MAIN ----------------------------------#
# img = cv2.imread('Untitled.png', cv2.IMREAD_COLOR)
img = cv2.imread('tasbih.jpg', cv2.IMREAD_COLOR)

M_i, N_i, _ = img.shape

# initial contours points
# initial_points = np.array([[309, 221], [334, 198], [363, 176], [398, 169],
#                             [423, 193], [435, 216], [465, 223], [503, 225], 
#                             [526, 244], [515, 267], [505, 295], [510, 321],
#                             [522, 353], [503, 381], [465, 386], [437, 401],
#                             [415, 421], [380, 437], [344, 416], [320, 395],
#                             [286, 380], [254, 373], [240, 341], [238, 310],
#                             [236, 278], [241, 238], [274, 220]])

initial_points = np.array([[233, 261], [270, 242], [315, 230], [368, 223], 
                           [405, 196], [445, 176], [494, 168], [524, 186], 
                           [533, 222], [533, 254], [554, 282], [585, 299], 
                           [630, 304], [676, 295], [730, 286], [783, 286], 
                           [823, 294], [872, 300], [918, 320], [957, 341], 
                           [925, 371], [875, 384], [824, 393], [769, 390], 
                           [719, 392], [676, 394], [645, 403], [622, 426], 
                           [603, 460], [575, 485], [534, 511], [481, 520], 
                           [451, 507], [423, 489], [418, 511], [426, 550], 
                           [420, 582], [390, 604], [347, 605], [309, 583], 
                           [291, 547], [285, 506], [291, 470], [300, 437], 
                           [309, 404], [314, 374], [277, 382], [216, 353], 
                           [203, 322], [210, 289]])

initial_points = addMiddlePoints(initial_points)
initial_points = np.stack([initial_points[:, 1], initial_points[:, 0]]).T

#img_con = drawClosedContour(img, initial_points, 1)
#utl.showImg(img_con, 0.5)

# window size
K = 5
# number of states
M = K**2
# number of points on contour
N = initial_points.shape[0]

points = initial_points[0:N]
# viterbi implementation


# table for viterbi algorithm
table = np.zeros((M, M, N), dtype=np.float64)
# shows predecessor of each state
path = np.zeros((M, M, N), dtype=np.int64)

alpha = 2
gamma = 1

img_grad = utl.calImageGradient(img, 3, 'Scharr')
#utl.showImg(img_grad, 0.5)


all_energy = np.zeros(M, dtype=np.float64)

for step in range(50):
    # average of length between points
    d_avg = calAverageDistance(points)
    table[:, :, :] = 0
    path[:, :, :] = 0
    all_energy[:] = 0
    for i in range(M):
    # i for iterating over each state of the first point
        for j in range(1, N):
        # j for iterating over each coloumn            
            for k in range(M):
            # k for iterating on every row in each coloumn
                state_vector = np.array([k//K, k%K], dtype=np.int64)        
                next_point = points[j] + state_vector - (K-1)//2
                if j == 1:
                    # no need to compare options - last state is the first point
                    state_vector = np.array([i//K, i%K], dtype=np.int64)        
                    current_point = points[0] + state_vector - (K-1)//2                        
                    energy = calEnergy(img_grad, current_point, next_point, d_avg, alpha, gamma)
                    path[i, k, j] = i
                    table[i, k, j] = energy                     
                else:
                    all_energy[:] = 0
                    # consider all M options of previous coloumn
                    for t in range(M):
                        state_vector = np.array([t//K, t%K], dtype=np.int64)        
                        current_point = points[j-1] + state_vector - (K-1)//2                        
                        energy = calEnergy(img_grad, current_point, next_point, d_avg, alpha, gamma)
                        all_energy[t] = energy + table[i, t, j-1]
                    
                    # assign best option
                    best_state = np.argmin(all_energy)
                    path[i, k, j] = best_state
                    table[i, k, j] = all_energy[best_state] 
        
        # add E(v_n, v_0) to the last coloumn
        state_vector = np.array([i//K, i % K], dtype=np.int64)
        next_point = points[0] + state_vector - (K-1)//2
        for q in range(M):
            state_vector = np.array([q//K, q % K], dtype=np.int64)  
            current_point = points[-1] + state_vector - (K-1)//2
            table[i, q, -1] += calEnergy(img_grad, current_point, next_point, d_avg, alpha, gamma)
            
    # finding path and updating points:
    min_index = np.argmin(table[:, :, -1])
    row_table = min_index // M
    best_state_index = min_index % M
    best_state = path[row_table, best_state_index, -1]
    for o in range(N-1, -1, -1):
        state_vector = np.array([best_state//K, best_state % K], dtype=np.int64)  
        points[o] += state_vector - (K-1)//2
        best_state = path[row_table, best_state, o]

    img_con = drawClosedContour(img, points, 1)
    cv2.imwrite('./test/res-{}.jpg'.format(step), img_con)
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
