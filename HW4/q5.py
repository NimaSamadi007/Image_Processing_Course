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
    return (d_avg / ((N-1)**2))

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
img = cv2.imread('tasbih.jpg', cv2.IMREAD_COLOR)
M_i, N_i, _ = img.shape

# generating points on an circle around the tasbih
theta = np.linspace(0, 2*np.pi, 80, endpoint=False)
initial_points = np.stack([390+250*np.cos(theta), 420+250*np.sin(theta)]).T
initial_points = initial_points.astype(np.int32)
img_grad = utl.calImageGradient(img, 2, 'Scharr')

# window size
K = 3
# number of states
M = K**2
# number of points on contour
N = initial_points.shape[0]

points = initial_points
# viterbi implementation

# table for viterbi algorithm
table = np.zeros((M, M, N), dtype=np.float64)
# shows predecessor of each state
path = np.zeros((M, M, N), dtype=np.int64)

alpha = 5
gamma = 1
coeff = 0.25
grad_avg_level = (np.sum(img_grad) / img_grad.size)
gradient_thr = 0.4 * grad_avg_level


all_energy = np.zeros(M, dtype=np.float64)

# at most it will take 500 step to reach the boundries
max_step = 500
for step in range(max_step):
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
    previous_points = np.copy(points)
    for o in range(N-1, -1, -1):
        state_vector = np.array([best_state//K, best_state % K], dtype=np.int64)  
        points[o] += state_vector - (K-1)//2
        best_state = path[row_table, best_state, o]
    img_con = drawClosedContour(img, points, 2)
    cv2.imwrite('./test/res-{}.jpg'.format(step), img_con)
    print("At iteration {}".format(step))
    
    contour_grad_level = np.sum(img_grad[points[:, 0], points[:, 1]]) / N
    
    # check termination condition
    if contour_grad_level >= 12 * grad_avg_level:
        print("Contour is reached the boundries ...")
        break
    
    # if contour is not moving, push it
    if np.sum(np.abs(previous_points - points)) <= 5:
        middle_point = np.sum(points, axis=0) / points.shape[0]
        distance_with_middle = (points - middle_point) * coeff
        for i in range(points.shape[0]):
            if img_grad[points[i, 0], points[i, 1]] <= gradient_thr:
                points[i] -= distance_with_middle[i].astype(np.int64)
    print("----------------------------")

cv2.imwrite('res11.jpg', img_con)
print("Done!")




