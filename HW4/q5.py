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
        # cv2.line(img_cop, points_cv[i], points_cv[(i+1) % N], (0, 255, 0), thic)
        cv2.circle(img_cop, (points_cv[i, 0], points_cv[i, 1]), 5, (0, 255, 0), -1)
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

def randomShifter(points):
    "Shifts a sequence of points to the left"
    N = points.shape[0]
    index = np.random.randint(low=1, high=N)
    # print(index)

    shifted_points = np.zeros(points.shape, dtype=points.dtype)
    shifted_points[0:N-index] = points[index:]
    shifted_points[-index:] = points[:index]
    
    return shifted_points
#---------------------------- MAIN ----------------------------------#
# img = cv2.imread('Untitled.png', cv2.IMREAD_COLOR)
img = cv2.imread('tasbih.jpg', cv2.IMREAD_COLOR)

M_i, N_i, _ = img.shape

# theta = np.linspace(0, 2*np.pi, 50)
# v_x = np.int64(400 + 250*np.cos(theta))
# v_y = np.int64(450 + 300*np.sin(theta))

initial_points = np.array([[244, 210], [294, 199], [385, 181], [376, 163], 
                            [436, 140], [503, 135], [555, 163], [581, 201], 
                            [610, 243], [639, 281], [686, 294], [733, 318], 
                            [737, 370], [699, 399], [674, 428], [646, 469], 
                            [616, 507], [581, 540], [536, 566], [496, 540], 
                            [457, 540], [451, 588], [417, 614], [369, 635], 
                            [302, 636], [271, 590], [248, 525], [252, 469], 
                            [250, 418], [199, 399], [163, 354], [166, 303], 
                            [186, 257], [211, 225]])

# initial_points = np.stack([v_x, v_y]).T

# initial_points = addMiddlePoints(initial_points)

initial_points = np.stack([initial_points[:, 1], initial_points[:, 0]]).T

# img_con = drawClosedContour(img, initial_points, 1)
# utl.showImg(img_con, 0.5)

# window size
K = 3
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


alpha = 0.1
gamma = 2
# sigma = 3

img_grad = utl.calImageGradient(img, 3, 'Scharr')

# img_grad2 = utl.calImageGradient(img, 5, 'Scharr')

# utl.showImg(utl.scaleIntensities(img_grad1, 'C'), 0.5, 't', False)
# utl.showImg(utl.scaleIntensities(img_grad, 'C'), 0.5)
# utl.showRange(img_grad)

all_energy = np.zeros(M, dtype=np.float64)

max_step = 150
for step in range(max_step):
    # points = randomShifter(points)
    # average of length between points
    # d_avg = calAverageDistance(points)
    d_avg = 0
    # print(d_avg)
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
    # previous_points = np.copy(points)
    for o in range(N-1, -1, -1):
        state_vector = np.array([best_state//K, best_state % K], dtype=np.int64)  
        points[o] += state_vector - (K-1)//2
        best_state = path[row_table, best_state, o]
    img_con = drawClosedContour(img, points, 1)
    cv2.imwrite('./test/res-{}.jpg'.format(step), img_con)
    print("going to step {}".format(step))
    # print(np.sum(np.abs(previous_points - points)))
    print("----------------------------")
    # if not np.sum(np.abs(previous_points - points)) or step > max_step:
    #     break
    # else:
    #     step += 1
print("Done!")

#%%
x = points[0:10]

y = randomShifter(x)

img_con = drawClosedContour(img, x, 1)
img_con2 = drawClosedContour(img, y, 1)
utl.showImg(img_con, 0.5, 't', False)
utl.showImg(img_con2, 0.5)



