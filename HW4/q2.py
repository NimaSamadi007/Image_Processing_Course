import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl

#---------------------------- FUNC -----------------------#
def isExist(arr, val, thr):
    "checkes if val exists in arr with a threshold"
    for i in range(len(arr)):
        if (np.sum((val - arr[i])**2))**(0.5) <= thr:
            return i
    # not found
    return -1
#---------------------------- MAIN -----------------------#
img = cv2.imread('./park.jpg', cv2.IMREAD_COLOR)
img_resized = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)

#img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Luv)

M, N,_ = img_resized.shape

equivalent_color = (-1)*np.ones((M, N), dtype=np.float64)

radius = 15
color_thr = radius ** (0.5)

# distinct color found
distinct_color = []

for i in range(M):
    print("At row {}".format(i))
    for j in range(N):
        if equivalent_color[i, j] != -1:
            # this pixel has been filled previously
            continue
        current_point = img_resized[i, j, :]

        all_x_indices = []
        all_y_indices = []
        #print(j)
        while 1:        
            distance = (img_resized.astype(np.float64) - current_point.astype(np.float64))**2
            # points that are in a circle
            indices = np.nonzero(np.sum(distance, axis=2)**(0.5) <= radius)
            all_x_indices.extend(list(indices[0]))
            all_y_indices.extend(list(indices[1]))

            near_points = img_resized[indices]
            new_point = np.sum(near_points, axis=0) / (near_points.shape[0])
            
            diff = np.sum((new_point.astype(np.float64) - current_point.astype(np.float64))**2)
            #print(diff)
            if diff ** (0.5) <= 0.3:
                # check if it is a distinct color or not
                all_indices = np.array([all_x_indices, all_y_indices]).T
                all_indices = list(set(map(tuple, all_indices)))
                index = isExist(distinct_color, new_point, color_thr)
                if index >= 0:
                    equivalent_color[i, j] = index
                    for k in range(len(all_indices)):
                        equivalent_color[all_indices[k][0], all_indices[k][1]] = index
                else:    
                    equivalent_color[i, j] = len(distinct_color)
                    for k in range(len(all_indices)):
                        equivalent_color[all_indices[k][0], all_indices[k][1]] = len(distinct_color)
                    distinct_color.append(new_point)
                break
            else:
                current_point = np.copy(new_point)
        #print("----------------")


print(len(distinct_color))
#img = cv2.imread('./park.jpg', cv2.IMREAD_COLOR)
#img_resized = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
#img_resized = img_resized.astype(np.float64)
# assign average color
for i in range(len(distinct_color)):
#for i in range(1):
    indices = np.nonzero(equivalent_color == i)
    #print(img_resized[indices])
    #print("For {}".format(i))
    #print(indices)
    #print(len(indices[0]))
    #print(len(indices[0]))
    #print(np.sum(img_resized[indices], axis=0))
    #print(len(indices[0]))
    img_resized[indices] = np.sum(img_resized[indices], axis=0) / len(indices[0])
    #img_resized[indices] = (i/len(distinct_color))*255
    #print(img_resized[indices])
    #print("-----------------------------")
    
cv2.imwrite('./test.jpg', img_resized.astype(np.uint8))

#%%

for i in range(M):
    for j in range(N):
        if isExist(distinct_color, equivalent_color[i, j, :], color_thr) == -1:
            distinct_color.append(equivalent_color[i, j, :])
            


#%%
#plt.show()
cv2.imwrite('./test.jpg', equivalent_color.astype(np.uint8))
 
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

ax.scatter(equivalent_color[:, :, 0], equivalent_color[:, :, 1], 
           equivalent_color[:, :, 2], s=10, c='k')

#%%

x = []
y = []

x.extend(list(indices[0]))
y.extend(list(indices[1]))

x.extend(list(indices[0]))
y.extend(list(indices[1]))

t = np.array([x, y])
t = t.T

t_set = list(set(map(tuple, t)))
