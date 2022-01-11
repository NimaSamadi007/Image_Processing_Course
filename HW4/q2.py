import numpy as np
import cv2

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
img = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
M, N,_ = img.shape

equivalent_color = (-1)*np.ones((M, N), dtype=np.float64)

radius = 15
color_thr = 15
# better to change this to 0.5 to increase the speed
diff_thr = 0.3
# diff_thr = 0.3
# distinct color found
distinct_color = []

for i in range(M):
    print("At row {}".format(i))
    for j in range(N):
        if equivalent_color[i, j] == -1:
        # this pixel hasn't been filled previously
            current_point = img[i, j, :]
            all_x_indices = []
            all_y_indices = []
            while 1:
                distance = (img.astype(np.float64) - current_point.astype(np.float64))**2
                # points that are in a circle
                indices = np.nonzero(np.sum(distance, axis=2)**(0.5) <= radius)
                                
                near_points = img[indices]
                # new center point
                new_point = np.sum(near_points, axis=0) / (near_points.shape[0])
                diff = np.sum((new_point.astype(np.float64) - current_point.astype(np.float64))**2)
                
                # just add pixels that their labels haven't assigned yet
                indices = np.nonzero((np.sum(distance, axis=2)**(0.5) <= radius) & (equivalent_color == -1))              
                if indices[0].shape[0]:
                    all_x_indices.extend(list(indices[0]))
                    all_y_indices.extend(list(indices[1]))

                if diff ** (0.5) <= diff_thr:
                    # there might be some replicants but after when assigning the replicants will
                    # not have any effect
                    all_indices = np.array([all_x_indices, all_y_indices]).T
                    index = isExist(distinct_color, new_point, color_thr)
                    if index >= 0:
                        equivalent_color[i, j] = index
                        equivalent_color[all_indices[:, 0], all_indices[:, 1]] = index
                    else:
                        equivalent_color[i, j] = len(distinct_color)
                        equivalent_color[all_indices[:, 0], all_indices[:, 1]] = len(distinct_color)
                        distinct_color.append(new_point)    
                    break
                else:
                    current_point = np.copy(new_point)

print("Replacing average color ...")
for i in range(len(distinct_color)):
    indices = np.nonzero(equivalent_color == i)
    if len(indices[0]):
        img[indices] = np.sum(img[indices], axis=0) / len(indices[0])
    else:
        print("None was found!")

img = cv2.resize(img, None, fx=5, fy=5, interpolation=cv2.INTER_LINEAR)
cv2.imwrite('./res05.jpg', img.astype(np.uint8))
print("Done!")            
