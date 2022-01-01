import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl


#---------------------------- MAIN -----------------------#
img = cv2.imread('./park.jpg', cv2.IMREAD_COLOR)
img_resized = cv2.resize(img, None, fx=0.1, fy=0.1, interpolation=cv2.INTER_AREA)
#img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Luv)

M, N,_ = img_resized.shape
equivalent_color = np.zeros(img_resized.shape, dtype=np.uint8)


radius = 4

for i in range(M):
    print("At row {}".format(i))
    for j in range(N):
        current_point = img_resized[i, j, :]    
        #print(j)
        while 1:
        #    fig = plt.figure()
        #    ax = fig.add_subplot(projection='3d')
        
        #    ax.scatter(current_point[0], current_point[1], 
        #           current_point[2], s=10, c='k')
        
            distance = (img_resized.astype(np.float64) - current_point.astype(np.float64))**2
            # points that are in a circle
            near_points = img_resized[np.nonzero(np.sum(distance, axis=2) <= radius)]
            new_point = np.sum(near_points, axis=0) / (near_points.shape[0])
            
            diff = np.sum((new_point.astype(np.float64) - current_point.astype(np.float64))**2)
            if diff ** (0.5) <= 2:
                # terminate new point finding
                equivalent_color[i, j, :] = new_point.astype(np.uint8)
                break        
        
        #    ax.scatter(near_points[:, 0], near_points[:, 1], 
        #               near_points[:, 2], s=10)
        #    ax.scatter(current_point[0], current_point[1], 
        #               current_point[2], s=10, c='r')
    
    
#plt.show()

    