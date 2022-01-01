import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl


#---------------------------- MAIN -----------------------#
img = cv2.imread('./park.jpg', cv2.IMREAD_COLOR)
img_resized = cv2.resize(img, None, fx=0.2, fy=0.2, interpolation=cv2.INTER_AREA)
#img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2Luv)

M, N,_ = img_resized.shape


window_size = 3

ext_img = np.zeros((M+2*window_size, N+2*window_size, 3), dtype=np.uint8)
ext_img[window_size:-window_size, window_size:-window_size, :] = img_resized

print(np.sum(ext_img[0:6, 0:6, :], axis=(0, 1)) / ((2*window_size+1) ** 2))
               
"""
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(img_resized[:, :, 0], img_resized[:, :, 1], 
           img_resized[:, :, 2], s=1)
plt.show()
"""