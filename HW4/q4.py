import numpy as np
import cv2
import utils as utl

img = cv2.imread('birds.jpg', cv2.IMREAD_COLOR)
M, N, _ = img.shape
img_sup = np.copy(img)

img = cv2.GaussianBlur(img, ksize=(3, 3), 
                       sigmaX=1, borderType=cv2.BORDER_CONSTANT)
img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

obj = cv2.ximgproc.createSuperpixelSLIC(img, cv2.ximgproc.SLICO, 25)

#%%
obj.iterate(20)
label_mask = obj.getLabelContourMask(False)
img_sup[label_mask == 255] = 0
#%%
cv2.imwrite('test-q4.jpg', img_sup)


