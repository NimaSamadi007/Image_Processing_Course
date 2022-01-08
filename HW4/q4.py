import numpy as np
import cv2
import utils as utl

img = cv2.imread('birds.jpg', cv2.IMREAD_COLOR)
M, N, _ = img.shape
img_sup = np.copy(img)
img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)
obj = cv2.ximgproc.createSuperpixelSLIC(img, cv2.ximgproc.SLICO, 50)

#%%
obj.iterate(20)
label_mask = obj.getLabelContourMask(False)
img_sup[label_mask == 255] = 0
print(obj.getNumberOfSuperpixels())
#%%
img_cop = np.copy(img)
img_labels = obj.getLabels()
cv2.imwrite('test2-q4.jpg', img_sup)
#%%
for i in range(obj.getNumberOfSuperpixels()):
    print(i)
    region = img_cop[img_labels == i]
    val = np.sum(region, axis=0) / region.shape[0]
    if val[0] <= (60 * 255 / 100) and val[0] >= (40 * 255 / 100):
        img_cop[img_labels == i] = (100 * 255 / 100)

img_cop = cv2.cvtColor(img_cop, cv2.COLOR_Lab2RGB)
cv2.imwrite('test-q4.jpg', img_cop)
    

#%%
import numpy as np
import cv2
import utils as utl

img = cv2.imread('birds.jpg', cv2.IMREAD_COLOR)
img = cv2.GaussianBlur(img, ksize=(7, 7), 
                       sigmaX=2, borderType=cv2.BORDER_CONSTANT)

imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

imgray[(imgray <= 120) & (imgray >= 80)] = 0
# utl.showImg(imgray, 0.2)

img_section = np.copy(imgray[1900:2300, 1400:2500])

contours, hierarchy = cv2.findContours(img_section, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

shift_val = np.array([1400, 1900])
shift_val = shift_val.reshape([1, 1, 2])
contours = list(contours)
for i in range(len(contours)):
    contours[i] += shift_val

cv2.drawContours(img, contours, -1, (0,255,0), 3)

# print(contours)

cv2.imwrite('test3.jpg', img)








