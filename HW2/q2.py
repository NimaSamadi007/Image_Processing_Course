import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl

img = cv2.imread('Greek-ship.jpg', cv2.IMREAD_COLOR)
patch = cv2.imread('patch.png', cv2.IMREAD_COLOR)

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)


sigma = 5
gauss_kernel = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
img_blured = cv2.filter2D(img_gray, -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT)

img_blured = utl.scaleIntensities(img_blured, 'M')

# utl.showImg(img_blured, 0.2, "gray img", False)

patch_bin = np.copy(patch)
thr = 130
patch_bin[patch_bin > thr] = 255
patch_bin[patch_bin <= thr] = 0

# thr = 125
# img_bin[img_bin > thr] = 0
# img_bin[img_bin <= thr] = 255

# utl.showImg(patch_bin, 1, "bin patch", False)

result = cv2.matchTemplate(img_blured, patch_bin, cv2.TM_CCOEFF_NORMED)
cv2.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
result = result * 255

# utl.showRange(result)
# utl.showImg(result.astype(np.uint8), 0.3, "result")


thr1 = 185.01
thr2 = 185.02

locations = np.array(np.nonzero((result >= thr1) & (result <= thr2)))
print(locations.shape)



for i in range(0, locations.shape[1]):
    pt1 = (locations[1, i], locations[0, i])
    pt2 = (locations[1, i]+patch.shape[1], locations[0, i]+patch.shape[0])
    img = cv2.rectangle(img, pt1, pt2, 
                        color=(0, 0, 255), thickness=5, lineType=8, shift=0)
    img_blured = cv2.rectangle(img_blured, pt1, pt2,
                        color=(128, 128, 128), thickness=-1, shift=0)


# utl.showImg(img_blured, 0.3, "black")

patch_bin = cv2.resize(patch_bin, None, fx=0.6, fy=0.5, interpolation=cv2.INTER_LINEAR)
result = cv2.matchTemplate(img_blured, patch_bin, cv2.TM_CCOEFF_NORMED)
cv2.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
result = result * 255

# utl.showImg(patch_bin, 0.3, "second patch", False)
utl.showImg(result.astype(np.uint8), 0.3, "second res", False)


thr1 = 218
thr2 = 218.2

locations = np.array(np.nonzero((result >= thr1) & (result <= thr2)))
print(locations.shape)

for i in range(0, locations.shape[1]):
    pt1 = (locations[1, i], locations[0, i])
    pt2 = (locations[1, i]+patch.shape[1], locations[0, i]+patch.shape[0])
    img = cv2.rectangle(img, pt1, pt2, 
                        color=(0, 0, 255), thickness=5, lineType=8, shift=0)

utl.showImg(img, 0.2, "found")

cv2.imwrite("res15.jpg", img)