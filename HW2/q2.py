import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl
import time

## ------------------- FUNCTIONS ----------------------- ##
def myMatchTempl(img, patch):
    "matches the patch with an img with zero mean cross correlation merit"
    M, N = patch.shape
    
    patch_norm = patch - np.sum(patch) / (M * N)
    # correlation using filter2D
    corr = cv2.filter2D(img, cv2.CV_64F, patch_norm, borderType=cv2.BORDER_CONSTANT)
    # concatenate image from each side by half size
    result = corr[(M-1)//2:-(M-1)//2, (N-1)//2:-(N-1)//2]

    # result2 = 

    # normalize result using NORM_MINMAX
    cv2.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    result *= 255
    return result
## ----------------- MAIN ----------------------------- ##

img = cv2.imread('Greek-ship.jpg', cv2.IMREAD_COLOR)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

patch = cv2.imread('patch.png', cv2.IMREAD_COLOR)
patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

# image bluring to enhance detection
sigma = 5
gauss_kernel = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
img_blured = cv2.filter2D(img_gray, -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT)
img_blured = utl.scaleIntensities(img_blured, 'M')

# making patch binary with a threshold
patch_bin = np.copy(patch)
thr = 130
patch_bin[patch_bin > thr] = 255
patch_bin[patch_bin <= thr] = 0

# since masts on the ship are small, I resize patch
patch_bin = cv2.resize(patch_bin, None, fx=0.5, fy=0.4, interpolation=cv2.INTER_LINEAR)
# perform matching:
result = myMatchTempl(img_blured, patch_bin)

# thresholding image:
thr1 = 227
thr2 = 227.01
thr3 = 200
thr4 = 200.08
locations = np.array(np.nonzero( ((result >= thr1) & (result <= thr2)) | ((result >= thr3) & (result <= thr4)) ))

for i in range(0, locations.shape[1]):
    pt1 = (locations[1, i], locations[0, i])
    pt2 = (locations[1, i]+patch.shape[1], locations[0, i]+patch.shape[0])
    img = cv2.rectangle(img, pt1, pt2, 
                        color=(0, 0, 255), thickness=5, lineType=8, shift=0)

cv2.imwrite("res15.jpg", img)
