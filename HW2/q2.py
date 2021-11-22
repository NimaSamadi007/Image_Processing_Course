import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl
import time

## ------------------- FUNCTIONS ----------------------- ##
def myMatchTempl(img, patch):
# My implementation of cv2.matchTemplate function
# using cv2.TM_CCOEFF_NORMED.
# img and patch must be grayscale (one channel). 
# It is suggested to convert patch to binary img

    # shapes:
    M_p, N_p = patch.shape
    M_i, N_i = img.shape
    # result matrix
    result = np.zeros((M_i-M_p+1, N_i-N_p+1), dtype=np.float64)
    # normalize patch by subtracting its average:
    patch_norm = patch.astype(np.float64) - np.sum(patch) / (M_p*N_p)

    #performing convolution:
    for i in range(result.shape[0]):
        print("at index: {}".format(i))
        for j in range(result.shape[1]):
            img_norm = img[i:i+M_p, j:j+N_p].astype(np.float64) - np.sum(img[i:i+M_p, j:j+N_p]) / (M_p*N_p)
            result[i, j] = np.sum(img_norm * patch_norm) / np.sqrt(np.sum(patch_norm ** 2) * np.sum(img_norm ** 2)) 

    return result

## ----------------- MAIN ----------------------------- ##

img = cv2.imread('Greek-ship.jpg', cv2.IMREAD_COLOR)
img = cv2.resize(img, None, fx=0.8, fy=0.8)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

patch = cv2.imread('patch.png', cv2.IMREAD_COLOR)
patch = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

patch = cv2.resize(patch, None, fx=0.5, fy=0.5)

sigma = 5
gauss_kernel = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
img_blured = cv2.filter2D(img_gray, -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT)
img_blured = utl.scaleIntensities(img_blured, 'M')

patch_bin = np.copy(patch)
thr = 130
patch_bin[patch_bin > thr] = 255
patch_bin[patch_bin <= thr] = 0

# time_origin = time.process_time()
# res = myMatchTempl(img_gray, patch)
# elappsed_time = time.process_time() - time_origin

# print("It took {} s !".format(elappsed_time))

# print(img_gray.shape)
# print(patch.shape)
# print(res.shape)

# cv2.normalize(res, res, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
# res *= 255

result = cv2.matchTemplate(img_blured, patch_bin, cv2.TM_CCOEFF_NORMED)
cv2.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
result = result * 255

# utl.showImg(res.astype(np.uint8), 1, 'my res', False)
# utl.showImg(result.astype(np.uint8), 1, 'cv res', False)
# utl.showImg(patch_bin, 1, 'bin patch')

# print(np.sqrt(np.sum((res - result)**2)))



print(img.shape)
print(img_blured.shape)
print(patch.shape)
print(result.shape)

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


patch_bin = cv2.resize(patch_bin, None, fx=0.6, fy=0.5, interpolation=cv2.INTER_LINEAR)
result = cv2.matchTemplate(img_blured, patch_bin, cv2.TM_CCOEFF_NORMED)
cv2.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
result = result * 255

thr1 = 218
thr2 = 218.2

locations = np.array(np.nonzero((result >= thr1) & (result <= thr2)))
print(locations.shape)

for i in range(0, locations.shape[1]):
    pt1 = (locations[1, i], locations[0, i])
    pt2 = (locations[1, i]+patch.shape[1], locations[0, i]+patch.shape[0])
    img = cv2.rectangle(img, pt1, pt2, 
                        color=(0, 0, 255), thickness=5, lineType=8, shift=0)

# cv2.imwrite("res15.jpg", img)
utl.showImg(img, 0.2, 'found')