import numpy as np
import cv2
import matplotlib.pyplot as plt
import utils as utl

def imgSharpener(img, sigma, alpha):
    gauss_kernel = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
    img_smoothed = cv2.filter2D(img.astype(float), -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT) 
    unsharp_mask = img.astype(float) - img_smoothed.astype(float)
    return img.astype(float) + alpha * unsharp_mask.astype(float)


def calProperIndex(x, y, x_thr, y_thr, M, N):
    xi = x
    yi = y

    if x <= M - x_thr:
        xf = x + x_thr + 1
    else:
        xf = M

    if y <= N - y_thr:
        yf = y + y_thr + 1
    else:
        yf = N
    return (xi, xf), (yi, yf)


img = cv2.imread('Greek-ship.jpg', cv2.IMREAD_COLOR)
patch = cv2.imread('patch.png', cv2.IMREAD_COLOR)

M1, N1, _ = img.shape
M2, N2, _ = patch.shape

# img_sharpened = imgSharpener(img, 10, 1)
# img_sharpened_r = utl.scaleIntensities(img_sharpened)

# patch_sharpened = imgSharpener(patch, 10, 5)
# patch_sharpened_r = utl.scaleIntensities(patch_sharpened)
sigma = 5
gauss_kernel = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
img_blured = cv2.filter2D(img, -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT)

img_blured = utl.scaleIntensities(img_blured, 'M')
# utl.showImg(img, 0.2, 'sharpened', False)


# SSD:
result = cv2.matchTemplate(img_blured, patch, cv2.TM_CCORR_NORMED)
cv2.normalize(result, result, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
result = result * 255

# utl.showImg(result.astype(np.uint8), 0.2, 'result', False)
# utl.showImg(img.astype(np.uint8), 0.2, 'img')


thr = 3.5
locations = np.array(np.nonzero(result >= 255 - thr))
vals = result[result >= 255 - thr]

print(locations.shape)
print(vals.shape)

local_max = []


local_max = locations

# print(local_max[])

fig, axes = plt.subplots(3)
axes[0].plot(local_max[0, :])
axes[1].plot(local_max[1, :])
axes[2].scatter(local_max[0, :], local_max[1, :])
plt.show()

pre_pt1 = (local_max[1, 0], local_max[0, 0])
pre_pt2 = (local_max[1, 0]+patch.shape[1], local_max[0, 0]+patch.shape[0])
img = cv2.rectangle(img, pre_pt1, pre_pt2, 
                    color=(0, 0, 255), thickness=5, lineType=8, shift=0)

for i in range(1, local_max.shape[1]):
    pt1 = (local_max[1, i], local_max[0, i])
    pt2 = (local_max[1, i]+patch.shape[1], local_max[0, i]+patch.shape[0])
    # if np.abs(pt1[0] - pre_pt1[0]) >= 100 and np.abs(pt1[1] - pre_pt1[1]) >= 1: 
    img = cv2.rectangle(img, pt1, pt2, 
                        color=(0, 0, 255), thickness=5, lineType=8, shift=0)
        # pre_pt1 = pt1
        # pre_pt2 = pt2

utl.showImg(img, 0.2, 'result')
