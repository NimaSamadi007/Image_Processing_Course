import numpy as np
import cv2
import utils as utl

img = cv2.imread('birds.jpg', cv2.IMREAD_COLOR)
M, N, _ = img.shape

sigma = 10
gauss_kernel = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
img_smoothed = cv2.filter2D(img.astype(np.float64), -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT) 
unsharp_mask = img.astype(np.float64) - img_smoothed.astype(np.float64) # unsharp mask
alpha = 32
img_sharpend = img.astype(float) + alpha * unsharp_mask.astype(float)
img_sharpend = utl.scaleIntensities(img_sharpend, 'C')


fg_model = np.zeros((1, 65), dtype="float")
bg_model = np.zeros((1, 65), dtype="float")

rect = (0, 1368, N-1, M-1)

mask = np.zeros((M, N), dtype=np.uint8)
mask, bg_model, fg_model = cv2.grabCut(img_sharpend, mask, rect, bg_model,
                                       fg_model, 1, mode=cv2.GC_INIT_WITH_RECT)

out_mask = np.nonzero((mask == cv2.GC_BGD) | (mask == cv2.GC_PR_BGD))

img[out_mask] = 0

cv2.imwrite('test-q4.jpg', img)