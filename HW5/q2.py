import numpy as np
import cv2 as cv
import utils as utl

source_img = cv.imread('res05.jpg', cv.IMREAD_COLOR)
target_img = cv.imread('res06.jpg', cv.IMREAD_COLOR)
# mask_col = cv.imread('res06-mask.jpg', cv.IMREAD_GRAYSCALE)

Ms, Ns, _ = source_img.shape
Mt, Nt, _ = target_img.shape



# source (x, y) coordinates
x_s, y_s = 39, 307
# target (x, y) coordinates
x_t, y_t = 475, 1432
# height and width of the region
height, width = 249, 250


# utl.showImg(source_img[x_s:x_s+height, y_s:y_s+width, :], 1, 'source', False)
# utl.showImg(target_img[x_t:x_t+height, y_s:y_s+width, :], 1)

M, N = height, width

# Possion blending for each channel
source_region = source_img[x_s:x_s+height, y_s:y_s+width, 0]
target_region = target_img[x_t:x_t+height, y_s:y_s+width, 0]

# source image laplacian calculation:
laplacian_kernel = np.array([[0, -1, 0],
                             [-1, 4, -1],
                             [0, -1, 0]], np.float64)
source_laplacian = cv.filter2D(source_region.astype(np.float64), -1, laplacian_kernel, cv.BORDER_CONSTANT)
utl.showRange(source_laplacian, 'Q')

# laplacian_repr = utl.scaleIntensities(source_laplacian)
# utl.showImg(laplacian_repr, 1)

# forming linear equations:
D = np.zeros((M-2, M-2), np.float64)
np.fill_diagonal(D, -4)
np.fill_diagonal(D[:-1, 1:], 1)
np.fill_diagonal(D[1:, :-1], 1)

A = np.zeros( ((M-2)*(N-2), (M-2)*(N-2)), np.float64)




