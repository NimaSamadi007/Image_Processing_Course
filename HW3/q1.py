import numpy as np
import cv2
import utils as utl

img1 = cv2.imread('./im01.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('./im02.jpg', cv2.IMREAD_COLOR)

img1_edges = cv2.Canny(img1, 100, 250)
img2_edges = cv2.Canny(img2, 100, 250)

# threshold for converting edges to binary
thr = 10
img1_edges[img1_edges > thr] = 255
img1_edges[img1_edges <= thr] = 0
img2_edges[img2_edges > thr] = 255
img2_edges[img2_edges <= thr] = 0

cv2.imwrite('res01.jpg', img1_edges)
cv2.imwrite('res02.jpg', img2_edges)

