from matplotlib.pyplot import axis
import numpy as np
# import matplotlib.pyplot as plt
import cv2
from numpy.lib.function_base import append

# my processing function - make sure that this file is imported 
import processingFunctions as pf

def changeColor(img, inputColorRange, color):
    M, N, K = img.shape
    colorMask = np.ones( (M, N, K), dtype=img.dtype) * (-1)
    for i in range(M):
        for j in range(N):
            if img[i, j, 0] >= inputColorRange[0] and img[i, j, 0] <= inputColorRange[1]:
                colorMask[i, j, : ] = img[i, j, : ]
                colorMask[i, j, 0] = np.round(color) 
            elif img[i, j, 0] >= inputColorRange[2] and img[i, j, 0] <= inputColorRange[3]:
                colorMask[i, j, : ] = img[i, j, : ]
                colorMask[i, j, 0] = np.round(color) 
    return colorMask


def filterUsingMatrices(img, ksize):
    M, N, K = img.shape
    M_kernel, N_kernel = ksize

    padded_img = np.zeros((M+(M_kernel-1), N+(N_kernel-1), K), dtype=img.dtype) 
    padded_img[(M_kernel-1)//2 : M+(M_kernel-1)//2, (N_kernel-1)//2:N+(N_kernel-1)//2, :] = img

    filtered_img = np.zeros((M, N, K), dtype=np.float64)

    for i in range(M_kernel):
        for j in range(N_kernel):
            filtered_img += padded_img[i:i+M, j:j+N, :]
    return (np.round((1/(M_kernel * N_kernel)) * filtered_img)).astype(img.dtype)


I = cv2.imread('Flowers.jpg', cv2.IMREAD_UNCHANGED)

I_hsv = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)    

pinkRange = (0, 20/2, 260/2, 360/2) # in full hsv, color range: [0 to 5] U [260 to 360]
yellowCol = 60 / 2 # in full hsv

colMask = changeColor(I_hsv, pinkRange, yellowCol) # col mask is in hsv
    
# bluring:
kernelSize = (15, 15)

I_filtered = filterUsingMatrices(I, kernelSize)
I_filtered_hsv = cv2.cvtColor(I_filtered, cv2.COLOR_BGR2HSV)    

M, N, K = colMask.shape

for i in range(M):
    for j in range(N):
        if colMask[i, j, 0] != -1:
            I_filtered_hsv[i, j, 1:] = colMask[i, j, 1:]
            I_filtered_hsv[i, j, 0] = colMask[i, j, 0]


I_changed = cv2.cvtColor(I_filtered_hsv, cv2.COLOR_HSV2BGR)

# pf.showImg(I_changed.astype(np.uint8), (960, 960), 'changed Image')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('res06.jpg', I_changed)