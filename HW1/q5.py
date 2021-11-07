import numpy as np
import time
import cv2
# my processing function - make sure that this file is imported 
import processingFunctions as pf


# ----------------------- FUNCTIONS ------------------------------------

# Applyes the (3, 3) box kernel on input image
def applyBoxFilter(img):
    M, N, K = img.shape

    # adds two row and column for kernel: (3, 3)
    padded_img = np.zeros((M+2, N+2, K), dtype=img.dtype) 
    padded_img[1:M+1, 1:N+1, :] = img
    filtered_img = np.zeros((M, N, K), dtype=img.dtype)

    # (i, j) in filteredImg is (i_prim, j_prim) = (i+1, i+j) in padded_img
    for i in range(M):
        for j in range(N):
            i_prim = i+1
            j_prim = j+1
            filtered_img[i, j, :] = calCorr(padded_img[i_prim-1:i_prim+2, j_prim-1:j_prim+2, :])
    return filtered_img

# calculate cross correlation with a box filter
def calCorr(img_portion):
    corr_value = np.zeros(3, dtype=img_portion.dtype)
    if img_portion.shape != (3, 3, 3):
        raise ValueError("Shape doesn't match")
    for i in range(3):
        corr_value[i] = (np.round((np.sum(img_portion[:, :, i])) / 9)).astype(img_portion.dtype)
    return corr_value

def filterUsingMatrices(img):
    M, N, K = img.shape

    # adds two row and column for kernel: (3, 3)
    padded_img = np.zeros((M+2, N+2, K), dtype=img.dtype) 
    padded_img[1:M+1, 1:N+1, :] = img

    filtered_img = np.zeros((M, N, K), dtype=np.float64)

    for i in range(3):
        for j in range(3):
            filtered_img += padded_img[i:i+M, j:j+N, :]
    return (np.round((1/9) * filtered_img)).astype(img.dtype)

# ----------------------- MAIN CODE ------------------------------------

img = cv2.imread('Pink.jpg', cv2.IMREAD_UNCHANGED)

# method 1: using opencv functions
time_origin = time.process_time()
filtered_img_method1 = cv2.boxFilter(img, ddepth=-1, ksize=(3, 3), normalize=1, borderType=cv2.BORDER_CONSTANT)
time_length = time.process_time() - time_origin
print("Method 1 took {} seconds".format(time_length))
# print("============================")
# print(filtered_img_method1)

# method 2: using double for
time_origin = time.process_time()
filtered_img_method2 = applyBoxFilter(img)
time_length = time.process_time() - time_origin
print("Method 2 took {} seconds".format(time_length))
# print("============================")
# print(filtered_img_method2)

# method 3: using 9 matrices
time_origin = time.process_time()
filtered_img_method3 = filterUsingMatrices(img)
time_length = time.process_time() - time_origin
print("Method 3 took {} seconds".format(time_length))
# print("============================")
# print(filtered_img_mehtod3)

print("errors: ")
print( np.sum(np.abs(filtered_img_method2.astype(np.int32) - filtered_img_method1.astype(np.int32))) )
print( np.sum(np.abs(filtered_img_method2.astype(np.int32) - filtered_img_method3.astype(np.int32))) )

cv2.imwrite('res07.jpg', filtered_img_method1)
cv2.imwrite('res08.jpg', filtered_img_method2)
cv2.imwrite('res09.jpg', filtered_img_method3)

# pf.showImg(filtered_img_method1, (960, 960), 'method 1')
# pf.showImg(filtered_img_mehtod3, (960, 960), 'method 2')
# cv2.waitKey(0)
# cv2.destroyAllWindows()