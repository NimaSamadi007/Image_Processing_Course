import numpy as np
import cv2

# Utility functions:
def repeatRow(row, num):
    row_format = row.reshape(1, row.shape[0])
    repeater = np.ones((num, 1), dtype=row.dtype)
    return repeater @ row_format

def repeatCol(col, num):
    col_format = col.reshape(col.shape[0], 1)
    repeater = np.ones((1, num), dtype=col.dtype)
    return col_format @ repeater

def showRange(matrix):
    print("The abs range is [{}, {}] and the type is {}".format(np.abs(np.amin(matrix)), 
                                                                np.abs(np.amax(matrix)),
                                                                matrix.dtype))

def calGaussFilter(dsize, sigma, normal=False):
    # calculates gaussian kernel 
    ## normal: normalizes filter
    M, N = dsize
    u_row = np.arange(N) - N // 2
    U_matrix = repeatRow(u_row, M)
    v_col = np.arange(M) - M // 2
    V_matrix = repeatCol(v_col, N)
    filter = np.exp(-(U_matrix ** 2 + V_matrix ** 2)/(2*(sigma**2)))
    if normal:
        return filter / (np.sum(filter))
    else:
        return filter

def showImg(img, res_factor, title='input image', wait_flag=True):
    res = (int(img.shape[1]*res_factor), int(img.shape[0]*res_factor))
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    if wait_flag:
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return

def scaleIntensities(img, mode='Z'):
    ## scale intensities to be shown as a picture
    ## modes:
    #   1) Z: scales negatives to zero (default)
    #   2) M: adds -min(img) to image
    img_scaled = np.copy(img) 
    if mode == 'Z':
        img_scaled[img < 0] = 0
    elif mode == 'M':
        if np.amin(img) < 0:
            img_scaled = img + (-np.amin(img))
    else:
        raise ValueError("Unknown mode!")
    if np.amax(img_scaled):
        img_scaled = (img_scaled / np.amax(img_scaled)) * 255
    else:
        img_scaled *= 255
    return img_scaled.astype(np.uint8)

def calImgFFT(img):
    img_fft = np.zeros(img.shape, dtype=np.complex128)
    for i in range(3):
        img_fft[:, :, i] = np.fft.fftshift(np.fft.fft2(img[:, :, i]))
    return img_fft

def calImgIFFT(img_fft):
    img = np.zeros(img_fft.shape, dtype=np.complex128)
    for i in range(3):
        img[:, :, i] = np.fft.ifft2(np.fft.ifftshift(img_fft[:, :, i]))
    return img

# warping function
def myWarpFunction(img, trans_matrix, dsize):
    # opencv format
    # N: width, M: height of the image
    M, N = dsize
    warped_img = np.zeros((M, N, 3), dtype=np.uint8)
    inverse_M = np.linalg.inv(trans_matrix)
    for i in range(M): # i: height (y in opencv, x in numpy)
        for j in range(N): # j: width (x in opencv, y in numpy)
            corr_pixel = inverse_M @ np.array([j, i, 1], dtype=np.float64).reshape(3, 1)
            corr_pixel = np.array([corr_pixel[0], corr_pixel[1]]) / corr_pixel[2]
            # in numpy format, assign each pixel in warped image
            assignPixels(img, warped_img, corr_pixel, i, j)
    
    return warped_img

def assignPixels(img, warped_img, corr_pixel, i, j):
    # assigns warped image pixels for each channel from 
    # original image
    # x, y are in opencv format
    M, N, _ = img.shape
    y = int(corr_pixel[0])
    x = int(corr_pixel[1])
    a = corr_pixel[0] - y
    b = corr_pixel[1] - x
    A = np.array([1-a, a], dtype=np.float64).reshape(1, 2)
    B = np.array([1-b, b], dtype=np.float64).reshape(2, 1)
    for k in range(3):
        img_mat = np.array([[img[x % M, y % N, k], img[(x+1) % M, y, k]], 
                            [img[x % N, (y+1) % N, k], img[(x+1) % M, (y+1) % N, k]]])     
        warped_img[i, j, k] = (A @ img_mat @ B).astype(np.uint8)
    return