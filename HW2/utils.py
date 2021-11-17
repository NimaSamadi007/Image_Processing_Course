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

def calGaussFilter(dsize, sigma):
    # calculates gaussian kernel - size must be odd
    M, N = dsize
    u_row = np.arange(N) - N // 2
    U_matrix = repeatRow(u_row, M)
    v_col = np.arange(M) - M // 2
    V_matrix = repeatCol(v_col, N)
    return np.exp(-(U_matrix ** 2 + V_matrix ** 2)/(2*(sigma**2)))

def showImg(img, res_factor, title='input image'):
    res = (int(img.shape[1]*res_factor), int(img.shape[0]*res_factor))
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    return

def scaleGrayIntensities(img, mode='Z'):
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
    img_scaled = (img_scaled / np.amax(img_scaled)) * 255
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
