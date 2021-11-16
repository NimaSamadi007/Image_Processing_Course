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
    print("The range is [{}, {}] and the type is {}".format(np.amin(matrix), 
                                                            np.amax(matrix),
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
    img_scaled = (img_scaled / np.amax(img_scaled)) * 255
    return img_scaled.astype(np.uint8)
