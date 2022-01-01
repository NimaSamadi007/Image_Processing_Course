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

def showRange(arr, mode='ABS'):
    if mode == 'ABS':
        print("Abs range = [{}, {}] \n Type = {}, Shape = {}".format(np.abs(np.amin(arr)), 
                                                                np.abs(np.amax(arr)),
                                                                arr.dtype,
                                                                arr.shape))
    else:
        print("Range = [{}, {}] \n Type = {}, Shape = {}".format(np.amin(arr), 
                                                                np.amax(arr),
                                                                arr.dtype,
                                                                arr.shape))


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
    if res_factor >= 1:
        img_show = cv2.resize(img, res, interpolation=cv2.INTER_LINEAR)
    else:
        img_show = cv2.resize(img, res, interpolation=cv2.INTER_AREA)

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
    #   3) C: cut intensities greater than 255 and less than 0
    img_scaled = np.copy(img) 
    if mode == 'Z':
        img_scaled[img < 0] = 0
    elif mode == 'M':
        if np.amin(img) < 0:
            img_scaled = img + (-np.amin(img))
    elif mode == 'C':
        img_scaled[img_scaled > 255] = 255
        img_scaled[img_scaled < 0] = 0
        return img_scaled.astype(np.uint8)
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
    "warps image using trans_matrix in numpy format format"
    M, N = dsize
    warped_img = np.zeros((M, N, 3), dtype=np.uint8)
    inverse_M = np.linalg.inv(trans_matrix)
    for i in range(M): 
        for j in range(N): 
            corr_pixel = inverse_M @ np.array([i, j, 1], dtype=np.float64).reshape(3, 1)
            corr_pixel = np.array([corr_pixel[0], corr_pixel[1]]) / corr_pixel[2]
            if corr_pixel[0] + 1 >= 0 and corr_pixel[1] + 1 >= 0:
                assignPixels(img, warped_img, corr_pixel, i, j)
            else:
                warped_img[i, j, :] = 0
    return warped_img

def assignPixels(img, warped_img, corr_pixel, i, j):
    # assigns warped image pixels for each channel from 
    # original image
    M, N, _ = img.shape
    x = int(corr_pixel[0])
    y = int(corr_pixel[1])
    a = corr_pixel[0] - x
    b = corr_pixel[1] - y
    A = np.array([1-a, a], dtype=np.float64).reshape(1, 2)
    B = np.array([1-b, b], dtype=np.float64).reshape(2, 1)    
    for k in range(3):
        if x < M and y < N:
            elem11 = img[x, y, k]
        else:
            elem11 = 0
        if x < M and (y+1) < N:
            elem12 = img[x, y+1, k]
        else:
            elem12 = 0
        if (x+1) < M and y < N:
            elem21 = img[x+1, y, k]
        else:
            elem21 = 0
        if (x+1) < M and (y+1) < N:
            elem22 = img[x+1, y+1, k]
        else:
            elem22 = 0
        img_mat = np.array([[elem11, elem12], 
                            [elem21, elem22]])     
        warped_img[i, j, k] = (A @ img_mat @ B).astype(np.uint8)
    return

def cv2numpy(tran):
    "converts transformation in opencv format to numpy format ( (x,y) in numpy is (y, x) in opencv)"
    swaped_tran = np.copy(tran)
    # swap first and second col:
    swaped_tran[:, [0, 1]] = swaped_tran[:, [1, 0]]
    # swap first and second row:
    swaped_tran[[0, 1], :] = swaped_tran[[1, 0], :]
    return swaped_tran

def findPath(path_mat, arr, M, N):
    path = []
    current_ind = np.argmin(path_mat[:, -1])
    path.append(current_ind)
    for j in range(N-1, 0, -1):
        val = path_mat[current_ind, j] - arr[current_ind, j]
        if current_ind == 0:
            possible_indices = np.where(path_mat[0:current_ind+2, j-1] == val)[0]
            current_ind = possible_indices[0]    
        elif current_ind == M-1:
            possible_indices = np.where(path_mat[current_ind-1:current_ind+1, j-1] == val)[0]
            current_ind += (possible_indices[0] - 1)    
        else:
            possible_indices = np.where(path_mat[current_ind-1:current_ind+2, j-1] == val)[0]
            current_ind += (possible_indices[0] - 1)
        # print(current_ind)
        path.append(current_ind)
    
    return np.flip(path)

def findMinCut(matrix, mode="COL"):
    "finds mincut in cols or rows of matrix"
    "COL mode: finds cut in columns of matrix (row min cut)"
    "ROW mode: finds cut in rows of matrix (coloumn min cut)"
    "returns path and path matrix in output"
    if mode == "COL":
        arr = np.copy(matrix)
    elif mode == "ROW":
        arr = np.copy(matrix.T)
    else:
        raise ValueError("Unknown mode inserted!")

    M, N = arr.shape
    path_mat = np.zeros((M, N), dtype=np.float64)
    path_mat[:, 0] = arr[:, 0]

    for j in range(1, N):
        for i in range(M):
            if i == 0:
                path_mat[i, j] = arr[i, j] + np.amin(path_mat[i:i+2, j-1])
            elif i == M-1:
                path_mat[i, j] = arr[i, j] + np.amin(path_mat[i-1:i+1, j-1])
            else:
                path_mat[i, j] = arr[i, j] + np.amin(path_mat[i-1:i+2, j-1])

    # find cut path by back substitution:
    path = findPath(path_mat, arr, M, N)
    if mode == "COL":
        return path.reshape(1, path.shape[0])
    else:
        path = path.T
        path_mat = path_mat.T
        return path.reshape(path.shape[0], 1)

def findLocalMax(mat, level_thr, noise_power):
    "Finds local maximum of matrix, ignores values less than threshold"
    "Noise is added to img to eliminate value equalities"
    
    M, N = mat.shape
    # append zeros to left, right, top and bottom of matrix
    ext_mat  = np.zeros((M+2, N+2), dtype=np.float64)
    ext_mat[1:-1, 1:-1] = mat
    less_indices = np.nonzero(ext_mat <= level_thr)
    
    noise = np.random.normal(0, noise_power, ext_mat.shape)
    # add noise:
    ext_mat += noise
    # ignore values less than threshold
    ext_mat[less_indices] = 0
    # find local maximum:
    x_max = []
    y_max = []
    for i in range(M):
        for j in range(N):
            # compare with all 8 neigbours
            if (ext_mat[i+1, j+1] > ext_mat[i, j] and
                    ext_mat[i+1, j+1] > ext_mat[i, j+1] and
                    ext_mat[i+1, j+1] > ext_mat[i, j+2] and
                    ext_mat[i+1, j+1] > ext_mat[i+1, j] and
                    ext_mat[i+1, j+1] > ext_mat[i+1, j+2] and
                    ext_mat[i+1, j+1] > ext_mat[i+2, j] and
                    ext_mat[i+1, j+1] > ext_mat[i+2, j+1] and
                    ext_mat[i+1, j+1] > ext_mat[i+2, j+2]) :
                x_max.append(i)
                y_max.append(j)
    return np.array([x_max, y_max])
        
    
    