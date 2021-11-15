import numpy as np
import matplotlib.pyplot as plt
import cv2

## -------------------------- FUNCIONS ----------------------- ##
def showImg(img, res_factor, title='input image'):
    res = (int(img.shape[1]*res_factor), int(img.shape[0]*res_factor))
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    return

# scales picture's intensities to the range of [0, 255] 
# img: numpy array
def scaleIntensities(img):
    if np.amin(img) < 0:
        img_scaled = img + (-np.amin(img))
    else:
        img_scaled = np.copy(img)
    img_scaled = (img_scaled / np.amax(img_scaled)) * 255
    return img_scaled.astype(np.uint8)

def calAbsFFT(img_fft):
    fft_amp = np.abs(img_fft)
    fft_amp = scaleIntensities(fft_amp)
    return fft_amp.astype(np.uint8)

def showRange(matrix):
    print("The range is [{}, {}] and the type is {}".format(np.amin(matrix), 
                                                            np.amax(matrix),
                                                            matrix.dtype))
## -------------------------- MAIN ----------------------- ##

img = cv2.imread('./flowers.blur.png', cv2.IMREAD_COLOR)

"""
## Part a)
sigma = 1

gauss_kernal = cv2.getGaussianKernel(6*sigma+1, sigma)

gauss_kernal_mat = gauss_kernal @ gauss_kernal.T # matrix form
gauss_kernal_mat_repr = cv2.resize(gauss_kernal_mat, (500, 500))
gauss_kernal_mat_repr = scaleIntensities(gauss_kernal_mat_repr)
cv2.imwrite('res01.jpg', gauss_kernal_mat_repr)

# smooth image
img_smoothed = cv2.filter2D(img.astype(float), -1, gauss_kernal, borderType=cv2.BORDER_CONSTANT) 
img_smoothed_repr = scaleIntensities(img_smoothed)
cv2.imwrite('res02.jpg', img_smoothed_repr)

unsharp_mask = img.astype(float) - img_smoothed.astype(float) # unsharp mask
unsharp_mask_repr = scaleIntensities(unsharp_mask)
cv2.imwrite('res03.jpg', unsharp_mask_repr)

alpha = 1.8
print(np.amin(img), np.amax(img))
print(np.amin(unsharp_mask), np.amax(unsharp_mask))
img_sharpend = img.astype(float) + alpha * unsharp_mask.astype(float)
print(np.amin(img_sharpend), np.amax(img_sharpend))
img_sharpend = scaleIntensities(img_sharpend)
cv2.imwrite('res04.jpg', img_sharpend)

# showImg(img, 0.8, 'img')
# showImg(img_sharpend, 0.8, 'sharpened')
# cv2.waitKey(0)
# cv2.destroyAllWindows()


## Part b)
sigma = 2
gauss_kernal = cv2.getGaussianKernel(6*sigma+1, sigma)
gauss_kernal_mat = gauss_kernal @ gauss_kernal.T 
laplacian_gauss = cv2.Laplacian(gauss_kernal_mat, ddepth=-1, ksize=1, 
                                borderType=cv2.BORDER_CONSTANT) # laplacian 3 * 3 kernel

laplacian_gauss_repr = laplacian_gauss * (255 / np.amax(laplacian_gauss))
laplacian_gauss_repr = laplacian_gauss_repr.astype(np.uint8)
laplacian_gauss_repr = cv2.resize(laplacian_gauss_repr, (500, 500))
cv2.imwrite('res05.jpg', laplacian_gauss_repr)

unsharp_mask = cv2.filter2D(img.astype(float), -1, laplacian_gauss, borderType=cv2.BORDER_CONSTANT)
print(np.amin(unsharp_mask), np.amax(unsharp_mask))
unsharp_mask_repr = scaleIntensities(unsharp_mask)
cv2.imwrite('res06.jpg', unsharp_mask.astype(np.uint8))

k = 1.1
img_sharpend = img.astype(float) - k * (unsharp_mask.astype(float))
print(np.amin(img_sharpend), np.amax(img_sharpend))
img_sharpend = scaleIntensities(img_sharpend)
cv2.imwrite('res07.jpg', img_sharpend)

showImg(img, 0.8, 'img')
showImg(img_sharpend, 0.8, 'sharpened')
cv2.waitKey(0)
cv2.destroyAllWindows()

"""
## Part c)
img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
img_fft = np.fft.fft2(img_hsv[:, :, 2].astype(float))
img_fft_shifted = np.fft.fftshift(img_fft)


fft_amp_log = np.log(np.abs(img_fft_shifted))

fft_amp_log_repr = scaleIntensities(fft_amp_log)


# showImg(fft_amp_log_repr, 0.8, 'log fft')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('res08.jpg', fft_amp_log_repr)

# low pass filter in spatial domain
sigma = 30
gauss_kernal = cv2.getGaussianKernel(6*sigma+1, sigma)
gauss_kernal_mat = gauss_kernal @ gauss_kernal.T 

M, N = img_fft.shape
gauss_ext = np.zeros((M, N), dtype=gauss_kernal_mat.dtype)
gauss_ext[M//2-3*sigma : M//2+3*sigma+1, N//2-3*sigma : N//2+3*sigma+1] = gauss_kernal_mat

lowpass_filter = np.fft.fft2(gauss_ext)
lowpass_filter = np.fft.fftshift(lowpass_filter)

# high pass filter:


# highpass_filter_repr = calAbsFFT(highpass_filter)
lowpass_filter_repr = calAbsFFT(lowpass_filter)


# H = 1 - np.abs(lowpass_filter)
H = 1 - lowpass_filter
highpass_filter_repr = scaleIntensities(H)

# showImg(lowpass_filter_repr, 1, 'lowpass filter')
# showImg(highpass_filter_repr, 1, 'highpass filter')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

k = 1
img_filtered_freq = 1 + k * H * img_fft_shifted

img_filtered_freq = np.fft.ifftshift(img_filtered_freq)
img_filtered = np.fft.ifft2(img_filtered_freq)

img_filtered_repr = calAbsFFT(img_filtered)

img_hsv[:, :, 2] = img_filtered_repr
img_rgb = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

showImg(img_rgb, 1, 'filtered img')
cv2.waitKey(0)
cv2.destroyAllWindows()

## Part d)


