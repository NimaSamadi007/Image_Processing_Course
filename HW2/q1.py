import numpy as np
import matplotlib.pyplot as plt
import cv2
import utils as utl

img = cv2.imread('./flowers.blur.png', cv2.IMREAD_COLOR)
"""
## Part a)
sigma = 1
gauss_kernel = utl.calGaussFilter((3, 3), sigma, True)

gauss_kernal_repr = cv2.resize(gauss_kernel, (500, 500))
gauss_kernal_repr = utl.scaleIntensities(gauss_kernal_repr)
cv2.imwrite('res01.jpg', gauss_kernal_repr)

# smooth image
img_smoothed = cv2.filter2D(img.astype(np.float64), -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT) 
img_smoothed_repr = utl.scaleIntensities(img_smoothed, 'M')
cv2.imwrite('res02.jpg', img_smoothed_repr)

unsharp_mask = img.astype(np.float64) - img_smoothed.astype(np.float64) # unsharp mask


unsharp_mask_repr = utl.scaleIntensities(unsharp_mask, 'M')
cv2.imwrite('res03.jpg', unsharp_mask_repr)

alpha = 2
img_sharpend = img.astype(float) + alpha * unsharp_mask.astype(float)

img_sharpend = utl.scaleIntensities(img_sharpend, 'M')
cv2.imwrite('res04.jpg', img_sharpend)

## Part b)
sigma = 1
gauss_kernal = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
laplacian_gauss = cv2.Laplacian(gauss_kernal, ddepth=-1, ksize=1, 
                                borderType=cv2.BORDER_CONSTANT) # laplacian 3 * 3 kernel

laplacian_gauss_repr = utl.scaleIntensities(laplacian_gauss, 'M')
laplacian_gauss_repr = cv2.resize(laplacian_gauss_repr, (500, 500))
cv2.imwrite('res05.jpg', laplacian_gauss_repr)

unsharp_mask = cv2.filter2D(img.astype(float), -1, laplacian_gauss, borderType=cv2.BORDER_CONSTANT)
unsharp_mask_repr = utl.scaleIntensities(unsharp_mask, 'M')
cv2.imwrite('res06.jpg', unsharp_mask.astype(np.uint8))

k = 1.2
img_sharpend = img.astype(float) - k * (unsharp_mask.astype(float))
img_sharpend = utl.scaleIntensities(img_sharpend, 'M')
cv2.imwrite('res07.jpg', img_sharpend)

"""

## Part c)
img_fft = utl.calImgFFT(img)

fft_amp_repr = utl.scaleIntensities(np.log(np.abs(img_fft)), 'M')
cv2.imwrite('res08.jpg', fft_amp_repr)

# low pass filter in frequency domain
M, N,_ = img_fft.shape

sigma = 40
lowpass_filter = utl.calGaussFilter((M, N), sigma, False)

# high pass filter:
H = 1 - lowpass_filter
H_repr = utl.scaleIntensities(np.abs(H))
cv2.imwrite('res09.jpg', H_repr)

k = 0.8
filter = 1 + k * H
filter_ext = np.stack([filter, filter, filter], axis=2)
img_filtered_freq = filter_ext * img_fft

img_filtered_freq_repr = utl.scaleIntensities(np.log(np.abs(img_filtered_freq)))

cv2.imwrite('res10.jpg', img_filtered_freq_repr)

img_filtered = utl.calImgIFFT(img_filtered_freq)
img_filtered = utl.scaleIntensities(np.abs(img_filtered))

cv2.imwrite('res11.jpg', img_filtered)

## Part d)
img_fft = utl.calImgFFT(img)
M, N, _ = img_fft.shape

v_row = np.arange(N) - N//2
u_col = np.arange(M) - M//2

v_matrix = utl.repeatRow(v_row, M)
u_matrix = utl.repeatCol(u_col, N)

lap_mat = 4 * (np.pi ** 2) * (v_matrix ** 2 + u_matrix ** 2)
lap_mat_ext = np.stack([lap_mat, lap_mat, lap_mat], axis=2).astype(img_fft.dtype)
int_mat = lap_mat_ext * img_fft
int_mat_repr = utl.scaleIntensities(np.log(1+np.abs(int_mat)))
cv2.imwrite('res12.jpg', int_mat_repr)

mask_img = utl.calImgIFFT(int_mat)
mask_img_repr = utl.scaleIntensities(np.abs(mask_img), 'M')
cv2.imwrite('res13.jpg', mask_img_repr)

k = 5*10**(-7)
sharpened_freq = img_fft + k * int_mat
sharpened_img = utl.calImgIFFT(sharpened_freq)
sharpened_img = utl.scaleIntensities(np.abs(sharpened_img))
cv2.imwrite('res14.jpg', sharpened_img)
