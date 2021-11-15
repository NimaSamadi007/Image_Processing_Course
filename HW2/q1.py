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
    fft_amp = np.log(np.abs(img_fft))
    fft_amp[np.nonzero(fft_amp == -np.inf)] = np.nextafter(0, 1)
    fft_amp = scaleIntensities(fft_amp)

    return fft_amp.astype(np.uint8)

## -------------------------- MAIN ----------------------- ##

img = cv2.imread('./flowers.blur.png', cv2.IMREAD_COLOR)

"""
## Part a)
sigma = 1

gauss_kernal = cv2.getGaussianKernel(6*sigma+1, sigma)

gauss_kernal_mat = gauss_kernal @ gauss_kernal.T # matrix form
gauss_kernal_mat_repr = scaleIntensities(gauss_kernal_mat)
gauss_kernal_mat_repr = cv2.resize(gauss_kernal_mat_repr, (500, 500))
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

showImg(img, 0.8, 'img')
showImg(img_sharpend, 0.8, 'sharpened')
cv2.waitKey(0)
cv2.destroyAllWindows()
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

log_fft_amp = calAbsFFT(img_fft_shifted)

# showImg(log_fft_amp, 0.8, 'log fft')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('res08.jpg', log_fft_amp)

# low pass gaussian filter
sigma = 2
gauss_kernal = cv2.getGaussianKernel(6*sigma+1, sigma)
gauss_kernal_mat = gauss_kernal @ gauss_kernal.T 

M, N = img_fft_shifted.shape

low_pass_filter = np.fft.fft2(gauss_kernal_mat)
low_pass_shifted = np.fft.fftshift(low_pass_filter)

low_pass_ext = np.zeros((M, N), dtype=img_fft.dtype)
test = calAbsFFT(low_pass_shifted)
test = cv2.resize(test, (500, 500))
low_pass_ext[M//2-3*sigma : M//2+3*sigma+1, N//2-3*sigma : N//2+3*sigma+1] = low_pass_shifted

showImg(test, 0.8, 'highpass filter')
cv2.waitKey(0)
cv2.destroyAllWindows()


H = 1 - low_pass_ext

high_pass_repr = calAbsFFT(H)
high_pass_repr = cv2.resize(high_pass_repr, (500, 500))

print(high_pass_repr)

cv2.imwrite('res09.jpg', high_pass_repr)

# showImg(high_pass_repr, 0.8, 'highpass filter')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# 1 + k * H * F calculation:

k = 1
filtered_img_fft = 1 + k * H * img_fft_shifted

filtered_img_fft_repr = calAbsFFT(filtered_img_fft)
# H_f = calAbsFFT(H)

# showImg(filtered_img_fft_repr, 0.8, 'filtered img fft')
# cv2.waitKey(0)
# cv2.destroyAllWindows()

cv2.imwrite('res10.jpg', filtered_img_fft_repr)

# filtered_img = np.fft.ifftshift(filtered_img_fft)
# filtered_img = np.fft.ifft2(filtered_img)
# print(filtered_img.shape)


# high_pass_filter_repr = cv2.resize(high_pass_filter, (500, 500))
# cv2.imwrite('res01.jpg', gauss_kernal_mat_repr)

# fig.tight_layout()
# plt.show()

## Part d)


