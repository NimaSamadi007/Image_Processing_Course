import numpy as np
import cv2
from numpy.core.fromnumeric import amax, amin

## -------------------------- FUNCIONS ----------------------- ##
def showImg(img, res, title='input image'):
    img_show = cv2.resize(img, res)
    cv2.imshow(title, img_show)
    return

# scales picture's intensities to the range of [0, 255] 
# img: numpy array
def scaleIntensities(img):
    img_scaled = img + (-np.amin(img))
    img_scaled[np.nonzero(img_scaled > 255)] = 255
    return img_scaled.astype(np.uint8)



## -------------------------- MAIN ----------------------- ##

img = cv2.imread('./flowers.blur.png')

## Part a)
sigma = 2

gauss_kernal = cv2.getGaussianKernel(6*sigma+1, sigma)

gauss_kernal_mat = gauss_kernal @ gauss_kernal.T # matrix form
gauss_kernal_mat_repr = gauss_kernal_mat * (255 / np.amax(gauss_kernal_mat)) # scale values to [0, 255] range
gauss_kernal_mat_repr = gauss_kernal_mat_repr.astype(np.uint8) # change type to show picture
gauss_kernal_mat_repr = cv2.resize(gauss_kernal_mat_repr, (200, 200))
cv2.imwrite('res01.jpg', gauss_kernal_mat_repr)

# smooth image
img_smoothed = cv2.filter2D(img, -1, gauss_kernal, borderType=cv2.BORDER_CONSTANT) 
cv2.imwrite('res02.jpg', img_smoothed)

unsharp_mask = img.astype(int) - img_smoothed.astype(int) # unsharp mask

unsharp_mask_repr = scaleIntensities(unsharp_mask)
cv2.imwrite('res03.jpg', unsharp_mask_repr)

alpha = 1
img_sharpend_1 = img.astype(int) + alpha * unsharp_mask
img_sharpend_1 = scaleIntensities(img_sharpend_1)
cv2.imwrite('res04.jpg', img_sharpend_1)

## Part b)
sigma = 2
gauss_kernal = cv2.getGaussianKernel(6*sigma+1, sigma)
gauss_kernal_mat = gauss_kernal @ gauss_kernal.T 
laplacian_gauss = cv2.Laplacian(gauss_kernal_mat, -1, 
                                borderType=cv2.BORDER_CONSTANT)

print(laplacian_gauss.shape)
laplacian_gauss_repr = laplacian_gauss * (255 / np.amax(laplacian_gauss))
laplacian_gauss_repr = laplacian_gauss_repr.astype(np.uint8)
laplacian_gauss_repr = cv2.resize(laplacian_gauss_repr, (200, 200))
cv2.imwrite('res05.jpg', laplacian_gauss_repr)

unsharp_mask = cv2.filter2D(img, -1, laplacian_gauss, borderType=cv2.BORDER_CONSTANT)
cv2.imwrite('res06.jpg', (unsharp_mask + 128).astype(np.uint8))

k = 5
img_sharpend_2 = img.astype(int) + k * (unsharp_mask.astype(int))
img_sharpend_2 = scaleIntensities(img_sharpend_2)
cv2.imwrite('res07.jpg', img_sharpend_2)

## Part c)

## Part d)


