from typing import final
import numpy as np
import cv2
import utils as utl


near_img = cv2.imread('./res19-near.jpg', cv2.IMREAD_COLOR)
far_img = cv2.imread('./res20-far.jpg', cv2.IMREAD_COLOR)

M1, N1, _ = near_img.shape
M2, N2, _ = far_img.shape

print("Near image shape: {}".format(near_img.shape))
print("Far image shape: {}".format(far_img.shape))

M, N = M1, N1


pts_near = np.array([[254, 124], [168, 129], [240, 544]])
pts_far = np.array([[409, 220], [324, 224], [352, 641]])

transform_matrix = cv2.getAffineTransform(pts_far.astype(np.float32), 
                                          pts_near.astype(np.float32))

far_img_changed = cv2.warpAffine(far_img, transform_matrix, (N, M))

cv2.imwrite('res21-near.jpg', near_img)
cv2.imwrite('res22-far.jpg', far_img_changed)

near_img_fft = utl.calImgFFT(near_img)
far_img_fft = utl.calImgFFT(far_img_changed)

near_img_fft_abs = utl.scaleGrayIntensities(np.log(np.abs(near_img_fft)), 'M')
far_img_fft_abs = utl.scaleGrayIntensities(np.log(np.abs(far_img_fft)), 'M')

cv2.imwrite('res23-dft-near.jpg', near_img_fft_abs)
cv2.imwrite('res24-dft-far.jpg', far_img_fft_abs)

# low pass and high pass filters:
s = 8
lowpass_filter = utl.calGaussFilter((M, N), s)
r = 50
highpass_filter = 1 - utl.calGaussFilter((M, N), r)

lowpass_filter_repr = (lowpass_filter * 255).astype(np.uint8)
highpass_filter_repr = (highpass_filter * 255).astype(np.uint8)

cv2.imwrite('res25-highpass-{}.jpg'.format(r), highpass_filter_repr)
cv2.imwrite('res26-lowpass-{}.jpg'.format(s), lowpass_filter_repr)

# filter corresponding images
lowpass_filter = np.stack([lowpass_filter, lowpass_filter, lowpass_filter], axis=2)
highpass_filter = np.stack([highpass_filter, highpass_filter, highpass_filter], axis=2)

far_img_filtered = far_img_fft * lowpass_filter.astype(far_img_fft.dtype)
near_img_filtered = near_img_fft * highpass_filter.astype(near_img_fft.dtype)

far_img_filtered_repr = utl.scaleGrayIntensities(np.log(1+np.abs(far_img_filtered)), 'Z')
near_img_filtered_repr = utl.scaleGrayIntensities(np.log(1+np.abs(near_img_filtered)), 'Z')

cv2.imwrite('res27-highpassed.jpg', near_img_filtered_repr)
cv2.imwrite('res28-lowpassed.jpg', far_img_filtered_repr)

alpha = 0.45
hybrid_img_fft = alpha * far_img_filtered + (1-alpha) * near_img_filtered

hybrid_img_fft_repr = utl.scaleGrayIntensities(np.log(1+np.abs(hybrid_img_fft)), 'Z')

cv2.imwrite('res29-hybrid.jpg', hybrid_img_fft_repr)

hybrid_img = utl.calImgIFFT(hybrid_img_fft)

hybrid_img = utl.scaleGrayIntensities(np.abs(hybrid_img), 'M')


cv2.imwrite('final.jpg', hybrid_img)

