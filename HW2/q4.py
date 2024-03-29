import numpy as np
import cv2
import utils as utl

## ------------------- FUNCTIONS ------------------------- ##
def calAffineTran(pts1, pts2):
    ## Fits an affine transformation from pts1 to pts2 in opencv format
    ## Equation is in the form of AX = B (same as course notations)
    if pts1.shape != (3, 2) or pts2.shape != (3, 2):
        raise ValueError("points shapes must be (3, 2)")
    else:
        A = np.zeros((6, 6), dtype=np.float64)
        for i in range(3):
            xi = pts1[i, 0]
            yi = pts1[i, 1]
            A[2*i:2*(i+1), :] = np.array([[xi, yi, 1, 0, 0, 0],
                                          [0, 0, 0, xi, yi, 1]], dtype=np.float64)

    B = pts2.reshape(6, 1).astype(np.float64)
    # X = A^(-1) * B
    X = np.linalg.inv(A) @ B
    transform_matrix = np.zeros((3, 3), dtype=np.float64)
    transform_matrix[2, 2] = 1
    transform_matrix[0:2, :] = X.reshape(2, 3)
    return transform_matrix
## ------------------- MAIN ------------------------------ ##

# always transform far image to the near image - 
# far image is the bigger image and the near image is the smaller image


img1 = cv2.imread('./res19-near.jpg', cv2.IMREAD_COLOR)
img2 = cv2.imread('./res20-far.jpg', cv2.IMREAD_COLOR)

# points are in opencv format
pts1 = np.array([[499, 410], [107, 402], [205, 241]])
pts2 = np.array([[843, 530], [307, 530], [447, 347]])

M1, N1, _ = img1.shape
M2, N2, _ = img2.shape

if M1 > M2 and N1 > N2:
    M, N = M2, N2
    far_img = img1
    pts_far = pts1

    near_img = img2
    pts_near = pts2
else:
    M, N = M1, N1
    far_img = img2
    pts_far = pts2

    near_img = img1
    pts_near = pts1


transform_matrix = calAffineTran(pts_far, pts_near)
## convert to numpy format
transform_matrix = utl.cv2numpy(transform_matrix)
far_img_changed = utl.myWarpFunction(far_img, transform_matrix, (M, N))

cv2.imwrite('res21-near.jpg', near_img)
cv2.imwrite('res22-far.jpg', far_img_changed)

near_img_fft = utl.calImgFFT(near_img)
far_img_fft = utl.calImgFFT(far_img_changed)

near_img_fft_abs = utl.scaleIntensities(np.log(np.abs(near_img_fft)), 'M')
far_img_fft_abs = utl.scaleIntensities(np.log(np.abs(far_img_fft)), 'M')

cv2.imwrite('res23-dft-near.jpg', near_img_fft_abs)
cv2.imwrite('res24-dft-far.jpg', far_img_fft_abs)

# low pass and high pass filters:
s = 6
lowpass_filter = utl.calGaussFilter((M, N), s)
r = 80
highpass_filter = 1 - utl.calGaussFilter((M, N), r)

lowpass_filter_repr = (lowpass_filter * 255).astype(np.uint8)
highpass_filter_repr = (highpass_filter * 255).astype(np.uint8)

cv2.imwrite('res25-highpass-{}.jpg'.format(r), highpass_filter_repr)
cv2.imwrite('res26-lowpass-{}.jpg'.format(s), lowpass_filter_repr)

# filter corresponding images
lowpass_filter = np.stack([lowpass_filter for _ in range(3)], axis=2)
highpass_filter = np.stack([highpass_filter for _ in range(3)], axis=2)

far_img_filtered = far_img_fft * lowpass_filter.astype(far_img_fft.dtype)
near_img_filtered = near_img_fft * highpass_filter.astype(near_img_fft.dtype)

far_img_filtered_repr = utl.scaleIntensities(np.log(1+np.abs(far_img_filtered)), 'M')
near_img_filtered_repr = utl.scaleIntensities(np.log(1+np.abs(near_img_filtered)), 'M')

cv2.imwrite('res27-highpassed.jpg', near_img_filtered_repr)
cv2.imwrite('res28-lowpassed.jpg', far_img_filtered_repr)

alpha = 0.4
hybrid_img_fft = alpha * far_img_filtered + (1-alpha) * near_img_filtered

hybrid_img_fft_repr = utl.scaleIntensities(np.log(np.abs(hybrid_img_fft)), 'M')

cv2.imwrite('res29-hybrid.jpg', hybrid_img_fft_repr)

hybrid_img = utl.calImgIFFT(hybrid_img_fft)

hybrid_img = utl.scaleIntensities(np.abs(hybrid_img), 'M')

cv2.imwrite('res30-hybrid-near.jpg', hybrid_img)
hybrid_img_small = cv2.resize(hybrid_img, None, fx=0.15, fy=0.15, interpolation=cv2.INTER_AREA)
cv2.imwrite('res31-hybrid-far.jpg', hybrid_img_small)
