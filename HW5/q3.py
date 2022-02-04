import numpy as np
import cv2 as cv
import utils as utl

#-------------------------------- FUNCTION ------------------------------#
def maskGenerator(M, N, intersect, thr):
    "Generates a binary mask which decreses linearly"
    mask = np.zeros((M, N), np.float64)

    mask[:, 0:intersect-thr] = 1
    mask[:, intersect-thr:intersect+thr] = np.linspace(1, 0, 2*thr, endpoint=False)
    mask[:, intersect+thr:] = 0    
    # convert to three channel mask in order to perform on RGB img
    mask = np.stack([mask for _ in range(3)], axis=2)
    
    return mask

def applyFeathring(img1, img2, intersect, thr):
    "Applies feathring to two image using a mask"
    M, N, _ = img1.shape
    
    # generate mask:    
    mask = maskGenerator(M, N, intersect, thr)
    # apply the mask to both images
    img1_feathered = (img1.astype(np.float64)) * mask
    img2_feathered = (img2.astype(np.float64)) * (1-mask)
    
    # merge both images:    
    merged_img = img1_feathered+img2_feathered
    
    return merged_img
#-------------------------------- MAIN ------------------------------#
img1_orig = cv.imread('res08.jpg', cv.IMREAD_COLOR)
img2_orig = cv.imread('res09.jpg', cv.IMREAD_COLOR)

M, N, _ = img1_orig.shape

# coressponded points in both images which will be used in warping
points1 = np.array([[470, 371], [505, 361], [484, 641], [518, 631]])
points2 = np.array([[445, 290], [515, 256], [465, 698], [533, 692]])

# finding transformation
pers_mat = cv.getPerspectiveTransform(points1.astype(np.float32), 
                                      points2.astype(np.float32))
# warping img
warped_img = cv.warpPerspective(img1_orig, pers_mat, (N, M), cv.INTER_LINEAR, cv.BORDER_CONSTANT, 1)

cv.imwrite('res08-warped.jpg', warped_img)

img1 = np.copy(img2_orig)
img2 = np.copy(warped_img)

final_img = np.zeros((M, N, 3), np.float64)
intersect = 500

sigma = 2
initial_window_size = 2
iterations = int(np.log2(intersect / initial_window_size))
for i in range(1, iterations):
    print("At step {}".format(i))
    img1_blured = cv.GaussianBlur(img1, (6*sigma+1, 6*sigma+1), sigma, cv.BORDER_CONSTANT)
    img2_blured = cv.GaussianBlur(img2, (6*sigma+1, 6*sigma+1), sigma, cv.BORDER_CONSTANT)

    img1_laplacian = img1.astype(np.float64)-img1_blured.astype(np.float64)
    img2_laplacian = img2.astype(np.float64)-img2_blured.astype(np.float64)
        
    merged_img = applyFeathring(img1_laplacian, img2_laplacian, intersect, initial_window_size * (2**(i-1)))
    
    final_img += merged_img

    img1 = np.copy(img1_blured)
    img2 = np.copy(img2_blured)

# final staep, add low frequencies:
img1_blured = cv.GaussianBlur(img1, (6*sigma+1, 6*sigma+1), sigma, cv.BORDER_CONSTANT)
img2_blured = cv.GaussianBlur(img2, (6*sigma+1, 6*sigma+1), sigma, cv.BORDER_CONSTANT)
merged_img = applyFeathring(img1_blured, img2_blured, intersect, initial_window_size * (2**i))

final_img += merged_img

final_img_rep = utl.scaleIntensities(final_img, 'C')
cv.imwrite('res10.jpg', final_img_rep)
print("Done!")
