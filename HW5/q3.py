import numpy as np
import cv2 as cv
import utils as utl

img1 = cv.imread('res08.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('res09-init.jpg', cv.IMREAD_COLOR)

M, N, _ = img1.shape

print(img1.shape)
print(img2.shape)

points1 = np.array([[368, 199], [440, 576], [287, 577], [366, 924]])
# points1 = np.array([[368, 199], [440, 577], [287, 577]])
points2 = np.array([[381, 268], [438, 471], [338, 469], [381, 718]])
# points2 = np.array([[367, 251], [438, 469], [338, 469]])


pers_mat = cv.getPerspectiveTransform(points2.astype(np.float32), 
                                      points1.astype(np.float32))

# affine_mat = cv.getAffineTransform(points2.astype(np.float32), points1.astype(np.float32))

warped_img2 = cv.warpPerspective(img2, pers_mat, (N, M), cv.INTER_LINEAR, cv.BORDER_CONSTANT, 1)
# warped_img2 = cv.warpAffine(img2, affine_mat, (N, M), cv.INTER_LINEAR, cv.BORDER_CONSTANT, 1)

# utl.showImg(warped_img2, 0.5)

cv.imwrite('res08-sim.jpg', warped_img2)

#%%
import numpy as np
import cv2 as cv
import utils as utl


def maskGenerator(M, N, intersect, thr):
    "Generates a binary mask which decreses linearly"
    mask = np.zeros((M, N), np.float64)

    mask[:, 0:intersect-thr] = 1
    mask[:, intersect-thr:intersect+thr] = np.linspace(1, 0, 2*thr, endpoint=False)
    mask[:, intersect+thr:] = 0    
    # convert to three channel mask in order to perform on RGB img
    mask = np.stack([mask for _ in range(3)], axis=2)
    
    return mask


img1 = cv.imread('res08.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('res08-sim.jpg', cv.IMREAD_COLOR)

M, N, _ = img1.shape

print(img1.shape)
print(img2.shape)

intersect = 372
thr = 70
mask = maskGenerator(M, N, intersect, thr)

img1_feathered = (img1.astype(np.float64)) * mask
img2_feathered = (img2.astype(np.float64)) * (1-mask)

img1_repr = img1_feathered.astype(np.uint8)
img2_repr = img2_feathered.astype(np.uint8)

utl.showImg(img1_repr, 0.5, 'img1', False)
utl.showImg(img2_repr, 0.5, 'img2', False)

merged_img = img1_feathered+img2_feathered
utl.showImg(merged_img.astype(np.uint8), 0.5)

# mask_repr = (mask * 255).astype(np.uint8)
# utl.showImg(mask_repr, 0.5, 'n', False)
# utl.showImg(255-mask_repr, 0.5)

