import numpy as np
import cv2 as cv
import utils as utl

img1 = cv.imread('res08.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('res09.jpg', cv.IMREAD_COLOR)

M, N, _ = img1.shape

print(img1.shape)
print(img2.shape)

points1 = np.array([[470, 371], [505, 361], [484, 641], [518, 631]])
points2 = np.array([[445, 290], [515, 256], [465, 698], [533, 692]])


# img1_points = utl.drawPoints(img1, points1, 5)
# img2_points = utl.drawPoints(img2, points2, 5)

# utl.showImg(img1_points, 0.5, 'img1', False)
# utl.showImg(img2_points, 0.5)


# affine_mat = cv.getAffineTransform(points2.astype(np.float32),
#                                    points1.astype(np.float32))

pers_mat = cv.getPerspectiveTransform(points1.astype(np.float32), 
                                        points2.astype(np.float32))

warped_img = cv.warpPerspective(img1, pers_mat, (N, M), cv.INTER_LINEAR, cv.BORDER_CONSTANT, 1)
# warped_img2 = cv.warpAffine(img2, affine_mat, (N, M), cv.INTER_LINEAR, cv.BORDER_CONSTANT, 1)

# utl.showImg(img1_points, 0.5, 'img1', False)
utl.showImg(warped_img, 0.5)

cv.imwrite('res09-sim.jpg', warped_img)

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

def applyFeathring(img1, img2, intersect, thr):
    "Applies feathirng to two image using a mask"
    M, N, _ = img1.shape
    
    # generate mask:    
    mask = maskGenerator(M, N, intersect, thr)
    # apply the mask to both images
    img1_feathered = (img1.astype(np.float64)) * mask
    img2_feathered = (img2.astype(np.float64)) * (1-mask)
    
    # merge both images:    
    merged_img = img1_feathered+img2_feathered
    
    return merged_img

    
img1 = cv.imread('res09.jpg', cv.IMREAD_COLOR)
img2 = cv.imread('res09-sim.jpg', cv.IMREAD_COLOR)

M, N, _ = img1.shape

final_img = np.zeros((M, N, 3), np.float64)
intersect = 488

sigma = 2
step = 3
iterations = int(np.log2(intersect / step))
for i in range(1, iterations):
    print("At step {}".format(i))
    img1_blured = cv.GaussianBlur(img1, (3*sigma+1, 3*sigma+1), sigma, cv.BORDER_CONSTANT)
    img2_blured = cv.GaussianBlur(img2, (3*sigma+1, 3*sigma+1), sigma, cv.BORDER_CONSTANT)

    img1_laplacian = img1.astype(np.float64)-img1_blured.astype(np.float64)
    img2_laplacian = img2.astype(np.float64)-img2_blured.astype(np.float64)
        
    merged_img = applyFeathring(img1_laplacian, img2_laplacian, intersect, step * (2**(i-1)))
    
    final_img += merged_img
    final_img_repr = utl.scaleIntensities(final_img, 'M')
    cv.imwrite('final-{}.jpg'.format(i), final_img_repr)    

    img1 = np.copy(img1_blured)
    img2 = np.copy(img2_blured)

# final staep, add low frequencies:
img1_blured = cv.GaussianBlur(img1, (6*sigma+1, 6*sigma+1), sigma, cv.BORDER_CONSTANT)
img2_blured = cv.GaussianBlur(img2, (6*sigma+1, 6*sigma+1), sigma, cv.BORDER_CONSTANT)
merged_img = applyFeathring(img1_blured, img2_blured, intersect, step * (2**i))

x_repr = utl.scaleIntensities(merged_img, 'M')
cv.imwrite('repr.jpg', x_repr)    

final_img += merged_img

final_img_rep = utl.scaleIntensities(final_img, 'M')
cv.imwrite('res10.jpg', final_img_rep)    
