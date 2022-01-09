import numpy as np
import cv2
import utils as utl


def sharpenImage(img, sigma, alpha):
    gauss_kernel = utl.calGaussFilter((6*sigma+1, 6*sigma+1), sigma, True)
    img_smoothed = cv2.filter2D(img.astype(np.float64), -1, gauss_kernel, borderType=cv2.BORDER_CONSTANT) 
    unsharp_mask = img.astype(np.float64) - img_smoothed.astype(np.float64) 
    img_sharpend = img.astype(np.float64) + alpha * unsharp_mask.astype(np.float64)
    img_sharpend = utl.scaleIntensities(img_sharpend, 'C')
    
    return img_sharpend    

# ---------------------------- MAIN -------------------------------- #

img = cv2.imread('birds.jpg', cv2.IMREAD_COLOR)
orig_img = np.copy(img)
M, N, _ = img.shape

# sharpen the image to enhance detection
img_sharp = sharpenImage(img, 10, 10)

# two parameters used in grabCut as buffer
fg_model = np.zeros((1, 65), dtype=np.float64)
bg_model = np.zeros((1, 65), dtype=np.float64)

# rectangle that surrounds foreground
rect_pos = (0, 0, N, 2600)

mask = np.zeros((M, N), dtype=np.uint8)
# run grabCut with respect to rect_pos

print("Running grab cut using first rectangle ...")
mask, bg_model, fg_model = cv2.grabCut(img_sharp, mask, rect_pos, bg_model,
                                       fg_model, 1, mode=cv2.GC_INIT_WITH_RECT)

# throw foreground away
foreground_mask = np.nonzero((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
img[foreground_mask] = 0

# sharp result again and rund another time grabcut
img_sharp = sharpenImage(img, 10, 10)

fg_model = np.zeros((1, 65), dtype=np.float64)
bg_model = np.zeros((1, 65), dtype=np.float64)

# rectangle that surronds the lower half
rect_pos = (0, 2500, N, M-2500)

mask = np.zeros((M, N), dtype=np.uint8)
print("Running grab cut using second rectangle ...")
mask, bg_model, fg_model = cv2.grabCut(img_sharp, mask, rect_pos, bg_model,
                                       fg_model, 1, mode=cv2.GC_INIT_WITH_RECT)

foreground_mask = np.nonzero((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
img[foreground_mask] = 0

# Remove dot noises using morphology
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
img_gray[img_gray > 0] = 255

# erode image to eliminate noises
erosion_kernel = np.ones((3, 3), dtype=np.uint8)
eroded_img = cv2.erode(img_gray, erosion_kernel, iterations=1)
# dilate eroded image to recover lost points
dilated_img = cv2.dilate(eroded_img, erosion_kernel, iterations = 1)

# apply mask
img[dilated_img == 0] = 0 
cv2.imwrite('test-q4.jpg', img)


#%%
orig_img = cv2.imread('birds.jpg', cv2.IMREAD_COLOR)
img = cv2.imread('test-q4.jpg', cv2.IMREAD_COLOR)
img_median = cv2.medianBlur(img, 21)
cv2.imwrite('test/median.jpg', img_median)

img_edges = cv2.Canny(img_median, 140, 400)

# kernel = np.ones((3, 3), np.uint8)
# dilation = cv2.dilate(img_edges, kernel, iterations = 3)
# erosion = cv2.erode(dilation, kernel, iterations = 1)

cv2.imwrite('test/edg.jpg', img_edges)

conts, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
x = 0
for i in range(len(conts)):
    print(i)
    if conts[i].shape[0] > x:
        x = conts[i].shape[0]
    if conts[i].shape[0] < 350 and conts[i].shape[0] > 125: 
        cv2.drawContours(orig_img, conts, i, (0,255,0), 3)
# cv2.drawContours(orig_img, conts, -1, (0,255,0), 3)

print("-------------")
print(x)
cv2.imwrite('test/final.jpg', orig_img)




