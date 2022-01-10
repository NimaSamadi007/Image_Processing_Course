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
M, N, _ = img.shape
votes_matrix = np.zeros((M, N), dtype=int)
vote_thr = 14
temp_mat = np.zeros((M, N), dtype=np.uint8)
for q in range(15):
    print("At step {}".format(q))
    img_temp = np.copy(img)    

    # sharpen the image to enhance detection
    img_sharp = sharpenImage(img_temp, 10, 10)
    
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
    img_temp[foreground_mask] = 0
    
    # sharp result again and run another time grabcut
    img_sharp = sharpenImage(img_temp, 10, 10)
    
    fg_model[:] = 0
    bg_model[:] = 0
    mask[:, :] = 0
    
    # rectangle that surronds the lower half
    rect_pos = (0, 2500, N, M-2500)
    
    print("Running grab cut using second rectangle ...")
    mask, bg_model, fg_model = cv2.grabCut(img_sharp, mask, rect_pos, bg_model,
                                           fg_model, 1, mode=cv2.GC_INIT_WITH_RECT)
    
    foreground_mask = np.nonzero((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD))
    img_temp[foreground_mask] = 0
    
    # Remove dot-shape noises using morphology
    img_gray = cv2.cvtColor(img_temp, cv2.COLOR_RGB2GRAY)
    img_gray[img_gray > 0] = 255
    
    # erode image to eliminate noises
    erosion_kernel = np.ones((3, 3), dtype=np.uint8)
    eroded_img = cv2.erode(img_gray, erosion_kernel, iterations=3)
    # dilate eroded image to recover lost points - use the same kernel
    dilated_img = cv2.dilate(eroded_img, erosion_kernel, iterations=3)
    
    # apply mask
    img_temp[dilated_img == 0] = 0 
    # apply median blur to reduce noise
    img_median = cv2.medianBlur(img_temp, 21)    
    # apply canny to find edges
    img_edges = cv2.Canny(img_median, 200, 450)
    # find contours
    conts, _ = cv2.findContours(img_edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    temp_mat[:, :] = 0
    # draw proper contours
    for i in range(len(conts)):
        if conts[i].shape[0] <= 310 and conts[i].shape[0] >= 120: 
            cv2.drawContours(temp_mat, conts, i, 255, 3)

    # add votes
    votes_matrix[temp_mat == 255] += 1    
    print("---------------")

# now take vote on votes_matrix value and draw final contours    
votes_matrix[votes_matrix < vote_thr] = 0
votes_matrix[votes_matrix >= vote_thr] = 1
# draw contours
img[votes_matrix == 1] = [0, 255, 0]
cv2.imwrite('res10.jpg', img)
